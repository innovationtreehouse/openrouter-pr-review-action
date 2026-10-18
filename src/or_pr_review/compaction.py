"""Deterministic compaction of re-review history (bot reviews, replies, comments).

Everything here stays untrusted prompt data; prompt.py fences the result.
Overflow never fails the review: lower-priority items are evicted (oldest
first within a section) and an explicit omission line says how many.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from or_pr_review.github_ops import STATUS_MARKER

MAX_REPLY_CHARS = 2_000
NEWEST_REVIEW_BYTES = 32_000
SETTLED_ARGUMENT_CHARS = 200
DIGEST_FALLBACK_CHARS = 300
DIGEST_MAX_CHARS = 600
CODE_BLOCK_MAX_LINES = 20
CODE_BLOCK_KEEP_LINES = 5
_OMISSION_RESERVE = 64

_MARKER_RE = re.compile(r"<!--[\s\S]*?-->")
_COMMIT_RE = re.compile(r"\*\*Commit:\*\* `([0-9a-fA-F]{7,64})`")
_VERDICT_RE = re.compile(r"\*\*Verdict:\*\* `([^`\n]+)`")
_ROUND_RE = re.compile(r"^### Round (\d+) resolution", re.MULTILINE)
_ISSUE_RE = re.compile(r"^#### \S+ Issue \d+ — (.+?)(?: \(updates `[^`]*`\))?$")
_SEVERITY_RE = re.compile(r"`(bug|risk|nit)`")
_STATUS_PREFIXES = ("Reviewing with OpenRouter",)

# Allocation order, highest priority first; display order is _DISPLAY.
_PRIORITY = ("open_replies", "comments", "latest_review", "digests", "settled_replies")
_DISPLAY = ("digests", "latest_review", "open_replies", "settled_replies", "comments")
_HEADINGS = {
    "digests": "Earlier review digests (oldest first):",
    "latest_review": "Most recent earlier review:",
    "open_replies": "Replies on carried findings:",
    "settled_replies": "Replies on settled findings:",
    "comments": "PR comments:",
}
_SEPARATORS = {"digests": "\n"}
_SUMMARY_HEADING = "Summary of older history (model-generated, untrusted):"

SummaryHook = Callable[[list[str]], str | None]


@dataclass(frozen=True)
class CompactedHistory:
    text: str
    bytes_before: int
    bytes_after: int
    dropped: dict[str, int] = field(default_factory=dict)
    summarized: bool = False

    def stats_line(self) -> str:
        dropped = ",".join(f"{k}:{v}" for k, v in self.dropped.items() if v) or "0"
        return (
            f"history compaction: before={self.bytes_before} "
            f"after={self.bytes_after} dropped={dropped}"
        )


def compact_history(
    *,
    budget: int,
    bot_login: str,
    review_bodies: list[str] | None = None,
    replies: list[tuple[str, str, str]],
    comments: list[tuple[str, str]],
    carried_ids: set[str] | None = None,
    summary_for_evicted: SummaryHook | None = None,
) -> CompactedHistory:
    """Compact history into at most `budget` UTF-8 bytes.

    `review_bodies` is None outside rebase scope. A reply is "open" when
    `carried_ids` is None or contains its finding id; others are settled and
    shrink to one line.
    """
    before = sum(_size(body) for body in review_bodies or [])
    before += sum(_size(body) for _, _, body in replies) + sum(_size(b) for _, b in comments)
    dropped: dict[str, int] = {}
    sections: dict[str, list[str]] = {name: [] for name in _PRIORITY}
    preamble = ""
    if review_bodies is not None:
        reviews = _dedupe_reviews(review_bodies, dropped)
        if reviews:
            sections["latest_review"] = [_clip_head(reviews[-1], NEWEST_REVIEW_BYTES)]
            sections["digests"] = [_digest(body) for body in reviews[:-1]]
        preamble = (
            f"Historical bot reviews ({len(review_bodies)} posted, {len(reviews)} after "
            "de-duplication). These describe earlier code; current code must establish "
            "every conclusion."
        )
    for (finding_id, login), (body, earlier) in _group_replies(replies).items():
        more = f" (+{earlier} earlier replies)" if earlier else ""
        if carried_ids is None or finding_id in carried_ids:
            sections["open_replies"].append(
                f"Reply to finding {finding_id} (from {login}){more}:\n{_clip_reply(body)}"
            )
        else:
            argued = " ".join(body.split())[:SETTLED_ARGUMENT_CHARS]
            sections["settled_replies"].append(
                f"- finding {finding_id}: {login} argued {argued}{more}"
            )
    for login, body in comments:
        cleaned = _clean_comment(body)
        if login == bot_login or not cleaned or _is_status(body):
            dropped["comments_filtered"] = dropped.get("comments_filtered", 0) + 1
            continue
        sections["comments"].append(f"PR comment (from {login}):\n{_clip_reply(cleaned)}")

    fixed = _size(preamble) + 2 if preamble else 0
    kept, evicted = _allocate(sections, budget - fixed - _OMISSION_RESERVE)
    summary = ""
    if evicted and summary_for_evicted is not None:
        reserve = budget // 4
        retry_kept, retry_evicted = _allocate(
            sections, budget - fixed - _OMISSION_RESERVE - reserve
        )
        try:
            produced = summary_for_evicted([text for _, text in retry_evicted])
        except Exception:  # noqa: BLE001 - a summary must never fail the review
            print("warning: history summary failed; omitting it", file=sys.stderr)
            produced = None
        candidate = f"{_SUMMARY_HEADING}\n{produced.strip()}" if produced else ""
        # All or nothing: a clipped summary would silently lose what the caller relies on.
        if produced and produced.strip() and _size(candidate) + 2 <= reserve:
            summary = candidate
            kept, evicted = retry_kept, retry_evicted
    for name, _ in evicted:
        dropped[name] = dropped.get(name, 0) + 1

    blocks = [preamble] if preamble else []
    if summary:
        blocks.append(summary)
    for name in _DISPLAY:
        if kept[name]:
            sep = _SEPARATORS.get(name, "\n\n")
            blocks.append(_HEADINGS[name] + "\n" + sep.join(kept[name]))
    if evicted:
        verb = "summarized" if summary else "omitted"
        blocks.append(f"[{len(evicted)} older items {verb}]")
    text = "\n\n".join(blocks)
    return CompactedHistory(text, before, _size(text), dropped, summarized=bool(summary))


def _allocate(
    sections: dict[str, list[str]], budget: int
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Greedy fill in priority order, newest first; each cost over-counts its separator."""
    remaining = budget
    kept: dict[str, list[str]] = {name: [] for name in sections}
    evicted: list[tuple[str, str]] = []
    for name in _PRIORITY:
        items = sections[name]
        if not items:
            continue
        sep = _size(_SEPARATORS.get(name, "\n\n"))
        head = _size(_HEADINGS[name]) + 1 + 2
        chosen: list[str] = []
        if head < remaining:
            room = remaining - head
            for item in reversed(items):
                cost = _size(item) + sep
                if cost > room:
                    if not chosen and room - sep >= 200:
                        item = _clip_head(item, room - sep)
                        cost = _size(item) + sep
                    else:
                        break
                chosen.insert(0, item)
                room -= cost
            if chosen:
                remaining = room
        kept[name] = chosen
        evicted.extend((name, item) for item in items[: len(items) - len(chosen)])
    return kept, evicted


def _dedupe_reviews(bodies: list[str], dropped: dict[str, int]) -> list[str]:
    """Strip markers, drop status-only bodies, keep only the newest review per commit."""
    cleaned: list[tuple[str | None, str]] = []
    for body in bodies:
        text = _MARKER_RE.sub("", body).strip()
        if not text or _is_status(body) or _is_status(text):
            dropped["reviews_status"] = dropped.get("reviews_status", 0) + 1
            continue
        match = _COMMIT_RE.search(text)
        cleaned.append((match.group(1).lower() if match else None, text))
    newest = {sha: index for index, (sha, _) in enumerate(cleaned) if sha}
    result = [text for index, (sha, text) in enumerate(cleaned) if not sha or newest[sha] == index]
    if len(result) < len(cleaned):
        dropped["reviews_duplicate"] = len(cleaned) - len(result)
    return result


def _digest(body: str) -> str:
    commit, verdict = _COMMIT_RE.search(body), _VERDICT_RE.search(body)
    if not commit and not verdict:
        return "- " + " ".join(body.split())[:DIGEST_FALLBACK_CHARS]
    lines = body.splitlines()
    findings: list[str] = []
    for index, line in enumerate(lines):
        heading = _ISSUE_RE.match(line)
        if heading:
            meta = next((ln for ln in lines[index + 1 :] if ln.strip()), "")
            severity = _SEVERITY_RE.search(meta)
            findings.append(f"[{severity.group(1) if severity else '?'}] {heading.group(1)[:100]}")
    round_match = _ROUND_RE.search(body)
    parts = [f"round {round_match.group(1)}"] if round_match else []
    parts.append(f"commit {commit.group(1)[:12]}" if commit else "commit ?")
    parts.append(f"verdict {verdict.group(1) if verdict else '?'}")
    parts.append(f"{len(findings)} finding(s)" + (": " + "; ".join(findings) if findings else ""))
    text = "- " + " · ".join(parts)
    return text if len(text) <= DIGEST_MAX_CHARS else text[:DIGEST_MAX_CHARS] + "…"


def _group_replies(
    replies: list[tuple[str, str, str]],
) -> dict[tuple[str, str], tuple[str, int]]:
    """Newest reply per (finding, author) with its earlier count; order follows the newest."""
    grouped: dict[tuple[str, str], tuple[str, int]] = {}
    for finding_id, login, body in replies:
        key = (finding_id, login)
        earlier = grouped.pop(key)[1] + 1 if key in grouped else 0
        grouped[key] = (body, earlier)
    return grouped


def _clean_comment(body: str) -> str:
    """Drop quoted lines and collapse long fenced code blocks."""
    out: list[str] = []
    block: list[str] | None = None
    for line in _MARKER_RE.sub("", body).splitlines():
        if block is not None:
            block.append(line)
            if line.lstrip().startswith("```"):
                out.extend(_collapse(block))
                block = None
        elif line.lstrip().startswith("```"):
            block = [line]
        elif not line.lstrip().startswith(">"):
            out.append(line)
    if block is not None:
        out.extend(_collapse(block))
    return "\n".join(out).strip()


def _collapse(block: list[str]) -> list[str]:
    if len(block) <= CODE_BLOCK_MAX_LINES:
        return block
    elided = len(block) - CODE_BLOCK_KEEP_LINES
    return [*block[:CODE_BLOCK_KEEP_LINES], f"…[{elided} lines elided]", "```"]


def _is_status(body: str) -> bool:
    return STATUS_MARKER in body or body.lstrip().startswith(_STATUS_PREFIXES)


def _clip_reply(body: str) -> str:
    text = body.strip()
    return text[:MAX_REPLY_CHARS] + "…[clipped]" if len(text) > MAX_REPLY_CHARS else text


def _clip_head(text: str, max_bytes: int) -> str:
    """Keep the beginning of `text` within `max_bytes`, marking the cut."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = "\n…[clipped]"
    if max_bytes < _size(marker):
        return encoded[: max(0, max_bytes)].decode("utf-8", errors="ignore")
    return encoded[: max_bytes - _size(marker)].decode("utf-8", errors="ignore") + marker


def _size(text: str) -> int:
    return len(text.encode("utf-8"))
