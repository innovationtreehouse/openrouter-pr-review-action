"""Measure the size of a RE-REVIEW user message for a PR (read-only GitHub GETs).

Rebuilds the prompt sections with the action's own code (loop.py, prompt.py,
github_ops.py). The diff is fetched only to measure its size; it is never
printed or stored. Verify mode is modelled as a rebase review (the largest case).

    PYTHONPATH=src python bench/context_budget.py --repo o/r --pr 1 --pr 2 [--impl]

--impl: also run compaction.compact_history (rebase 64KB, verify 16KB, no model).
"""

from __future__ import annotations

import argparse
import re
import subprocess

from or_pr_review import loop as L
from or_pr_review import prompt as P
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.github_ops import GitHub

BOT = "github-actions[bot]"
MAX_DIFF = 300 * 1024  # DEFAULT_MAX_DIFF_KB embed cap
COMMIT_RE = re.compile(r"\*\*Commit:\*\*\s*`([0-9a-f]{7,40})`")


# Baseline renderers vendored verbatim from fork main 29a5faa (loop.py); the
# compaction branch removed them from the action.
import re  # noqa: E402

from or_pr_review.errors import ActionError  # noqa: E402

MAX_REPLIES_BYTES = L.MAX_REPLIES_BYTES
MAX_REBASE_CONTEXT_BYTES = L.MAX_REBASE_CONTEXT_BYTES
MAX_REPLY_CHARS = 2_000
OMISSION_MARKER = "…[older entries omitted]\n"
_CONTEXT_SEPARATOR = "\n\n"


def render_agent_context(
    finding_replies: list[tuple[str, str, str]],
    issue_comments: list[tuple[str, str]],
) -> str:
    """Render comment-thread replies and PR comments into a bounded prompt block.

    Finding replies are the adjudication signal, so they get the byte budget
    first; issue comments fill what remains. Both clip from the FRONT
    (dropping the oldest entries) so overflow never evicts the newest reply.
    """
    reply_lines: list[str] = []
    for finding_id, login, body in finding_replies:
        reply_lines.append(f"Reply to finding {finding_id} (from {login}):")
        reply_lines.append(_clip_reply(body))
        reply_lines.append("")
    comment_lines: list[str] = []
    for login, body in issue_comments:
        comment_lines.append(f"PR comment (from {login}):")
        comment_lines.append(_clip_reply(body))
        comment_lines.append("")
    reply_text = _clip_tail("\n".join(reply_lines).strip(), MAX_REPLIES_BYTES)
    reply_bytes = len(reply_text.encode("utf-8"))
    remaining = MAX_REPLIES_BYTES - reply_bytes
    comment_source = "\n".join(comment_lines).strip()
    if comment_source and reply_text:
        remaining -= len(_CONTEXT_SEPARATOR.encode("utf-8"))
    comment_text = _clip_tail(comment_source, max(0, remaining))
    parts = [part for part in (reply_text, comment_text) if part]
    return _CONTEXT_SEPARATOR.join(parts)


def render_rebase_context(
    review_bodies: list[str],
    finding_replies: list[tuple[str, str, str]],
    issue_comments: list[tuple[str, str]],
) -> str:
    """Bounded historical evidence, including fixes absent from today's ledger.

    The caller supplies only reviews from the configured bot. Their text and
    all replies remain untrusted prompt data, never a source of current state.
    Hidden ledger/receipt encodings are unnecessary in this narrative context.
    """
    reviews = [re.sub(r"<!--[\s\S]*?-->", "", body).strip() for body in review_bodies]
    reviews = [body for body in reviews if body]
    heading = (
        f"Historical bot reviews ({len(reviews)} available, oldest first). "
        "These describe earlier code; current code must establish every conclusion.\n\n"
    )
    review_text = _clip_tail("\n\n--- Earlier review ---\n\n".join(reviews), 47_500)
    replies = render_agent_context(finding_replies, issue_comments)
    result = heading + review_text + "\n\nHistorical replies and PR discussion:\n\n" + replies
    if len(result.encode("utf-8")) > MAX_REBASE_CONTEXT_BYTES:
        raise ActionError("rebase review history exceeds its context budget")
    return result


def _clip_tail(text: str, max_bytes: int) -> str:
    """Keep the newest (trailing) portion of an oldest-first block."""
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = OMISSION_MARKER.encode("utf-8")
    content_budget = max_bytes - len(marker)
    if content_budget <= 0:
        return encoded[-max_bytes:].decode("utf-8", errors="ignore")
    clipped = encoded[-content_budget:].decode("utf-8", errors="ignore")
    return OMISSION_MARKER + clipped


def _clip_reply(body: str) -> str:
    text = body.strip()
    if len(text) > MAX_REPLY_CHARS:
        return text[:MAX_REPLY_CHARS] + "…[clipped]"
    return text


def b(s: str) -> int:
    return len(s.encode("utf-8"))


def gh_get(path: str, accept: str | None = None) -> str:
    cmd = ["gh", "api"] + (["-H", f"Accept: {accept}"] if accept else []) + [path]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def measure(repo: str, pr: int, impl: bool) -> dict:
    token = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
    gh = GitHub(token=token, repository=repo)
    import json

    meta = json.loads(gh_get(f"repos/{repo}/pulls/{pr}"))
    diff = gh_get(f"repos/{repo}/pulls/{pr}", "application/vnd.github.diff")
    diff_raw = b(diff)
    embedded = diff.encode()[:MAX_DIFF].decode(errors="ignore")

    bodies = gh.list_bot_review_bodies(pr, BOT)
    ledger = L.latest_ledger(bodies, repo=repo, pr_number=pr)
    state = None
    if ledger:
        state = L.LoopState(mode="verify", round_number=ledger.round_number + 1,
                            prior_findings=ledger.findings, generation=ledger.generation)
    replies_rebase = gh.list_rebase_replies(pr, BOT)
    comments = gh.list_recent_issue_comments(pr)
    history = render_rebase_context(bodies, replies_rebase, comments)
    carried = {f.id for f in state.prior_findings} if state else set()
    verify_replies_raw = gh.list_finding_replies(pr, generation=state.generation) if state else []
    verify_ctx = render_agent_context([r for r in verify_replies_raw if r[0] in carried], comments)

    # Duplicate bot reviews of the same commit, and bytes dropped by _clip_tail.
    shas = [m.group(1) if (m := COMMIT_RE.search(x)) else None for x in bodies]
    keyed = [s for s in shas if s]
    dup = len(keyed) - len(set(keyed))
    stripped = [s for s in (re.sub(r"<!--[\s\S]*?-->", "", x).strip() for x in bodies) if s]
    pre = b("\n\n--- Earlier review ---\n\n".join(stripped))
    reviews_part = history.split("\n\nHistorical replies and PR discussion:")[0]
    kept = b(reviews_part) - b(f"Historical bot reviews ({len(stripped)} available, oldest first). "
                               "These describe earlier code; current code must establish every conclusion.\n\n")
    kept -= b(OMISSION_MARKER) if OMISSION_MARKER in reviews_part else 0
    dropped = max(0, pre - kept)

    collected = CollectedReview(
        pr_number=pr, title=meta["title"], body=meta.get("body") or "", head_sha=meta["head"]["sha"],
        base_ref=meta["base"]["ref"], head_ref=meta["head"]["ref"],
        plan=DiffPlan("rebase", "full-pr", None, None, None),
        truncation=Truncation(embedded, diff_raw > MAX_DIFF, diff_raw, b(embedded), 300),
        mode="verify" if state else "initial",
    )
    ledger_block = P._loop_block(state, "", rebase=True) if state else ""
    user = P._user_prompt(collected, custom_instructions="", loop=state, agent_replies=history,
                          path_profiles=None)
    system = P._system_prompt(tone="professional", mode=collected.mode, full_pr=False, rebase=True)
    fenced_history = P._fence(history) if state else ""
    body_b = b(collected.body)
    overhead = b(user) - body_b - b(embedded) - b(ledger_block) - b(fenced_history if state else "")
    rows = {
        "system prompt": b(system),
        "metadata/policy/profiles (approx, remainder)": overhead,
        "PR body": body_b,
        "ledger block (prior findings)": b(ledger_block),
        "verify replies block (render_agent_context)": b(verify_ctx),
        "rebase history block (render_rebase_context)": b(history) if state else 0,
        "diff (embedded, capped 300KB)": b(embedded),
    }
    # Without a ledger the action stays in initial mode and sends no history.
    hist_if_loop = b(history)
    comp = None
    if impl:
        from or_pr_review import compaction

        comp = compaction.compact_history(
            budget=L.MAX_REBASE_CONTEXT_BYTES, bot_login=BOT, review_bodies=bodies,
            replies=replies_rebase, comments=comments, carried_ids={f"{state.generation}/{i}" for i in carried} if state else None,
            summary_for_evicted=None)  # deterministic only, no model calls
        vcomp = compaction.compact_history(
            budget=L.MAX_REPLIES_BYTES, bot_login=BOT, replies=[r for r in verify_replies_raw if r[0] in carried],
            comments=comments, summary_for_evicted=None)
        rows["rebase history AFTER compaction"] = comp.bytes_after
        rows["verify replies AFTER compaction"] = vcomp.bytes_after
    user_total = b(user)
    return dict(pr=pr, rows=rows, user_total=user_total, system=b(system), ledger=bool(ledger),
                n_bot=len(bodies), dup=dup, dropped=dropped, diff_raw=diff_raw,
                hist_if_loop=hist_if_loop, comp=comp, adds=meta["additions"], dels=meta["deletions"],
                n_replies=len(replies_rebase), n_comments=len(comments))


def render(r: dict) -> str:
    t = lambda n: f"{n // 4:,}"
    out = [f"### PR #{r['pr']}  (ledger marker: {'yes' if r['ledger'] else 'NO - action would run initial, history not sent; history rows shown for reference'})",
           f"bot reviews: {r['n_bot']} (same-commit duplicates: {r['dup']}); thread replies: {r['n_replies']}; "
           f"issue comments kept: {r['n_comments']}; _clip_tail dropped: {r['dropped']:,} B; "
           f"diff raw {r['diff_raw']:,} B (+{r['adds']}/-{r['dels']})", "",
           "| section | bytes | est. tokens |", "|---|---:|---:|"]
    for k, v in r["rows"].items():
        out.append(f"| {k} | {v:,} | {t(v)} |")
    if not r["ledger"]:
        out.append(f"| (history if loop were active) | {r['hist_if_loop']:,} | {t(r['hist_if_loop'])} |")
    # verify replies block is an alternative to the rebase block, not additive.
    out += [f"| **user message total** | {r['user_total']:,} | {t(r['user_total'])} |", "",
            "Per lane, user message resent on N tool turns (est. input tokens, no caching; system + user per turn):", "",
            "| N | tokens |", "|---:|---:|"]
    per = (r["user_total"] + r["system"]) // 4
    for n in (1, 8, 12):
        out.append(f"| {n} | {per * n:,} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int, action="append", required=True)
    ap.add_argument("--impl", action="store_true", help="also size or_pr_review.compaction.compact_history if present")
    ap.add_argument("--dump", help="dir to write compacted history text (private content!)")
    a = ap.parse_args()
    for pr in a.pr:
        r = measure(a.repo, pr, a.impl)
        print(render(r))
        if r["comp"]:
            c = r["comp"]
            print(f"compaction: {c.stats_line()}\n")
            if a.dump:
                open(f"{a.dump}/compacted-{pr}.txt", "w").write(c.text)


if __name__ == "__main__":
    main()
