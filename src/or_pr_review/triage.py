"""Diff-budget triage: deterministic per-file packing of an over-budget diff.

When the collected diff exceeds ``max_diff_kb`` the old behavior was a raw
byte truncation — every file beyond the cut vanished from the prompt and the
verdict was forced ``partial`` forever. Triage instead splits the diff into
per-file segments, classifies each changed file as hand-written source or
generated/vendored/lock-class (via ``.gitattributes`` ``linguist-generated``
/ ``linguist-vendored``, built-in path heuristics, and an optional
caller-owned glob input following the ``path_profiles`` trust model —
workflow configuration only, never PR content), and packs hand-written
hunks into the budget first; whole-file deletions demote alongside
generated files. A demoted file keeps a visible STUB in the embedded diff —
its ``diff --git`` header, add/delete counts, and first hunk header — plus
an explicit note that the file is tool-readable (or, for a deletion, absent
from the checkout) and still requires a coverage entry, so the per-file coverage contract keeps
covering every changed file.

Only when files must be dropped entirely (no segment and no stub embedded)
does truncation still force a ``partial`` verdict.

This module owns the diff-header and path-glob parsing so `collect` can
import it without a cycle (`prompt` imports `collect`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import takewhile

from or_pr_review.errors import ActionError

# git quotes paths containing non-ASCII or special characters (core.quotePath
# default): `diff --git "a/pa\303\244th" "b/pa\303\244th"`. Each side may be
# quoted independently.
_DIFF_GIT_RE = re.compile(
    r'^diff --git (?:"a/((?:[^"\\]|\\.)*)"|a/(.+)) (?:"b/((?:[^"\\]|\\.)*)"|b/(.+))$'
)

MAX_GENERATED_GLOBS = 200
GENERATED_GLOBS_MAX_BYTES = 8_000

# A committed machine-written snapshot (ground-truth JSON dumps, coverage
# maps) is recognizable by a data-file suffix plus an outsized diff segment.
# Deliberately size-based with no path allowlist: it engages only when the
# diff already exceeds max_diff_kb, and demotion costs packing priority
# only — the stub, the coverage obligation, and tool access all remain. A
# hand-maintained large JSON can opt out with `-linguist-generated` at the
# reviewed commit (which outranks this heuristic), or by raising
# max_diff_kb.
LARGE_SNAPSHOT_SUFFIXES = (".json", ".jsonl", ".geojson")
LARGE_SNAPSHOT_DIFF_BYTES = 100 * 1024

_LOCKFILE_BASENAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "bun.lockb",
        "composer.lock",
        "gemfile.lock",
        "cargo.lock",
        "poetry.lock",
        "uv.lock",
        "pipfile.lock",
        "packages.lock.json",
        "go.sum",
        "flake.lock",
        "gradle.lockfile",
    }
)
_GENERATED_SUFFIXES = (".lock", ".min.js", ".min.css", ".map")
_VENDORED_SEGMENTS = frozenset({"node_modules", "vendor", "third_party"})

# Stub classification reasons; also rendered into the stub lines.
REASON_GENERATED = "generated/vendored/lock-class"
REASON_OVER_BUDGET = "hand-written, demoted to fit the embed budget"
REASON_DELETED = "whole-file deletion"

# Reserved bytes for the trailing omitted-files marker when files must be
# dropped entirely.
_OMITTED_MARKER_RESERVE = 512


def _unquote_git_path(raw: str) -> str:
    """Decode git's C-style path quoting (octal byte escapes, \\t, \\\", …)."""
    out = bytearray()
    index = 0
    length = len(raw)
    escapes = {
        "n": b"\n",
        "t": b"\t",
        "r": b"\r",
        "a": b"\a",
        "b": b"\b",
        "f": b"\f",
        "v": b"\v",
        '"': b'"',
        "\\": b"\\",
    }
    while index < length:
        character = raw[index]
        if character == "\\" and index + 1 < length:
            nxt = raw[index + 1]
            if nxt in "01234567":
                digits = raw[index + 1 : index + 4]
                octal = ""
                for digit in digits:
                    if digit in "01234567":
                        octal += digit
                    else:
                        break
                out.append(int(octal, 8) & 0xFF)
                index += 1 + len(octal)
                continue
            if nxt in escapes:
                out += escapes[nxt]
                index += 2
                continue
        out += character.encode("utf-8")
        index += 1
    return out.decode("utf-8", errors="replace")


def paths_from_git_header(line: str) -> tuple[str, str] | None:
    """(old_path, new_path) from a `diff --git` header, unquoting as needed."""
    # With two unquoted paths, `` b/`` is both the separator and a legal part
    # of either path. The regex below would silently bind the last occurrence.
    # Resolve ordinary/equal-side cases here; split_diff uses ---/+++ metadata
    # for the remaining ambiguous cases.
    if line.startswith("diff --git a/") and '"' not in line:
        remainder = line[len("diff --git a/") :]
        candidates = [match.start() for match in re.finditer(r" b/", remainder)]
        if len(candidates) != 1:
            equal = [
                (remainder[:position], remainder[position + 3 :])
                for position in candidates
                if remainder[:position] == remainder[position + 3 :]
            ]
            if len(equal) == 1:
                return equal[0]
            return None
        position = candidates[0]
        return remainder[:position], remainder[position + 3 :]
    match = _DIFF_GIT_RE.match(line)
    if not match:
        return None
    quoted_old, plain_old, quoted_new, plain_new = match.groups()
    old = _unquote_git_path(quoted_old) if quoted_old is not None else plain_old
    new = _unquote_git_path(quoted_new) if quoted_new is not None else plain_new
    if old is None or new is None:
        return None
    return old, new


def accounted_path(old_path: str, new_path: str) -> str:
    """Return the one repository path used for coverage and path accounting.

    Changes use the new-side path. Deletions have no new-side path and remain
    accounted under the old path; a rename therefore contributes only its
    destination, which is present in the reviewed checkout.
    """
    if new_path in {"/dev/null", "dev/null"}:
        return old_path
    return new_path


def accounted_paths_from_diff(diff: str) -> tuple[str, ...]:
    """Unique changed paths in diff order using the stateful diff parser."""
    parsed = split_diff(diff)
    if parsed is None:
        return ()
    _preamble, segments = parsed
    found: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        path = accounted_path(segment.old_path, segment.new_path)
        if path in {"", "/dev/null", "dev/null"} or path in seen:
            continue
        seen.add(path)
        found.append(path)
    return tuple(found)


def path_glob_regex(pattern: str) -> re.Pattern[str]:
    """GitHub-Actions-style path glob: `*`/`?` never cross `/`, `**` does.

    fnmatch would let `src/*.py` match `src/pkg/nested.py` (and is
    case-insensitive on Windows/macOS, diverging from CI) — the wrong
    semantics for path scoping. `**/` matches zero or more whole segments,
    so `src/**/*.json` covers `src/a.json` as well as `src/deep/a.json`.
    """
    return re.compile("^" + _glob_core(pattern) + "$")


def _glob_core(pattern: str) -> str:
    """Compile glob metacharacters without choosing regex anchoring."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return "".join(parts)


def parse_generated_globs(raw: str | None) -> list[str] | None:
    """Validate the caller-owned generated_paths input (trusted workflow config).

    A JSON array of path-glob strings naming files to treat as
    generated/vendored during diff-budget triage. Demotion changes only the
    packing priority of an over-budget diff — never coverage accountability —
    so validation is about shape and size, not content policy.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if len(text.encode("utf-8")) > GENERATED_GLOBS_MAX_BYTES:
        raise ActionError(f"generated_paths exceeds {GENERATED_GLOBS_MAX_BYTES:,} UTF-8 bytes")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionError(f"generated_paths is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) > MAX_GENERATED_GLOBS:
        raise ActionError(
            f"generated_paths must be a JSON array of at most {MAX_GENERATED_GLOBS} globs"
        )
    globs: list[str] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, str) or not item.strip():
            raise ActionError(f"generated_paths[{index}] must be a non-empty glob string")
        globs.append(item.strip())
    return globs


@dataclass(frozen=True)
class AttrRule:
    """One `.gitattributes` line's linguist verdict: pattern + generated flag."""

    regex: re.Pattern[str]
    generated: bool


def parse_gitattributes(text: str) -> tuple[AttrRule, ...]:
    """`linguist-generated` / `linguist-vendored` rules from .gitattributes.

    Best-effort gitignore-style matching: a pattern without `/` matches the
    basename at any depth; a pattern with `/` matches the full path from the
    repository root (`**` crosses segments, `*`/`?` do not). Later rules win,
    and an explicit `-linguist-generated` / `=false` unmarks. Misreads only
    shift packing priority; they never remove a file from review.
    """
    rules: list[AttrRule] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        pattern, attrs = tokens[0], tokens[1:]
        verdict: bool | None = None
        for attr in attrs:
            name, _, value = attr.partition("=")
            flag = name.lstrip("-")
            if flag not in {"linguist-generated", "linguist-vendored"}:
                continue
            if name.startswith("-") or value.lower() == "false":
                verdict = False
            else:
                verdict = True
        if verdict is None:
            continue
        rules.append(AttrRule(regex=_gitattributes_regex(pattern), generated=verdict))
    return tuple(rules)


def _gitattributes_regex(pattern: str) -> re.Pattern[str]:
    anchored = pattern.startswith("/")
    body = pattern.lstrip("/")
    core = _glob_core(body)
    if anchored or "/" in body:
        return re.compile("^" + core + "$")
    # No slash: match the basename at any depth (gitignore semantics).
    return re.compile("(^|/)" + core + "$")


def _gitattributes_verdict(path: str, rules: tuple[AttrRule, ...]) -> bool | None:
    verdict: bool | None = None
    for rule in rules:
        if rule.regex.search(path):
            verdict = rule.generated
    return verdict


def _heuristic_generated(path: str, segment_bytes: int) -> bool:
    lowered = path.lower().replace("\\", "/")
    basename = lowered.rsplit("/", 1)[-1]
    if basename in _LOCKFILE_BASENAMES:
        return True
    if basename.endswith(_GENERATED_SUFFIXES):
        return True
    if any(segment in _VENDORED_SEGMENTS for segment in lowered.split("/")[:-1]):
        return True
    if basename.endswith(LARGE_SNAPSHOT_SUFFIXES) and segment_bytes > LARGE_SNAPSHOT_DIFF_BYTES:
        return True
    return False


def classify_generated(
    path: str,
    *,
    segment_bytes: int = 0,
    attr_rules: tuple[AttrRule, ...] = (),
    caller_regexes: tuple[re.Pattern[str], ...] = (),
) -> bool:
    """True when the path should be demoted behind hand-written source.

    Precedence: caller globs (deliberate workflow config) > explicit
    `.gitattributes` linguist rules (either direction) > built-in heuristics.
    """
    if any(regex.match(path) for regex in caller_regexes):
        return True
    verdict = _gitattributes_verdict(path, attr_rules)
    if verdict is not None:
        return verdict
    return _heuristic_generated(path, segment_bytes)


@dataclass
class DiffSegment:
    """One file's slice of a unified diff, header line included."""

    header_line: str
    old_path: str
    new_path: str
    lines: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        """The path the review should account the file under."""
        return accounted_path(self.old_path, self.new_path)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    @property
    def byte_len(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def is_deleted(self) -> bool:
        """Whether the reviewed-head checkout cannot supply this file."""
        # Extended headers only: an added hunk line `++ /dev/null` reads as `+++ /dev/null`.
        header = list(takewhile(lambda line: not line.startswith("@@"), self.lines))
        return (
            self.new_path in {"/dev/null", "dev/null"}
            or "+++ /dev/null" in header
            or any(line.startswith("deleted file mode ") for line in header)
            or any(
                line.startswith("Binary files ") and line.endswith(" and /dev/null differ")
                for line in header
            )
        )

    def counts(self) -> tuple[int, int]:
        adds = dels = 0
        for line in self.lines:
            if line.startswith("+") and not line.startswith("+++"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1
        return adds, dels

    def hunk_headers(self) -> list[str]:
        return [line for line in self.lines if line.startswith("@@")]


def _path_from_file_header(line: str, prefix: str) -> str | None:
    """Extract a path from a unified-diff ---/+++ metadata line."""
    if not line.startswith(prefix):
        return None
    raw = line[len(prefix) :]
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = _unquote_git_path(raw[1:-1])
    marker = prefix[:3]
    if raw == "/dev/null":
        return raw
    expected = "a/" if marker == "---" else "b/"
    if raw.startswith(expected):
        return raw[2:]
    # Git normally emits the side prefix, but accepting an already-stripped
    # path makes the fallback tolerant of hand-produced unified diffs.
    return raw


def _path_from_rename_header(line: str, prefix: str) -> str | None:
    """Extract a path from git's pure-rename metadata lines."""
    if not line.startswith(prefix):
        return None
    raw = line[len(prefix) :]
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = _unquote_git_path(raw[1:-1])
    return raw


def _ambiguous_unquoted_header(line: str) -> bool:
    if not line.startswith("diff --git a/") or '"' in line:
        return False
    remainder = line[len("diff --git a/") :]
    return sum(1 for _ in re.finditer(r" b/", remainder)) > 1


def split_diff(diff: str) -> tuple[str, list[DiffSegment]] | None:
    """(preamble, per-file segments), or None when no `diff --git` parses."""
    segments: list[DiffSegment] = []
    preamble: list[str] = []
    current: DiffSegment | None = None
    for line in (diff or "").split("\n"):
        header = paths_from_git_header(line)
        if header is not None:
            current = DiffSegment(header_line=line, old_path=header[0], new_path=header[1])
            current.lines.append(line)
            segments.append(current)
            continue
        if _ambiguous_unquoted_header(line):
            # The header itself is ambiguous when a path contains `` b/``.
            # Keep collecting this segment and resolve it from its ---/+++
            # lines below, which are unambiguous and authoritative.
            current = DiffSegment(header_line=line, old_path="", new_path="")
            current.lines.append(line)
            segments.append(current)
            continue
        if current is not None and not current.old_path:
            path = _path_from_file_header(line, "--- ")
            if path is None:
                path = _path_from_rename_header(line, "rename from ")
            if path is not None:
                current.old_path = path
        if current is not None and not current.new_path:
            path = _path_from_file_header(line, "+++ ")
            if path is None:
                path = _path_from_rename_header(line, "rename to ")
            if path is not None:
                current.new_path = path
        if current is None:
            preamble.append(line)
        else:
            current.lines.append(line)
    if not segments:
        return None
    if any(not segment.old_path or not segment.new_path for segment in segments):
        # Without both sides there is no safe way to classify or account this
        # segment. The collector will use its raw-cut fallback instead.
        return None
    # Drop the one trailing empty element str.split produced from the final
    # newline. Only the LAST segment can carry it — an empty last line on any
    # other segment is a genuine empty context line and must survive.
    last = segments[-1]
    if diff.endswith("\n") and last.lines and last.lines[-1] == "":
        last.lines.pop()
    preamble_text = "\n".join(preamble)
    if preamble_text:
        preamble_text += "\n"
    return preamble_text, segments


def build_stub(segment: DiffSegment, reason: str) -> str:
    """A stub keeps the file inside the embedded diff without its hunks.

    The original `diff --git` header line survives verbatim so every
    downstream consumer that walks headers (changed-path listing, coverage
    expectations, path profiles) still sees the file. The bracket lines can
    never be mistaken for diff content by the hunk parsers.
    """
    adds, dels = segment.counts()
    hunks = segment.hunk_headers()
    first_hunk = hunks[0][:120] if hunks else "(no hunk header)"
    second_line = (
        "[This file was deleted in this PR and its removed lines are not embedded; "
        "it is absent from the checkout, so only the counts above describe it. "
        "It STILL REQUIRES a coverage entry.]\n"
        if reason == REASON_DELETED
        else "[This file changed in this PR but its hunks are not embedded. Its "
        "current content is in the checkout — sweep it with read_file/grep "
        "(deleted lines exist only as the counts above); it STILL REQUIRES "
        "a coverage entry.]\n"
    )
    return (
        f"{segment.header_line}\n"
        f"[diff stubbed by budget triage: {reason}; +{adds}/-{dels} across "
        f"{len(hunks)} hunk(s); first hunk: {first_hunk}]\n"
        f"{second_line}"
    )


@dataclass(frozen=True)
class PackedDiff:
    text: str
    stubbed: tuple[tuple[str, str], ...]  # (path, reason)
    dropped: tuple[str, ...]


def plan_packing(
    diff: str,
    limit: int,
    *,
    attr_rules: tuple[AttrRule, ...] = (),
    caller_regexes: tuple[re.Pattern[str], ...] = (),
) -> PackedDiff | None:
    """Deterministically pack an over-budget diff into `limit` bytes.

    Generated-class files and whole-file deletions are demoted to stubs
    first (unless the stub would be larger than the hunks themselves). If the
    total still exceeds the budget, the largest remaining hand-written segment is demoted next, and
    so on — maximizing how many files keep their hunks embedded. Only when
    even the all-stubs form exceeds the budget are files dropped entirely
    (first-fit keeps what fits; the caller must surface drops as partial).

    Returns None when the diff has no parseable `diff --git` headers.
    """
    parsed = split_diff(diff)
    if parsed is None:
        return None
    preamble, segments = parsed

    seg_bytes = [segment.byte_len for segment in segments]
    stub_bytes = {
        reason: [len(build_stub(s, reason).encode("utf-8")) for s in segments]
        for reason in (REASON_GENERATED, REASON_OVER_BUDGET, REASON_DELETED)
    }
    stubbed: list[str | None] = []  # reason, or None = embed
    for index, segment in enumerate(segments):
        generated = classify_generated(
            segment.path,
            segment_bytes=seg_bytes[index],
            attr_rules=attr_rules,
            caller_regexes=caller_regexes,
        )
        # A stub bigger than the hunks it replaces is pure loss.
        if segment.is_deleted and stub_bytes[REASON_DELETED][index] < seg_bytes[index]:
            stubbed.append(REASON_DELETED)
        elif generated and stub_bytes[REASON_GENERATED][index] < seg_bytes[index]:
            stubbed.append(REASON_GENERATED)
        else:
            stubbed.append(None)

    def _piece(index: int) -> int:
        reason = stubbed[index]
        return seg_bytes[index] if reason is None else stub_bytes[reason][index]

    def total() -> int:
        return len(preamble.encode("utf-8")) + sum(_piece(index) for index in range(len(segments)))

    while total() > limit:
        candidates = [
            (seg_bytes[i], i)
            for i in range(len(segments))
            if stubbed[i] is None and seg_bytes[i] > stub_bytes[REASON_OVER_BUDGET][i]
        ]
        if not candidates:
            break
        # Largest first; original order breaks ties deterministically.
        _size, index = max(candidates, key=lambda item: (item[0], -item[1]))
        stubbed[index] = REASON_OVER_BUDGET

    dropped: list[str] = []
    include = [True] * len(segments)
    if total() > limit:
        # Even the all-stubs form does not fit: keep pieces first-fit in
        # original order and drop the rest entirely (surfaced as partial).
        budget = limit - _OMITTED_MARKER_RESERVE
        used = len(preamble.encode("utf-8"))
        for index, segment in enumerate(segments):
            piece = _piece(index)
            if used + piece > budget:
                include[index] = False
                dropped.append(segment.path)
            else:
                used += piece

    parts: list[str] = [preamble] if preamble else []
    stub_list: list[tuple[str, str]] = []
    for index, segment in enumerate(segments):
        if not include[index]:
            continue
        reason = stubbed[index]
        if reason is None:
            parts.append(segment.text)
        else:
            parts.append(build_stub(segment, reason))
            stub_list.append((segment.path, reason))
    if dropped:
        marker = (
            f"[{len(dropped)} changed file(s) beyond the embed budget were "
            f"omitted entirely: {', '.join(dropped[:10])}"
            f"{f' (+{len(dropped) - 10} more)' if len(dropped) > 10 else ''}]\n"
        )
        if len(marker.encode("utf-8")) > _OMITTED_MARKER_RESERVE:
            # Long paths could blow the byte reserve the drop loop left for
            # this marker; the count alone always fits.
            marker = (
                f"[{len(dropped)} changed file(s) beyond the embed budget were omitted entirely]\n"
            )
        parts.append(marker)
    return PackedDiff(
        text="".join(parts),
        stubbed=tuple(stub_list),
        dropped=tuple(dropped),
    )
