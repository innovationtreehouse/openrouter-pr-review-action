"""Diff-budget triage: classification, packing, and verdict semantics."""

from __future__ import annotations

import pytest

from or_pr_review.collect import pack_diff, truncate_diff
from or_pr_review.errors import ActionError
from or_pr_review.prompt import changed_paths_from_diff, diff_right_side_lines
from or_pr_review.publish import decide_verdict
from or_pr_review.triage import (
    REASON_DELETED,
    REASON_GENERATED,
    REASON_OVER_BUDGET,
    accounted_path,
    accounted_paths_from_diff,
    build_stub,
    classify_generated,
    parse_generated_globs,
    parse_gitattributes,
    path_glob_regex,
    paths_from_git_header,
    split_diff,
)


def _file_diff(path: str, added_lines: int, *, width: int = 60) -> str:
    """A parseable single-file diff segment with `added_lines` additions."""
    body = "".join(f"+{'x' * width} line {i}\n" for i in range(added_lines))
    return (
        f"diff --git a/{path} b/{path}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{added_lines} @@\n"
        f"{body}"
    )


# ---------------------------------------------------------------- classification


def test_lockfile_and_vendored_heuristics() -> None:
    for path in (
        "package-lock.json",
        "sub/dir/yarn.lock",
        "poetry.lock",
        "go.sum",
        "assets/app.min.js",
        "assets/app.js.map",
        "vendor/lib/code.py",
        "web/node_modules/pkg/index.js",
    ):
        assert classify_generated(path), path
    for path in ("src/main.py", "docs/guide.md", "lock.py", "data/small.json"):
        assert not classify_generated(path), path


def test_large_json_snapshot_heuristic_needs_size() -> None:
    path = "src/data/ground-truth/rule-coverage.json"
    assert not classify_generated(path, segment_bytes=10_000)
    assert classify_generated(path, segment_bytes=200_000)


def test_gitattributes_linguist_rules() -> None:
    rules = parse_gitattributes(
        "# comment\n"
        "*.snap linguist-generated\n"
        "docs/** linguist-vendored=true\n"
        "src/data/*.json linguist-generated=true\n"
        "src/data/hand.json -linguist-generated\n"
    )
    assert classify_generated("tests/__snapshots__/a.snap", attr_rules=rules)
    assert classify_generated("docs/site/index.html", attr_rules=rules)
    assert classify_generated("src/data/big.json", attr_rules=rules)
    # Explicit unmark wins over the earlier glob AND over heuristics.
    assert not classify_generated("src/data/hand.json", segment_bytes=500_000, attr_rules=rules)
    assert not classify_generated("src/main.py", attr_rules=rules)


def test_gitattributes_last_match_wins() -> None:
    rules = parse_gitattributes("*.json linguist-generated\n*.json -linguist-generated\n")
    assert not classify_generated("a.json", attr_rules=rules)


def test_caller_globs_take_precedence() -> None:
    rules = parse_gitattributes("src/data/hand.json -linguist-generated\n")
    caller = (path_glob_regex("src/data/**"),)
    assert classify_generated("src/data/hand.json", attr_rules=rules, caller_regexes=caller)


def test_parse_generated_globs_validation() -> None:
    assert parse_generated_globs(None) is None
    assert parse_generated_globs("  ") is None
    assert parse_generated_globs('["a/**", " b.lock "]') == ["a/**", "b.lock"]
    with pytest.raises(ActionError, match="not valid JSON"):
        parse_generated_globs("not json")
    with pytest.raises(ActionError, match="JSON array"):
        parse_generated_globs('{"a": 1}')
    with pytest.raises(ActionError, match="non-empty glob string"):
        parse_generated_globs('["ok", ""]')
    with pytest.raises(ActionError, match="non-empty glob string"):
        parse_generated_globs("[42]")
    with pytest.raises(ActionError, match="UTF-8 bytes"):
        parse_generated_globs("[" + ",".join(['"x"'] * 4000) + "]")


# ---------------------------------------------------------------------- splitting


def test_split_diff_segments_and_preamble() -> None:
    diff = "some preamble\n" + _file_diff("a.py", 2) + _file_diff("b.py", 3)
    parsed = split_diff(diff)
    assert parsed is not None
    preamble, segments = parsed
    assert preamble == "some preamble\n"
    assert [segment.path for segment in segments] == ["a.py", "b.py"]
    assert preamble + "".join(segment.text for segment in segments) == diff


def test_split_diff_unparseable_returns_none() -> None:
    assert split_diff("no headers here\njust text\n") is None
    assert split_diff("") is None


def test_segment_counts_and_hunks() -> None:
    diff = _file_diff("a.py", 4)
    _, segments = split_diff(diff)  # type: ignore[misc]
    assert segments[0].counts() == (4, 0)
    assert segments[0].hunk_headers() == ["@@ -0,0 +1,4 @@"]


def test_deleted_file_accounts_under_old_path() -> None:
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-old\n"
    )
    _, segments = split_diff(diff)  # type: ignore[misc]
    assert segments[0].path == "gone.py"


def test_deleted_file_stub_is_not_tool_readable() -> None:
    deleted = (
        "diff --git a/gone.lock b/gone.lock\n"
        "deleted file mode 100644\n"
        "--- a/gone.lock\n"
        "+++ /dev/null\n"
        "@@ -1,300 +0,0 @@\n"
        + "".join(f"-removed-{index:03d}-" + "x" * 40 + "\n" for index in range(300))
    )
    truncation = pack_diff(deleted, 1)
    assert truncation.stubbed_files == ("gone.lock",)
    assert truncation.dropped_files == ()
    assert not truncation.forces_partial
    assert REASON_DELETED in truncation.text
    assert "current content is in the checkout" not in truncation.text


def test_binary_deleted_file_stub_is_not_tool_readable() -> None:
    deleted = (
        "diff --git a/gone.lock b/gone.lock\n"
        "deleted file mode 100644\n"
        "index 0123456..0000000\n"
        "GIT binary patch\n"
        "literal 2000\n" + "z" * 2000 + "\n"
    )
    parsed = split_diff(deleted)
    assert parsed is not None
    assert parsed[1][0].is_deleted
    truncation = pack_diff(deleted, 1)
    assert truncation.stubbed_files == ("gone.lock",)
    assert truncation.dropped_files == ()
    assert not truncation.forces_partial
    assert REASON_DELETED in truncation.text
    assert "current content is in the checkout" not in truncation.text


def test_binary_dev_null_marker_alone_identifies_deletion() -> None:
    diff = (
        "diff --git a/path and name.bin b/path and name.bin\n"
        "Binary files a/path and name.bin and /dev/null differ\n"
    )
    parsed = split_diff(diff)
    assert parsed is not None
    assert parsed[1][0].is_deleted


def test_renamed_file_accounts_only_under_new_path() -> None:
    diff = (
        "diff --git a/old.py b/new.py\n"
        "similarity index 90%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "--- a/old.py\n"
        "+++ b/new.py\n"
    )
    _, segments = split_diff(diff)  # type: ignore[misc]
    assert segments[0].path == "new.py"
    assert accounted_path("old.py", "new.py") == "new.py"
    assert accounted_paths_from_diff(diff) == ("new.py",)


def test_ambiguous_unquoted_header_uses_file_headers() -> None:
    # `` b/`` occurs in the old path and in the separator, so the diff --git
    # line cannot be split safely without consulting ---/+++.
    header = "diff --git a/old b/file.txt b/new.txt"
    assert paths_from_git_header(header) is None
    diff = f"{header}\nsimilarity index 90%\n--- a/old b/file.txt\n+++ b/new.txt\n"
    parsed = split_diff(diff)
    assert parsed is not None
    assert parsed[1][0].old_path == "old b/file.txt"
    assert parsed[1][0].new_path == "new.txt"
    assert accounted_paths_from_diff(diff) == ("new.txt",)


def test_ambiguous_pure_rename_uses_rename_metadata() -> None:
    diff = (
        "diff --git a/old b/file.txt b/new.txt\n"
        "similarity index 100%\n"
        "rename from old b/file.txt\n"
        "rename to new.txt\n"
    )
    parsed = split_diff(diff)
    assert parsed is not None
    assert parsed[1][0].path == "new.txt"
    assert accounted_paths_from_diff(diff) == ("new.txt",)


def test_binary_and_mode_only_headers_are_accountable() -> None:
    diff = (
        "diff --git a/image.bin b/image.bin\n"
        "new file mode 100644\n"
        "Binary files /dev/null and b/image.bin differ\n"
        "diff --git a/script.sh b/script.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    parsed = split_diff(diff)
    assert parsed is not None
    assert [segment.path for segment in parsed[1]] == ["image.bin", "script.sh"]
    assert accounted_paths_from_diff(diff) == ("image.bin", "script.sh")


# ------------------------------------------------------------------------ packing


def test_under_budget_diff_is_untouched() -> None:
    diff = _file_diff("a.py", 5) + _file_diff("package-lock.json", 5)
    truncation = pack_diff(diff, 300)
    assert not truncation.truncated
    assert truncation.text == diff
    assert truncation.stubbed_files == ()
    assert not truncation.forces_partial


def test_over_budget_stubs_generated_files_first() -> None:
    source = _file_diff("src/main.py", 8)
    lock = _file_diff("package-lock.json", 200)
    truncation = pack_diff(source + lock, 2)
    assert truncation.truncated
    assert truncation.stubbed_files == ("package-lock.json",)
    assert truncation.dropped_files == ()
    assert not truncation.forces_partial
    # Hand-written hunks survive verbatim; the lock file keeps a header stub.
    assert source in truncation.text
    assert f"[diff stubbed by budget triage: {REASON_GENERATED}" in truncation.text
    assert "+200/-0" in truncation.text
    assert "STILL REQUIRES a coverage entry" in truncation.text


def test_stub_preserves_changed_path_accounting() -> None:
    source = _file_diff("src/main.py", 8)
    lock = _file_diff("package-lock.json", 200)
    truncation = pack_diff(source + lock, 2)
    assert changed_paths_from_diff(truncation.text) == [
        "src/main.py",
        "package-lock.json",
    ]
    # A stub exposes no anchorable right-side lines: findings on stubbed
    # files stay body-only instead of risking a rejected batched review.
    assert "package-lock.json" not in diff_right_side_lines(truncation.text)


def test_over_budget_source_demotes_largest_hand_written_file() -> None:
    small = _file_diff("small.py", 5)
    large = _file_diff("large.py", 300)
    truncation = pack_diff(small + large, 1)
    assert truncation.stubbed_files == ("large.py",)
    assert truncation.dropped_files == ()
    assert small in truncation.text
    assert f"[diff stubbed by budget triage: {REASON_OVER_BUDGET}" in truncation.text
    assert not truncation.forces_partial


def test_tiny_generated_file_embeds_when_stub_is_larger() -> None:
    tiny_lock = _file_diff("tiny.lock", 1, width=5)
    filler = _file_diff("filler.py", 500)
    truncation = pack_diff(tiny_lock + filler, 2)
    assert "tiny.lock" not in truncation.stubbed_files
    assert tiny_lock in truncation.text


def _deleted_file_diff(path: str, removed_lines: int) -> str:
    """A whole-file deletion segment shaped like the deleted fixtures above."""
    return (
        f"diff --git a/{path} b/{path}\n"
        "deleted file mode 100644\n"
        f"--- a/{path}\n"
        "+++ /dev/null\n"
        f"@@ -1,{removed_lines} +0,0 @@\n" + "".join(f"-{'x' * 40}\n" for _ in range(removed_lines))
    )


def test_many_deleted_files_all_stub_and_none_drop() -> None:
    deleted = "".join(_deleted_file_diff(f"d{i:03}.py", 30) for i in range(400))
    kept = _file_diff("src/kept.py", 3)
    max_diff_kb = 250
    diff = deleted + kept
    assert len(diff.encode("utf-8")) > max_diff_kb * 1024
    truncation = pack_diff(diff, max_diff_kb)
    assert len(truncation.stubbed_files) == 400
    assert truncation.dropped_files == ()
    assert not truncation.forces_partial
    assert kept in truncation.text
    assert len(truncation.text.encode("utf-8")) <= max_diff_kb * 1024


def test_deletion_stubs_before_same_size_hand_written_edit() -> None:
    deleted = _deleted_file_diff("gone.py", 30)
    kept = _file_diff("src/kept.py", 18)
    assert abs(len(deleted.encode("utf-8")) - len(kept.encode("utf-8"))) < 64
    # One full segment plus one stub fits; both full segments do not.
    truncation = pack_diff(deleted + kept, 2)
    assert "gone.py" in truncation.stubbed_files
    assert "src/kept.py" not in truncation.stubbed_files
    assert kept in truncation.text


def test_genuine_overflow_drops_files_and_forces_partial() -> None:
    # Hundreds of files whose stubs alone exceed a 1 KB budget.
    diff = "".join(_file_diff(f"f{i:03}.py", 3) for i in range(400))
    truncation = pack_diff(diff, 1)
    assert truncation.truncated
    assert truncation.dropped_files
    assert truncation.forces_partial
    assert "omitted entirely" in truncation.text
    assert truncation.notice is not None
    assert "must not be treated as clean" in truncation.notice
    assert len(truncation.text.encode("utf-8")) <= 1024


def test_unparseable_over_budget_diff_falls_back_to_raw_cut() -> None:
    blob = "x" * 4000
    truncation = pack_diff(blob, 1)
    legacy = truncate_diff(blob, 1)
    assert truncation.truncated
    assert truncation.stubbed_files == ()
    assert truncation.forces_partial
    assert truncation.text == legacy.text


def test_packing_is_deterministic() -> None:
    diff = (
        _file_diff("a.py", 40)
        + _file_diff("package-lock.json", 300)
        + _file_diff("b.py", 60)
        + _file_diff("c.py", 200)
    )
    first = pack_diff(diff, 5)
    second = pack_diff(diff, 5)
    assert first == second


def test_packed_size_respects_budget() -> None:
    diff = (
        _file_diff("a.py", 40)
        + _file_diff("package-lock.json", 300)
        + _file_diff("b.py", 60)
        + _file_diff("c.py", 200)
    )
    truncation = pack_diff(diff, 5)
    assert len(truncation.text.encode("utf-8")) <= 5 * 1024
    assert truncation.embedded_bytes == len(truncation.text.encode("utf-8"))


def test_pack_diff_uses_gitattributes_and_caller_globs() -> None:
    handwritten = _file_diff("src/logic.py", 30)
    data = _file_diff("src/data/snapshot.dat", 300)
    truncation = pack_diff(
        handwritten + data,
        4,
        gitattributes_text="src/data/** linguist-generated\n",
    )
    assert truncation.stubbed_files == ("src/data/snapshot.dat",)
    assert handwritten in truncation.text

    truncation = pack_diff(
        handwritten + data,
        4,
        generated_globs=["src/data/**"],
    )
    assert truncation.stubbed_files == ("src/data/snapshot.dat",)


def test_stub_first_hunk_header_is_bracketed_not_hunk() -> None:
    _, segments = split_diff(_file_diff("a.py", 3))  # type: ignore[misc]
    stub = build_stub(segments[0], REASON_GENERATED)
    assert stub.startswith("diff --git a/a.py b/a.py\n")
    for line in stub.splitlines()[1:]:
        assert line.startswith("["), line
    assert "first hunk: @@ -0,0 +1,3 @@" in stub


# ---------------------------------------------------------- verdict and notices


def test_stub_only_truncation_keeps_verdict_and_notice_says_covered() -> None:
    source = _file_diff("src/main.py", 8)
    lock = _file_diff("package-lock.json", 200)
    truncation = pack_diff(source + lock, 2)
    assert not truncation.forces_partial
    assert truncation.notice is not None
    assert "still covers every changed file" in truncation.notice
    assert "must not be treated as clean" not in truncation.notice
    verdict = decide_verdict(issues=[], truncated=truncation.forces_partial, successful_lanes=1)
    assert verdict == "clean"


def test_dropped_files_notice_is_partial() -> None:
    diff = "".join(_file_diff(f"f{i:03}.py", 3) for i in range(400))
    truncation = pack_diff(diff, 1)
    assert truncation.notice is not None
    assert "partial" in truncation.notice
    verdict = decide_verdict(issues=[], truncated=truncation.forces_partial, successful_lanes=1)
    assert verdict == "partial"


def test_split_diff_keeps_empty_context_line_between_segments() -> None:
    first = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x\n"
        "+y\n"
        "\n"  # genuine empty context line (trailing whitespace stripped)
    )
    second = _file_diff("b.py", 2)
    parsed = split_diff(first + second)
    assert parsed is not None
    _, segments = parsed
    assert segments[0].text == first
    assert "".join(segment.text for segment in segments) == first + second


def test_split_diff_round_trips_without_trailing_newline() -> None:
    diff = _file_diff("a.py", 2).rstrip("\n")
    parsed = split_diff(diff)
    assert parsed is not None
    _, segments = parsed
    assert segments[0].text == diff + "\n"


def test_omitted_marker_falls_back_to_count_for_long_paths() -> None:
    long_dir = "very/long/directory/name/that/keeps/going/" * 4
    diff = "".join(_file_diff(f"{long_dir}file{i:03}.py", 3) for i in range(200))
    truncation = pack_diff(diff, 1)
    assert truncation.dropped_files
    assert "omitted entirely]" in truncation.text
    assert len(truncation.text.encode("utf-8")) <= 1024


def test_double_star_slash_matches_zero_segments() -> None:
    # The documented generated_paths example shape: src/data/**/*.json must
    # match direct children as well as nested paths.
    regex = path_glob_regex("src/data/**/*.json")
    assert regex.match("src/data/snapshot.json")
    assert regex.match("src/data/ground-truth/foo.json")
    assert not regex.match("src/data/foo.jsonl")
    leading = path_glob_regex("**/*.lock")
    assert leading.match("root.lock")
    assert leading.match("a/b/deep.lock")


def test_gitattributes_double_star_slash_matches_zero_segments() -> None:
    rules = parse_gitattributes("src/**/generated/*.json linguist-generated\n")
    assert classify_generated("src/generated/a.json", attr_rules=rules)
    assert classify_generated("src/x/y/generated/a.json", attr_rules=rules)


def test_added_line_resembling_dev_null_header_is_not_a_deletion() -> None:
    diff = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -0,0 +1 @@\n+++ /dev/null\n"
    _, segments = split_diff(diff)  # type: ignore[misc]
    assert not segments[0].is_deleted
