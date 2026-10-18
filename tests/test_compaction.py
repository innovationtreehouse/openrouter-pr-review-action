from __future__ import annotations

from or_pr_review.compaction import compact_history
from or_pr_review.github_ops import STATUS_MARKER
from or_pr_review.loop import MAX_REBASE_CONTEXT_BYTES, MAX_REPLIES_BYTES
from or_pr_review.merge import MergedIssue, format_issue_block

BOT = "github-actions[bot]"


def review(sha: str, *, verdict: str = "fail", tag: str = "", filler: int = 2_000) -> str:
    issues = [
        MergedIssue(f"Null deref {tag}", "body " + "x" * filler, "bug", "a.py", 3),
        MergedIssue(f"Naming {tag}", "short", "nit", "b.py", None),
    ]
    blocks = [format_issue_block(n, issue) for n, issue in enumerate(issues, start=1)]
    return "\n".join(
        [
            "## OpenRouter pull-request review",
            "<!-- openrouter-review-ledger:v1:QUJD -->",
            "",
            f"**Verdict:** `{verdict}`",
            "**Scope:** `rebase` (full-pr)",
            f"**Commit:** `{sha}`",
            "",
            "### Round 2 resolution",
            "",
            "### Findings",
            "",
            *blocks,
        ]
    )


def compact(**kwargs):
    kwargs.setdefault("budget", MAX_REBASE_CONTEXT_BYTES)
    kwargs.setdefault("bot_login", BOT)
    kwargs.setdefault("replies", [])
    kwargs.setdefault("comments", [])
    return compact_history(**kwargs)


def test_dedupes_by_commit_keeping_newest_and_strips_markers():
    bodies = [review("a" * 40, tag="FIRSTTAG"), review("a" * 40, tag="rerun"), review("b" * 40)]
    result = compact(review_bodies=bodies)
    assert "FIRSTTAG" not in result.text
    assert "rerun" in result.text and "<!--" not in result.text
    assert result.dropped["reviews_duplicate"] == 1


def test_status_bodies_dropped():
    bodies = [f"{STATUS_MARKER}\nReviewing with OpenRouter (2 lanes)", review("a" * 40)]
    result = compact(review_bodies=bodies)
    assert "Reviewing with" not in result.text
    assert result.dropped["reviews_status"] == 1


def test_older_reviews_become_digests_newest_full():
    bodies = [review("a" * 40, tag="old"), review("b" * 40, tag="new")]
    text = compact(review_bodies=bodies).text
    digest = "- round 2 · commit aaaaaaaaaaaa · verdict fail · 2 finding(s): "
    assert digest + "[bug] Null deref old; [nit] Naming old" in text
    assert "body xxx" in text.split("Most recent earlier review:")[1]
    assert text.count("body xxx") == 1


def test_digest_falls_back_to_prefix_when_unparseable():
    text = compact(review_bodies=["free-form " + "y" * 500, review("b" * 40)]).text
    assert "- free-form " + "y" * 290 + "\n" in text


def test_replies_grouped_and_settled_shortened():
    replies = [
        ("g/r1-1", "dev", "first"),
        ("g/r1-1", "dev", "second"),
        ("g/r1-1", "dev", "third"),
        ("g/r1-2", "dev", "argued " + "z" * 400),
    ]
    text = compact(replies=replies, carried_ids={"g/r1-1"}).text
    assert "Reply to finding g/r1-1 (from dev) (+2 earlier replies):\nthird" in text
    assert "first" not in text and "second" not in text
    assert "z" * 194 not in text
    assert "- finding g/r1-2: dev argued argued " + "z" * 193 in text


def test_comments_filtered_and_cleaned():
    code = "\n".join(["```py", *[f"line{n}" for n in range(30)], "```"])
    comments = [
        (BOT, "bot chatter"),
        ("dev", "Reviewing with OpenRouter (1 lane)"),
        ("dev", f"> quoted bot text\nmy answer\n{code}"),
    ]
    result = compact(comments=comments)
    assert "bot chatter" not in result.text and "quoted" not in result.text
    assert "my answer" in result.text and "line3" in result.text
    assert "line10" not in result.text and "…[27 lines elided]" in result.text
    assert result.dropped["comments_filtered"] == 2


def test_priority_eviction_never_exceeds_budget_or_raises():
    bodies = [review(f"{n:040x}", tag=f"r{n}", filler=6_000) for n in range(20)]
    replies = [(f"g/r1-{n}", "dev", f"open{n} " + "Ω" * 1_900) for n in range(20)]
    settled = [(f"old/r1-{n}", "dev", "s" * 300) for n in range(50)]
    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    for budget in (MAX_REPLIES_BYTES, MAX_REBASE_CONTEXT_BYTES, 3_000, 500):
        result = compact(
            budget=budget,
            review_bodies=bodies,
            replies=replies + settled,
            comments=comments,
            carried_ids={f"g/r1-{n}" for n in range(20)},
        )
        assert len(result.text.encode()) <= budget
        assert result.bytes_after == len(result.text.encode())
        assert "older items omitted]" in result.text
    small = compact(
        budget=MAX_REPLIES_BYTES,
        review_bodies=bodies,
        replies=replies + settled,
        comments=comments,
        carried_ids={f"g/r1-{n}" for n in range(20)},
    )
    # Open replies win the budget; newest survive, oldest go first.
    assert "open19 " in small.text and "open0 " not in small.text
    assert small.dropped["settled_replies"] == 50 and small.dropped["digests"] == 19
    text = small.text
    assert text.index("open17 ") < text.index("open18 ") < text.index("open19 ")


def test_summary_hook_gets_evicted_items_within_budget():
    seen: list[list[str]] = []

    def summarize(items: list[str]) -> str:
        seen.append(items)
        return "S" * 3_000

    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    result = compact(budget=MAX_REPLIES_BYTES, comments=comments, summary_for_evicted=summarize)
    assert len(seen) == 1 and any("comment0 " in item for item in seen[0])
    assert all(f"comment{n} " not in result.text for n in range(len(seen[0])))
    assert "Summary of older history (model-generated, untrusted):" in result.text
    assert "older items summarized]" in result.text and result.summarized
    assert len(result.text.encode()) <= MAX_REPLIES_BYTES


def test_summary_hook_none_or_error_falls_back_to_plain_omission():
    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    plain = compact(budget=MAX_REPLIES_BYTES, comments=comments)

    def boom(items):
        raise RuntimeError("network")

    for hook in (lambda items: None, boom):
        result = compact(budget=MAX_REPLIES_BYTES, comments=comments, summary_for_evicted=hook)
        assert result.text == plain.text


def test_oversized_summary_is_dropped_not_clipped():
    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    plain = compact(budget=MAX_REPLIES_BYTES, comments=comments)
    result = compact(
        budget=MAX_REPLIES_BYTES, comments=comments, summary_for_evicted=lambda items: "S" * 5_000
    )
    assert result.text == plain.text and not result.summarized


def test_hook_not_called_without_eviction():
    calls = []
    compact(comments=[("dev", "hi")], summary_for_evicted=lambda items: calls.append(items))
    assert not calls


def test_realistic_fixture_stats_line():
    heads = [f"{n % 10:040x}" for n in range(15)]  # 5 duplicate heads
    bodies = [review(sha, tag=str(n), filler=2_600) for n, sha in enumerate(heads)]
    result = compact(review_bodies=bodies, comments=[("dev", "ping")])
    line = result.stats_line()
    assert line.startswith("history compaction: before=")
    assert "reviews_duplicate:5" in line
    assert result.bytes_after < result.bytes_before / 4


FOOTER = "\n\n---\n🤖 Posted by [Claude Code](https://claude.com/claude-code) on behalf of @dev"


def test_footers_stripped_and_settled_clip_keeps_whole_characters():
    body = "Fixed in c6e3169. " + "é🤖" * 150 + FOOTER
    text = compact(
        replies=[("g/r1-1", "dev", body), ("g/r1-2", "dev", "Done." + FOOTER)],
        comments=[("dev", "See above." + FOOTER)],
        carried_ids={"g/r1-2"},
    ).text
    assert "Posted by" not in text and "---" not in text and "�" not in text
    assert "Reply to finding g/r1-2 (from dev):\nDone." in text
    assert "- finding g/r1-1: dev argued Fixed in c6e3169. " in text and "…" in text


def test_footer_kept_when_not_a_signature():
    text = compact(comments=[("dev", "Table\n---\nreal content here")]).text
    assert "real content here" in text


def test_digest_round_prefix_is_consistent():
    initial = (
        review("a" * 40)
        .replace("### Round 2 resolution\n", "")
        .replace("**Scope:**", "**Mode:** `initial`\n**Scope:**")
    )
    unknown = review("b" * 40).replace("### Round 2 resolution\n", "")
    text = compact(review_bodies=[initial, unknown, review("c" * 40)]).text
    assert "- round 1 · commit aaaaaaaaaaaa" in text
    assert "- round ? · commit bbbbbbbbbbbb" in text


def ledger(ident, title, severity="bug"):
    from or_pr_review.loop import LedgerFinding

    return LedgerFinding(ident, severity, "a.py", 3, title, "evidence", "open")


def test_newest_review_references_carried_findings_and_keeps_others():
    body = review("a" * 40, tag="X")  # issues: [bug] "Null deref X", [nit] "Naming X"
    text = compact(review_bodies=[body], ledger_findings=[ledger("r1-1", "Null deref X")]).text
    assert "- `r1-1` [bug] Null deref X — see Prior findings" in text
    assert "body xxx" not in text  # carried finding body replaced
    assert "#### 🔵 Issue 2 — Naming X" in text and "short" in text  # not in ledger: kept
    assert "**Verdict:** `fail`" in text


def test_reference_matches_updates_tag_and_requires_same_severity():
    body = review("a" * 40, tag="X").replace("Null deref X", "Renamed title (updates `r2-4`)")
    text = compact(
        review_bodies=[body],
        ledger_findings=[ledger("r2-4", "Old title"), ledger("r9-9", "Naming X", "bug")],
    ).text
    assert "- `r2-4` [bug] Old title — see Prior findings" in text
    assert "#### 🔵 Issue 2 — Naming X" in text  # severity nit != bug: no match


def test_pr2305_shaped_review_shrinks_to_ledger_references():
    from or_pr_review.merge import MergedIssue, format_issue_block

    titles = [f"Finding number {n} about the reconcile path" for n in range(9)]
    blocks = [
        format_issue_block(n + 1, MergedIssue(t, "detail " * 270, "bug", "x.ts", 10))
        for n, t in enumerate(titles)
    ]
    body = "## OpenRouter pull-request review\n\n**Verdict:** `issues`\n**Mode:** `initial`\n"
    body += f"**Commit:** `{'d' * 40}`\n\n### Findings\n\n" + "\n".join(blocks)
    carried = [ledger(f"r1-{n + 1}", t) for n, t in enumerate(titles)]
    result = compact(review_bodies=[body], ledger_findings=carried)
    assert len(body.encode()) > 16_000
    assert result.bytes_after < 2_000
    assert result.text.count("see Prior findings") == 9
