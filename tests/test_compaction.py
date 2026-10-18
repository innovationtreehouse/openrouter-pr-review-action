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
        return "S" * 10_000

    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    result = compact(budget=MAX_REPLIES_BYTES, comments=comments, summary_for_evicted=summarize)
    assert len(seen) == 1 and any("comment0 " in item for item in seen[0])
    assert all(f"comment{n} " not in result.text for n in range(len(seen[0])))
    assert "Summary of older history (model-generated, untrusted):" in result.text
    assert "older items summarized]" in result.text
    assert len(result.text.encode()) <= MAX_REPLIES_BYTES


def test_summary_hook_none_or_error_falls_back_to_plain_omission():
    comments = [(f"u{n}", f"comment{n} " + "c" * 1_900) for n in range(30)]
    plain = compact(budget=MAX_REPLIES_BYTES, comments=comments)

    def boom(items):
        raise RuntimeError("network")

    for hook in (lambda items: None, boom):
        result = compact(budget=MAX_REPLIES_BYTES, comments=comments, summary_for_evicted=hook)
        assert result.text == plain.text


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
