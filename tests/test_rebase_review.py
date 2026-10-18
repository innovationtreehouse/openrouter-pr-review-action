from __future__ import annotations

import json
from dataclasses import replace

import pytest

from or_pr_review import cli
from or_pr_review.compaction import compact_history
from or_pr_review.errors import ActionError, DivergedRangeError, SchemaError
from or_pr_review.github_ops import GitHub
from or_pr_review.loop import (
    MAX_REBASE_CONTEXT_BYTES,
    Ledger,
    LedgerFinding,
    apply_round,
    encode_ledger,
)
from or_pr_review.merge import MergedIssue
from or_pr_review.prompt import build_messages
from or_pr_review.review_context import freeze_context, restore_context
from or_pr_review.schema import Resolution

REPO, GEN = "owner/repo", "abcdef123456"
HEAD, OLD, BASE = "a" * 40, "b" * 40, "c" * 40
DIFF = "diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n@@ -1 +1 @@\n-guard()\n+save()\n"


def finding(ident, status="open", severity="bug"):
    return LedgerFinding(
        ident, severity, "api.py", 1, "Missing guard", "POST bypasses guard", status
    )


class Source:
    def __init__(self):
        first = Ledger(1, (finding("r1-1"), finding("r1-2")), OLD, GEN)
        self.current = Ledger(
            3,
            (finding("r1-2", "disputed"), finding("r3-1"), finding("r3-2", severity="nit")),
            OLD,
            GEN,
        )
        self.bodies = [
            "## OpenRouter pull-request review\nOriginal fixed finding r1-1: missing guard\n"
            + encode_ledger(first, repo=REPO, pr_number=1),
            "## OpenRouter pull-request review\nr1-1 fixed correctly: guard added. "
            "r1-2 disputed: caller validates first.\n"
            + encode_ledger(self.current, repo=REPO, pr_number=1),
        ]
        self.compares = 0
        self.full_reads = []

    def list_bot_review_bodies(self, number, bot_login):
        return self.bodies.copy()

    def list_rebase_replies(self, number, bot_login):
        return [(f"{GEN}/r1-1", "developer", "guard added in earlier commit")]

    def list_recent_issue_comments(self, number):
        return [("developer", "Rebase changed the caller")]

    def pr_view(self, number):
        return {"headRefOid": HEAD, "baseRefOid": BASE, "baseRefName": "main", "headRefName": "fix"}

    def compare_diff(self, before, after):
        self.compares += 1
        raise DivergedRangeError("not a fast-forward")

    def commit_diff(self, sha):
        return DIFF

    def pr_diff(self, number, *, base_sha, head_sha):
        self.full_reads.append((base_sha, head_sha))
        return DIFF


def environment(**overrides):
    return {
        "PR_NUMBER": "1",
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_TOKEN": "test",
        "REVIEW_MODE": "auto",
        "REVIEW_SCOPE": "latest-commit",
        "EVENT_ACTION": "synchronize",
        "HEAD_SHA": HEAD,
        "MODELS": "vendor/model",
        **overrides,
    }


def test_real_divergent_collection_rechecks_disputes_and_preserves_fixed_history(monkeypatch):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)
    collected, state, history = cli._collect_with_loop(environment())
    assert collected.plan.scope == "rebase"
    assert collected.plan.kind == "full-pr"
    assert source.full_reads == [(BASE, HEAD)]
    assert state.generation == GEN and state.round_number == 4
    assert {f.id: f for f in state.prior_findings} == {f.id: f for f in source.current.findings}
    assert not state.retired_prior
    assert cli._expected_resolution_ids(state, collected) == {"r1-2", "r3-1", "r3-2"}
    assert "Original fixed finding r1-1" in history
    assert "guard added in earlier commit" in history
    assert "Rebase changed the caller" in history
    system, user = build_messages(collected, loop=state, agent_replies=history)
    assert "bug, risk, and nit" in system["content"]
    assert "do not restart a nit sweep" not in system["content"]
    assert "specific new evidence" in system["content"]
    assert "Already disputed and settled — do not re-raise" not in user["content"]
    assert "previous disposition: disputed" in user["content"]
    assert "```text\nHistorical bot reviews" in user["content"]

    outcome = apply_round(
        state,
        [
            MergedIssue(
                "Guard lost during rebase", "Earlier r1-1 fix was removed", "bug", "api.py", 1
            )
        ],
        {
            "r1-2": Resolution("r1-2", "not_fixed", "Updated caller no longer validates"),
            "r3-1": Resolution("r3-1", "fixed", "Guard verified"),
            "r3-2": Resolution("r3-2", "disputed", "Intentional label"),
        },
        reassess_disputes=True,
    )
    statuses = {f.id: f.status for f in outcome.ledger.findings}
    assert statuses == {"r1-2": "open", "r3-2": "disputed", "r4-1": "open"}
    assert outcome.open_bug_count == 2


def test_rebase_prepared_context_freezes_history_and_scope(monkeypatch):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)
    envelope = cli._prepare_execution(environment())
    frozen = restore_context(envelope)
    source.bodies.clear()
    assert frozen.collected.plan.scope == "rebase"
    assert frozen.execution is not None
    assert "r1-1 fixed correctly" in frozen.execution.agent_replies
    assert frozen.loop.generation == GEN
    with pytest.raises(SchemaError, match="inconsistent"):
        freeze_context(
            REPO,
            replace(frozen.collected, plan=replace(frozen.collected.plan, kind="single-commit")),
            frozen.loop,
            20,
        )


def test_explicit_rebase_requires_history_and_avoids_incremental_compare(monkeypatch):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)
    collected, _, _ = cli._collect_with_loop(environment(REVIEW_SCOPE="rebase"))
    assert collected.plan.scope == "rebase" and source.compares == 0
    source.bodies.clear()
    with pytest.raises(ActionError, match="requires an existing ledger"):
        cli._collect_with_loop(environment(REVIEW_SCOPE="rebase"))
    with pytest.raises(ActionError, match="requires an existing ledger"):
        cli._collect_with_loop(environment(REVIEW_SCOPE="rebase", REVIEW_MODE="initial"))


def test_rebase_rejects_a_newer_ledger_during_history_acquisition(monkeypatch):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)
    original = source.list_bot_review_bodies
    reads = 0

    def changed(number, bot_login):
        nonlocal reads
        reads += 1
        bodies = original(number, bot_login)
        if reads > 1:
            bodies.append(
                encode_ledger(replace(source.current, round_number=4), repo=REPO, pr_number=1)
            )
        return bodies

    monkeypatch.setattr(source, "list_bot_review_bodies", changed)
    with pytest.raises(ActionError, match="review history changed"):
        cli._collect_with_loop(environment())


@pytest.mark.parametrize("changed_field", ["headRefOid", "baseRefOid"])
def test_rebase_rejects_identity_change_during_full_collection(monkeypatch, changed_field):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)
    original = source.pr_view
    calls = 0

    def view(number):
        nonlocal calls
        calls += 1
        value = original(number)
        if calls > 1:
            value[changed_field] = "d" * 40
        return value

    monkeypatch.setattr(source, "pr_view", view)
    with pytest.raises(ActionError, match="changed while collecting"):
        cli._collect_with_loop(environment(REVIEW_SCOPE="rebase"))


def test_rebase_history_read_failure_is_not_silently_ignored(monkeypatch):
    source = Source()
    monkeypatch.setattr(cli, "_github", lambda env: source)

    def fail(*args):
        raise ActionError("history unavailable")

    monkeypatch.setattr(source, "list_rebase_replies", fail)
    with pytest.raises(ActionError, match="history unavailable"):
        cli._collect_with_loop(environment())


def test_history_is_bounded_with_visible_omission_and_hidden_markers_removed():
    context = compact_history(
        budget=MAX_REBASE_CONTEXT_BYTES,
        bot_login="github-actions[bot]",
        review_bodies=["old " + "Ω" * 40_000, "latest finding <!-- secret-metadata --> fixed"],
        replies=[(f"{GEN}/r1-1", "developer", "Ω" * 40_000)],
        comments=[("developer", "latest explanation")],
    ).text
    assert len(context.encode()) <= MAX_REBASE_CONTEXT_BYTES
    assert "latest finding" in context and "secret-metadata" not in context
    assert "latest explanation" in context


def test_rebase_replies_are_generation_qualified_and_ignore_spoofed_roots():
    comments = [
        {
            "id": 1,
            "user": {"login": "github-actions[bot]"},
            "body": f"<!-- or-finding:{GEN}:r1-1 -->",
        },
        {
            "id": 2,
            "user": {"login": "github-actions[bot]"},
            "body": "<!-- or-finding:111111111111:r1-1 -->",
        },
        {"id": 3, "user": {"login": "attacker"}, "body": f"<!-- or-finding:{GEN}:r9-1 -->"},
        *[
            {"in_reply_to_id": parent, "user": {"login": "dev"}, "body": "rebuttal"}
            for parent in (1, 2, 3)
        ],
    ]

    def runner(args, **kwargs):
        assert "--paginate" in args
        return json.dumps([comments])

    github = GitHub(token="test", repository=REPO, runner=runner)
    assert [row[0] for row in github.list_rebase_replies(1, "github-actions[bot]")] == [
        f"{GEN}/r1-1",
        "111111111111/r1-1",
    ]
