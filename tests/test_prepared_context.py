from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.compaction import compact_history
from or_pr_review.errors import SchemaError
from or_pr_review.loop import LoopState
from or_pr_review.review_context import (
    CONTEXT_VERSION,
    FROZEN_RUNTIME_KEYS,
    PreparedExecution,
    freeze_context,
    freeze_runtime,
    restore_context,
)
from or_pr_review.review_plan import ReviewLane, ReviewPlan
from or_pr_review.review_policy import ResolvedPolicy

REPOSITORY = "example/project"
HEAD = "a" * 40


def _collected(*, policy: ResolvedPolicy | None = None) -> CollectedReview:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    return CollectedReview(
        1,
        "title",
        "",
        HEAD,
        "main",
        "change",
        DiffPlan("full-pr", "full-pr", "b" * 40, HEAD, None),
        Truncation(diff, False, len(diff), len(diff), 300),
        "initial",
        ("a.py",),
        policy_base_sha="b" * 40 if policy else "",
        review_policy=policy,
    )


def _plan(**changes: object) -> ReviewPlan:
    value = ReviewPlan(
        "code",
        "standard",
        "baseline",
        (ReviewLane("openai/gpt-5", True, "OpenAI", "priority"),),
        "openai/gpt-5.6-luna",
        "low",
        50,
        600,
        1320,
        "c" * 64,
        False,
    )
    return replace(value, **changes)


def _execution(**changes: object) -> PreparedExecution:
    value = PreparedExecution(
        _plan(),
        freeze_runtime(
            {
                "ROAST_LEVEL": "professional",
                "CUSTOM_INSTRUCTIONS": "check boundaries",
                "PATH_PROFILES": '[{"paths":["*.py"],"instructions":"check types"}]',
                "PERSONA": "",
                "JUDGE_NEEDED": "true",
                "OPENROUTER_TIMEOUT_SECONDS": "180",
                "FAIL_ON": "bugs",
                "BOT_LOGIN": "github-actions[bot]",
                "OPENROUTER_API_KEY": "must never be captured",
            }
        ),
        "agent reply",
        1_700_000_000_000,
        1_700_001_320_000,
        "https://github.com/example/project/actions/runs/123",
        1,
    )
    return replace(value, **changes)


def _snapshot(**changes: object) -> dict:
    context = freeze_context(
        REPOSITORY, _collected(), LoopState("initial", 1), 50, execution=_execution()
    )
    context.update(changes)
    return context


def _redigest(snapshot: dict) -> None:
    snapshot["sha256"] = hashlib.sha256(
        json.dumps(
            snapshot["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def test_prepared_execution_round_trip_and_legacy_call_signature() -> None:
    snapshot = _snapshot()
    restored = restore_context(json.loads(json.dumps(snapshot)))
    assert restored.execution == _execution()
    assert restored.execution.plan.lanes[0].provider == "OpenAI"
    assert "OPENROUTER_API_KEY" not in json.loads(restored.execution.runtime_json)
    old_caller = restore_context(
        freeze_context(REPOSITORY, _collected(), LoopState("initial", 1), 50)
    )
    assert old_caller.execution is None


def test_runtime_freeze_is_canonical_and_allow_list_only() -> None:
    encoded = freeze_runtime(
        {
            "FAIL_ON": " ANY ",
            "TOKEN": "no",
            "ROAST_LEVEL": " PlayFul ",
            "JUDGE_NEEDED": "YES",
            "OPENROUTER_TIMEOUT_SECONDS": "",
            "ALL_ROLE_DEADLINE_SECONDS": " 42 ",
        }
    )
    assert json.loads(encoded) == {
        "ALL_ROLE_DEADLINE_SECONDS": "42",
        "FAIL_ON": "any",
        "JUDGE_NEEDED": "true",
        "OPENROUTER_TIMEOUT_SECONDS": "180",
        "ROAST_LEVEL": "playful",
    }
    assert set(json.loads(encoded)) <= FROZEN_RUNTIME_KEYS


@pytest.mark.parametrize(
    "runtime",
    [
        '{"TOKEN":"secret"}',
        '{"ROAST_LEVEL":"professional","ROAST_LEVEL":"playful"}',
        '{"OPENROUTER_TIMEOUT_SECONDS":"NaN"}',
        '{"JUDGE_NEEDED":true}',
        '{"PATH_PROFILES":"[\\"bad\\"]"}',
    ],
)
def test_runtime_restore_rejects_unknown_secrets_and_bad_shape(runtime: str) -> None:
    snapshot = _snapshot()
    snapshot["payload"]["execution"]["runtime_json"] = runtime
    _redigest(snapshot)
    with pytest.raises(SchemaError):
        restore_context(snapshot)


@pytest.mark.parametrize(
    "execution",
    [
        lambda: _execution(agent_replies="x" * 16_001),
        lambda: _execution(runtime_json=freeze_runtime({"PERSONA": "x" * 1_001})),
        lambda: _execution(runtime_json=freeze_runtime({"CUSTOM_INSTRUCTIONS": "x" * 16_001})),
        lambda: _execution(source_run_url="https://github.com/other/repo/actions/runs/123"),
    ],
)
def test_prepared_execution_bounded_strings(execution) -> None:
    with pytest.raises(SchemaError):
        freeze_context(REPOSITORY, _collected(), LoopState("initial", 1), 50, execution=execution())


@pytest.mark.parametrize(
    "plan",
    [
        lambda: _plan(lanes=(ReviewLane("openai/gpt-5", False),)),
        lambda: _plan(lanes=(ReviewLane("openai/gpt-5", True), ReviewLane("openai/gpt-5", False))),
        lambda: _plan(lanes=(ReviewLane("bad model", True),)),
        lambda: _plan(lanes=(ReviewLane("openai/gpt-5", True, "not a provider"),)),
        lambda: _plan(level="deep", trigger="baseline"),
    ],
)
def test_plan_routes_lanes_and_trigger_are_validated(plan) -> None:
    with pytest.raises(SchemaError):
        freeze_context(
            REPOSITORY, _collected(), LoopState("initial", 1), 50, execution=_execution(plan=plan())
        )


def test_policy_profile_mismatch_and_weakening_are_rejected() -> None:
    policy = ResolvedPolicy("b" * 40, "security", "deep", (), (), (), "d" * 64)
    with pytest.raises(SchemaError):
        freeze_context(
            REPOSITORY,
            _collected(policy=policy),
            LoopState("initial", 1),
            50,
            execution=_execution(),
        )
    matching_but_weak = _plan(profile="security", level="standard", trigger="baseline")
    with pytest.raises(SchemaError):
        freeze_context(
            REPOSITORY,
            _collected(policy=policy),
            LoopState("initial", 1),
            50,
            execution=_execution(plan=matching_but_weak),
        )


def test_absolute_deadline_shape_and_exact_types_are_validated() -> None:
    with pytest.raises(SchemaError):
        freeze_context(
            REPOSITORY,
            _collected(),
            LoopState("initial", 1),
            50,
            execution=_execution(deadline_unix_ms=1_700_001_320_001),
        )
    snapshot = _snapshot()
    snapshot["payload"]["execution"]["started_unix_ms"] = True
    _redigest(snapshot)
    with pytest.raises(SchemaError):
        restore_context(snapshot)


@pytest.mark.parametrize("field", ["runtime_json", "agent_replies", "plan"])
def test_execution_changes_are_digest_covered(field: str) -> None:
    snapshot = _snapshot()
    payload = snapshot["payload"]["execution"]
    if field == "runtime_json":
        payload[field] = freeze_runtime({"FAIL_ON": "any"})
    elif field == "agent_replies":
        payload[field] = "changed"
    else:
        payload[field]["effort"] = "high"
    with pytest.raises(SchemaError, match="digest"):
        restore_context(snapshot)


def test_restore_rejects_old_context_version() -> None:
    snapshot = _snapshot()
    snapshot["version"] = CONTEXT_VERSION - 1
    with pytest.raises(SchemaError, match="unsupported"):
        restore_context(snapshot)


def test_prepared_context_agent_replies_roundtrip_near_cap() -> None:
    replies = [
        ("r1-1", "dev", "OLD reply " + "x" * 4_000),
        ("r1-2", "dev", "NEWEST reply " + "🔍" * 500),
    ]
    comments = [(f"user{n}", f"comment {n} " + "y" * 1_500) for n in range(8)]
    rendered = compact_history(
        budget=16_000, bot_login="bot", replies=replies, comments=comments
    ).text
    assert "NEWEST reply" in rendered
    assert len(rendered.encode("utf-8")) <= 16_000
    execution = _execution(agent_replies=rendered)
    snapshot = freeze_context(
        REPOSITORY,
        replace(_collected(), mode="verify"),
        LoopState("verify", 2),
        50,
        execution=execution,
    )
    restored = restore_context(snapshot)
    assert restored.execution is not None
    assert restored.execution.agent_replies == rendered
