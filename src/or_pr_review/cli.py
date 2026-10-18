"""CLI entry for the composite action roles: setup, lane, judge, all."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from or_pr_review.brief import BriefSession
from or_pr_review.collect import (
    DEFAULT_MAX_DIFF_KB,
    DIVERGED_NOTICE,
    CollectedReview,
    collect_review,
    head_sha_from_pr,
    parse_mode,
    parse_scope,
    resolve_mode,
)
from or_pr_review.compaction import compact_history
from or_pr_review.errors import ActionError, SchemaError
from or_pr_review.github_ops import GitHub, require_full_sha, upsert_status_comment
from or_pr_review.harness import (
    DEFAULT_LANE_TIMEOUT_SECONDS,
    ProgressFn,
    openrouter_chat,
    parse_max_tool_turns,
    require_openrouter_key,
    run_lane,
    sanitize_anchors,
)
from or_pr_review.judge import (
    deterministic_union_with_cap,
    partition_reviewable_lanes,
    run_llm_judge,
)
from or_pr_review.loop import (
    MAX_REBASE_CONTEXT_BYTES,
    MAX_REPLIES_BYTES,
    Ledger,
    LoopState,
    apply_round,
    apply_severity_floor,
    decide_loop_state,
    encode_ledger,
    latest_ledger,
    merge_resolutions,
    round_report,
)
from or_pr_review.merge import MergedIssue, issues_from_single_lane
from or_pr_review.models import (
    LANE_CAP,
    judge_is_needed,
    matrix_json,
    models_json,
    parse_judge_model,
    parse_model_routes,
    parse_models,
    parse_slug,
)
from or_pr_review.prompt import (
    build_messages,
    changed_paths_from_diff,
    diff_right_side_lines,
    parse_path_profiles,
    review_policy_block,
)
from or_pr_review.publish import (
    canonical_receipt_json,
    decide_verdict,
    fail_on_should_fail,
    inline_review_comments,
    render_incomplete,
    render_review_parts,
)
from or_pr_review.redaction import redact
from or_pr_review.review_context import (
    FROZEN_RUNTIME_KEYS,
    MAX_CONTEXT_BYTES,
    PreparedExecution,
    ReviewContext,
    freeze_context,
    freeze_runtime,
    restore_context,
)
from or_pr_review.review_plan import (
    ReviewPlan,
    parse_review_profiles,
    resolve_review_plan,
)
from or_pr_review.review_policy import resolve_policy
from or_pr_review.schema import (
    MAX_COVERAGE_ENTRIES,
    LaneResult,
    coverage_count_mismatches,
    failed_lane,
    parse_lane_artifact,
)
from or_pr_review.triage import parse_generated_globs
from or_pr_review.workspace import materialize_commit

_ACTIVE_ENV: dict[str, str] = {}

# The organization workflow gives this action 25 minutes. Stop OpenRouter
# work at 22 minutes so review publication and composite cleanup cannot lose a
# completed lane at the hard GitHub cancellation boundary (the PR358 failure).
JOB_BUDGET_SECONDS = 22 * 60
BRIEF_TIMEOUT_SECONDS = 120
# Ledger plus a 12 KB brief marker must leave room in the 60 KB review body.
MAX_LEDGER_WITH_BRIEF = 32_000
POST_RESERVE_SECONDS = 3 * 60
JUDGE_SCHEDULING_MARGIN_SECONDS = 5
JUDGE_BUDGET_SECONDS = 60
MIN_JUDGE_WINDOW_SECONDS = 15
LANE_COLLECTION_GRACE_SECONDS = 5
DEFAULT_BOT_LOGIN = "github-actions[bot]"
_JOB_DEADLINE_KEY = "_OR_PR_REVIEW_JOB_DEADLINE_MONOTONIC"


@dataclass(frozen=True)
class JudgeOutcome:
    """The merge decision and its publication metadata.

    Keeping the diagnostics beside the issues prevents a later caller from
    re-filtering a different lane set while constructing the final verdict.
    """

    issues: list[MergedIssue]
    note: str
    cost: float | None
    ran: bool
    environment_diagnostics: list[tuple[str, str]]

    def __iter__(self):
        """Compatibility for callers that previously unpacked the 4-tuple."""
        yield from (self.issues, self.note, self.cost, self.ran)


@dataclass(frozen=True)
class PreparedContext:
    """Immutable context plus the exact envelope that lane artifacts carry."""

    context: ReviewContext
    digest: str
    envelope: dict[str, Any]

    @property
    def repository(self) -> str:
        return self.context.repository

    @property
    def collected(self) -> CollectedReview:
        return self.context.collected

    @property
    def loop(self) -> LoopState:
        return self.context.loop

    @property
    def execution(self) -> PreparedExecution:
        assert self.context.execution is not None
        return self.context.execution


@dataclass(frozen=True)
class PreparedLaneInputs:
    """Shared, immutable all-role inputs prepared exactly once."""

    frozen_env: dict[str, str]
    workspace: Path | None
    messages: list[dict[str, Any]]
    expect_coverage: bool
    expected_paths: set[str] | None
    expected_resolution_ids: set[str] | None


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    global _ACTIVE_ENV
    args = list(sys.argv[1:] if argv is None else argv)
    environ = dict(env) if env is not None else dict(os.environ)
    role = (args[0] if args else environ.get("ROLE") or "all").strip().lower()
    environ["ROLE"] = role
    _ACTIVE_ENV = environ
    try:
        if role == "policy":
            from or_pr_review.policy_cli import main as policy_main

            return policy_main(args[1:])
        policy_mode = (environ.get("REVIEW_POLICY") or "off").strip().lower()
        if policy_mode not in {"off", "base"}:
            raise ActionError("review_policy must be off or base")
        if role in {"all", "lane", "judge"} and not _prepared_requested(environ):
            environ[_JOB_DEADLINE_KEY] = str(time.monotonic() + _job_budget_seconds(environ))
        if role == "setup":
            return _role_setup(environ)
        if role == "lane":
            return _role_lane(environ)
        if role == "judge":
            return _role_judge(environ)
        if role == "all":
            return _role_all(environ)
        raise ActionError(f"unknown role {role!r}; expected setup, lane, judge, or all")
    except SchemaError as exc:
        _error(f"schema mismatch (fail-closed): {exc}")
        if role != "policy":
            _best_effort_incomplete(environ, stage=role, reason=redact(str(exc)))
        return 1
    except ActionError as exc:
        _error(redact(str(exc)))
        if role != "policy":
            _best_effort_incomplete(environ, stage=role, reason=redact(str(exc)))
        return 1
    except Exception as exc:  # noqa: BLE001 — unexpected bugs are operational failures
        _error(f"unexpected action error: {redact(str(exc))}")
        print(redact(traceback.format_exc()), file=sys.stderr)
        return 1


def _validate_prepared_context_inputs(env: dict[str, str]) -> None:
    """Reject half-supplied frozen context before any GitHub or provider work."""
    path = (env.get("REVIEW_CONTEXT_FILE") or "").strip()
    digest = (env.get("REVIEW_CONTEXT_SHA256") or "").strip()
    if bool(path) ^ bool(digest):
        raise ActionError("review_context_file and review_context_sha256 must be supplied together")


def _role_setup(env: dict[str, str]) -> int:
    if _prepared_requested(env):
        _validate_prepared_context_inputs(env)
        supplied = (env.get("REVIEW_CONTEXT_FILE") or "").strip()
        if supplied:
            context, restored = _load_prepared_context(env)
        else:
            context = _prepare_execution(env)
            restored = restore_context(context)
        assert restored.execution is not None
        plan = restored.execution.plan
        _write_prepared_outputs(context, publisher_judge_required=True)
        path = _write_prepared_context(env, context)
        _set_output("review_context_file", str(path))
        _set_output("review_context_sha256", _context_digest(context))
        _set_output("head_sha", restored.collected.head_sha)
        print(f"prepared {plan.profile}/{plan.level} with {len(plan.lanes)} lane(s)")
        return 0
    slugs = _validate_inputs(env, full_roster=True)
    needed = _judge_needed(env, slugs)
    judge_model = parse_judge_model(env.get("JUDGE_MODEL"))
    _write_setup_outputs(slugs, needed, judge_model)
    print(f"parsed {len(slugs)} model lane(s) (cap {LANE_CAP}): {', '.join(slugs)}")
    if needed:
        print(f"judge will run with `{judge_model}`")
    else:
        print("judge skipped: one review lane (one reviewer = no judge)")
    return 0


def _write_setup_outputs(slugs: list[str], needed: bool, judge_model: str) -> None:
    _write_lane_setup_outputs(slugs)
    _write_judge_outputs(needed, judge_model)


def _write_lane_setup_outputs(slugs: list[str]) -> None:
    _set_output("models_json", models_json(slugs))
    _set_output("matrix", matrix_json(slugs))
    _set_output("lane_count", str(len(slugs)))


def _write_judge_outputs(needed: bool, judge_model: str) -> None:
    _set_output("judge_needed", "true" if needed else "false")
    _set_output("judge_model", judge_model)


def _role_lane(env: dict[str, str]) -> int:
    if _prepared_requested(env):
        return _role_lane_prepared(env)
    slugs = _validate_inputs(env)
    index = _int_env(env, "LANE_INDEX", 0)
    if index < 0:
        raise ActionError(f"LANE_INDEX {index} is invalid")
    override = (env.get("LANE_MODEL") or "").strip()
    if override:
        model = parse_slug(override, what="lane_model")
    elif index < len(slugs):
        model = slugs[index]
    elif len(slugs) == 1:
        # Reusable workflow matrix jobs pass models=<one slug> plus the global
        # matrix index. Keep that index so lane-N.json artifacts do not collide.
        model = slugs[0]
    else:
        raise ActionError(f"LANE_INDEX {index} is out of range for {len(slugs)} model(s)")
    result, collected, state = _run_one_lane(env, model)
    path = _write_lane_file(env, index, result)
    _set_output("lane_file", str(path))
    _set_output("lane_ok", "true" if result.ok else "false")
    if result.ok:
        print(f"lane {index} `{model}` ok: {len(result.findings)} finding(s)")
    else:
        print(f"lane {index} `{model}` failed-open: {result.error}")
    if not _judge_needed(env, slugs):
        return _finish(env, [result], collected=collected, loop=state)
    return 0


def _role_judge(env: dict[str, str]) -> int:
    if _prepared_requested(env):
        return _role_judge_prepared(env)
    expected = _validate_inputs(env)
    directory = Path(env.get("LANE_RESULTS_DIR") or "")
    if not directory.is_dir():
        raise ActionError("LANE_RESULTS_DIR is missing or not a directory")
    lanes = _load_lane_dir(directory, expected)
    contexts = [restore_context(lane.review_context) for lane in lanes if lane.review_context]
    if not contexts:
        raise SchemaError("no matrix publication context is available; rerun the lanes")
    context = contexts[0]
    if any(other != context for other in contexts[1:]):
        raise SchemaError("matrix lanes collected different review contexts; rerun the lanes")
    if (
        context.repository != (env.get("GITHUB_REPOSITORY") or "").strip()
        or str(context.collected.pr_number) != (env.get("PR_NUMBER") or "").strip()
        or context.max_tool_turns != parse_max_tool_turns(env.get("MAX_TOOL_TURNS"))
        or (
            (env.get("HEAD_SHA") or "").strip()
            and context.collected.head_sha != env["HEAD_SHA"].strip().lower()
        )
    ):
        raise SchemaError("matrix publication context does not match this judge job")
    return _finish(env, lanes, collected=context.collected, loop=context.loop)


def _validate_inputs(env: dict[str, str], *, full_roster: bool = False) -> list[str]:
    _validate_profile_inputs(env)
    slugs = parse_models(env.get("MODELS"))
    routes = parse_model_routes(env.get("MODEL_ROUTES"))
    if (
        full_roster
        and parse_review_profiles(env.get("REVIEW_PROFILES")) is None
        and set(routes) - set(slugs)
    ):
        raise ActionError("model_routes keys must match configured models")
    _job_budget_seconds(env)
    _env_flag(env, "JUDGE_NEEDED", judge_is_needed(slugs))
    _env_flag(env, "STATUS_COMMENTS", True)
    parse_judge_model(env.get("JUDGE_MODEL"))
    parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    parse_mode(env.get("REVIEW_MODE") or "auto")
    fail_on = (env.get("FAIL_ON") or "never").strip().lower()
    if fail_on not in {"never", "bugs", "any"}:
        raise ActionError("fail_on must be never, bugs, or any")
    roast = (env.get("ROAST_LEVEL") or "professional").strip().lower()
    if roast not in {"professional", "playful"}:
        raise ActionError("roast_level must be professional or playful in v1")
    max_diff = _int_env(env, "MAX_DIFF_KB", DEFAULT_MAX_DIFF_KB)
    if max_diff <= 0:
        raise ActionError("max_diff_kb must be a positive integer")
    parse_max_tool_turns(env.get("MAX_TOOL_TURNS"))
    openrouter_timeout = _int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180)
    if openrouter_timeout < 1 or openrouter_timeout > 600:
        raise ActionError("openrouter_timeout_seconds must be an integer from 1 through 600")
    _configured_all_role_deadline_seconds(env)
    override = (env.get("LANE_MODEL") or "").strip()
    if override:
        parse_slug(override, what="lane_model")
    _bot_login(env)
    custom = env.get("CUSTOM_INSTRUCTIONS") or ""
    if len(custom.encode("utf-8")) > 16_000:
        raise ActionError("custom_instructions exceeds 16,000 UTF-8 bytes")
    parse_path_profiles(env.get("PATH_PROFILES"))
    parse_generated_globs(env.get("GENERATED_PATHS"))
    return slugs


def _prepared_requested(env: dict[str, str]) -> bool:
    """Whether this invocation must consume/produce the frozen plan contract."""
    return (
        (env.get("REVIEW_POLICY") or "off").strip().lower() == "base"
        or bool((env.get("REVIEW_PROFILES") or "").strip())
        or (env.get("REVIEW_LEVEL") or "auto").strip().lower() == "deep"
        or bool((env.get("REVIEW_CONTEXT_FILE") or "").strip())
        or bool((env.get("REVIEW_CONTEXT_SHA256") or "").strip())
    )


def _validate_profile_inputs(env: dict[str, str]) -> None:
    """Reject profile configuration mistakes before any GitHub/provider work."""
    level = (env.get("REVIEW_LEVEL") or "auto").strip().lower()
    if level not in {"auto", "deep"}:
        raise ActionError("review_level must be auto or deep")
    registry = parse_review_profiles(env.get("REVIEW_PROFILES"))
    if registry is not None:
        mixed = [
            name
            for name in ("MODELS", "JUDGE_MODEL", "EFFORT", "JUDGE_NEEDED")
            if (env.get(name) or "").strip()
        ]
        if mixed:
            raise ActionError("review_profiles cannot be mixed with " + ", ".join(mixed).lower())
        routes = parse_model_routes(env.get("MODEL_ROUTES"))
        available = {
            lane.model
            for profile in registry.profiles
            for panel in (profile.standard, profile.deep)
            if panel is not None
            for lane in panel.lanes
        }
        unknown = set(routes) - available
        if unknown:
            raise ActionError("model_routes names model(s) absent from review_profiles")
    elif level == "deep":
        # Resolve produces the same diagnostic later; this earlier failure
        # guarantees no collection/API activity for an impossible request.
        raise ActionError("review_level=deep requires a configured deep review profile")
    lane_ceiling = _int_env(env, "LANE_TIMEOUT_SECONDS", DEFAULT_LANE_TIMEOUT_SECONDS)
    if not 1 <= lane_ceiling <= 1800:
        raise ActionError("lane_timeout_seconds must be an integer from 1 through 1800")


def _judge_needed(env: dict[str, str], slugs: list[str] | None = None) -> bool:
    inferred = judge_is_needed(slugs if slugs is not None else parse_models(env.get("MODELS")))
    return _env_flag(env, "JUDGE_NEEDED", inferred)


def _context_digest(envelope: dict[str, Any]) -> str:
    digest = envelope.get("sha256")
    if type(digest) is not str:
        raise SchemaError("prepared context is missing its digest")
    return digest


def _prepared_plan(env: dict[str, str], collected: CollectedReview, state: LoopState) -> ReviewPlan:
    registry = parse_review_profiles(env.get("REVIEW_PROFILES"))
    policy = collected.review_policy
    return resolve_review_plan(
        registry,
        profile=policy.profile if policy is not None else "code",
        minimum=policy.minimum if policy is not None else "standard",
        requested_level=(env.get("REVIEW_LEVEL") or "auto").strip().lower(),
        mode=state.mode,
        models=parse_models(env.get("MODELS")),
        judge_model=parse_judge_model(env.get("JUDGE_MODEL")),
        routes=parse_model_routes(env.get("MODEL_ROUTES")),
        effort=(env.get("EFFORT") or "").strip(),
        max_tool_turns=parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
        job_budget_seconds=_job_budget_seconds(env),
        lane_timeout_seconds=_int_env(env, "LANE_TIMEOUT_SECONDS", DEFAULT_LANE_TIMEOUT_SECONDS),
    )


def _prepare_execution(env: dict[str, str]) -> dict[str, Any]:
    """Collect the loop once and freeze every paid-work input for a profile run."""
    # Setup owns the absolute clock: collection/policy resolution consumes the
    # same finite budget as lanes and publication, never a free prelude.
    started_ms = int(time.time() * 1000)
    _validate_inputs(env, full_roster=True)
    collect_env = dict(env)
    if (env.get("REVIEW_LEVEL") or "auto").strip().lower() == "deep" and parse_scope(
        env.get("REVIEW_SCOPE") or "full-pr"
    ) != "rebase":
        # A manually deep review is exhaustive from the outset; preserve the
        # requested/auto loop mode instead of resetting it to initial.
        collect_env["REVIEW_SCOPE"] = "full-pr"
    collected, state, replies = _collect_with_loop(collect_env)
    collected = _with_review_policy(env, collected, state)
    plan = _prepared_plan(env, collected, state)
    if plan.level == "deep" and collected.plan.kind != "full-pr":
        # Policy escalation happens after the incremental collection.  Reuse
        # the already-selected loop/replies, only replacing the diff under the
        # pinned identity; this cannot silently turn a verify round into reset.
        full_env = dict(
            env,
            REVIEW_MODE=state.mode,
            REVIEW_SCOPE="full-pr",
            HEAD_SHA=collected.head_sha,
        )
        full = _collect(full_env)
        if full.head_sha != collected.head_sha or full.policy_base_sha != collected.policy_base_sha:
            raise SchemaError(
                "head/base identity changed while acquiring the required full-PR deep diff"
            )
        collected = replace(
            full,
            review_policy=collected.review_policy,
            body=collected.body,
            body_brief=collected.body_brief,
            brief_marker=collected.brief_marker,
        )
    execution = PreparedExecution(
        plan=plan,
        runtime_json=freeze_runtime(env),
        agent_replies=replies,
        started_unix_ms=started_ms,
        deadline_unix_ms=started_ms + plan.job_budget_seconds * 1000,
        source_run_url=(env.get("RUN_URL") or "").strip(),
        run_attempt=_int_env(env, "GITHUB_RUN_ATTEMPT", 1),
    )
    return freeze_context(
        (env.get("GITHUB_REPOSITORY") or "").strip(),
        collected,
        state,
        plan.max_tool_turns,
        execution=execution,
    )


def _write_prepared_context(env: dict[str, str], context: dict[str, Any]) -> Path:
    explicit = (env.get("REVIEW_CONTEXT_OUTPUT_FILE") or "").strip()
    if explicit:
        path = Path(explicit)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        directory = Path(env.get("ALL_LANE_RESULTS_DIR") or _work_dir(env))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "review-context.json"
    path.write_text(
        json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return path


def _write_prepared_outputs(
    context: dict[str, Any],
    *,
    publisher_judge_required: bool | None = None,
) -> None:
    restored = restore_context(context)
    assert restored.execution is not None
    plan = restored.execution.plan
    models = [lane.model for lane in plan.lanes]
    # Setup matrix jobs gate the publisher on judge_needed. That output names
    # whether a judge job must run, not whether an LLM judge call occurs.
    # Frozen runtime and _prepared_judge_needed retain the LLM decision.
    judge_for_outputs = (
        publisher_judge_required
        if publisher_judge_required is not None
        else _prepared_judge_needed(restored)
    )
    _write_setup_outputs(models, judge_for_outputs, plan.judge_model)
    _set_output(
        "policy_digest",
        restored.collected.review_policy.digest if restored.collected.review_policy else "",
    )
    _set_output("policy_base_sha", restored.collected.policy_base_sha)
    _set_output("review_profile", plan.profile)
    _set_output("review_level", plan.level)
    _set_output("review_trigger", plan.trigger)
    _set_output("registry_digest", plan.registry_digest)


def _load_prepared_context(env: dict[str, str]) -> tuple[dict[str, Any], ReviewContext]:
    path = (env.get("REVIEW_CONTEXT_FILE") or "").strip()
    expected = (env.get("REVIEW_CONTEXT_SHA256") or "").strip()
    if not path or not expected:
        raise ActionError(
            "prepared lane/judge requires review_context_file and review_context_sha256"
        )
    try:
        envelope = _read_bounded_json(Path(path), what="prepared context")
    except (OSError, json.JSONDecodeError, UnicodeError, RecursionError, SchemaError) as exc:
        raise SchemaError(f"prepared context file is unreadable: {exc}") from exc
    context = restore_context(envelope)
    if _context_digest(envelope) != expected:
        raise SchemaError("prepared context digest does not match review_context_sha256")
    execution = context.execution
    if execution is None:
        raise SchemaError("prepared context has no execution plan")
    supplied = {
        "GITHUB_REPOSITORY": context.repository,
        "PR_NUMBER": str(context.collected.pr_number),
        "HEAD_SHA": context.collected.head_sha,
    }
    for key, value in supplied.items():
        raw = (env.get(key) or "").strip()
        actual = raw.lower() if key == "HEAD_SHA" else raw
        if actual and actual != value:
            raise SchemaError(f"prepared context {key.lower()} does not match this job")
    if (env.get("RUN_URL") or "").strip() != execution.source_run_url:
        raise SchemaError("prepared context source_run_url does not match this job")
    if str(execution.run_attempt) != (env.get("GITHUB_RUN_ATTEMPT") or "1").strip():
        raise SchemaError("prepared context run_attempt does not match this job")
    return envelope, context


def _frozen_execution_env(env: dict[str, str], context: Any) -> dict[str, str]:
    assert context.execution is not None
    plan = context.execution.plan
    frozen = dict(env)
    for key in FROZEN_RUNTIME_KEYS:
        frozen.pop(key, None)
    frozen.update(json.loads(context.execution.runtime_json))
    frozen["MODELS"] = ",".join(lane.model for lane in plan.lanes)
    frozen["JUDGE_MODEL"] = plan.judge_model
    frozen["EFFORT"] = plan.effort
    frozen["MAX_TOOL_TURNS"] = str(plan.max_tool_turns)
    frozen["LANE_TIMEOUT_SECONDS"] = str(plan.lane_timeout_seconds)
    frozen["JOB_BUDGET_SECONDS"] = str(plan.job_budget_seconds)
    frozen["JUDGE_NEEDED"] = "true" if _prepared_judge_needed(context) else "false"
    routes = {
        lane.model: {
            **({"provider": lane.provider} if lane.provider else {}),
            **({"service_tier": lane.service_tier} if lane.service_tier else {}),
        }
        for lane in plan.lanes
        if lane.provider or lane.service_tier
    }
    frozen["MODEL_ROUTES"] = json.dumps(routes, separators=(",", ":"))
    remaining = max(0.0, context.execution.deadline_unix_ms / 1000 - time.time())
    frozen[_JOB_DEADLINE_KEY] = str(time.monotonic() + remaining)
    return frozen


def _prepared_judge_needed(context: Any) -> bool:
    """Use frozen explicit legacy intent, falling back to the panel shape."""
    runtime = json.loads(context.execution.runtime_json)
    return _env_flag(
        {"JUDGE_NEEDED": runtime.get("JUDGE_NEEDED", "")},
        "JUDGE_NEEDED",
        judge_is_needed([lane.model for lane in context.execution.plan.lanes]),
    )


def _read_bounded_json(path: Path, *, what: str) -> Any:
    """Read untrusted artifacts without allocating past their published cap."""
    with path.open("rb") as handle:
        raw = handle.read(MAX_CONTEXT_BYTES + 1)
    if len(raw) > MAX_CONTEXT_BYTES:
        raise SchemaError(f"{what} exceeds the 16 MiB artifact limit")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SchemaError(f"{what} has duplicate key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except SchemaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SchemaError(f"{what} is not strict JSON") from exc


def _resolve_issues(
    env: dict[str, str],
    slugs: list[str],
    lanes: list[LaneResult],
    successful: list[LaneResult],
) -> JudgeOutcome:
    lane_payloads = [lane.to_dict() for lane in successful]
    if not successful:
        return JudgeOutcome([], "skipped (no successful lanes)", None, False, [])
    reviewable_payloads, diagnostics = partition_reviewable_lanes(lane_payloads)
    needed = _judge_needed(env, slugs)
    if len(successful) == 1 and (len(slugs) > 1 or not needed):
        print("judge skipped: one successful lane; posting that lane directly")
        reason = (
            "skipped (single review lane; one reviewer = no judge)"
            if len(lanes) == 1
            else "skipped (one successful review lane; no merge needed)"
        )
        return JudgeOutcome(
            issues_from_single_lane(successful[0]), reason, None, False, diagnostics
        )
    if not needed:
        # Defensive only: model validation currently makes multiple lanes
        # imply a judge unless the caller explicitly overrides judge_needed.
        issues, note = _capped_union_note(
            lane_payloads, "skipped by configuration (deterministic union)"
        )
        return JudgeOutcome(
            issues,
            note,
            None,
            False,
            diagnostics,
        )
    judge_model = parse_judge_model(env.get("JUDGE_MODEL"))
    if diagnostics and not any(lane.get("findings") for lane in reviewable_payloads):
        print("judge skipped: successful lanes contained only review-environment diagnostics")
        return JudgeOutcome(
            [], "skipped (review environment unavailable)", None, False, diagnostics
        )
    judge_timeout = _judge_request_timeout(env)
    if judge_timeout is None:
        print(
            "judge skipped near the job deadline; using the cap-aware deterministic union",
            flush=True,
        )
        issues, note = _capped_union_note(
            lane_payloads, f"`{judge_model}` (deadline fallback: deterministic union)"
        )
        return JudgeOutcome(
            issues,
            note,
            None,
            False,
            diagnostics,
        )
    # One absolute boundary includes retries and policy/payload preparation.
    judge_deadline = _judge_deadline(env, judge_timeout)
    key = require_openrouter_key(env)
    print(
        f"judge running with `{judge_model}` (reasoning effort=minimal, "
        f"request timeout={judge_timeout}s)",
        flush=True,
    )
    try:
        policy_kwargs = {}
        if successful[0].review_context:
            saved = restore_context(successful[0].review_context)
            if saved.collected.review_policy is not None:
                policy_kwargs["policy_guidance"] = review_policy_block(saved.collected)
        issues, mode, judge_cost = run_llm_judge(
            model=judge_model,
            lanes=lane_payloads,
            api_key=key,
            timeout=judge_timeout,
            deadline=judge_deadline,
            **policy_kwargs,
        )
    except SchemaError as exc:
        # The lane artifacts already passed our schema and anchor gates. A
        # malformed judge answer falls back to the validated lane findings;
        # make the publication cap and degraded merge explicit on the review.
        print(
            f"warning: judge schema failed ({redact(str(exc))}); using the "
            "cap-aware deterministic union",
            flush=True,
        )
        issues, note = _capped_union_note(
            lane_payloads, f"`{judge_model}` (schema fallback: deterministic union)"
        )
        return JudgeOutcome(
            issues,
            note,
            getattr(exc, "cost_usd", None),
            True,
            diagnostics,
        )
    except ActionError as exc:
        # Review lanes are the source evidence; a judge transport failure
        # falls back to their completed work. Judge schema failures use the
        # same explicit, cap-aware deterministic-union degradation above.
        attempted = getattr(exc, "attempted", True)
        stage = "transport" if attempted else "preparation"
        print(
            f"warning: judge {stage} failed ({redact(str(exc))}); using the "
            "cap-aware deterministic union",
            flush=True,
        )
        issues, note = _capped_union_note(
            lane_payloads, f"`{judge_model}` ({stage} fallback: deterministic union)"
        )
        return JudgeOutcome(
            issues,
            note,
            None,
            attempted,
            diagnostics,
        )
    # Merge outcomes are visible on the posted review, not only in the job
    # log: readers must be able to tell a clean merge from a repaired or
    # fallback post. Source accounting is lossless before the global cap.
    if mode == "merged":
        return JudgeOutcome(issues, f"`{judge_model}`", judge_cost, True, diagnostics)
    return JudgeOutcome(
        issues,
        f"`{judge_model}` ({mode}: source coverage preserved before publication cap)",
        judge_cost,
        True,
        diagnostics,
    )


def _capped_union_note(lanes: list[dict[str, Any]], note: str) -> tuple[list[MergedIssue], str]:
    """Use the judge's cap-aware union and disclose any omitted findings."""
    issues, dropped = deterministic_union_with_cap(lanes)
    suffix = f" (capped+{dropped})" if dropped else ""
    return issues, f"{note}{suffix}"


def _role_all(env: dict[str, str]) -> int:
    if _prepared_requested(env):
        return _role_all_prepared(env)
    slugs = _validate_inputs(env, full_roster=True)
    needed = _judge_needed(env, slugs)
    # role=all needs the matrix metadata immediately, but judge outputs are
    # emitted once by _finish alongside the other public result outputs.
    _write_lane_setup_outputs(slugs)
    collected, state, agent_replies = _collect_with_loop(env)
    collected = _with_review_policy(env, collected, state)
    context = freeze_context(
        (env.get("GITHUB_REPOSITORY") or "").strip(),
        collected,
        state,
        parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
    )
    _maybe_status(
        env,
        collected.pr_number,
        f"Reviewing with OpenRouter ({len(slugs)} lane(s): {', '.join(f'`{s}`' for s in slugs)}).",
    )
    work = _work_dir(env)
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected, state, agent_replies)
    expect_coverage, expected_paths = _coverage_expectations(state, collected)
    expected_ids = _expected_resolution_ids(state, collected)
    remaining = _remaining_job_seconds(env)
    lane_timeout, judge_reserve = _lane_budget(
        remaining,
        judge_needed=needed,
        shares_job_with_judge=True,
    )
    lane_dir = Path(env.get("ALL_LANE_RESULTS_DIR") or (work / "lanes"))
    lane_dir.mkdir(parents=True, exist_ok=True)

    deadline_seconds = _all_role_deadline_seconds(env, remaining, judge_reserve)
    if remaining is not None:
        deadline_seconds = min(
            deadline_seconds,
            max(0, int(remaining - max(POST_RESERVE_SECONDS, judge_reserve))),
        )
    # Finish inside the collector's deadline, including explicit shorter overrides.
    # Do not extend the job or consume the judge/publication reserve.
    lane_timeout = min(
        lane_timeout,
        max(0.0, deadline_seconds - min(LANE_COLLECTION_GRACE_SECONDS, deadline_seconds / 2)),
    )
    print(
        f"lane budget={lane_timeout:g}s; collection deadline={deadline_seconds}s; "
        f"judge/publication reserve={judge_reserve}s",
        flush=True,
    )

    def _one(index: int, model: str) -> LaneResult:
        return _invoke_lane(
            env,
            model,
            messages,
            workspace,
            expect_coverage=expect_coverage,
            expect_resolutions=state.mode == "verify",
            expected_paths=expected_paths,
            expected_resolution_ids=expected_ids,
            lane_timeout=lane_timeout,
            progress=partial(_persist_lane_progress, lane_dir, index, model),
        )

    def run_lane_index(index: int) -> LaneResult:
        return _one(index, slugs[index])

    lanes = _collect_bounded_lanes(
        len(slugs),
        run_lane_index,
        deadline_seconds,
        on_lane=lambda index, lane: _persist_and_log_lane(
            lane_dir, index, slugs[index], lane, collected.head_sha, context
        ),
        salvage_progress=lambda index, lane: _restore_lane_progress(lane_dir, index, lane),
        model_for_index=lambda index: slugs[index],
    )
    return _finish(env, lanes, collected=collected, loop=state)


def _run_prepared_lane(
    env: dict[str, str],
    context: PreparedContext,
    index: int,
    *,
    inputs: PreparedLaneInputs | None = None,
    collector_deadline_monotonic: float | None = None,
) -> LaneResult:
    plan = context.execution.plan
    lane_plan = plan.lanes[index]
    frozen = inputs.frozen_env if inputs is not None else _frozen_execution_env(env, context)
    try:
        if inputs is None:
            work = _work_dir(frozen)
            workspace = _prepare_workspace(frozen, context.collected, work)
            messages = _messages(
                frozen, context.collected, context.loop, context.execution.agent_replies
            )
            coverage, paths = _coverage_expectations(context.loop, context.collected)
            inputs = PreparedLaneInputs(
                frozen,
                workspace,
                messages,
                coverage,
                paths,
                _expected_resolution_ids(context.loop, context.collected),
            )
        # Workspace materialization can consume the last usable time.  Check
        # after it completes, before looking up the paid-provider key.
        remaining = _remaining_job_seconds(frozen)
        lane_budget, _judge_reserve = _lane_budget(
            remaining,
            judge_needed=_prepared_judge_needed(context),
            shares_job_with_judge=True,
            lane_ceiling=plan.lane_timeout_seconds,
        )
        lane_timeout = min(plan.lane_timeout_seconds, lane_budget)
        if collector_deadline_monotonic is not None:
            remaining_cap = max(0.0, collector_deadline_monotonic - time.monotonic())
            lane_timeout = min(
                lane_timeout,
                max(
                    0,
                    int(remaining_cap - min(LANE_COLLECTION_GRACE_SECONDS, remaining_cap / 2)),
                ),
            )
        reserve = max(POST_RESERVE_SECONDS, _judge_reserve)
        if remaining is None or remaining <= reserve or lane_timeout < 1:
            result = failed_lane(
                lane_plan.model, "prepared review deadline expired before this lane could start"
            )
        else:
            result = _invoke_lane(
                frozen,
                lane_plan.model,
                inputs.messages,
                inputs.workspace,
                expect_coverage=inputs.expect_coverage,
                expect_resolutions=context.loop.mode == "verify",
                expected_paths=inputs.expected_paths,
                expected_resolution_ids=inputs.expected_resolution_ids,
                lane_timeout=lane_timeout,
            )
    except ActionError as exc:
        result = failed_lane(lane_plan.model, redact(str(exc)))
    result.head_sha = context.collected.head_sha
    result.lane_index = index
    result.required = lane_plan.required
    result.context_sha256 = context.digest
    result.review_context = context.envelope
    return result


def _context_digest_from_context(context: PreparedContext) -> str:
    return context.digest


def _prepared_context(envelope: dict[str, Any], restored: ReviewContext) -> PreparedContext:
    return PreparedContext(restored, _context_digest(envelope), envelope)


def _role_lane_prepared(env: dict[str, str]) -> int:
    _validate_prepared_context_inputs(env)
    envelope, restored = _load_prepared_context(env)
    context = _prepared_context(envelope, restored)
    plan = context.execution.plan
    index = _int_env(env, "LANE_INDEX", 0)
    if not 0 <= index < len(plan.lanes):
        raise ActionError("LANE_INDEX is out of range for the prepared review plan")
    override = (env.get("LANE_MODEL") or "").strip()
    if override and override != plan.lanes[index].model:
        raise SchemaError("prepared lane model override does not match its assigned plan lane")
    result = _run_prepared_lane(env, context, index)
    path = _write_lane_file(env, index, result)
    _set_output("lane_file", str(path))
    _set_output("lane_ok", "true" if result.ok else "false")
    # Matrix lanes fail open so the judge can publish surviving evidence and
    # the artifact uploader still receives this explicit failed result.
    return 0


def _role_all_prepared(env: dict[str, str]) -> int:
    _validate_prepared_context_inputs(env)
    supplied = (env.get("REVIEW_CONTEXT_FILE") or "").strip()
    if supplied:
        envelope, restored = _load_prepared_context(env)
        _write_prepared_outputs(envelope)
        _set_output("review_context_file", supplied)
        _set_output("review_context_sha256", _context_digest(envelope))
        _set_output("head_sha", restored.collected.head_sha)
    else:
        envelope = _prepare_execution(env)
        restored = restore_context(envelope)
        _write_prepared_outputs(envelope)
        context_path = _write_prepared_context(env, envelope)
        _set_output("review_context_file", str(context_path))
        _set_output("review_context_sha256", _context_digest(envelope))
        _set_output("head_sha", restored.collected.head_sha)
    context = _prepared_context(envelope, restored)
    directory = Path(env.get("ALL_LANE_RESULTS_DIR") or (_work_dir(env) / "lanes"))
    directory.mkdir(parents=True, exist_ok=True)
    frozen = _frozen_execution_env(env, context)
    # The all-role compatibility path is one shared job.  Materialize the
    # immutable checkout and prompt once, then start every lane concurrently.
    work = _work_dir(frozen)
    workspace = _prepare_workspace(frozen, context.collected, work)
    coverage, expected_paths = _coverage_expectations(context.loop, context.collected)
    inputs = PreparedLaneInputs(
        frozen,
        workspace,
        _messages(frozen, context.collected, context.loop, context.execution.agent_replies),
        coverage,
        expected_paths,
        _expected_resolution_ids(context.loop, context.collected),
    )
    plan = context.execution.plan
    _maybe_status(
        frozen,
        context.collected.pr_number,
        f"Reviewing with OpenRouter ({len(plan.lanes)} lane(s): "
        f"{', '.join(f'`{lane.model}`' for lane in plan.lanes)}).",
    )
    remaining = _remaining_job_seconds(frozen)
    _lane_timeout, judge_reserve = _lane_budget(
        remaining,
        judge_needed=_prepared_judge_needed(context),
        shares_job_with_judge=True,
        lane_ceiling=plan.lane_timeout_seconds,
    )
    deadline_seconds = _all_role_deadline_seconds(frozen, remaining, judge_reserve)
    # An explicit cap must not outlive the frozen absolute execution deadline
    # or consume the judge/publication reserve.
    if remaining is not None:
        deadline_seconds = min(
            deadline_seconds,
            max(0, int(remaining - max(POST_RESERVE_SECONDS, judge_reserve))),
        )

    collection_deadline = time.monotonic() + deadline_seconds

    def one(index: int) -> LaneResult:
        return _run_prepared_lane(
            frozen,
            context,
            index,
            inputs=inputs,
            collector_deadline_monotonic=collection_deadline,
        )

    lanes = _collect_all_prepared_lanes(plan, context, directory, one, deadline_seconds)
    return _finish(
        frozen,
        lanes,
        collected=context.collected,
        loop=context.loop,
        prepared_context=context,
    )


def _collect_bounded_lanes(
    count: int,
    run_one: Any,
    deadline_seconds: float,
    *,
    on_lane: Any,
    salvage_progress: Any | None = None,
    model_for_index: Any | None = None,
) -> list[LaneResult]:
    """Run up to ``count`` lanes concurrently under one collection deadline."""
    if count == 0:
        return []
    # A timed-out pool cannot stop requests already in flight. The lane
    # clock's per-request clamp guarantees those stragglers end within
    # their lane deadline after non-waiting shutdown returns control.
    pool = ThreadPoolExecutor(max_workers=min(count, LANE_CAP))
    timed_out = False
    by_index: dict[int, LaneResult] = {}
    futures = {pool.submit(run_one, index): index for index in range(count)}
    pending = set(futures)
    try:
        try:
            for future in as_completed(futures, timeout=max(0, deadline_seconds)):
                pending.discard(future)
                index = futures[future]
                try:
                    lane = future.result()
                except Exception as exc:  # noqa: BLE001
                    model = model_for_index(index) if model_for_index is not None else "unknown"
                    lane = failed_lane(model, redact(str(exc)))
                by_index[index] = lane
                on_lane(index, lane)
        except FutureTimeoutError:
            timed_out = True
            completed = len(by_index)
            for future in pending:
                index = futures[future]
                model = model_for_index(index) if model_for_index is not None else "unknown"
                if future.done():
                    try:
                        lane = future.result()
                    except Exception as exc:  # noqa: BLE001
                        lane = failed_lane(model, redact(str(exc)))
                else:
                    future.cancel()
                    lane = failed_lane(
                        model,
                        "role=all deadline reached before every lane finished; "
                        f"salvaging {completed}/{count} completed lane(s)",
                    )
                    if salvage_progress is not None:
                        salvage_progress(index, lane)
                by_index[index] = lane
                on_lane(index, lane)
    finally:
        pool.shutdown(wait=not timed_out, cancel_futures=timed_out)
    return [by_index[index] for index in range(count)]


def _collect_all_prepared_lanes(
    plan: ReviewPlan,
    context: PreparedContext,
    directory: Path,
    run_one: Any,
    deadline_seconds: float,
) -> list[LaneResult]:
    """Concurrent bounded collector with immediate persistence and salvage."""
    count = len(plan.lanes)
    return _collect_bounded_lanes(
        count,
        run_one,
        deadline_seconds,
        on_lane=lambda index, lane: _persist_and_log_lane(
            directory,
            index,
            plan.lanes[index].model,
            lane,
            context.collected.head_sha,
            context.envelope,
        ),
        salvage_progress=lambda index, lane: _restore_lane_progress(directory, index, lane),
        model_for_index=lambda index: plan.lanes[index].model,
    )


def _role_judge_prepared(env: dict[str, str]) -> int:
    _validate_prepared_context_inputs(env)
    envelope, restored = _load_prepared_context(env)
    context = _prepared_context(envelope, restored)
    directory = Path(env.get("LANE_RESULTS_DIR") or "")
    if not directory.is_dir():
        raise ActionError("LANE_RESULTS_DIR is missing or not a directory")
    lanes = _load_prepared_lane_dir(directory, context)
    return _finish(
        _frozen_execution_env(env, context),
        lanes,
        collected=context.collected,
        loop=context.loop,
        prepared_context=context,
    )


def _run_one_lane(env: dict[str, str], model: str) -> tuple[LaneResult, CollectedReview, LoopState]:
    collected, state, agent_replies = _collect_with_loop(env)
    context = freeze_context(
        (env.get("GITHUB_REPOSITORY") or "").strip(),
        collected,
        state,
        parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
    )
    work = _work_dir(env)
    workspace = _prepare_workspace(env, collected, work)
    messages = _messages(env, collected, state, agent_replies)
    _maybe_status(env, collected.pr_number, f"Lane `{model}` is reviewing via OpenRouter.")
    expect_coverage, expected_paths = _coverage_expectations(state, collected)
    remaining = _remaining_job_seconds(env)
    lane_timeout, _judge_reserve = _lane_budget(
        remaining,
        judge_needed=_judge_needed(env, parse_models(env.get("MODELS"))),
        shares_job_with_judge=False,
    )
    result = _invoke_lane(
        env,
        model,
        messages,
        workspace,
        expect_coverage=expect_coverage,
        expect_resolutions=state.mode == "verify",
        expected_paths=expected_paths,
        expected_resolution_ids=_expected_resolution_ids(state, collected),
        lane_timeout=lane_timeout,
    )
    result.head_sha = collected.head_sha
    result.review_context = context
    return result, collected, state


def _invoke_lane(
    env: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    workspace: Path | None,
    *,
    expect_coverage: bool = False,
    expect_resolutions: bool = False,
    expected_paths: set[str] | None = None,
    expected_resolution_ids: set[str] | None = None,
    lane_timeout: float | None = None,
    progress: ProgressFn | None = None,
) -> LaneResult:
    try:
        key = require_openrouter_key(env)
        lane_kwargs: dict[str, Any] = parse_model_routes(env.get("MODEL_ROUTES")).get(model, {})
        if progress is not None:
            lane_kwargs["progress"] = progress
        if lane_timeout is not None:
            lane_kwargs["lane_timeout"] = lane_timeout
        return run_lane(
            model=model,
            messages=messages,
            api_key=key,
            workspace=workspace,
            max_tool_turns=parse_max_tool_turns(env.get("MAX_TOOL_TURNS")),
            effort=(env.get("EFFORT") or "").strip(),
            timeout=_int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180),
            # Tool-less runs have no inert checkout; the anchor gate then
            # checks against the workflow's own full checkout of the head.
            anchor_root=_source_root(env) if workspace is None else None,
            expect_coverage=expect_coverage,
            expect_resolutions=expect_resolutions,
            expected_paths=expected_paths,
            expected_resolution_ids=expected_resolution_ids,
            **lane_kwargs,
        )
    except ActionError:
        raise
    except Exception as exc:  # noqa: BLE001
        return failed_lane(model, redact(str(exc)))


def _new_generation() -> str:
    """A fresh nonce per initial round.

    Deriving the token from the reviewed SHA would reuse it when a loop is
    reset at the same commit, letting old inline threads pair with new
    same-numbered findings.
    """
    return secrets.token_hex(6)


def _coverage_expectations(
    state: LoopState, collected: CollectedReview
) -> tuple[bool, set[str] | None]:
    """Whether this run enforces the coverage manifest, and for which paths.

    A diff naming more paths than a manifest may hold would make every lane
    unsatisfiable (the prompt demands every file while the parser caps the
    array), so enforcement degrades to unenforced with a visible notice.
    """
    if state.mode != "initial" and collected.plan.kind != "full-pr":
        return False, None
    paths = set(changed_paths_from_diff(collected.diff))
    if len(paths) > MAX_COVERAGE_ENTRIES:
        print(
            f"notice: {len(paths)} diff paths exceed the coverage manifest cap "
            f"({MAX_COVERAGE_ENTRIES}); coverage enforcement is skipped for this run"
        )
        return False, None
    return True, paths


def _expected_resolution_ids(
    state: LoopState, collected: CollectedReview | None = None
) -> set[str] | None:
    """Prior finding ids a verify lane must explicitly resolve."""
    if state.mode != "verify":
        return None
    findings = (
        state.prior_findings
        if collected is not None and collected.plan.scope == "rebase"
        else state.open_prior
    )
    return {finding.id for finding in findings}


def _lane_budget(
    remaining: float | None,
    *,
    judge_needed: bool,
    shares_job_with_judge: bool,
    lane_ceiling: int = DEFAULT_LANE_TIMEOUT_SECONDS,
) -> tuple[int, int]:
    """Return the lane timeout and same-job judge reserve."""
    judge_reserve = 0
    if shares_job_with_judge and judge_needed:
        # Reviewers produce the evidence. The tool-free judge gets one bounded
        # window shared by all retries; timeout falls back to the validated union.
        judge_reserve = (
            POST_RESERVE_SECONDS + JUDGE_BUDGET_SECONDS + JUDGE_SCHEDULING_MARGIN_SECONDS
        )
    if remaining is None:
        return lane_ceiling, judge_reserve
    reserve = (
        max(POST_RESERVE_SECONDS, judge_reserve) if shares_job_with_judge or not judge_needed else 0
    )
    timeout = max(1, min(lane_ceiling, int(remaining - reserve)))
    return timeout, judge_reserve


def _bot_login(env: dict[str, str]) -> str:
    login = (env.get("BOT_LOGIN") or "").strip() or DEFAULT_BOT_LOGIN
    if len(login) > 100 or any(character.isspace() for character in login):
        raise ActionError("bot_login must be a GitHub login of at most 100 characters")
    return login


def _resolve_loop(
    env: dict[str, str], github: GitHub, pr_number: int
) -> tuple[Ledger | None, LoopState]:
    """Recover the loop position before collecting the diff.

    State recovery fails closed: a corrupted newest ledger raises instead of
    silently resetting to round 1, and a state-free synchronize run under
    latest-commit scope is refused (an "initial" review of one push could
    report clean without ever seeing the rest of the PR).
    """
    mode_input = parse_mode(env.get("REVIEW_MODE") or "auto")
    event_action = (env.get("EVENT_ACTION") or "").strip().lower()
    scope = parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    repo = (env.get("GITHUB_REPOSITORY") or "").strip()
    ledger: Ledger | None = None
    if mode_input in {"verify", "auto"}:
        bodies = github.list_bot_review_bodies(pr_number, _bot_login(env))
        ledger = latest_ledger(bodies, repo=repo, pr_number=pr_number)
        if ledger is None and mode_input == "verify":
            raise ActionError(
                "review_mode is verify but no prior review-loop state exists on "
                "this PR; run an initial review first"
            )
        if ledger is None and event_action == "synchronize" and scope == "latest-commit":
            raise ActionError(
                "this synchronize run collects only the latest commit and no prior "
                "review-loop state exists, so carried findings cannot be verified; "
                "run an initial full-PR review first (review_mode: initial, "
                "review_scope: full-pr)"
            )
    mode, round_number = decide_loop_state(
        review_mode=mode_input, event_action=event_action, ledger=ledger
    )
    if scope == "rebase" and (mode != "verify" or ledger is None):
        raise ActionError("rebase scope requires an existing ledger and auto or verify mode")
    prior = ledger.findings if ledger is not None and mode == "verify" else ()
    generation = ledger.generation if ledger is not None and mode == "verify" else ""
    prior, retired = apply_severity_floor(
        prior, round_number if mode == "verify" and scope != "rebase" else 1
    )
    return ledger, LoopState(
        mode=mode,
        round_number=round_number,
        prior_findings=prior,
        generation=generation,
        retired_prior=retired,
    )


def _collect_with_loop(
    env: dict[str, str], *, with_replies: bool = True
) -> tuple[CollectedReview, LoopState, str]:
    pr_number = _pr_number(env)
    github = _github(env)
    ledger, state = _resolve_loop(env, github, pr_number)
    env_for_collect = dict(env)
    env_for_collect["REVIEW_MODE"] = state.mode
    if state.mode == "verify" and ledger is not None and ledger.reviewed_sha:
        # Continuity: verify everything since the last successfully published
        # review, not just this push's webhook range, so a run racing a
        # cancelled older run can never skip the commits that run covered.
        env_for_collect["EVENT_BEFORE"] = ledger.reviewed_sha
    collected = _collect(env_for_collect)
    if (
        state.mode == "verify"
        and ledger is not None
        and collected.plan.fallback_notice == DIVERGED_NOTICE
    ):
        # History was rewritten (force-push): the last reviewed SHA is no
        # longer an ancestor, so a latest-commit verify can never cover the
        # rewritten work — and its partial verdict would never republish the
        # ledger, repeating the identical failed round forever. Expand the
        # acquisition to the full PR while preserving finding identities,
        # dispositions, and replies: rewritten commits do not reset a review.
        # Transient compare failures (timeouts, 5xx) carry a different notice
        # and never reset: they stay a single-commit partial round and retry
        # naturally on the next push.
        print(
            "notice: history diverged from the last reviewed commit "
            f"({ledger.reviewed_sha[:12]}); collecting a rebase review with existing history"
        )
        env_full = dict(env_for_collect)
        env_full.update(
            REVIEW_SCOPE="rebase", HEAD_SHA=collected.head_sha, EVENT_AFTER=collected.head_sha
        )
        full = _collect(env_full)
        if full.head_sha != collected.head_sha or full.policy_base_sha != collected.policy_base_sha:
            raise ActionError(
                "head/base changed while acquiring the rebase diff; retry this review"
            )
        collected = full
        state = replace(state, prior_findings=ledger.findings, retired_prior=())
    agent_replies, summarized = "", False
    # Shared-context roles only: legacy matrix lanes must collect identical contexts.
    shared = with_replies and env.get("ROLE") in {"all", "setup"}
    session = _brief_session(env, github, collected, state) if shared else None
    if collected.plan.scope == "rebase" and with_replies:
        bodies = github.list_bot_review_bodies(pr_number, _bot_login(env))
        if latest_ledger(bodies, repo=env["GITHUB_REPOSITORY"], pr_number=pr_number) != ledger:
            raise ActionError("review history changed while acquiring rebase context; retry")
        # History is part of this review's contract. A failed history read must
        # not silently downgrade a rebase into a review without its rebuttals.
        history = compact_history(
            budget=MAX_REBASE_CONTEXT_BYTES,
            bot_login=_bot_login(env),
            review_bodies=bodies,
            replies=github.list_rebase_replies(pr_number, _bot_login(env)),
            comments=github.list_recent_issue_comments(pr_number),
            carried_ids={f"{state.generation}/{finding.id}" for finding in state.prior_findings},
            ledger_findings=state.prior_findings,
            summary_for_evicted=session.summarize_evicted if session else None,
        )
        print(history.stats_line(), file=sys.stderr)
        return _apply_brief(session, collected, history.summarized), state, history.text
    if with_replies and state.mode == "verify":
        try:
            # Replies to findings the severity floor retired would reintroduce
            # the retired context and invite re-adjudication; only threads for
            # carried findings (open or disputed) reach the prompt.
            carried_ids = {finding.id for finding in state.prior_findings}
            history = compact_history(
                budget=MAX_REPLIES_BYTES,
                bot_login=_bot_login(env),
                replies=[
                    reply
                    for reply in github.list_finding_replies(pr_number, generation=state.generation)
                    if reply[0] in carried_ids
                ],
                comments=github.list_recent_issue_comments(pr_number),
                summary_for_evicted=session.summarize_evicted if session else None,
            )
            print(history.stats_line(), file=sys.stderr)
            agent_replies, summarized = history.text, history.summarized
        except ActionError as exc:
            print(
                "warning: could not fetch reviewer replies; verifying from "
                f"commits only: {redact(str(exc))}"
            )
    return _apply_brief(session, collected, summarized), state, agent_replies


def _review_event(env: dict[str, str], verdict: str, open_issues: int) -> str:
    """APPROVE only when opted in and the review is clean; never REQUEST_CHANGES."""
    mode = (env.get("APPROVE") or "off").strip().lower()
    if mode not in {"off", "clean"}:
        raise ActionError("approve must be 'off' or 'clean'")
    return "APPROVE" if mode == "clean" and verdict == "clean" and not open_issues else "COMMENT"


def _post_review(
    github: GitHub,
    number: int,
    body: str,
    sha: str,
    comments: list[dict[str, Any]],
    event: str,
) -> dict[str, Any]:
    extra: dict[str, Any] = {"comments": comments} if comments else {}
    if event == "COMMENT":
        return github.create_review(number, body, sha, **extra)
    try:
        return github.create_review(number, body, sha, event=event, **extra)
    except ActionError as exc:
        # Repos that forbid Actions approvals still get the review, as a comment.
        print(f"warning: approval was rejected; posting as a comment: {redact(str(exc))}")
        return github.create_review(number, body, sha, **extra)


def _brief_session(
    env: dict[str, str], github: GitHub, collected: CollectedReview, state: LoopState
) -> BriefSession | None:
    """Re-reviews only; an empty compaction_model keeps the deterministic path."""
    if state.mode != "verify":
        return None
    compaction = (env.get("COMPACTION") or "auto").strip().lower()
    if compaction not in {"auto", "off"}:
        raise ActionError("compaction must be 'auto' or 'off'")
    raw_model = (env.get("COMPACTION_MODEL") or "").strip()
    model = parse_slug(raw_model, what="compaction_model") if raw_model else ""
    if compaction == "off":
        model = ""
    bodies: list[str] = []
    if model:
        try:
            bodies = github.list_bot_review_bodies(collected.pr_number, _bot_login(env))
        except ActionError:
            bodies = []
    timeout = min(_int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180), BRIEF_TIMEOUT_SECONDS)

    def chat(payload: dict[str, Any]) -> dict[str, Any]:
        key = require_openrouter_key(env)
        return openrouter_chat(key, payload, timeout=timeout, deadline=time.monotonic() + timeout)

    return BriefSession(
        body=collected.body,
        model=model,
        chat=chat,
        cached_bodies=bodies,
        route=parse_model_routes(env.get("MODEL_ROUTES")).get(model, {}) if model else {},
        repo=(env.get("GITHUB_REPOSITORY") or "").strip(),
        pr=collected.pr_number,
    )


def _apply_brief(
    session: BriefSession | None, collected: CollectedReview, summarized: bool = False
) -> CollectedReview:
    if session is None:
        return collected
    # The history may have dropped an oversized summary; then the body block renders it.
    session.rendered_in_history = session.rendered_in_history and summarized
    body, block = session.compact_body()
    session.log(block)
    return replace(collected, body=body, body_brief=block, brief_marker=session.marker())


def _collect(env: dict[str, str]) -> CollectedReview:
    pr_number = _pr_number(env)
    github = _github(env)
    scope = parse_scope(env.get("REVIEW_SCOPE") or "full-pr")
    mode = resolve_mode(parse_mode(env.get("REVIEW_MODE") or "auto"), env.get("EVENT_ACTION"))
    max_diff_kb = _int_env(env, "MAX_DIFF_KB", DEFAULT_MAX_DIFF_KB)
    return collect_review(
        pr_number=pr_number,
        scope=scope,
        mode=mode,
        before_sha=env.get("EVENT_BEFORE"),
        after_sha=env.get("EVENT_AFTER"),
        head_sha=env.get("HEAD_SHA"),
        max_diff_kb=max_diff_kb,
        source=github,
        gitattributes_text=_gitattributes_text(env),
        generated_globs=parse_generated_globs(env.get("GENERATED_PATHS")),
    )


def _gitattributes_text(env: dict[str, str]) -> str:
    """Best-effort .gitattributes AT THE REVIEWED COMMIT, for triage.

    `git show <head>:.gitattributes` against the reviewed checkout's object
    store pins the read to the reviewed commit and fails soft (empty string,
    heuristics-only packing) when that checkout does not contain the commit.

    Trust note: .gitattributes is repository content, so on a PR it is
    contributor-controlled. Honoring linguist-generated is still strictly
    safer than the pre-triage behavior it replaces: a demoted file keeps its
    stub, its coverage obligation, and tool access, while under the raw byte
    cut an attacker could push hand-written code beyond the cutoff entirely
    (no stub, no coverage, permanent partial). Demotion can never remove a
    file from review.
    """
    root = _source_root(env)
    head = (env.get("HEAD_SHA") or env.get("EVENT_AFTER") or "").strip()
    if root is None or not head:
        return ""
    try:
        head = require_full_sha(head, "head")
    except ActionError:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "show", f"{head}:.gitattributes"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _source_root(env: dict[str, str]) -> Path | None:
    """The reviewed repository checkout, if the caller supplied one."""
    raw = (env.get("SOURCE_WORKSPACE") or env.get("GITHUB_WORKSPACE") or "").strip()
    if not raw:
        return None
    root = Path(raw).resolve()
    return root if root.is_dir() else None


def _work_dir(env: dict[str, str]) -> Path:
    """The action-owned temporary work directory, consistently resolved."""
    return Path(env.get("WORK") or env.get("RUNNER_TEMP") or "/tmp").resolve()


def _checkout_has_commit(root: Path | None, sha: str) -> bool:
    """Whether ``root`` is checked out exactly at the reviewed ``sha``.

    A directory existing is insufficient in a reusable judge job: that job
    also has an action checkout. Object availability is insufficient too: a
    checkout can contain the reviewed commit while its worktree is at another
    commit, which would validate anchors against the wrong file contents.
    """
    if root is None:
        return False
    try:
        checked = require_full_sha(sha, "reviewed")
    except ActionError:
        return False
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    checkout_head = (getattr(proc, "stdout", "") or "").strip().lower()
    if proc.returncode != 0 or checkout_head != checked:
        print(
            "notice: reviewed checkout is not at the lane commit; leaving judge anchors body-only"
        )
        return False
    return True


def _prepare_workspace(env: dict[str, str], collected: CollectedReview, work: Path) -> Path | None:
    if parse_max_tool_turns(env.get("MAX_TOOL_TURNS")) == 0:
        return None
    source = _source_root(env)
    if source is None:
        raise ActionError("SOURCE_WORKSPACE is missing; cannot materialize the reviewed checkout")
    dest = work / "inert-checkout"
    if dest.exists() and any(dest.iterdir()):
        return dest
    try:
        # Stubbed files carry a tool-readability contract, so they may exceed
        # the normal materialization cap (bounded by the oversized ceiling).
        return materialize_commit(
            source,
            collected.head_sha,
            dest,
            oversized_ok=frozenset(collected.truncation.stubbed_files),
        )
    except ActionError as exc:
        # Fail closed: the prompt mandates blast-radius tool use, so a
        # silently tool-less run could post an unmarked glance review.
        raise ActionError(
            "inert checkout unavailable; refusing a tool-less review "
            f"(set max_tool_turns: 0 to review without tools): {redact(str(exc))}"
        ) from exc


def _messages(
    env: dict[str, str],
    collected: CollectedReview,
    state: LoopState | None = None,
    agent_replies: str = "",
) -> list[dict[str, str]]:
    custom = env.get("CUSTOM_INSTRUCTIONS") or ""
    tone = (env.get("ROAST_LEVEL") or "professional").strip().lower()
    # persona is reserved and unused; passed through so a later release can read it.
    return build_messages(
        collected,
        custom_instructions=custom,
        tone=tone,
        persona=env.get("PERSONA") or "",
        loop=state,
        agent_replies=agent_replies,
        path_profiles=parse_path_profiles(env.get("PATH_PROFILES")),
    )


def _with_review_policy(
    env: dict[str, str], collected: CollectedReview, state: LoopState
) -> CollectedReview:
    if (env.get("REVIEW_POLICY") or "off").strip().lower() != "base":
        return collected
    root = _source_root(env)
    if root is None:
        raise ActionError("review_policy=base requires the full-depth source checkout")
    policy = resolve_policy(
        root,
        collected.policy_base_sha,
        collected.head_sha,
        tuple(f.file for f in state.prior_findings if f.file),
        timeout=_int_env(env, "GITHUB_TIMEOUT_SECONDS", 120),
    )
    if parse_review_profiles(env.get("REVIEW_PROFILES")) is None:
        if policy.profile not in {"code", "docs"}:
            raise ActionError(f"unsupported review policy profile: {policy.profile}")
        if policy.minimum != "standard":
            raise ActionError(
                "deep review requires a configured deep profile; guidance-only mode cannot "
                "fulfill it"
            )
    _set_output("policy_digest", policy.digest)
    _set_output("policy_base_sha", policy.base_sha)
    print(
        f"review policy: {len(policy.files)} file(s), "
        f"{policy.profile}/{policy.minimum}, {policy.digest}"
    )
    return replace(collected, review_policy=policy)


def _finish(
    env: dict[str, str],
    lanes: list[LaneResult],
    collected: CollectedReview | None = None,
    loop: LoopState | None = None,
    prepared_context: Any | None = None,
) -> int:
    if collected is None or loop is None:
        collected, loop, _replies = _collect_with_loop(env, with_replies=False)
    # Lanes stamp the commit they actually reviewed; mixed artifacts are
    # irreconcilable and fail closed before anything posts.
    reviewed_sha = _common_lane_sha(lanes) or collected.head_sha
    successful = [lane for lane in lanes if lane.ok]
    plan = prepared_context.execution.plan if prepared_context is not None else None
    slugs = (
        [lane.model for lane in plan.lanes] if plan is not None else parse_models(env.get("MODELS"))
    )
    _write_judge_outputs(
        _judge_needed(env, slugs),
        plan.judge_model if plan is not None else parse_judge_model(env.get("JUDGE_MODEL")),
    )
    judge_outcome = _resolve_issues(env, slugs, lanes, successful)
    issues = judge_outcome.issues
    # Judge output bypasses the per-lane anchor gate, so gate the merged
    # issues too when a checkout demonstrably contains the reviewed head.
    # MergedIssue duck-types the
    # file/line/title fields the gate touches; single-lane issues were
    # already gated in run_lane and pass through unchanged.
    finish_root = _source_root(env)
    if _checkout_has_commit(finish_root, reviewed_sha):
        issues = sanitize_anchors(issues, finish_root)  # type: ignore[arg-type]

    prior_ids = _expected_resolution_ids(loop, collected) or set()
    resolutions = merge_resolutions([lane.resolutions for lane in successful], prior_ids)
    outcome = apply_round(
        loop, issues, resolutions, reassess_disputes=collected.plan.scope == "rebase"
    )
    issues = outcome.issues

    github = _github(env)
    stale_notice: str | None = None
    live_head = _live_head(github, collected.pr_number)
    if live_head is None:
        stale_notice = (
            "The current PR head could not be confirmed. "
            f"This review is pinned to commit {reviewed_sha[:12]} and is partial; "
            "it cannot publish authoritative loop state until the live head is verified."
        )
    elif live_head != reviewed_sha:
        stale_notice = (
            "The PR head advanced after this review's diff was collected. "
            f"This review is pinned to commit {reviewed_sha[:12]} and does not "
            "cover the newest push."
        )
    notices: list[str] = []
    if stale_notice:
        notices.append(stale_notice)
    if judge_outcome.environment_diagnostics:
        affected = ", ".join(
            f"`{model or 'unknown'}` ({title})"
            for model, title in judge_outcome.environment_diagnostics
        )
        notices.append(
            "One or more lanes reported that the supplied review environment "
            f"could not be inspected: {affected}. This review is partial and "
            "must not be treated as a clean pass."
        )
    if loop.mode == "initial" or collected.plan.kind == "full-pr":
        diff_path_set = set(changed_paths_from_diff(collected.diff))
        for lane in lanes:
            if lane.ok and lane.coverage:
                for note in coverage_count_mismatches(lane.findings, lane.coverage, diff_path_set):
                    notices.append(f"`{lane.model}`: {note}")
    # Diff-budget triage: stub-only truncation (every changed file embedded
    # or stubbed) does not force partial — only dropped files or a raw byte
    # cut do, so dense PRs keep verdicts, ledger publication, and loop
    # continuity. The stub contract is that TOOLS sweep the stubbed files,
    # so a tool-less run cannot honor it: stubs + max_tool_turns=0 stays a
    # partial review.
    truncation_partial = collected.truncation.forces_partial
    for lane in successful:
        if lane.dropped_findings:
            truncation_partial = True
            notices.append(
                f"`{lane.model}` omitted {lane.dropped_findings} finding(s) at the lane cap; "
                "the strongest severities were retained. This review is partial."
            )
    if collected.truncation.stubbed_files and parse_max_tool_turns(env.get("MAX_TOOL_TURNS")) == 0:
        truncation_partial = True
        notices.append(
            "Diff-budget triage stubbed "
            f"{len(collected.truncation.stubbed_files)} file(s) but tools are "
            "disabled (max_tool_turns: 0), so the stubbed files could not be "
            "swept. This review is partial."
        )
    verdict = decide_verdict(
        issues=issues,
        truncated=truncation_partial or bool(judge_outcome.environment_diagnostics),
        successful_lanes=len(successful),
        fallback=collected.plan.fallback_notice is not None,
        stale=stale_notice is not None,
    )
    if verdict == "clean" and outcome.open_issue_count:
        # Carried findings from earlier rounds are still open.
        verdict = "issues"

    profile_satisfied = True
    panel_status = ""
    receipt: dict[str, Any] | None = None
    if plan is not None:
        required_failures = [
            item
            for expected, item in zip(plan.lanes, lanes, strict=True)
            if expected.required and not item.ok
        ]
        optional_failures = [
            item
            for expected, item in zip(plan.lanes, lanes, strict=True)
            if not expected.required and not item.ok
        ]
        remaining = _remaining_job_seconds(env)
        expired = remaining is not None and remaining <= 0
        environment_failed = bool(judge_outcome.environment_diagnostics)
        # A profile may be degraded by an optional lane, but it is only
        # satisfied when the resulting review is still authoritative.  This
        # prevents a stale, expired, partial-diff, or all-failed optional
        # panel from publishing a clean ledger.
        profile_satisfied = not required_failures and not expired and not environment_failed
        if required_failures:
            panel_status = "required_missing"
            verdict = "partial" if successful else "error"
            notices.append(
                "One or more required review-plan lanes did not complete; "
                "the profile is unsatisfied."
            )
        elif (
            optional_failures
            or (len(plan.lanes) > 1 and not judge_outcome.ran)
            or "fallback" in judge_outcome.note
        ):
            panel_status = "degraded"
        else:
            panel_status = "complete"
        if expired and panel_status == "complete":
            panel_status = "degraded"
        if not profile_satisfied and verdict in {"clean", "issues"}:
            verdict = "partial" if successful else "error"
        if verdict in {"partial", "error"}:
            profile_satisfied = False
            if panel_status == "complete":
                panel_status = "degraded"
        _set_output("review_profile", plan.profile)
        _set_output("review_level", plan.level)
        _set_output("review_trigger", plan.trigger)
        _set_output("registry_digest", plan.registry_digest)
        _set_output("profile_satisfied", "true" if profile_satisfied else "false")
        _set_output("panel_status", panel_status)
        _set_output("review_context_sha256", _context_digest_from_context(prepared_context))
        receipt = {
            "version": 1,
            "repository": prepared_context.repository,
            "pr_number": collected.pr_number,
            "head_sha": reviewed_sha,
            "policy_base_sha": collected.policy_base_sha,
            "policy_digest": collected.review_policy.digest if collected.review_policy else "",
            "profile": plan.profile,
            "level": plan.level,
            "trigger": plan.trigger,
            "registry_digest": plan.registry_digest,
            "context_sha256": _context_digest_from_context(prepared_context),
            "required_models": [lane.model for lane in plan.lanes if lane.required],
            "successful_models": [lane.model for lane in successful],
            "panel_status": panel_status,
            "profile_satisfied": profile_satisfied,
            "verdict": verdict,
            "scope": collected.plan.scope,
            "mode": loop.mode,
            "run_url": prepared_context.execution.source_run_url
            or (env.get("RUN_URL") or "").strip(),
            "run_attempt": prepared_context.execution.run_attempt,
        }

    # The generation token scopes inline finding markers to this loop
    # generation; a reset mints a new one so old threads can never pair with
    # new same-numbered findings.
    generation = loop.generation if loop.mode == "verify" and loop.generation else _new_generation()
    hidden_marker: str | None = None
    if verdict in {"clean", "issues"}:
        # A partial or error run never publishes authoritative loop state;
        # the previous marker remains the retry boundary, so a truncated or
        # fallback diff cannot permanently skip unseen code.
        hidden_marker = encode_ledger(
            replace(outcome.ledger, reviewed_sha=reviewed_sha, generation=generation),
            repo=(env.get("GITHUB_REPOSITORY") or "").strip(),
            pr_number=collected.pr_number,
        )

    # The brief cache rides beside the ledger; drop it before it could crowd the body.
    if collected.brief_marker and len(hidden_marker or "") <= MAX_LEDGER_WITH_BRIEF:
        hidden_marker = "\n".join(m for m in (hidden_marker, collected.brief_marker) if m)
    bodies = render_review_parts(
        collected=collected,
        lanes=lanes,
        issues=issues,
        verdict=verdict,
        run_url=env.get("RUN_URL") or "",
        judge_note=judge_outcome.note,
        judge_cost=judge_outcome.cost,
        judge_ran=judge_outcome.ran,
        reviewed_sha=reviewed_sha,
        extra_notices=notices or None,
        hidden_marker=hidden_marker,
        round_lines=round_report(loop, outcome) or None,
        receipt=receipt,
    )
    comments: list[dict[str, Any]] = []
    if verdict in {"clean", "issues"}:
        # Partial runs never post inline comments: their diff (stale head,
        # truncated, or fallback) is not what the anchors were computed
        # against, and a same-round retry would re-issue the same finding ids.
        comments = inline_review_comments(
            issues,
            allowed_lines=diff_right_side_lines(collected.diff),
            generation=generation,
        )
    review_url = ""
    event = _review_event(env, verdict, outcome.open_issue_count)
    try:
        posted = _post_review(
            github, collected.pr_number, bodies[0], reviewed_sha, comments, event
        )
        html = posted.get("html_url")
        review_url = html if isinstance(html, str) else ""
    except ActionError as exc:
        _maybe_status(
            env,
            collected.pr_number,
            render_incomplete(
                stage="post-review",
                reason=redact(str(exc)),
                run_url=env.get("RUN_URL") or "",
            ),
        )
        raise ActionError(f"failed to post the GitHub review: {exc}") from exc

    if receipt is not None:
        _set_output("review_receipt_file", str(_write_review_receipt(env, receipt)))

    for continuation in bodies[1:]:
        try:
            github.create_issue_comment(collected.pr_number, continuation)
        except ActionError as exc:
            print(f"warning: could not post a continuation comment: {redact(str(exc))}")

    _maybe_status(
        env,
        collected.pr_number,
        f"OpenRouter review posted (`{verdict}`). {review_url}".strip(),
    )

    _set_output("verdict", verdict)
    _set_output("issue_count", str(outcome.open_issue_count))
    _set_output("bug_count", str(outcome.open_bug_count))
    _set_output("round", str(loop.round_number))
    _set_output("review_url", review_url)

    if verdict == "error":
        _error("every model lane failed; nothing structured arrived to post")
        return 1

    if plan is not None and not profile_satisfied:
        _error("review profile is unsatisfied")
        return 1

    # Every role passes through _validate_inputs before reaching _finish.
    fail_on = (env.get("FAIL_ON") or "never").strip().lower()
    if fail_on_should_fail(
        fail_on,
        issues,
        open_issue_count=outcome.open_issue_count,
        open_bug_count=outcome.open_bug_count,
    ):
        _error(
            f"fail_on={fail_on} matched {outcome.open_issue_count} open finding(s) "
            f"({outcome.open_bug_count} bug)"
        )
        return 1
    return 0


def _common_lane_sha(lanes: list[LaneResult]) -> str | None:
    """The one commit every lane reviewed, or None when no lane recorded one.

    Mixed artifacts (lanes that reviewed different commits) fail closed:
    merging findings from two different code states would attribute results
    to a commit no lane actually reviewed.
    """
    shas = {lane.head_sha for lane in lanes if lane.head_sha}
    if len(shas) > 1:
        listed = ", ".join(sorted(sha[:12] for sha in shas))
        raise SchemaError(
            f"lane artifacts reviewed different commits ({listed}); refusing to merge them"
        )
    return next(iter(shas), None)


def _live_head(github: GitHub, pr_number: int) -> str | None:
    try:
        pr = github.pr_view(pr_number)
    except ActionError as exc:
        print(f"warning: could not re-check the live PR head: {redact(str(exc))}")
        return None
    return head_sha_from_pr(pr)


def _load_lane_dir(directory: Path, expected: list[str]) -> list[LaneResult]:
    files = sorted(directory.rglob("lane-*.json"))
    by_index: dict[int, Path] = {}
    for path in files:
        stem = path.stem  # lane-0
        if not stem.startswith("lane-"):
            continue
        suffix = stem.split("-", 1)[1]
        if suffix.isdigit():
            if int(suffix) in by_index:
                raise SchemaError(f"duplicate matrix lane index: {suffix}")
            by_index[int(suffix)] = path

    lanes: list[LaneResult] = []
    for index, model in enumerate(expected):
        path = by_index.get(index)
        if path is None:
            # also accept a lone file for a 1-lane job
            if len(expected) == 1 and files:
                path = files[0]
            else:
                lanes.append(
                    failed_lane(model, "lane artifact missing (job failed or was cancelled)")
                )
                continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaError(f"{path.name} is not valid JSON: {exc}") from exc
        artifact = parse_lane_artifact(payload)
        context = restore_context(artifact.review_context)
        if artifact.model != model or artifact.head_sha != context.collected.head_sha:
            raise SchemaError("matrix artifact model or reviewed head does not match its context")
        lanes.append(artifact)
    return lanes


def _load_prepared_lane_dir(directory: Path, context: PreparedContext) -> list[LaneResult]:
    """Load only artifacts bound to the standalone setup context.

    Do not use a surviving lane as a source of truth: setup is the authority,
    so a cancelled first lane cannot erase the matrix publication context.
    """
    assert context.execution is not None
    plan = context.execution.plan
    files = sorted(directory.rglob("*.json"))
    by_index: dict[int, Path] = {}
    for path in files:
        if not path.stem.startswith("lane-"):
            raise SchemaError(f"prepared lane artifacts include unexpected file: {path.name}")
        suffix = path.stem.removeprefix("lane-")
        if not suffix.isdigit():
            raise SchemaError(f"prepared lane artifact has invalid name: {path.name}")
        index = int(suffix)
        if index in by_index:
            raise SchemaError(f"duplicate matrix lane index: {index}")
        by_index[index] = path
    unexpected = set(by_index) - set(range(len(plan.lanes)))
    if unexpected:
        raise SchemaError("prepared lane artifacts contain an out-of-plan index")
    lanes: list[LaneResult] = []
    digest = context.digest
    for index, lane_plan in enumerate(plan.lanes):
        path = by_index.get(index)
        if path is None:
            missing = failed_lane(
                lane_plan.model, "lane artifact missing (job failed or was cancelled)"
            )
            missing.head_sha = context.collected.head_sha
            missing.lane_index = index
            missing.required = lane_plan.required
            missing.context_sha256 = digest
            missing.review_context = context.envelope
            lanes.append(missing)
            continue
        try:
            artifact = parse_lane_artifact(_read_bounded_json(path, what=path.name))
        except (OSError, json.JSONDecodeError, UnicodeError, RecursionError, SchemaError) as exc:
            raise SchemaError(f"{path.name} is not valid JSON: {exc}") from exc
        if (
            artifact.lane_index != index
            or artifact.model != lane_plan.model
            or artifact.required != lane_plan.required
            or artifact.context_sha256 != digest
            or artifact.head_sha != context.collected.head_sha
            or artifact.review_context != context.envelope
        ):
            raise SchemaError("prepared lane artifact does not match its assigned frozen context")
        # Also strict-parse the embedded context; equality above detects a
        # forged digest/envelope pair while restore enforces its structure.
        if restore_context(artifact.review_context) != context.context:
            raise SchemaError("prepared lane artifact carries a different full context")
        lanes.append(artifact)
    return lanes


def _write_lane_file(env: dict[str, str], index: int, result: LaneResult) -> Path:
    # Persist outside the action's mktemp WORK dir so upload-artifact can
    # still see the file after the composite cleanup step.
    explicit = (env.get("LANE_RESULTS_DIR") or "").strip()
    if explicit:
        directory = Path(explicit)
    else:
        directory = Path(env.get("RUNNER_TEMP") or "/tmp") / "or-pr-review-lanes"
    return _persist_lane_artifact(directory, index, result)


def _all_role_deadline_seconds(
    env: dict[str, str], remaining: float | None, judge_reserve: int
) -> int:
    """Bound lane collection while preserving the judge/publication window."""
    configured = _configured_all_role_deadline_seconds(env)
    if configured is not None:
        return configured
    if remaining is None:
        return 1500
    return max(int(remaining - max(POST_RESERVE_SECONDS, judge_reserve)), 0)


def _configured_all_role_deadline_seconds(env: dict[str, str]) -> int | None:
    raw = (env.get("ALL_ROLE_DEADLINE_SECONDS") or "").strip()
    if not raw:
        return None
    deadline = _int_env(env, "ALL_ROLE_DEADLINE_SECONDS", 1500)
    if deadline < 1:
        raise ActionError("all_role_deadline_seconds must be a positive integer")
    return deadline


def _job_budget_seconds(env: dict[str, str]) -> int:
    budget = _int_env(env, "JOB_BUDGET_SECONDS", JOB_BUDGET_SECONDS)
    if budget < 1:
        raise ActionError("job_budget_seconds must be a positive integer")
    return budget


# Aggregate telemetry only: never persist prompts, tool arguments or model text.
_PROGRESS_FIELDS = frozenset(
    {
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "known_cost_usd",
        "attempted_requests",
        "cost_observed_responses",
        "cost_complete",
        "cost_usd",
        "requests",
        "tool_rounds",
        "retries",
        "provider",
        "requested_service_tier",
        "served_service_tiers",
        "service_tier_observed_responses",
        "service_tier_complete",
        "service_tier_confirmed",
        "last_http_status",
        "transport_timeouts",
        "idle_timeouts",
        "elapsed_deadlines",
        "connect_timeouts",
        "connection_errors",
    }
)


def _persist_lane_progress(
    directory: Path, index: int, model: str, snapshot: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in snapshot.items() if key in _PROGRESS_FIELDS}
    payload.update(schema="or-pr-review/lane-progress/1", model=model)
    path = directory / f"progress-{index}.json"
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    pending.replace(path)


def _restore_lane_progress(directory: Path, index: int, lane: LaneResult) -> None:
    try:
        snapshot = json.loads((directory / f"progress-{index}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for key in _PROGRESS_FIELDS:
        if key in snapshot and hasattr(lane, key):
            setattr(lane, key, snapshot[key])
    diagnostics = ", ".join(
        f"{key}={snapshot[key]}"
        for key in (
            "transport_timeouts",
            "idle_timeouts",
            "elapsed_deadlines",
            "connect_timeouts",
            "connection_errors",
        )
        if isinstance(snapshot.get(key), int)
        and not isinstance(snapshot[key], bool)
        and snapshot[key] >= 0
    )
    if diagnostics and lane.error:
        lane.error += f"; checkpoint transport history (includes recovered errors): {diagnostics}"
    # An in-flight request may be billable even if earlier costs were complete.
    # Preserve observed costs separately; never claim a total for an interrupted lane.
    lane.cost_usd = None
    lane.cost_complete = False


def _persist_lane_artifact(directory: Path, index: int, result: LaneResult) -> Path:
    """Persist an all-role lane immediately, before waiting for siblings."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"lane-{index}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _write_review_receipt(env: dict[str, str], receipt: dict[str, Any]) -> Path:
    """Write exactly the canonical JSON embedded in the posted receipt marker."""
    directory = Path(env.get("ALL_LANE_RESULTS_DIR") or (_work_dir(env) / "lanes"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "review-receipt.json"
    canonical = canonical_receipt_json(receipt)
    path.write_bytes(canonical.encode("utf-8"))
    return path


def _persist_and_log_lane(
    directory: Path,
    index: int,
    model: str,
    lane: LaneResult,
    head_sha: str,
    context: dict[str, Any],
) -> None:
    lane.head_sha = head_sha
    lane.review_context = context
    _persist_lane_artifact(directory, index, lane)
    print(
        f"lane {index} `{model}` persisted ({'ok' if lane.ok else 'failed-open'})",
        flush=True,
    )


def _remaining_job_seconds(env: dict[str, str]) -> float | None:
    raw = (env.get(_JOB_DEADLINE_KEY) or "").strip()
    if not raw:
        return None
    try:
        deadline = float(raw)
    except ValueError:
        return None
    return max(0.0, deadline - time.monotonic())


def _judge_request_timeout(env: dict[str, str]) -> int | None:
    """Skip impractical windows; only shared jobs cap the judge at one minute."""
    configured = _int_env(env, "OPENROUTER_TIMEOUT_SECONDS", 180)
    window = configured if env.get("ROLE") == "judge" else min(configured, JUDGE_BUDGET_SECONDS)
    remaining = _remaining_job_seconds(env)
    if remaining is not None:
        usable = int(remaining - POST_RESERVE_SECONDS - JUDGE_SCHEDULING_MARGIN_SECONDS)
        window = min(window, usable)
    return window if window >= MIN_JUDGE_WINDOW_SECONDS else None


def _judge_deadline(env: dict[str, str], timeout: int) -> float | None:
    if env.get("ROLE") != "judge":
        return time.monotonic() + timeout
    # A dedicated matrix judge does not compete with reviewers. Keep its
    # configured inactivity timeout and bound retries by its own job deadline.
    remaining = _remaining_job_seconds(env)
    if remaining is None:
        return None
    return time.monotonic() + max(
        0.0, remaining - POST_RESERVE_SECONDS - JUDGE_SCHEDULING_MARGIN_SECONDS
    )


def _github(env: dict[str, str]) -> GitHub:
    token = (env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or "").strip()
    repository = (env.get("GITHUB_REPOSITORY") or "").strip()
    timeout = _int_env(env, "GITHUB_TIMEOUT_SECONDS", 120)
    source = _source_root(env)
    return GitHub(
        token=token,
        repository=repository,
        timeout=timeout,
        source_workspace=str(source) if source is not None else None,
    )


def _pr_number(env: dict[str, str]) -> int:
    raw = (env.get("PR_NUMBER") or "").strip()
    if not raw:
        raise ActionError("pr_number is empty (set the input or run on a pull_request event)")
    try:
        number = int(raw)
    except ValueError as exc:
        raise ActionError("pr_number must be an integer") from exc
    if number <= 0:
        raise ActionError("pr_number must be a positive integer")
    return number


def _int_env(env: dict[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ActionError(f"{name} must be an integer") from exc


def _env_flag(env: dict[str, str], name: str, default: bool) -> bool:
    """Parse the action's documented yes/no spellings, preserving defaults."""
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    raise ActionError(f"{name.lower()} must be true or false")


def _maybe_status(env: dict[str, str], pr_number: int, body: str) -> None:
    try:
        enabled = _env_flag(env, "STATUS_COMMENTS", True)
    except ActionError:
        # This helper also runs from main's error handler. Invalid input must
        # not raise a second exception while reporting the first one.
        return
    if not enabled:
        return
    try:
        upsert_status_comment(_github(env), pr_number=pr_number, body=body)
    except ActionError as exc:
        print(f"warning: status comment failed: {redact(str(exc))}")


def _best_effort_incomplete(env: dict[str, str], *, stage: str, reason: str) -> None:
    try:
        pr_number = _pr_number(env)
    except ActionError:
        return
    _maybe_status(
        env,
        pr_number,
        render_incomplete(stage=stage, reason=reason, run_url=env.get("RUN_URL") or ""),
    )


def _set_output(name: str, value: str) -> None:
    path = _ACTIVE_ENV.get("GITHUB_OUTPUT") or os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    # Every output produced by this CLI is single-line. Refuse accidental
    # command-file injection instead of maintaining an unnecessary heredoc
    # protocol with a collision-prone fixed delimiter.
    if "\r" in value or "\n" in value:
        raise ActionError(f"GitHub output {name!r} must be a single line")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _error(message: str) -> None:
    print(f"::error::{redact(message)}", file=sys.stderr)
    print(redact(message), file=sys.stderr)
