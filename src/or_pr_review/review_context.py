"""Versioned publication snapshots carried by matrix lane artifacts.

The digest detects inconsistent artifacts; it is not a signature. These files
must come from the trusted lane jobs of the same workflow run.
"""

from __future__ import annotations

import hashlib
import json
import re
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from or_pr_review.brief import BRIEF_MARKER_RE, MAX_MARKER_BYTES
from or_pr_review.collect import CollectedReview
from or_pr_review.errors import SchemaError
from or_pr_review.loop import MAX_UNCOMPACTED_HISTORY_BYTES, LoopState
from or_pr_review.models import parse_slug
from or_pr_review.prompt import parse_path_profiles
from or_pr_review.review_plan import ReviewPlan
from or_pr_review.schema import SEVERITIES, valid_review_path

MAX_CONTEXT_BYTES = 16 * 1024 * 1024
CONTEXT_VERSION = 3
MAX_RUNTIME_BYTES = 40 * 1024
FROZEN_RUNTIME_KEYS = frozenset(
    {
        "ROAST_LEVEL",
        "CUSTOM_INSTRUCTIONS",
        "PATH_PROFILES",
        "PERSONA",
        "JUDGE_NEEDED",
        "OPENROUTER_TIMEOUT_SECONDS",
        "FAIL_ON",
        "BOT_LOGIN",
        "ALL_ROLE_DEADLINE_SECONDS",
    }
)
_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_PROVIDER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}")


@dataclass(frozen=True)
class ReviewContext:
    repository: str
    collected: CollectedReview
    loop: LoopState
    max_tool_turns: int
    execution: PreparedExecution | None = None


@dataclass(frozen=True)
class PreparedExecution:
    """The fully resolved, replayable execution settings for a lane artifact."""

    plan: ReviewPlan
    runtime_json: str
    agent_replies: str
    started_unix_ms: int
    deadline_unix_ms: int
    source_run_url: str
    run_attempt: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(kind: Any, value: Any) -> Any:
    """Restore only the statically declared dataclasses and their primitive fields."""
    origin, args = get_origin(kind), get_args(kind)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return _decode(choice, value)
            except SchemaError:
                pass
    elif origin is Literal:
        if type(value) is str and value in args:
            return value
    elif origin is tuple:
        if isinstance(value, list):
            return tuple(_decode(args[0], item) for item in value)
    elif is_dataclass(kind):
        if isinstance(value, dict) and set(value) == {field.name for field in fields(kind)}:
            hints = get_type_hints(kind)
            return kind(**{name: _decode(hints[name], item) for name, item in value.items()})
    elif kind in (str, int, bool, type(None)) and type(value) is kind:
        return value
    raise SchemaError("review context has an invalid field or shape")


def _runtime_object(runtime_json: str) -> dict[str, str]:
    """Decode the deliberately tiny allow-listed execution environment."""
    try:
        if len(runtime_json.encode("utf-8", "strict")) > MAX_RUNTIME_BYTES:
            raise SchemaError("prepared runtime exceeds 40 KiB")
    except UnicodeEncodeError as exc:
        raise SchemaError("prepared runtime is not valid UTF-8") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError(f"prepared runtime has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            runtime_json,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except SchemaError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as exc:
        raise SchemaError("prepared runtime is not strict JSON") from exc
    if type(value) is not dict or set(value) - FROZEN_RUNTIME_KEYS:
        raise SchemaError("prepared runtime has an invalid key or shape")
    if any(type(key) is not str or type(item) is not str for key, item in value.items()):
        raise SchemaError("prepared runtime values must be strings")
    if any(0xD800 <= ord(char) <= 0xDFFF for item in value.values() for char in item):
        raise SchemaError("prepared runtime is not valid Unicode")
    return value


def _validate_runtime(runtime_json: str) -> None:
    runtime = _runtime_object(runtime_json)
    tone = runtime.get("ROAST_LEVEL")
    if tone is not None and tone not in {"professional", "playful"}:
        raise SchemaError("prepared runtime has an invalid roast level")
    custom = runtime.get("CUSTOM_INSTRUCTIONS")
    if custom is not None:
        try:
            if len(custom.encode("utf-8", "strict")) > 16_000:
                raise SchemaError("prepared runtime custom instructions exceed 16,000 UTF-8 bytes")
        except UnicodeEncodeError as exc:
            raise SchemaError("prepared runtime custom instructions are not valid UTF-8") from exc
    path_profiles = runtime.get("PATH_PROFILES")
    if path_profiles is not None:
        try:
            profiles = parse_path_profiles(path_profiles)
        except Exception as exc:  # trusted parser exposes ActionError, never leak its type here
            raise SchemaError("prepared runtime has invalid path profiles") from exc
        if any(
            path.startswith(("/", "~")) or re.match(r"[A-Za-z]:[\\/]", path)
            for profile in profiles or []
            for path in profile["paths"]
        ):
            raise SchemaError("prepared runtime path profiles must not contain local paths")
    persona = runtime.get("PERSONA")
    if persona is not None:
        try:
            if len(persona.encode("utf-8", "strict")) > 1_000:
                raise SchemaError("prepared runtime persona exceeds 1,000 UTF-8 bytes")
        except UnicodeEncodeError as exc:
            raise SchemaError("prepared runtime persona is not valid UTF-8") from exc
    judge_needed = runtime.get("JUDGE_NEEDED")
    if judge_needed is not None and judge_needed not in {"", "true", "false"}:
        raise SchemaError("prepared runtime has an invalid judge-needed flag")
    timeout = runtime.get("OPENROUTER_TIMEOUT_SECONDS")
    if timeout is not None and (
        re.fullmatch(r"[1-9][0-9]*", timeout) is None or not 1 <= int(timeout) <= 600
    ):
        raise SchemaError("prepared runtime has an invalid OpenRouter timeout")
    fail_on = runtime.get("FAIL_ON")
    if fail_on is not None and fail_on not in {"never", "bugs", "any"}:
        raise SchemaError("prepared runtime has an invalid fail_on setting")
    login = runtime.get("BOT_LOGIN")
    if login is not None and (len(login) > 100 or any(char.isspace() for char in login)):
        raise SchemaError("prepared runtime has an invalid bot login")
    all_deadline = runtime.get("ALL_ROLE_DEADLINE_SECONDS")
    if (
        all_deadline is not None
        and all_deadline != ""
        and re.fullmatch(r"[1-9][0-9]*", all_deadline) is None
    ):
        raise SchemaError("prepared runtime has an invalid all-role deadline")


def freeze_runtime(env: dict[str, str]) -> str:
    """Freeze only non-secret prompt/execution settings, in canonical JSON."""
    runtime = {key: value for key, value in env.items() if key in FROZEN_RUNTIME_KEYS}
    if any(type(key) is not str or type(value) is not str for key, value in runtime.items()):
        raise SchemaError("prepared runtime values must be strings")
    # Freeze the meanings accepted by the CLI, rather than ambient spelling.
    # Artifacts remain strict: restore accepts only these canonical spellings.
    runtime["ROAST_LEVEL"] = (runtime.get("ROAST_LEVEL") or "professional").strip().lower()
    runtime["FAIL_ON"] = (runtime.get("FAIL_ON") or "never").strip().lower()
    try:
        runtime["OPENROUTER_TIMEOUT_SECONDS"] = str(
            int((runtime.get("OPENROUTER_TIMEOUT_SECONDS") or "180").strip() or "180")
        )
    except ValueError as exc:
        raise SchemaError("prepared runtime has an invalid OpenRouter timeout") from exc
    judge = (runtime.get("JUDGE_NEEDED") or "").strip().lower()
    if judge in {"1", "yes"}:
        judge = "true"
    elif judge in {"0", "no"}:
        judge = "false"
    runtime["JUDGE_NEEDED"] = judge
    deadline = (runtime.get("ALL_ROLE_DEADLINE_SECONDS") or "").strip()
    try:
        runtime["ALL_ROLE_DEADLINE_SECONDS"] = str(int(deadline)) if deadline else ""
    except ValueError as exc:
        raise SchemaError("prepared runtime has an invalid all-role deadline") from exc
    if "BOT_LOGIN" in runtime:
        runtime["BOT_LOGIN"] = runtime["BOT_LOGIN"].strip()
    encoded = _canonical(runtime).decode("utf-8")
    _validate_runtime(encoded)
    return encoded


def _valid_slug(value: str) -> bool:
    try:
        return (
            len(value.encode("utf-8", "strict")) <= 200
            and parse_slug(value, what="prepared execution model") == value
        )
    except Exception:
        return False


def _validate_execution(context: ReviewContext) -> None:
    execution = context.execution
    if execution is None:
        return
    plan = execution.plan
    if (
        not _PROFILE_RE.fullmatch(plan.profile)
        or plan.level not in {"standard", "deep"}
        or plan.trigger not in {"baseline", "policy", "manual"}
        or (plan.level == "standard" and plan.trigger != "baseline")
        or (plan.level == "deep" and plan.trigger not in {"policy", "manual"})
        or (plan.legacy and plan.level != "standard")
        or not re.fullmatch(r"[0-9a-f]{64}", plan.registry_digest)
        or plan.effort not in {"", "none", "minimal", "low", "medium", "high", "xhigh"}
        or not 0 <= plan.max_tool_turns <= 1000
        or plan.max_tool_turns != context.max_tool_turns
        or not 1 <= plan.lane_timeout_seconds <= 1800
        or not 240 <= plan.job_budget_seconds <= 3600
        or plan.lane_timeout_seconds > plan.job_budget_seconds - 180
        or not _valid_slug(plan.judge_model)
        or not 1 <= len(plan.lanes) <= 4
    ):
        raise SchemaError("prepared execution has an invalid review plan")
    models = [lane.model for lane in plan.lanes]
    if len(models) != len(set(models)) or any(not _valid_slug(model) for model in models):
        raise SchemaError("prepared execution has invalid lane models")
    if not plan.legacy and not any(lane.required for lane in plan.lanes):
        raise SchemaError("prepared execution requires a required lane")
    for lane in plan.lanes:
        if (
            lane.provider is not None and not _PROVIDER_RE.fullmatch(lane.provider)
        ) or lane.service_tier not in {None, "default", "flex", "priority"}:
            raise SchemaError("prepared execution has an invalid model route")
    policy = context.collected.review_policy
    if policy is not None:
        if plan.profile != policy.profile or (policy.minimum == "deep" and plan.level != "deep"):
            raise SchemaError("prepared execution weakens or mismatches review policy")
    elif not plan.legacy and plan.profile != "code":
        raise SchemaError("prepared execution without policy must use the code profile")
    _validate_runtime(execution.runtime_json)
    try:
        replies_bytes = len(execution.agent_replies.encode("utf-8", "strict"))
    except UnicodeEncodeError as exc:
        raise SchemaError("prepared execution replies are not valid UTF-8") from exc
    reply_limit = MAX_UNCOMPACTED_HISTORY_BYTES  # compaction: none; the producer enforces tighter budgets
    if replies_bytes > reply_limit:
        raise SchemaError(f"prepared execution replies exceed {reply_limit:,} UTF-8 bytes")
    try:
        source_url_bytes = len(execution.source_run_url.encode("utf-8", "strict"))
    except UnicodeEncodeError as exc:
        raise SchemaError("prepared execution source run URL is not valid UTF-8") from exc
    if (
        execution.started_unix_ms <= 0
        or execution.deadline_unix_ms <= 0
        or execution.deadline_unix_ms - execution.started_unix_ms != plan.job_budget_seconds * 1000
        or not 1 <= execution.run_attempt <= 1000
        or source_url_bytes > 512
    ):
        raise SchemaError("prepared execution has invalid absolute deadline metadata")
    if execution.source_run_url and not re.fullmatch(
        rf"https://[A-Za-z0-9][A-Za-z0-9.-]*/{re.escape(context.repository)}/actions/runs/[0-9]+",
        execution.source_run_url,
    ):
        raise SchemaError("prepared execution has an invalid source run URL")


def freeze_context(
    repository: str,
    collected: CollectedReview,
    loop: LoopState,
    max_tool_turns: int,
    *,
    execution: PreparedExecution | None = None,
) -> dict[str, Any]:
    payload = asdict(ReviewContext(repository, collected, loop, max_tool_turns, execution))
    # Normalize tuples exactly as the artifact writer will. Validate before any
    # paid request so an oversized/invalid context cannot strand completed work.
    payload = json.loads(_canonical(payload))
    envelope = {
        "version": CONTEXT_VERSION,
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }
    restore_context(envelope)
    return envelope


def restore_context(envelope: object) -> ReviewContext:
    if not isinstance(envelope, dict) or set(envelope) != {"version", "sha256", "payload"}:
        raise SchemaError("matrix lane is missing a valid publication context; rerun its lanes")
    if type(envelope["version"]) is not int or envelope["version"] != CONTEXT_VERSION:
        raise SchemaError("unsupported review context version; rerun the lanes with this action")
    try:
        encoded = _canonical(envelope)
        digest = hashlib.sha256(_canonical(envelope["payload"])).hexdigest()
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SchemaError("review context is not bounded JSON") from exc
    if len(encoded) > MAX_CONTEXT_BYTES:
        raise SchemaError("review context exceeds the 16 MiB artifact limit")
    if envelope["sha256"] != digest:
        raise SchemaError("review context digest mismatch")
    context = _decode(ReviewContext, envelope["payload"])
    collected, loop = context.collected, context.loop
    try:
        embedded_bytes = len(collected.diff.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise SchemaError("review context diff is not valid UTF-8 text") from exc
    if (
        not re.fullmatch(r"[^/\s]+/[^/\s]+", context.repository)
        or collected.pr_number < 1
        or not re.fullmatch(r"[0-9a-f]{40}", collected.head_sha)
        or collected.plan.to_sha != collected.head_sha
        or loop.mode != collected.mode
        or loop.mode not in {"initial", "verify"}
        or (
            collected.plan.scope == "rebase"
            and (loop.mode != "verify" or collected.plan.kind != "full-pr" or loop.retired_prior)
        )
        or not 1 <= loop.round_number <= 999
        or not re.fullmatch(r"[0-9a-f]{0,12}", loop.generation)
        or not 0 <= context.max_tool_turns <= 1000
        or collected.truncation.max_diff_kb < 1
        or collected.truncation.original_bytes < 0
        or collected.truncation.embedded_bytes != embedded_bytes
        or collected.truncation.original_bytes < collected.truncation.embedded_bytes
        or len(collected.body_brief) > 64_000
        or (collected.brief_marker and not BRIEF_MARKER_RE.fullmatch(collected.brief_marker))
        or len(collected.brief_marker) > MAX_MARKER_BYTES
    ):
        raise SchemaError("review context contains inconsistent publication metadata")
    policy = collected.review_policy
    if policy is not None:
        if (
            not re.fullmatch(r"[0-9a-f]{40}", policy.base_sha)
            or policy.base_sha != collected.policy_base_sha
            or not re.fullmatch(r"[0-9a-f]{64}", policy.digest)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", policy.profile)
            or policy.minimum not in {"standard", "deep"}
            or len(policy.files) > 32
            or len({item.path for item in policy.files}) != len(policy.files)
            or len(policy.matches) > 64
            or any(not valid_review_path(path) for path in policy.changed_paths)
        ):
            raise SchemaError("review context contains invalid policy metadata")
        total_bytes = 0
        for item in policy.files:
            try:
                size = len(item.content.encode("utf-8"))
            except UnicodeError as exc:
                raise SchemaError("review policy is not valid UTF-8") from exc
            total_bytes += size
            if (
                not valid_review_path(item.path)
                or item.path.split("/")[-1] != "REVIEW.md"
                or not re.fullmatch(r"[0-9a-f]{40}", item.blob_sha)
                or size > 16 * 1024
                or total_bytes > 64 * 1024
                or not set(item.scope_paths).issubset(policy.changed_paths)
            ):
                raise SchemaError("review context contains invalid scoped policy guidance")
        for match in policy.matches:
            if (
                match.file not in {item.path for item in policy.files}
                or match.minimum not in {"standard", "deep"}
                or not set(match.paths).issubset(policy.changed_paths)
            ):
                raise SchemaError("review context contains invalid policy rule matches")
    findings = (*loop.prior_findings, *loop.retired_prior)
    if len(findings) > 200 or len({item.id for item in findings}) != len(findings):
        raise SchemaError("review context has invalid carried finding identities")
    for item in findings:
        if (
            not re.fullmatch(r"r\d{1,3}-\d{1,3}", item.id)
            or item.severity not in SEVERITIES
            or item.status not in {"open", "disputed"}
            or (item.file is not None and not valid_review_path(item.file))
            or (item.line is not None and item.line < 1)
            or len(item.title) > 300
            or len(item.evidence) > 616
            or len(item.models) > 8
        ):
            raise SchemaError("review context has an invalid carried finding")
    _validate_execution(context)
    return context
