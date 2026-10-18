"""Thin OpenRouter chat-completions harness with optional read-only tools."""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import math
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any

from or_pr_review.errors import ActionError, LaneError
from or_pr_review.http_transport import (
    HttpConnectTimeout,
    HttpElapsedTimeout,
    HttpIdleTimeout,
    bounded_urlopen,
)
from or_pr_review.models import GEMINI_MAX_RESPONSE_TOKENS, base_chat_payload
from or_pr_review.redaction import redact
from or_pr_review.schema import (
    SCHEMA_VERSION,
    Finding,
    LaneResult,
    Resolution,
    failed_lane,
    findings_json_schema,
    parse_lane_payload,
    parse_model_findings,
    valid_review_path,
    validate_coverage,
)
from or_pr_review.workspace import MAX_OVERSIZED_MATERIALIZED_FILE, READ_ONLY_TOOLS, tracked_paths

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "https://github.com/FlyOverCoderKY/openrouter-pr-review-action"
APP_TITLE = "OpenRouter PR Review Action"
DEFAULT_TIMEOUT = 180
# Compatibility import for Python callers that used the former harness-level
# constant. New code should import GEMINI_MAX_RESPONSE_TOKENS from models.
MAX_GEMINI_RESPONSE_TOKENS = GEMINI_MAX_RESPONSE_TOKENS
# Lane ceilings are additionally clamped by the caller's remaining job budget
# after its separate judge and publication reserves.
DEFAULT_LANE_TIMEOUT_SECONDS = 18 * 60
# Transient-error policy: each request retries a few times with backoff, and
# a mid-loop failure that survives the retries triggers one salvage attempt
# in _run_loop instead of discarding the gathered evidence.
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)
MAX_HTTP_ATTEMPTS = 4
MAX_RATE_LIMIT_ATTEMPTS = 7
BACKOFF_BASE_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 30.0
RATE_LIMIT_JITTER_SECONDS = 5.0
# First-pass budget matches the sibling Grok action's default max_turns=50.
# Follow-up jobs may pass a lower value (sibling callers often use 30).
DEFAULT_MAX_TOOL_TURNS = 50
MAX_TOOL_TURNS = DEFAULT_MAX_TOOL_TURNS
MAX_TOOL_TURNS_LIMIT = 1000
# Aggregate cap on tool-observation bytes per lane. Every observation is
# resent on every later request, so unbounded reads grow the transcript
# quadratically; past this cap the loop withdraws tools and asks for the
# JSON finish.
MAX_OBSERVATION_BYTES = 600_000
BLAST_RADIUS_NUDGE = (
    "You returned a review without using tools. The embedded diff is not the "
    "whole repository. Use read_file, grep, and/or list_dir to check blast "
    "radius: tests that inventory workflow or other filenames, README and "
    "code-map docs, and sibling CI files. Then finish with the JSON object "
    '{"findings": [...]}.'
)
BUDGET_EXHAUSTED_NOTICE = (
    "Tool budget exhausted. Finish now with the JSON object "
    '{"findings": [...]} and no further tool calls.'
)
FINALIZE_RETRY_NOTICE = (
    "Your last message was not the required JSON object. Return ONLY the JSON "
    'object {"findings": [...]} now, with no commentary and no tool calls.'
)
DEADLINE_FINALIZE_NOTICE = (
    "The review lane is approaching its wall-clock deadline. Stop using tools "
    "and return the best complete JSON result now from the embedded diff and "
    "all evidence gathered so far."
)
MISSING_SIGNATURE_FINALIZE_NOTICE = (
    "The tool request could not be safely continued because its provider "
    "reasoning signature was missing. Do not call tools. Return the best "
    'complete JSON result now as {"findings": [...]} from the embedded diff '
    "and evidence already gathered."
)


class OpenRouterHTTPError(LaneError):
    """Terminal OpenRouter HTTP error with aggregate-safe routing metadata."""

    def __init__(self, message: str, *, provider: str | None) -> None:
        super().__init__(message)
        self.provider = provider


# Bound on stub-repair rounds if a model keeps emitting tool calls after the
# tool budget was withdrawn.
MAX_REPAIR_ROUNDS = 3
# Repository tools run out-of-process so a pathological regex or filesystem
# stall cannot pin a review thread past its lane deadline. The worker package
# is found through action.yml's PYTHONPATH or through an installed package.
# Python safe-path mode prevents a package in the reviewed checkout from
# shadowing that trusted import.
MAX_TOOL_CALL_SECONDS = 30


ChatFn = Callable[[dict[str, Any]], dict[str, Any]]
SleepFn = Callable[[float], None]
ProgressFn = Callable[[dict[str, Any]], None]


class _LoopPhase(Enum):
    EXPLORE = auto()
    NUDGE = auto()
    FINALIZE = auto()
    REPAIR = auto()


@dataclass
class _LoopState:
    phase: _LoopPhase
    use_schema: bool
    turns: int = 0
    repairs: int = 0
    observation_bytes: int = 0
    last_error: LaneError | None = None
    finalize_reason: str | None = None
    deadline_limited: bool = False
    finalize_retried: bool = False
    salvage_attempted: bool = False
    successful_responses: int = 0

    @property
    def tools_active(self) -> bool:
        return self.phase in {_LoopPhase.EXPLORE, _LoopPhase.NUDGE}

    @property
    def force_tool(self) -> bool:
        return self.phase is _LoopPhase.NUDGE

    @property
    def signature_finalizing(self) -> bool:
        return self.finalize_reason == "signature"

    def enter_finalize(self, reason: str, *, salvaged: bool = False) -> None:
        self.phase = _LoopPhase.FINALIZE
        self.use_schema = True
        self.finalize_reason = reason
        if reason == "deadline":
            # Deadline pressure is orthogonal to the immediate finalize reason.
            # A later signature recovery may repair only from time still left.
            self.deadline_limited = True
        if salvaged:
            self.salvage_attempted = True


@dataclass
class _LaneClock:
    """Own lane-wall-clock arithmetic for real and injected chat transports."""

    deadline: float | None
    reserve_seconds: float
    request_deadline: float | None = None

    def ensure_active(self, now: float) -> None:
        if self.deadline is not None and now >= self.deadline:
            raise LaneError("OpenRouter lane wall-clock deadline exhausted")

    def finalize_due(self, *, now: float, tools_active: bool) -> bool:
        return bool(
            self.deadline is not None
            and tools_active
            and now >= self.deadline - self.reserve_seconds
        )

    def prepare_request(
        self,
        *,
        now: float,
        tools_active: bool,
    ) -> float | None:
        self.ensure_active(now)
        if self.deadline is None:
            self.request_deadline = None
        elif tools_active:
            self.request_deadline = self.deadline - self.reserve_seconds
        else:
            self.request_deadline = self.deadline
        return self.request_deadline

    def tool_deadline(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - self.reserve_seconds


def require_openrouter_key(env: dict[str, str]) -> str:
    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ActionError(
            "OPENROUTER_API_KEY is missing. Set it on the job env "
            "(not as an action input) from a repository secret."
        )
    return key


def parse_max_tool_turns(
    raw: str | None,
    *,
    default: int = DEFAULT_MAX_TOOL_TURNS,
    what: str = "max_tool_turns",
) -> int:
    """Parse a tool-turn budget. 0 disables tools. Empty uses `default`."""
    text = (raw or "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ActionError(f"{what} must be an integer") from exc
    if value < 0 or value > MAX_TOOL_TURNS_LIMIT:
        raise ActionError(f"{what} must be 0 through {MAX_TOOL_TURNS_LIMIT}")
    return value


def openrouter_chat(
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    sleep: SleepFn = time.sleep,
    stats: dict[str, int] | None = None,
    deadline: float | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        if stats is not None and attempt > 1:
            stats["attempted_requests"] = stats.get("attempted_requests", 0) + 1
            if progress_callback is not None:
                progress_callback()
        request_timeout = _bounded_request_timeout(timeout, deadline)
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": APP_TITLE,
            },
        )
        try:
            # A live response may take longer than one socket inactivity window.
            # The absolute stage deadline still bounds headers, trickling bodies,
            # worker shutdown and all retries, without preempting useful data.
            limits: dict[str, float] = {}
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LaneError("OpenRouter lane request budget exhausted")
                limits["total_timeout"] = remaining
            with bounded_urlopen(request, timeout=request_timeout, **limits) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if stats is not None:
                stats["last_http_status"] = exc.code
            full_error_body = exc.read().decode("utf-8", errors="replace")
            err_body = full_error_body[:800]
            provider = _provider_from_error_body(full_error_body)
            max_attempts = MAX_RATE_LIMIT_ATTEMPTS if exc.code == 429 else MAX_HTTP_ATTEMPTS
            if exc.code in RETRYABLE_STATUS and attempt < max_attempts:
                _count_retry(stats)
                retry_after = exc.headers.get("Retry-After")
                delay = _retry_delay(attempt, retry_after)
                if exc.code == 429 and retry_after is None:
                    delay = min(
                        delay + _stable_rate_limit_jitter(body, attempt),
                        MAX_RETRY_AFTER_SECONDS,
                    )
                try:
                    _sleep_before_retry(delay, sleep=sleep, deadline=deadline)
                except LaneError as deadline_error:
                    raise OpenRouterHTTPError(
                        f"{deadline_error}; last OpenRouter HTTP {exc.code}: {redact(err_body)}",
                        provider=provider,
                    ) from exc
                continue
            # OpenRouter's zero-completion insurance makes a terminal HTTP
            # error response non-billable. Any earlier successful requests in
            # the lane retain their separately reported cost.
            raise OpenRouterHTTPError(
                f"OpenRouter HTTP {exc.code}: {redact(err_body)}",
                provider=provider,
            ) from exc
        except TimeoutError as exc:
            if stats is not None:
                stats["transport_timeouts"] = stats.get("transport_timeouts", 0) + 1
                if isinstance(exc, HttpIdleTimeout):
                    stats["idle_timeouts"] = stats.get("idle_timeouts", 0) + 1
                elif isinstance(exc, HttpElapsedTimeout):
                    stats["elapsed_deadlines"] = stats.get("elapsed_deadlines", 0) + 1
                elif isinstance(exc, HttpConnectTimeout):
                    stats["connect_timeouts"] = stats.get("connect_timeouts", 0) + 1
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                _sleep_before_retry(_retry_delay(attempt, None), sleep=sleep, deadline=deadline)
                continue
            raise LaneError("OpenRouter request timed out") from exc
        except urllib.error.URLError as exc:
            if stats is not None:
                stats["connection_errors"] = stats.get("connection_errors", 0) + 1
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                _sleep_before_retry(_retry_delay(attempt, None), sleep=sleep, deadline=deadline)
                continue
            raise LaneError(f"OpenRouter request failed: {redact(str(exc.reason))}") from exc
        except (http.client.HTTPException, OSError) as exc:
            if stats is not None:
                stats["connection_errors"] = stats.get("connection_errors", 0) + 1
            # A connection that drops mid-body raises from response.read() as
            # http.client.IncompleteRead / RemoteDisconnected or a bare
            # ConnectionError — none of which are URLError — and is exactly as
            # transient as a 502. Common on flaky routes (free tiers).
            if attempt < MAX_HTTP_ATTEMPTS:
                _count_retry(stats)
                _sleep_before_retry(_retry_delay(attempt, None), sleep=sleep, deadline=deadline)
                continue
            raise LaneError(
                f"OpenRouter connection failed mid-response: {redact(str(exc))}"
            ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LaneError(f"OpenRouter returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LaneError("OpenRouter returned a non-object JSON body")
    return parsed


def _count_retry(stats: dict[str, int] | None) -> None:
    if stats is not None:
        stats["retries"] = stats.get("retries", 0) + 1


def _bounded_request_timeout(timeout: int, deadline: float | None) -> float:
    """Clamp socket inactivity to remaining stage time; elapsed is separate."""
    if deadline is None:
        return float(timeout)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LaneError("OpenRouter lane request budget exhausted")
    return min(float(timeout), remaining)


def _sleep_before_retry(delay: float, *, sleep: SleepFn, deadline: float | None) -> None:
    """Back off only when doing so cannot cross the current request budget."""
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= delay:
            raise LaneError("OpenRouter lane request budget exhausted before retry")
    sleep(delay)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
        except ValueError:
            pass
    return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_AFTER_SECONDS)


def _stable_rate_limit_jitter(body: bytes, attempt: int) -> float:
    """Desynchronize concurrent benchmark lanes without nondeterministic tests."""

    digest = hashlib.sha256(body + attempt.to_bytes(4, "big")).digest()
    fraction = int.from_bytes(digest[:2], "big") / 65_535
    return fraction * RATE_LIMIT_JITTER_SECONDS


def _provider_from_error_body(body: str) -> str | None:
    """Read only the aggregate provider name from an OpenRouter error body."""

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return None
    provider = metadata.get("provider_name")
    if not isinstance(provider, str) or not provider.strip():
        return None
    return provider.strip()


def run_lane(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    workspace: Path | None,
    max_tool_turns: int = MAX_TOOL_TURNS,
    effort: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    provider_order: list[str] | None = None,
    provider_data_collection: str | None = None,
    provider_zdr: bool = False,
    service_tier: str | None = None,
    # Anchor-gate reference tree for tool-less runs (workspace None): a full
    # checkout of the reviewed head, used only for path/line existence checks.
    anchor_root: Path | None = None,
    chat: ChatFn | None = None,
    expect_coverage: bool = False,
    expect_resolutions: bool = False,
    expected_paths: set[str] | None = None,
    expected_resolution_ids: set[str] | None = None,
    lane_timeout: float = DEFAULT_LANE_TIMEOUT_SECONDS,
    progress: ProgressFn | None = None,
) -> LaneResult:
    started = time.monotonic()
    deadline = started + lane_timeout
    stats: dict[str, int] = {}
    clock = _LaneClock(
        deadline=deadline,
        # Preserve a finish window, capped at half of short lanes. The first
        # finish can use the whole tail; retry only from time actually left.
        reserve_seconds=min(2 * timeout, max(1, lane_timeout // 2)),
        request_deadline=deadline,
    )
    send = chat or (
        lambda payload: openrouter_chat(
            api_key,
            payload,
            timeout=timeout,
            stats=stats,
            deadline=clock.request_deadline,
            progress_callback=emit_progress,
        )
    )
    conversation = list(messages)
    tools = list(READ_ONLY_TOOLS) if workspace is not None and max_tool_turns > 0 else None
    usage: dict[str, int | float] = {}
    meta: dict[str, Any] = {}
    if service_tier is not None:
        meta["requested_service_tier"] = service_tier
    progress_warning_emitted = False

    def emit_progress() -> None:
        nonlocal progress_warning_emitted
        if progress is None:
            return
        snapshot: dict[str, Any] = {
            "elapsed_ms": elapsed_ms(started),
        }
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "known_cost_usd"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                snapshot[key] = value
        attempts = _attempted_requests(stats)
        snapshot["attempted_requests"] = attempts
        cost_observed = stats.get("cost_observed_responses", 0)
        snapshot["cost_observed_responses"] = cost_observed
        snapshot["cost_complete"] = attempts > 0 and attempts == cost_observed
        if attempts > 0 and attempts == cost_observed:
            known_cost = usage.get("known_cost_usd", 0.0)
            if isinstance(known_cost, (int, float)) and not isinstance(known_cost, bool):
                snapshot["cost_usd"] = known_cost
        for key in (
            "requests",
            "tool_rounds",
            "retries",
            "last_http_status",
            "transport_timeouts",
            "idle_timeouts",
            "elapsed_deadlines",
            "connect_timeouts",
            "connection_errors",
        ):
            value = stats.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                snapshot[key] = value
        provider = meta.get("provider")
        if isinstance(provider, str) and provider:
            snapshot["provider"] = provider
        requested_tier = meta.get("requested_service_tier")
        if isinstance(requested_tier, str):
            snapshot["requested_service_tier"] = requested_tier
        served = meta.get("served_service_tiers")
        if isinstance(served, list) and served:
            snapshot["served_service_tiers"] = list(served)
        if isinstance(requested_tier, str) or (isinstance(served, list) and served):
            tier_observed = stats.get("service_tier_observed_responses", 0)
            snapshot["service_tier_observed_responses"] = tier_observed
            served_list = served if isinstance(served, list) else []
            snapshot["service_tier_complete"] = (
                attempts > 0 and attempts == tier_observed and None not in served_list
            )
            snapshot["service_tier_confirmed"] = (
                snapshot["service_tier_complete"]
                and isinstance(requested_tier, str)
                and served_list == [requested_tier]
            )
        try:
            progress(snapshot)
        except Exception:
            # Telemetry is best-effort and must never abort already-billed
            # review work. Keep the warning aggregate and secret-free.
            if not progress_warning_emitted:
                print(
                    "warning: aggregate progress checkpoint failed",
                    file=sys.stderr,
                    flush=True,
                )
                progress_warning_emitted = True

    parse_diagnostics: dict[str, int] = {}

    def _validate(content: str) -> tuple[list[Finding], list[Resolution], list[tuple[str, int]]]:
        findings, resolutions, coverage = parse_lane_payload(
            content,
            model,
            expect_coverage=expect_coverage,
            expect_resolutions=expect_resolutions,
            diagnostics=parse_diagnostics,
        )
        if expect_coverage and expected_paths:
            problem = validate_coverage(coverage, set(expected_paths))
            if problem:
                raise LaneError(problem)
        if expect_resolutions and expected_resolution_ids:
            provided = {resolution.id for resolution in resolutions}
            missing = sorted(expected_resolution_ids - provided)
            if missing:
                named = ", ".join(missing[:5])
                raise LaneError(
                    f"resolutions are incomplete; missing entries for carried finding(s): {named}"
                )
        return findings, resolutions, coverage

    try:
        findings, resolutions, coverage = _run_loop(
            model=model,
            conversation=conversation,
            tools=tools,
            workspace=workspace,
            max_tool_turns=max_tool_turns,
            effort=effort,
            send=send,
            usage=usage,
            stats=stats,
            meta=meta,
            provider_order=provider_order,
            provider_data_collection=provider_data_collection,
            provider_zdr=provider_zdr,
            service_tier=service_tier,
            response_schema=findings_json_schema(
                include_coverage=expect_coverage,
                include_resolutions=expect_resolutions,
            ),
            validate_final=_validate,
            clock=clock,
            progress=emit_progress,
        )
        gate_root = workspace if workspace is not None else anchor_root
        if gate_root is not None:
            findings = sanitize_anchors(findings, gate_root)
    except LaneError as exc:
        # Progress callbacks are optional. Preserve timeout/connection history
        # even when a deadline masks the preceding error. HTTP exceptions already
        # carry their immediate status; a sticky earlier status can mislead here.
        diagnostics = ", ".join(
            f"{key}={stats[key]}"
            for key in (
                "transport_timeouts",
                "idle_timeouts",
                "elapsed_deadlines",
                "connect_timeouts",
                "connection_errors",
            )
            if key in stats
        )
        error = str(exc)
        if diagnostics:
            error += f"; lane transport history (includes recovered errors): {diagnostics}"
        failed = failed_lane(model, redact(error), elapsed_ms=elapsed_ms(started))
        _attach_stats(failed, stats, usage)
        provider = meta.get("provider")
        failed.provider = provider if isinstance(provider, str) else None
        _attach_service_tier_telemetry(failed, stats, meta)
        return failed

    result = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model=model,
        findings=findings,
        error=None,
        elapsed_ms=elapsed_ms(started),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        provider=meta.get("provider") if isinstance(meta.get("provider"), str) else None,
        resolutions=resolutions,
        coverage=coverage,
        dropped_findings=parse_diagnostics.get("dropped_findings", 0),
    )
    _attach_stats(result, stats, usage)
    _attach_service_tier_telemetry(result, stats, meta)
    return result


def _attempted_requests(stats: dict[str, int]) -> int:
    attempted = stats.get("attempted_requests")
    if attempted is None:
        attempted = stats.get("requests", 0)
    return attempted


def _attach_stats(result: LaneResult, stats: dict[str, int], usage: dict[str, int | float]) -> None:
    result.requests = stats.get("requests")
    attempted = _attempted_requests(stats)
    result.attempted_requests = attempted
    result.tool_rounds = stats.get("tool_rounds")
    result.retries = stats.get("retries")
    result.thought_signature_tool_turns = stats.get("thought_signature_tool_turns")
    result.thought_signature_recoveries = stats.get("thought_signature_recoveries")
    result.sanitized_tool_turns = stats.get("sanitized_tool_turns")
    result.cached_tokens = usage.get("cached_tokens")
    known_cost = usage.get("known_cost_usd")
    result.known_cost_usd = (
        float(known_cost)
        if isinstance(known_cost, (int, float)) and not isinstance(known_cost, bool)
        else None
    )
    cost_observed = stats.get("cost_observed_responses", 0)
    result.cost_observed_responses = cost_observed
    result.cost_complete = bool(attempted and attempted == cost_observed)
    if result.cost_complete:
        result.cost_usd = result.known_cost_usd if result.known_cost_usd is not None else 0.0
    result.salvaged = bool(stats.get("salvaged"))
    if result.prompt_tokens is None:
        result.prompt_tokens = usage.get("prompt_tokens")
    if result.completion_tokens is None:
        result.completion_tokens = usage.get("completion_tokens")


def _attach_service_tier_telemetry(
    result: LaneResult,
    stats: dict[str, int],
    meta: dict[str, Any],
) -> None:
    """Attach optional service-tier telemetry on durable lane artifacts.

    Routing defaults are unchanged; this only records what was requested and
    what each response reported. A requested tier is routing intent. OpenRouter
    reports the served tier on each response, and missing, null, or mixed values
    must remain visible rather than being inferred from that request intent.
    """

    requested = meta.get("requested_service_tier")
    result.requested_service_tier = requested if isinstance(requested, str) else None
    served = meta.get("served_service_tiers", [])
    result.served_service_tiers = list(served) if isinstance(served, list) else []
    observed = stats.get("service_tier_observed_responses", 0)
    result.service_tier_observed_responses = observed
    attempted = _attempted_requests(stats)
    result.service_tier_complete = bool(
        attempted and attempted == observed and None not in result.served_service_tiers
    )
    result.service_tier_confirmed = bool(
        result.service_tier_complete
        and result.served_service_tiers == [result.requested_service_tier]
    )


def _run_loop(
    *,
    model: str,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    workspace: Path | None,
    max_tool_turns: int,
    effort: str,
    send: ChatFn,
    usage: dict[str, int | float],
    stats: dict[str, int] | None = None,
    meta: dict[str, Any] | None = None,
    provider_order: list[str] | None = None,
    provider_data_collection: str | None = None,
    provider_zdr: bool = False,
    service_tier: str | None = None,
    response_schema: dict[str, Any] | None = None,
    validate_final: Callable[[str], tuple[list[Finding], list[Resolution], list[tuple[str, int]]]]
    | None = None,
    clock: _LaneClock,
    progress: Callable[[], None] | None = None,
) -> tuple[list[Finding], list[Resolution], list[tuple[str, int]]]:
    try:
        payload_base, protocol = base_chat_payload(
            model=model,
            response_schema=response_schema or findings_json_schema(),
            reasoning={"effort": effort} if effort else None,
            provider_order=provider_order,
            provider_data_collection=provider_data_collection,
            provider_zdr=provider_zdr,
            service_tier=service_tier,
        )
    except ValueError as exc:
        raise LaneError(str(exc)) from exc

    tools_active = bool(tools) and workspace is not None
    # JSON schema on the first tool-enabled turn pushes a glance-and-clean
    # empty findings object. Offer tools without a schema until the model
    # stops calling them (or the budget is gone).
    state = _LoopState(
        phase=_LoopPhase.EXPLORE if tools_active else _LoopPhase.FINALIZE,
        use_schema=not tools_active,
    )

    def retry_finalization(
        notice: str,
        *,
        error: LaneError | None = None,
        message: dict[str, Any] | None = None,
    ) -> bool:
        """Make the one allowed tool-free structured-finish repair request."""
        if state.finalize_retried:
            return False
        state.finalize_retried = True
        state.last_error = error
        if message is not None:
            conversation.append(
                _assistant_record(
                    message,
                    preserve_provider_metadata=protocol.preserve_provider_metadata,
                )
            )
        # Some provider protocols reject adjacent user entries. Merge notices
        # so a deadline/failure sequence remains valid for every provider.
        _append_user_notice(conversation, notice)
        state.phase = _LoopPhase.REPAIR
        state.use_schema = True
        if state.finalize_reason is None:
            state.finalize_reason = "repair"
        return True

    def sanitize_gemini_history() -> None:
        if not protocol.serial_tool_calls:
            return
        sanitized = _sanitize_tool_history_for_finalize(conversation)
        if stats is not None and sanitized:
            stats["sanitized_tool_turns"] = stats.get("sanitized_tool_turns", 0) + sanitized

    def enter_salvage(
        reason: str,
        notice: str,
        *,
        signature_recovery: bool = False,
    ) -> None:
        """Enter a tool-free finish and record its shared salvage telemetry."""
        state.enter_finalize(reason, salvaged=True)
        if stats is not None:
            stats["salvaged"] = 1
            if signature_recovery:
                stats["thought_signature_recoveries"] = (
                    stats.get("thought_signature_recoveries", 0) + 1
                )
        _append_user_notice(conversation, notice)

    try:
        while True:
            now = time.monotonic()
            clock.ensure_active(now)
            if (
                clock.finalize_due(now=now, tools_active=state.tools_active)
                and not state.deadline_limited
            ):
                enter_salvage("deadline", DEADLINE_FINALIZE_NOTICE)

            # Tool exploration may use only the leading portion of the lane
            # budget.  Preserve the protected tail for a structured finish.
            # This calculation also runs for injected chat fakes, keeping the
            # tested control path identical even though they do no network I/O.
            clock.prepare_request(
                now=now,
                tools_active=state.tools_active,
            )
            payload = {
                **payload_base,
                "messages": conversation,
            }
            if not state.use_schema:
                payload.pop("response_format", None)
            replay_tools = (
                protocol.serial_tool_calls
                and not state.tools_active
                and _conversation_has_tool_protocol(conversation)
            )
            if state.tools_active or replay_tools:
                payload["tools"] = tools
                if state.tools_active:
                    payload["tool_choice"] = "required" if state.force_tool else "auto"
                else:
                    payload["tool_choice"] = "none"
                # Gemini 3 strictly validates thought signatures across a
                # function-calling turn.  Parallel calls have intermittently
                # arrived without a usable signature for every call, causing
                # the next otherwise-valid request to fail with HTTP 400
                # INVALID_ARGUMENT.  Keep Gemini's tool transcript serial so
                # there is exactly one signature-bearing call to round-trip.
                if protocol.serial_tool_calls and state.tools_active:
                    payload["parallel_tool_calls"] = False
            if stats is not None:
                stats["requests"] = stats.get("requests", 0) + 1
                stats["attempted_requests"] = stats.get("attempted_requests", 0) + 1
            # Persist before sending: a transport interruption can happen
            # after OpenRouter accepts and bills the request but before this
            # process receives a response with cost telemetry.
            if progress is not None:
                progress()
            try:
                response = send(payload)
                state.successful_responses += 1
                _absorb_usage(usage, response, meta, stats)
                if progress is not None:
                    progress()
                # In-body errors (HTTP 200 whose JSON carries an error object,
                # a common OpenRouter provider-failure shape) must reach the
                # same schema-fallback and salvage handling as HTTP errors.
                message = _assistant_message(response)
            except LaneError as exc:
                if isinstance(exc, OpenRouterHTTPError):
                    if exc.provider and meta is not None:
                        meta["provider"] = exc.provider
                if progress is not None:
                    progress()
                print(f"lane request rejected: {_request_shape(payload)}", file=sys.stderr)
                if state.signature_finalizing:
                    # An unsigned tool turn gets one safe, tool-free finish.
                    # Retrying it with another transcript mutation would no
                    # longer be the promised recovery from the last valid
                    # provider-authenticated history.
                    raise
                message = str(exc)
                lowered = message.lower()
                schema_rejected = "response_format" in lowered or "json_schema" in lowered
                if state.use_schema and schema_rejected:
                    state.use_schema = False
                    state.last_error = exc
                    if state.deadline_limited:
                        state.finalize_retried = True
                    continue
                if state.turns > 0 and protocol.is_signature_rejection(lowered):
                    # The provider rejected an ostensibly signed tool history.
                    # Strip the protocol blocks once, retain their observations
                    # as attributed plain evidence, and make one no-tools
                    # structured finish rather than replaying poisoned history.
                    if _looks_like_context_overflow(lowered):
                        # Signature/context recovery converts tool records to
                        # attributed plain evidence, so discard all oversized
                        # observations before that conversion. Generic
                        # transport salvage below can retain the last two.
                        _shrink_tool_history(conversation, keep_last=0)
                    sanitize_gemini_history()
                    enter_salvage(
                        "signature",
                        MISSING_SIGNATURE_FINALIZE_NOTICE,
                        signature_recovery=True,
                    )
                    continue
                deadline_pressure = clock.finalize_due(
                    now=time.monotonic(),
                    tools_active=state.tools_active,
                )
                if deadline_pressure and not state.salvage_attempted:
                    # The exploration-stage request budget expired.  This is
                    # expected deadline control, not a failed lane: finalize
                    # from the embedded diff even if no tool round completed.
                    enter_salvage("deadline", DEADLINE_FINALIZE_NOTICE)
                    continue
                if state.turns > 0 and not state.salvage_attempted:
                    # Salvage: a mid-loop failure that survived the HTTP
                    # retries must not discard every gathered observation.
                    # Ask for a final JSON answer from the evidence so far.
                    if _looks_like_context_overflow(lowered):
                        _shrink_tool_history(conversation)
                    enter_salvage(
                        "transport",
                        (
                            "The previous request failed and will not be retried. "
                            "Do not call tools. Return your findings NOW as the "
                            'JSON object {"findings": [...]} from the evidence '
                            "you have already gathered."
                        ),
                    )
                    continue
                if state.deadline_limited and not state.finalize_retried:
                    # A failed finish may be repaired from remaining time.
                    # Do not interrupt a live response to reserve this retry.
                    retry_finalization(
                        (
                            "The protected finalize request failed. Do not call tools. "
                            "Return the final structured JSON now from the evidence "
                            f"already gathered. Problem: {exc}"
                        ),
                        error=exc,
                    )
                    continue
                raise
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if state.signature_finalizing:
                    raise LaneError(
                        "Gemini returned another tool call during thought-signature recovery"
                    )
                if state.phase is _LoopPhase.NUDGE:
                    state.phase = _LoopPhase.EXPLORE
                if protocol.serial_tool_calls and not state.tools_active:
                    # The provider ignored tool_choice=none. Do not append or
                    # execute the unexpected turn; finish once from attributed
                    # plain evidence without replaying tool protocol.
                    sanitize_gemini_history()
                    enter_salvage(
                        "signature",
                        MISSING_SIGNATURE_FINALIZE_NOTICE,
                        signature_recovery=True,
                    )
                    continue
                if protocol.serial_tool_calls and not _has_round_trippable_gemini_signature(
                    message
                ):
                    # Gemini 3 requires the thought signature from each
                    # function-calling step to appear unchanged in the next
                    # request. Never execute a call whose response cannot be
                    # replayed: doing so would both waste local work and poison
                    # every subsequent provider request, including salvage.
                    sanitize_gemini_history()
                    enter_salvage(
                        "signature",
                        MISSING_SIGNATURE_FINALIZE_NOTICE,
                        signature_recovery=True,
                    )
                    continue
                if protocol.serial_tool_calls and stats is not None:
                    stats["thought_signature_tool_turns"] = (
                        stats.get("thought_signature_tool_turns", 0) + 1
                    )
                conversation.append(
                    _assistant_record(
                        message,
                        preserve_provider_metadata=protocol.preserve_provider_metadata,
                    )
                )
                if not state.tools_active or workspace is None:
                    # The model produced tool calls this loop never solicited
                    # (or cannot service). Answer every call with a stub so
                    # the transcript stays valid, then insist on the finish.
                    state.repairs += 1
                    if state.repairs > MAX_REPAIR_ROUNDS:
                        raise LaneError(
                            "model kept issuing tool calls after the tool budget was withdrawn"
                        )
                    for call in tool_calls:
                        conversation.append(_stub_tool_result(call))
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NOTICE})
                    state.use_schema = True
                    continue
                state.turns += 1
                if stats is not None:
                    stats["tool_rounds"] = state.turns
                if progress is not None:
                    progress()
                for call in tool_calls:
                    observation = _run_one_tool(workspace, call, deadline=clock.tool_deadline())
                    conversation.append(observation)
                    text = observation.get("content")
                    if isinstance(text, str):
                        state.observation_bytes += len(text.encode("utf-8"))
                if (
                    state.turns >= max_tool_turns
                    or state.observation_bytes >= MAX_OBSERVATION_BYTES
                ):
                    # Withdraw tools BEFORE the next request: the loop must
                    # never solicit a tool call it will not execute (a
                    # dangling assistant tool_calls entry is an invalid
                    # conversation). The observation-byte cap bounds the
                    # resent transcript the same way the turn budget does.
                    state.enter_finalize("budget")
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NOTICE})
                continue

            content = _message_text(message)
            if state.phase is _LoopPhase.EXPLORE and state.turns == 0:
                # Glance-and-clean: one forced blast-radius tool pass.
                state.phase = _LoopPhase.NUDGE
                conversation.append(
                    _assistant_record(
                        message,
                        preserve_provider_metadata=protocol.preserve_provider_metadata,
                    )
                )
                conversation.append({"role": "user", "content": BLAST_RADIUS_NUDGE})
                continue
            if content.strip():
                try:
                    if validate_final is not None:
                        validated = validate_final(content)
                    else:
                        validated = (parse_model_findings(content, model), [], [])
                except LaneError as exc:
                    if not retry_finalization(
                        f"{FINALIZE_RETRY_NOTICE} Problem: {exc}",
                        error=exc,
                        message=message,
                    ):
                        raise
                    # The tool-backed path runs without response_format, so a
                    # malformed natural finish gets exactly one
                    # schema-enforced, tool-free redo before failing open.
                    continue
                return validated
            if state.last_error:
                raise state.last_error
            if retry_finalization(
                (f"{FINALIZE_RETRY_NOTICE} Problem: the previous assistant message was empty."),
                message=message,
            ):
                # Some otherwise healthy providers occasionally return an
                # empty assistant message after several successful tool
                # rounds. Treat that like a malformed finish: preserve the
                # gathered evidence and make one bounded, schema-enforced,
                # tool-free finalization request instead of discarding the
                # entire review.
                continue
            raise LaneError("OpenRouter returned an empty assistant message")
    finally:
        if stats is not None:
            stats["tool_rounds"] = state.turns


def _request_shape(payload: dict[str, Any]) -> str:
    """Describe a request's protocol fields without any message content."""
    messages = payload.get("messages") or []
    roles = ",".join(str(m.get("role", "?"))[0] for m in messages if isinstance(m, dict))
    return (
        f"response_format={'response_format' in payload} tools={len(payload.get('tools') or [])} "
        f"tool_choice={payload.get('tool_choice')} "
        f"parallel_tool_calls={payload.get('parallel_tool_calls')} "
        f"messages={len(messages)} roles={roles}"
    )


def _looks_like_context_overflow(lowered_error: str) -> bool:
    if "prompt is too long" in lowered_error or "too many tokens" in lowered_error:
        return True
    return "context" in lowered_error and (
        "length" in lowered_error
        or "exceed" in lowered_error
        or "window" in lowered_error
        or "token" in lowered_error
    )


def _conversation_has_tool_protocol(conversation: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant" and bool(message.get("tool_calls"))
        for message in conversation
    )


def _shrink_tool_history(
    conversation: list[dict[str, Any]], *, keep_last: int = 2, keep_bytes: int = 2_000
) -> None:
    """Truncate old tool observations in place so a salvage request can fit."""
    tool_indexes = [
        index for index, message in enumerate(conversation) if message.get("role") == "tool"
    ]
    shrinkable = tool_indexes[:-keep_last] if keep_last else tool_indexes
    for index in shrinkable:
        content = conversation[index].get("content")
        if not isinstance(content, str):
            continue
        data = content.encode("utf-8")
        if len(data) <= keep_bytes:
            continue
        clipped = data[:keep_bytes].decode("utf-8", errors="ignore")
        conversation[index] = {
            **conversation[index],
            "content": clipped + "\n[observation truncated after a context overflow]",
        }


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    error = response.get("error")
    if isinstance(error, dict):
        raise LaneError(redact(str(error.get("message") or error)))
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LaneError("OpenRouter response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LaneError("OpenRouter choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LaneError("OpenRouter choice is missing a message")
    return message


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _assistant_record(
    message: dict[str, Any], *, preserve_provider_metadata: bool = False
) -> dict[str, Any]:
    # Provider adapters can attach opaque continuity metadata to an assistant
    # message. Reconstructing only the fields known to this harness drops that
    # metadata (including Gemini thought signatures). Echo the complete
    # message exactly as returned, using a deep copy so later local mutations
    # cannot alter the provider-authenticated object.
    if preserve_provider_metadata:
        record = copy.deepcopy(message)
        record["role"] = "assistant"
        return record

    record = {"role": "assistant", "content": copy.deepcopy(message.get("content") or "")}
    if message.get("tool_calls"):
        record["tool_calls"] = copy.deepcopy(message["tool_calls"])
    if message.get("reasoning_details"):
        record["reasoning_details"] = copy.deepcopy(message["reasoning_details"])
    elif message.get("reasoning"):
        record["reasoning"] = copy.deepcopy(message["reasoning"])
    return record


def _has_round_trippable_gemini_signature(message: dict[str, Any]) -> bool:
    """Return whether a Gemini tool step carries an opaque signature we echo.

    OpenRouter normally exposes Gemini signatures as a google-gemini-v1
    reasoning detail. Direct thought-signature fields are also accepted for
    compatibility with adapters that retain Google's function-call metadata.
    Plain reasoning/reasoning_content text is useful context but is not a
    cryptographic thought signature.
    """
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict) or detail.get("format") != "google-gemini-v1":
                continue
            signature = detail.get("signature")
            data = detail.get("data")
            if isinstance(signature, str) and signature.strip():
                return True
            if (
                detail.get("type") == "reasoning.encrypted"
                and isinstance(data, str)
                and data.strip()
            ):
                return True

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
        return False
    first = calls[0]
    function = first.get("function")
    candidates = [first]
    if isinstance(function, dict):
        candidates.append(function)
    for candidate in candidates:
        for key in ("thought_signature", "thoughtSignature"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _sanitize_tool_history_for_finalize(conversation: list[dict[str, Any]]) -> int:
    """Replace tool protocol history with plain evidence for safe salvage.

    A provider 400 can mean an earlier tool-call signature is unacceptable.
    Re-sending the same function-call blocks poisons a no-tools salvage request
    too. Remove assistant function-call messages and convert their tool results
    to ordinary user evidence, retaining the observations without asking the
    provider to authenticate the broken tool protocol.
    """
    sanitized: list[dict[str, Any]] = []
    call_provenance: dict[str, tuple[str, str]] = {}
    removed_turns = 0
    for message in conversation:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            removed_turns += 1
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = str(call.get("id") or "unknown")
                    function = call.get("function")
                    if not isinstance(function, dict):
                        function = {}
                    call_provenance[call_id] = (
                        str(function.get("name") or "unknown"),
                        _canonical_tool_arguments(function.get("arguments")),
                    )
            assistant_text = _message_text(message).strip()
            if assistant_text:
                sanitized.append({"role": "assistant", "content": assistant_text})
            continue
        if message.get("role") == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, ensure_ascii=False)
            call_id = str(message.get("tool_call_id") or "unknown")
            name, arguments = call_provenance.get(call_id, ("unknown", "{}"))
            _append_user_notice(
                sanitized,
                (
                    "Previously gathered read-only tool evidence:\n"
                    f"Tool: {name}\n"
                    f"Arguments: {arguments}\n"
                    "Result:\n"
                    f"{content}"
                ),
            )
            continue
        sanitized.append(message)
    conversation[:] = sanitized
    return removed_turns


def _append_user_notice(conversation: list[dict[str, Any]], content: str) -> None:
    """Append user text without creating Gemini-invalid adjacent user turns."""
    if conversation and conversation[-1].get("role") == "user":
        previous = _message_text(conversation[-1]).strip()
        conversation[-1] = {
            **conversation[-1],
            "content": f"{previous}\n\n{content}" if previous else content,
        }
        return
    conversation.append({"role": "user", "content": content})


def _canonical_tool_arguments(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False)
    try:
        return json.dumps(value if value is not None else {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _stub_tool_result(call: object) -> dict[str, Any]:
    call_id = "unknown"
    if isinstance(call, dict):
        call_id = str(call.get("id") or "unknown")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "error: tool budget exhausted; this call was not executed",
    }


def _run_one_tool(
    workspace: Path, call: object, *, deadline: float | None = None
) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"role": "tool", "tool_call_id": "unknown", "content": "error: malformed tool call"}
    call_id = str(call.get("id") or "unknown")
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"error: invalid tool arguments: {exc}",
        }
    remaining = MAX_TOOL_CALL_SECONDS
    if deadline is not None:
        remaining = min(remaining, deadline - time.monotonic())
    if remaining <= 0:
        result = "error: lane tool budget exhausted before this call"
        return {"role": "tool", "tool_call_id": call_id, "content": result}
    try:
        process = subprocess.run(
            [sys.executable, "-P", "-m", "or_pr_review.tool_worker", str(workspace), name],
            input=json.dumps(arguments),
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            result = f"error: repository tool exceeded its {remaining:.1f}s deadline"
        else:
            result = f"error: repository tool process failed: {redact(str(exc))}"
        return {"role": "tool", "tool_call_id": call_id, "content": result}
    if process.returncode != 0:
        detail = redact((process.stderr or "").strip()[:400])
        result = f"error: repository tool process exited {process.returncode}: {detail}"
        return {"role": "tool", "tool_call_id": call_id, "content": result}
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        result = f"error: repository tool returned invalid JSON: {exc}"
        return {"role": "tool", "tool_call_id": call_id, "content": result}
    value = payload.get("result") if isinstance(payload, dict) else None
    result = value if isinstance(value, str) else "error: repository tool returned no result"
    return {"role": "tool", "tool_call_id": call_id, "content": result}


def _absorb_usage(
    usage: dict[str, int | float],
    response: dict[str, Any],
    meta: dict[str, Any] | None = None,
    stats: dict[str, int] | None = None,
) -> None:
    if meta is not None:
        provider = response.get("provider")
        if isinstance(provider, str) and provider.strip():
            # Last response wins; OpenRouter may reroute between requests.
            meta["provider"] = provider.strip()
        if "service_tier" in response:
            tier = response.get("service_tier")
            if tier is None or (isinstance(tier, str) and tier in {"default", "flex", "priority"}):
                if stats is not None:
                    stats["service_tier_observed_responses"] = (
                        stats.get("service_tier_observed_responses", 0) + 1
                    )
                if meta is not None:
                    tiers = meta.setdefault("served_service_tiers", [])
                    if isinstance(tiers, list) and tier not in tiers:
                        tiers.append(tier)
    block = response.get("usage")
    if not isinstance(block, dict):
        return
    for key in ("prompt_tokens", "completion_tokens"):
        value = block.get(key)
        if isinstance(value, int):
            usage[key] = usage.get(key, 0) + value
    spend = response_spend(block)
    if spend is not None:
        usage["known_cost_usd"] = usage.get("known_cost_usd", 0) + spend
        if stats is not None:
            stats["cost_observed_responses"] = stats.get("cost_observed_responses", 0) + 1
    details = block.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            usage["cached_tokens"] = usage.get("cached_tokens", 0) + cached


def response_spend(block: dict[str, Any]) -> float | None:
    """What this request actually cost the operator, from one usage block.

    Non-BYOK responses put the charge in `cost` AND mirror the same figure
    in cost_details.upstream_inference_cost — summing both double-counts.
    BYOK responses put 0 in `cost` (plus any OpenRouter BYOK fee) and the
    provider-billed spend in upstream_inference_cost. So: BYOK = upstream
    plus any positive `cost`; otherwise `cost` alone. A BYOK block with no
    upstream figure is an unknown, not a $0 observation.
    """

    def _valid(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    cost = block.get("cost")
    details = block.get("cost_details")
    upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
    if block.get("is_byok") is True:
        if not _valid(upstream):
            return None
        fee = cost if _valid(cost) and cost > 0 else 0.0
        return float(upstream) + float(fee)
    return float(cost) if _valid(cost) else None


def sanitize_anchors(findings: list[Finding], workspace: Path) -> list[Finding]:
    """Deterministic anchor sanity gate: null objectively impossible locations.

    A finding whose path is not tracked at the reviewed commit becomes
    body-only; a line beyond the end of its materialized file loses the line
    anchor. The finding itself always survives. The gate errs toward keeping
    anchors: the inert snapshot deliberately omits oversized and non-regular
    files, so path existence is judged against the commit's tracked-path
    manifest when one is present (falling back to the filesystem, where a
    directory also counts as existing), and the line check runs only on files
    that were actually materialized, counted the way the read-only tools
    number lines. Live-checkout paths are inspected without following symlink
    components, and line counting is capped to avoid rereading unbounded files.
    Each adjustment is logged.
    """
    manifest = tracked_paths(workspace)
    sanitized: list[Finding] = []
    for finding in findings:
        if finding.file is None:
            sanitized.append(finding)
            continue
        # Findings ordinarily pass through parse_finding, but this public
        # helper also receives programmatically constructed Finding objects.
        # Never let an invalid path escape the review workspace during the
        # filesystem checks below.
        if not valid_review_path(finding.file):
            _log_redacted_warning(
                f"anchor gate: invalid review path; finding "
                f"{finding.title[:60]!r} becomes body-only"
            )
            sanitized.append(replace(finding, file=None, line=None))
            continue
        target = workspace / finding.file
        tracked = finding.file in manifest if manifest is not None else None
        path_state = _anchor_path_state(workspace, finding.file)
        exists = tracked if tracked is not None else path_state != "missing"
        if not exists:
            _log_redacted_warning(
                f"anchor gate: `{finding.file}` is not tracked at the reviewed "
                f"commit; finding {finding.title[:60]!r} becomes body-only"
            )
            sanitized.append(replace(finding, file=None, line=None))
            continue
        # A tracked member can be absent from an inert materialization because
        # it was a link, special file, or too large. Preserve that uncheckable
        # anchor. A present link/non-regular path, especially in a live checkout,
        # is never followed or offered as a GitHub file anchor.
        if path_state == "unsafe":
            _log_redacted_warning(
                f"anchor gate: `{finding.file}` is a symlink or non-regular path; "
                f"finding {finding.title[:60]!r} becomes body-only"
            )
            sanitized.append(replace(finding, file=None, line=None))
            continue
        if finding.line is not None and path_state == "regular":
            line_count = _bounded_lf_line_count(target)
            if line_count is not None and finding.line > line_count:
                _log_redacted_warning(
                    f"anchor gate: line {finding.line} is beyond the end of "
                    f"`{finding.file}` ({line_count} line(s)); dropping the line "
                    f"anchor of {finding.title[:60]!r}"
                )
                sanitized.append(replace(finding, line=None))
                continue
        sanitized.append(finding)
    return sanitized


def _anchor_path_state(workspace: Path, relative: str) -> str:
    """Return regular, unsafe, or missing without following symlink components."""
    current = workspace
    parts = Path(relative).parts
    try:
        for index, part in enumerate(parts):
            current /= part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                return "unsafe"
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                return "unsafe"
        return "regular" if stat.S_ISREG(info.st_mode) else "unsafe"
    except (FileNotFoundError, OSError):
        return "missing"


def _bounded_lf_line_count(path: Path) -> int | None:
    """Count LF-delimited lines without loading files beyond the hard snapshot cap."""
    total = 0
    newline_count = 0
    last_byte: int | None = None
    try:
        if path.stat().st_size > MAX_OVERSIZED_MATERIALIZED_FILE:
            return None
        with path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                total += len(chunk)
                if total > MAX_OVERSIZED_MATERIALIZED_FILE:
                    return None
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1]
    except OSError:
        return None
    if total == 0:
        return 0
    return newline_count + (0 if last_byte == ord("\n") else 1)


def _log_redacted_warning(message: str) -> None:
    """Emit the few model-derived diagnostics this module needs safely."""
    print(redact(message), file=sys.stderr, flush=True)


def elapsed_ms(started: float) -> int:
    """Return monotonic elapsed milliseconds for lane and benchmark records."""
    return int((time.monotonic() - started) * 1000)


def response_message_text(response: dict[str, Any]) -> str:
    """Public wrapper for parsing a chat-completions assistant message."""
    return _message_text(_assistant_message(response))
