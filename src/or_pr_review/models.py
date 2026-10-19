"""Parse the comma-separated OpenRouter models list and enforce the lane cap."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from or_pr_review.errors import ActionError

LANE_CAP = 4
DEFAULT_MODEL = "x-ai/grok-4.6"
# Verified live on OpenRouter 2026-08-29. Merge/de-dupe only; not a second reviewer.
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-luna"

# OpenRouter holds credit for max_tokens while a request is in flight, so an
# uncapped model reserves its full native maximum (65536 for Opus) and runs a
# low balance dry. Every model family gets this cap.
MAX_RESPONSE_TOKENS = 32_768
GEMINI_MAX_RESPONSE_TOKENS = MAX_RESPONSE_TOKENS

# OpenRouter slugs look like provider/model or provider/model:variant.
# Do not invent slugs; callers pass catalogue ids.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(:[A-Za-z0-9._-]+)?$")


@dataclass(frozen=True)
class ModelProtocolProfile:
    """Model-family behavior that OpenRouter's generic API does not describe.

    OpenRouter may route one model through several providers, so this policy
    follows the requested model slug rather than the eventual provider name.
    """

    max_response_tokens: int | None = None
    preserve_provider_metadata: bool = False
    serial_tool_calls: bool = False
    json_schema_output: bool = True

    def is_signature_rejection(self, error: str) -> bool:
        if not self.serial_tool_calls:
            return False
        lowered_error = error.lower()
        if "thought_signature" in lowered_error or "thought signature" in lowered_error:
            return True
        # Google sometimes redacts the field-level detail and returns only a
        # generic INVALID_ARGUMENT after accepting several tool turns.
        return "invalid_argument" in lowered_error or "invalid argument" in lowered_error


_GEMINI_MAJOR_RE = re.compile(r"^google/gemini-(\d+)")


def _gemini_major(model: str) -> int:
    match = _GEMINI_MAJOR_RE.match(model)
    return int(match.group(1)) if match else 0


def _gemini_signs_tool_calls(model: str) -> bool:
    """Gemini 3 and later reject tool history with a missing thought signature.

    Observed on 3.x and on google/gemini-4-argon (generic INVALID_ARGUMENT after
    several accepted tool turns), so the gate is the generation, not one prefix.
    """
    return _gemini_major(model) >= 3


def model_protocol_profile(model: str) -> ModelProtocolProfile:
    """Resolve stable request/transcript behavior without a live API lookup."""

    if not model.startswith("google/gemini-"):
        return ModelProtocolProfile(max_response_tokens=MAX_RESPONSE_TOKENS)
    return ModelProtocolProfile(
        max_response_tokens=GEMINI_MAX_RESPONSE_TOKENS,
        preserve_provider_metadata=True,
        serial_tool_calls=_gemini_signs_tool_calls(model),
        # gemini-4-argon rejects any request carrying response_format json_schema
        # (INVALID_ARGUMENT), so a lane that hit the tool budget could never
        # finish. Its natural JSON finish already parses; rely on that instead.
        json_schema_output=_gemini_major(model) < 4,
    )


def parse_model_routes(raw: str | None) -> dict[str, dict[str, Any]]:
    """Validate trusted per-model routing; never apply one lane's tier to another."""
    if not raw or not raw.strip():
        return {}
    if len(raw.encode("utf-8")) > 8_000:
        raise ActionError("model_routes exceeds 8,000 UTF-8 bytes")
    try:
        routes = json.loads(raw)
    except ValueError as exc:
        raise ActionError("model_routes must be a JSON object") from exc
    if not isinstance(routes, dict) or len(routes) > LANE_CAP:
        raise ActionError("model_routes must be an object with at most four model keys")
    parsed: dict[str, dict[str, Any]] = {}
    for model, route in routes.items():
        slug = parse_slug(model, what="model_routes key")
        if slug != model:
            raise ActionError("model_routes keys must not contain surrounding whitespace")
        if not isinstance(route, dict) or not route or set(route) - {"provider", "service_tier"}:
            raise ActionError("each model_routes entry accepts only provider and service_tier")
        kwargs: dict[str, Any] = {}
        if "provider" in route:
            provider = route["provider"]
            if not isinstance(provider, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}", provider
            ):
                raise ActionError("model_routes provider must be an OpenRouter provider slug")
            kwargs["provider_order"] = [provider]
        if "service_tier" in route:
            tier = route["service_tier"]
            if not isinstance(tier, str) or tier not in {"default", "flex", "priority"}:
                raise ActionError("model_routes service_tier must be default, flex, or priority")
            kwargs["service_tier"] = tier
        parsed[slug] = kwargs
    return parsed


def provider_policy(
    *,
    order: list[str] | None = None,
    data_collection: str | None = None,
    zdr: bool = False,
) -> dict[str, Any] | None:
    """Build the shared OpenRouter routing/data policy.

    ``ValueError`` deliberately leaves callers free to translate invalid
    action configuration into their own public error type.
    """

    if data_collection not in {None, "allow", "deny"}:
        raise ValueError("provider_data_collection must be allow, deny, or unset")
    policy: dict[str, Any] = {}
    if order:
        policy.update({"order": list(order), "allow_fallbacks": False})
    if data_collection:
        policy["data_collection"] = data_collection
    if zdr:
        policy["zdr"] = True
    return policy or None


def base_chat_payload(
    *,
    model: str,
    response_schema: dict[str, Any],
    reasoning: dict[str, str] | None = None,
    provider_order: list[str] | None = None,
    provider_data_collection: str | None = None,
    provider_zdr: bool = False,
    service_tier: str | None = None,
) -> tuple[dict[str, Any], ModelProtocolProfile]:
    """Build fields shared by lane and judge chat-completion requests."""

    profile = model_protocol_profile(model)
    payload: dict[str, Any] = {"model": model, "usage": {"include": True}}
    if profile.json_schema_output:
        payload["response_format"] = {"type": "json_schema", "json_schema": response_schema}
    if profile.max_response_tokens is not None:
        payload["max_tokens"] = profile.max_response_tokens
    if reasoning:
        payload["reasoning"] = dict(reasoning)
    policy = provider_policy(
        order=provider_order,
        data_collection=provider_data_collection,
        zdr=provider_zdr,
    )
    if policy is not None:
        payload["provider"] = policy
    if service_tier is not None:
        payload["service_tier"] = service_tier
    return payload, profile


def parse_slug(raw: str, *, what: str) -> str:
    slug = raw.strip()
    if not _SLUG_RE.fullmatch(slug):
        raise ActionError(
            f"invalid OpenRouter {what} slug {slug!r}; "
            "expected provider/model (optional :variant), e.g. google/gemini-3.1-flash-lite"
        )
    return slug


def parse_judge_model(raw: str | None, *, default: str = DEFAULT_JUDGE_MODEL) -> str:
    text = (raw or "").strip()
    return parse_slug(text or default, what="judge_model")


def judge_is_needed(slugs: list[str]) -> bool:
    """One configured review lane (or a future single-persona run) skips the judge."""
    return len(slugs) >= 2


def parse_models(raw: str | None, *, default: str = DEFAULT_MODEL) -> list[str]:
    """Split a comma-separated models list. Empty input uses the default slug."""
    text = (raw or "").strip()
    if not text:
        slugs = [default]
    else:
        slugs = [part.strip() for part in text.split(",") if part.strip()]
    if not slugs:
        raise ActionError("models is empty after parsing; provide at least one OpenRouter slug")
    if len(slugs) > LANE_CAP:
        raise ActionError(
            f"models lists {len(slugs)} lanes; the hard cap is {LANE_CAP}. "
            f"Shorten the list to {LANE_CAP} or fewer OpenRouter slugs."
        )
    return [parse_slug(slug, what="model") for slug in slugs]


def matrix_payload(slugs: list[str]) -> list[dict[str, object]]:
    """GitHub Actions matrix.include entries (index + model)."""
    return [{"index": index, "model": slug} for index, slug in enumerate(slugs)]


def models_json(slugs: list[str]) -> str:
    return json.dumps(slugs)


def matrix_json(slugs: list[str]) -> str:
    return json.dumps(matrix_payload(slugs))
