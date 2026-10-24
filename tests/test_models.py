from __future__ import annotations

import json

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.models import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    GEMINI_MAX_RESPONSE_TOKENS,
    LANE_CAP,
    MAX_RESPONSE_TOKENS,
    ModelProtocolProfile,
    base_chat_payload,
    judge_is_needed,
    matrix_json,
    model_protocol_profile,
    parse_judge_model,
    parse_models,
    provider_policy,
)


def test_empty_models_uses_default_grok_slug() -> None:
    assert parse_models("") == [DEFAULT_MODEL]
    assert parse_models(None) == ["x-ai/grok-4.6"]


def test_comma_separated_list_trims_and_keeps_order() -> None:
    slugs = parse_models(" x-ai/grok-4.6, anthropic/claude-sonnet-4.6 ,google/gemini-2.5-pro ")
    assert slugs == [
        "x-ai/grok-4.6",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-2.5-pro",
    ]


def test_duplicate_slugs_are_separate_lanes() -> None:
    slugs = parse_models("x-ai/grok-4.6,x-ai/grok-4.6")
    assert slugs == ["x-ai/grok-4.6", "x-ai/grok-4.6"]


def test_variant_suffix_is_allowed() -> None:
    assert parse_models("x-ai/grok-4.6:beta") == ["x-ai/grok-4.6:beta"]


def test_four_lanes_is_at_cap() -> None:
    slugs = parse_models(
        "x-ai/grok-4.6,anthropic/claude-sonnet-4.6,google/gemini-2.5-pro,openai/gpt-5"
    )
    assert len(slugs) == LANE_CAP == 4


def test_five_lanes_fails_clearly() -> None:
    with pytest.raises(ActionError, match="hard cap is 4"):
        parse_models(
            "x-ai/grok-4.6,anthropic/claude-sonnet-4.6,"
            "google/gemini-2.5-pro,openai/gpt-5,meta-llama/llama-4-maverick"
        )


def test_only_commas_is_empty_error() -> None:
    with pytest.raises(ActionError, match="empty after parsing"):
        parse_models(",,,")


def test_invalid_slug_rejected() -> None:
    with pytest.raises(ActionError, match="invalid OpenRouter model slug"):
        parse_models("grok-4.6")
    with pytest.raises(ActionError, match="invalid OpenRouter model slug"):
        parse_models("x-ai/grok 4.6")


def test_judge_model_defaults_and_parses() -> None:
    assert parse_judge_model("") == DEFAULT_JUDGE_MODEL == "openai/gpt-5.6-luna"
    assert parse_judge_model("anthropic/claude-haiku-4.5") == "anthropic/claude-haiku-4.5"
    with pytest.raises(ActionError, match="judge_model"):
        parse_judge_model("not-a-slug")


def test_one_lane_skips_judge() -> None:
    assert not judge_is_needed(["x-ai/grok-4.6"])
    assert judge_is_needed(["x-ai/grok-4.6", "openai/gpt-4.1-nano"])


def test_matrix_json_includes_indexes() -> None:
    payload = json.loads(matrix_json(["x-ai/grok-4.6", "anthropic/claude-sonnet-4.6"]))
    assert payload == [
        {"index": 0, "model": "x-ai/grok-4.6"},
        {"index": 1, "model": "anthropic/claude-sonnet-4.6"},
    ]


def test_model_protocol_profile_is_model_family_policy() -> None:
    assert model_protocol_profile("example/model") == ModelProtocolProfile(
        max_response_tokens=MAX_RESPONSE_TOKENS
    )
    assert model_protocol_profile("google/gemini-2.5-pro") == ModelProtocolProfile(
        max_response_tokens=GEMINI_MAX_RESPONSE_TOKENS,
        preserve_provider_metadata=True,
    )
    gemini_3 = model_protocol_profile("google/gemini-3.1-flash-lite")
    assert gemini_3.serial_tool_calls
    assert gemini_3.is_signature_rejection("provider returned INVALID_ARGUMENT")
    assert not model_protocol_profile("example/model").is_signature_rejection(
        "provider returned invalid_argument"
    )


@pytest.mark.parametrize(
    "model", ["google/gemini-4-argon", "google/gemini-4.0-flash", "google/gemini-10-pro"]
)
def test_later_gemini_generations_keep_signature_protections(model: str) -> None:
    profile = model_protocol_profile(model)
    assert profile.serial_tool_calls
    assert profile.is_signature_rejection("Request contains an invalid argument.")


def test_gemini_4_and_later_omit_the_json_schema_response_format() -> None:
    newer, _ = base_chat_payload(model="google/gemini-4-argon", response_schema={"name": "r"})
    older, _ = base_chat_payload(
        model="google/gemini-3.1-flash-lite", response_schema={"name": "r"}
    )
    assert "response_format" not in newer
    assert older["response_format"]["type"] == "json_schema"


def test_provider_policy_centralizes_routing_and_data_controls() -> None:
    assert provider_policy() is None
    assert provider_policy(order=["Baseten"], data_collection="deny", zdr=True) == {
        "order": ["Baseten"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }
    with pytest.raises(ValueError, match="allow, deny, or unset"):
        provider_policy(data_collection="sometimes")


def test_base_chat_payload_applies_protocol_and_shared_fields() -> None:
    payload, profile = base_chat_payload(
        model="google/gemini-3.1-flash-lite",
        response_schema={"name": "result"},
        reasoning={"effort": "minimal"},
        provider_data_collection="deny",
    )

    assert profile.serial_tool_calls
    assert payload == {
        "model": "google/gemini-3.1-flash-lite",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "result"},
        },
        "usage": {"include": True},
        "max_tokens": GEMINI_MAX_RESPONSE_TOKENS,
        "reasoning": {"effort": "minimal"},
        "provider": {"data_collection": "deny"},
    }


def test_base_chat_payload_optionally_requests_service_tier() -> None:
    plain, _ = base_chat_payload(model="example/model", response_schema={"name": "result"})
    requested, _ = base_chat_payload(
        model="example/model",
        response_schema={"name": "result"},
        service_tier="flex",
    )

    assert "service_tier" not in plain
    assert requested["service_tier"] == "flex"
