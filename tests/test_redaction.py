from __future__ import annotations

from or_pr_review.redaction import looks_like_dotenv, redact, secret_values


def test_redact_literal_secrets() -> None:
    text = redact("token=abc Authorization: Bearer super-secret", extra=["super-secret"])
    assert "super-secret" not in text
    assert "[redacted]" in text


def test_redact_env_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-please-hide")
    text = redact("calling with sk-or-v1-please-hide")
    assert "sk-or-v1-please-hide" not in text


def test_redact_discovers_secrets_added_after_an_earlier_call(monkeypatch) -> None:
    assert redact("ordinary output") == "ordinary output"

    monkeypatch.setenv("LATE_PROVIDER_API_KEY", "late-provider-secret")

    assert redact("received late-provider-secret") == "received [redacted]"


def test_redact_common_json_key_shapes() -> None:
    text = redact(
        '{"api_key": "sk-json-secret", "OPENAI_API_KEY": "sk-openai-secret", '
        '"access_token": "token-json-secret"}'
    )
    assert "sk-json-secret" not in text
    assert "sk-openai-secret" not in text
    assert "token-json-secret" not in text
    assert '"api_key": "[redacted]"' in text


def test_redact_multi_segment_secret_key_names() -> None:
    text = redact(
        '{"AZURE_OPENAI_API_KEY": "sk-azure-secret"} CUSTOM_SERVICE_TOKEN=custom-service-secret'
    )
    assert "sk-azure-secret" not in text
    assert "custom-service-secret" not in text


def test_redact_quoted_assignment_with_escaped_quote() -> None:
    text = redact(r'token="super-secret\"suffix"')
    assert "super-secret" not in text
    assert "suffix" not in text
    assert text == 'token="[redacted]"'


def test_redact_prose_api_key_and_bearer_forms() -> None:
    text = redact(
        "Invalid API key: sk-or-v1-prose-secret; API-key=sk-api-secret; "
        "Authorization: Bearer sk-bearer-secret"
    )
    assert "sk-or-v1-prose-secret" not in text
    assert "sk-api-secret" not in text
    assert "sk-bearer-secret" not in text
    assert "Bearer [redacted]" in text


def test_secret_values_covers_provider_and_suffix_env_names(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-env-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google-env-secret")
    monkeypatch.setenv("CUSTOM_SERVICE_TOKEN", "custom-token-secret")
    monkeypatch.setenv("SHORT_TOKEN", "abc")
    values = secret_values()
    assert "sk-openai-env-secret" in values
    assert "sk-anthropic-env-secret" in values
    assert "AIza-google-env-secret" in values
    assert "custom-token-secret" in values
    assert "abc" not in values
    assert redact("keys: sk-openai-env-secret custom-token-secret") == (
        "keys: [redacted] [redacted]"
    )


def test_short_literal_extra_is_not_globally_replaced() -> None:
    assert redact("abc appears in this prose", extra=["abc"]) == ("abc appears in this prose")


def test_looks_like_dotenv() -> None:
    assert looks_like_dotenv("FOO=1\nBAR=2\nBAZ=3\n")
    assert not looks_like_dotenv("just a paragraph of review text\nwith two lines")


_BODY_429 = (
    '{"error":{"message":"Rate limit exceeded: free-models-per-day-high-balance. ","code":429,'
    '"metadata":{"headers":{"X-RateLimit-Limit":"50"},"limit_source":"x","provider_name":null}},'
    '"user_id":"user_2abcDEF"}'
)


def test_scrub_error_body_json_keeps_only_message() -> None:
    from or_pr_review.redaction import scrub_error_body

    out = scrub_error_body(_BODY_429, 429)
    assert f"HTTP 429: {out}" == "HTTP 429: Rate limit exceeded: free-models-per-day-high-balance."
    assert "user_" not in out and "RateLimit" not in out
    assert scrub_error_body('{"error":{"message":"bad","code":"x"}}', 400) == "bad (code x)"


def test_scrub_error_body_non_json_truncated_and_scrubbed() -> None:
    from or_pr_review.redaction import scrub_error_body

    body = (
        'upstream "user_id":"user_2abcDEF" Authorization: Bearer tok123 X-RateLimit-Limit: 50 '
        + "x" * 500
    )
    out = scrub_error_body(body)
    assert len(out) < 260
    for leaked in ("user_2abcDEF", "tok123", ": 50"):
        assert leaked not in out


def test_scrub_error_body_never_emits_openrouter_key() -> None:
    from or_pr_review.redaction import scrub_error_body

    key = "sk-or-v1-abc123def456"
    assert key not in scrub_error_body(f"invalid key {key}")
    assert key not in scrub_error_body(f'{{"error":{{"message":"bad key {key}"}}}}')
