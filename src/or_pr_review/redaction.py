"""Never print API keys, tokens, or .env values."""

from __future__ import annotations

import json
import os
import re

_SECRET_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "XAI_API_KEY",
)

_KEY_ASSIGN_RE = re.compile(
    r"(?ix)"
    r"(?P<prefix>\b(?:[A-Z][A-Z0-9_]*(?:_API_KEY|_TOKEN|_SECRET)|"
    r"OPENROUTER_API_KEY|GITHUB_TOKEN|GH_TOKEN|XAI_API_KEY|"
    r"api\s*[_-]?\s*key|token|secret|password|authorization)\b"
    r"[\"']?\s*[:=]\s*)"
    r"(?:"
    r"(?P<quote>[\"'])(?P<quoted>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"
    r"|(?P<bare>[^\s,}\]]+)"
    r")"
)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)([^\s\"',}\]]+)")
_ENV_LINE_RE = re.compile(
    r"(?m)^([A-Za-z_][A-Za-z0-9_]*)=([^\n]+)$",
)
_PLACEHOLDER_VALUES = frozenset({"", "null", "none", "true", "false"})
_MIN_LITERAL_SECRET_LENGTH = 8


def secret_values() -> list[str]:
    names = set(_SECRET_ENV_NAMES)
    names.update(
        name for name in os.environ if name.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET"))
    )
    values = {
        raw
        for name in names
        if (raw := os.environ.get(name, "").strip())
        and len(raw) >= _MIN_LITERAL_SECRET_LENGTH
        and raw.lower() not in _PLACEHOLDER_VALUES
    }
    return sorted(values, key=len, reverse=True)


def _redact_assignment(match: re.Match[str]) -> str:
    quoted = match.group("quoted")
    bare = match.group("bare")
    value = quoted if quoted is not None else bare
    if value is None or value.lower() in _PLACEHOLDER_VALUES:
        return match.group(0)
    # Preserve a quoted Bearer scheme while hiding only its credential.
    if quoted is not None and value.lower().startswith("bearer "):
        return (
            match.group("prefix")
            + match.group("quote")
            + _BEARER_RE.sub(r"\1[redacted]", value, count=1)
            + match.group("quote")
        )
    # A preceding Bearer pass already replaced the credential.
    if bare is not None and bare.lower() == "bearer":
        return match.group(0)
    if quoted is not None:
        return match.group("prefix") + match.group("quote") + "[redacted]" + match.group("quote")
    return match.group("prefix") + "[redacted]"


def redact(text: str, extra: list[str] | None = None) -> str:
    if not text:
        return text
    out = text
    for value in list(extra or []) + secret_values():
        # Very short values (for example a test token of "abc") are common
        # words and would cause surprising global replacement.
        if (
            value
            and len(value.strip()) >= _MIN_LITERAL_SECRET_LENGTH
            and value.strip().lower() not in _PLACEHOLDER_VALUES
            and value in out
        ):
            out = out.replace(value, "[redacted]")
    out = _BEARER_RE.sub(r"\1[redacted]", out)
    out = _KEY_ASSIGN_RE.sub(_redact_assignment, out)
    return out


_SCRUB_RES = (
    re.compile(r"(?i)[\"']?user_id[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,}\]]+)"),
    re.compile(r"(?i)\bauthorization\b[\"']?\s*[:=]?\s*(?:bearer\s+)?[^\s,}\]]*"),
    re.compile(r"(?i)\bbearer\s+[^\s\"',}\]]+"),
    re.compile(r"\bsk-or-[A-Za-z0-9_.-]*"),
    re.compile(r"(?i)[\"']?x-ratelimit-[\w-]+[\"']?\s*[:=]\s*(?:\"[^\"]*\"|[^\s,}\]]+)"),
)
ERROR_BODY_LIMIT = 200


def scrub_error_body(body: str, status: int | None = None) -> str:
    """Reduce a provider error body to a message safe for a public comment."""
    try:
        err = json.loads(body).get("error")
        msg = err.get("message") if isinstance(err, dict) else None
        if isinstance(msg, str) and msg.strip():
            code = err.get("code")
            text = msg.strip() + (f" (code {code})" if code not in (None, "", status) else "")
            return redact(_scrub(text))
    except (ValueError, AttributeError):
        pass
    return redact(_scrub(body[:ERROR_BODY_LIMIT]))


def _scrub(text: str) -> str:
    for rx in _SCRUB_RES:
        text = rx.sub("[redacted]", text)
    return text


def looks_like_dotenv(text: str) -> bool:
    """Heuristic: a block of KEY=value lines that should not be logged."""
    lines = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 2:
        return False
    hits = sum(1 for line in lines if _ENV_LINE_RE.match(line))
    return hits >= 2 and hits / len(lines) >= 0.7
