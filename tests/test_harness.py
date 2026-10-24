from __future__ import annotations

import email.message
import inspect
import io
import json
import urllib.error
from pathlib import Path

import pytest

from or_pr_review.errors import ActionError, LaneError
from or_pr_review.harness import (
    BLAST_RADIUS_NUDGE,
    BUDGET_EXHAUSTED_NOTICE,
    DEADLINE_FINALIZE_NOTICE,
    DEFAULT_MAX_TOOL_TURNS,
    MAX_GEMINI_RESPONSE_TOKENS,
    MAX_TOOL_TURNS,
    MISSING_SIGNATURE_FINALIZE_NOTICE,
    OpenRouterHTTPError,
    _assistant_record,
    _LaneClock,
    _LoopPhase,
    _LoopState,
    parse_max_tool_turns,
    require_openrouter_key,
    run_lane,
)
from or_pr_review.models import GEMINI_MAX_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS


def test_require_key_fail_closed() -> None:
    with pytest.raises(ActionError, match="OPENROUTER_API_KEY"):
        require_openrouter_key({})


def test_lane_clock_gives_finish_the_remaining_budget() -> None:
    clock = _LaneClock(deadline=100.0, reserve_seconds=20.0)

    assert (
        clock.prepare_request(
            now=50.0,
            tools_active=True,
        )
        == 80.0
    )
    assert clock.finalize_due(now=80.0, tools_active=True)
    assert (
        clock.prepare_request(
            now=80.0,
            tools_active=False,
        )
        == 100.0
    )
    assert (
        clock.prepare_request(
            now=90.0,
            tools_active=False,
        )
        == 100.0
    )
    assert clock.tool_deadline() == 80.0


def test_loop_state_phase_owns_tool_and_finalize_flags() -> None:
    state = _LoopState(phase=_LoopPhase.NUDGE, use_schema=False)
    assert state.tools_active
    assert state.force_tool

    state.enter_finalize("signature", salvaged=True)

    assert not state.tools_active
    assert state.use_schema
    assert state.signature_finalizing
    assert state.salvage_attempted

    state.enter_finalize("deadline", salvaged=True)
    state.enter_finalize("signature", salvaged=True)
    assert state.deadline_limited
    assert state.signature_finalizing


def test_run_lane_captures_provider_and_pins_routing(tmp_path: Path) -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return {
            "provider": "Baseten",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    result = run_lane(
        model="z-ai/glm-5.3-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        provider_order=["baseten"],
        chat=chat,
    )
    assert result.ok
    assert result.provider == "Baseten"
    assert payloads[0]["max_tokens"] == MAX_RESPONSE_TOKENS
    assert payloads[0]["provider"] == {"order": ["baseten"], "allow_fallbacks": False}
    # Round-trips through the lane artifact.
    from or_pr_review.schema import parse_lane_artifact

    assert parse_lane_artifact(result.to_dict()).provider == "Baseten"


def test_run_lane_reuses_its_validated_final_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful response should not be parsed once in the loop and again at exit."""
    from or_pr_review import harness
    from or_pr_review.schema import Finding, Resolution

    original = harness.parse_lane_payload
    calls = 0

    def count_parse(
        payload: object,
        model_id: str,
        *,
        expect_coverage: bool = False,
        expect_resolutions: bool = False,
        diagnostics: dict[str, int] | None = None,
    ) -> tuple[list[Finding], list[Resolution], list[tuple[str, int]]]:
        nonlocal calls
        calls += 1
        return original(
            payload,
            model_id,
            expect_coverage=expect_coverage,
            expect_resolutions=expect_resolutions,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(harness, "parse_lane_payload", count_parse)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        chat=lambda _payload: _findings_reply(),
    )

    assert result.ok
    assert calls == 1


def test_run_lane_enforces_explicit_benchmark_data_policy(tmp_path: Path) -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return {
            "provider": "DeepInfra",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    result = run_lane(
        model="z-ai/glm-5.3-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        provider_order=["DeepInfra"],
        provider_data_collection="deny",
        provider_zdr=True,
        chat=chat,
    )

    assert result.ok
    assert payloads[0]["provider"] == {
        "order": ["DeepInfra"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }


def test_gemini_lane_caps_reserved_output_tokens(tmp_path: Path) -> None:
    payloads: list[dict] = []

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        chat=lambda payload: payloads.append(payload) or _findings_reply(),
    )

    assert result.ok
    assert payloads[0]["max_tokens"] == GEMINI_MAX_RESPONSE_TOKENS
    assert MAX_GEMINI_RESPONSE_TOKENS == GEMINI_MAX_RESPONSE_TOKENS


@pytest.mark.parametrize(
    ("data_collection", "zdr", "expected"),
    [
        ("deny", False, {"data_collection": "deny"}),
        (None, True, {"zdr": True}),
    ],
)
def test_run_lane_supports_unpinned_provider_data_policy(
    tmp_path: Path,
    data_collection: str | None,
    zdr: bool,
    expected: dict[str, object],
) -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return {
            "provider": "example",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        provider_data_collection=data_collection,
        provider_zdr=zdr,
        chat=chat,
    )

    assert result.ok
    assert payloads[0]["provider"] == expected


def test_run_lane_rejects_invalid_provider_data_collection(tmp_path: Path) -> None:
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        provider_data_collection="sometimes",
        chat=lambda _payload: {},
    )

    assert not result.ok
    assert "provider_data_collection" in (result.error or "")


def test_lane_parses_structured_findings(tmp_path: Path) -> None:
    def chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings":[{"title":"Race","body":"check-then-act",'
                            '"severity":"risk","file":"db.py","line":9}]}'
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cost": 0.004,
            },
        }

    (tmp_path / "db.py").write_text("\n".join(f"line{i}" for i in range(1, 20)), encoding="utf-8")
    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=0,
    )
    assert result.ok
    assert result.findings[0].title == "Race"
    # A real path with an in-range line passes the anchor gate untouched.
    assert result.findings[0].file == "db.py"
    assert result.findings[0].line == 9
    assert result.prompt_tokens == 10
    assert result.cost_usd == 0.004


def test_anchor_gate_nulls_impossible_locations(tmp_path: Path) -> None:
    from or_pr_review.harness import sanitize_anchors
    from or_pr_review.schema import Finding

    (tmp_path / "real.py").write_text("one\ntwo\n", encoding="utf-8")

    def finding(title: str, file: str | None, line: int | None) -> Finding:
        return Finding(title=title, body="b", severity="risk", file=file, line=line, model_id="m")

    out = sanitize_anchors(
        [
            finding("valid anchor", "real.py", 2),
            finding("ghost path", "no/such.py", 3),
            finding("line beyond eof", "real.py", 99),
            finding("fileless blast radius", None, None),
        ],
        tmp_path,
    )
    # Never drops a finding — only the impossible parts of its anchor.
    assert [f.title for f in out] == [
        "valid anchor",
        "ghost path",
        "line beyond eof",
        "fileless blast radius",
    ]
    assert (out[0].file, out[0].line) == ("real.py", 2)
    assert (out[1].file, out[1].line) == (None, None)
    assert (out[2].file, out[2].line) == ("real.py", None)
    assert (out[3].file, out[3].line) == (None, None)


def test_anchor_gate_rejects_programmatic_traversal_before_filesystem_lookup(
    tmp_path: Path,
) -> None:
    """The public gate also protects Finding objects that bypass parsing."""
    from or_pr_review.harness import sanitize_anchors
    from or_pr_review.schema import Finding

    # This file is deliberately reachable from tmp_path via ``..``. The
    # anchor still must be removed rather than probing outside the checkout.
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret = 'not review context'\n", encoding="utf-8")
    finding = Finding(
        title="unsafe direct finding",
        body="b",
        severity="risk",
        file="../outside.py",
        line=1,
        model_id="m",
    )

    out = sanitize_anchors([finding], tmp_path)

    assert (out[0].file, out[0].line) == (None, None)


def test_anchor_gate_respects_snapshot_holes_and_directories(tmp_path: Path) -> None:
    from or_pr_review.harness import sanitize_anchors
    from or_pr_review.schema import Finding

    workspace = tmp_path / "inert-checkout"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "small.py").write_text("one\n", encoding="utf-8")
    # The manifest records the commit's FULL tracked list — including a large
    # generated file that materialization deliberately skipped.
    (tmp_path / "inert-checkout.paths").write_text(
        "pkg/small.py\npackage-lock.json\n", encoding="utf-8"
    )

    def finding(title: str, file: str | None, line: int | None):
        return Finding(title=title, body="b", severity="risk", file=file, line=line, model_id="m")

    out = sanitize_anchors(
        [
            finding("tracked but not materialized", "package-lock.json", 4021),
            finding("directory citation", "pkg", None),
            finding("truly untracked", "ghost.py", 1),
        ],
        workspace,
    )
    # A snapshot hole is NOT a ghost path: anchor kept, line uncheckable so kept.
    assert (out[0].file, out[0].line) == ("package-lock.json", 4021)
    # Directories are not valid file anchors and become body-only.
    assert (out[1].file, out[1].line) == (None, None)
    assert (out[2].file, out[2].line) == (None, None)


def test_anchor_gate_counts_lines_like_the_read_tools(tmp_path: Path) -> None:
    from or_pr_review.harness import sanitize_anchors
    from or_pr_review.schema import Finding

    # GitHub and the read tools see one line because only LF is a boundary;
    # Unicode separators and bare CR remain part of that line.
    (tmp_path / "odd.py").write_bytes("a b\rc\n".encode())
    finding = Finding(
        title="cites beyond the only LF-delimited line",
        body="b",
        severity="risk",
        file="odd.py",
        line=3,
        model_id="m",
    )
    out = sanitize_anchors([finding], tmp_path)
    assert (out[0].file, out[0].line) == ("odd.py", None)


def test_anchor_gate_never_follows_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review.harness import sanitize_anchors
    from or_pr_review.schema import Finding

    outside = tmp_path.parent / "outside-anchor.py"
    outside.write_text("secret\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    finding = Finding(
        title="symlink citation",
        body="b",
        severity="risk",
        file="linked.py",
        line=1,
        model_id="m",
    )
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path == link:
            pytest.fail("anchor gate must not open a symlink")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    out = sanitize_anchors([finding], tmp_path)

    assert (out[0].file, out[0].line) == (None, None)


def test_anchor_gate_does_not_read_files_beyond_snapshot_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness
    from or_pr_review.schema import Finding

    target = tmp_path / "large.py"
    target.write_bytes(b"one\ntwo\n")
    monkeypatch.setattr(harness, "MAX_OVERSIZED_MATERIALIZED_FILE", 4)
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path == target:
            pytest.fail("oversized anchor target must not be read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    finding = Finding(
        title="large citation",
        body="b",
        severity="risk",
        file="large.py",
        line=99,
        model_id="m",
    )

    out = harness.sanitize_anchors([finding], tmp_path)

    # The file is real, but its line count is intentionally left unverified.
    assert (out[0].file, out[0].line) == ("large.py", 99)


def test_run_lane_applies_the_anchor_gate(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("one\ntwo\n", encoding="utf-8")

    def chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings":['
                            '{"title":"ghost","body":"b","severity":"risk","file":"no.py","line":1},'
                            '{"title":"eof","body":"b","severity":"nit","file":"real.py","line":99}'
                            "]}"
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=0,
    )
    assert result.ok
    assert (result.findings[0].file, result.findings[0].line) == (None, None)
    assert (result.findings[1].file, result.findings[1].line) == ("real.py", None)


def test_run_lane_gates_toolless_runs_via_anchor_root(tmp_path: Path) -> None:
    def chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings":[{"title":"ghost","body":"b","severity":"risk",'
                            '"file":"no.py","line":1}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        anchor_root=tmp_path,
        chat=chat,
        max_tool_turns=0,
    )
    assert result.ok
    assert (result.findings[0].file, result.findings[0].line) == (None, None)


def test_lane_tool_call_then_findings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = {"n": 0}

    def chat(payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"findings":[]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 3, "cost": 0.003},
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=4,
    )
    assert result.ok
    assert result.findings == []
    assert calls["n"] == 2
    assert result.prompt_tokens == 25
    assert result.completion_tokens == 5
    assert result.cost_usd == 0.004


def test_lane_bad_json_fail_opens() -> None:
    def chat(_payload: dict) -> dict:
        return {"choices": [{"message": {"content": "sorry I cannot"}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert result.findings == []
    assert result.error


def test_lane_http_style_error_fail_opens() -> None:
    from or_pr_review.errors import LaneError

    def chat(_payload: dict) -> dict:
        raise LaneError("OpenRouter HTTP 429: rate limited")

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert "429" in (result.error or "")


def test_default_first_pass_tool_budget_is_fifty() -> None:
    assert DEFAULT_MAX_TOOL_TURNS == 50
    assert MAX_TOOL_TURNS == 50
    assert inspect.signature(run_lane).parameters["max_tool_turns"].default == 50
    assert parse_max_tool_turns(None) == 50
    assert parse_max_tool_turns("") == 50
    assert parse_max_tool_turns("30") == 30
    assert parse_max_tool_turns("0") == 0
    with pytest.raises(ActionError, match="integer"):
        parse_max_tool_turns("nope")
    with pytest.raises(ActionError, match="0 through"):
        parse_max_tool_turns("-1")
    with pytest.raises(ActionError, match="0 through"):
        parse_max_tool_turns("1001")


def _tool_reply() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ],
                }
            }
        ]
    }


def _gemini_tool_reply(*, call_ids: tuple[str, ...] = ("call_1",)) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                        for call_id in call_ids
                    ],
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "data": "opaque-google-signature",
                            "id": "sig-1",
                            "format": "google-gemini-v1",
                            "index": 0,
                        }
                    ],
                    "reasoning_content": "provider reasoning alias",
                    "provider_metadata": {"opaque": "keep-exactly"},
                }
            }
        ]
    }


def _findings_reply() -> dict:
    return {"choices": [{"message": {"content": '{"findings":[]}'}}]}


def test_tools_enabled_omits_json_schema_on_first_turn(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert "response_format" not in payloads[0]
    assert payloads[0].get("tools")
    assert payloads[0].get("tool_choice") == "auto"


def test_zero_tool_finish_is_nudged_then_required(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _findings_reply()
        if len(payloads) == 2:
            assert any(item.get("content") == BLAST_RADIUS_NUDGE for item in payload["messages"])
            assert payload.get("tool_choice") == "required"
            assert "response_format" not in payload
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert len(payloads) == 3


def test_default_budget_allows_more_than_legacy_eight_turns(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = {"n": 0}

    def chat(_payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] <= 12:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert calls["n"] == 13


def test_schema_used_when_tools_disabled() -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert "response_format" in payloads[0]
    assert "tools" not in payloads[0]


def _assert_valid_tool_pairing(messages: list[dict]) -> None:
    """Every assistant tool_calls entry must be followed by matching tool results."""
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                index += 1
                assert index < len(messages), f"missing tool result for {call.get('id')!r}"
                nxt = messages[index]
                assert nxt.get("role") == "tool", f"dangling tool_calls before {nxt!r}"
                assert nxt.get("tool_call_id") == call.get("id")
        index += 1


def test_assistant_record_preserves_reasoning() -> None:
    details = [{"type": "reasoning.text", "text": "plan"}]
    record = _assistant_record(
        {"content": "", "tool_calls": [{"id": "c1"}], "reasoning_details": details}
    )
    assert record["reasoning_details"] == details
    plain = _assistant_record({"content": "x", "reasoning": "thoughts"})
    assert plain["reasoning"] == "thoughts"
    bare = _assistant_record({"content": "x"})
    assert "reasoning" not in bare and "reasoning_details" not in bare


def test_assistant_record_preserves_complete_provider_message_verbatim() -> None:
    message = _gemini_tool_reply()["choices"][0]["message"]
    record = _assistant_record(message, preserve_provider_metadata=True)

    assert record == message
    assert record is not message
    assert record["tool_calls"] is not message["tool_calls"]
    assert record["reasoning_content"] == "provider reasoning alias"
    assert record["provider_metadata"] == {"opaque": "keep-exactly"}


def test_assistant_record_drops_unrecognized_metadata_for_other_providers() -> None:
    message = _gemini_tool_reply()["choices"][0]["message"]
    record = _assistant_record(message)

    assert "provider_metadata" not in record
    assert "reasoning_content" not in record
    assert record["tool_calls"] == message["tool_calls"]
    assert record["reasoning_details"] == message["reasoning_details"]


@pytest.mark.parametrize("call_ids", [("call_1",), ("call_1", "call_2")])
def test_gemini_signed_tool_message_round_trips_exactly(
    tmp_path: Path, call_ids: tuple[str, ...]
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    signed_message = _gemini_tool_reply(call_ids=call_ids)["choices"][0]["message"]
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return {"choices": [{"message": signed_message}]}
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=2,
    )

    assert result.ok
    assert result.thought_signature_tool_turns == 1
    echoed = next(
        message
        for message in payloads[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert echoed == signed_message
    assert [
        message["tool_call_id"]
        for message in payloads[1]["messages"]
        if message.get("role") == "tool"
    ] == list(call_ids)


def test_gemini_missing_signature_finalizes_without_executing_bad_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        if len(payloads) == 2:
            return _tool_reply()
        assert "tools" not in payload
        assert "response_format" in payload
        assert payload["messages"][-1]["role"] == "user"
        assert payload["messages"][-1]["content"].endswith(MISSING_SIGNATURE_FINALIZE_NOTICE)
        assert not any(
            left.get("role") == right.get("role") == "user"
            for left, right in zip(payload["messages"], payload["messages"][1:], strict=False)
        )
        assert not any(message.get("tool_calls") for message in payload["messages"])
        assert not any(message.get("role") == "tool" for message in payload["messages"])
        assert any(
            'Tool: read_file\nArguments: {"path": "a.py"}\nResult:\n'
            in str(message.get("content", ""))
            for message in payload["messages"]
        )
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.tool_rounds == 1
    assert result.thought_signature_recoveries == 1
    assert result.thought_signature_tool_turns == 1
    assert result.sanitized_tool_turns == 1
    assert len(payloads) == 3


def test_gemini_signed_fifty_turn_chain_round_trips(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) <= 50:
            reply = _gemini_tool_reply(call_ids=(f"call_{len(payloads)}",))
            detail = reply["choices"][0]["message"]["reasoning_details"][0]
            detail["data"] = f"opaque-signature-{len(payloads)}"
            detail["id"] = f"sig-{len(payloads)}"
            return reply
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=51,
    )

    assert result.ok
    assert result.tool_rounds == 50
    assert result.thought_signature_tool_turns == 50
    assert len(payloads) == 51
    final_history = payloads[-1]["messages"]
    signed_turns = [message for message in final_history if message.get("tool_calls")]
    assert len(signed_turns) == 50
    assert [message["reasoning_details"][0]["data"] for message in signed_turns] == [
        f"opaque-signature-{index}" for index in range(1, 51)
    ]


def test_gemini_provider_400_after_tools_uses_sanitized_salvage(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        if len(payloads) == 2:
            raise LaneError("OpenRouter HTTP 400: INVALID_ARGUMENT thought signature")
        assert "tools" not in payload
        assert "response_format" in payload
        assert not any(message.get("tool_calls") for message in payload["messages"])
        assert not any(message.get("role") == "tool" for message in payload["messages"])
        assert any(
            'Tool: read_file\nArguments: {"path": "a.py"}\nResult:\n'
            in str(message.get("content", ""))
            for message in payload["messages"]
        )
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.thought_signature_tool_turns == 1
    assert result.sanitized_tool_turns == 1
    assert len(payloads) == 3


@pytest.mark.parametrize(
    "provider_error",
    [
        "OpenRouter HTTP 400: INVALID_ARGUMENT",
        "OpenRouter HTTP 400: Request contains an invalid argument.",
    ],
)
def test_gemini_generic_invalid_argument_after_tools_uses_sanitized_salvage(
    tmp_path: Path, provider_error: str
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        if len(payloads) == 2:
            raise LaneError(provider_error)
        assert "tools" not in payload
        assert "response_format" in payload
        assert not any(message.get("tool_calls") for message in payload["messages"])
        assert not any(message.get("role") == "tool" for message in payload["messages"])
        assert any(
            'Tool: read_file\nArguments: {"path": "a.py"}\nResult:\n'
            in str(message.get("content", ""))
            for message in payload["messages"]
        )
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.thought_signature_tool_turns == 1
    assert result.thought_signature_recoveries == 1
    assert result.sanitized_tool_turns == 1
    assert len(payloads) == 3


def test_gemini_invalid_argument_context_overflow_shrinks_before_salvage(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("x" * 10_000, encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        if len(payloads) == 2:
            raise LaneError("OpenRouter HTTP 400: INVALID_ARGUMENT context length exceeded")
        messages = payload["messages"]
        assert any(
            "observation truncated after a context overflow" in str(message.get("content", ""))
            for message in messages
        )
        assert not any(
            left.get("role") == right.get("role") == "user"
            for left, right in zip(messages, messages[1:], strict=False)
        )
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert result.ok
    assert result.thought_signature_recoveries == 1


def test_gemini_signature_salvage_gets_one_json_repair(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        if len(payloads) == 2:
            raise LaneError("OpenRouter HTTP 400: INVALID_ARGUMENT")
        if len(payloads) == 3:
            return {"choices": [{"message": {"content": "not valid JSON"}}]}
        assert "response_format" in payload
        assert "tools" not in payload
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.thought_signature_recoveries == 1
    assert len(payloads) == 4


def test_gemini_tool_call_during_signature_finalize_fails_visibly(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        # The first unsigned call enters signature recovery without executing
        # the tool. A second tool call must not turn that bounded recovery into
        # another tool loop.
        return _tool_reply() if len(payloads) == 1 else _gemini_tool_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )

    assert not result.ok
    assert result.requests == 2
    assert result.tool_rounds == 0
    assert result.thought_signature_recoveries == 1
    assert "another tool call during thought-signature recovery" in (result.error or "")


def test_unsolicited_tool_calls_fail_after_bounded_repair_rounds() -> None:
    from or_pr_review import harness

    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        # Copy the history because the lane intentionally mutates its shared
        # conversation after each request.
        payloads.append(json.loads(json.dumps(payload)))
        return _tool_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )

    assert not result.ok
    assert result.requests == harness.MAX_REPAIR_ROUNDS + 1
    assert "kept issuing tool calls after the tool budget was withdrawn" in (result.error or "")
    assert all("tools" not in payload for payload in payloads)
    assert all("response_format" in payload for payload in payloads)
    for payload in payloads:
        _assert_valid_tool_pairing(payload["messages"])


def test_gemini_deadline_finish_replays_signed_tool_protocol_with_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = {"value": 0.0}
    payloads: list[dict] = []
    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            now["value"] = 6.0
            return _gemini_tool_reply()
        assert payload.get("tools")
        assert payload.get("tool_choice") == "none"
        assert "response_format" in payload
        assert any(message.get("tool_calls") for message in payload["messages"])
        assert any(message.get("role") == "tool" for message in payload["messages"])
        now["value"] = 7.0
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        lane_timeout=10,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.thought_signature_tool_turns == 1
    assert result.sanitized_tool_turns is None
    assert len(payloads) == 2


def test_gemini_deadline_signature_rejection_sanitizes_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = {"value": 0.0}
    payloads: list[dict] = []
    request_deadlines: list[float | None] = []
    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])
    original_prepare = harness._LaneClock.prepare_request

    def capture_request_deadline(clock: harness._LaneClock, **kwargs: object) -> float | None:
        result = original_prepare(clock, **kwargs)
        request_deadlines.append(result)
        return result

    monkeypatch.setattr(harness._LaneClock, "prepare_request", capture_request_deadline)

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            now["value"] = 6.0
            return _gemini_tool_reply()
        if len(payloads) == 2:
            assert payload.get("tools")
            assert payload.get("tool_choice") == "none"
            raise LaneError(
                "OpenRouter HTTP 400: INVALID_ARGUMENT function call has invalid thought_signature"
            )
        if len(payloads) == 3:
            assert "tools" not in payload
            assert "response_format" in payload
            assert not any(message.get("tool_calls") for message in payload["messages"])
            assert not any(message.get("role") == "tool" for message in payload["messages"])
            assert any(
                'Tool: read_file\nArguments: {"path": "a.py"}\nResult:\n'
                in str(message.get("content", ""))
                for message in payload["messages"]
            )
            now["value"] = 7.0
            return {"choices": [{"message": {"content": "not valid JSON"}}]}
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        lane_timeout=10,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.thought_signature_tool_turns == 1
    assert result.thought_signature_recoveries == 1
    assert result.sanitized_tool_turns == 1
    assert len(payloads) == 4
    assert request_deadlines == [5.0, 10.0, 10.0, 10.0]


def test_budget_withdrawal_never_solicits_unserviceable_calls(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        _assert_valid_tool_pairing(payload["messages"])
        if len(payloads) <= 2:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=2,
    )
    assert result.ok
    assert len(payloads) == 3
    final = payloads[2]
    assert "tools" not in final
    assert "tool_choice" not in final
    assert "response_format" in final
    assert final["messages"][-1] == {"role": "user", "content": BUDGET_EXHAUSTED_NOTICE}


def test_unsolicited_tool_calls_are_stubbed_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        _assert_valid_tool_pairing(payload["messages"])
        if len(payloads) <= 2:
            # The second tool_calls reply arrives after tools were withdrawn.
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=1,
    )
    assert result.ok
    assert len(payloads) == 3
    assert "tools" not in payloads[1]
    stubs = [
        message
        for message in payloads[2]["messages"]
        if message.get("role") == "tool" and "not executed" in message.get("content", "")
    ]
    assert stubs and stubs[-1]["tool_call_id"] == "call_1"


def test_gemini_4_budget_finish_sends_no_json_schema(tmp_path: Path) -> None:
    # gemini-4-argon rejected every budget finish that carried response_format.
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        assert payload["tool_choice"] == "none"
        return _findings_reply()

    result = run_lane(
        model="google/gemini-4-argon",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=1,
    )
    assert result.ok
    assert result.salvaged is not True
    assert len(payloads) == 2
    assert not any("response_format" in payload for payload in payloads)


def test_malformed_finish_gets_one_schema_enforced_retry(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        if len(payloads) == 2:
            return {"choices": [{"message": {"content": "sorry, prose only"}}]}
        assert "response_format" in payload
        assert "tools" not in payload
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert len(payloads) == 3


def test_gemini_3_disables_parallel_tool_calls_only_while_tools_are_active(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _gemini_tool_reply()
        return _findings_reply()

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=1,
    )

    assert result.ok
    assert payloads[0]["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in payloads[1]
    assert payloads[1].get("tools")
    assert payloads[1].get("tool_choice") == "none"
    assert any(message.get("tool_calls") for message in payloads[1]["messages"])
    assert result.thought_signature_tool_turns == 1
    assert result.sanitized_tool_turns is None


def test_pre_gemini_3_models_keep_the_generic_tool_protocol(tmp_path: Path) -> None:
    """The signature workaround starts at Gemini 3; older generations keep the default."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            # An unsigned generic tool reply remains serviceable for a model
            # that has no empirically established signature contract.
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=1,
    )

    assert result.ok
    assert result.tool_rounds == 1
    assert result.thought_signature_tool_turns is None
    assert "parallel_tool_calls" not in payloads[0]
    assert "tools" not in payloads[1]


def test_non_gemini_tool_calls_keep_provider_default_parallelism(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=1,
    )

    assert result.ok
    assert "parallel_tool_calls" not in payloads[0]


def test_empty_finish_gets_one_schema_enforced_retry(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        if len(payloads) == 2:
            return {"choices": [{"message": {"content": ""}}]}
        assert "response_format" in payload
        assert "tools" not in payload
        assert "previous assistant message was empty" in payload["messages"][-1]["content"]
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert len(payloads) == 3


def test_empty_finish_fails_open_after_single_retry() -> None:
    calls = {"n": 0}

    def chat(_payload: dict) -> dict:
        calls["n"] += 1
        return {"choices": [{"message": {"content": ""}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert calls["n"] == 2


def test_malformed_finish_fails_open_after_single_retry() -> None:
    calls = {"n": 0}

    def chat(_payload: dict) -> dict:
        calls["n"] += 1
        return {"choices": [{"message": {"content": "still not json"}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert not result.ok
    assert calls["n"] == 2


def test_contradictory_resolution_gets_one_retry_then_converges() -> None:
    from or_pr_review.loop import LedgerFinding, LoopState, apply_round

    payloads: list[dict] = []
    contradictory = (
        '{"findings": [], "resolutions": ['
        '{"id": "r1-1", "status": "fixed_incorrectly", '
        '"note": "Actually fixed correctly in the current head."}]}'
    )
    consistent = (
        '{"findings": [], "resolutions": ['
        '{"id": "r1-1", "status": "fixed", '
        '"note": "Fixed correctly in the current head."}]}'
    )

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        content = contradictory if len(payloads) == 1 else consistent
        return {"choices": [{"message": {"content": content}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "verify"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        expect_resolutions=True,
        expected_resolution_ids={"r1-1"},
        chat=chat,
    )

    assert result.ok
    assert len(payloads) == 2
    assert "response_format" in payloads[1]
    assert "note contradicts" in payloads[1]["messages"][-1]["content"]
    assert result.resolutions[0].status == "fixed"

    prior = LedgerFinding(
        id="r1-1",
        severity="bug",
        file="src/app.py",
        line=3,
        title="Wrong enum",
        evidence="The implementation selects the wrong enum.",
        status="open",
    )
    outcome = apply_round(
        LoopState(mode="verify", round_number=2, prior_findings=(prior,)),
        [],
        {resolution.id: resolution for resolution in result.resolutions},
    )
    assert outcome.ledger.findings == ()
    assert outcome.open_issue_count == 0
    assert outcome.open_bug_count == 0


def test_repeated_resolution_contradiction_fails_visibly_after_retry() -> None:
    calls = {"n": 0}
    contradictory = (
        '{"findings": [], "resolutions": ['
        '{"id": "r1-1", "status": "fixed_incorrectly", '
        '"note": "Actually fixed correctly in the current head."}]}'
    )

    def chat(_payload: dict) -> dict:
        calls["n"] += 1
        return {"choices": [{"message": {"content": contradictory}}]}

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "verify"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        expect_resolutions=True,
        expected_resolution_ids={"r1-1"},
        chat=chat,
    )

    assert not result.ok
    assert calls["n"] == 2
    assert result.resolutions == []
    assert "note contradicts" in (result.error or "")


def _http_error(
    code: int, retry_after: str | None = None, body: bytes = b"body"
) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", code, "err", headers, io.BytesIO(body)
    )


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"choices": []}'


def test_openrouter_chat_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import harness

    attempts = {"n": 0}

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _http_error(429, retry_after="1")
        return _FakeResponse()

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    stats: dict[str, int] = {}
    parsed = harness.openrouter_chat(
        "sk-test", {"model": "m"}, timeout=5, sleep=sleeps.append, stats=stats
    )
    assert parsed == {"choices": []}
    assert attempts["n"] == 3
    assert sleeps == [1.0, 1.0]
    assert stats["retries"] == 2


@pytest.mark.parametrize(
    ("retry_after", "expected_sleep"),
    [
        ("not-a-delay", 2.0),
        ("999999999999", 30.0),
    ],
)
def test_openrouter_chat_bounds_or_ignores_bad_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str,
    expected_sleep: float,
) -> None:
    attempts = 0

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _http_error(429, retry_after=retry_after)
        return _FakeResponse()

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []

    from or_pr_review import harness

    assert harness.openrouter_chat("sk-test", {"model": "m"}, timeout=5, sleep=sleeps.append) == {
        "choices": []
    }
    assert sleeps == [expected_sleep]


def test_openrouter_chat_retries_mid_body_connection_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import http.client

    from or_pr_review import harness

    attempts = {"n": 0}

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            # A truncated chunked response raises from response.read(), not
            # from urlopen — it is neither URLError nor HTTPError.
            raise http.client.IncompleteRead(b"partial")
        if attempts["n"] == 2:
            raise ConnectionResetError("peer reset")
        return _FakeResponse()

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    stats: dict[str, int] = {}
    parsed = harness.openrouter_chat(
        "sk-test", {"model": "m"}, timeout=5, sleep=sleeps.append, stats=stats
    )
    assert parsed == {"choices": []}
    assert attempts["n"] == 3
    assert stats["retries"] == 2


def test_openrouter_chat_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import harness

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        raise _http_error(503)

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    with pytest.raises(LaneError, match="HTTP 503"):
        harness.openrouter_chat("sk-test", {"model": "m"}, timeout=5, sleep=sleeps.append)
    assert len(sleeps) == harness.MAX_HTTP_ATTEMPTS - 1
    assert sleeps == [2.0, 4.0, 8.0]


def test_openrouter_chat_rate_limit_retries_longer_with_stable_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    body = (
        b'{"error":{"message":"Provider returned error","code":429,'
        b'"metadata":{"provider_name":"Google"}}}'
    )

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        raise _http_error(429, body=body)

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    with pytest.raises(harness.OpenRouterHTTPError) as raised:
        harness.openrouter_chat(
            "sk-test", {"model": "google/gemini-3.8-flash"}, timeout=5, sleep=sleeps.append
        )

    assert raised.value.provider == "Google"
    assert len(sleeps) == harness.MAX_RATE_LIMIT_ATTEMPTS - 1
    assert sleeps == [
        min(
            harness._retry_delay(index, None)
            + harness._stable_rate_limit_jitter(b'{"model": "google/gemini-3.8-flash"}', index),
            harness.MAX_RETRY_AFTER_SECONDS,
        )
        for index in range(1, harness.MAX_RATE_LIMIT_ATTEMPTS)
    ]
    assert max(sleeps) <= harness.MAX_RETRY_AFTER_SECONDS


def test_lane_preserves_terminal_http_provider_and_nonbillable_cost() -> None:
    from or_pr_review import harness

    def chat(_payload: dict) -> dict:
        raise harness.OpenRouterHTTPError("OpenRouter HTTP 429: rate limited", provider="Google")

    result = run_lane(
        model="google/gemini-3.8-flash",
        messages=[{"role": "user", "content": "verify"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        chat=chat,
    )

    assert result.ok is False
    assert result.provider == "Google"
    assert result.cost_usd is None


def test_terminal_http_error_does_not_hide_prior_unknown_cost(tmp_path: Path) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = 0

    def chat(_payload: dict) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _tool_reply()
        raise harness.OpenRouterHTTPError("OpenRouter HTTP 429: rate limited", provider="Google")

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "verify"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        chat=chat,
    )

    assert result.ok is False
    assert result.provider == "Google"
    assert result.cost_usd is None


def test_http_error_provider_is_parsed_beyond_display_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    body = json.dumps(
        {
            "error": {
                "message": "x" * 1_000,
                "metadata": {"provider_name": "Google"},
            }
        }
    ).encode()

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        raise _http_error(400, body=body)

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    with pytest.raises(harness.OpenRouterHTTPError) as raised:
        harness.openrouter_chat("sk-test", {"model": "m"}, timeout=5)

    assert raised.value.provider == "Google"


def test_rate_limit_deadline_preserves_provider_and_nonbillable_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    body = b'{"error":{"metadata":{"provider_name":"Google"}}}'

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        raise _http_error(429, body=body)

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    with pytest.raises(harness.OpenRouterHTTPError) as raised:
        harness.openrouter_chat(
            "sk-test",
            {"model": "google/gemini-3.8-flash"},
            timeout=5,
            sleep=lambda _delay: None,
            deadline=harness.time.monotonic() + 1,
        )

    assert raised.value.provider == "Google"
    assert "budget exhausted" in str(raised.value)


def test_openrouter_chat_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from or_pr_review import harness

    def fake_urlopen(
        _request: object, timeout: int, total_timeout: float | None = None
    ) -> _FakeResponse:
        raise _http_error(400)

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    with pytest.raises(LaneError, match="HTTP 400"):
        harness.openrouter_chat("sk-test", {"model": "m"}, timeout=5, sleep=sleeps.append)
    assert sleeps == []


def test_lane_salvages_after_midloop_failure(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        if len(payloads) == 2:
            raise LaneError("OpenRouter HTTP 500: upstream unavailable")
        assert "response_format" in payload
        assert "tools" not in payload
        assert "already gathered" in payload["messages"][-1]["content"]
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert result.salvaged is True
    assert result.requests == 3
    assert result.tool_rounds == 1


def test_lane_does_not_salvage_after_out_of_credit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        raise OpenRouterHTTPError("OpenRouter HTTP 402: more credits", provider=None, status=402)

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert not result.ok
    assert result.salvaged is not True
    assert len(payloads) == 2


def test_lane_wall_clock_forces_structured_finish_before_job_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR358 regression: a tool loop must not run until GitHub cancels the job."""
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = {"value": 0.0}
    payloads: list[dict] = []

    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            # The first provider turn consumes the exploration budget.  A
            # pre-fix lane would simply continue requesting tools.
            now["value"] = 6.0
            return _tool_reply()
        assert "tools" not in payload
        assert "response_format" in payload
        assert payload["messages"][-1] == {
            "role": "user",
            "content": DEADLINE_FINALIZE_NOTICE,
        }
        now["value"] = 7.0
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        lane_timeout=10,
    )
    assert result.ok
    assert result.salvaged is True
    assert result.requests == 2
    assert result.tool_rounds == 1
    assert now["value"] < 10


@pytest.mark.parametrize("model", ["x-ai/grok-4.6", "google/gemini-3.8-flash"])
def test_deadline_finalize_transport_failure_keeps_one_repair_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = {"value": 0.0}
    payloads: list[dict] = []
    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            now["value"] = 6.0
            if model.startswith("google/gemini-"):
                return _gemini_tool_reply()
            return _tool_reply()
        if len(payloads) == 2:
            now["value"] = 8.0
            raise LaneError("OpenRouter HTTP 429: retry later")
        if model.startswith("google/gemini-"):
            assert payload["tool_choice"] == "none"
        else:
            assert "tools" not in payload
        assert "response_format" in payload
        assert "protected finalize request failed" in payload["messages"][-1]["content"]
        assert not any(
            left.get("role") == right.get("role") == "user"
            for left, right in zip(payload["messages"], payload["messages"][1:], strict=False)
        )
        now["value"] = 9.0
        return _findings_reply()

    result = run_lane(
        model=model,
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        lane_timeout=10,
    )

    assert result.ok
    assert result.salvaged is True
    assert result.requests == 3
    assert now["value"] < 10


@pytest.mark.parametrize("request_timeout", [60, 180])
@pytest.mark.parametrize("exploration_times_out", [False, True])
def test_deadline_finish_and_repair_receive_full_http_allowances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_timeout: int,
    exploration_times_out: bool,
) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = {"value": 0.0}
    finishes: list[float] = []
    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def fake_urlopen(
        request: urllib.request.Request, timeout: float, total_timeout: float | None = None
    ) -> io.BytesIO:
        payload = json.loads(request.data)
        if "tools" in payload:
            now["value"] += timeout
            if exploration_times_out and now["value"] >= 732 - 2 * request_timeout:
                raise TimeoutError("exploration allowance exhausted")
            return io.BytesIO(json.dumps(_tool_reply()).encode())
        finishes.append(timeout)
        assert timeout == request_timeout
        if len(finishes) == 1:
            now["value"] += timeout
            raise TimeoutError("finish stalled")
        now["value"] += timeout * 0.75
        return io.BytesIO(json.dumps(_findings_reply()).encode())

    monkeypatch.setattr(harness, "bounded_urlopen", fake_urlopen)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        timeout=request_timeout,
        lane_timeout=732,
    )

    assert result.ok
    assert result.salvaged
    assert finishes == [request_timeout, request_timeout]
    assert now["value"] < 732


def test_http_retries_cannot_cross_lane_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    now = {"value": 10.0}
    observed_timeouts: list[float] = []

    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        observed_timeouts.append(timeout)
        now["value"] += timeout
        raise TimeoutError("provider stalled")

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    sleeps: list[float] = []
    with pytest.raises(LaneError, match="request budget exhausted"):
        harness.openrouter_chat(
            "sk-test",
            {"model": "m"},
            timeout=180,
            sleep=sleeps.append,
            deadline=15.0,
        )
    assert observed_timeouts == [5.0]
    assert sleeps == []


@pytest.mark.parametrize("failure", ["timeout", "url", "connection", "http"])
def test_failed_lane_retains_transport_evidence_without_progress(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from or_pr_review import harness

    now = {"value": 0.0}
    monkeypatch.setattr(harness.time, "monotonic", lambda: now["value"])

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        now["value"] += timeout
        if failure == "timeout":
            raise TimeoutError("private transport details")
        if failure == "url":
            raise urllib.error.URLError("private transport details")
        if failure == "connection":
            raise ConnectionError("private transport details")
        raise _http_error(503, body=b"upstream unavailable")

    monkeypatch.setattr(harness, "bounded_urlopen", fake_urlopen)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        max_tool_turns=0,
        lane_timeout=5,
        workspace=None,
    )

    assert not result.ok
    error = result.to_dict()["error"]
    assert "budget exhausted" in error
    expected = {
        "timeout": "transport_timeouts=1",
        "url": "connection_errors=1",
        "connection": "connection_errors=1",
    }
    if failure == "http":
        assert "last OpenRouter HTTP 503" in error
        assert error.count("503") == 1
        assert "lane transport history" not in error
    else:
        assert f"lane transport history (includes recovered errors): {expected[failure]}" in error
    assert "private transport details" not in error
    assert "sk-test" not in error
    assert result.cost_usd is None
    assert not result.cost_complete


def test_transport_history_does_not_replace_later_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    calls = 0

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> io.BytesIO:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(429)
        if calls == 2:
            raise TimeoutError("recovered timeout")
        return io.BytesIO(b'{"choices":[{"message":{"content":"invalid JSON"}}]}')

    monkeypatch.setattr(harness, "bounded_urlopen", fake_urlopen)
    monkeypatch.setattr(harness, "_sleep_before_retry", lambda *_args, **_kwargs: None)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
    )

    assert not result.ok
    assert "JSON" in result.error
    assert (
        "lane transport history (includes recovered errors): transport_timeouts=1" in result.error
    )
    assert "429" not in result.error
    assert result.cost_usd is None


def test_repository_tool_timeout_is_killed_and_returned_as_observation(
    tmp_path: Path,
) -> None:
    from or_pr_review import harness

    (tmp_path / "pathological.txt").write_text("a" * 30_000 + "!\n", encoding="utf-8")
    call = {
        "id": "tool-1",
        "function": {
            "name": "grep",
            "arguments": '{"pattern": "(a+)+$", "path": "pathological.txt"}',
        },
    }
    started = harness.time.monotonic()
    observation = harness._run_one_tool(tmp_path, call, deadline=started + 0.25)
    assert observation["tool_call_id"] == "tool-1"
    assert "exceeded its" in observation["content"]
    assert "deadline" in observation["content"]
    assert harness.time.monotonic() - started < 5


def test_repository_tool_rejects_invalid_arguments_without_starting_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid arguments must not start the tool worker")

    monkeypatch.setattr(harness.subprocess, "run", unexpected_run)
    observation = harness._run_one_tool(
        tmp_path,
        {
            "id": "tool-invalid",
            "function": {"name": "read_file", "arguments": "[1, 2]"},
        },
    )

    assert observation == {
        "role": "tool",
        "tool_call_id": "tool-invalid",
        "content": "error: invalid tool arguments: arguments must be an object",
    }


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (7, "", "worker failed", "repository tool process exited 7: worker failed"),
        (0, "not-json", "", "repository tool returned invalid JSON"),
    ],
)
def test_repository_tool_reports_worker_protocol_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
    expected: str,
) -> None:
    from or_pr_review import harness

    completed = harness.subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )
    monkeypatch.setattr(harness.subprocess, "run", lambda *_args, **_kwargs: completed)
    observation = harness._run_one_tool(
        tmp_path,
        {
            "id": "tool-worker",
            "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
        },
    )

    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "tool-worker"
    assert expected in observation["content"]


def test_repository_tool_worker_uses_safe_path_against_checkout_shadowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    untrusted = tmp_path / "untrusted"
    shadow = untrusted / "or_pr_review"
    shadow.mkdir(parents=True)
    marker = tmp_path / "shadow-ran"
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "tool_worker.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('owned')\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("trusted\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    monkeypatch.chdir(untrusted)

    observation = harness._run_one_tool(
        workspace,
        {
            "id": "tool-safe-path",
            "function": {"name": "read_file", "arguments": '{"path":"safe.txt"}'},
        },
    )

    assert observation["content"].replace("\r\n", "\n") == "trusted\n"
    assert not marker.exists()


def test_salvage_shrinks_old_observations_on_context_overflow(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x" * 5000 + "\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) <= 3:
            return _tool_reply()
        if len(payloads) == 4:
            raise LaneError("OpenRouter HTTP 400: maximum context length exceeded")
        tool_texts = [
            message["content"] for message in payload["messages"] if message.get("role") == "tool"
        ]
        assert any("[observation truncated" in text for text in tool_texts)
        assert all("[observation truncated" not in text for text in tool_texts[-2:])
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=10,
    )
    assert result.ok
    assert result.salvaged is True
    assert len(payloads) == 5


def test_failed_lane_still_reports_usage_and_stats(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            reply = _tool_reply()
            reply["usage"] = {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cost": 0.001,
                "prompt_tokens_details": {"cached_tokens": 5},
            }
            return reply
        raise LaneError("OpenRouter HTTP 500: down")

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert not result.ok
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    assert result.cached_tokens == 5
    assert result.known_cost_usd == 0.001
    assert result.cost_usd is None
    assert result.cost_complete is False
    assert result.salvaged is True
    assert result.requests == 3
    assert result.tool_rounds == 1


def test_observation_budget_withdraws_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    monkeypatch.setattr(harness, "MAX_OBSERVATION_BYTES", 100)
    (tmp_path / "a.py").write_text("x" * 200 + "\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        max_tool_turns=50,
    )
    assert result.ok
    assert len(payloads) == 2
    assert "tools" not in payloads[1]
    assert payloads[1]["messages"][-1] == {"role": "user", "content": BUDGET_EXHAUSTED_NOTICE}


def test_initial_lane_enforces_coverage_with_schema_retry(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        if len(payloads) == 2:
            return _findings_reply()  # missing coverage → finalize retry
        schema = payload["response_format"]["json_schema"]["schema"]
        assert "coverage" in schema["required"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings": [], "coverage": [{"path": "a.py", "findings": 0}]}'
                        )
                    }
                }
            ]
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
        expect_coverage=True,
        expected_paths={"a.py"},
    )
    assert result.ok
    assert result.coverage == [("a.py", 0)]
    assert len(payloads) == 3
    assert "coverage is missing" in payloads[2]["messages"][-1]["content"]


def test_initial_lane_fails_open_when_coverage_misses_a_file(tmp_path: Path) -> None:
    def chat(_payload: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings": [], "coverage": [{"path": "a.py", "findings": 0}]}'
                        )
                    }
                }
            ]
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
        expect_coverage=True,
        expected_paths={"a.py", "b.py"},
    )
    assert not result.ok
    assert "does not account" in (result.error or "")


def test_in_body_error_reaches_salvage(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _tool_reply()
        if len(payloads) == 2:
            # HTTP 200 whose body carries an error object — must salvage,
            # not discard the gathered evidence.
            return {"error": {"message": "provider exploded upstream"}}
        assert "response_format" in payload
        assert "tools" not in payload
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        chat=chat,
    )
    assert result.ok
    assert result.salvaged is True
    assert len(payloads) == 3


def test_in_body_schema_rejection_downgrades_schema() -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return {"error": {"message": "response_format json_schema is not supported"}}
        assert "response_format" not in payload
        return _findings_reply()

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
    )
    assert result.ok
    assert len(payloads) == 2


def test_verify_lane_requires_complete_resolutions() -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return {"choices": [{"message": {"content": '{"findings": [], "resolutions": []}'}}]}
        assert "missing entries" in payload["messages"][-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"findings": [], "resolutions": '
                            '[{"id": "r1-1", "status": "fixed", "note": ""}]}'
                        )
                    }
                }
            ]
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        chat=chat,
        expect_resolutions=True,
        expected_resolution_ids={"r1-1"},
    )
    assert result.ok
    assert result.resolutions[0].id == "r1-1"
    assert len(payloads) == 2


def test_lane_artifact_roundtrip_with_stats_fields() -> None:
    from or_pr_review.schema import SCHEMA_VERSION, LaneResult, parse_lane_artifact

    lane = LaneResult(
        schema_version=SCHEMA_VERSION,
        ok=True,
        model="x-ai/grok-4.6",
        findings=[],
        error=None,
        elapsed_ms=12,
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=80,
        cost_usd=0.0123,
        requests=5,
        tool_rounds=3,
        retries=1,
        salvaged=True,
        head_sha="a" * 40,
    )
    parsed = parse_lane_artifact(lane.to_dict())
    assert parsed.cached_tokens == 80
    assert parsed.cost_usd == 0.0123
    assert parsed.requests == 5
    assert parsed.tool_rounds == 3
    assert parsed.retries == 1
    assert parsed.salvaged is True
    assert parsed.head_sha == "a" * 40


def test_run_lane_sums_byok_upstream_cost_and_requests_usage(tmp_path: Path) -> None:
    payloads: list[dict] = []
    progress: list[dict[str, int | float | str | bool]] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        return {
            "choices": [{"message": {"content": '{"findings": []}'}}],
            # BYOK shape: OpenRouter credits are 0 (any positive cost is
            # the BYOK fee) and the provider-billed spend arrives in
            # cost_details.upstream_inference_cost.
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "cost": 0.002,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": 0.009},
            },
        }

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        chat=chat,
        progress=progress.append,
    )
    assert result.ok
    assert result.cost_usd == pytest.approx(0.011)  # 0.009 upstream + 0.002 BYOK fee
    assert payloads[0]["usage"] == {"include": True}
    assert len(progress) == 2
    pre_send = progress[0]
    post_response = progress[-1]
    assert pre_send["elapsed_ms"] >= 0
    assert pre_send["attempted_requests"] == 1
    assert pre_send["cost_observed_responses"] == 0
    assert pre_send["cost_complete"] is False
    assert "cost_usd" not in pre_send
    assert {key: value for key, value in post_response.items() if key != "elapsed_ms"} == {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "known_cost_usd": pytest.approx(0.011),
        "attempted_requests": 1,
        "cost_observed_responses": 1,
        "cost_complete": True,
        "cost_usd": pytest.approx(0.011),
        "requests": 1,
    }
    # Round-trips through the lane artifact.
    from or_pr_review.schema import parse_lane_artifact

    assert parse_lane_artifact(result.to_dict()).cost_usd == pytest.approx(0.011)


def test_run_lane_progress_records_current_tool_round(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    responses = iter(
        [
            _tool_reply(),
            {
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.001},
            },
        ]
    )
    progress: list[dict[str, int | float | str]] = []

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        chat=lambda _payload: next(responses),
        progress=progress.append,
    )

    assert result.ok
    assert any(snapshot.get("tool_rounds") == 1 for snapshot in progress)


def test_run_lane_progress_failure_does_not_abort_paid_work(tmp_path: Path) -> None:
    def broken_progress(_snapshot: dict[str, int | float | str]) -> None:
        raise OSError("disk full")

    result = run_lane(
        model="x-ai/grok-4.6",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        chat=lambda _payload: {
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.001},
        },
        progress=broken_progress,
    )

    assert result.ok
    assert result.cost_usd == pytest.approx(0.001)


def test_response_spend_policy() -> None:
    from or_pr_review.harness import response_spend

    # Non-BYOK mirrors the charge into cost_details: count it once.
    assert response_spend(
        {"cost": 0.011, "is_byok": False, "cost_details": {"upstream_inference_cost": 0.011}}
    ) == pytest.approx(0.011)
    # BYOK: upstream is the spend; positive cost is the BYOK fee on top.
    assert response_spend(
        {"cost": 0.0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.009}}
    ) == pytest.approx(0.009)
    # BYOK with no upstream figure is unknown spend, not a $0 observation.
    assert response_spend({"cost": 0.0, "is_byok": True}) is None
    # Free non-BYOK routes legitimately cost $0.
    assert response_spend({"cost": 0}) == 0.0


def test_run_lane_forwards_requested_service_tier_through_tool_and_finish(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    payloads: list[dict] = []
    responses = iter(
        [
            {
                **_tool_reply(),
                "service_tier": "flex",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.001},
            },
            {
                "service_tier": "priority",
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.002},
            },
        ]
    )

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        service_tier="flex",
        chat=lambda payload: payloads.append(payload) or next(responses),
    )

    assert result.ok
    assert [payload["service_tier"] for payload in payloads] == ["flex", "flex"]
    assert result.requested_service_tier == "flex"
    assert result.served_service_tiers == ["flex", "priority"]
    assert result.service_tier_observed_responses == 2
    assert result.service_tier_complete is True
    assert result.service_tier_confirmed is False


def test_run_lane_forwards_service_tier_during_schema_repair(tmp_path: Path) -> None:
    payloads: list[dict] = []

    def chat(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            raise LaneError("response_format rejected")
        return {
            "service_tier": "flex",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"cost": 0},
        }

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        service_tier="flex",
        chat=chat,
    )

    assert result.ok
    assert len(payloads) == 2
    assert all(payload["service_tier"] == "flex" for payload in payloads)
    assert result.service_tier_observed_responses == 1
    assert result.service_tier_complete is False


def test_run_lane_forwards_service_tier_during_finalization_repair(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    payloads: list[dict] = []
    responses = iter(
        [
            {**_tool_reply(), "service_tier": "flex", "usage": {"cost": 0}},
            {
                "service_tier": "flex",
                "choices": [{"message": {"content": "not valid findings JSON"}}],
                "usage": {"cost": 0},
            },
            {
                "service_tier": "flex",
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"cost": 0},
            },
        ]
    )

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        service_tier="flex",
        chat=lambda payload: payloads.append(payload) or next(responses),
    )

    assert result.ok
    assert len(payloads) == 3
    assert all(payload["service_tier"] == "flex" for payload in payloads)
    assert "response_format" in payloads[1]
    assert "response_format" in payloads[2]
    assert result.service_tier_confirmed is True


def test_run_lane_keeps_missing_and_mixed_served_tier_telemetry_distinct(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    responses = iter(
        [
            {
                **_tool_reply(),
                "service_tier": "flex",
                "usage": {"cost": 0.001},
            },
            {
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"cost": 0.001},
            },
        ]
    )
    missing = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        service_tier="flex",
        chat=lambda _payload: next(responses),
    )

    assert missing.served_service_tiers == ["flex"]
    assert missing.service_tier_observed_responses == 1
    assert missing.service_tier_complete is False
    assert missing.service_tier_confirmed is False

    mixed_responses = iter(
        [
            {**_tool_reply(), "service_tier": "default", "usage": {"cost": 0.001}},
            {
                "service_tier": "priority",
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"cost": 0.001},
            },
        ]
    )
    mixed = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        service_tier="flex",
        chat=lambda _payload: next(mixed_responses),
    )

    assert mixed.served_service_tiers == ["default", "priority"]
    assert mixed.service_tier_complete is True
    assert mixed.requested_service_tier == "flex"
    assert mixed.service_tier_confirmed is False


def test_run_lane_keeps_partial_cost_but_withholds_incomplete_total(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    responses = iter(
        [
            {**_tool_reply(), "usage": {"cost": 0.001}},
            {"choices": [{"message": {"content": '{"findings": []}'}}], "usage": {}},
        ]
    )

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        chat=lambda _payload: next(responses),
    )

    assert result.ok
    assert result.known_cost_usd == pytest.approx(0.001)
    assert result.cost_usd is None
    assert result.cost_observed_responses == 1
    assert result.cost_complete is False


def test_run_lane_failed_call_after_known_cost_keeps_total_unknown(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    calls = 0

    def chat(_payload: dict) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {**_tool_reply(), "usage": {"cost": 0.001}}
        raise LaneError("transport failed")

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=1,
        chat=chat,
    )

    assert not result.ok
    assert result.known_cost_usd == pytest.approx(0.001)
    assert result.cost_usd is None
    assert result.cost_complete is False


def test_run_lane_progress_marks_unaccounted_request_before_interruption(tmp_path: Path) -> None:
    progress: list[dict[str, int | float | str | bool]] = []

    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        chat=lambda _payload: (_ for _ in ()).throw(LaneError("interrupted")),
        progress=progress.append,
    )

    assert not result.ok
    assert progress[-1]["requests"] == 1
    assert progress[-1]["attempted_requests"] == 1
    assert progress[-1]["cost_observed_responses"] == 0
    assert progress[-1]["cost_complete"] is False
    assert "cost_usd" not in progress[-1]


def test_openrouter_chat_internal_retry_counts_one_logical_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from or_pr_review import harness

    calls = 0
    success_body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001},
        }
    ).encode()

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(429, body=b'{"error":{"message":"rate limited"}}')

        class _SuccessResponse(_FakeResponse):
            def read(self) -> bytes:
                return success_body

        return _SuccessResponse()

    stats: dict[str, int] = {}
    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    response = harness.openrouter_chat(
        "sk-test",
        {"model": "example/model"},
        timeout=5,
        sleep=lambda _delay: None,
        stats=stats,
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert calls == 2
    assert stats["attempted_requests"] == 1
    assert stats["retries"] == 1


def test_run_lane_openrouter_retry_keeps_total_cost_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    success_body = json.dumps(
        {
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001},
        }
    ).encode()

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(429, body=b'{"error":{"message":"rate limited"}}')

        class _SuccessResponse(_FakeResponse):
            def read(self) -> bytes:
                return success_body

        return _SuccessResponse()

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
    )

    assert result.ok
    assert calls == 2
    assert result.attempted_requests == 2
    assert result.cost_observed_responses == 1
    assert result.known_cost_usd == pytest.approx(0.001)
    assert result.cost_complete is False
    assert result.cost_usd is None


def test_run_lane_openrouter_retry_progress_counts_inflight_before_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    progress: list[dict[str, int | float | str | bool]] = []
    urlopen_calls = 0
    tool_body = json.dumps(
        {
            **_tool_reply(),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001},
        }
    ).encode()

    def fake_urlopen(
        _request: object, timeout: float, total_timeout: float | None = None
    ) -> _FakeResponse:
        nonlocal urlopen_calls
        urlopen_calls += 1
        if urlopen_calls == 1:

            class _ToolResponse(_FakeResponse):
                def read(self) -> bytes:
                    return tool_body

            return _ToolResponse()
        if urlopen_calls == 2:
            raise _http_error(429, body=b'{"error":{"message":"rate limited"}}')
        checkpoint = progress[-1]
        assert checkpoint["attempted_requests"] == 3
        assert checkpoint["known_cost_usd"] == pytest.approx(0.001)
        assert checkpoint["cost_observed_responses"] == 1
        assert checkpoint["cost_complete"] is False
        assert "cost_usd" not in checkpoint
        raise KeyboardInterrupt

    monkeypatch.setattr("or_pr_review.harness.bounded_urlopen", fake_urlopen)
    with pytest.raises(KeyboardInterrupt):
        run_lane(
            model="example/model",
            messages=[{"role": "user", "content": "review"}],
            api_key="sk-test",
            workspace=tmp_path,
            max_tool_turns=1,
            progress=progress.append,
        )

    assert urlopen_calls == 3
    assert progress[-1]["attempted_requests"] == 3
    assert progress[-1]["known_cost_usd"] == pytest.approx(0.001)
    assert progress[-1]["cost_observed_responses"] == 1
    assert progress[-1]["cost_complete"] is False
    assert "cost_usd" not in progress[-1]


def test_run_lane_records_served_tier_without_request(tmp_path: Path) -> None:
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        max_tool_turns=0,
        chat=lambda _payload: {
            "service_tier": "priority",
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"cost": 0.001},
        },
    )

    assert result.ok
    assert result.requested_service_tier is None
    assert result.served_service_tiers == ["priority"]
    assert result.service_tier_observed_responses == 1
    assert result.service_tier_complete is True
    assert result.service_tier_confirmed is False


def test_live_finalize_can_use_tail_beyond_one_idle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from or_pr_review import harness

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    now = [0.0]
    attempts = []
    monkeypatch.setattr(harness.time, "monotonic", lambda: now[0])

    def transport(request, *, timeout, total_timeout=None):
        payload = json.loads(request.data)
        attempts.append((timeout, total_timeout, "tools" in payload))
        if "tools" in payload:
            # Exploration reaches its cutoff with evidence already collected.
            now[0] = 702.0
            return io.BytesIO(json.dumps(_tool_reply()).encode())
        assert timeout == 180
        assert total_timeout == 360
        # Active bytes keep this response alive. No speculative repair gets
        # to interrupt its useful 240-second generation at the old 180s limit.
        now[0] += 240
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "findings": [
                                            {
                                                "title": "Preserved finding",
                                                "body": "Evidence from a.py.",
                                                "severity": "risk",
                                                "file": "a.py",
                                                "line": 1,
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(harness, "bounded_urlopen", transport)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=tmp_path,
        timeout=180,
        lane_timeout=1062,
    )
    assert result.ok
    assert result.salvaged
    assert not result.retries
    assert [finding.title for finding in result.findings] == ["Preserved finding"]
    assert len(attempts) == 2
    assert now[0] == 942


@pytest.mark.parametrize("kind", ["idle_timeouts", "elapsed_deadlines", "connect_timeouts"])
def test_failed_lane_distinguishes_idle_from_elapsed_timeout(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from or_pr_review import harness
    from or_pr_review.http_transport import HttpConnectTimeout, HttpElapsedTimeout, HttpIdleTimeout

    now = [0.0]
    monkeypatch.setattr(harness.time, "monotonic", lambda: now[0])

    def transport(request, *, timeout, total_timeout=None):
        now[0] += total_timeout
        error_type = {
            "idle_timeouts": HttpIdleTimeout,
            "elapsed_deadlines": HttpElapsedTimeout,
            "connect_timeouts": HttpConnectTimeout,
        }[kind]
        raise error_type("private transport detail")

    monkeypatch.setattr(harness, "bounded_urlopen", transport)
    result = run_lane(
        model="example/model",
        messages=[{"role": "user", "content": "review"}],
        api_key="sk-test",
        workspace=None,
        max_tool_turns=0,
        lane_timeout=5,
    )
    assert not result.ok
    assert "transport_timeouts=1" in result.error
    assert f"{kind}=1" in result.error
    assert "private transport detail" not in result.error
