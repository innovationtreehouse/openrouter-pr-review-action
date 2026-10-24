"""Pick mode: decision state, minimal compaction, answer parsing and fallback."""

import copy
import json
from pathlib import Path

import pytest

from or_pr_review import cli, pick
from or_pr_review.brief import BriefItem, ReviewBrief
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.loop import LoopState
from or_pr_review.schema import Finding, LaneResult

CAP = 48_000
FIXTURES = Path(__file__).resolve().parents[1] / "bench" / "judge-fixtures"


def lane(model, findings):
    return LaneResult(
        1,
        True,
        model,
        findings=[
            Finding(f["title"], f["body"], f["severity"], f["file"], f["line"], model)
            for f in findings
        ],
        head_sha="b" * 40,
        tool_rounds=3,
        elapsed_ms=4200,
    )


def two_lanes():
    data = json.loads((FIXTURES / "ab-duplicates-complements.json").read_text())
    return [lane(entry["model"], entry["findings"]) for entry in data["lanes"]]


def new_file_diff(path, lines, width=60):
    body = "".join(f"+line {n} {'x' * width}\n" for n in range(1, lines + 1))
    return (
        f"diff --git a/{path} b/{path}\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{lines} @@\n{body}"
    )


def context_diff(path, lines, changed_at):
    out = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"]
    out.append(f"@@ -1,{lines} +1,{lines} @@ def handler():")
    for n in range(1, lines + 1):
        out.append(f"+changed line {n}" if n == changed_at else f" context line {n} {'y' * 40}")
    return "\n".join(out) + "\n"


def collected(diff="", body=""):
    return CollectedReview(
        7,
        "Add retries",
        body,
        "b" * 40,
        "main",
        "feature",
        DiffPlan("full-pr", "full-pr", "c" * 40, "b" * 40, None),
        Truncation(diff, False, len(diff), len(diff), 300),
        "initial",
    )


def make_state(lanes, *, diff="", body="", bodies=(), comments=(), cap=CAP):
    items = pick.build_items(
        collected=collected(diff, body),
        loop=None,
        lanes=lanes,
        review_bodies=list(bodies),
        replies=[],
        comments=list(comments),
        setup={},
    )
    return pick.State(items, cap)


def section(state, name):
    return {i.key: copy.deepcopy(i.value) for i in state.items if i.section == name}


def brief_stub(log):
    def brief_fn(chunk):
        assert len(chunk.encode()) <= pick.CHUNK_BYTES
        log.append(len(chunk))
        return ReviewBrief(intent="Adds retries.", claims=(BriefItem("retries", chunk[:20]),))

    return brief_fn


def failing_brief(_chunk):
    raise AssertionError("no compaction call expected")


def test_state_that_fits_is_not_compacted():
    state = make_state(two_lanes(), diff=new_file_diff("src/jobs.py", 50), body="Fixes #1.")
    assert pick.fit_state(state, two_lanes(), failing_brief) == 0
    rendered = state.render()
    assert rendered["compaction"]["applied"] == []
    assert list(rendered)[:3] == ["compaction", "lanes", "change"]
    assert rendered["lanes"]["lane_1"]["findings"][0]["id"] == "lane_1.F1"
    assert rendered["lanes"]["lane_1"]["tool_rounds"] == 3


def test_state_size_is_measured_per_item_and_matches_the_payload():
    state = make_state(two_lanes(), diff=new_file_diff("src/jobs.py", 300), body="b" * 2000)
    assert abs(len(json.dumps(state.render())) - state.total()) < 200


def test_just_over_cap_gets_only_the_smallest_reduction():
    comment = "<!-- " + "m" * 3000 + " -->"
    body = "Intent.\n" + comment
    review = "## Old review\n<!-- " + "n" * 1000 + " -->"
    state = make_state(two_lanes(), body=body, bodies=[review])
    state.cap = state.total() - 100
    before_prior = section(state, "prior_rounds")
    assert pick.fit_state(state, two_lanes(), failing_brief) == 0
    assert state.fits()
    assert {k: v[0] for k, v in state.steps.items()} == {("strip html comments", "pr"): 1}
    assert section(state, "prior_rounds") == before_prior


def test_tier_a_alone_when_pr_text_suffices():
    lanes = two_lanes()
    state = make_state(lanes, diff=new_file_diff("src/jobs.py", 100), body="word " * 12_000)
    untouched = {name: section(state, name) for name in ("lanes", "change", "prior_rounds")}
    log = []
    assert pick.fit_state(state, lanes, brief_stub(log)) == 1
    assert {step for step, _ in state.steps} == {"summarize pr text (model)"}
    assert {name: section(state, name) for name in untouched} == untouched
    assert section(state, "pr")["body"]["original_bytes"] > 60_000


def test_tiers_a_and_b_when_the_diff_is_also_large():
    lanes = two_lanes()
    diff = new_file_diff("src/jobs.py", 900) + new_file_diff("src/input.py", 10)
    state = make_state(lanes, diff=diff, body="word " * 12_000)
    before_lanes = section(state, "lanes")
    pick.fit_state(state, lanes, brief_stub([]))
    steps = {step for step, _ in state.steps}
    assert steps == {"summarize pr text (model)", "summarize diff per file"}
    assert state.steps[("summarize diff per file", "change")][0] == 1  # largest file only
    assert "diff" in section(state, "change")["src/input.py"]
    assert section(state, "lanes") == before_lanes


def test_cited_lines_survive_diff_compaction_verbatim():
    lanes = two_lanes()  # cites src/jobs.py:42 and :44
    diff = context_diff("src/jobs.py", 2000, changed_at=42)
    state = make_state(lanes, diff=diff)
    state.cap = 20_000
    pick.fit_state(state, lanes, failing_brief)
    text = json.dumps(section(state, "change")["src/jobs.py"])
    assert "+changed line 42" in text
    assert f"context line 44 {'y' * 40}" in text
    assert "def handler():" in text
    assert ("collapse diff context", "change") in state.steps


def test_collapsed_then_summarized_file_keeps_cited_line_numbers():
    diff = context_diff("a.py", 400, changed_at=300)
    collapsed = pick.collapse_context(diff, {300})
    summary = pick.summarize_file(collapsed, {300, 302})
    assert "L300: +changed line 300" in summary
    assert f"L302:  context line 302 {'y' * 40}" in summary


# Calls = body chunks (+1 reduce when the briefs are still over cap/8) + comment chunks.
@pytest.mark.parametrize("scale,expected_calls", [(1, 1), (10, 7), (100, 58)])
def test_compaction_converges_at_any_scale(scale, expected_calls):
    lanes = two_lanes()
    files = "".join(new_file_diff(f"gen/f{n}.py", 60) for n in range(scale * 10))
    body = "".join(f"Paragraph {n}: {'z' * 90}\n" for n in range(scale * CAP // 100))
    comments = [(f"user{n}", "please check " + "q" * 400) for n in range(scale * 5)]
    state = make_state(lanes, diff=files, body=body, comments=comments)
    before = section(state, "lanes")
    log = []
    calls = pick.fit_state(state, lanes, brief_stub(log))
    assert state.fits() and len(json.dumps(state.render())) <= CAP
    assert calls == len(log) == expected_calls
    assert section(state, "lanes") == before


def test_single_five_megabyte_file_converges_without_model_calls():
    lanes = two_lanes()
    diff = new_file_diff("src/jobs.py", 80_000, width=50)
    assert len(diff) > 5_000_000
    state = make_state(lanes, diff=diff)
    before = section(state, "lanes")
    assert pick.fit_state(state, lanes, failing_brief) == 0
    assert state.fits()
    assert f"L42: +line 42 {'x' * 50}" in section(state, "change")["src/jobs.py"]["summary"]
    assert section(state, "lanes") == before


def test_terminal_step_keeps_finding_identity_and_is_bounded():
    many = [
        dict(title=f"Finding {n}", body="e" * 2000, severity="bug", file="a.py", line=n)
        for n in range(1, 40)
    ]
    lanes = [lane("x/a", many), lane("x/b", many)]
    state = make_state(lanes, diff=new_file_diff("a.py", 100), cap=12_000)
    pick.fit_state(state, lanes, failing_brief)
    assert state.terminal and state.fits()
    findings = section(state, "lanes")["lane_2"]["findings"]
    assert len(findings) == 39
    assert all(set(f) == {"id", "title", "severity", "location"} for f in findings)
    assert {f["title"] for f in findings} == {f"Finding {n}" for n in range(1, 40)}
    assert [f["id"] for f in findings] == [f"lane_2.F{n}" for n in range(1, 40)]


def test_compaction_model_failure_raises_for_fallback():
    state = make_state(two_lanes(), body="word " * 12_000)
    with pytest.raises(pick.PickError):
        pick.fit_state(state, two_lanes(), lambda _chunk: None)


def answers(probs=None, choice=None, safe=None):
    out = {"better_lane": {"choice": choice, "probabilities": probs or {}}}
    for lid, p in (safe or {}).items():
        out[f"{lid}_safe"] = {"noul": p}
    return {"answers": out}


def test_parse_pick_takes_the_most_probable_known_lane():
    ids = ["lane_1", "lane_2"]
    got = pick.parse_pick(answers({"lane_1": 0.3, "lane_2": 0.7}, "lane_1", {"lane_2": 0.9}), ids)
    assert got == ("lane_2", 0.7, {"lane_2": 0.9})
    assert pick.parse_pick(answers(choice="lane_1"), ids) == ("lane_1", None, {})


@pytest.mark.parametrize(
    "response",
    [answers({"lane_9": 1.0}, "lane_9"), {"answers": {}}, {"error": "x"}, None],
)
def test_parse_pick_rejects_unknown_or_missing_answers(response):
    with pytest.raises(pick.PickError):
        pick.parse_pick(response, ["lane_1", "lane_2"])


class GitHub:
    def __init__(self):
        self.posted = []

    def pr_view(self, _number):
        return {"headRefOid": "b" * 40}

    def create_review(self, _number, body, _sha, **kwargs):
        self.posted.append((body, kwargs.get("comments") or []))
        return {"html_url": "https://example.test/r"}

    def list_bot_review_bodies(self, *_a):
        return []

    def list_finding_replies(self, *_a, **_k):
        return []

    def list_recent_issue_comments(self, *_a):
        return [("dev", "Please look at the retry path.")]


ENV = {
    "MODELS": "synthetic/reviewer-a,synthetic/reviewer-b",
    "JUDGE_MODE": "pick",
    "OPENROUTER_API_KEY": "sk-test",
    "FAIL_ON": "never",
    "GITHUB_REPOSITORY": "o/r",
}


def resolve(monkeypatch, env, body=""):
    monkeypatch.setattr(cli, "_github", lambda _env: GitHub())
    lanes = two_lanes()
    return cli._resolve_issues(
        env, [x.model for x in lanes], lanes, lanes, collected(body=body), LoopState("initial", 1)
    )


def test_decision_failure_falls_back_to_the_labeled_union(monkeypatch):
    def boom(**_kwargs):
        raise OSError("HTTP transport worker failed")

    monkeypatch.setattr(pick, "ask", boom)
    outcome = resolve(monkeypatch, ENV)
    assert outcome.picked is None
    assert outcome.note.startswith("`baseten/mammoth-1:free` (pick decision fallback")
    assert len(outcome.issues) == 3  # the union: both lanes' code findings


def test_compaction_failure_falls_back_to_the_labeled_union(monkeypatch):
    monkeypatch.setattr(pick, "ask", lambda **_k: pytest.fail("decision must not run"))
    monkeypatch.setattr(cli, "build_brief", lambda *_a, **_k: None)
    env = {**ENV, "COMPACTION_MODEL": "x/compact", "DECISION_STATE_MAX": "8000"}
    outcome = resolve(monkeypatch, env, body="word " * 12_000)
    assert "pick compaction fallback: deterministic union" in outcome.note
    no_model = resolve(monkeypatch, {**ENV, "DECISION_STATE_MAX": "8000"}, body="w " * 9000)
    assert "pick compaction fallback" in no_model.note


def test_pick_mode_posts_the_chosen_lane_verbatim(monkeypatch, tmp_path):
    github = GitHub()
    sent = {}

    def ask(**kwargs):
        sent.update(kwargs)
        return answers({"lane_1": 0.8, "lane_2": 0.2}, "lane_1", {"lane_1": 0.9, "lane_2": 0.4})

    monkeypatch.setattr(pick, "ask", ask)
    monkeypatch.setattr(cli, "_github", lambda _env: github)
    monkeypatch.setattr(cli, "_set_output", lambda *_a: None)
    monkeypatch.setattr(cli, "_maybe_status", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_checkout_has_commit", lambda *a: False)
    lanes = two_lanes()
    diff = context_diff("src/jobs.py", 60, changed_at=42) + context_diff("src/input.py", 30, 18)
    env = {**ENV, "SOURCE_WORKSPACE": str(tmp_path)}
    assert cli._finish(env, lanes, collected(diff=diff), LoopState("initial", 1)) == 0
    body, comments = github.posted[0]
    assert "**Verdict:** `issues`" in body  # lane_2's checkout diagnostic is not posted
    assert "**Judge:** Picked by `baseten/mammoth-1:free` from 2 lanes: " in body
    assert "`synthetic/reviewer-a` (lane_1; p=0.80; safe-to-post lane_1=0.90" in body
    for finding in lanes[0].findings:
        assert finding.title in body and finding.body in body
    assert lanes[1].findings[0].title not in body
    assert {c["line"] for c in comments} == {42, 18}
    assert sent["state"]["pr"]["human_comments"] == [
        {"author": "dev", "body": "Please look at the retry path."}
    ]
    assert set(sent["qs"]) == {"better_lane", "lane_1_safe", "lane_2_safe"}


def test_invalid_judge_mode_is_rejected():
    with pytest.raises(cli.ActionError):
        cli._judge_mode({"JUDGE_MODE": "vote"})
    with pytest.raises(cli.ActionError):
        cli._decision_settings({"DECISION_ENDPOINT": "../chat"})
