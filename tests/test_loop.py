from __future__ import annotations

import base64
import json

import pytest

from or_pr_review.errors import ActionError
from or_pr_review.loop import (
    LEDGER_PREFIX,
    LEDGER_V1_EVIDENCE_COMPAT_LIMIT,
    LEDGER_V1_TITLE_COMPAT_LIMIT,
    LEDGER_VERSION,
    Ledger,
    LedgerFinding,
    LoopState,
    apply_round,
    apply_severity_floor,
    decide_loop_state,
    encode_ledger,
    extract_ledger,
    latest_ledger,
    merge_resolutions,
    round_report,
)
from or_pr_review.merge import MergedIssue
from or_pr_review.schema import Resolution

REPO = "o/r"


def _finding(
    ident: str = "r1-1",
    status: str = "open",
    severity: str = "bug",
    evidence: str = "the failure detail",
) -> LedgerFinding:
    return LedgerFinding(
        id=ident,
        severity=severity,
        file="src/app.py",
        line=3,
        title="Missing auth check",
        evidence=evidence,
        status=status,
        models=("x-ai/grok-4.6",),
    )


def _raw_ledger_marker(*, title: str, evidence: str) -> str:
    payload = {
        "lv": LEDGER_VERSION,
        "repo": REPO,
        "pr": 7,
        "sha": "",
        "gen": "",
        "round": 1,
        "findings": [
            {
                "id": "r1-1",
                "sev": "bug",
                "file": None,
                "line": None,
                "title": title,
                "ev": evidence,
                "st": "open",
                "m": [],
            }
        ],
    }
    token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"{LEDGER_PREFIX}{token} -->"


def test_ledger_roundtrip_and_binding() -> None:
    ledger = Ledger(round_number=2, findings=(_finding(),), reviewed_sha="a" * 40)
    marker = encode_ledger(ledger, repo=REPO, pr_number=7)
    assert marker.startswith(LEDGER_PREFIX)
    assert LEDGER_PREFIX == f"<!-- openrouter-review-ledger:v{LEDGER_VERSION}:"
    decoded = extract_ledger(marker, repo=REPO, pr_number=7)
    assert decoded is not None
    assert decoded.round_number == 2
    assert decoded.reviewed_sha == "a" * 40
    assert decoded.findings[0].evidence == "the failure detail"
    assert decoded.findings[0].models == ("x-ai/grok-4.6",)
    assert extract_ledger(marker, repo="other/repo", pr_number=7) is None
    assert extract_ledger(marker, repo=REPO, pr_number=8) is None


@pytest.mark.parametrize(
    ("field", "limit"),
    (
        ("title", LEDGER_V1_TITLE_COMPAT_LIMIT),
        ("evidence", LEDGER_V1_EVIDENCE_COMPAT_LIMIT),
    ),
)
def test_ledger_v1_compatibility_limits(field: str, limit: int) -> None:
    values = {"title": "t", "evidence": "e"}
    values[field] = "x" * limit
    at_limit = _raw_ledger_marker(**values)
    decoded = extract_ledger(at_limit, repo=REPO, pr_number=7)
    assert decoded is not None
    assert getattr(decoded.findings[0], field) == "x" * limit

    values[field] = "x" * (limit + 1)
    over_limit = _raw_ledger_marker(**values)
    assert extract_ledger(over_limit, repo=REPO, pr_number=7) is None


def test_latest_ledger_newest_marker_is_authoritative() -> None:
    valid = encode_ledger(
        Ledger(round_number=1, findings=(_finding(),), reviewed_sha="a" * 40),
        repo=REPO,
        pr_number=7,
    )
    junk = LEDGER_PREFIX + base64.b64encode(b'{"nope": 1}').decode("ascii") + " -->"
    assert latest_ledger([valid], repo=REPO, pr_number=7) is not None
    with pytest.raises(ActionError, match="reset"):
        latest_ledger([valid, junk], repo=REPO, pr_number=7)
    assert latest_ledger(["no marker here"], repo=REPO, pr_number=7) is None
    assert latest_ledger([], repo=REPO, pr_number=7) is None


def test_decide_loop_state_matrix() -> None:
    ledger = Ledger(round_number=3, findings=(), reviewed_sha="")
    assert decide_loop_state(review_mode="initial", event_action="synchronize", ledger=ledger) == (
        "initial",
        1,
    )
    assert decide_loop_state(review_mode="auto", event_action="synchronize", ledger=None) == (
        "initial",
        1,
    )
    assert decide_loop_state(review_mode="auto", event_action="synchronize", ledger=ledger) == (
        "verify",
        4,
    )
    assert decide_loop_state(review_mode="verify", event_action="", ledger=ledger) == (
        "verify",
        4,
    )
    assert decide_loop_state(review_mode="auto", event_action="opened", ledger=ledger) == (
        "verify",
        4,
    )


def test_apply_round_folds_resolutions_and_numbers_new_issues() -> None:
    state = LoopState(
        mode="verify",
        round_number=2,
        prior_findings=(
            _finding("r1-1"),
            _finding("r1-2", severity="risk"),
            _finding("r1-3", status="disputed"),
        ),
    )
    new_issue = MergedIssue("New race", "check-then-act", "risk", "db.py", 9, ["x-ai/grok-4.6"])
    resolutions = {"r1-1": Resolution(id="r1-1", status="fixed", note="added check")}
    outcome = apply_round(state, [new_issue], resolutions)
    statuses = {finding.id: finding.status for finding in outcome.ledger.findings}
    assert "r1-1" not in statuses  # fixed → dropped
    assert statuses["r1-2"] == "open"  # unaddressed carries forward
    assert statuses["r1-3"] == "disputed"
    assert outcome.issues[0].id == "r2-1"
    assert statuses["r2-1"] == "open"
    assert outcome.open_issue_count == 2
    assert outcome.open_bug_count == 0
    lines = "\n".join(outcome.resolution_lines)
    assert "✅" in lines and "r1-1" in lines
    assert "⏳" in lines and "r1-2" in lines
    report = round_report(state, outcome)
    assert report[0] == "### Round 2 resolution"
    assert round_report(LoopState(mode="initial", round_number=1), outcome) == []


def test_disputed_resolution_settles_a_finding() -> None:
    state = LoopState(mode="verify", round_number=3, prior_findings=(_finding("r1-1"),))
    resolutions = {"r1-1": Resolution(id="r1-1", status="disputed", note="by design")}
    outcome = apply_round(state, [], resolutions)
    assert outcome.ledger.findings[0].status == "disputed"
    assert outcome.open_issue_count == 0
    assert "🤝" in outcome.resolution_lines[0]


def test_followup_evidence_keeps_one_open_id_and_every_lane_body():
    state = LoopState(mode="verify", round_number=3, prior_findings=(_finding(),))
    issues = [
        MergedIssue(
            "Missing check in caller A",
            "Caller A still bypasses auth.",
            "risk",
            "src/app.py",
            10,
            ["model-a"],
            prior_finding_id="r1-1",
        ),
        MergedIssue(
            "Missing check in caller B",
            "Caller B still bypasses auth.",
            "bug",
            "src/app.py",
            20,
            ["model-b"],
            prior_finding_id="r1-1",
        ),
    ]
    outcome = apply_round(state, issues, {"r1-1": Resolution("r1-1", "not_fixed", "still open")})
    assert outcome.open_issue_count == 1
    assert outcome.open_bug_count == 1
    assert len(outcome.issues) == 1
    assert outcome.issues[0].id == "r1-1"
    assert "Caller A" in outcome.issues[0].body and "Caller B" in outcome.issues[0].body
    assert "the failure detail" in outcome.ledger.findings[0].evidence
    assert "Caller A" in outcome.ledger.findings[0].evidence
    assert "Caller B" in outcome.ledger.findings[0].evidence
    assert outcome.issues[0].title == outcome.ledger.findings[0].title
    assert outcome.issues[0].line == outcome.ledger.findings[0].line
    assert set(outcome.ledger.findings[0].models) == {"x-ai/grok-4.6", "model-a", "model-b"}


@pytest.mark.parametrize("status", ["fixed", "disputed"])
def test_link_cannot_hide_a_new_finding_behind_settled_status(status):
    state = LoopState(mode="verify", round_number=2, prior_findings=(_finding(),))
    issue = MergedIssue(
        "A new failure", "New evidence", "bug", "src/app.py", 3, prior_finding_id="r1-1"
    )
    outcome = apply_round(state, [issue], {"r1-1": Resolution("r1-1", status, "settled")})
    assert outcome.issues[0].id == "r2-1"
    assert outcome.open_issue_count == 1


@pytest.mark.parametrize("prior_id,path", [("r9-9", "src/app.py"), ("r1-1", "other.py")])
def test_unconfirmed_links_stay_independent(prior_id, path):
    state = LoopState(mode="verify", round_number=2, prior_findings=(_finding(),))
    issue = MergedIssue("Another failure", "Evidence", "bug", path, 3, prior_finding_id=prior_id)
    outcome = apply_round(state, [issue], {})
    assert outcome.issues[0].id == "r2-1"
    assert outcome.open_issue_count == 2


def test_unanchored_links_stay_independent():
    from dataclasses import replace

    state = LoopState(
        mode="verify", round_number=2, prior_findings=(replace(_finding(), file=None),)
    )
    issue = MergedIssue("Another failure", "Evidence", "bug", None, None, prior_finding_id="r1-1")
    outcome = apply_round(state, [issue], {})
    assert outcome.open_issue_count == 2
    assert outcome.issues[0].id == "r2-1"


def test_linked_evidence_over_budget_stays_visible_as_separate_finding():
    state = LoopState(mode="verify", round_number=2, prior_findings=(_finding(),))
    issue = MergedIssue(
        "Long evidence", "x" * 8000, "bug", "src/app.py", 3, prior_finding_id="r1-1"
    )
    outcome = apply_round(state, [issue, issue], {})
    assert [row.id for row in outcome.issues] == ["r2-1", "r2-2"]
    assert outcome.open_issue_count == 3
    assert outcome.ledger.findings[0].evidence == "the failure detail"


def test_apply_severity_floor_retires_nits_from_round_two() -> None:
    findings = (
        _finding("r1-1"),
        _finding("r1-2", severity="nit"),
        _finding("r1-3", severity="risk"),
        _finding("r1-4", severity="nit", status="disputed"),
    )
    carried, retired = apply_severity_floor(findings, 1)
    assert carried == findings
    assert retired == ()
    carried, retired = apply_severity_floor(findings, 2)
    assert [finding.id for finding in carried] == ["r1-1", "r1-3"]
    assert [finding.id for finding in retired] == ["r1-2", "r1-4"]


def test_round_report_states_severity_floor_retirement() -> None:
    # Build the state the way the CLI does — through the partition — so the
    # asserts pin that retirement actually strips nits out of loop state.
    findings = (
        _finding("r1-1"),
        _finding("r1-2", severity="nit"),
        _finding("r1-3", severity="nit"),
    )
    carried, retired = apply_severity_floor(findings, 2)
    state = LoopState(
        mode="verify",
        round_number=2,
        prior_findings=carried,
        retired_prior=retired,
    )
    resolutions = {"r1-1": Resolution(id="r1-1", status="fixed", note="done")}
    outcome = apply_round(state, [], resolutions)
    report = "\n".join(round_report(state, outcome))
    assert "2 nit finding(s) from earlier rounds retired" in report
    assert "severity floor" in report
    # Retired findings leave the ledger and the open counts entirely.
    assert all(finding.severity != "nit" for finding in outcome.ledger.findings)
    assert outcome.ledger.findings == ()
    assert outcome.open_issue_count == 0
    # No retirement, no line.
    bare = LoopState(mode="verify", round_number=2, prior_findings=(_finding("r1-1"),))
    assert "retired" not in "\n".join(round_report(bare, apply_round(bare, [], resolutions)))


def test_round_report_nits_only_backlog_reports_only_retirement() -> None:
    findings = (
        _finding("r1-1", severity="nit"),
        _finding("r1-2", severity="nit"),
    )
    carried, retired = apply_severity_floor(findings, 2)
    assert carried == ()
    state = LoopState(mode="verify", round_number=2, prior_findings=carried, retired_prior=retired)
    outcome = apply_round(state, [], {})
    report = "\n".join(round_report(state, outcome))
    # The retirement line IS the resolution report; the empty-backlog
    # fallback would contradict it.
    assert "No prior findings were open" not in report
    assert "2 nit finding(s) from earlier rounds retired" in report
    assert outcome.open_issue_count == 0


def test_merge_resolutions_is_conservative() -> None:
    prior = {"r1-1"}
    fixed = Resolution(id="r1-1", status="fixed", note="")
    not_fixed = Resolution(id="r1-1", status="not_fixed", note="still broken")
    disputed = Resolution(id="r1-1", status="disputed", note="by design")
    assert merge_resolutions([[fixed], [fixed]], prior)["r1-1"].status == "fixed"
    assert merge_resolutions([[fixed], [not_fixed]], prior)["r1-1"].status == "not_fixed"
    assert merge_resolutions([[not_fixed], [disputed]], prior)["r1-1"].status == "disputed"
    assert merge_resolutions([[Resolution(id="zz", status="fixed", note="")]], prior) == {}


def test_pr354_wrong_enum_fix_remains_open_with_consistent_disposition() -> None:
    """Regression: a genuinely wrong implementation enum remains open."""
    state = LoopState(mode="verify", round_number=2, prior_findings=(_finding("r1-1"),))
    resolution = Resolution(
        id="r1-1",
        status="fixed_incorrectly",
        note="The implementation still uses the wrong enum value.",
    )
    outcome = apply_round(state, [], {"r1-1": resolution})
    report = "\n".join(round_report(state, outcome))

    assert outcome.open_issue_count == 1
    assert outcome.open_bug_count == 1
    assert outcome.ledger.findings[0].id == "r1-1"
    assert "fixed incorrectly" in report
    assert "wrong enum value" in report
    assert "Open findings after this round: 1" in report


def test_pr355_all_fixed_correctly_produces_clean_ledger_and_counts() -> None:
    """Regression: authoritative fixed statuses converge the review to clean."""
    state = LoopState(
        mode="verify",
        round_number=2,
        prior_findings=(
            _finding("r1-1"),
            _finding("r1-2", severity="risk"),
        ),
    )
    resolutions = {
        finding.id: Resolution(
            id=finding.id,
            status="fixed",
            note="Fixed correctly in the current head.",
        )
        for finding in state.prior_findings
    }
    outcome = apply_round(state, [], resolutions)
    report = "\n".join(round_report(state, outcome))

    assert outcome.ledger.findings == ()
    assert outcome.open_issue_count == 0
    assert outcome.open_bug_count == 0
    assert report.count("fixed correctly") == 2
    assert "fixed incorrectly" not in report
    assert "Open findings after this round: 0" in report


def test_encode_ledger_trims_to_fit_and_keeps_open_findings() -> None:
    open_findings = tuple(
        LedgerFinding(
            id=f"r1-{n}",
            severity="bug",
            file=None,
            line=None,
            title="T" * 160,
            evidence="E" * 600,
            status="open",
            models=(),
        )
        for n in range(1, 60)
    )
    disputed = tuple(
        LedgerFinding(
            id=f"r2-{n}",
            severity="nit",
            file=None,
            line=None,
            title="D" * 160,
            evidence="E" * 600,
            status="disputed",
            models=(),
        )
        for n in range(1, 60)
    )
    marker = encode_ledger(
        Ledger(round_number=2, findings=open_findings + disputed, reviewed_sha=""),
        repo=REPO,
        pr_number=1,
    )
    assert len(marker) <= 40_000
    decoded = extract_ledger(marker, repo=REPO, pr_number=1)
    assert decoded is not None
    open_ids = {finding.id for finding in decoded.findings if finding.status == "open"}
    assert len(open_ids) == 59  # every open finding survives trimming


def test_encode_ledger_overflow_fails_visibly() -> None:
    too_many = tuple(
        LedgerFinding(
            id=f"r1-{n}",
            severity="nit",
            file=None,
            line=None,
            title="t",
            evidence="",
            status="open",
            models=(),
        )
        for n in range(1, 202)
    )
    with pytest.raises(ActionError, match="overflow"):
        encode_ledger(
            Ledger(round_number=1, findings=too_many, reviewed_sha=""),
            repo=REPO,
            pr_number=1,
        )


def test_ledger_generation_roundtrip() -> None:
    ledger = Ledger(
        round_number=1,
        findings=(_finding(),),
        reviewed_sha="a" * 40,
        generation="1234567890ab",
    )
    marker = encode_ledger(ledger, repo=REPO, pr_number=7)
    decoded = extract_ledger(marker, repo=REPO, pr_number=7)
    assert decoded is not None
    assert decoded.generation == "1234567890ab"
