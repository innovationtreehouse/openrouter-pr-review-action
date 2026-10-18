import pytest

from or_pr_review.cli import _review_event
from or_pr_review.errors import ActionError


@pytest.mark.parametrize(
    ("approve", "verdict", "open_issues", "event"),
    [
        (None, "clean", 0, "COMMENT"),  # default off
        ("off", "clean", 0, "COMMENT"),
        ("clean", "clean", 0, "APPROVE"),
        (" Clean ", "clean", 0, "APPROVE"),
        ("clean", "clean", 1, "COMMENT"),
        ("clean", "issues", 2, "COMMENT"),
        ("clean", "partial", 0, "COMMENT"),
        ("clean", "error", 0, "COMMENT"),
    ],
)
def test_review_event(approve, verdict, open_issues, event) -> None:
    env = {} if approve is None else {"APPROVE": approve}
    assert _review_event(env, verdict, open_issues) == event


def test_review_event_rejects_unknown_mode() -> None:
    with pytest.raises(ActionError, match="approve must be"):
        _review_event({"APPROVE": "always"}, "clean", 0)


class _FakeGitHub:
    def __init__(self, refuse_approve: bool) -> None:
        self.refuse_approve = refuse_approve
        self.events: list[str] = []

    def create_review(self, number, body, sha, comments=None, event="COMMENT"):
        self.events.append(event)
        if event == "APPROVE" and self.refuse_approve:
            raise ActionError("422 GitHub Actions is not permitted to approve pull requests.")
        return {"html_url": "u"}


def test_refused_approval_posts_as_comment() -> None:
    from or_pr_review.cli import _post_review

    gh = _FakeGitHub(refuse_approve=True)
    assert _post_review(gh, 1, "b", "c" * 40, [], "APPROVE") == {"html_url": "u"}
    assert gh.events == ["APPROVE", "COMMENT"]


def test_comment_failure_is_not_retried() -> None:
    from or_pr_review.cli import _post_review

    class Broken(_FakeGitHub):
        def create_review(self, *a, **k):
            self.events.append(k.get("event", "COMMENT"))
            raise ActionError("boom")

    gh = Broken(refuse_approve=False)
    with pytest.raises(ActionError, match="boom"):
        _post_review(gh, 1, "b", "c" * 40, [], "COMMENT")
    assert gh.events == ["COMMENT"]
