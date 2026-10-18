"""Measure the size of a RE-REVIEW user message for a PR (read-only GitHub GETs).

Rebuilds the prompt sections with the action's own code (loop.py, prompt.py,
github_ops.py). The diff is fetched only to measure its size; it is never
printed or stored. Verify mode is modelled as a rebase review (the largest case).

    PYTHONPATH=src python bench/context_budget.py --repo o/r --pr 1 --pr 2 [--impl]

--impl: if or_pr_review.compaction.compact_history exists, also report its
output size for the history block (called as compact_history(bodies, replies,
comments) -> str; adjust the call when that module lands).
"""

from __future__ import annotations

import argparse
import re
import subprocess

from or_pr_review import loop as L
from or_pr_review import prompt as P
from or_pr_review.collect import CollectedReview, DiffPlan, Truncation
from or_pr_review.github_ops import GitHub

BOT = "github-actions[bot]"
MAX_DIFF = 300 * 1024  # DEFAULT_MAX_DIFF_KB embed cap
COMMIT_RE = re.compile(r"\*\*Commit:\*\*\s*`([0-9a-f]{7,40})`")


def b(s: str) -> int:
    return len(s.encode("utf-8"))


def gh_get(path: str, accept: str | None = None) -> str:
    cmd = ["gh", "api"] + (["-H", f"Accept: {accept}"] if accept else []) + [path]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def measure(repo: str, pr: int, impl: bool) -> dict:
    token = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
    gh = GitHub(token=token, repository=repo)
    import json

    meta = json.loads(gh_get(f"repos/{repo}/pulls/{pr}"))
    diff = gh_get(f"repos/{repo}/pulls/{pr}", "application/vnd.github.diff")
    diff_raw = b(diff)
    embedded = diff.encode()[:MAX_DIFF].decode(errors="ignore")

    bodies = gh.list_bot_review_bodies(pr, BOT)
    ledger = L.latest_ledger(bodies, repo=repo, pr_number=pr)
    state = None
    if ledger:
        state = L.LoopState(mode="verify", round_number=ledger.round_number + 1,
                            prior_findings=ledger.findings, generation=ledger.generation)
    replies_rebase = gh.list_rebase_replies(pr, BOT)
    comments = gh.list_recent_issue_comments(pr)
    history = L.render_rebase_context(bodies, replies_rebase, comments)
    carried = {f.id for f in state.prior_findings} if state else set()
    verify_replies_raw = gh.list_finding_replies(pr, generation=state.generation) if state else []
    verify_ctx = L.render_agent_context([r for r in verify_replies_raw if r[0] in carried], comments)

    # Duplicate bot reviews of the same commit, and bytes dropped by _clip_tail.
    shas = [m.group(1) if (m := COMMIT_RE.search(x)) else None for x in bodies]
    keyed = [s for s in shas if s]
    dup = len(keyed) - len(set(keyed))
    stripped = [s for s in (re.sub(r"<!--[\s\S]*?-->", "", x).strip() for x in bodies) if s]
    pre = b("\n\n--- Earlier review ---\n\n".join(stripped))
    reviews_part = history.split("\n\nHistorical replies and PR discussion:")[0]
    kept = b(reviews_part) - b(f"Historical bot reviews ({len(stripped)} available, oldest first). "
                               "These describe earlier code; current code must establish every conclusion.\n\n")
    kept -= b(L.OMISSION_MARKER) if L.OMISSION_MARKER in reviews_part else 0
    dropped = max(0, pre - kept)

    collected = CollectedReview(
        pr_number=pr, title=meta["title"], body=meta.get("body") or "", head_sha=meta["head"]["sha"],
        base_ref=meta["base"]["ref"], head_ref=meta["head"]["ref"],
        plan=DiffPlan("rebase", "full-pr", None, None, None),
        truncation=Truncation(embedded, diff_raw > MAX_DIFF, diff_raw, b(embedded), 300),
        mode="verify" if state else "initial",
    )
    ledger_block = P._loop_block(state, "", rebase=True) if state else ""
    user = P._user_prompt(collected, custom_instructions="", loop=state, agent_replies=history,
                          path_profiles=None)
    system = P._system_prompt(tone="professional", mode=collected.mode, full_pr=False, rebase=True)
    fenced_history = P._fence(history) if state else ""
    body_b = b(collected.body)
    overhead = b(user) - body_b - b(embedded) - b(ledger_block) - b(fenced_history if state else "")
    rows = {
        "system prompt": b(system),
        "metadata/policy/profiles (approx, remainder)": overhead,
        "PR body": body_b,
        "ledger block (prior findings)": b(ledger_block),
        "verify replies block (render_agent_context)": b(verify_ctx),
        "rebase history block (render_rebase_context)": b(history) if state else 0,
        "diff (embedded, capped 300KB)": b(embedded),
    }
    # Without a ledger the action stays in initial mode and sends no history.
    hist_if_loop = b(history)
    if impl:
        try:
            from or_pr_review import compaction

            rows["history AFTER compaction"] = b(compaction.compact_history(bodies, replies_rebase, comments))
        except ImportError:
            rows["history AFTER compaction"] = -1
    user_total = b(user)
    return dict(pr=pr, rows=rows, user_total=user_total, system=b(system), ledger=bool(ledger),
                n_bot=len(bodies), dup=dup, dropped=dropped, diff_raw=diff_raw,
                hist_if_loop=hist_if_loop, adds=meta["additions"], dels=meta["deletions"],
                n_replies=len(replies_rebase), n_comments=len(comments))


def render(r: dict) -> str:
    t = lambda n: f"{n // 4:,}"
    out = [f"### PR #{r['pr']}  (ledger marker: {'yes' if r['ledger'] else 'NO - action would run initial, history not sent; history rows shown for reference'})",
           f"bot reviews: {r['n_bot']} (same-commit duplicates: {r['dup']}); thread replies: {r['n_replies']}; "
           f"issue comments kept: {r['n_comments']}; _clip_tail dropped: {r['dropped']:,} B; "
           f"diff raw {r['diff_raw']:,} B (+{r['adds']}/-{r['dels']})", "",
           "| section | bytes | est. tokens |", "|---|---:|---:|"]
    for k, v in r["rows"].items():
        out.append(f"| {k} | {v:,} | {t(v)} |")
    if not r["ledger"]:
        out.append(f"| (history if loop were active) | {r['hist_if_loop']:,} | {t(r['hist_if_loop'])} |")
    # verify replies block is an alternative to the rebase block, not additive.
    out += [f"| **user message total** | {r['user_total']:,} | {t(r['user_total'])} |", "",
            "Per lane, user message resent on N tool turns (est. input tokens, no caching; system + user per turn):", "",
            "| N | tokens |", "|---:|---:|"]
    per = (r["user_total"] + r["system"]) // 4
    for n in (1, 8, 12):
        out.append(f"| {n} | {per * n:,} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int, action="append", required=True)
    ap.add_argument("--impl", action="store_true", help="also size or_pr_review.compaction.compact_history if present")
    a = ap.parse_args()
    for pr in a.pr:
        print(render(measure(a.repo, pr, a.impl)))


if __name__ == "__main__":
    main()
