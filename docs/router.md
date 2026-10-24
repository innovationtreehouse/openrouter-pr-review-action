# Routed review: `review.yml` and `router.json`

`.github/workflows/review.yml` is a reusable workflow that runs the whole routed review:
**route** (pick review profiles and a reviewer rung) → **review** (this action) → **adjudicate**
(is the posted review complete? if not, pick a stronger rung) → one **rerun** → **recommend**
(advisory line plus the `ai-recommends-approve` label) → **scrub** (exact dollar costs become a
word) → **answer** (reply to trusted questions left for the reviewer). The decision steps use a
Jev-protocol model on OpenRouter's `systemone` endpoint; every decision has a deterministic fallback
and is logged as one `ai-review-decision {json}` line.

A consumer needs the caller workflow in the README plus a config directory on a trusted branch:

```
.github/ai-review/profiles.json     # required: [{"name", "paths": [globs], "instructions"}]
.github/ai-review/instructions.md   # required: base custom_instructions for every review
.github/ai-review/router.json       # optional: overrides of the defaults below
```

## Trust model

- The router and the action run from the commit the caller pins (`job.workflow_sha`), never from
  the PR. The config directory is read from `config_ref` (default: the repository's default
  branch) with a sparse checkout, so a PR or a stacked parent branch cannot edit its own reviewer.
- The job runs only for same-repository, non-draft, non-Dependabot PRs carrying the `ai-review`
  label, as the per-repo workflows did. Under `pull_request` the caller workflow file itself still
  comes from the PR; anyone who can push a branch can change the caller.
- Only the review bot and `trusted_associations` reach a model: PR body, comments, replies and
  questions from anyone else are ignored. PR text is always data, never instructions.

Workflow inputs: `config_ref` (trusted branch, default the default branch), `config_dir`
(default `.github/ai-review`), `missing_config` (`warn` skips the review and passes; `error`
fails). Secret: `OPENROUTER_API_KEY`.

## `router.json`

Every key is optional. An unknown key, a value of the wrong type (`"false"` is not `false`), a rung missing any of its seven keys, a string containing a newline, or a broken ladder fails every router step, and the job with it. Defaults:

| Key | Default | Meaning |
| --- | --- | --- |
| `decision_model` | `baseten/mammoth-1:free` | Router, judge and pick-mode decision model |
| `decision_state_max` | `48000` | Decision-state cap in bytes, at least 16000; the token cap is a quarter of it. Jev 1.14 takes `1500000` |
| `compaction_model` | `nvidia/nemotron-3.5-super-vl:free` | The action's `compaction_model` (re-review brief), and the router's summarizer for trusted PR text the deterministic fit had to condense. `""` disables both |
| `judge_mode` | `pick` | Action `judge_mode` for multi-lane rungs |
| `approve` | `off` | Action `approve` input |
| `first_round_lowest` | `true` | A PR's first review uses the lowest rung unless the fixed ladder says `high` or trusted PR text asks for a tier |
| `rungs` | free, mid, high, top (below) | The ladder, cheapest first. Replaces the whole list |
| `free_exit` | `0.985` | Leave the free rung only when P(any paid rung) reaches this |
| `profile_threshold` | `0.5` | A profile applies at this probability |
| `complete_threshold` | `0.2` | A review is complete at this probability; below it a rerun is considered |
| `min_credit_usd` | `5` | Below this OpenRouter credit, route to the free rung and order no reruns |
| `large_lines`, `large_files`, `very_large_lines` | `800`, `25`, `2500` | Fixed-ladder fallback: `mid` above the first two, `high` above the third |
| `high_profiles` | `migrations`, `security-boundary`, `iam-and-secrets`, `state-safety` | Fixed-ladder fallback goes `high` when one applies |
| `trusted_associations` | `OWNER`, `MEMBER`, `COLLABORATOR` | Whose PR text, comments and questions count |

Each rung: `name`, `model` (a comma list is several lanes the judge picks between),
`max_tool_turns`, `max_diff_kb`, `effort` (`""` for the action default), `bar`, and `description`
(what the decision model reads when choosing). The ladder must start with `free` and contain
`mid` and `high`; any other rung is optional. Default ladder:

| Rung | Model | Turns | Diff KB | Effort | Bar |
| --- | --- | --- | --- | --- | --- |
| `free` | `nvidia/nemotron-3-ultra-550b-a55b:free,recursive/flint-v1.0:free` | 50 | 300 | medium | 0 |
| `mid` | `deepseek/deepseek-v4-pro` | 30 | 300 | | 0.5 |
| `high` | `google/gemini-4-argon` | 12 | 150 | | 0.8 |
| `top` | `anthropic/claude-opus-5.5` | 8 | 120 | | 0.975 |

Among paid rungs, and for reruns, the router moves to a rung only when P(this rung or higher)
clears its `bar`; a rerun also needs `1 - P(complete)` to clear it. Example: an extra `low` rung
between `free` and `mid`:

```json
{
  "rungs": [
    {"name": "free", "model": "nvidia/nemotron-3-ultra-550b-a55b:free,recursive/flint-v1.0:free",
     "max_tool_turns": 50, "max_diff_kb": 300, "effort": "medium", "bar": 0,
     "description": "A cent or two a review. Routine changes: docs, tests, small refactors, config, renames."},
    {"name": "low", "model": "nvidia/nemotron-3.5-lightning", "max_tool_turns": 30, "max_diff_kb": 300,
     "effort": "", "bar": 0.25,
     "description": "A few cents a review. For routine code changes when the free tier is likely to miss something but the mid tier is not needed."},
    {"name": "mid", "model": "z-ai/glm-5.3", "max_tool_turns": 30, "max_diff_kb": 300, "effort": "", "bar": 0.5,
     "description": "Quarters a review. Real logic changes, one risky area, or a large diff. The default for anything beyond routine."},
    {"name": "high", "model": "google/gemini-4-argon", "max_tool_turns": 12, "max_diff_kb": 150, "effort": "", "bar": 0.8,
     "description": "Around a dollar a review. Only when the mid reviewer is likely to miss something serious."},
    {"name": "top", "model": "anthropic/claude-opus-5.5", "max_tool_turns": 8, "max_diff_kb": 120, "effort": "", "bar": 0.975,
     "description": "Dollars a review. Choose this almost never."}
  ]
}
```

## Tier requests

A trusted PR description, comment or review that asks for a reviewer tier ("please run this on
Argon", "use the high lane") is passed to the decision model, which by default grants it. A granted
request is a floor: it never lowers the rung, and it is refused only when absurd for the change (a
whitespace change asking for `top`). It also lifts `first_round_lowest` for that round.

## Decision-state compaction

Route and adjudicate send the decision model the PR title, the per-file diff, every earlier AI
review round (body, inline comments and, under `files` as `round_N/<path>`, the diff it covered),
trusted PR text and, for adjudicate, the posted review. Nothing is cut. Under `decision_state_max`
the state goes unchanged. Over it, earlier rounds' diffs fold first into per-round counts
(largest first, until the state fits), as the old router dropped them first. Then pick mode's
tiered compactor (`pick.fit_state`) runs, largest items first, stopping as soon as the state fits:
strip HTML comments, drop duplicates, collapse unchanged diff context, reduce diffs to file and
hunk headers, fold the remaining files into per-directory counts, digest earlier rounds to one line
each, condense trusted PR text and the posted review (they sit in `lanes`, which goes last), and
finally one line per section. The state header records every step. This fit calls no model, so it
works at any input size.

The newest trusted text that asks for a tier is never condensed. If other trusted text had to be
condensed (cut to its first 280 characters), the router refits with each such text map-reduced by
`compaction_model` to its share of the cap instead; if the model fails, the condensed state stands.

## Running the router by hand

`python3 -P -m or_pr_review.router <route|adjudicate|recommend|scrub|answer>` with
`PYTHONPATH=src`, `AI_REVIEW_CFG` (the config directory), `PR_NUMBER`, `GITHUB_REPOSITORY`,
`GITHUB_OUTPUT`, `GH_TOKEN` and, for route, adjudicate and answer, `OPENROUTER_API_KEY`.
