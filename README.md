# openrouter-pr-review-action

An MIT-licensed GitHub Action that reviews pull requests using one or more models through [OpenRouter](https://openrouter.ai). It reads the diff and repository context, posts structured findings, and tracks unresolved findings across follow-up reviews. It works with human authors and coding agents; it does not make fixes or merge PRs.

Author: **Nathan (FlyOverCoderKY) / RetireGolden, LLC**.

**Start here:** [Setup](#setup) · [Routed review](#copy-paste-routed-review-decision-model-tiers-judge-rerun) · [Choose models](#choosing-models) · [Repository guidance](#repository-review-guidance) · [Trusted profiles](docs/review-profiles.md) · [Copy-paste workflow](#copy-paste-one-lane-grok-via-openrouter) · [Agentic loops and merge gates](#agentic-loops-and-merge-gates) · [Troubleshooting](#troubleshooting) · [Inputs](#inputs)

## Setup

1. Enable GitHub Actions in the consuming repository and allow this action under any organization action restrictions. The examples use GitHub-hosted `ubuntu-latest`. A custom runner needs Bash, Git, GitHub CLI (`gh`), Python 3.11+, and access to GitHub and OpenRouter; no application dependency installation is needed.
2. Create an OpenRouter key with credit available and save it under **Settings → Secrets and variables → Actions → New repository secret**, named `OPENROUTER_API_KEY`. An organization secret also works if this repository has access. Keep the value out of chat and committed files.
3. Save the [one-lane workflow below](#copy-paste-one-lane-grok-via-openrouter) as `.github/workflows/openrouter-review.yml`. It includes initial and follow-up reviews, checkout, permissions, and separate concurrency groups. The examples pin this action to a full commit SHA; update that pin deliberately when adopting fixes.
4. Choose a model using the [benchmark guidance](#choosing-models). Start with one lane to keep setup and cost simple. API usage is paid through your OpenRouter account; GitHub runner usage is separate. Tool/time budgets are limits on work, not a dollar spending cap.
5. Open a small, non-draft PR from a branch in the same repository. Check that the **first-pass** job posts a review with the expected model and head SHA. Wait for it to finish, then push a change and verify that **follow-up** posts the next round. This smoke test makes paid model calls.

The examples intentionally skip drafts, fork PRs, and Dependabot PRs. They are a setup for trusted same-repository contributors. Fork review requires a separately designed trusted workflow; changing the event to `pull_request_target` alone is not a safe installation fix. See [Untrusted pull requests](#untrusted-pull-requests).

For a PR already open when you install the workflow, trigger an initial event first (for example, mark a draft ready for review, or reopen an eligible PR). A push alone starts a verification round and cannot bootstrap a missing initial review. If the initial run is partial or fails, resolve its cause and obtain a complete initial review before relying on follow-ups.

### Ask your coding agent to install it

Copy this prompt into your coding agent:

```text
Set up FlyOverCoderKY/openrouter-pr-review-action in this repository. Read its
README and action.yml at the exact revision you will use, plus this repository's
agent instructions and existing workflows. Pin the action to a full commit SHA.
Use the README's same-repository, non-draft initial + follow-up workflow, keeping
the two concurrency groups separate. Use ubuntu-latest, a 25-minute job timeout,
full-depth checkout at the PR head, persist-credentials: false, contents: read,
and pull-requests: write. Do not install or execute this repository's code in
the review job. Start with one model; consult the linked benchmark and explain
the choice and its limitations. Keep fail_on: never for the initial rollout.
Use the OPENROUTER_API_KEY Actions secret, never a literal key. Tell me if the
secret or repository settings need my attention; do not request the key in chat.
Preserve existing CI and merge rules, check whether an existing review bot would
duplicate this one, and explain any GITHUB_TOKEN-trigger limitation for my agents.
Validate the workflow without calling a model, then describe how to smoke-test
both review rounds and what paid API usage that test will trigger. Do not enable
automatic merging or treat job success as proof of a clean review.
```

## Choosing models

Use the [Review Benchmark leaderboard](https://bench.flyovercoder.com/leaderboard) for model recommendations and measured quality, completion, latency, and cost. Read the [methodology](https://bench.flyovercoder.com/methodology) and each result's status and configuration alongside its rank. The board is a small, harness-specific comparison; AI-adjudicated results are provisional, and a model's position does not certify its performance on every repository.

- **One reviewer:** set `models` to one OpenRouter slug. The default is `x-ai/grok-4.6`; it is a configuration default, not a claim that it is always the best value.
- **Multiple reviewers:** supply up to four comma-separated slugs. Each lane reviews the change, adding API cost. A strong individual ranking does not establish that two models catch different defects; assess the extra useful findings before keeping a second lane.
- **Judge:** `judge_model` combines reviewer findings. Choose it separately from the reviewers; its merge benchmark answers a different question from the review leaderboard. The default and rationale are [below](#copy-paste-multi-model-bake-off).

Translate a board recommendation into the exact OpenRouter model slug in `models`, and check its availability in your account. Keep the chosen slug and action SHA explicit in your workflow so a later comparison has a known starting point. The README links to the evolving results instead of maintaining a second ranking table.

This project is independent of its sibling [`grok-pr-review-action`](https://github.com/FlyOverCoderKY/grok-pr-review-action). You do not need that action, the Grok CLI, or an `XAI_API_KEY` to use this one.

## What it does

- One invocation = one review. Callers own concurrency and merge gating.
- `models` is a comma-separated list of OpenRouter slugs. **List length is the lane count** (hard-capped at **4**; the action fails clearly if you ask for more).
- **One lane:** by default the judge is skipped and that lane’s structured findings are posted directly. Set `judge_needed: true` to force the configured judge for a single lane. The finding can still name the model that produced it.
- **Two or more lanes:** parallel review lanes (same prompt on every lane in v1), then an OpenRouter **judge** union-merges them into **one** GitHub review when the shared job budget leaves a safe judge window. Every input finding is identity-tracked, the judge must account for every id, only duplicates whose source location, title and evidence agree may merge, and any unaccounted or over-broadly merged finding is deterministically restored verbatim before the shared 80-finding publishing cap is applied. A judge transport/schema failure or an exhausted judge window posts the validated deterministic union and labels that degradation in the review instead of discarding completed lanes. If a repaired or fallback union exceeds the cap, its visible mode reports how many lower-severity findings were omitted. Attribution looks like:

  ```text
  #### 🔴 Issue 1 — Missing auth check

  `src/api.py:42` · `bug` · identified by x-ai/grok-4.6 and anthropic/claude-sonnet-4.6
  ```

  then the issue body (severity emoji: 🔴 bug / 🟠 risk / 🔵 nit).

- Prefer a **setup → matrix lanes → judge** workflow so wall clock is roughly the slowest lane + judge, not the sum. GitHub **bills the minutes in parallel**. A single-job `role: all` run can still fan lanes in-process; that job is billed as one runner.
- If a **lane** fails, that lane fail-opens and the judge (or the single-lane poster) continues on whatever structured results arrived. A lane that reports only temporary checkout/tool-access failures produces a visible **partial** review, never an unmarked clean pass. Action-wide collection, input, artifact, and posting errors still fail the job; judge transport/schema errors preserve validated lane findings via a visibly labeled deterministic union, subject to the documented publishing cap.

Persona lanes are **out of v1** (path-scoped guidance is available via `path_profiles`). The `persona` input is a reserved unused hook so a later persona feature does not require a rewrite. A future **single-persona** run should skip the judge the same way (one reviewer = no judge).

## Repository review guidance

Optional `REVIEW.md` files give reviewers durable project context: invariants,
supported behavior, sources of truth, and concrete failure modes to check.
Root guidance applies across the PR; nested files add guidance for their own
directories. They never exclude changed files or replace your CI rules.

Use a pinned action revision containing this feature and add this input to
both direct-action review steps (`role: all`, with a full-depth checkout):

```yaml
with:
  review_policy: base
```

The workflow examples in this README pin a revision that contains this feature.
Leave `review_policy` off for unchanged behavior; matrix runs without policy
keep working as before. Setting `review_policy: base` on `setup`, `lane`, or
`judge` roles requires the prepared flow: setup needs a full-depth checkout of
the reviewed repository and writes the guidance into a context artifact, and
every lane and the judge must receive that artifact through
`review_context_file` and `review_context_sha256` (see
[prepared matrix execution](docs/review-profiles.md#prepared-matrix-execution)).
Policy-enabled matrix roles without the prepared context fail before model
calls. The shipped reusable workflow does not accept `review_policy`; use the
direct action, or a workflow of your own that threads the setup artifact.

A useful first `REVIEW.md` can be plain Markdown:

```markdown
# Review guidance

Persisted documents must remain readable after an upgrade. When storage
formats change, follow the reader, writer, migration, and restore paths.
The contract is documented in docs/storage.md; verify claims against code.

For a bug, identify its trigger, affected behavior, and supporting code.
Uncertainty about a requirement is a question, not proof of a defect.
```

Guidance comes from the **immutable target-branch commit**, not the PR's file
contents. A PR adding or changing `REVIEW.md` therefore uses the existing policy;
its new guidance starts applying after merge. The review reports the source
commit and effective-policy digest. All lanes and the merge judge receive the
same frozen guidance, including during verification rounds.

See [the policy reference](docs/review-policy.md) for hierarchy, metadata,
validation limits, local previews, privacy, and troubleshooting. For named
standard/deep panels, bounds, prepared matrix artifacts, and the authorization
boundary, read [trusted review profiles](docs/review-profiles.md).

When asking a coding agent to install this feature, add: “Read the policy
reference at the pinned action revision. Enable target-branch REVIEW.md guidance
for both all-role review steps, write concise contracts verified against local
sources, and run local lint and explain without model calls. Preserve existing
CI, model settings, and the findings ledger. Verify the receipt on the next
normal PR after the guidance lands.”

### Copyable all-role profile workflow

Use this for a manually requested deep review. The pin below is a reviewed
action revision that contains the `review_profiles` input; do not point at an
old tag and assume the feature is there. The handler that authorizes a deep request is a separate integration.

```yaml
name: OpenRouter profile review
on:
  workflow_dispatch:
    inputs:
      pr_number:
        required: true
        type: number
      review_level:
        required: true
        type: choice
        options: [auto, deep]
        default: deep
jobs:
  review:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    permissions:
      contents: read
      pull-requests: write
    steps:
      - name: Resolve PR head
        id: pr
        env:
          GH_TOKEN: ${{ github.token }}
          # No checkout has run yet, so gh cannot infer the repository from a git remote.
          GH_REPO: ${{ github.repository }}
        run: echo "head_sha=$(gh pr view '${{ inputs.pr_number }}' --json headRefOid --jq .headRefOid)" >> "$GITHUB_OUTPUT"
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ steps.pr.outputs.head_sha }}
          fetch-depth: 0
          persist-credentials: false
      - uses: FlyOverCoderKY/openrouter-pr-review-action@212775ffea22e806cddcb706c73a3df26fbcb6d0
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ inputs.pr_number }}
          review_level: ${{ inputs.review_level }}
          review_profiles: ${{ vars.REVIEW_PROFILES_JSON }}
          review_scope: full-pr
          review_mode: auto
          fail_on: never
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Replace the checkout `ref` with the resolved PR head SHA in the workflow that
you install; the PR number must never accidentally select the default branch.
Keep full history and do not execute checked-out PR code. A profile run emits
the resolved profile, level, trigger, registry digest, panel status, context
digest/artifact, satisfaction flag, and reviewed head SHA.

## Auth

Real auth is **`OPENROUTER_API_KEY` against `https://openrouter.ai/api/v1/chat/completions` only**.

1. Create an API key at [openrouter.ai](https://openrouter.ai).
2. Store it as the repository secret `OPENROUTER_API_KEY`.
3. Pass it as a **job environment variable**, not as an action input (so it stays out of input logs):

```yaml
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

The action fails closed if the key is empty (when a lane or the multi-lane judge needs it), never prints the key, and never documents or accepts `XAI_API_KEY`.

## Privacy

This action uses OpenRouter as an external processor. The prompt sends PR metadata, the selected diff, `custom_instructions`, matching `path_profiles`, and enabled applicable `REVIEW.md` guidance to OpenRouter, which routes them to the **upstream model provider** for each slug. When a lane uses `read_file`, `grep`, or `list_dir`, those paths and file contents can also be sent as model context.

Do not put secrets, credentials, regulated personal data, or unrelated confidential material in PR text, diffs, repository files, `custom_instructions`, or `path_profiles`. Obtain organizational approval before enabling the action on private or regulated repositories.

## Untrusted pull requests

PR title, body, diffs, and repository files are treated as **untrusted data**. The action does **not** execute PR code. Before the model runs, tracked files from the reviewed commit are materialized into an inert, size-bounded checkout. Tools are read-only (`read_file`, `grep`, `list_dir`). There is no shell, no writes, no network except OpenRouter, no subagents, and no memory. Secret-like paths (`.env`, keys, credential files) are refused.

The checkout must contain the reviewed commit object (`fetch-depth: 0` is the safest setup). GitHub does not give repository secrets or a write token to ordinary `pull_request` workflows from public forks. Do not work around that by blindly checking out and executing fork code under `pull_request_target`.

The default first-pass tool budget is **50** read-only rounds (`max_tool_turns`), matching the sibling Grok action's default `max_turns`. The review prompt tells the model to use those tools for **blast radius** — filename-inventory tests, README / code-map docs, and sibling CI files — not just the embedded diff. A workflow-only PR can still break a test that requires every `.github/workflows/*.yml` to be listed in docs. Follow-up jobs may pass a lower budget (sibling callers often use `30`).

## Copy-paste: one-lane Grok via OpenRouter

First-pass + latest-commit follow-up. The default `models` value is one slug (`x-ai/grok-4.6`), so the **judge does not run**.

```yaml
name: OpenRouter PR review

on:
  pull_request:
    types: [opened, reopened, ready_for_review, synchronize]

jobs:
  first-pass:
    if: >-
      github.event.action != 'synchronize' &&
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    timeout-minutes: 25
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      group: or-review-first-pass-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}

      - uses: FlyOverCoderKY/openrouter-pr-review-action@212775ffea22e806cddcb706c73a3df26fbcb6d0
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: full-pr
          review_mode: auto
          fail_on: never
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

  follow-up:
    if: >-
      github.event.action == 'synchronize' &&
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    timeout-minutes: 25
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      group: or-review-follow-up-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}

      - uses: FlyOverCoderKY/openrouter-pr-review-action@212775ffea22e806cddcb706c73a3df26fbcb6d0
        with:
          github_token: ${{ github.token }}
          pr_number: ${{ github.event.pull_request.number }}
          models: x-ai/grok-4.6
          review_scope: latest-commit
          review_mode: verify
          fail_on: never
          max_tool_turns: "30"
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

The examples pin `212775ffea22e806cddcb706c73a3df26fbcb6d0`, the `main` revision that adds rebase reviews on top of the review-integrity, deadline, and profile work; a branch push does not update existing consumers' pins. Do not put first-pass and follow-up in one concurrency group that `synchronize` can cancel.

When a `latest-commit` follow-up detects that the last reviewed commit is no longer an ancestor, it automatically selects **`rebase`**: a full-PR review with earlier review context and preserved finding identities. Missing compare inputs or transient compare errors still embed the **single latest head commit**, with a visible **partial** verdict. For `full-pr` and `rebase`, if GitHub rejects the patch because it exceeds its line limit, the action computes the complete `git diff base...head` from the full-depth workflow checkout before applying the normal diff-budget triage. A truncated diff is never treated as clean.

The initial round is the exhaustive pass: the prompt requires a per-file, all-severity sweep (bug, risk, **and** nit), and each coverage entry claims a completed sweep of that file. From verify round 2 onward a **severity floor** applies: carried `nit` findings are retired — stated visibly in the round's resolution section, never silently — so follow-up rounds track the bug/risk backlog to convergence instead of re-adjudicating nits forever.

## Copy-paste: multi-model bake-off

Use the reusable workflow so lanes are **separate GitHub jobs**. Wall clock ≈ slowest lane + judge. GitHub bills those minutes **in parallel**. This example runs an initial review only. For a complete loop, the one-job example above also accepts multiple slugs; alternatively add a separate reusable-workflow follow-up job with `synchronize`, `review_mode: verify`, `review_scope: latest-commit`, and its own concurrency group.

```yaml
name: OpenRouter bake-off

on:
  pull_request:
    types: [opened, reopened, ready_for_review]

jobs:
  review:
    if: >-
      github.event.pull_request.draft == false &&
      github.event.pull_request.head.repo.full_name == github.repository &&
      github.event.pull_request.user.login != 'dependabot[bot]' &&
      github.actor != 'dependabot[bot]'
    concurrency:
      group: or-review-first-pass-${{ github.repository }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    uses: FlyOverCoderKY/openrouter-pr-review-action/.github/workflows/pr-review.yml@212775ffea22e806cddcb706c73a3df26fbcb6d0
    permissions:
      contents: read
      pull-requests: write
    with:
      models: x-ai/grok-4.6,anthropic/claude-sonnet-4.6
      review_scope: full-pr
      review_mode: initial
      # Optional. Default is openai/gpt-5.6-luna (merge only).
      # judge_model: google/gemini-3.1-flash-lite  # measured lower-latency override
    secrets:
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Two or more slugs schedule a judge. When using the direct action, `judge_needed: true` can also force it for one configured lane; the reusable workflow does not expose that override. The default judge is `openai/gpt-5.6-luna`, selected by the repeatable synthetic judge benchmark after it retained 50/50 expected findings with 100% precision, zero duplicates, correct verdicts, and no repair/fallback across five runs per fixture. It is a coverage-checked **union-merge** of already-structured findings plus JSON schema — not a second reviewer and not a filter: identity-tracked coverage plus a conservative source-evidence merge check restore anything the judge drops or over-merges. Judged output preserves validated lane findings up to the shared 80-finding publishing cap; if a repaired or fallback union exceeds that cap, it retains the strongest severities and reports the omitted count in the visible judge mode. If only one of several configured lanes succeeds, that survivor posts directly because there is nothing to merge. Thinking/reasoning is pinned to `minimal`. In single-job `role: all`, reviewers get priority: the shared budget reserves 60 seconds total for the judge, plus 180 seconds for publication and a 5-second scheduling margin. Judge retries share one absolute deadline; they do not each reserve another request window. If judging times out or less than 15 seconds remain for a useful request, the posted review explicitly uses the deterministic union of validated lane findings. Dedicated `role: judge` jobs retain the configured per-request inactivity timeout, with retries bounded by their own remaining job time after publication reserve. Give `role: all` callers at least a 25-minute job timeout (the reusable matrix workflow uses separate jobs and is not constrained by this shared envelope).

Alternatives (do not change the default unless you mean to):

| Slug | When to use |
| --- | --- |
| `google/gemini-3.1-flash-lite` | Lower-latency legacy default; the decision benchmark found lower recall and precision |
| `anthropic/claude-haiku-4.5` | An unbenchmarked cross-vendor override |

A judge schema or transport failure is labeled on the posted review and falls back to the deterministic coverage-preserving union, subject to the same visible 80-finding publishing cap. Invalid lane artifacts or action-wide contract errors still fail closed.

### Pick mode (`judge_mode: pick`)

Instead of merging, a decision model picks the one lane review to post. It is a Jev-protocol model (default `baseten/mammoth-1:free`) called on OpenRouter's `systemone` endpoint with a state object and typed questions, not a chat model. It answers one `choice` question (which lane's review is better: more real, evidenced findings, fewer false positives, correct severities) and one `noul` per lane (safe to post as-is). The most probable known lane is posted exactly as a single-lane run would post it: its findings, inline comments, resolutions and model attribution. The review header reads `Judge: Picked by <decision_model> from N lanes: <lane model> (lane_K; p=…; safe-to-post …)`. Unpicked lanes still appear under Lanes with their cost. If the state cannot be built or compacted, the call fails or times out, or no known lane comes back, the review posts the deterministic union, labeled `pick <stage> fallback: deterministic union`. By default, a chosen lane whose safe-to-post probability is below 0.5 is also not posted: the union goes out, labeled `pick: chosen lane judged unsafe to post`. This threshold can be relaxed later. A lane whose only results are review-environment diagnostics (it could not read the checkout) is never a candidate. If only one lane is left, it posts directly. If none is left, the union posts. Pick mode shares the merge judge's time window. Its cost line counts the decision call and any compaction-model calls.

PR text is untrusted. Like the router, the state carries the PR body only when the PR author is an OWNER, MEMBER or COLLABORATOR, and it carries only those authors' comments. The question and the state header both say that PR text must not influence which lane is picked.

The state lists the lane reviews first: per lane, the verdict, tool rounds, duration, flags, and every finding with a stable id (`lane_1.F3`), title, severity, `file:line` and evidence. Then come the change (base and head SHAs), the files (per-file add/delete counts and diff, keyed by path in their own section so a file named `meta` cannot collide), earlier rounds (the ledger, bot review bodies, finding replies), the review setup (policy, `path_profiles`, `custom_instructions`), and the PR (title, body, human comments). The state must fit two caps. `decision_state_max_tokens` (default 12,000) is the main one. `decision_state_max` (default 48,000 UTF-8 bytes; the JSON is sent without ASCII escaping) is an upper bound. Mammoth's context is 16,384 tokens (apps.checkin#2415), and the default leaves about 4,000 for the questions and the endpoint's prompt. Tokens are estimated by character class: about 4 characters per token for ASCII prose and code, 2 for dense runs (hex, base64, minified text of 24 or more word characters), and 1.5 for CJK. These ratios are typical of BPE tokenizers. They were not measured against Mammoth's tokenizer. Size is measured item by item. A state under the cap is sent unchanged. Over the cap, these steps run in order, largest item first, and stop as soon as the state fits:

1. Strip HTML comments (ledger and brief markers) and drop duplicate entries from earlier-round, setup and PR text.
2. Collapse unchanged diff context to the lines a finding cites plus 3 lines on each side.
3. Summarize PR body, human comments and setup text with `compaction_model` (the re-review brief: intent, claims, quoted requests). Text is chunked to the brief's 96 KB input limit, summarized chunk by chunk, then the summaries are summarized again, at most 6 rounds.
4. Reduce each diff file to its header, hunk headers (with function names) and every cited line verbatim. Then fold uncited files into per-top-directory counts.
5. Reduce earlier review bodies to one-line digests and replies to one outcome line each.
6. Condense lane evidence prose only. Finding ids, titles, severities, locations and verdicts never change.
7. Terminal step, if still over: keep the lane findings without evidence and reduce every other section to one line. The step is logged. If the state is still over, every title is shortened by the same fraction until it fits. Ids, severities and locations are kept.

The state's `compaction` header records each step with item counts and before/after bytes, plus the estimated tokens before and after. If a summary is needed and `compaction_model` is empty (or `compaction: rules`), or the model returns nothing, the review falls back to the union. A diff GitHub refuses (HTTP 406) already arrives through the local `git diff base...head` fallback.

In `role: all`, reviewers finish up to five seconds before the coordinator stops
collecting results, leaving time to save their final diagnostics. The judge and
publication allowances described above remain outside that collection deadline.
HTTP attempts run in isolated workers with a socket inactivity timeout and an
absolute elapsed-time deadline. A response that keeps delivering data can continue
past the inactivity duration, but never beyond the current lane-stage deadline.
Without a stage deadline, the configured HTTP timeout also bounds total elapsed
time. A separate watchdog bounds DNS, connection, and header setup by the configured
HTTP timeout. Active success/error bodies, retries, and shutdown remain bounded
by the absolute stage deadline; GitHub transport calls retain their per-operation elapsed limit.
Tool exploration reserves two inactivity allowances for structured finalization,
capped at half the lane budget. The first finish can use that entire remaining
window. A retry or schema repair uses only time actually left; no speculative
repair reservation cuts a live finish short. Overall lane, judge, publication,
and job budgets do not increase. Incomplete responses never count as a review.

When available, all-role artifact uploads include `progress-N.json` checkpoints alongside the final
lane files. These contain aggregate request/tool/retry counts, observed usage and
costs, last HTTP error status, transport timeout and connection failure counts, and
provider metadata when known. They contain no prompts, tool arguments, or model
output. A timeout preserves observed costs as incomplete and never turns an
unfinished lane into a successful review.
Final failed lane results also retain aggregate timeout and connection-error history
in their error text, including prepared-context reviews that do not emit progress
checkpoints. The `idle_timeouts`, `connect_timeouts`, and `elapsed_deadlines` counters distinguish silent socket waits, connection/header watchdog expiry, and the worker reaching its absolute stage deadline. Interrupted-lane checkpoint artifacts retain these counters too. This history can include recovered errors; the leading error describes
the terminal failure. HTTP failures already include their immediate status.

Matrix lane artifacts carry a versioned publication context: the collected PR metadata,
diff and completeness accounting, prior loop state, and tool-turn policy. The judge
uses that saved context without fetching the diff or prior ledger again. A newer
push produces a stale, commit-pinned partial summary with no inline comments or new
authoritative ledger. Every available lane must agree on its context, model slot,
and reviewed commit; mismatches fail before judging or posting. Missing sibling
jobs still fail open when another lane supplied a valid context.

Use the same action revision for lanes and judge. Legacy artifacts without this
context cannot be replayed through `role: judge`; rerun the lanes. Current
`role: all` artifacts also carry this context for recovery. Contexts are limited
to 16 MiB per lane and checked before model
execution. They contain repository content and must remain trusted workflow
artifacts. Their checksum detects inconsistency, not malicious replacement.
Lane collection is still independent: a metadata or ledger change between lanes
is detected and requires a rerun rather than silently choosing one lane's state.
Pin `head_sha` to the original reviewed commit in both lane and judge jobs (as the
reusable workflow does). It declares the job's intended target; it is not a live
head lookup. If the final live-head lookup fails, publication is partial with no
new ledger. A missing or invalid context fails the job and posts a best-effort
incomplete status instead of reconstructing review state.

## Copy-paste: routed review (decision-model tiers, judge, rerun)

`review.yml` picks the reviewer per PR from a cost ladder, judges the posted review, reruns it once on a stronger rung when it is incomplete, and adds an advisory approve recommendation. Routing config lives on a trusted branch of the consumer; see [docs/router.md](docs/router.md) for `router.json`, the trust model and the decision-state compaction. Add the `ai-review` label to a PR to review it.

```yaml
name: openrouter pr review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review, labeled]

jobs:
  review:
    uses: innovationtreehouse/openrouter-pr-review-action/.github/workflows/review.yml@<full-commit-sha>
    permissions:
      contents: read
      pull-requests: write
    # with:
    #   config_ref: development    # trusted branch with .github/ai-review (default: default branch)
    #   missing_config: error      # fail instead of skipping when the config is absent
    secrets:
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

`review.yml` sets its own job concurrency group (`ai-review-routed-<PR>-...`). Do not add a workflow-level `concurrency:` block that reuses that name; drop the old per-repo `openrouter-review-*` block when migrating.

## Recommended caller concurrency

This action runs **one review per invocation**. It does not implement workflow concurrency or org-specific merge gates (those belong in your reusable caller).

Use **separate first-pass and follow-up jobs**, or distinct concurrency groups, so a `synchronize` run cannot cancel an in-progress `full-pr` review.

## Agentic loops and merge gates

The initial review must finish before the fixing agent starts its review/fix/push cycle. A verify round uses the prior trusted review ledger and carries unresolved findings forward. Keep the posting identity (`bot_login`) consistent, and leave the action's hidden review metadata intact. A custom PAT or GitHub App token requires the matching posting login.

**Job success is not a clean-review gate.** `fail_on: never` posts findings without failing for them. `fail_on: bugs` and `fail_on: any` fail for matching open findings, but neither independently rejects every `partial` result. A caller requiring a clean result should inspect `verdict` as well. For the direct action example, give the action step `id: review`, then append this step in each review job:

```yaml
      - name: Require a complete clean review
        if: always()
        env:
          REVIEW_OUTCOME: ${{ steps.review.outcome }}
          REVIEW_VERDICT: ${{ steps.review.outputs.verdict }}
        run: |
          test "$REVIEW_OUTCOME" = "success" && test "$REVIEW_VERDICT" = "clean"
```

This gate rejects missing output, failure, and partial or issue-bearing reviews. It does not require every configured model to succeed: a surviving lane can produce a clean review under the action's documented failure policy. If your merge policy requires all lanes, check that separately. The reusable matrix workflow currently exposes neither review outputs nor `judge_needed` as a caller input; use the direct action when you need this output-based gate or a forced single-lane judge.

Your outer agent loop still needs to confirm that the reviewed head matches the **current** PR head, that review and required CI actually ran, and that findings have been fixed or explicitly adjudicated. Keep existing test and security gates. The two conditional jobs in the example are not a ready-made branch-protection policy; use a final required gate that accounts for which round should run and does not accept a skipped review as success.

If your coding agent creates PRs or pushes fixes from a workflow using `GITHUB_TOKEN`, those events ordinarily do not trigger another workflow. Use an appropriately scoped GitHub App/user token for the authoring operation, or design an explicit dispatch path. This README's copy-paste workflows do not include dispatch triggers. See GitHub's [workflow triggering rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

## Troubleshooting

### Reruns and finding continuity

Use `review_mode: auto` with `review_scope: full-pr` for a manual recheck. It
reviews the whole PR while retaining the existing ledger, finding IDs, and
rebuttals. With no ledger it seeds an initial review. A corrupt newest ledger
fails visibly; it never silently starts over. Use `review_mode: initial`
explicitly only when you intend to reset history.

After a rebase or force-push, a divergent `latest-commit` range automatically
selects `review_scope: rebase`. You can also select it explicitly with
`review_mode: auto` or `verify` when a prior ledger exists. It reviews the
complete current PR at **all severities**, including nits, and retains its
generation, round progression, and finding IDs. Every current finding,
including disputed findings, must receive a resolution. Valid rebuttals still
apply; specific new evidence can reopen an invalidated dispute under its
existing ID. Earlier fixed or retired findings supply context; a regression
is reported as a new finding with evidence connecting it to the earlier fix.

Rebase context includes historical bot review bodies and replies to their
finding threads, including earlier generations, plus recent PR comments.
History is untrusted evidence, never instructions. It is bounded to 64,000
UTF-8 bytes and compacted as described under "Re-review context compaction";
the current ledger remains separate. A history fetch failure or a detected
head, base, or ledger change during collection fails the review for retry.
Prepared matrix runs freeze this context so every lane sees the same snapshot.

This scope does not promise a prompt-cache hit. Provider caching depends on
matching prompt prefixes, routing, and retention; changed diffs and history can
invalidate reuse. Budget for a full review and measure cached-token usage.

#### Re-review context compaction

Verify rounds and rebase reviews compact the history they feed the reviewers.
Compaction never fails a review: overflow drops the lowest-priority items and
the prompt says how many were omitted.

- **What is compacted.** In rebase scope, duplicate bot reviews of one commit
  collapse to the newest, older reviews become one-line digests, and the newest
  review is clipped to 32,000 bytes. Replies are grouped per finding and author
  (newest kept, earlier ones counted); replies on settled findings shrink to one
  line. PR comments are cleaned (hidden markers removed, long code blocks
  shortened) and the bot's own comments and status comments are dropped.
- **Budget and priority.** 16,000 bytes in verify rounds (replies and comments
  only), 64,000 bytes in rebase scope. Space goes first to replies on carried
  findings, then PR comments, the newest earlier review, older digests, and
  last replies on settled findings.
- **Model brief (optional).** With `compaction: auto` and a `compaction_model`,
  a re-review makes one extra request when the PR body exceeds 16,000 bytes or
  the history overflows its budget. It returns a structured brief (intent,
  claims, non-goals, test claims, open questions, finding arguments). Every
  item must carry a quote that appears verbatim in the source; items whose
  quote does not are dropped. A compacted body is replaced by its first 600
  bytes, its closing-keyword lines (`Fixes #N` and the like) and the brief;
  evicted history is replaced by the brief's rendering. The request runs at
  medium reasoning effort.
  Initial reviews never use it, and neither do legacy matrix lanes.
- **Cache.** The brief is stored in a hidden bot-authored marker on the posted
  review, keyed by the sha256 of its source text, and reused while the source is
  unchanged. A failed brief is cached the same way, so later rounds on the
  same source skip the request. A marker over 12,000 bytes, or a ledger too large to leave room for
  it, is not written.
- **Routing.** The request uses the `compaction_model` entry in `model_routes`
  (provider pin, service tier) when there is one, and a timeout of
  `OPENROUTER_TIMEOUT_SECONDS` capped at 120 seconds.
- **Failure.** Any error, invalid JSON, or empty brief falls back to the full
  body and the deterministic history. The review proceeds.

Operators see these stderr lines:

```text
history compaction: before=<bytes> after=<bytes> dropped=<section:count,...|0>
review brief: <skipped|reused|built|failed|failed-cached> (source_bytes=<n>, brief_bytes=<n>)
```

`dropped` counts evicted items per section, plus `comments_filtered`. A failed
summary also logs `warning: history summary failed; omitting it`.

The `compaction` input picks one of three modes for re-reviews:

- `auto` (default): deterministic compaction plus the model brief when
  `compaction_model` is set.
- `rules`: deterministic compaction only, no brief. `off` is a deprecated
  alias for `rules` and still works.
- `none`: no compaction and no brief. Every earlier bot review, reply and PR
  comment goes to the prompt verbatim (hidden markers, status comments and the
  bot's own comments still removed) up to 256,000 bytes and logs
  `history compaction: none, verbatim bytes=<n>`. Larger history falls back to
  `auto` and logs
  `history compaction: none, history <bytes> bytes exceeds the 256000 cap; falling back to auto`.

Follow-up reviewers put incomplete fixes in the existing finding's resolution.
Additional evidence can name `prior_finding_id` to retain that open finding's ID;
same-file linked evidence from multiple lanes is published together with every
body retained, only when the original and all additional evidence fit in the
ledger's evidence budget as well as the publishing body limit. Malformed,
unknown, settled, unanchored, cross-file, or oversized links remain separate
visible findings. Distinct defects must not be
linked merely because they share a file. Grouping is not proof of correctness;
the fixing agent still evaluates the evidence.

Agents polling GitHub must paginate **reviews, inline comments, and issue
comments**. A busy review loop can fill the first reviews page before the latest
bot verdict appears; inline replies are read from the separate comments endpoint.
Match the bot author and explicit reviewed commit SHA,
then read every continuation part: large reviews put their remaining findings
in issue comments. A part-one count is not the total finding count. For example,
`gh api --paginate repos/OWNER/REPO/pulls/NUMBER/reviews` reads every review page;
also paginate `pulls/NUMBER/comments` and `issues/NUMBER/comments`.

| Symptom | What to check |
| --- | --- |
| Review jobs are skipped | The examples exclude drafts, fork PRs, and Dependabot PRs. Mark an eligible draft ready to start its initial review. These exclusions need to be accounted for in merge policy. |
| No run appears after an agent pushes | Check event filters and whether the authoring operation used `GITHUB_TOKEN`; see the trigger rules above. |
| `OPENROUTER_API_KEY` is empty / authentication fails | Check the Actions secret name, organization-secret repository access, OpenRouter key validity, credit, and model access. Do not print the secret to debug it. |
| GitHub reports permission denied / resource not accessible | Check `contents: read`, `pull-requests: write`, organization policy, and token identity. Fork and [Dependabot events](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/dependabot-on-actions) have additional token/secret restrictions. |
| Verify says no prior review exists | Finish an initial `full-pr` review first. A failed or partial initial review does not publish authoritative loop state. Check `bot_login` if the posting token changed. |
| Review is partial, or tools cannot inspect files | Read the posted diagnostic. Check full-depth checkout, reviewed commit availability, tool/diff budgets, and whether a new push made the result stale. Do not treat an empty findings list as clean. |
| Green job, but findings remain | `fail_on` defaults to `never`. Choose a finding policy and, if required, the explicit verdict gate above. |
| Cost is higher than expected | Count lanes, judge calls, follow-up rounds, and duplicate installed review workflows. Reported provider cost can be incomplete; compare it with your OpenRouter usage. |

## Inputs

`model_routes` is optional trusted configuration keyed by exact model slugs. It
is separate from the profile registry; use only provider identifiers verified
for your account and action revision.

| Input | Default | Notes |
| --- | --- | --- |
| `role` | `all` | `all` (collect + lanes + optional judge + post) \| `setup` (parse matrix) \| `lane` \| `judge` |
| `models` | `x-ai/grok-4.6` | Comma-separated OpenRouter slugs. Length = lane count. Cap 4. |
| `model_routes` | empty | Trusted JSON keyed by exact review model slug, with optional `provider` and `service_tier` (`default`, `flex`, `priority`). Provider pins disable fallbacks. At most four entries and 8,000 UTF-8 bytes. Unlisted lanes and the judge retain their existing routing. Never interpolate PR content. |
| `judge_model` | `openai/gpt-5.6-luna` | Independent of `models`. Used whenever judging is enabled. |
| `judge_needed` | _empty_ | `true` enables the judge (and forces it for one configured lane); `false` skips it. Empty infers from `models`. A sole survivor from a multi-lane run posts directly. |
| `judge_mode` | `merge` | `merge` union-merges findings with `judge_model`. `pick` lets `decision_model` choose one lane review to post ([pick mode](#pick-mode-judge_mode-pick)). |
| `decision_model` | `baseten/mammoth-1:free` | Pick mode only. Jev-protocol decision model slug. |
| `decision_endpoint` | `systemone` | Pick mode only. OpenRouter API path the decision model answers on. |
| `decision_state_max` | `48000` | Pick mode only. Upper bound on the decision state in UTF-8 bytes (minimum 4,000). |
| `decision_state_max_tokens` | `12000` | Pick mode only. Decision state cap in estimated tokens (minimum 1,000). Over either cap the state is compacted step by step, never cut blind. |
| `github_token` | `${{ github.token }}` | Needs `pull-requests: write` to post the review. |
| `github_timeout_seconds` | `120` | Per-operation `gh` or fallback local-git timeout; 1–600. |
| `job_budget_seconds` | `1320` | Total action budget measured from Python startup. Keep this below the enclosing job timeout so publication and cleanup retain time. |
| `all_role_deadline_seconds` | _empty_ | Optional cap for concurrent lanes in `role=all`. Empty derives the cap from `job_budget_seconds` after judge and publication reserves. |
| `pr_number` | PR that triggered the workflow | Required for `workflow_dispatch`. |
| `fail_on` | `never` | Finding policy: `never` \| `bugs` \| `any`. Operational/schema errors always fail. |
| `roast_level` | `professional` | `professional` \| `playful`. |
| `custom_instructions` | _empty_ | Extra prompt text, max 16,000 UTF-8 bytes. Never put secrets here. |
| `path_profiles` | _empty_ | Caller-owned additive review profiles: JSON `[{name?, paths: [globs], instructions}]`, applied only when a changed path matches (`*`/`?` stay within a path segment, `**` crosses). Sharpen attention; never narrow the review. Trusted workflow config only — never interpolate PR content, never put secrets here. Max 20 profiles and 16,000 UTF-8 bytes. |
| `review_policy` | `off` | `base` loads hierarchical target-branch `REVIEW.md` guidance from full-depth source checkout. Prepared matrix roles consume the setup snapshot. Invalid policy fails before model calls. See the [reference](docs/review-policy.md). |
| `review_profiles` | _empty_ | Version-1 trusted named standard/deep registry. Maximum 32 KiB, 8 profiles, 4 lanes per panel. Cannot combine with `models`, `judge_model`, `judge_needed`, or `effort`. See [trusted review profiles](docs/review-profiles.md). |
| `review_level` | `auto` | `auto` follows policy; `deep` raises the request and requires a deep panel. Retains the ledger; does not grant merge authority. |
| `lane_timeout_seconds` | `1080` | Caller ceiling for prepared profile lanes (1–1800 seconds). The selected profile must fit it and the job budget. |
| `review_context_file` | _empty_ | Setup snapshot path for prepared matrix lane/judge roles; use only the same workflow run/attempt artifact. |
| `review_context_sha256` | _empty_ | Expected setup snapshot digest; required with `review_context_file` and not a substitute for artifact provenance. |
| `review_context_output_file` | _empty_ | Optional destination path for `role=setup` to write `review-context.json` (parent directories are created). Use this when an external orchestrator needs the snapshot outside the action work directory. |
| `status_comments` | `true` | Live status comment on the PR. |
| `max_diff_kb` | `300` | Embedded diff cap. Over-budget diffs go through **diff-budget triage**: generated/vendored/lock-class files (the reviewed commit's `.gitattributes` `linguist-generated`/`linguist-vendored`, lockfile heuristics, large committed JSON snapshots, `generated_paths`) and whole-file deletions demote to stubs first, then the largest hand-written files, so hand-written hunks keep the budget. A stubbed file stays in the embedded diff (header + counts + first-hunk reference), is materialized (unless deleted) into the inert checkout for the tools even past the normal 1 MB cap (up to 8 MB), and still requires a coverage entry — so a fully stubbed-or-embedded diff keeps its real verdict and review-loop continuity. `.gitattributes` is repository content (PR-author-controlled); honoring it only shifts packing priority — a demoted file keeps its stub, coverage obligation, and tool access, which is strictly safer than the raw byte cut it replaces (where tail files vanished entirely). Files dropped entirely, an unparseable diff's raw byte cut, or stubs with tools disabled (`max_tool_turns: 0`) ⇒ `partial`, never clean. |
| `generated_paths` | _empty_ | Extra globs (JSON array of strings) classified as generated/vendored during diff-budget triage. Demotion only shifts packing priority — never excludes a file from review. Trusted workflow config only — never interpolate PR content. Max 8,000 UTF-8 bytes, 200 globs. |
| `review_scope` | `full-pr` | `full-pr` \| `latest-commit` \| `rebase`. Initial rounds require `full-pr`; `rebase` requires an existing ledger and auto/verify mode. Divergent follow-ups select it automatically. |
| `review_mode` | `auto` | `auto` continues an existing ledger on any event, or seeds an initial review when none exists. `initial` explicitly resets history; `verify` requires an existing ledger. |
| `effort` | _empty_ | Optional OpenRouter reasoning effort for **review lanes**. |
| `max_tool_turns` | `50` | Read-only tool rounds against the inert checkout. `0` disables tools. First-pass default matches the sibling Grok `max_turns`. Follow-up jobs may pass `30`. |
| `openrouter_timeout_seconds` | `180` | Connection/header setup and socket inactivity limit per OpenRouter HTTP attempt (1–600 seconds). Active responses are bounded by the remaining lane-stage deadline; without one, this also limits total elapsed time. |
| `lane_index` | `0` | Zero-based matrix index used by `role=lane` artifact naming. Normally supplied by the reusable workflow. |
| `lane_model` | _empty_ | Optional validated model override for `role=lane`. Normally supplied through matrix plumbing. |
| `lane_results_dir` | _empty_ | Lane artifact output/input directory used by `role=lane` and `role=judge`. Normally supplied by orchestration. |
| `head_sha` | _empty_ | Full reviewed commit SHA resolved by reusable-workflow setup. Internal lane/judge plumbing; pull-request runs use the event head by default. |
| `approve` | `off` | `off` or `clean`. `clean` submits a clean review (no open findings) as an APPROVE; anything else stays a COMMENT. The action never requests changes. Needs "Allow GitHub Actions to create and approve pull requests"; if GitHub refuses, the review posts as a COMMENT. |
| `compaction` | `auto` | `auto`, `rules` or `none`; re-reviews only. `auto` compacts history and adds the model brief when `compaction_model` is set. `rules` compacts with no brief. `none` sends history verbatim up to 256,000 bytes and falls back to `auto` above that. `off` is a deprecated alias for `rules`. See "Re-review context compaction". |
| `compaction_model` | _empty_ | OpenRouter slug that writes the re-review brief and summarizes evicted history. Empty skips the brief. Uses its `model_routes` entry if present. |
| `bot_login` | `github-actions[bot]` | Identity the action posts reviews as. Review-loop ledger state is only trusted from this login. Change it when `github_token` is a PAT or App token. |
| `persona` | _empty_ | **Reserved, unused in v1.** Future single-persona runs should skip the judge. |

Prepared matrix consumers require `review_context_file` and `review_context_sha256`
together. When both are supplied to `role: all`, the action loads the frozen
setup snapshot and does not recollect policy or loop state. The digest detects
mismatches only; it is not authenticity.

## Outputs

| Output | Meaning |
| --- | --- |
| `verdict` | `clean` \| `issues` \| `partial` \| `error` |
| `issue_count` | Open findings after this round (carried-over plus new; nits retired by the severity floor excluded) |
| `bug_count` | Open bug-severity findings after this round |
| `round` | Review-loop round number this run performed |
| `review_url` | Posted GitHub review |
| `policy_digest` | Effective guidance SHA-256 when policy is enabled |
| `policy_base_sha` | Immutable target commit supplying guidance |
| `models_json` | Parsed slug array (`setup` / `all`) |
| `matrix` | `[{index, model}, …]` for a GitHub Actions matrix |
| `lane_count` | Number of lanes |
| `judge_needed` | Whether the judge is enabled for the configured lanes |
| `judge_model` | Judge slug used when judging is enabled |
| `lane_file` | Written lane JSON (`role=lane`) |
| `lane_ok` | `true` when `role=lane` produced a valid structured artifact |
| `review_profile` | Resolved trusted profile name |
| `review_level` | Resolved `standard` or `deep` level |
| `review_trigger` | `baseline`, `policy`, or `manual` selection reason |
| `registry_digest` | Digest of the trusted registry and exact model routes |
| `profile_satisfied` | `true` only when required lanes and runtime constraints are satisfied |
| `panel_status` | `complete`, `degraded`, `required_missing`, or `error` |
| `review_context_sha256` | Digest expected by prepared matrix consumers |
| `review_context_artifact` | Prepared context artifact from setup or all-role execution (retention 30 days) |
| `review_receipt_artifact` | Canonical prepared review receipt artifact (retention 30 days) |
| `head_sha` | Immutable reviewed PR head resolved by setup |

## Source evidence and incomplete output

The judge proposes grouping; published text, severity, attribution and anchors are
rebuilt from validated lane findings. Uncertain matches stay separate, which can
produce more duplicate-looking comments than free-form semantic merging. Judge
rewrites cannot downgrade a source bug or substitute unsupported evidence.

Lane overflow retains the strongest severities, records `dropped_findings` in the
artifact, and produces a visible partial review without an authoritative ledger.
Malformed candidates anywhere in the response still fail validation. The separate
union publishing cap remains visible in the judge mode. Reported judge charges
are included even when its answer is unusable; unknown attempted charges are
labeled incomplete.

## Local tests

CI on this repo does not need a live `OPENROUTER_API_KEY`.

For measuring review recall against fixtures with known defects (which DOES
spend tokens and needs `OPENROUTER_API_KEY`), see the offline bench in
[bench/README.md](bench/README.md).

```bash
python3 -m pip install -e '.[dev]'
ruff check src tests bench
pytest
```

## License

[MIT](LICENSE). Copyright (c) 2026 Fly Over Coder.
