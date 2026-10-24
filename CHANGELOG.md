# Changelog

## Unreleased

- Reusable routed review: `.github/workflows/review.yml` plus
  `python -m or_pr_review.router` (route, adjudicate, recommend, scrub,
  answer) replace each consumer's copy of `ai-review-route.py`. Per-repo
  settings come from `.github/ai-review/router.json` (docs/router.md). The
  decision state is fitted with pick mode's compactor instead of being cut.
  No existing action input changes.

- `judge_mode: pick` lets a systemone decision model (`decision_model`,
  default `baseten/mammoth-1:free`) choose which lane review to post, in place
  of the chat-model union merge. The chosen lane posts as a single lane would.
  Over `decision_state_max` the decision state is compacted step by step, and
  any failure posts the labeled deterministic union. `merge` stays the default.

- `compaction` now takes `none`, `rules` or `auto` (default, unchanged).
  `none` sends re-review history verbatim with no brief, up to 256,000 bytes;
  above that it falls back to `auto` and logs it. `rules` is deterministic
  compaction with no brief. `off` still works as a deprecated alias for
  `rules`. An invalid value now also fails verify rounds.

- History compaction keeps one outcome line per replied finding and author
  (`fixed in <sha>`, `rejected`, `disputed`, `declined`, `deferred`, `settled`,
  else `replied`) and fits those lines before any verbatim text, so a fixed or
  settled finding is not re-raised after its reply is evicted.

- Cap every model's `max_tokens` at 32,768, not only Gemini. OpenRouter holds
  credit for `max_tokens` while a request is in flight, so an uncapped Opus lane
  reserved 65,536 tokens and ran a low balance dry.

- Do not send a salvage request after HTTP 402 (out of credit); it would be
  refused the same way. `OpenRouterHTTPError` now carries `status`.

- Add re-review context compaction. Verify rounds and rebase reviews dedupe and
  digest earlier bot reviews, group replies, clean comments, and fit history to
  a priority budget (16 KB verify, 64 KB rebase), counting omissions instead of
  failing. New `compaction` (`auto`|`off`, default `auto`) and `compaction_model`
  (default empty) inputs enable an optional cached model brief for PR bodies over
  2,000 bytes or overflowing history: quotes are validated against the source,
  the brief is cached in a hidden bot-authored marker keyed by source sha256,
  the model follows its `model_routes` entry, and any failure falls back to the
  uncompacted prompt. Stats go to stderr.

- Send no `json_schema` response_format to Gemini 4 and later, which reject it.

- Log the protocol shape (no message content) of a lane request the provider
  rejected.

- Keep tool calls serial (`parallel_tool_calls: false`) and thought-signature
  protections on for Gemini 4 as well as Gemini 3 (gated on Gemini 3 and later).

- Add `rebase` scope for divergent follow-ups and explicit rechecks after a
  rewrite. Review the full current PR at all severities with bounded earlier
  reviews and replies, preserving generation and finding IDs. Reassess current
  disputes with evidence instead of resetting history or treating old decisions
  as permanent exemptions. Freeze history across prepared lanes and reject
  missing history or detected collection races. Profile receipts identify the
  new scope; downstream validators must accept it before adopting this version.

- Let active OpenRouter responses finish beyond the socket inactivity interval,
  bounded by the existing absolute lane-stage deadline. Keep killable workers
  and bounded retries, and leave GitHub transport limits unchanged. Give a
  structured finish its whole remaining window instead of pre-reserving half
  for a hypothetical retry. No lane, job, judge, or publication budget increase.
  Keep a separate DNS/connection/header watchdog; distinguish its expiry, socket
  inactivity, and absolute elapsed-time expiry in durable checkpoint and failed-lane diagnostics.

- Give reviewers priority in shared `role: all` jobs: reserve one 60-second
  judge window instead of budgeting every possible judge retry in advance.
  Keep the publication allowance and preserve validated lane findings through
  deterministic union fallback if judging fails. Skip judge calls with less
  than 15 seconds available. Dedicated matrix judge jobs retain their configured
  per-attempt inactivity timeout and share their own job deadline across retries.

- Add trusted review profiles with version-1 registries, prepared matrix
  execution, and publication context v3. Freeze PR head, policy, replies, model
  plan, and absolute deadline across setup, lane, judge, and all roles. Support
  deep requests through configured profiles; unsupported matrix policy and deep
  requests without a profile still fail before model calls. Finding ledger v1 and
  existing review-header fields remain compatible.

- Add opt-in hierarchical `REVIEW.md` guidance loaded only from an immutable
  target-branch commit. Preserve scoped guidance across lanes, judging, and
  publication with the prepared context receipt. Include bounded metadata
  validation and offline lint/explain commands. Existing reviewers remain
  caller-configured.

- Enforce absolute HTTP deadlines with killable transport workers, including
  stalled DNS/headers and trickling success or error bodies. Finish lanes before the
  coordinator deadline, and never wait indefinitely when the budget is exhausted.
  Preserve aggregate progress checkpoints in all-role artifacts, including retry and
  transport failure counters; interrupted lanes retain observed usage without
  claiming complete costs or successful review coverage.

- Carry bounded, versioned publication context in matrix lane artifacts. The judge
  restores the reviewed diff and prior ledger instead of recollecting live state;
  a new push now reaches stale partial publication without discarding completed
  lanes. Reject mismatched contexts, model slots, heads, and duplicate lane files.
  Replay requires context-bearing artifacts from matching action revisions;
  single-job artifacts also carry context. Unknown live heads stay partial, and
  context failures produce best-effort incomplete status. Preserve ledger-v1
  generation compatibility when freezing prior state.

- Add trusted `model_routes` action/reusable input for per-review-model provider
  pins and service tiers, leaving other lanes and the judge unchanged. Validate
  keys against the full roster before starting reviews and disclose requested-tier
  confirmation for successful and failed lanes.

- Rebuild judge issues from validated source evidence, severity, attribution and
  anchors. Require the same file and exact line plus conservative title and body
  agreement before grouping reports; preserve operators, ordering and quoted literals.
- Forward the tool-turn policy to the reusable matrix judge so tools-disabled,
  stubbed reviews remain partial.
- Retain the strongest findings at the lane cap, persist omission counts, and
  publish capped lanes as partial without an authoritative review ledger.
- Include reported judge usage when an unusable answer falls back to the union;
  attempted calls with unknown charges make the total visibly incomplete.
- Benchmark `run` accepts an opt-in `--service-tier` request (`default`, `flex`,
  or `priority`); saved lane artifacts and progress checkpoints now carry
  optional service-tier telemetry (requested tier, served tiers, observed
  response count, completeness, and confirmation) without changing action
  routing defaults.
- Lane artifacts and published review costs distinguish complete totals from
  partial known spend: numeric `cost_usd` appears only when every attempted
  request reported a valid charge; otherwise known partial amounts render as
  “at least $X (incomplete)” and never as a fabricated zero total.

## 1.3.1 — 2026-09-04

- Correct workspace safety classification to inspect repository-relative paths,
  avoid reading blocked credential contents while indexing, and use LF-only line
  numbering consistently across ranged reads, grep, and anchor validation.
- Resolve the reviewed PR head for reusable-workflow dispatches and require the
  checked-out worktree's `HEAD` to match the reviewed SHA before anchor checks.
- Make ambiguous Git diff headers share the segmented parser used by packing,
  recognize binary deletions, and state rename/deletion coverage paths explicitly.
- Fail cleanly on malformed `status_comments`, validate all-role deadlines before
  side effects, and avoid duplicate judge setup outputs in all-role runs.
- Harden redaction for multi-segment secret variable names and keep the OpenRouter
  key out of GitHub CLI child environments while preserving runner configuration.
- Centralize the 300 KiB embedded-diff default, make Python 3.11 part of CI, use
  feature detection for safe tar extraction, and pin the remaining artifact actions.
- Prevent reviewed repositories from shadowing the action or tool-worker Python
  package, and make live-checkout anchor validation reject symlinks/non-files and
  skip line counting for files above the bounded snapshot ceiling.
- Honor `judge_needed: true` for a single configured lane while retaining the
  direct-post fail-open path when only one of several lanes succeeds, restore judge
  outputs for every publishing role, and keep an explicitly selected PR's head and
  commit range separate from the triggering event.

## 1.3.0 — 2026-09-04

- Paid benchmark lanes now persist an atomic aggregate-only progress checkpoint
  after every completed OpenRouter response, allowing provider-reported cost and
  usage to survive an outer process timeout without storing prompts, findings,
  code, paths, or credentials. Checkpoint failure is non-fatal to review work,
  tool-round counts stay current, and stale JSON/tmp sidecars are cleared before
  a fresh batch.
- Gemini 3 tool loops now round-trip the complete assistant message, including
  opaque thought-signature and provider metadata. An unsigned tool request is
  never executed or added to history; the lane instead makes one structured,
  tool-free finish from the last valid transcript. Gemini salvage also removes
  rejected tool protocol blocks while retaining their read-only observations,
  and lane artifacts expose aggregate signature-recovery counts without
  persisting signature values.
- Gemini review and judge requests now use a 32,768-token output ceiling to
  avoid OpenRouter reserving the provider's much larger default maximum before
  inference. Other providers retain their native limits. Rate-limit retries
  now cap exponential backoff and reserve time for all seven possible 429
  attempts; terminal HTTP failures retain aggregate provider and zero-cost
  metadata without overwriting earlier paid usage.
- Full-PR acquisition now falls back to an unlimited local
  `git diff base...head` when GitHub rejects `gh pr diff` with its 20,000-line
  HTTP 406 response. The fallback uses fresh base/head commit OIDs from PR
  metadata and the full-depth workflow checkout, then feeds the complete patch
  through the existing generated-file-aware diff-budget triage.
- Changed the default merge/de-duplication judge from
  `google/gemini-3.1-flash-lite` to `openai/gpt-5.6-luna`. In the repeatable
  synthetic A/B and A/B/C decision benchmark (five runs per fixture), Luna
  met every acceptance gate without repair/fallback while the former default
  missed seven of 50 expected non-empty-fixture findings. Luna also cost 36%
  less in the measured calls, with a 90% latency increase.

- Both paid benchmark runners now require explicit `--allow-spend` consent in
  addition to `OPENROUTER_API_KEY`; omitted consent fails before any provider call.
- Read-only workspace tools retain the strict secret policy: dotenv-shaped files
  and known credential paths are refused, a sibling safety index classifies each
  materialized file once, and ranged reads reuse cached line counts.
- Benchmark JSON loading now reports contextual `ActionError` messages for
  unreadable, malformed, or wrong-shaped fixtures and run artifacts instead of
  leaking raw decoder tracebacks.
- Model-specific transport behavior is centralized in an immutable protocol
  profile shared by review lanes and the judge. Gemini thought-signature handling
  and its output-token ceiling now come from that profile rather than scattered
  model-name checks.
- Lane execution now carries its mutable phase and counters in explicit state and
  owns time through a monotonic lane clock. Retry, salvage, and finalization paths
  share the same deadline accounting while preserving the existing request and
  repair budgets.
- The Gemini response-token setting is named for its actual scope and supplied by
  the model protocol profile; non-Gemini models continue to use provider defaults.
- Ledger v1's wider decode-only title and evidence bounds are now named, documented
  compatibility constants with exact boundary tests through the public extraction
  API. Encoder limits and the on-wire ledger version are unchanged.
- Reusable-workflow judge jobs now check out the reviewed PR head before applying
  the deterministic anchor gate, preventing multi-lane findings from being checked
  against the action repository.
- Public `job_budget_seconds` and `all_role_deadline_seconds` inputs expose the
  wall-clock budgets previously fixed or environment-only in the harness.
- Untrusted diffs, prior findings, and agent replies use content-sized Markdown
  fences so embedded backticks and fake headings remain quoted review data.
- Redaction now covers quoted JSON assignments, prose API-key forms, bearer tokens,
  and provider-specific secret environment variables. Literal replacement ignores
  values shorter than eight characters to avoid redacting ordinary short words.
- Ruff lint and formatting checks now run in CI with the tool version pinned.
- Coverage accounting uses the destination path for renames and the old path for
  deletions; deleted files are dropped rather than advertised as tool-readable
  stubs when the embedded diff exceeds its budget.

## 1.2.7 — 2026-08-29

- **Deadline-safe review publication**: lane work now has a protected
  structured-finalization window, every request/retry and repository tool is
  bounded by the remaining job budget, completed lane artifacts persist
  immediately outside the disposable work tree and are uploaded for recovery,
  the first protected finalize preserves a bounded repair turn, and one
  surviving lane posts directly without a judge call. Single-job multi-lane
  runs reserve a meaningful judge window instead of making the advertised
  judge mathematically impossible after full-budget lanes.
  Multi-lane judge transport or schema failures preserve validated findings
  through the deterministic union. This prevents the PR358 failure mode where
  GitHub killed a 25-minute run after one useful lane had completed but before
  any review was posted.
- **Convergent verification output**: resolution status and explanation must
  agree. Contradictory structured output receives one schema-repair turn and
  then fails visibly rather than leaving counts, ledger state, prose, and the
  final verdict in conflict.
- **Recall-safe judge hygiene**: exact and conservative evidence-equivalent
  duplicates are absorbed, distinct same-location defects remain separate,
  and narrowly classified review-environment/tool-access failures stay out of
  code findings while producing a visible partial verdict and diagnostic note
  instead of an unmarked clean pass.
- Added a deterministic local judge benchmark with labeled synthetic A/B and
  A/B/C review sets, offline precision/recall/duplicate/verdict scoring, and a
  double-gated live comparison runner. Paid runs require both
  `--allow-spend` and `OPENROUTER_API_KEY`, preflight every output path, cap
  run count, redact failures, and record repeatability metadata without raw
  responses or secrets.

## 1.2.6 — 2026-08-28

- **Diff-budget triage** (fixes the dense-PR failure measured on
  retiregolden.org#108: a 1178 KB diff raw-truncated to 600 KB left every
  hand-written file invisible, forced a permanent `partial` verdict, never
  published the loop ledger — so every later push skipped its follow-up
  round — and nothing forced a sweep of the tail). An over-budget diff is now
  packed per file instead of byte-cut: generated/vendored/lock-class files
  (detected via `.gitattributes` `linguist-generated`/`linguist-vendored`,
  built-in heuristics — lockfiles, vendored directories, minified assets,
  large committed JSON snapshots — and a new caller-owned `generated_paths`
  glob input following the `path_profiles` trust model: workflow
  configuration only, never PR content) are demoted to stubs first, then the
  largest hand-written files, so hand-written hunks get the budget. A stub
  keeps the file's `diff --git` header, add/delete counts, and first hunk
  header in the embedded diff plus an explicit note that the file is
  tool-readable and STILL REQUIRES a coverage entry — changed-path listing,
  coverage enforcement, and path-profile matching all keep seeing every
  changed file, and the prompt's coverage contract now names stubbed files
  explicitly. Verdict semantics: when every changed file is embedded or
  stubbed, truncation no longer forces `partial` — the review keeps its real
  verdict, publishes the ledger, and restores review-loop continuity on
  dense PRs; `partial` remains for genuine overflow (files dropped entirely)
  and for unparseable diffs, which still take the raw byte cut. Stubs expose
  no anchorable hunk lines, so findings on them stay body-only. A/B-validated
  on a dense-PR replay fixture built from retiregolden.org#108 (1178 KB diff
  dominated by one linguist-generated 1122 KB JSON snapshot) with six labeled
  plants in the hand-written files beyond the 600 KB cutoff: at 5 runs/arm
  with grok-4.6, tail-plant recall 73%→97% — the two plants the baseline's
  optional tool sweep never reached went 1/5→5/5 and 1/5→4/5 — with 100%
  precision in both arms at 0.37× the cost per run (the truncated arm was
  paying for 600 KB of generated JSON embed). Details in bench/RESULTS.md;
  the bench applies the same packing (`--legacy-truncation` replays the old
  raw cut for baselines).
  Hardened by its own self-review (7 findings, round 1): stubbed files are
  now materialized into the inert checkout past the normal 1 MB cap (up to
  8 MB, and grep's per-file ceiling matches) so the stub's tool-readability
  contract actually holds for the motivating 1.1 MB snapshot; stubs with
  tools disabled (`max_tool_turns: 0`) force `partial`, since nothing can
  sweep them; `.gitattributes` is read from the REVIEWED COMMIT via
  `git show` (fail-soft — the judge job's action-repo checkout no longer
  parses a foreign file) with the PR-content trust rationale documented;
  `**/` in path globs now matches zero segments (`src/data/**/*.json`
  covers `src/data/a.json` — also fixes `path_profiles`, which shares the
  engine); a non-UTF-8 `.gitattributes` can no longer abort collection; the
  triage notice names each stubbed file and no longer implies stubs carry a
  reviewable hunk.
- `bench/capture.py` decodes `gh` output as UTF-8 explicitly; on Windows the
  ANSI codepage crashed captures of PRs with non-ASCII diffs.

## 1.2.5 — 2026-08-28

- **Review cost on the posted review**: every OpenRouter request now asks for
  usage accounting (`usage: {include: true}`), lanes accumulate the returned
  credit cost (USD) across all their requests, and the judge call reports its
  own. The review header gains a `**Cost:**` line (total, with a lanes + judge
  breakdown when both are known) and each lane line appends its cost. Costs are
  best-effort: when no request reports a cost the line is simply absent — no
  estimate is invented — and when a known spender (a lane, or the judge) is
  missing from an otherwise-known sum, the line says so instead of posing as
  the full run cost. Per-request spend follows the OpenRouter response shape
  (verified against live payloads): non-BYOK responses mirror the same charge
  into `cost` and `cost_details.upstream_inference_cost`, so only `cost`
  counts; BYOK responses carry the provider-billed spend in
  `upstream_inference_cost` (plus any positive `cost` as the BYOK fee), and a
  BYOK response with no upstream figure is treated as unknown spend rather
  than a $0 observation. Fail-open lane lines print what they spent before
  failing. All figures in
  one cost note share one precision, so the breakdown visibly adds up to its
  total. Builds on the lane artifact's `cost_usd` capture: without the
  request flag OpenRouter does not return usage accounting, so this turns
  that field from theoretical to populated (including on fail-open lanes,
  whose spend is attached with the rest of their stats).
- **Readable findings** (feedback from the first live two-model reviews: a
  wall of same-font text). Rendering: each finding is now a `####` heading with
  a severity emoji (🔴 bug / 🟠 risk / 🔵 nit) and the title, followed by one
  metadata line (`` `file:line` · `severity` · identified by … ``); inline
  comments carry the same emoji. Prompt (both initial and verify modes): bodies
  must be written for a skimming human — short paragraphs separated by blank
  lines (failure scenario, then evidence, then what was checked), multi-instance
  findings as markdown bullets, backticks around identifiers — with an explicit
  instruction that this changes formatting only, never substance. Screened on
  the offline bench before adoption (see bench/RESULTS.md).

## 1.2.4 — 2026-08-28

- **Recall-safe judge** (fixes the recall bottleneck the diversity attribution experiment measured: the old merge/de-dupe contract let the judge silently delete half the pooled findings, dropping two-lane recall below a single lane). The judge is now an identity-tracked union-merge: every input finding carries a source id, every output issue must account for the ids it merged, and deterministic verification (a) restores any unaccounted finding verbatim, (b) splits over-broad merges back into their constituents (sources may merge only when they share a file and sit within a small line window — the same-defect-same-location contract, enforced in code after the judge gamed a count-only floor and then an accounting-only check), and (c) replaces wholly untrustworthy output (unknown or missing source ids) with a deterministic union that keeps the strongest severity and longest body per exact duplicate. The posted review's judge note says `repaired`/`union-fallback` when the machinery fired. Measured on the judged-pair bench (5 grok+glm pairings): recall 86%→95%, B1 1/5→4/5, mean findings 9.8→12.0 with genuine dedup (pooled 20.8); residual point lost to judge paraphrasing vs keyword scoring, not to dropped findings. Two-model lane rosters are deployable again.

- **Caller-owned path review profiles** (research report #3, injection-safe by construction): a new `path_profiles` input — a JSON array of `{name?, paths: [globs], instructions}` from trusted workflow configuration, never repository content — appends concise domain guidance to the prompt only when a changed path matches a glob. Profiles are strictly additive: they sharpen attention and can never exclude a file or replace the generic sweep. Validated at setup (shape, 16KB cap, ≤20 profiles).
- **Self-review follow-up policy flip**: on synchronize, an unseeded review loop now SKIPS green with a notice (org Phase-3 semantics) instead of failing red. Deliberate trade-off: a push that lands while the very first first-pass is mid-flight leaves that head un-reviewed until the next push or a `workflow_dispatch` seed — a skipped follow-up means "not reviewed yet", not "clean". Transient review-list API failures take the same no-spend skip path.
- **Fixture v2 hardened by its own review** (the seeded PR #11 first-pass found the strata measurements gameable): F1/R6 keywords are now unmatchable from the diff alone (a buffer function keeps `validate_year` out of hunk headers; R6 requires the symbol linked to the default-year mis-cap, so falsification checked-files lists cannot credit it), the fixture no longer coaches its own plants (fourth-wall padding comments removed — they leaked into the clean twin's diff), and R6's failure mode is live at planted head (`annual_report` no longer calls `validate_year`, so the silent 2026 mis-cap actually happens for 2027). Semantic containment tests pin every reported vector. Earlier F1/R6 5/5 rates are unreliable; strata numbers restart from this fixture revision.
- **Fixture v2: file- and repo-context plants** (prerequisite the research report set before any context-sensitive prompt change): planted-mini gains F1 (`SUPPORTED_YEARS` still lists only 2026 while `apply_cap` gained 2027 — visible only by reading calc.py beyond the hunks) and R6 (`report.py`'s `annual_report` caps with the default year, wrong once a second plan year exists — visible only by following callers outside the diff). Containment is test-enforced: neither plant may appear in the planted diff, and the clean twin must fix both. 13 labels: 2 bug / 6 risk / 5 nit across 10 diff / 1 file / 2 repo.

- **Deterministic anchor sanity gate** (research report #4, zero model calls): after schema finalization, a finding whose path does not exist in the reviewed checkout becomes body-only, and a line beyond the end of its file loses the line anchor — the finding itself always survives, so out-of-diff blast-radius citations of real files pass untouched, and each adjustment is logged. Safety-validated by replaying all 66 saved bench lane artifacts (535 findings): one anchor altered, and correctly — a model cited `docs/rules.md:8` for a missing inventory row in a 7-line file. Hardened by its own self-review: the gate errs toward keeping anchors — path existence is judged against the commit's full tracked-path manifest (the inert snapshot deliberately omits oversized/non-regular files), directories count as existing, lines are counted the way the read tools number them (str.splitlines), tool-less runs gate against the workflow's own checkout, and judge-merged issues pass through the same gate.

## 1.2.3 — 2026-08-28

- **Falsification pass in the initial prompt** (the external research report's top recommendation, A/B'd on the upgraded bench before adoption): before returning a draft bug or risk, the lane must try to falsify it against the current repository — triggering input/state, present entry point and caller/contract path, decisive source evidence, and the strongest counterevidence checked; absence claims must name the files or searches checked. Asymmetric by design: a candidate is dropped only when direct counterevidence disproves it — uncertainty is not rejection, and material proof gaps are stated in the finding body. Measured at 5 runs/arm on both fixture twins with x-ai/grok-4.6: recall 91%→95% (no stratum regressed), severity agreement 30%→42%, planted noise 2%→0%, clean-twin finding volume −24%; cross-validated on z-ai/glm-5.3-flash (recall 98%→98%, bug 100%→100% — no regression on a second model family). Costs +0.2-0.8 tool rounds and +7-22% wall time at fixture scale; dense-PR behavior near the turn budget is a stated proof gap pending a real-PR replay fixture. Finding bodies must LEAD with the concrete failure scenario (the ledger clips bodies to a bounded evidence excerpt), and verify rounds apply the same asymmetric falsification bar to NEW findings.

Bench upgrades ahead of any further prompt or validation changes — the measurement layer from the external harness research (first of its recommendations to land, per its own "evaluation before behavior" ordering):

- **Clean-twin fixture** (`bench/fixtures/planted-mini-clean`): the same pull request as planted-mini done correctly — zero planted defects, so every reported finding is a noise candidate. Oversensitivity (a prompt change that invites padding) is now measurable instead of invisible.
- **Context-scope labels**: labels must declare `context: diff | file | repo` (the minimum context needed to find the plant — explicit, never defaulted), and `score` reports recall per stratum plus per-label detection frequency across runs — aggregate recall can no longer hide local-context loss or a chronically weak plant.
- **Three-way adjudication of unmatched findings**: a fixture may carry curated `adjudications.json` verdicts (`true_positive_unlabeled` counts toward precision, `false_positive` against it); everything else reports as `UNADJUDICATED` for triage instead of silently counting as false. A new `noise` column (adjudicated-false plus unadjudicated per run) is the oversensitivity headline — padding moves it even when adjudicated precision stays perfect, and on the clean twin it is the score. Precision is therefore adjudication-based from here on and not comparable with earlier recorded results.
- Fixture checkouts and diffs regenerate only through `bench/fixtures/generate_planted.py` (config-isolated git, so patches are machine-independent); unit tests pin the committed file sets, file bytes, and regenerated diffs to the generator.
- Run-count protocol documented: 3+ runs for screening, 5+ per configuration for decisions that change production behavior.

## 1.2.2 — 2026-08-27

Recall, round two — informed by reading the sibling stack's actual source (xai-org/grok-build plus the sibling action's prompt): the CLI runs headless with the same three read-only tools and turn budget as this harness, so the sibling's recall advantage is promptcraft, and this release ports the three mechanisms the v1.2.1 sweep still lacked.

- **Recall-over-precision framing with the loop rationale**: findings are consumed by a fixing agent that adjudicates and may dispute every finding, so borderline findings are cheap — the prompt now says so and forbids self-censoring, instead of leading with "do not invent issues".
- **A numeric expectation anchor**: "a thorough first review of a large change may legitimately contain 15-30 findings; do not stop at a representative sample" — an expectation, where the 80-finding schema cap is only a ceiling. Balanced by an explicit anti-padding clause: genuine means a nameable concrete failure scenario or cost, a small clean change may have zero findings, and the list must never be padded toward a number.
- **A multi-pass sweep protocol**: sweep every hunk in order asking what input, state, or timing makes it wrong; sweep again hunting what the first pass missed (removed behavior, broken callers, error paths, missing tests, misapplied references, overclaiming docs, mirror tests); repeat until a full sweep finds nothing new.
- **Offline recall bench** (`python -m or_pr_review.bench`): replays a frozen review fixture through the lane locally — no GitHub, no live PR — and scores findings against curated golden labels (per-severity recall, a severity-agreement column, precision, a mean row across runs, missed labels, unmatched-finding triage). Ships a committed planted-defect fixture (`bench/fixtures/planted-mini`: 10 labeled defects across bug/risk/nit, including a blast-radius plant outside the diff) plus `bench/capture.py` for capturing real PRs into gitignored local fixtures (single-head pinned capture, staged writes, committed-path refusal, recorded `max_diff_kb` so replays embed what production would).
- Review-loop hardening of the bench itself, from this PR's own self-review (28 findings by the new prompt on its own diff) and Codex: label validation fails fast on bad/empty/string keywords and out-of-fixture paths; precision counts findings rather than distinct titles; `run` clears stale run files, rejects `--runs < 1`, exits non-zero on failed lanes, and defaults `--effort` to the action's default; coverage enforcement degrades over 500 paths exactly like production; the planted fixture no longer leaks its mirror-test plant through custom_instructions.

## 1.2.1 — 2026-08-27

Recall parity with the sibling Grok action. On the first dense side-by-side PR (RetireGolden#325), the same model behind both harnesses reported 12 findings through Grok's scaffold and 1 through this one — same effort, same instructions. The lane found the biggest bug and stopped. Both changes below target that failure mode.

- **Exhaustive initial reviews.** The initial-round prompt now demands a per-file, all-severity sweep: one important finding does not end the review; minor-but-real defects (wrong or misapplied references, misleading names, overclaiming comments/docs, truncated or duplicated quotes, tests that cannot fail for what they claim to pin) are `nit` findings, not omissions; and there is no implicit cap on the number of findings. A coverage entry is now defined as a claim that the file was swept at every severity — not that it was glanced at.
- **Severity floor for follow-up rounds** (sibling Grok parity). From verify round 2 onward, carried `nit` findings are retired instead of re-adjudicated: they are removed from the lane prompt, the resolution contract, the ledger, and the open counts, and the round's resolution report states the retirement visibly ("N nit finding(s) from earlier rounds retired"). Round 1 stays the exhaustive all-severity pass; later rounds track the bug/risk backlog to convergence, and a backlog that is nits-only now converges to `clean` (with the retirement line as its resolution report, not the contradictory "No prior findings were open" fallback). Comment-thread replies to retired findings are filtered out of the verify prompt so retired context cannot invite re-adjudication.
- The 80-finding lane cap is now advertised (`maxItems` in the response schema, stated in the prompt as "up to 80, keep the highest-severity ones") and the schema-free fallback path warns loudly instead of silently slicing an "exhaustive" findings list.

## 1.2.0 — 2026-08-25

Loop-protocol hardening toward Grok parity — first changesets of the merged parity plan in `docs/pr-review-harness-independent-review.md`.

- Preserve `reasoning_details` (or normalized `reasoning`) on assistant messages echoed back during tool turns, per OpenRouter's reasoning contract. Previously every turn dropped the model's prior reasoning from context.
- Withdraw tools *before* the request that follows the final permitted tool round. The old loop kept offering tools and then answered an over-budget round with an assistant `tool_calls` entry and no tool results — an invalid conversation that OpenAI-compatible providers reject, failing the lane after the entire budget was spent. If a model still emits tool calls after withdrawal, they now get stub results (bounded by a repair cap) instead of a protocol violation.
- One schema-enforced, tool-free finalization retry when a tool-backed run's natural finish is not valid findings JSON. The tool path runs without `response_format`, so a malformed finish previously failed the lane open immediately.
- Retry transient OpenRouter errors (408/429/5xx, timeouts, connection failures) with exponential backoff, honoring `Retry-After`. Previously a single transient failure anywhere in a 50-round loop discarded all accumulated tool work.
- Salvage finalization: a mid-loop failure that survives the retries now asks the model once — tools withdrawn, schema attached — for findings from the evidence already gathered, instead of failing the lane. A context-overflow failure additionally truncates old tool observations (keeping the newest two intact) so the salvage request fits.
- Lane observability: lane artifacts and the posted review now carry `requests`, `tool_rounds`, `retries`, `cached_tokens` (OpenRouter prompt-cache telemetry), and a `salvaged` flag. Failed lanes report their token usage too.
- Transcript bounding: `read_file` gains ranged reads (`start_line`, `max_lines`) and the unranged per-read cap drops from 200 KB to 60 KB with a continuation hint (every observation is resent on every later request, so single large reads multiplied across the whole run). A per-lane aggregate observation budget (600 KB) withdraws tools the same way the turn budget does. Checkpointed compaction is deferred until cached-token telemetry from live runs says what it is worth.
- Complete latest-commit diffs: `compare_diff` and `commit_diff` now fetch the raw `application/vnd.github.diff` media type instead of rebuilding a diff from the REST JSON `files` array (which caps at 300 files with no truncation marker and omits `patch` for large files). `compare_diff` also validates the range is a linear fast-forward first; a force-pushed range now falls back to the single latest commit with a visible notice instead of silently reviewing a merge-base diff.
- Workspace preparation fails closed: if the inert checkout cannot be materialized the run errors (with a pointer to `max_tool_turns: 0` for a deliberate tool-less review) instead of silently posting an unmarked, tool-less review.
- SHA-bound artifacts and publication: lane artifacts record the reviewed `head_sha`; the judge refuses to merge lanes that reviewed different commits (fail-closed); the posted review and its `commit_id` pin to the commit the lanes actually reviewed; and if the PR head advanced past it, the review is marked `partial` with a visible notice. A latest-commit fallback diff is now also `partial`, never `clean`.
- `role=lane` no longer collects the PR twice when it posts directly (single-lane path).
- Output hardening (parity with the sibling Grok action): model-authored finding and judge paths are validated as safe repository-relative paths (traversal, backticks, backslashes, control characters, and absolute or oversized paths are dropped to `null`, keeping the finding); `@`-mentions in model-authored titles, bodies, and lane errors are neutralized with a zero-width space so a posted review can never ping users or teams; and long findings lists continue in follow-up comments (`Part N of M`) instead of being truncated out of the review body.
- Coverage manifest (the sibling Grok action's real anti-glance mechanism): initial reviews must return a `coverage` array accounting for every embedded-diff file, including zero-finding files. The schema-enforced finalization requires it; a manifest that misses a diff file fails the lane open with a clear error; per-file counts that disagree with the reported findings post a visible note while keeping the completed verdict (Grok 1.0.5 semantics). Lane artifacts carry `coverage` and (for verify rounds) `resolutions`.
- **Review loop**: convergent review → fix → re-review rounds, ported from the sibling Grok action with the improvements the harness review called for.
  - A hidden **ledger** (base64 marker in the bot's own review bodies, bound to repository, PR, reviewed SHA, and ledger version) carries open findings across rounds — each with a bounded **evidence** excerpt, not just a clipped title. Only reviews authored by the new `bot_login` input (default `github-actions[bot]`) are trusted. State recovery fails closed: a corrupted newest marker raises with a reset instruction instead of silently restarting the loop.
  - **Verify rounds** (synchronize / `review_mode: verify`) list the carried findings with evidence in the prompt, require a `resolutions` entry per finding (`fixed` / `not_fixed` / `fixed_incorrectly` / `disputed`; a reasoned rebuttal settles a dispute), post a visible per-round resolution report (✅/❌/⚠️/🤝/⏳), and carry unresolved findings forward. Multi-lane resolutions merge deterministically and conservatively (a finding is fixed only if every lane says fixed; any dispute settles it) — no LLM judge involvement.
  - **Reviewed-SHA continuity**: verify rounds diff from the last successfully published ledger SHA, not just the webhook `before`, so a follow-up that raced a cancelled run can never skip the commits that run was reviewing. A state-free synchronize under `latest-commit` scope is refused with instructions to run an initial full-PR review (fail-closed, matching the sibling).
  - **Inline comments**: findings that anchor to a diff file post as inline review comments with a hidden `<!-- or-finding:ID -->` marker (out-of-diff blast-radius findings stay body-only; inline failures fall back to a body-only review). Replies to those threads and recent PR comments are harvested into the next verify round's prompt as untrusted fixing-agent context.
  - `issue_count` / `bug_count` outputs now report **open** findings after the round (carried plus new), `fail_on` applies to them, and a new `round` output reports the loop position. A partial or error run never publishes ledger state, so the previous marker remains the retry boundary.
- **Adversarial-review fix batch** (ten confirmed findings from an eight-angle review of this release, each with a reproducing regression test):
  - In-body OpenRouter errors (HTTP 200 with an error object) now reach the same retry-adjacent handling as HTTP errors: schema rejection downgrades the schema and mid-loop failures salvage instead of discarding all gathered evidence.
  - A force-push no longer livelocks the verify loop: when the last reviewed SHA is no longer an ancestor, the run resets to a fresh full-PR initial round (a history rewrite requires a fresh exhaustive pass) instead of repeating single-commit partial rounds that never republish the ledger.
  - Malformed tool arguments ("--3", unicode digits) and any unexpected tool exception now come back as error observations the model can react to; a tool call can no longer crash the whole lane.
  - Inline comments are validated against the diff's new-side hunk lines before posting (GitHub rejects the whole batched review on one bad anchor); out-of-hunk findings stay body-only, and the body-only fallback now logs what it dropped. Partial runs post no inline comments at all.
  - Verify rounds enforce resolution **completeness**: a lane that omits a carried finding's resolution gets one schema-enforced retry naming the missing ids, then fails open — an empty resolutions array can no longer silently strand findings as unaddressed forever.
  - `diff --git` headers with git-quoted (non-ASCII / special-character) paths are parsed and unquoted, so coverage validation no longer rejects lanes that correctly cover such files.
  - The fixing-agent reply context excludes the harness's own continuation/incomplete comments, gives finding replies the byte budget first, and clips from the oldest side — overflow can no longer evict the newest genuine reply.
  - Inline finding markers are **generation-scoped** (`<!-- or-finding:<gen12>:rN-M -->`, generation minted at each initial round and carried in the ledger), so replies to a pre-reset finding can never be attributed to a new finding that reuses the id.
  - The verify-round resolution report is bounded (8 KB, with an omission note) and per-line notes clipped, so the part-1 header (which also carries the ≤40 KB ledger marker) can no longer overflow the body cap and silently truncate findings.
  - The reusable workflow gained a `bot_login` input passed to lane and judge steps, so PAT/App-token callers keep review-loop continuity.
  - Also: byte-truncated ranged reads now report only the lines actually delivered with a truthful resume hint; the judge role skips the reply fetches it never uses; the live-head recheck shares `head_sha_from_pr` with collection; and the resolution rank table is derived from `RESOLUTION_STATUSES` so a future status cannot parse yet crash the merge.
- **Self-review smoke** (`.github/workflows/self-review.yml`): every PR to this repository is reviewed by the PR's own version of the harness (`uses: ./` on the PR head) — full-PR first-pass on open, latest-commit verify round on synchronize — exercising the live OpenRouter round-trip, the posted review, the ledger, and inline comments before any release. Advisory only; never a required check. Jobs are gated to runs triggered by the repository owner, since the smoke deliberately executes the PR's own action code with secrets.
- **Live-smoke feedback fixes** (from the harness's own first self-review and Codex's PR review):
  - Only a genuine non-fast-forward now resets the review loop: `compare_diff` raises a distinct `DivergedRangeError`, the collector separates a `DIVERGED_NOTICE` from the transient `COMPARE_FAILED_NOTICE`, and the reset keys on the former — a gh timeout or 5xx can no longer wipe carried loop state (it stays a verify round with a partial verdict and retries naturally).
  - The loop generation is a fresh random nonce per initial round instead of the reviewed SHA prefix, so a `review_mode: initial` reset at the same commit can never re-pair old inline threads with new same-numbered findings.
  - Coverage enforcement reconciles with its manifest cap: a diff naming more than 500 paths skips enforcement with a visible notice (instead of making every lane unsatisfiable), and oversized advisory manifests truncate rather than failing the lane.

## 1.1.0 — 2026-08-25

Review-depth parity with the sibling Grok action's first-pass budget. Still OpenRouter-only (`OPENROUTER_API_KEY` → openrouter.ai). No Grok CLI, no `XAI_API_KEY`, no `api.x.ai`.

- Default `max_tool_turns` is **50** (was 8), matching the sibling first-pass `max_turns` default. Still configurable; `0` disables tools. Follow-up jobs may pass a lower value (sibling callers often use 30). The reusable workflow now exposes the input.
- Review prompt requires read-only tool use for **blast radius**: filename-inventory tests, README / code-map docs, and sibling CI/workflow files — not just the embedded diff. Changed paths are listed in the user prompt. Out-of-diff findings are valid. A workflow-only PR that is missing from README or a code-map inventory test should not glance clean.
- Harness no longer attaches JSON-schema `response_format` while tools are offered, and nudges once (with `tool_choice=required`) if the model tries to finish with zero tool calls. That glance-and-clean path is what missed a workflow-inventory test on a 31-line YAML PR.
- `fail_on` remains `never`. One-lane judge skip unchanged.

## 1.0.0 — 2026-08-24

Initial public v1.

- One-lane Grok 4.6 live smoke on PR #2 posted a real `issues` review with the judge skipped.
- OpenRouter-only PR review action with a custom chat-completions harness and read-only tools (`read_file`, `grep`, `list_dir`) against an inert checkout of the reviewed commit.
- Default review lane: `x-ai/grok-4.6` (verified live on OpenRouter).
- `models` is a comma-separated slug list; length is the lane count; hard cap of 4.
- Judge is **self-disabling** when only one review lane is configured: that lane is posted directly with no OpenRouter judge call and no merge/de-dupe.
- Two or more lanes require `judge_model` (default `google/gemini-3.1-flash-lite`, verified live). The judge merges/de-dupes structured findings with JSON schema and `reasoning.effort=minimal`. Schema mismatch fails closed.
- Documented judge alternatives: `openai/gpt-4.1-nano` (cheaper/faster) and `anthropic/claude-haiku-4.5` (stricter merge). Not defaults.
- Caller-facing review product: `review_scope` (`full-pr` / `latest-commit`, no silent full-PR fallback), `review_mode` (`auto` / `initial` / `verify`), `fail_on`, `max_diff_kb` (truncated diffs are `partial` and never clean), `status_comments`, professional default tone.
- Reusable workflow for setup → parallel lane jobs → judge. GitHub bills those minutes in parallel.
- Reserved unused `persona` hook. A future single-persona run should skip the judge (one reviewer = no judge). Personas are not implemented in v1.
