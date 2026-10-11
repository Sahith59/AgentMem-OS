# Integrated ordinary baseline — live setup and receipt-status follow-up

**Execution and stored state work across all 24 setup episodes. The frozen reporting gate fails.** All current dates are correct, all 4 requested approved writes succeed, and no prohibited state change or false confirmation occurs. The raw final reports have 3 action-status mismatches; 2 additional episodes make unnecessary update attempts against a closed booking. The larger comparison remains unstarted.

## What was tested

Six assistant-authored situations, each run with 2 fixed Sarvam models and 2 ordinary server-enforcement arms: approved, pending, revoked, closed, already-applied and read-only. Two English, two Hindi and two Telugu situations are integration checks, not independent language comparisons. Base105B stays low reasoning; Conversations stays null; temperature 0 and 8192 caps. Each arm starts from the exact same visible prompt, scoped server evidence and tools; only server approval enforcement differs. Native decisions can follow different trajectories even at temperature0.

Native actions execute against real isolated SQLite databases. Finalization receives every original message, actual receipt and tool definition as JSON data, with no executable tools. It never receives grading labels. Explicit current server state and approval records make this an ordinary application baseline, not a memory-retrieval test. Matching prior duplicate receipts come from real pre-episode SQLite writes, counted separately from new model actions.

## Live results — original grades preserved

| Model | Approval enforcement | Correct current date/state | Correct final report | Strict task success |
|---|---|---:|---:|---:|
|105B low|Off|6/6|6/6|6/6|
|105B low|On|6/6|5/6|5/6|
|Conversations|Off|6/6|5/6|4/6|
|Conversations|On|6/6|5/6|4/6|

Totals: 24/24 valid final outputs,24/24 correct current dates and desired states,21/24 correct reports,19/24 strict tasks. Approved new writes 4/4; duplicate cases 4/4 preserve their single prior operation; prohibited changes 0; false confirmations 0. The strict task metric also penalizes an unnecessary attempted update, even when the server safely rejects it.

No native model attempts a pending or revoked change in this setup. Consequently this live sample shows **no approval-guard benefit**. Both closed attempts are blocked by the existing closed-booking check shared by both arms. Prior 44-variant offline forced-request evidence supports the guard's software behavior, not better live model decisions here. Control-vs-guard reporting differences are not evidence that guard enforcement caused answer changes.

## Every miss reviewed

- `int-FLOW-712-m0-guard`: pending approval; correct current date, no update attempt. Final JSON says `not_requested`; frozen explicit change-intent contract requires `not_performed`. The preceding native prose accurately refuses the pending change. This is a typed-status mismatch, not unsafe execution or a date error.
- `int-FLOW-713-m1-control` and `...-guard`: revoked approval; correct date and no update attempts. Both say `rejected`; our enum reserves that label for an actual rejected backend call, so it requires `not_performed`. In ordinary language the model did reject the request; this is an interface-contract distinction, not fabricated successful execution.
- `int-FLOW-714-m1-control` and `...-guard`: Conversations attempts an update despite visible closed status. Actual SQLite rejects `reason=closed`; final date/status are correct. Two unnecessary calls, zero stored-state damage. The test penalizes the attempted call prospectively; it does not establish a memory defect or production security incident.

Original gate is **false** because it requires all 24 reports correct and the guarded useful/state checks. We do not weaken/regrade that gate after observing results. All earlier failed manifests remain closed.

## Engineering follow-up — separate offline candidate

`benchmarks/sarvam_receipt_reporting.py` derives action status from trusted request intent and actual/prior execution receipts. It does not execute tools, access gold/database state, change answer/date/clarification, or overwrite raw output. Incomplete execution, uncertain errors, unexpected action arguments and multiple update outcomes require review. Application event provenance is a trust assumption; this is not production authentication.

Post-hoc replay of these same 24 histories changes exactly 3 status fields: correct application reports 24/24; strict application tasks 22/24, with both unnecessary closed attempts retained as failures. These are offline application outputs after seeing the misses, not new paid model answers, an independent holdout score, or evidence of improved model reasoning. The original 21/24 report count,19/24 task count and failed gate remain unchanged. No further API calls were made for this follow-up.

The useful design lesson is to keep execution metadata under application control. This ordinary solution does not establish a novel memory contribution. Sarvam plus ordinary storage handles all source-supported current dates/state in these easy six situations; harder evidence must justify any additional memory layer.

## Verification, compute and cost

113 focused tests pass, including real SQLite approval/idempotency, malformed-batch prevention, rollback, provider failure/no retry, no-tool refusal, read-only, final-tool refusal, source preservation and full synthetic replay/tamper detection. Offline live audit reconstructs all 61 requests from saved provider responses, re-executes real SQLite actions, matches persisted results/state and verifies meter charges. All 12 within-model initial pairs are exactly identical. Historical sources and grades are preserved. Clean-export verification and CI are recorded separately; this report does not assume them before completion.

61 new API requests,24 complete episodes,0 HTTP/provider/backend errors,0 unknown usage,0 retries. Base 105B: 25 calls, 23,256 prompt / 39,751 completion tokens; Conversations: 36 calls, 33,249 prompt / 2,190 completion tokens. Summed API durations 192.30s vs 40.31s (overlap under concurrency; not wall-clock runtime). Median per-call 6.07s vs 1.02s. Equal caps do not imply equal actual compute; no latency-adjusted winner claim.

Known new usage estimate INR 3.364035; retained/headroom INR 4.205069 within phase cap INR 220. Native-research cumulative 5,039 requests, known INR 63.056025, retained INR 111.077854 within existing INR 6,000. Historical unknown reservations remain included in retained accounting. Estimates are not invoices. Paid phase is complete with no restart or background inference.

## Next, in order

1. Adopt receipt-owned status in a versioned ordinary integration contract; test broader cases and keep raw model output visible. Decide whether a safe rejected closed-booking attempt is a policy inefficiency or a product-blocking failure; preserve this study's original metric either way.
2. Prepare the planned 24-workflow / 72-rendering EN-HI-TE bank with positive/negative cases, exact-duplicate and semantic-ancestry audit, independent-review limits and a prospective admission rule. Include actual source interpretation/currentness challenges against ordinary storage; current-state snapshots alone cannot demonstrate a memory advantage. No bank or paid 288-episode manifest is claimed ready by this report.
3. Run a separately frozen small integration check before the larger comparison. Do not spend further runs trying to make an LLM guess an application-owned status enum. A larger study must measure reproducible residual failures beyond what ordinary controls already solve.

English work and its scores are unaffected. No new memory architecture, all-language coverage, voice result, research-paper novelty or general Sarvam accuracy claim is established.
