# Receipt-owned reporting connected; source challenge comparison prepared

**The new live integration gate passes. The larger comparison is ready and has not started.** This is verified progress in application behavior and experimental readiness, not an improved English baseline, general Sarvam accuracy score or novel memory result.

## What changed

The new runner uses the existing ordinary SQLite backend and fixed Sarvam models, then applies receipt-owned status during the actual workflow. It preserves each raw model answer and creates a separate application output. Only `action_status` can change. Dates and clarification flags remain exactly what the model returned. The projector reads trusted request intent and actual/prior execution receipts; it never reads grading labels or uses a database lookup to repair the answer.

The finalizer now answers the actual question. The previous current-date-only instruction could not validly handle a question asking for the customer's latest requested date. Complete original messages, source records and receipts remain available, with no executable final tools. Historical runners, raw results and the prior failed gate remain unchanged and reproduce exactly.

## Prospective live integration result

Six existing mechanics situations × two models, guarded backend only: **12 episodes, 30 API requests**. These repeated setup cases are excluded from the larger bank. Same models/profiles: `sarvam-105b` low reasoning and `sarvam-105b-conversations` null; temperature 0, output cap 8192.

| Result | 105B low | Conversations | Total |
|---|---:|---:|---:|
| Correct raw model report |4/6|4/6|8/12|
| Correct application report |6/6|6/6|12/12|
| Correct stored state |6/6|6/6|12/12|
| Strict task success, including unnecessary-call check |6/6|5/6|11/12|

Both required approved changes executed successfully. Neither pending nor revoked changes executed. No prohibited stored-state change, duplicate write, invalid projected answer, provider/backend error, unknown usage or retry occurred.

The four raw-report misses are all status fields: both models label a pending change `not_requested`; base 105B labels a no-tool revoked refusal `rejected`; Conversations labels a read-only current-date answer `confirmed`. All dates and clarification flags are correct. Under this explicit contract, a current booking's confirmation does not establish a new action. The application emits `not_performed`, `not_performed`, `not_performed` and `not_requested` from actual intent/receipts. Raw responses stay visible.

One Conversations episode attempts to modify a visibly closed booking. SQLite rejects it and the final date/status are correct. It remains a strict-task failure and an unnecessary-call finding. The prospectively declared setup gate checks correct application reports, useful approved writes and safe desired state; it does not require zero unnecessary safely rejected calls. **That new gate passes. The old PR55 gate stays failed.** No historical scoring threshold was edited.

This demonstrates the software correction in a new live workflow. It does not demonstrate improved model reasoning: raw-report correctness is only 8/12, and the generalized reporting prompt changed. Do not compare this as a causal accuracy gain against earlier 21/24 reports or the offline projection. The models themselves still confuse some status categories.

## Larger source challenge bank

24 scenario instances across eight known families, each in English, Hindi and Telugu: **72 distinct visible inputs**. The families cover requested versus confirmed dates, corrections/missing information, conditional preferences, booking references, delayed evidence, permission, approval time/scope, and idempotency with newer state.

Thirty of 45 read-only questions cannot be answered by copying the initial current booking date. Six additional renderings require successful new writes. These are offline bank-coverage checks, not model failures or measured gains. There are two positive new-write scenarios, so refusing all changes cannot pass. Versioned duplicate fixtures use actual prior SQLite operations; an old retry must preserve a newer confirmed date.

All language triples share the same server fixtures and expected meanings. Exact question/source matches against the earlier native bank: zero; within-bank exact visible duplicates: zero. Known semantic ancestry is explicitly shared with earlier correction, conditional, identity, backend outcome, effective-date, update and retry families. Three related instances per family and their translations are correlated. New IDs and wording do not create a holdout.

Assistant source review is complete; independent native-language review is absent. Text sources and questions are translated, while machine-readable fields/system instructions remain English. Action arguments are supplied explicitly, so this does not evaluate multilingual argument extraction. Every case supplies an authoritative initial server snapshot. Some questions need original source interpretation beyond that snapshot, but this is not a long-context retrieval, voice, cross-session persistence or production-authentication study. No memory architecture is being tested.

Preflight caught and removed a contradiction before calls: read-only cases 01–12 had an unused valid approval, including the case describing approval as pending. Their unrelated approvals are now absent. Frozen source/labels record the corrected fixtures; no model outputs were used for that change. CASE_REVIEW.md records the interpretation for every scenario.

## Comparison package and next execution

The prepared manifest contains **288 episodes = 72 inputs × two fixed models × two backend-enforcement arms**, at most 1,152 API calls. Each within-model pair begins with exactly the same source input, server evidence and tools. Only server approval enforcement differs. Independent native action trajectories can vary; no guard benefit can be claimed on cases where no approval violation was attempted.

The runner recomputes and verifies the setup gate before credential access. A changed source, bank, protocol, budget or gate blocks execution. Four workers share 90 RPM pacing, below the founder-reported 120 RPM. No retry or resuming a started manifest. Stop after a batch with two infrastructure/backend errors. Per-case factual, status, state, safety and unnecessary-call results remain separate. Analyze the eight families and 24 scenario instances as well as the 72 language renderings; do not treat 288 episodes as independent customers.

**No comparison API calls have occurred.** It is prepared under the existing research authorization for the next phase; no new key or repeated blanket budget permission is needed. Do not claim a larger result until that run is complete and replayed. The next task is to execute this frozen comparison once and inspect every failure before proposing a memory design.

## Prospective evaluation clarification

A final pre-execution review found that reporting a real but unauthorized state change must be separated from making a factual error. The original primary grades measure the desired task outcome and remain unchanged. A separately frozen [attribution contract](ATTRIBUTION_CONTRACT.md) adds actual answer truth, execution status, desired state and prohibited changes as distinct measurements. It cannot turn an unsafe write into success or repair an answer. The new wrapper binds this contract before entering the original runner. No larger-comparison outputs were available when defining this distinction. Setup results are unaffected.

## Verification and accounting

129 focused tests pass, including live-path status projection, preservation of deliberately wrong dates, incomplete/provider/final-tool failures, conditional/source label contrasts, duplicate/currentness initialization, full synthetic replay, network restoration after the gate and tampered-gate refusal before key access. Ruff and diff checks pass. Original PR55 raw/projection results still reproduce exactly; its failed gate remains false.

Offline replay reconstructs all 30 live requests from saved responses, re-executes the actual SQLite operations and matches the persisted result, application projection, database state and meter charges. New setup known usage estimate: **INR 2.990055**, retained/headroom **INR 3.737582**. Native-research cumulative: **5,069 requests**, known **INR 66.046080**, retained **INR 114.815436**. Retained includes historical unknown reservations and is not an invoice.

Base 105B used 13 calls, 11,769 prompt / 41,729 completion tokens, 189.01 summed API seconds, 12.02s median per call. Conversations used 17 calls, 16,444 prompt / 1,052 completion tokens, 18.41 summed API seconds, 1.02s median per call. Summed durations overlap under concurrency. Equal caps do not imply equal actual compute; no speed or cost advantage is claimed for the base model.

Setup cap INR 110; prepared comparison cap INR 2,590; phase cap INR 2,700 within the existing INR 6,000 ceiling. Conservative maxima are caps, not expected charges. The existing phase ledger includes setup spending, so the comparison cannot silently reset it. No inference is active. Clean-export verification, public archive hashes and PR/main CI are recorded separately; publication is not assumed before those checks finish. Founder files and the English track remain protected.
