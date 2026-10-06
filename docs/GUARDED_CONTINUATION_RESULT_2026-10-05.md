# Normalized continuation: stopped result and source audit

October 5, 2026 (America/New_York). Status: approved execution stopped without retry; no English closure or accuracy improvement demonstrated.

## Actual result

The continuation made 24 new requests, reusing all 25 prior responses unchanged. Combined ledger: 49 unique provider responses, 48 accepted jobs, one new planner validation failure, six complete three-arm comparisons. The original run's separate failed checkpoint remains unchanged.

| Method | Correct / complete questions |
| --- | --- |
| Fixed Luna baseline | 5/6 |
| Fixed Luna draft/revision control | 5/6 |
| Fixed Luna guarded source focus | 5/6 |

Focus has zero gains and zero losses against either control. Four of six completed cases received applied focus; one kept the original packet because the focus did not fit, and one because of uncertain assistant qualification. These six are a partial, development-exposed sample, not a completed 32-question result or a new full-benchmark score. Do not report 83.3% as a replacement English baseline. Historical English remains 423/500 (84.6%); 425/500 (85.0%) is a separate provisional Terra regrading of identical saved answers.

## Why execution stopped

At `22d2cb42:plan`, all source IDs were known and support/qualification lists were ordered and disjoint. Only the rejected-ID list was out of source order. The frozen contract rejected this response even though rejected IDs do not enter the answer context. No provider retry occurred. The earlier overlap recovery worked; this is a distinct, avoidable contract burden on the planner.

A new offline-only adapter canonicalizes all role lists into trusted source order before the existing strict checks. It preserves source membership, original text and selected union. Unknown IDs, duplicates, selected/rejected conflicts and other schema defects remain errors. No paid runner imports it. Replay succeeds on all seven saved plans and preserves the rendered contexts of all six previously valid plans. This is a reliability correction, not a measured accuracy gain.

## Concrete shared accuracy miss

For `gpt4_ab202e7f`, all three methods answer four kitchen items; the reference lists five. The frozen packet contains the four named items but omits the coffee-maker replacement evidence. The full cleaned conversation, session index43 / ID `answer_728deb4d_4`, turn0, May30, says the user received an espresso machine and donated the old coffee maker. The packet mentions espresso-to-milk ratios elsewhere, but does not deliver that upgrade/donation turn. This supports an upstream evidence-delivery gap; it does not support blaming the judge or prove that a larger model would fix the miss.

The source review used the reference after execution for diagnosis only. It did not alter any model request, grade, original artifact or candidate answer. Recovering the missing turn is not yet a demonstrated answer gain. A runtime retrieval repair must operate from the question and source corpus, without gold answers or answer-session IDs.

The other new completed cases were charity total ($5,850, with team/personal attribution caveat retained) and mortgage preapproval ($400,000, supported by the November30 user turn). Every arm agrees with the fixed judge on these outcomes. The first three cases retain the earlier audit.

## Accounting and integrity

- New requests:24; new reservation:$0.70037835; new receipt-derived upper usage:$0.04382065.
- Combined reservation:$1.40394505; combined receipt-derived upper usage:$0.08766035. These usage estimates are not invoices.
- All49 provider response IDs unique; unreconciled usage attempts:zero.
- All25 inherited jobs match reconstruction exactly; original checkpoint, package and approval hashes verify.
- Approved package:`d1342082bc937a3a7e894d699d373b92429477e057156b63eabfe380b05b5189`.
- No further paid calls after the stop. Remaining unspent allowance does not authorize changing the frozen implementation.

Validation:149 focused tests passed and Ruff passed. Independent GPT-6 Sol review reproduced the summary, inheritance, cost accounting, ordering defect and missing source finding, and found no blocking issue in the offline adapter. PR32 post-merge CI37406848435 passed before this result audit.

## Next work

1. Keep this stopped experiment and its six comparisons as evidence; no promotion or benchmark claim.
2. Check the evidence-delivery path for broad aggregation questions. Build a gold-blind bounded retrieval expansion with source receipts, strict context limits and regression controls; do not hardcode the coffee-maker example.
3. Before any resumed paid experiment, freeze the complete contract policy and explicit continuation lineage. Do not keep relaxing a running experiment or skip failed questions. Source expansion is a separate architectural intervention and must not be mixed silently into this comparison.
4. Require measured gains under the fixed models and controlled budget before a full benchmark and repeat. No defensible date for 90% follows from this run.

Local evidence: this directory's `checkpoint.json`, `incomplete-summary.json`, `offline-audit.json`; source corpus SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. The adapter is `benchmarks/evidence_role_normalization_v2.py`, with tests in `tests/test_evidence_role_normalization_v2.py`.
