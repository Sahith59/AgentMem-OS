# Guarded answer screen: stopped run and bounded continuation

October 5, 2026, America/New_York. Status: original paid run stopped; normalized continuation prepared, not approved or executed.

## Observed result

The approved 32-question comparison stopped after 25 attempts: 24 completed jobs and one planner contract error. Three questions have all three answers and judgments. Baseline, draft/revision and guarded focus each passed those three questions, with zero comparative gains or losses. This is insufficient evidence of improvement; English remains 423/500 (84.6%) historically, or 425/500 (85.0%) for the identical saved answers under provisional Terra.

The fourth planner put the same three existing user-turn IDs in support and qualification lists. The strict disjointness contract rejected the response. No source IDs were fabricated. Source review still leaves real semantic uncertainty about personal versus team charity amounts; fixing role overlap does not prove the eventual answer correct.

The original checkpoint, failed status, raw responses, approval and code remain unchanged. There were no retries. Reservation was $0.7035667; receipt-derived upper usage estimate was $0.0438397, not a provider invoice. All 25 responses have receipts; unreconciled attempts: zero.

## Engineering correction

A separately versioned adapter gives qualification precedence when a known source appears in both lists. It preserves the selected source union and original source text. Unknown IDs, invalid ordering, duplicates within a list, rejected-source overlap and other schema defects remain errors. Existing valid plans preserve their rendered context. Semantic correctness is explicitly not certified by normalization.

The continuation checks source artifact hashes, original approval, unchanged questions/models/prompts/gates, inherited responses and incremental spending limits. It reuses all 25 responses, including offline normalization of the rejected plan, and never overwrites the stopped run. Interrupted or unrelated failed dispatches cannot be inherited as completed work.

Validation: 141 focused tests passed; Ruff passed on the five new files. Independent GPT-6 Sol review checked the actual stopped result and continuation. Its provenance-tag, budget and pre-bootstrap directory concerns were fixed and tested. Offline inheritance of the actual 25 responses passed. No new model calls followed the stop.

## Exact continuation proposal

- Same 32 development-exposed questions, Luna answerer/planner, Terra judge and extraction data.
- 231 remaining requests maximum, no retries; $6.77 additional reservation cap.
- Combined reservation ceiling: $7.4735667, below the original $7.50 ceiling. Remaining calculated reservation: $6.76207345.
- Package SHA-256: `d1342082bc937a3a7e894d699d373b92429477e057156b63eabfe380b05b5189`.
- Approval template is inactive. Changed output acceptance requires exact approval for this new package under the workspace working agreement; the old approval stays bound to its original package.
- Pass requires all 32 comparisons, at least three net gains against each control, at most one loss against each, at least eight applied-focus cases, then source/grade review. No full benchmark follows automatically.

The two-stage controls match calls and output caps, not exact realized input tokens or cost. These are exposed development cases, not a fresh generalization test. A successful screen must precede a full benchmark and repeat. Neither a 90% result nor a date for it is established.

Local evidence: `codex-memory-2026-09-08/plans/2026-10-05-guarded-answer-screen/paid-screen-001/{checkpoint,incomplete-summary}.json`; `normalized-continuation-001/{package,approval-template,offline-replay}.json`. The earlier normalized-continuation-draft.json is superseded and must not execute.
