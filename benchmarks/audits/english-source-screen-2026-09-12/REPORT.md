# Source-supplement screen readiness — September 12–13, 2026 EDT

The measured English score remains **418/500 = 83.6%, twice**. No new paid call or answer-quality result exists. This package is ready for a scoped comparison; it does not prove the remaining gaps fixed or promise90%.

## What changed

The prior source audit showed that preserving the whole original packet and appending complete source turns could restore omitted detail without deleting working evidence. The executable screen now freezes the original150-question selection, original answer prompt, original model settings, original judge templates, all baseline/candidate packet hashes, source artifacts, runtime and cost policy. Generation receives question/date/context only; cohort and gold labels do not select generation behavior.

Models remain `gpt-5.6-luna` for answers and `gpt-4o` for judging, with the exact prior caps and temperature. No extraction runs or default changes. Returned models are restricted to those observed in both original repeats: `gpt-5.6-luna` and `gpt-4o-2024-08-06`. Provider drift halts execution instead of quietly changing the comparison.

The new runtime reconstructs saved requests, answers, grades, shared judgments, provider receipts and cost accounting before resume. The older general V2 runner checked binding/ledger/sharing but did not reconstruct completed answers/grades from saved responses on resume; the new tests demonstrate rejection of such tampering. Historical runners and paid artifacts remain unchanged.

## Evidence and remaining gaps

- Reverified500 packets and4,190 added turns against scoped source text, roles, offsets and hashes. Every original packet remains intact;197 annotated turns are regained and none lost.
- All72 stable misses remain in the screen.38 gain annotated turns;36 of these are answerable cases and2 are abstention cases. More evidence in an abstention case does not make the requested fact answerable.
- Among64 answerable persistent misses, every annotated source turn is present in13 original packets and34 candidate packets.30 candidate packets still lack at least one annotated turn. Complete raw-turn coverage is not a proof of sufficient or unambiguous evidence; facts can also carry information absent from raw turns.
- Source witnesses establish omitted details such as page250, yoga3/week,15-minute weekday wake offsets and12figurines. New QA behavior is still unmeasured.
- Counting, date reasoning, current-state interpretation and unsupported entity substitutions remain open. Judge/reference disputes stay separate; primary labels are unchanged. Extraction-versus-migration/filtering attribution is unresolved. No all-gap closure claim.

## Frozen decision gates

The150 pairs comprise72 stable misses,20 disagreements and58 deterministic stable-pass controls, including all30 abstentions. Require all of:

1. Complete150 pairs with no unresolved attempts, selective regrades or retries.
2. At least10 net correct gains: candidate correct minus baseline correct≥10.
3. At most2 paired losses on stable-pass controls.
4. Candidate correct on at least56/58 controls. This stricter floor was added **before any paid outputs**: both arms failing a previously stable control would otherwise escape the paired-loss gate.
5. No net abstention decline across all30 abstentions.

All500 questions are development-exposed. This selected150-case score is not a new full500 headline or a held-out estimate. A pass supports considering a separately approved full500 measurement; a failure blocks broad promotion. Report every gain and loss. No automatic larger run.

## Package and cost

- Canonical package SHA256: `3a32272ab417756f01a8d83f98df246b83901a94e5f39d4679a34d5fb71aba34`.
- Active package: `execution-v2/package.json` in the local source-supplement-screen-001 run directory.
- Full conservative reservation: **$8.9676845**; proposed rounded cap: **$8.97** (`8970000000`nanoUSD).
- Maximum300 generated answers and300 judge calls,600 total with no retries. Identical within-question judge requests may share a verdict and reduce actual calls.
- Proposed paid output: `/Volumes/Sahith_SSD/AgentMem-OS/codex-memory-2026-09-08/runs/english-improvement-2026-09-11/paid-source-supplement-screen-001`. It does not yet exist. No founder approval record has been created.
- Prices checked against [official Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [official GPT-4o documentation](https://developers.openai.com/api/docs/models/gpt-4o): Luna$0.20input/$1.20output per million tokens, with1.25xcache-write input premium; GPT-4o$2.50input/$10output. Inputs remain below the272kbound for the long-context pricing threshold.
- Cap uses UTF-8 bytes plus1024wrapper tokens and full output limits. Integer reservations are durably saved before dispatch and never automatically refunded. This bounds this runner at frozen prices; it is not an account-wide cap. Usage estimates include reported cache writes and ignore read discounts; actual invoice reconciliation is separate.

The earlier `execution/package.json` is superseded because a trailing blank line was removed from the runtime during staged diff review. Hash binding correctly required a new package. No settings, population, source packets, gates or budget changed. The earlier synthetic receipts are preserved; use execution-v2 and offline-v2 only.

## Validation

59 focused tests pass:43 new screen tests and16 existing source tests. Coverage includes corrupted answer/grade/request/ledger, source/runtime drift, invalid approval, no-network default, no SDK retries, budget stop before calls, active file lock, pending/timeout/truncated/ambiguous/oversized output, model drift, cache pricing, sharing provenance and clean prefix/completed resume.

The exact final package completed two socket-denied synthetic scenarios:600 distinct-answer calls and450 identical-answer calls with150 shared judgments. Both completed600 logical jobs and independently verified all150 pairs. Both completed resumes dispatched zero calls. Synthetic scores are not model evidence; all-correct mock arms deliberately fail the net-gain criterion.

The old1,591-file artifact manifest rehashes with zero mismatches. Existing founder .gitignore change and untracked benchmark/website files are preserved. Application defaults and all earlier paid results are unchanged.

Code/runbook commit: `0b2a620`. Compact evidence is committed separately; merge/CI receipts are recorded in durable memory when complete. Full source packets and synthetic checkpoints stay in local memory; the repository retains the builder, verifier, runnable recipe, compact receipts and full file hashes. Rebuilding elsewhere requires the original local source artifacts and creates a new path-bound package to review.

## Next action

Founder review of exact150-pair scope and$8.97cap is the only pending input for this new paid screen. Earlier paid-run approvals are consumed. No worksheet or manual example judging is required. After approval, create the exact bound record, reverify hashes and source receipts, run serially, independently verify all150 paired outcomes, and report the failures as well as gains before deciding on further work.
