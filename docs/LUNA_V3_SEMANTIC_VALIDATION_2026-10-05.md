# Fixed-Luna v3 semantic validation: prepared, not executed

Latest measured status: the approved run completed23calls with19checks passed and4failed; the gate failed. See [result and limitations](LUNA_V3_PAID_RESULT_2026-10-05.md). The following is the preserved pre-run protocol.

The next experiment tests the new evidence-role selector against frozen source labels. It keeps Luna, answerer settings, provisional Terra evaluator and stored extraction corpus unchanged. There are no new model outputs or English scores. The preceding paid v2 result remains12/13 on exposed development fixtures, with the all-case gate failed.

## Population and scoring

| Population | Calls | Frozen check |
| --- | ---: | --- |
| Existing exposed development | 13 | Original required/excluded source membership on support/qualification union; old labels unchanged |
| New internal semantic validation | 9 | One predeclared acceptable evidence set, allowed source roles, and expected sufficiency |
| Capacity awareness | 1 | Valid subset within eight-turn cap and uncertainty when nine independent operands are required; not complete source coverage |

The internal cases cover unresolved uncertainty, later resolution, explicit denial, mixed positive/negative facts, corrections, conflicts, advice versus actions, and event dates versus observation order. Alternative acceptable sets are declared for resolved uncertainty and self-contained corrections before any outputs. Lists need not be exhaustive: irrelevant sources may be rejected or left unclassified. The capacity case can pass while omitting sources; its separate result must never be described as complete evidence selection.

The overall gate requires all23 checks to pass, with each population reported separately. Schema/provider failure stops before another dispatch; ordinary semantic failures remain in the denominator and do not trigger retries. No automatic promotion follows from a pass: inspect raw outputs and source receipts before a paired fixed-Luna answer screen with an equal-cost extra-call control. These short synthetic cases cannot establish performance on full conversation histories.

These are assistant-authored internal validation cases created after the v3 adapter was fixed, not independent human labels or an external holdout. Root reviewed source/question/label consistency. Two requested GPT-6 Sol reviews were unavailable due to model capacity; no independent review of this new set is claimed. Do not tune on the set after its first model result and continue describing it as untouched validation. Any revised cases or contract require a newly versioned package; preserve original results.

## Exact proposed run

- Model: `gpt-5.6-luna`, unchanged low-reasoning settings and2048 maximum completion tokens.
- 23 calls, no retries, no final answerer or judge calls.
- Frozen rate assumptions inherited from the previous diagnostic; conservative reservation **$0.08668355**, proposed hard cap **$0.09**. This is not a provider invoice.
- Canonical package SHA-256: `5a8042bdc294b987c599ebbbfbeaef0090fdd2e60317730c489490a200b77d4d`.
- Workspace memory package: `plans/2026-10-05-luna-semantic-validation/preparation-001/package.json`.
- Bound output: `plans/2026-10-05-luna-semantic-validation/paid-diagnostic-001/`.
- Approval template is inactive. No inference is authorized by preparation alone.

Use the prepared `execute_approved.py` launcher only after an approval file binds this package, cap and output directory. It reconstructs source/code/fixture hashes before dispatch and uses the existing credential without printing it. A failed or ambiguous attempt is preserved, not automatically replayed. A completed checkpoint can be inspected without new calls; do not treat an old approval as a new run allowance.

## Offline verification

119 focused tests pass, including11 new harness/scoring checks. New Python files pass Ruff. Tests cover label-free provider projection, separate population totals, role/sufficiency/membership failures, alternative evidence sets, capacity handling, paid-provider isolation, cap and package binding, failed-output usage receipts, no-retry/resume behavior and retention of semantic failures. Mock outcomes are harness checks, not measured selector performance. Frozen v2 package reconstruction still passes; neither the v3 adapter nor old model settings were changed for these cases.
