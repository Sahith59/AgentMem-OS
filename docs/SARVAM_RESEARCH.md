# Sarvam research: evidence, limits and next task

Updated October 13, 2026. We are investigating which problem warrants a product. We have measured integration gains and concrete failures, but no demonstrated unique multilingual memory advantage. This work is separate from the English baseline.

## Latest corrected comparison

All 240 native episodes were attempted once. Forty related scenarios, each rendered in English/Hindi/Telugu, were tested with two fixed Sarvam profiles. Both reporters share each native history; only their first instruction differs. All 120 episodes per model stay in the denominator, including unusable and unreached reports.

| Model | Control date + clarification correct | Candidate | Paired gains / losses |
|---|---:|---:|---:|
| Sarvam-105B, low reasoning | 90/120 (75%) | 105/120 (87.5%) | 20 / 5 |
| Sarvam-105B-Conversations | 95/120 (79.2%) | 96/120 (80%) | 4 / 3 |

**The acceptance gate failed. No candidate is promoted.** Conversations wrong non-null dates increased from 18 to 22, and strict task success fell 85→84. One native response consumed its 8,192-token output cap and returned no content or tool calls; the pipeline skipped both reporters. Thus 478 reporter calls were issued for 480 planned slots. All failures remain included.

Fourteen of the 20 base-model recoveries restore a previously unusable final output; six fix an answer that was usable but wrong. This is a useful reporting gain, not proof that a new memory architecture improved reasoning. The primary score includes the clarification flag. The candidate's supplementary 114/120 date-only result must not replace 105/120 as the headline.

Actual actions are shared across reporter arms: 47/48 required new writes completed, zero forbidden state changes, and 11 unnecessary Conversations attempts. The ordinary backend blocked those attempts. A correct old-date answer does not count as task completion when an authorized update was missed. Model-reported status and receipt-owned application status are separate metrics.

[Full report](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/REPORT.md), [interpretation](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/INTERPRETATION.md), [all failures and disagreements](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/FAILURE_REVIEW.md), [next task](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/NEXT_STEP.md).

## What we have learned so far

| Earlier step | Result | Interpretation |
|---|---|---|
| Early memory comparison and repeat | Pooled 47/48 for both arms; initial advantage reversed | No established memory advantage |
| Native capability study | Low strict scores included formatting and dataset issues | Those scores do not establish a vendor memory weakness |
| Ordinary storage/tool controls | Four unsafe unguarded Conversations writes; zero guarded; 23/24 legitimate writes completed | Ordinary host enforcement helps, with one missed legitimate action |
| Saved-history reporting sentence | Base 59→72/72; Conversations 70→71/72, including one regression | Narrow reporting effect; no new action or memory result |
| Initial fresh confirmation | Stopped 32/240 after 111 calls because our question was ambiguous | Test-design flaw preserved; not a clean confirmation |
| Corrected v3 confirmation | Base 90→105/120; Conversations 95→96/120, but gate failed | Real base-model gain with important remaining failures |

[Earlier research history](../benchmarks/audits/sarvam-reporter-wording-2026-10-11/SARVAM_HISTORY.md), [saved-history result](../benchmarks/audits/sarvam-reporter-wording-2026-10-11/REPORT.md), [stopped-run evidence](../benchmarks/audits/sarvam-fresh-confirmation-2026-10-11/REPORT.md).

V2 corrected 12 historical/current questions but was never executed. A further assistant preflight found 12 preference/request ambiguities, corrected in v3 before execution. No source, expected date or backend contract was changed after seeing v3 outcomes. The cases inherit known mechanisms and 32 earlier exposed episodes; they are development research, not an independent holdout. Assistant review covers every source and reporter slot; independent native-speaking human review is still missing.

## Concrete remaining problems and engineering

Candidate Conversations missed all 12 preferred-date questions involving two booking references, despite the target preference being present. It missed 7/12 unresolved-condition questions and 2/12 resolved-condition questions. These are source-present failures: they do not establish that persistent memory or retrieval is missing. Cause between model interpretation, prior actor prose and evidence layout remains to be isolated.

The [evidence-view component](SARVAM_EVIDENCE_VIEWS.md) separates original source, executed receipts and unverified actor claims, with exact reconstruction checks. A three-arm comparison harness distinguishes lossless grouping from omitting actor prose. Its prospective 108-call selection remains blocked and unexecuted: one selected capped native history lacks the final request and the parent fails completeness. We did not drop the case or weaken the gate. Code tests are not live accuracy results.

The skipped reporter after a capped, call-free native response is an identified integration recovery gap. An [opt-in reporting handoff](SARVAM_REPORTING_HANDOFF.md) is implemented and tested offline, then exercised in the one-case diagnostic below. Recovery must stop the actor, preserve the failed response, use only verified existing source/receipts, and expose no executable reporter tools or retries. The opt-in implementation remains separate from the frozen results; one observed recovery does not establish broad reliability.

## Observed capped-actor recovery

A separate, prospectively frozen two-call diagnostic succeeded on the sole eligible capped episode. Both existing Sarvam-105B reporters returned the correct null date, clarification required, and no action requested. No actor or backend action was repeated. All 240 prior routing records were checked; the original study and its failed gate remain unchanged. This is one English case and no candidate-over-control advantage was found here. [Result, protocol and raw evidence](../benchmarks/audits/sarvam-capped-recovery-2026-10-13/REPORT.md).

The opt-in recovery path is now exercised with real provider responses, not just software tests. It adds compute: candidate reporting alone took 17.63 seconds, so this does not establish a fast voice path. The diagnostic added ₹0.566850 known estimate / ₹0.708564 retained. Native research cumulative is now 7,037 calls, ₹271.224630 known / ₹371.289339 retained. No new English LongMemEval result or multilingual memory advantage follows.

The next proposed study compares all 240 saved histories across ordinary chronological evidence, lossless evidence groups, and those groups without nonempty actor prose. It targets source interpretation with fixed models and reports both gains and regressions. The 720-call proposal remains unexecuted; the older blocked 108-call screen is preserved. [Next protocol requirements](../benchmarks/audits/sarvam-capped-recovery-2026-10-13/NEXT_STEP.md).

## Accounting and reproducibility

V3 made 844 calls: known token-cost estimate ₹90.677925; retained with headroom ₹113.347725; no unknown-usage calls. At the close of v3, native research cumulative was 7,035 calls, ₹270.657780 known and ₹370.580775 retained under the existing total ₹6,000 research ceiling. Older separate pilots remain separately accounted in their history. Estimates use [published pricing](https://docs.sarvam.ai/api/getting-started/pricing), checked October 13: ₹15/₹60 per million input/output tokens, ignoring cache discounts and keeping 25% headroom. These are not invoices or an account-balance check.

The archive preserves source/gold separation, every request and receipt, actual SQLite state, code hashes, complete assistant reviews and negative results. Offline request/backend replay and a separate parser/accounting check agree. No failed request was retried and no missing output was imputed. See the archive README for clean replay commands.

## When a finding warrants a product

A candidate problem must reproduce on fresh reviewed cases, survive clear instructions and strong ordinary-storage controls, have a traceable cause, and matter in an actual user workflow. For memory specifically, missing persistence, identity or evidence selection must be causal. A prompt error, malformed test or reporting failure is not proof of a missing vendor memory capability.

A public benchmark remains required, with documented language/task coverage and independent review. Neither startup-program acceptance nor these synthetic scores establishes novelty or production readiness. The planned eventual language scope follows the relevant Sarvam service/model, with coverage earned through testing. See [INDIC_ROADMAP.md](INDIC_ROADMAP.md).
