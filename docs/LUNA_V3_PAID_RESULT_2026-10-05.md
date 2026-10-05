# V3 paid semantic diagnostic: failed gate and architectural findings

The founder approved the frozen23-call/$0.09/no-retry package. All23calls completed with locally valid plans;19checks passed and4failed. The all-case gate remains failed. No final answers or new English grades were generated. Historical423/500(84.6%) and provisional Terra425/500(85.0%) still score the same saved answers.

| Population | Passed | Interpretation |
| --- | ---: | --- |
| Exposed development |12/13| Same aggregate as v2; advice miss changed mechanism |
| Fresh internal semantic validation |6/9| New cases reveal remaining source-eligibility and role errors |
| Capacity awareness |1/1| Marked uncertain with only8of9required source turns selected; not full evidence coverage |

Do not combine these into an English accuracy percentage. All cases are short assistant-authored fixtures, not an independent external holdout. No independent review of this run is claimed.

## Four failures

1. **Original advice case:** v2 omitted the user's uncertainty statement. V3 includes it, but now also includes the excluded assistant suggestion as qualification. It remains uncertain. The original gate still fails; recovery of required evidence was offset by extra irrelevant focus.
2. **Fresh undecided booking:** selects both the user's indecision and assistant train advice as qualifications. The suggestion does not establish a booking. Expected uncertainty is correct; evidence membership fails.
3. **Fresh advice-only booking:** selects restaurant advice as qualification when no reservation is established. It correctly marks uncertain, but fails the frozen empty-selection requirement.
4. **Mixed positive/negative depot statement:** retains the correct whole source and says complete, but labels it support instead of qualification. Offline counterfactual replay proves that changing only this role yields byte-identical answer context. The frozen role gate fails, but this is not a demonstrated answer-quality defect.

There were no observed final-answer hallucinations because no final-answer calls occurred. Advice remains visibly attributed to the assistant; the risk is extra emphasis, not proven attribution failure in an answer. The original full packet is retained in every replay, including the capacity case where the ninth source remains in the baseline.

## Engineering progress versus outcome

Confirmed engineering improvements include strict structured outputs and source-ID validation, explicit evidence-role handling, original-text receipts, nonfit fallback without partial focus, label-free inference inputs, frozen scoring and bounded no-retry execution. All23paid outputs satisfied the local plan contract. Those are reliability properties of the experimental adapter/harness, not proof of complete product architecture robustness.

The v3 hypothesis is not validated for promotion: a qualification category lets irrelevant advice through and does not improve the13-case aggregate. Source membership still depends on a model's semantic judgment. Adding another bucket or tightening a prompt is not by itself evidence of a fix. We should not optimize audit-only labels as if they were English answer errors.

Next bounded engineering work: define query-relative source eligibility and separate proposal/advice from event evidence without globally excluding assistant turns (questions can ask what was recommended). Preserve explicit user uncertainty and mixed turns. Use paired advice-versus-action and time-scope cases; no benchmark-specific answers or name filters. Before another paid proposal, specify the expected change to actual rendered evidence and its failure controls. Keep role-only errors visible separately in any future protocol; do not retroactively relax this frozen gate. These fresh cases are now exposed and cannot remain an untouched validation set after tuning.

Only a preregistered paired answer experiment with the same Luna answerer, fixed evaluator and equal-compute control can demonstrate answer improvement. Full500measurement and an unchanged repeat follow only justified candidate evidence; independent transfer is needed for generalization. No defensible date for90%exists. We have not shown this change can supply the26additional correct answers needed to exceed90%from425/500on the provisional scale (451/500); a full-run outcome cannot be extrapolated from these fixtures.

## Audit and cost

Package canonical SHA-256 `5a8042bdc294b987c599ebbbfbeaef0090fdd2e60317730c489490a200b77d4d`.23attempts/responses, zero retries. Approvedcap$0.09; reservation$0.08668355; receipt-derived conservative cost$0.0083113, not invoice reconciliation. Model/settings unchanged. Recomputed summary, approval binding and original-source render spans for all23outputs verified. Raw paid responses and original labels are preserved in workspace memory under `plans/2026-10-05-luna-semantic-validation/paid-diagnostic-001/`. Public [verification](results/luna-selector-v3-2026-10-05.json). No further paid call is authorized by this completed approval.
