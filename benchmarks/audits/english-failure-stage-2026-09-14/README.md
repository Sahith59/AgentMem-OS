# English full500 failure-stage audit

The independently verified precision run remains **423/500 (84.6%)**. This post-run development audit covers all **77** official misses; it does not change any answer, verdict, benchmark reference, or score.

| Diagnostic category | Cases |
| --- | ---: |
| Every exact annotated source turn delivered; answer/reasoning/reference/judge remains | 25 |
| Evidence-sufficiency or judge | 11 |
| At least one required source turn has no linked extracted fact | 2 |
| At least one linked fact is absent from the final packet | 10 |
| Raw turn absent while linked fact lineage reaches the packet; semantic sufficiency unresolved | 29 |

The audit reconstructs all 1,000 saved generation/judge request hashes. Generation uses only the frozen prompt, context, question, date, and model settings; references enter only the later judge request. The audited retrieval sources contain no benchmark question IDs.

Question operations among misses are concentrated in aggregation/amount (35) and temporal/update (15), followed by direct recall/synthesis (12), evidence sufficiency (11), and preference/advice (4). Reaching 90% on this population requires 27 additional correct answers, so neither retrieval nor a model replacement alone is assumed sufficient.

Use `benchmarks/audit_english_failure_stages.py` with the exact frozen package, checkpoints, source annotations, and immutable facts database to reproduce the full row-level analysis. Earlier provisional lineage outputs are not represented here; the final audit requires `--facts-db` so a mutable live store cannot be mistaken for the frozen packet store.

The measured evidence-sufficiency prompt is available as the opt-in `--answerer balanced`. Historical `reasoning` remains the default so prior results remain reproducible.

## Answerer-capacity diagnostic

A frozen 40-case paired development diagnostic subsequently compared `gpt-5.6-luna` with `gpt-5.6-terra` while holding all packets, questions, prompts and GPT-4o judge requests fixed. On the 25 exact-evidence misses, Luna scored 3 and Terra scored 10, with eight gains and one loss. Both models scored 15/15 on triple-stable controls. The prospective capacity gate passed.

This supports answerer capacity as a material contributor, especially for latest-state selection, cross-session arithmetic, personalized evidence and exact-detail recovery. Fourteen exact-evidence targets remained wrong in both arms, concentrated in numeric operations, chronology/entity selection and preference/reference specificity. The verified full500 headline therefore remains 423/500 (84.6%); the targeted result is not a population score or a default-model promotion. See [`model-capacity-result.json`](model-capacity-result.json) for the compact integrity and outcome record.
