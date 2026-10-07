# Source-review handoff: useful review tooling, no new selector or score

October 7, 2026. Implementation 2f5264c; starting main 26d228d. **The offline review path is complete. Automatic semantic source selection remains unimplemented.** English stays 423/500 (84.6%, once) ; 425/500 is a Terra regrade of the same answers. No paid calls, new answer packets, new embeddings, model/corpus changes or default promotion.

## Why this task changed

The previous source-count audit gained five turns on historical misses, but four repeated known information or left a reference dispute. Exact-source novelty was being mistaken for useful new information until qualitative review corrected it. Existing ranking/focus/packing failures do not support another heuristic that treats keyword/date parsing as semantic completeness. Root and an independent GPT-6 Sol reviewer found no currently defensible automatic model-free proposer for all required relationships. This is a method gap, not a proof that such a method is impossible.

The implementation therefore makes the necessary semantic review concrete instead of presenting an automatic selector as done. `benchmarks/source_unit_review.py` accepts typed original snapshots, supplied source-bound rankings and supplied SourceUnits. It exports the entire eligible original pool, including distant corrections and qualifiers; exact question and old packet; source roles/dates/identity; proposed spans/edges and paired compiler previews/refusals. The old packet is explicitly legacy mixed evidence, not ground truth. Original baseline identity and temporal cleanliness remain uncertified.

Thirteen questions always remain UNREVIEWED: ten canonical dependency kinds, question relevance, new information and answer sufficiency. A declared dependency, empty dependency graph, literal body match or valid preview never changes this status. There is no semantic-complete input, automatic approval, model call, database connection, ranking algorithm or answer-path adapter. Separate reviewer findings do not mutate the dossier or automatically become runtime policy. Future sources are excluded from the review source pool; their IDs remain visible as exclusions. The full offline dossier has no answer-budget claim.

## All 500 reviewer-input audit

Before outputs, the policy, script, code and existing label-free hybrid runtime/hit records were hash-frozen. The first eight existing hybrid hits were represented as whole-turn example units. These are mechanical examples, not a new semantic proposal algorithm. No grade, reference answer or annotation file was loaded. Network was blocked.

Independent replay verifies:

- 500 dossiers with exact old packet bytes and original source metadata/order/text.
- 231,575 eligible source occurrences; 15,175 future occurrences excluded from source text.
- 3,988 supplied whole-hit units: 3,953 PREVIEW_ONLY and 35 refused by the 4,000-character compiler-preview cap.
- Exact hit order/hash, preview/report bindings, JSON round trips, all thirteen UNREVIEWED statuses and no answer/paid readiness.

The 4k cap applies only to the compiler preview. Review dossiers include full histories and are much larger; do not pass them to an answerer or count them as budget-fitting answer contexts. The reused inputs follow their frozen source-time contract; this does not certify the legacy baseline as production-time clean. Structural success is not evidence gained/lost, semantic selection accuracy or English closure.

## Three prospectively selected source reviews

Before outputs, select exactly three cases by SHA256('source-unit-review-v1:'+question), without outcome resampling. Root and two GPT-6 Sol reviewers examined all 24proposed turns, relevant old-packet witnesses and additional original context as documented separately. They did not load gold answers, grades or model answers. The corpus is nevertheless development-exposed, and the reviewers are assistants, not independent human annotators. No full semantic reading of all source histories is claimed.

| Case | Finding |
| --- | --- |
| Laptop-backpack delivery interval | Purchase and arrival dates already occur verbatim in the old packet; other six proposals are irrelevant. Cross-session item identity is plausible but not item-ID certified. The year is inferred from observation context. |
| Wells Fargo pre-approval | Two different amounts at different dates are already in the old packet. Six other proposals provide no approval amount, including a literal-new car brake-pad cost. Which approval the question means remains unresolved. |
| Bike expenses | The one proposed dollar amount is already present. Six other turns do not establish spending; a prior tune-up has no fee in its proposed turn. Seven whole bodies already occur in the baseline. Mileage, plans and insurance advice are not expense amounts. |

No useful new answer information was established by these 24proposals. This sample is a reviewer-path smoke check, not a500-case semantic result, selection accuracy estimate or proof that a broader proposer cannot improve. It reinforces the need to check relevance, novelty and qualifications together. We did not generate or score any answers.

## Validation and decision

83 focused tests pass, including 32 new dossier tests; Ruff and diff checks pass. Tests cover distant qualifiers, identical text with different roles/dates, future dependencies, forged spans/hashes, unresolved/budget refusal, all obligations remaining unreviewed, and inert source framing/Unicode separators. Pure test collection disabled plugins/conftest and emitted one expected asyncio_mode warning. It did not collect paid legacy tests or touch live stores.

Keep the tool offline. The automatic proposer is still open; no new paid comparison package is ready. The existing source-readiness gate remains failed, and the next paid answer comparison still needs a reviewed positive candidate. Do not add further review schemas or packing variants merely to count implementation work as accuracy progress.

The founder asked whether Sarvam can start in parallel. **Recommend a bounded text-memory preparation track now, independently of reaching 90% English.** Five excluded English-control fixtures and evaluator-only expected states are drafted; native rendering/review, API adapter and actual live integration remain pending. This does not silently close English, lower its target or authorize credits/API use. See PARALLEL_PLAN.md for scope, working-session estimates and the separate English closure gates. No 90% date is supported. The old September calendar is historical.

Large dossiers remain local; compact public artifacts bind their hashes and preserve every result. Public script copies require the original memory-relative layout and frozen inputs. Publication and exact-head/main CI receipts are recorded separately after they finish.
