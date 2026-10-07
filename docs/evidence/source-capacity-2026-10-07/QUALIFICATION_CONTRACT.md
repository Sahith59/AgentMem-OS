# Next source-unit contract — proposed, no runtime candidate yet

October7,2026. This replaces another topic-ranking/packing sweep as the next engineering boundary. It is not a validated semantic classifier, new score, permission to spend, or claim of novelty.

## Problem and concrete behavior

The current unit is an entire original turn plus its immediate neighbors. It preserves local text but can make a short fact unaffordable because neighboring advice is long. A compact unit should quote only the needed original material **and all material needed to interpret that quote**. Example: “I replaced it on Monday, only after the refund” cannot become “I replaced it”; “it” requires its original referent, Monday needs the observation/event-time context, and the refund condition must survive. The speaker remains explicit. An assistant recommendation cannot become a completed user action.

## Separate mechanical and semantic contracts

Mechanical representation (implementation target): a source snapshot ID and hash; original message ID, session and ordinal; speaker and complete observation timestamp; ordered quote ranges with exact offsets/hashes; dependencies linking a quote to original qualifier/antecedent ranges; omission ranges; and status `unresolved` or `reviewed-for-this-question`. Copy original strings only. Keep relative-date wording; normalized dates are derived data with inputs and rule provenance, never replacements for the original. Different dated events cannot deduplicate by content. Cycles, unknown IDs, changed source hashes, future dependencies or unresolved references prevent compact admission. A known dependency too large for budget prevents compact admission; it is not silently discarded.

Semantic obligations (evaluation target, not guaranteed by schema): entity identity, event identity, role/speaker authority, actual-versus-planned status, negation/conditions/corrections, temporal reference/window, quantity membership and repeated-event identity. A fully valid receipt or planner claim cannot mark these obligations satisfied. Every critical dependency in the next source-review set must survive; otherwise retain the original bounded context or report that evidence does not fit. Retaining a +/-1 window is itself only local-context preservation, not proof of whole-history semantic sufficiency.

Runtime selection must receive only the question and eligible source snapshot. No reference answer, annotation flag, historical grade, manually labelled benchmark source ID or capacity-oracle witness can be passed to it. Review labels remain evaluator-only. Do not assume a fact's legacy line citation maps to a whole message; use verified original message IDs or fail closed.

## Adversarial contract before all500 output

Use paired non-benchmark examples with the same topic and different answers, covering at least:

1. completed purchase versus considered purchase;
2. same named object on different dates;
3. explicit correction versus repeated statement;
4. ambiguous pronoun versus resolved object;
5. prior job tenure versus current role tenure;
6. “last time” versus a named weekday;
7. assistant advice versus user action, and a question specifically asking for advice previously given;
8. received versus purchased items in a count;
9. future dependency versus eligible source;
10. ambiguous friend/entity versus an explicitly named companion;
11. a qualifier outside +/-1;
12. missing information versus explicit denial.

These test the contract and audit renderer. Passing synthetic checks is not semantic accuracy. Review fresh source decisions separately; preserve failed labels/outcomes rather than retuning a case-specific rule. The existing v3 planner's support/qualification fields and failed semantic validation are prior work, not evidence this harder dependency problem is solved.

## Sequential gates

1. Implement the source-unit contract and explicit unresolved behavior; independently audit source/date/offset/dependency integrity. No production promotion.
2. Freeze one general, label-blind method for proposing these units. Specify exactly how qualification claims are checked; if no credible semantic method exists, stop and state that limitation instead of inventing a heuristic certificate. Any model call requires a concrete separately approved package and must be counted; do not bypass the agreed offline prerequisite by calling it a free audit.
3. Build all500 paired packets only when a defensible proposer exists. Keep Luna/extraction/Terra, temporal contract and matched compute fixed. Audit evidence gained AND lost, including quantities, negation, attribution and controls. Do not silently loosen the prior>=10 historical-miss source-gain and>=5 net source-case spending-triage gates. Those gates are necessary triage, not sufficient proof; qualitative review can still reject a candidate.
4. Only a reviewed positive offline candidate supports a proposed fixed-Luna/Terra answer comparison. Freeze actual answer/control/abstention gates, no-retry limits, token accounting and spend before approval. Full500 and repeat follow demonstrated answer gains and their own approval.

Neither this design nor the source-capacity audit supplies a date for90%. Do not automatically switch to Sarvam or tell the founder a breakthrough is assured.
