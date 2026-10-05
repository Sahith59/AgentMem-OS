# Source evidence roles: offline v3

The paid v2 selector passed12/13 exposed development checks. Its remaining miss was to reject the user's undecided statement while correctly marking the requested museum visit uncertain. This version adds a separate place for qualifying evidence. **No v3 model run or English accuracy gain has been measured.** V1/v2 source, fixtures and paid results remain unchanged.

## Boundary and behavior

The authoritative structured-output schema is constructed in `benchmarks/luna_evidence_plan_v3.py:plan_request`; the parser enforces its shape and additional source constraints. The provider is the same fixed Luna planner. The consumer is the original-source focus renderer. No new answerer, judge, extractor or production default is introduced.

- `support_turn_ids`: sources directly supporting the requested fact.
- `qualification_turn_ids`: relevant negation, uncertainty, correction or conflicting evidence needed to interpret the fact. A mixed turn goes here once, with its complete original text.
- `rejected_turn_ids`: considered but irrelevant sources. Advice may be relevant when the question asks what was advised.
- Omitted sources remain explicitly unclassified in audit output; these lists do not establish exhaustive review.

Lists must be disjoint, known, unique and ordered by the source input. Support and qualification have a combined eight-turn limit. Their union is rendered in original source order. Qualification-only evidence can support a definite negative answer; it does not automatically force uncertainty. Neither source order nor an earlier undecided statement proves what later happened.

The model's classifications and sufficiency are unverified claims. Only source text plus existing trusted ID/speaker/date prefixes enter the appended context. Generated requirements, classifications and rejection explanations are audit-only. The original packet stays intact, including rejected sources. If the entire focus block does not fit, append none of it: never discard a selected qualification to fit its supporting claim. Empty selection also retains the baseline unchanged.

## Verification and limits

108 focused offline tests pass, including23 new v3 checks. Coverage includes qualification-only and mixed turns, source-order union, exact source spans, no evaluator metadata in requests, no generated facts in rendered context, malformed/overlapping roles, total source limits, and whole-block fallback. New files pass Ruff. The original frozen v2 package reconstructs with canonical SHA `92f773710f7509e1cd083e226ce70ef7b465c5a84b624652f98f2c63ea49b93c`.

These are manually supplied response tests, not evidence that Luna assigns roles correctly. A deliberately incorrect but schema-valid selection remains accepted and labelled uncertified. The implementation cannot detect an important omitted ninth source or certify sufficiency; later semantic evaluation must measure these failures. This is an opt-in experimental adapter, not a production architecture replacement.

## Next measurement

Preserve the v2 failed all-case gate. In a future v3 diagnostic, score the rendered support/qualification union against the same frozen required/excluded IDs; separately assess roles and uncertainty. Never replace old labels to turn12/13 into a pass.

Before further inference, freeze source-reviewed cases for: unresolved uncertainty, uncertainty followed by completion, explicit negative answers, mixed negative/positive statements, corrections, unresolved conflicts, advice queried as advice versus action, and a critical source outside the eight-turn focus allowance. Allow alternative sufficient evidence sets where justified before outputs exist. Newly authored development examples are not an independent holdout; freeze a separate semantic validation set before its first outputs and do not tune on its result.

Keep the next paid diagnostic bounded, priced and separately authorized. No new paid package is prepared by this module alone. Only measured selection success justifies the paired fixed-Luna answer screen with an equal-cost extra-call control. English scores remain423/500 historical and425/500 under provisional Terra on the same saved answers; the judge difference is not architectural improvement. No 90% date is established.
