# Frozen complete-bundle allocation policy

October 7, 2026, America/New_York. IN_PROGRESS implementation; policy frozen before new candidate outputs. Starting main343b004 (PR38). Prior source-aware and lexical gates stay failed. All500 questions are exposed development data. No paid calls, new embeddings, generated summaries or model substitutions.

## One intervention

Retain the existing legacy E5+TF-IDF512/RRFk60 full ranking, conservative whole-source presence ledger, first8 novel anchors, original-session +/-1 atomic neighborhoods, complete baseline prefix, chronological whole-turn rendering, strict observed_at<=question_time eligibility,4,000 extra characters and40,000 total. Positive-only lexical option remains off. Models, prompts and generation budgets unchanged.

Change only admission: enumerate at most256 subsets of the fixed8 anchors. Each structurally valid anchor requires its complete eligible original-position neighborhood; gaps or future neighbors make it invalid. For each subset, union the required source IDs minus sources certified in the baseline. Charge every appended source exactly once, using the existing renderer's exact character cost including framing and the two-character baseline separator.

A packet covers an anchor only if ALL of its required sources are in certified baseline sources or emitted whole. Score all fixed top8 novel anchors whose bundles are covered, including incidental bundle completion, once each. Utility is math.fsum of their existing RRF scores, in original anchor order. No reward for unranked neighboring source IDs. Maximize utility among budget-fitting subsets; break ties by smaller actual rendered character cost, then lexicographically earlier covered-anchor ranks, then seed ranks, then stable source IDs. Empty packet is a candidate. Input ordering and source metadata remain deterministic. Report seed subset, covered anchors, utility, considered subsets, actual costs and every excluded anchor reason. Greedy default stays unchanged.

This objective can prefer several shorter lower-ranked bundles over one larger high-ranked bundle. Retrieval score is a relevance proxy, not correctness or sufficiency. All original text is preserved for each admitted neighborhood, but distractors and reference errors remain possible. Do not claim semantic completeness merely because neighbors are whole. This mechanism may help12 competition-blocked annotated turns across11 misses; it cannot rescue14 individually oversized bundles by shrinking text. No outcome-guided objective weights or alternate policy sweep.

## Matched control and gates

Control reproduces the corrected Oct6 source-aware v2 candidate: same inputs/rank/ledger/top8/bundles/budgets/rendering, greedy admission. Require byte-identical control packets and source receipts all500 before trusting comparison. Both arms build entirely before evaluator labels/historical grades load.

Readiness requires at least10/77 historical miss cases gaining annotated whole-turn text vs original baseline, at least5 net historical miss cases gaining minus losing vs matched greedy control, zero baseline losses, complete scope/time/offset/role/hash/budget checks and full source coverage for every admitted atomic bundle. Independently review all historical miss gains/losses, every historically-correct control-relative loss, and changed abstention cases. These are engineering spending-triage gates, not predicted answer gains or statistical significance. No paid proposal follows a failed gate; freeze and preserve the failure.

No semantic clipping, user-only filter, vocabulary change, larger context, answer/judge upgrade, timestamp relaxation, case-ID rule or gold phrase selection. A separate source-span design would need its own semantic contract and evidence; it is not smuggled into this allocation test.

## Paid timeline

No paid authorization is active. First complete this offline implementation/audit. Only a credible reviewed result supports a concrete fixed-Luna/Terra paired answer package and its separately requested approval. Only that comparison passing supports full500 and repeat. October8–9 is a conditional planning window, not a scheduled run or promise: an offline/screen failure postpones full500. No fixed date for90% is supported.
