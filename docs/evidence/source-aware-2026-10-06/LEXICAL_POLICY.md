# Frozen positive lexical voting experiment

October6,2026 America/New_York. One follow-up after the corrected source-aware candidate failed its frozen gate (7 historical misses gained annotated text, threshold10). Preserve that failure. Source review finds3 plausible missing-fact repairs vs baseline; only2 are candidate-specific vs matched control. No new answers.

Observed mechanism: legacy RRF assigns keyword rank contributions to every indexed document, including zero TFIDF matches. Offline diagnosis:36,318 zero-match passages/231,575;48 selected zero-match anchors across26 questions, including9anchors across6historical misses. Zero lexical score does not imply irrelevance. No zero-query-vector case among499 nonempty scopes;1empty scope. TFIDF512 drops some query terms in493cases, but vocabulary changes are OUT OF SCOPE.

One opt-in change: leave all dense rank contributions unchanged; contribute lexical rank credit only for strictly positive TFIDF cosine scores. Keep existing descending NumPy ordering and k60, encoder and vectors, TFIDF512, cutoff, baseline, conservative ledger, first8 novel anchors, atomic +/-1 neighborhoods, <=4000extra/40000total characters unchanged. Dense-only sources remain eligible. Legacy default stays unchanged. No weighting, new model, relevance threshold tuning, role deletion, vocabulary expansion or parameter sweep.

Matched control is exactly the corrected source-aware v2 candidate, reproduced with the legacy ranker. Candidate differs only by positive lexical voting. Freeze code/script/input hashes before generating500 pairs. Runtime receives no gold/reference/grades; evaluate annotations only after all packets exist.

Readiness requires >=10 historical misses gaining annotated whole-turn text vs original baseline, >=5 net historical miss cases gaining minus losing vs this matched control, zero baseline losses, complete scope/time/offset/budget/bundle checks, and source review of every changed historical miss plus correct/abstention regressions. These remain spending-triage criteria, not answer gains or significance. A negative result remains negative; do not revise criteria after observing results. No paid calls, new answers or automatic promotion follow this offline run.

Rationale source: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion describes contributions only for documents belonging to an individual retriever result set. Applying positive lexical membership here is a testable local design choice, not proof this will improve answers or a novel research claim.
