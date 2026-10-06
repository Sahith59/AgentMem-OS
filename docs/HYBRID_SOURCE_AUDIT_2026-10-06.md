# Hybrid retrieval and source preservation: all-500 offline result

October 6, 2026. **Integration complete; candidate fails the prospective readiness gate. No paid answer comparison is proposed.** The historical full-set result remains 423/500 (84.6%, once). Terra's 425/500 is a regrade of those same answers. This experiment generated no answers and made no paid LLM calls.

## What changed

The existing multilingual-E5 semantic plus TF-IDF keyword retrieval now feeds the source-preserving packer through `ContextAssembler.assemble_hybrid_source_packet`. Each hit retains its original source ID, date, role, turn position and session. Ranking filters future sources before indexing and retains the original reciprocal-rank-fusion order, including ties. Packing records source hashes and exact output offsets.

The experiment keeps the entire historical baseline packet, including semantic facts present in 446/500 packets. It selects eight anchors before packing, considers adjacent turns within their original sessions, and appends at most 4,000 characters within a 40,000-character total. It never clips a source or backfills with lower-ranked anchors. This is an opt-in experimental entry point; normal assembly is unchanged and this losing policy is not promoted.

The retriever also rejects malformed, complex or nonfinite embeddings, clears failed reindex state, and rejects nonfinite similarity results before ranking. These are reliability fixes, not measured accuracy gains.

## Measured result

The [protocol](evidence/hybrid-source-2026-10-06/PLAN.md) was frozen before the outputs. Its threshold was at least **10 of the 77 historical misses** gaining a previously absent annotated source turn, zero losses, and complete integrity. This is an engineering spending gate, not a statistical claim or a promise of ten corrected answers.

| Measure | Result |
| --- | ---: |
| Questions audited | 500/500 |
| Original packets preserved byte-for-byte as prefixes | 500/500 |
| Packets receiving additions | 496 |
| Appended source turns | 2,831 |
| Appended turns whose complete dated text was already present | 2,259 (79.8%) |
| Newly present annotated turns | 16 across 13 questions |
| Historical misses gaining an annotated turn | **3/77**, four turns |
| Annotated turns lost | 0 |
| Questions with any annotated turn present | 427 → 433, out of 479 |
| Questions with every annotated turn present | 357 → 362, out of 479 |
| Readiness | **FAIL** |

There are 896 annotated turns across 479 questions. The dataset contains 30 abstention questions; 21 of them have no annotated turns. Abstentions are separately reported. Zero loss is expected from preserving the complete original prefix; it does not show that the extra text helps Luna. Text presence is a proxy, not proof of baseline source identity, semantic completeness or answer correctness. The gold-session presence proxy improves from 484 to 490 questions with any gold session represented and from 430 to 440 with all represented; it has the same limitations.

## What this tells us about the architecture

**Adding relevance-ranked turns mostly repeats existing context.** The current policy does not account for which sources the baseline already contains. The measured duplication gives a specific reason to investigate source-aware allocation. It does not establish that eliminating duplicates alone will fix answers.

Among the 77 historical misses, 73 annotated turns remain absent under the text-presence check: 66 were outside the selected eight anchors and their neighbors, six had recorded packing nonfit, and one was correctly excluded by the time cutoff. These are turn counts, not a decomposition of 77 answer errors. Sources outside the selected eight may still exist lower in the full ranking.

| Previously investigated example | Observed failure in this candidate |
| --- | --- |
| Kitchen replacements: coffee-maker donation | Needed turn is outside the selected eight and neighbors; remains absent. |
| April workshops: April 17–18 | Needed turn is outside the selected eight and neighbors; remains absent. |
| Current-role tenure: two intervals | One interval ranks in the top eight but does not fit; the other is outside the selected eight and neighbors. Both remain absent. |

The three historical misses gaining annotated turns concern phone charging accessories, magazine subscriptions and charity fundraising. The latter two already have recorded source/reference or category ambiguities in earlier reviews. Even these three are not three demonstrated corrected answers. The [mechanism review](evidence/hybrid-source-2026-10-06/mechanism-review.json) preserves exact sources and omission reasons.

## Verification and limits

- Independent review checked all 500 input/grade/label joins, 23,867 question-local source-session occurrences, 1,475 future occurrences excluded before indexing, and all 2,831 appended receipts. No date, scope, text, hash, role, offset or budget violation was found. Runtime inputs omit answers, gold IDs, evidence annotations and historical grades; evaluation loads them only after all packets exist.
- **148 focused tests passed**, including source boundaries, time filtering, failed reindexing, input validation and the actual assembler entry point. Ruff on the new code/tests and `git diff --check` passed. The isolated tests report one expected `asyncio_mode` warning because plugins are disabled. The legacy paid API collection test was excluded.
- **216 controlled comparisons** against actual pre-change retrieval code preserve old search outputs across ties, snippets, neighbor widths and depth settings. This is scoped parity evidence, not proof for every possible input. Those comparisons preceded the additional nonfinite-result guard; the guarded version reproduces all 500 real packet bytes exactly.
- Local inference used the same existing model ID, `intfloat/multilingual-e5-small`, pinned to revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Historical weights were not revision-pinned, so byte-identical historical weights cannot be claimed. No stronger embedding, extraction, answerer or judge model was introduced. The local encoder retains its 512-token limit; emitted source turns remain whole.
- Local embedding of 231,309 unique prefixed texts took about 16.6 minutes. The audit used NumPy 2.1.3 and scikit-learn 1.5.2; isolated tests used scikit-learn 1.8.0. Exact model files, environment versions, code and input hashes are preserved.
- The local matrix operation emitted numerical warnings. A replay checked 231,575 passage/query dot products: all finite, maximum absolute difference from direct float64 contraction approximately 1.25e-7. No top-eight ordering changed; 15 lower-rank orderings differed. The added finite-result guard produced byte-identical packets in a versioned rerun. The warning's root cause is not proven, and replay does not retrospectively certify unsaved original intermediate arrays. Both original and rerun artifacts remain preserved.

All 500 questions have been repeatedly inspected during development. This is not a hidden validation set. The audit certifies appended-source integrity, not a new temporal audit of the frozen baseline itself. No completeness claim is made about every repository file or the entire internet.

## Research response and next bounded task

The negative result triggered a [primary-source review of competitors and papers](SOURCE_SELECTION_RESEARCH_2026-10-06.md). Hindsight documents source-linked redundancy control; Graphiti preserves episode and temporal links; Mem0 and Letta expose related entity and memory-budget mechanisms. These precedents support a hypothesis, not a transferable leaderboard score or a novelty claim.

The next engineering task is **a source-presence ledger and selection diagnosis, without another answer run**:

1. Bind existing packet content to original scoped sources using stored provenance where available. Otherwise accept only unambiguous date/role/body matches. Record ambiguous mappings explicitly; never merge distinct dated events or equate a paraphrased fact with its complete original turn.
2. Measure which missing sources fall below the top-eight cutoff versus which lose space to already-present evidence. Do this across all 500 with evaluator-only annotations. Do not tune to answer strings or the three named examples.
3. Freeze one source-aware selection policy that spends the same budget on missing sources and their necessary qualifiers. Treat an incomplete source bundle as incomplete rather than silently dropping its qualification. Candidate recall and packet allocation must remain separate checks.
4. Re-audit gains, losses, controls and all added-source receipts before any paid proposal. Only a credible offline result can justify a prospective fixed-Luna comparison with matched compute and actual token/cost reporting. A passing answer comparison must precede full-500 measurement, repetition and broader validation.

Source pointers, fusion and deduplication are established ideas. A possible research contribution would be a tested selection/qualification contract with explicit failure reporting under fixed models and budgets. That contribution and a score above 90% are unproved. This turn closes the requested integration/audit task; it does not close the English baseline.

## Evidence

[All-500 per-question results](evidence/hybrid-source-2026-10-06/audit-v2.json), [independent verification](evidence/hybrid-source-2026-10-06/independent-verification.json), [numeric replay](evidence/hybrid-source-2026-10-06/numeric-audit.json), [hash manifest and reproduction notes](evidence/hybrid-source-2026-10-06/README.md).

Large local inputs, embeddings and complete packet text remain in `../codex-memory-2026-09-08/plans/2026-10-06-hybrid-source-audit/`. Both packet files have SHA256 `46caf43c3a3fdd22b7860c4b69d9616ca1d59e48dcafcabe798d95d805e71680`. No historical results were overwritten.
