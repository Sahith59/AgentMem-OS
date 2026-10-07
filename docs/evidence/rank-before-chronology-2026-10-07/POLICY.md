# Rank before chronology — frozen candidate policy

October 7, 2026. Starting main 97d55d3. Offline only; no paid/model/embedding calls or model changes. All 500 cases are development-exposed. Historical 423/500 is unchanged.

## Reason for the bounded change

Automatic semantic compaction remains unproved. Existing lossless-neighbor-reuse diagnosis already found one partial opportunity; do not repeat it or ship a partial parser. A new independent reproduction identifies a concrete remaining raw-context flaw: `_order_evidence` admits an overshooting low-ranked chunk, sorts it ahead by date, then `_fit_to_budget` can remove the higher-ranked chunk. This is the old rank-versus-presentation failure class; do not claim a new invention. Fact-tier protections do not establish raw-tier safety.

One opt-in policy `whole_rank_v1`: for ordinary semantic raw retrieval with no nonempty reserve receipt, admit complete input chunks in rank order only when the entire proposed rendered section fits BOTH the existing four-characters-per-token limit and the actual current TokenCounter limit. Account for framing and separators. Skip nonfitting chunks with a disposition, never clip. Render admitted chunks chronologically under the legacy date-detection rule (or rank order for recall), then deliver that exact already-budgeted section without a later cut. A later candidate cannot evict an earlier admitted chunk. Input chunks may already be transformed by their retriever: preservation is relative to input chunks, not an invented original-source certificate.

Legacy remains the default. Cases with an existing nonempty adapter reserve take the exact legacy path in both arms; reserve routing/size and all fact/profile/recent sections are outside this change. Fixed question, corpus, query rankings, lexical vocabulary, temporal semantics, other tiers, precision-source supplement, models and 40k final cap remain unchanged. No classification by question ID, reference, historical grade or annotated source.

## Verification and prospective gate

Reproduce the actual assembler rank-0 deletion, test exact character/token boundaries, dense-token Unicode, unknown/partial dates, oversize first chunk, malicious framing, deterministic receipts, no-op legacy/default and reserve fallback. Audit opt-in assembly and ensure complete chunks cannot be truncated after reorder.

Reconstruct all 500 legacy corrected contexts exactly through the existing real assembler and fixed copied DB, then reproduce the historical precision supplement. Fail on any baseline mismatch. Use a disposable DB, block network, hash source DB before/after, do not load evaluator labels/grades into runtime selection. Freeze script/source/input hashes before candidate output. Preserve controls, candidate contexts and input/output traces.

Only after all contexts exist, evaluate evidence gained AND lost against the historical 423/500 packet and matched legacy control. Retain prior spending triage: at least 10 historical misses gaining annotated source and net at least 5 source-case gains versus control, no annotated source losses across all 500. Review every gain and loss plus uncertainty/abstention cases even if a numerical gate passes. Count source-body presence separately from receipt-certified identity; partial clauses do not count as complete annotated turns. A failed gate stays failed, with no parameter sweep. This is a candidate source-delivery test, not answer accuracy or English closure. No paid proposal until reviewed source evidence supports it.
