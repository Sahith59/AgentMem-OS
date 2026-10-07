# Raw context: admit by rank before presenting chronologically

October 7,2026. **An opt-in truncation fix is verified; the paid-run readiness gate fails.** Historical English remains 423/500 (84.6%, once). No new answers, paid calls, embeddings or model/corpus changes.

Ordinary raw evidence could exceed its section budget, be sorted by date, and then lose a higher-ranked newer chunk in the final cut. [The new packer](../llm/ranked_chunk_packing.py) admits whole input chunks only when the final rendered section fits both token and character budgets, including framing. It preserves accepted chunks without a later raw-section cut and records dispositions and exact receipts. `ContextAssembler(raw_evidence_policy="whole_rank_v1")` enables it; legacy stays default and existing nonempty reserve paths retain legacy behavior. Input-chunk preservation is not original-source or semantic certification.

All 500 historical corrected and precision contexts reproduce exactly. The candidate changes442ordinary cases and keeps58 reserve cases unchanged. It gains12 annotated turns and loses none; only5 gaining cases were historical misses, below the frozen10 minimum. Four of those repeat already-known information or leave a scope dispute. One adds a plausible age operand with timing uncertainty. Repairing10 real rank-0raw deletions does not establish10useful facts: six newly reach final context, and none directly states its question's answer.

134 focused tests pass with one intentional live-Redis skip. Independent review includes800 synthetic cases, all 500 structural replay, all896 annotated occurrences and separate semantic review. The first post-generation reader failed on literal Unicode separators; preserved artifacts and a reader-only correction are documented. No candidate regeneration, threshold adjustment or default promotion occurred.

Next source selection must preserve needed qualifications and deliver genuinely missing information. Only a reviewed positive candidate supports a smaller fixed-Luna/Terra matched answer comparison, before separately approved full500 and repeat. No 90% result/date, English closure or Sarvam activation is claimed.

[Detailed report](evidence/rank-before-chronology-2026-10-07/REPORT.md), [all 500 audit](evidence/rank-before-chronology-2026-10-07/audit.json), [semantic review](evidence/rank-before-chronology-2026-10-07/independent-semantic-review.json), [manifest](evidence/rank-before-chronology-2026-10-07/manifest.json).
