# Original-source units with declared dependencies

October7,2026. **The mechanical compiler is implemented and independently verified. Semantic selection and answer gains are still unproved.** English remains423/500 (84.6%, once);425/500 is a Terra regrade of the same answers. No paid calls or model/corpus/default changes.

A quote such as “I replaced it on Monday, only after the refund” needs its condition, date context and the identity of “it”. The new audit-only compiler can keep that quote and its declared dependencies together, including a referent outside the immediate neighbor window. If any declared part is unresolved, future-dated or too large, the entire unit is refused. It never drops a declared qualifier merely to fit the fact.

- [Canonical input types](../llm/source_unit_contract.py): immutable original quote spans and a rooted dependency graph, bound to the snapshot, exact question and cutoff.
- [Compiler](../llm/source_unit_compiler.py): exact text/hash/offset/role/date checks; cycle, overlap and orphan rejection; complete source omission ranges; atomic budget refusal.
- Output is a JSONL preview with reversible quoted text and a companion provenance report. Keep both together; receipt offsets refer to encoded JSON string tokens and decoded substrings. Generated dependency interpretations are never historical facts.

Every preview reports `semantic_completeness=NOT_CERTIFIED` and `answer_path_eligible=false`. There is no automatic source-unit proposer, model request, answer-packet adapter or production promotion. An undeclared critical negation can still be missing; the tests deliberately demonstrate that structural validity does not establish meaning.

Validation:

- 103 focused tests and Ruff/diff checks pass.
- Independent code review and700 additional adversarial probe cases pass.
- All500 real-source snapshots replay:231575 eligible source-turn occurrences,2008 batches,499 budget-refusal probes and76 future-source probes. Independent verification checks every preview/report hash, decoded text, original metadata and receipt. One case has no eligible sources.

The real replay uses full original turns and a deliberately large structural-test cap; its largest preview is276069characters. It does **not** demonstrate4k/40k answer-budget fit, successful source selection or improved answers. No labels, reference answers or historical grades enter it. All500 remain development-exposed; review is by an assistant, not independent human calibration.

The next open task is a general label-blind proposer with a credible method for checking semantic qualifications. Only a source-reviewed positive candidate supports the smaller fixed-Luna/Terra answer comparison. Actual answer gains must precede separately approved full500 and repeat. No90% date or English closure follows from this component.

[Detailed report](evidence/source-unit-compiler-2026-10-07/REPORT.md), [frozen structural audit](evidence/source-unit-compiler-2026-10-07/audit.json), [independent code review](evidence/source-unit-compiler-2026-10-07/independent-review.json), [independent replay](evidence/source-unit-compiler-2026-10-07/stress-verification.json), [artifact manifest](evidence/source-unit-compiler-2026-10-07/manifest.json).
