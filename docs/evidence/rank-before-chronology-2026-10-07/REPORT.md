# Rank before chronology: verified packing fix, failed source-readiness gate

October 7, 2026. Implementation d5689bf; starting main 97d55d3. **Mechanical repair verified. No answer gain measured; no paid proposal.** Historical English remains 423/500 (84.6%, once); Terra 425/500 regrades those same answers. Models and extraction corpus are unchanged; GPT-4o remains retired.

## Problem and change

The ordinary raw-context path can admit a final chunk that exceeds its budget, reorder chunks chronologically, and then truncate the result from the head. This can remove a fitting, higher-ranked newer chunk. The failure class was known; dedicated fact/reserve protections did not cover this remaining ordinary path. This is not a novelty claim.

The opt-in `ContextAssembler(raw_evidence_policy="whole_rank_v1")` admits complete input chunks in retrieval order only when the fully rendered section fits both actual TokenCounter tokens and the existing four-characters-per-token limit. Framing, separators and final chronology count toward both budgets. A later chunk cannot evict an earlier accepted chunk. Each input has a disposition; each accepted occurrence has an exact text/offset/hash receipt. There is no subsequent raw-section cut.

Legacy remains default. A nonempty existing adapter reserve uses the byte-identical legacy path. Retrieval rankings, other tiers, models, corpus, historical time semantics, precision-supplement algorithm and final 40,000-character cap stay fixed. Complete input chunks may already be transformed by their retriever; their receipts are not original-source identity or semantic qualification certificates.

## All 500 replay

The audit first reconstructs all 500 historical corrected contexts AND all 500 precision-supplement contexts exactly. It then generates candidates using the same retrieved inputs and adapter receipts. Frozen source/policy/input hashes, label-free runtime projection, blocked network and a disposable DB separate generation from evaluator-only labels. Source and copied DB hashes are unchanged.

442 ordinary cases change;58 reserve cases fall back exactly. Precision-supplement source membership changes in64 cases downstream because space/presence changes; do not describe those final packets as changing only raw text. All 500 final packets fit40k; no candidate raw section is cut by the outer cap.

| Cohort | Cases | Gain cases / turns | Loss cases / turns |
| --- | ---: | ---: | ---: |
| All | 500 | 12 / 12 | 0 / 0 |
| Historical misses | 77 | 5 / 5 | 0 / 0 |
| Historical correct | 423 | 7 / 7 | 0 / 0 |
| Abstention subset | 30 | 0 / 0 | 0 / 0 |

The frozen source-readiness gate **FAILS**: at least 10 historical miss gains were required; only 5 occur. Net 5 and zero annotated losses pass their separate conditions. No threshold change, parameter sweep or default promotion follows. Annotated complete-body presence is a proxy, not semantic sufficiency or an answer score. No annotated losses is not proof that no useful unannotated text was lost.

The raw-section rank invariant is repaired in10 real cases: a fitting rank 0chunk missing from legacy raw text is fully retained. Six newly reach the final packet. Qualitative review finds none directly supplies the requested answer fact: examples include Osiris mythology for Netflix viewing hours and cat-food planning for a user's weight loss. Rank preservation is useful engineering, but rank is not answer relevance.

## Review of all 12 gains

Seven gains add original support to already-correct historical answers. The five historical misses were reviewed against the original sources, old/new packets and saved answer:

| Case | Actual implication |
| --- | --- |
| 9d25d4e0, jewelry | Added emerald-earring purchase already appears in an extracted fact and another original user turn. Engagement-ring acquisition timing remains missing as a complete annotated turn. |
| ba358f49, age at Rachel's wedding | Adds missing age 32; baseline already contains wedding next year. Plausible missing operand, but exact age 33 depends on unspecified birthday/wedding timing. Age source 16:14 precedes question 23:52 on September 1, 2022. No answer repair measured. |
| bf659f65, albums/EPs | Added Midnight Sky purchase already appears in multiple facts. Signed-vinyl ownership and purchase/download scope still need interpretation; not a demonstrated third purchase. |
| d851d5ba, charity | Added $2,000 already appears in facts. Extra benefit-concert amount over $5,000 leaves the historical over $8,750 versus reference $3,750 scope dispute unresolved. No regrade. |
| e3038f8c, rare items | Added five books already appears in facts and saved answer. Original12figurines source is still absent; adding five books again does not fill that gap. |

Root review and separate GPT-6 Sol review agree: four gains repeat known content or leave a dispute unresolved; one adds a plausible missing operand with inference uncertainty. This does not predict one, five or six corrected answers. All 30 abstention cases have zero annotated gains/losses; changed contexts still need answer-level abstention validation. We have not claimed full semantic review of every changed byte across442 cases.

## Validation and preserved correction

134 focused tests pass; one live-Redis opt-in test is skipped. New files pass Ruff; seven pre-existing assembler lint diagnostics are unchanged. Diff checks pass. Independent review checks800 synthetic cases. Independent all 500 replay checks 13,138 raw receipts,58 fallbacks,652 legacy / 598 candidate precision receipts, exact inputs, controls, token/character limits and hashes. Separate evaluation replay checks all896 annotated occurrences and reproduces the failed gate. Final semantic review is separate from structural verification and is assistant review, not independent human annotation.

The first post-generation evaluator failed because Python `str.splitlines()` splits literal Unicode separators inside valid JSON strings. The original runner, freeze, traces and failed log remain intact. `evaluate_v2.py` and `mechanism_audit_v2.py` read physical file lines instead. No scoring rule, candidate packet, runtime policy or generation changed; no output regeneration occurred. `reader-correction.json` binds both versions and input hashes.

## Decision and next boundary

Keep this bounded fix opt-in. Do not spend on the current candidate or interpret source-count gains as new facts. Return to the unresolved general source-unit proposer: select missing answer-relevant facts with original entity/event/role/date/condition/reference/quantity context, and use the existing compiler to preserve declared dependencies. Do not add another packing-weight sweep or repeat the rejected fact-source bridge. Evaluation must distinguish a newly included source from information already present elsewhere in the packet and flag distractors/reference disputes. Evaluator-only witnesses and IDs must never enter runtime selection.

A reviewed positive source candidate must precede a smaller fixed-Luna/Terra answer comparison under a matched compute budget. Actual answer gains must precede separately approved full500 and repeat. All 500 are development-exposed; same-day benchmark time-contract ambiguity remains open. This change does not close the English baseline, establish 90%, or activate Sarvam. Paid500 is unscheduled. No paid calls, new embeddings or new answers occurred.

Public compact artifacts preserve results and hashes; large projection, DB, full contexts/traces and historical answer review inputs remain local. See the artifact manifest for bindings. Publication/CI receipts are recorded separately after checks actually finish.
