# Luna v2 paid selection diagnostic

The approved 13-call run completed with no retries. **12 of 13 exposed development cases passed; the frozen all-case gate failed.** All responses satisfied the local schema and source-ID contract. This is not a 92.3% English benchmark result. No final answers or new grades were generated; historical 423/500 and provisional Terra 425/500 remain scores of the same saved answers.

## Evidence

- Frozen package: `92f773710f7509e1cd083e226ce70ef7b465c5a84b624652f98f2c63ea49b93c` (preparation-v2-002).
- Same Luna settings; final answerer, judge and extraction corpus unchanged.
- 13 attempts, 13 responses, 13 valid plans, 12 selection passes, one selection failure, zero retries.
- Approved cap $0.05; reservation $0.0462113; receipt-derived conservative usage estimate $0.0037253, not an invoice.
- Recomputed summary and approval binding; replayed all 13 outputs through the renderer. All baseline packets remained intact and every appended source receipt matched its original text.
- Original raw provider responses, frozen labels, approval and checkpoint remain preserved in workspace memory. Public verification: [JSON](results/luna-selector-v2-2026-10-04.json).

The magazine case now selects only magazine evidence and rejects the book box. This is an encouraging observation on one exposed case; the previous run stopped there, so this is not a complete paired v1/v2 comparison or causal ablation.

## Remaining miss: evidence for uncertainty

The question asks which museum the user visited. One source is an assistant suggestion to visit Stone Museum; the other is the user's statement that they had not decided where to go. Luna rejected both sources, selected nothing and marked `uncertain`. The frozen fixture requires the user's uncertainty statement to be retained as supporting evidence, so the gate fails.

This is an evidence-selection miss, not an observed wrong museum answer. The model did not claim a visit; no final answer was generated. Offline replay returns `UNCHANGED_EMPTY_SELECTION`, retaining the complete original packet. The user's undecided statement also does not prove no later visit occurred; the appropriate conclusion is insufficient information.

The policy already mentions necessary negations and evidence of missing facts. This result suggests tension between selecting sources that support a requested completed event and retaining sources that explain why it cannot be established. Merely adding another sentence is not demonstrated to fix it.

## Next bounded work

Keep this failed gate and its labels unchanged. Define a source-bound distinction between affirmative evidence, contradiction/uncertainty evidence, and irrelevant evidence before another selector version. Test cases where uncertainty later resolves, where a negative statement contains a useful fact, where only advice exists, and where missing evidence is genuinely absent. Preserve baseline fallback; never insert generated explanations as source facts or hardcode benchmark answers.

Separate frozen development membership checks from future answer-sufficiency evaluation with alternative acceptable evidence sets. A fresh set must be frozen before its first model outputs; assistant-authored review is not independent human validation. Do not tune on that validation set after scoring.

No further paid run is authorized by this completed package. A paired fixed-Luna answer screen with an equal-cost extra-call control remains downstream of selection validation. Full500/repeat and disjoint transfer remain later gates. No architecture promotion, English closure or 90% claim follows from this run.
