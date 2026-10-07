# Source-aware selection: measured engineering gains, accuracy still unproved

October 6, 2026, America/New_York. **Two all-500 offline comparisons are complete. Neither passes its frozen readiness gate. No paid run is proposed.** Historical full-set accuracy remains **423/500 (84.6%, once)**: all500 saved answers identify `gpt-5.6-luna`, and all500 original grades identify `gpt-4o-2024-08-06`. Terra's 425/500 is a regrade of those same answers. GPT-4o remains retired for future execution. There are no new model answers in this work.

## What improved

The opt-in assembler now recognizes complete original sources already present in a historical context, with exact text offsets, hashes, date, role and session identity. It spends retrieval slots on other sources and admits an anchor only when its original adjacent turns are available together. Ambiguous or unsafe historical formatting cannot silently certify a source. Normal production assembly remains unchanged.

This addresses two concrete flaws: repeatedly spending context on the same source, and allowing an anchor to arrive without its local qualification. It certifies syntactic neighborhoods, not semantic completeness. Some qualifications occur farther away.

The first parser version had real identity defects: a short source could be confused with the prefix of a longer one, and a source could imitate section framing. Independent review caught these; the corrected v2 rejects ambiguous prefixes and malformed/nested frames. Original v1 artifacts remain preserved and marked uncertified. No original answer or grade was overwritten.

## All-500 source-aware comparison

Both arms use the same existing E5 vectors, TF-IDF512/RRFk60 ranking, eight anchors, whole original turns, atomic +/-1 neighborhoods, 4,000 extra characters and 40,000 total. The control does not skip or reuse already-present sources; the candidate does. The original baseline remains the complete prefix in both arms.

| Measure | Matched control | Source-aware candidate |
| --- | ---: | ---: |
| Appended source turns | 1,666 | 1,733 |
| Appended turns whose complete dated text already occurs in baseline | 991 (59.48%) | 14 (0.81%) |
| Historical misses gaining annotated text vs original baseline | 1/77 | 7/77 |
| Questions with all annotated turns present, of 479 | 359 | 365 |
| Historical-miss cases gaining/losing coverage vs control | — | 6 gained / 0 lost |

Candidate gains24 annotated turns across23 questions vs baseline. There are no baseline text losses. Relative to control, however, five historically correct cases lose six annotated turns that the control had added. Source review finds the core answer facts still in the baseline; answer stability is nevertheless unmeasured. Abstention examples also acquire distractors.

The prospective gate required at least10 historical misses gaining annotated text vs baseline and at least5 net historical-miss gains vs control, plus integrity and source review. **Seven is below ten: FAIL.** Reducing duplicates is a measured engineering improvement, not a measured accuracy increase.

## Which gains actually look useful?

All seven historical-miss gains were reviewed against the saved answer and original sources. Only three are plausible missing-fact repairs vs baseline; only two distinguish the candidate from matched control.

| Example | Missing information delivered | Scope of inference |
| --- | --- | --- |
| February museum visits (`80ec1f4f`) | Natural History Museum visit dated February 8 | Candidate-specific; could support two visits instead of the saved answer's one. |
| Rare collections (`e3038f8c`) | 12 rare figurines | Candidate-specific; could complete the saved sum of87 to99. |
| March bicycle service (`a9f6b44c`) | Commuter-bike tire replacement planned that month | Delivered by both arms; not evidence of candidate superiority. |

The four other gains repeat known facts or leave chronology/category/reference ambiguity unresolved. None of these are new correct model answers. The stored original sources support the two candidate-specific opportunities; a fresh fixed-Luna comparison would still be needed to establish that Luna uses them correctly.

## A ranking defect tested and rejected as the explanation

Legacy fusion gives every document a keyword rank vote, including zero-TF-IDF matches. A single opt-in policy removes only those zero-match keyword votes while retaining every dense vote. It changes no model, vocabulary, context budget or source rule. This is motivated by the result-membership definition in [Elastic's RRF documentation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), not a claimed new retrieval invention.

The frozen diagnostic found48 selected zero-keyword anchors across26 questions, nine across six historical misses. The follow-up exactly reproduces all500 corrected source-aware controls. It changes27 anchor lists and11 packets, including two historical-miss packets. **It gains zero and loses zero annotated turns versus control. Readiness FAIL.** It stays opt-in and is not promoted as an accuracy improvement.

The27-versus26 difference is explained:25 zero-anchor cases change selection, one keeps the same anchors, and two further cases only swap equal-score positive anchors under NumPy's unstable tie ordering. Those two swaps change neither membership nor packets. The two changed miss packets only exchange unrelated interview/personality material or remove an unrelated greeting. No missing answer fact is recovered.

## Where the remaining architecture work is concentrated

Among initially absent annotated turns in the77 historical misses, the source-aware audit finds43 still outside selected neighborhoods,14 whose whole neighborhood exceeds available context,12 competing for the remaining space, seven delivered and one excluded by the timestamp cutoff. These are turn counts, not an attribution of all77 answer failures.

Assistant turns consume150,945 of179,587 appended body characters in these misses:84.1%. This explains a concrete packing pressure. It does not show those turns are irrelevant or authorize deleting them. Earlier experiments already showed that naked hits lose dates and other qualifications.

The next bounded design should preserve linked original evidence while selecting the necessary surrounding context more precisely. The completed review below distinguishes necessary qualifications from generic assistant prose before another packet policy is designed. Do not sweep window sizes, remove all assistant turns, enlarge budgets or substitute models to manufacture a gain. The separate43-turn retrieval-recall problem remains visible.

The subsequent source review is complete for all26 budget-blocked turns, across21 historical misses. Every target is a user turn and fits its empty addition allowance with attribution:299–693 characters, against1,382–3,998 available. Fourteen cases have a plausible bounded qualification; seven remain uncertain. Review identifies10 plausible missing-evidence cases, four redundant or unproved repairs, six unresolved reference/identity cases, and one guitar case with relevant assistant comparisons. These are qualitative categories, not recovered answers. A target fitting an empty budget does not show it fits after earlier admissions or suffices for the whole question. No new candidate packets were constructed from this review.

This gives a concrete basis for the next architecture task: exact source spans linked to their dates, entities, conditions and surrounding references, with a refusal to compress when those links are uncertain. Automatic user-only clipping is rejected. The [case-by-case review](evidence/source-aware-2026-10-06/qualification-budget-review.json) preserves every target and neighboring source excerpt. The next experiment still requires a general, label-blind policy and a fresh matched audit.

## A benchmark time-contract issue also found

An independent audit finds908 complete dated source occurrences already in the frozen baseline that are later by clock time than the question, across67 cases. All are on the same calendar day. This does not retrospectively certify their source identity or prove leakage/cheating.

The official [timestamp generator](https://github.com/xiaowu0162/LongMemEval/blob/main/data/custom_history/sample_haystack_and_timestamp.py) assigns randomized clock times, and the [generation code](https://github.com/xiaowu0162/LongMemEval/blob/main/src/generation/run_generation.py) consumes provided history without a new minute-level filter in the examined history assembly. Together these support a benchmark timestamp-granularity mismatch, rather than treating every later same-day record as genuinely future information. That interpretation must remain distinct from a production strict-as-of contract.

No timestamp policy or prior grade changed here. Before a future paid package, explicitly state whether it evaluates benchmark-provided history or strict production as-of availability; freeze and disclose that input contract in both arms. Do not combine that change with the next packing ablation.

## Verification and practical next steps

Independent verification covers500 pairs,15,168 reconstructed presence certificates and3,399 appended receipts for corrected source-aware v2. The lexical comparison separately verifies500 reproduced controls,30,336 certificates across both arms and3,448 appended receipts. Scope, date, full text, hashes, offsets, roles, positions, original prefixes, budgets and admitted syntactic bundles pass. Known-renderer reconstruction is conservative evidence mapping, not cryptographic producer provenance for arbitrary caller input.

After the final retrieval change,107 focused offline tests and216 legacy-code parity comparisons pass. Earlier source-presence implementation passed186 broader focused tests. Core defaults remain unchanged. Ruff on all six new/changed code and test files and diff checks pass. Final lint caught one overlong signature; it was wrapped without changing the Python AST, with both source hashes recorded in `format-only-equivalence.json`. Frozen code/output files were not rewritten. Existing numerical matmul warnings still occur in the cached environment; finite guards pass, and the earlier independent float64 replay found no top-eight changes. The warning root cause remains unresolved. All500 questions are exposed development data; no hidden validation, research novelty or industry superiority is claimed.

For the founder's two-day sprint, finish the qualification/budget diagnosis and one justified candidate first. Only a source-reviewed positive offline gate should lead to a separately approved, frozen Luna/Terra answer comparison, with the same models, prompt, call count, context ceilings and realized token/cost accounting. A passing comparison must precede full500 and a repeat. Two days is a decision checkpoint;90% cannot be promised. Sarvam remains parked until the founder's closure decision. No further worksheet is assigned to the founder and no paid authority is assumed.

Evidence: corrected `audit-v2.json`, `independent-verification.json`, `qualitative-review.json`, `temporal-contract-audit.json`; lexical `audit-lexical.json` and `lexical-independent-verification.json`; frozen policies, scripts and hashes. Full packets and embedding inputs remain in local memory and are hash-bound, not uploaded. See the public evidence bundle for exact file locations and reproduction limits.
