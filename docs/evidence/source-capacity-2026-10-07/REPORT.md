# Why another packing tweak is insufficient

October 7, 2026. The new all500 source-capacity audit is complete and independently verified. **No accuracy improvement, runtime selector or paid package is produced.** Historical English remains423/500 (84.6%, once); Terra's425/500 regrades the same answers. No calls or model changes. Prior failed source-selection gates remain failed.

## What was measured

We gave an evaluator perfect knowledge of which annotated turn bodies are missing, then asked what the existing representation could possibly fit. This intentionally uses evaluation labels and is **not permitted as runtime selection or as a paid answer packet**. It searches exact unions of complete original +/-1 neighborhoods, reuses only existing certified baseline sources, preserves source roles/dates/text, rejects gaps/future neighbors, and counts all headers/framing under the unchanged4k addition/40k total limits. It computes the cheapest complete cover, including cases where a neighboring anchor covers the target more cheaply than the target's own window.

Of77 historical misses,42 lack at least one annotated turn body;29 already contain every annotated body;6 have no annotated turns. Neither annotations nor exact-text coverage establish semantic sufficiency or the cause of a wrong answer. In particular,80 historical correct cases also lack some annotated turn bodies, so all-annotation coverage is demonstrably not necessary for every correct answer.

| Missing-source cases among the42 | Existing first8 novel anchors | Any eligible original anchor |
| --- | ---: | ---: |
| Can fit at least one missing annotated turn |16|31|
| Can fit every missing annotated turn |5|16|
| Cannot reach any missing turn through a valid bundle |17|1|
| Can reach one, but cheapest bundle exceeds budget |9|10|

For all500,122 cases lack annotated bodies (214 missing turns). Any/all delivery fits in50/23 under the fixed pool, or87/51 with oracle ranking. This is a source-capacity diagnosis, **not a predicted benchmark score or a total accuracy ceiling**. Some annotations are redundant or ambiguous, alternate evidence can suffice, and Luna can still fail with the relevant source present.

The current84.6% requires27 net additional correct answers to reach90%,28 to exceed it. This audit does not prove90% impossible; it shows that improving missing-annotation delivery within the current top8/full-window family reaches only16 missed cases, even before usefulness and answer regressions. That mechanism alone cannot justify a27-answer improvement forecast. Other answer-use mechanisms remain open.

## What the diagnosis rules out

- More RRF-score allocation variants do not change this pool/budget limit. The prior optimizer already lost the March commuter-bike repair plan despite higher relevance utility.
- More raw text is not equivalent to useful evidence. Existing packets have median36,366 characters;342/500 already use at least36k. The median raw semantic-memory section is20,552 characters.
- Exact duplicate removal is too sparse to be a broad solution: repeated complete body strings occur in42/500 baselines, only5/77 misses (5,707 raw characters in those five before replacement framing). These are optimistic substring counts, not certified safe removals or semantic redundancy.
- A keyword/date parser cannot certify identity, prior-versus-current role, completed-versus-planned events, or what “last time” refers to. Do not implement case-specific synonyms or silently call lexical overlap a qualification certificate.

## Architecture consequence

There are separate retrieval and representation limits. A useful fact must be found across the eligible source pool; its original date, speaker, condition and referents must then fit. Preserving entire neighboring assistant responses is conservative but often expensive. Deleting those responses automatically is unsafe, as prior guitar, museum-companion, clinic and role-tenure reviews show.

The next source-unit contract must distinguish mechanical source binding from semantic qualification. Original offsets/hashes can certify an exact quote; they cannot certify that omitted context is unnecessary. A source unit therefore needs explicit entity/event/time/modality/role obligations, source-linked dependencies and an unresolved state. Unresolved references keep their context or refuse compaction. Generated explanations must never become historical facts.

The existing-fact-to-original-source index was checked and **rejected as a repeated experiment**. September decisionD-20260914-163 already measured a five-source bridge:10/65 missing annotations across9/34 frozen dossiers, without an answer gain; broader expansion added noise. The unchanged corpus contains100,865facts (read-only SQLite count verified). Legacy citations are line IDs produced by lexical overlap, not trustworthy semantic entailment or interchangeable original-message IDs. The stable message manifest explicitly requires a validated mapping or reconstruction. See fact-index-feasibility.json; no new bridge or citation remapping was executed.

The next implementation boundary is a qualification-preserving source unit, specified in QUALIFICATION_CONTRACT.md. Build its adversarial contract before another all500 selector: distinguish exact quotation integrity from semantic support; preserve linked dates/conditions/referents; retain full context or refuse compaction when a link is unresolved. Do not use a keyword rule or a model's “complete” flag as semantic proof. This contract is proposed, not an implemented successful selector. Retrieval recall remains a separate measured gap. A baseline-section replacement must be a separate arm with explicit losses; no silent40k expansion or gate relaxation.

## Verification and boundaries

37 focused tests pass, including capacity search versus exhaustive subset enumeration, shared qualifiers, Unicode costs, malformed inputs, future/gap rejection and existing atomic packing tests. Ruff and whitespace checks pass. An independent GPT-6 Sol verifier recomputed all500 minima using a separate per-target Cartesian search, checked every witness, source/presence binding, exact cost and summary. Evidence: audit.json, freeze.json, verification.json, POLICY.md and audit.py in this folder. Code is benchmark-only; no runtime caller or default changed.

All500 are exposed development data. This work is assistant review, not independent human calibration. Preserved baseline same-day timestamp discrepancies remain unresolved; new additions obey the existing strict cutoff. No gate, grade, model, corpus or cutoff was changed.

Paid sequence remains: reviewed positive offline runtime candidate -> concrete approved fixed-Luna/Terra matched answer comparison -> demonstrated answer gains -> separately approved full500 and repeat. No paid500 date is supported, and the historical roughly30-minute API runtime is not a multi-day prerequisite. Sarvam stays parked; no current paid authorization exists.
