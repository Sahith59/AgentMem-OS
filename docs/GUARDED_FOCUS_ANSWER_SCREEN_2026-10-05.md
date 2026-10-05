# Guarded focus: a final-answer experiment, not another label gate

Status: implemented and prepared offline; no paid calls or new English accuracy results. The v3 diagnostic remains failed (12/13 exposed,6/9 fresh semantic,1/1 capacity). We do not relabel it or claim that a conservative fallback fixes semantic selection.

## The bounded change

If a v3 plan is uncertain and classifies any assistant source as qualification, the guard declines the entire optional focus block. The original packet stays byte-identical, including all user and assistant sources. It never strips a selected qualifier while emphasizing its associated claim. Direct assistant support can still receive focus, including questions about recommendations. There are no question-name filters, answer strings or reference labels in this rule.

This addresses the observed extra-emphasis mechanism conservatively. It does not distinguish all advice from events: advice labelled support or a plan marked complete can pass through, and legitimate assistant qualifications may lose useful emphasis. Those are risks to measure. It cannot recover missing retrieved sources. Replay of all23saved v3outputs changes only the three advice-related focus blocks; all23baseline packets remain intact. That is offline behavior, not three corrected answers.

## Actual benchmark questions and controls

Select32of the existing500questions by ascending SHA256(`guarded-focus-outcome-screen-v1:<id>`), without using correctness labels, question type or gold answers. These500questions have been exposed during development; this is an exploratory sample, not a clean external holdout. The full source-pool verifier checks500cases and16,210original turns before building the package. The exact report path, bytes and source hashes are bound.

| Arm | Inference | Context and settings |
| --- | --- | --- |
| Baseline | One Luna answer | Original frozen packet, original prompt,4200output cap |
| Revision control | Luna draft, then Luna revision | Original packet; first call uses low reasoning and2048output cap, final4200; draft explicitly treated as untrusted model output |
| Guarded focus candidate | Luna v3 plan, then Luna answer | Same first-call low/2048allowance; final original prompt/4200with guarded original-source focus |

The two-stage arms have matching model identities, call counts and output caps. They do not have identical input lengths, actual reasoning-token use or exact measured cost; report those differences instead of claiming strict compute equivalence. Each arm is judged by the fixed provisional Terra contract. Gold/reference text is sent only to judges. Source corpus, final answerer settings, evaluator contract and original prompt remain fixed. Revision deliberately adds a draft-review instruction as its control intervention. Arm order is counterbalanced deterministically per case; no answer crosses between candidate and control.

There are8calls per case:5Luna calls and3Terra judgments,256maximum attempts. Identical answers are still separately graded; disclose judge variability and manually review every changed grade and applied focus case before any promotion. All attempted cases, empty focus, guarded fallback, nonfit and failures remain visible. Exceptions/truncated outputs stop the run without retry; reservations remain committed. Missing provider usage is explicitly unreconciled, not billed as zero.

## Decision rule

The exploratory numerical gate requires32complete pairs, at least8applied-focus cases, at least3net additional correct answers versus each control, and no more than1loss versus either control. The same case remains in all arms. It is a screening threshold, not a statistical significance claim;32cases cannot establish90%full-benchmark accuracy. A pass still requires source/grade review. A failure means no full500promotion of this candidate; preserve the result and identify whether retrieval, answer reasoning, focus or evaluator behavior caused the losses. Do not buy repeated prompt variants simply to get a green result.

This is a new outcome-based experimental protocol, not a retroactive waiver of the failed semantic gate. It tests whether guarded focus improves actual answers despite acknowledged selection limitations. An unguarded v3 selector is not promoted. Full500measurement and an unchanged repeat require their own justified proposal and authorization after meaningful candidate evidence.

## Verification and exact proposal

128focused offline tests pass, including9new guard/runner checks; Ruff passes. GPT-6 Sol review identified unknown-billing and exact-provenance binding gaps, both corrected with regression tests. Real source reconstruction verified16,210turns. Mock outputs exercise dispatch, budget, resume and scoring behavior only; they are not accuracy evidence.

Package canonical SHA256: `b2b77d3a02756c9e41869963bcdd5e3b761c3af8dc23cce48637a436a708b1f9`.
Workspace memory: `plans/2026-10-05-guarded-answer-screen/preparation-001/`.
Bound output: `plans/2026-10-05-guarded-answer-screen/paid-screen-001/`.
Conservative frozen-rate reservation: **$7.46564015**. Proposed cap: **$7.50**,256calls, no retries. No spend authorized by preparation. `execute_approved.py` requires an exact approval file and validates frozen code/package identity before constructing the provider; credentials are not printed. Prior diagnostic approvals are consumed and do not cover this run.

The current English headline remains84.6%under the historical evaluator;85.0%is the same saved answers under provisional Terra. No new lift or date for90%is supported yet.
