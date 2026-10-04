# Terra calibration and the next controlled Luna experiment

Historical English accuracy remains **423/500=84.6%, one run**, using Luna answers and GPT-4o grades. No new model calls or accuracy result accompany this implementation. GPT-4o remains retired for future calls; its records and historical source snapshots remain intact.

## What is implemented

`benchmarks/evaluator_v1` is a separate reference grader with a no-retry OpenAI transport, durable attempt reservations, explicit Terra-low settings, strict yes/no parsing, input/source hashes and receipt replay. It rejects truncated output, unrecognized models, malformed usage, changed inputs, reused receipt IDs and unresolved prior attempts. A paid approval binds one package, maximum attempts, budget and output directory. Validation requires a real passing development receipt; the saved-answer bridge requires a real passing validation receipt. The CLI defaults to an explicit offline `preflight` action; importing/building does not call a model or database.

The new grader retains the existing per-type LongMemEval rubric and adds a system instruction treating evaluated text as untrusted data. This is a **new internal scoring series**, not official GPT-4o scoring parity. Provider model aliases remain mutable; returned aliases/snapshots are recorded. Code changes require rebuilding packages, not weakening old hashes.

| Asset | Scope | Interpretation |
| --- | --- | --- |
| Development |32 judgments:26 from13 exposed synthetic scenarios,6 from3 exposed historical reference disputes | Debugging and first gate; require32/32 before validation |
| Internal validation |120 judgments from60 new paired source scenarios;60 positive/60 negative | Prospectively frozen before outputs, assistant authored/reviewed, published; not an independent hidden/human holdout |
| Saved-answer bridge |500 original Luna answers, checked against saved requests/responses and the old package | Changes only the judge; no regeneration |
| Focus inputs |500 frozen packets;16,210 whole original turns checked against authorized scopes | Input/provenance preparation, no selector quality measurement |

Validation covers all six question types plus five abstention scenarios, category/unit/event-identity errors, updates, temporal tolerance, partial preference satisfaction and instruction attacks. The positive/negative pair shares a scenario, so120 is not120 independent scenarios. These are mostly short synthetic cases; passing them does not establish transfer to all real ambiguous benchmark answers. The bridge needs a separate disagreement/source audit before using its judge scale for architecture claims.

The validation gate is at least114/120 correct, no more than3 false accepts, no more than3 false rejects, and zero errors on the two critical instruction-attack negatives. Report every rubric stratum and families with errors. These are project gates, not industry certification. Even zero false accepts among60 examples leaves a roughly4.9% one-sided95% binomial upper bound under independence assumptions; the authored sample is not a random sample of production traffic. If tuned after validation outputs, retire that validation set.

## One experimental evidence-selection lever

`benchmarks/evidence_focus.py` asks Terra to select up to8 original turns by opaque IDs from the source pool already delivered in the packet. It then appends only those original turns, including speaker and observation time, within the existing40,000-character total cap and4,000-character focus cap. Original context remains a byte-for-byte prefix. No selector-generated summary, arithmetic, answer or completeness assertion becomes answerer context. Schema/unknown-ID errors fail; an oversized selection is explicitly recorded as unchanged rather than clipped.

This is **opt-in experimental code**, not activated in production or promoted on mock tests. It differs from earlier rigid structured answers: Luna still reasons over the original prose with the original answer prompt. Corpus, extraction and Luna answerer remain fixed. It also differs from the earlier lexical supplement: selection is semantic and repeats existing original evidence; it retrieves no new facts. The caller must provide the authorized source pool; `prepare_focus` builds that pool from the frozen cache's question scope. IDs and evaluation labels never choose the selection. The provider sees question/date/source turns only.

The independent source-pool check verified500 packets and16,210 original turns, median31 turns per packet. Median spare capacity is3,634 characters; the smallest spare capacity is1 character. Budget non-fit therefore matters and must be reported as unchanged treatment. A correct quote can still be the wrong evidence: tests explicitly demonstrate that a book-box selection passes provenance but fails magazine membership. The13 published semantic fixtures are the next selector-quality gate, requiring all required evidence and no excluded evidence. Oracle-selected IDs in unit tests verify plumbing only.

**This lever cannot fix missing retrieval evidence or establish a90% result on its own.** Do not expand its scope after seeing outcomes. If it fails semantic/accuracy gates, keep the negative result and return to the appropriate missing-source or reasoning failure class.

## Execution order and cost

1. Approve and run32 development judgments with Terra-low. Stop on any incorrect judgment or execution error; no automatic retry.
2. Only if32/32 pass unchanged, run the120 internal validation judgments. Stop/preserve failures. Do not tune on this set and reuse it as fresh.
3. If validation passes, obtain a separately approved saved500-answer bridge and audit disagreements. Report old-judge/old-answers and new-judge/old-answers side by side; their difference is a grading shift.
4. Separately freeze/run the13-fixture Terra selector check. If it passes, prepare one matched Luna baseline/candidate screen with identical answer settings, prompts, corpus and judge; only the focus block differs. Freeze numeric gain, stable-control, abstention, latency and total-cost gates before paid outputs. A larger final run is not currently approved or ready.
5. If the controlled screen passes, do full500 and an unchanged repeat. A90% result on the new judge must be labeled with that judge; it cannot silently replace the old84.6% series. All500 questions have influenced development, so generalization still needs a separate benchmark.

The prepared first proposal is **at most152 Terra calls, no retries, $4.45 total reservation cap**: $0.94 development plus$3.51 validation. The unrounded worst-case reservation is$4.43249950. It includes byte-based input bounds, cache-write premium and the full2,048 completion-token cap including reasoning. It is a conservative dispatch bound, not an expected invoice or account-level spending control. The separately prepared500-judge bridge bounds$14.745885 and is **excluded** from this first approval. No selector or Luna generation calls are included.

Model settings/rates were checked October4 against [official Terra documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra): low reasoning supported, Chat Completions supported, standard short-context input/output rates$2/$12 per million tokens and cache writes1.25× input. The runner uses default service tier; reverify prices when freezing a later package. No account balance/access check has been performed.

## Reproduce without spending

From the repository root, build a new output directory (existing paths are rejected):

```sh
python -m benchmarks.evaluator_v1.build /absolute/new-output \
  --historical-package /absolute/old-full500-package.json \
  --historical-checkpoint /absolute/old-full500-checkpoint.json
python -m benchmarks.evaluator_v1 preflight /absolute/new-output/validation-package.json
```

The `run` action additionally requires `--directory` and `--approval`; an unapproved template cannot execute. Prerequisite receipt hashes are populated only after the preceding approved stage actually passes. Resume verifies the binding and all completed jobs; pending/error attempts require a reviewed continuation instead of automatic replay. Offline tests use injected fake providers with mode `offline-test`; their reports explicitly forbid model-quality claims.

The new evaluator does not unify every legacy product entrypoint. Deployment robustness, tenant isolation beyond the prepared scope adapter, fresh evaluation transfer and missing-source recovery remain separate work. This task adds a measurable, bounded experiment and preserves failures; it does not certify an industry-leading architecture.
