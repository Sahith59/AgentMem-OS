# Fixed-model architecture: first selector failure and repair

October 4, 2026. The founder approved the exact 13-call Luna diagnostic at a $0.04 cap and reiterated that improvement should come from architecture. **The frozen gate failed on its first attempt.** One provider response returned, zero plans were accepted, and 12 cases were not attempted. There was no retry. This is not a completed 13-case accuracy result. No English answers or grades changed: historical 423/500 and provisional Terra 425/500 still describe the same saved answers.

## What the paid response established

The first case asks for the number of magazine subscriptions. The source has two magazines and a separate book-box subscription. Luna selected both source turns and returned `sufficiency: sufficient`.

1. **Confirmed contract defect:** the v1 parser accepts only `complete` or `uncertain`, but the prompt explicitly named only `uncertain`. JSON mode did not constrain this enum. Rejecting the output was correct; failing to specify the full contract was our engineering mistake.
2. **Observed semantic gate failure:** independently decoding the opaque source IDs confirms that the excluded book-box turn also received selection. Simply translating `sufficient` into `complete` would not pass the frozen selection gate. One example does not establish the model's general error rate.

The request, provider response, failed checkpoint and original fixture labels remain unchanged. Reservation was $0.0030436; token-receipt cost using the conservative cache-write input rate is $0.0002934. This is an estimate, not invoice reconciliation. Source package canonical SHA-256: `017319737bef7572fe81b8d6c939aa5ce31035144ec20a40ed64fd3b93f90493`. Detailed artifacts and independent audit are in the external memory directory `plans/2026-10-04-luna-selector/paid-diagnostic-001/`.

## Implemented v2, still unmeasured

The new version uses the same GPT-5.6 Luna settings. It adds a strict output schema with all required fields and explicit enum values, plus separate selected and rejected source-ID lists. Local checks reject unknown, duplicated, overlapping and out-of-order IDs. Only selected original text receives extra focus; generated requirements and rejection lists are audit data. The original packet is preserved, so rejected turns still remain visible there. This experiment changes emphasis, not the available corpus or removal policy.

The model can still make a schema-valid semantic mistake. `complete` remains an unverified model claim. Documented support for [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) establishes the request format, not correct evidence selection or a successful live v2 request. The cost bound includes the output schema as input. Failed/truncated responses retain valid usage receipts separately from output acceptance; an independent review caught this accounting issue before v2 was frozen.

Validation: 84 focused checks pass across v1/v2 contracts, both diagnostic runners, source rendering and capacity bounds. New v2 files pass Ruff. The old v1 package still reconstructs byte-for-byte. No v2 provider call has occurred. The opt-in experimental path is implemented; no production default or English answerer is promoted.

## What public scores tell us—and do not tell us

Research used ECC deep-research, a GPT-6 Sol literature reviewer, a separate Sol failure/code reviewer, and root source verification. These are development tools, not benchmark inference models. This was a targeted primary-source review, not a competitor reproduction.

| Primary source | Protocol fact | Implication for this project |
| --- | --- | --- |
| [LongMemEval dataset definition](https://github.com/xiaowu0162/LongMemEval/blob/main/README.md) | `_s` contains the full history; the oracle file contains only evidence sessions | Identify the ingestion dataset separately from the reference file used for judging |
| [agentmemory run audit](https://github.com/JordanMcCann/agentmemory/blob/main/LEGITIMACY.md), [harness](https://github.com/JordanMcCann/agentmemory/blob/main/run_longmemeval_full.py) | The author-reported 481/500 run names the oracle dataset and Opus answerer; the current harness defaults to the oracle path, with an override available | Retrieval within an already filtered pool is not directly comparable with full-haystack retrieval. This is a protocol distinction, not a fraud finding |
| [Caura harness](https://github.com/caura-ai/caura-longmemeval) | Uses raw-turn search, whole-session expansion and a four-stage Gemini reader; reports 21 of 39 errors with answer turns delivered | Test compact layout and source-session expansion separately under Luna, preserving source coverage and measuring the larger budget |
| [SodaMem](https://arxiv.org/html/2608.08055) | Reports 92.8% best-of-three with the same Flash model reading and grading; stated costs exclude ingestion/judging | Provenance and temporal validity are useful mechanisms. Its headline is not our judge, model, repeat statistic or cost boundary |
| [WhenLoss](https://arxiv.org/html/2605.24579) | Separates fixed-reader oracle evidence, complete stored memory and retrieved memory; includes a two-call cost control | Diagnose write/retrieve/read gaps operationally, avoid unsupported causal attribution, and compare planning against an equally priced extra call |

## Next architecture decisions

Keep the extraction corpus, Luna answerer, provisional Terra evaluation contract and question population fixed. The immediate v2 diagnostic tests evidence decisions; it is not another answerer/model screen. The two versions isolate a contract/partition change, although that bundled change cannot alone determine which part improves model selection.

If the development gate passes, freeze a fresh source-reviewed semantic set before further tuning, then a paired fixed-Luna answer test. The first candidate is original-source focus. Include the existing baseline and an extra-Luna-call control with a matched inference budget; do not attribute an extra-compute gain entirely to architecture. Measure false inclusions, required-source coverage, answer gains/losses, all attempts, cost and latency. Source expansion and temporal membership/operation checks remain later independent changes. Do not combine a new graph, prompt, retrieval rule and calculator in one claimed causal comparison.

The 13 fixtures are exposed, assistant-authored development material. Their frozen requirement lists may be over-strict: the workshop's first/second-event turns can arguably suffice without the repeat-mention turn, and the explicit corrected amount can suffice without the earlier amount. Keep the recorded gate unchanged and report these limitations. A future evaluation should allow source-reviewed alternative sufficient sets and distinguish all labelled evidence recall from actual answer sufficiency.

Our 500 English questions have also been inspected repeatedly. A future 90% there would be an exposed development-benchmark result; disjoint conversation/data transfer is needed for a research generalization claim. No accuracy increase, 90% date, industry superiority or research novelty follows from this repair.
