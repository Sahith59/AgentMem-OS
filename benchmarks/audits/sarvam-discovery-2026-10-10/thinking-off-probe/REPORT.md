# Thinking-off diagnostic: completed output, weak compaction

October 10, 2026 (America/New_York). Exactly **one authorized Sarvam call**, zero retries or follow-up calls. Same `sarvam-105b`, source, prompt, temperature, JSON format and 4,096-token output cap as the failed request. The only request-field change was `reasoning_effort: null` instead of `"low"`.

| Observation | Original low-reasoning call | Approved null-reasoning call |
| --- | --- | --- |
| Completed final JSON | No | Yes |
| Finish reason | length | stop |
| Prompt / completion tokens | 362 / 4,096 | 366 / 301 |
| Reported reasoning tokens | 4,095 | 0 |
| Request latency | 17.087 seconds | 2.365 seconds |
| Current-rate usage estimate | ₹0.25119 | ₹0.02355 |
| Frozen-rate usage estimate | ₹0.31042656 | ₹0.03274968 |
| Retained reservation | ₹4.13761536 | ₹4.13761536 |
| Accepted extractions / scored answers | 0 / 0 | 1 / 0 |

These are two observations, not repeated latency/cost estimates or an accuracy comparison. Identical messages can receive different provider token counts under different reasoning settings; the recorded request diff is exactly one field. No inference about the provider's internal formatting follows.

## What passed

HTTP 200 returned the requested model, completed final JSON and valid usage. The final-content and fact parsers pass. The exact instruction **“For my parcel, use pickup counter K7.”** remains in a fact with source `s01`. Every returned value matches an input field. No confirmed-backend claim, fabricated validity date or condition was added. This resolves the missing-final-output symptom for this request only. Optional reasoning control is documented by [Sarvam](https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/how-to/adjust-the-models-thinking-level).

An offline check writes the actual returned facts and originals to a new isolated SQLite file, closes it and reopens it. Both candidate answer arms retain the same authorized original source. This checks persistence and evidence delivery; neither answer request was dispatched.

## What is still weak

The extractor produced **seven facts: six copied metadata fields and one copied source sentence**. It did not extract a concise pickup-counter attribute. Schema validity therefore overstates useful memory structure if reported alone. All seven items have `current` status; that label on record metadata is not evidence of business-state reasoning. The original wording still preserves the user's instruction, so this is not a demonstrated false confirmation or incorrect answer.

In offline packet construction, user-message length rises from 364 characters for full history to 1,349 for SQLite augmentation. These are characters, not model tokens. The original remains accessible in both arms. Extra verbosity is a measured property of this one extraction; whether it helps or harms answering is **unknown**. Do not fabricate or hand-rewrite a better extraction, and do not classify metadata copying as a general Sarvam defect from one case.

## Accounting and verification

The authorized conservative reservation was ₹4.13761536 (displayed ₹4.14); no retry or broader pilot followed. For this call, current uncached rates of ₹15 input / ₹60 output per million imply ₹0.02355. The original higher frozen rates imply ₹0.03274968. Neither is a verified invoice or account balance. See [published pricing](https://docs.sarvam.ai/api/getting-started/pricing).

Across both separate live attempts, known usage estimates total ₹0.27474 at current rates / ₹0.34317624 at frozen rates, with ₹8.27523072 retained reservations. The first pilot's 227 unattempted requests remain unattempted. Its failure receipt is unchanged; this is a separate diagnostic, not a repaired historical score.

Independent re-reading/recalculation verified approval and request bindings, unchanged parent checkpoint and source hashes, one-field diff, token/cost arithmetic, exact seven source matches and final-parser replay. Re-entering the executor refuses before credential loading or provider construction, with the checkpoint unchanged. All verification after the one live call was offline. Semantic review was performed by the same assistant against the original source, not by an independent human or paid model judge.

## Direction and next task

This is a useful configuration result, **not an architecture gain, memory benchmark score or novel product finding**. English scores are unchanged. We are testing ordinary existing capabilities before choosing a product, as required.

Next, prepare a smaller, separately versioned EN/HI/TE full-history-versus-SQLite comparison with explicit null reasoning and the same answer model and prompts. Include ordinary recall plus date, condition and source-authority contrasts. Freeze its scope and request derivation, test malformed/failing outputs and cost accounting, then obtain approval for that exact package. Keep the observed metadata-copying extraction unchanged for the diagnostic comparison so that a prompt rewrite is not mixed into the settings change. Inspect useful facts and final answers separately. Any later compact-extraction change is a separate intervention.

Do not resume the failed parent, auto-run the full 228 calls or claim all-language coverage. Short synthetic, assistant-reviewed EN/HI/TE cases remain development data. Native human validation, broader languages, matched-total-compute controls, fresh family-held-out cases and actual tool/platform integration remain pending. If simple controls work, that is evidence to retain the simple approach, not to invent a product gap.

## Artifacts

`request.json` contains the exact credential-free request. `result-summary.json` includes actual final content, usage and receipt/checkpoint hashes. `verification.json` and `offline-store-verification.json` contain offline checks. Raw provider receipt/IDs and the bound approval remain local. No credentials are published. The failed parent report and raw evidence are preserved.
