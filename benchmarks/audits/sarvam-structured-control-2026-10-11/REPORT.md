# Structured-output control: failed action compatibility

**The candidate failed. The planned 288-episode comparison was not run.** Strict JSON improved formatting on a small check but prevented useful action execution in that same configuration. We preserved the negative results and tested the ordinary control to understand the difference.

## Previous study, briefly
The larger native study completed 2,800 main and 264 diagnostic task episodes plus setup, using 4,928 requests. Conversations with ordinary SQLite kept the correct stored state in 140/140 main action renderings. Low reasoning helped base105B on a selected full-context sample, from 14/33 to 20/33, but did not improve every setup. Wrong amounts, stale dates, unsupported action claims and format failures remained.

That study contained 637 distinct visible inputs in 700 rows, including six cross-split repeats. We published those limitations and separate sensitivity counts. Its strict synthetic scores are not general Sarvam accuracy, an English500 score or evidence of a novel memory product.

## This experiment
Within each model, the paired requests changed only `response_format`. The evidence, question, tool schema, backend, temperature0, 8,192-token cap and reasoning setting stayed fixed. Base105B uses `low`; Conversations explicitly uses `null`. These profiles must be reported separately, not used for a causal comparison between models.

The original plan was 24 authored situations, each rendered in English, Hindi and Telugu: 72 inputs and 288 model/format episodes. The sources and labels were reviewed by the same assistant that authored them. They have no exact visible-input overlap with the earlier bank, but share known failure families. They are neither independently authored workflows nor independently native-reviewed language tests.

Before scaling, we ran three excluded compatibility situations with both models: an unknown English address, a copied Hindi colour and an authorized Telugu appointment update. The strict-schema gate required all six to pass. All six had valid JSON keys/types; only two passed the complete contract. Neither action episode called the update tool. Base105B even reported the requested new date as confirmed while the actual database still held the old date.

The gate failed, so the 288-episode comparison was closed with zero calls. A separate 12-episode matched diagnostic was then frozen before execution. It tested ordinary output versus strict schema on the same three excluded situations. This deliberate attribution repeat was reported separately from the original six observations. There were no automatic retries or replacements of failed results.

## Matched results

| Model | Ordinary output: valid / strict success | Strict schema: valid / strict success | Update actually executed |
| --- | --- | --- | --- |
| Base105B, low | 1/3; 0/3 | 3/3; 1/3 | Control 1/1; schema 0/1 |
| Conversations, null | 3/3; 1/3 | 3/3; 1/3 | Control 1/1; schema 0/1 |

Both controls invoked `update_appointment`. Their actual SQLite databases changed from 2026-10-22 to 2026-10-25, with version2→3, one TEST-KEY operation and an actual confirmed receipt. Both schema arms emitted final text without invoking the tool. Their databases stayed at the old date and version, with no operation.

The initial paired payloads differ only in `response_format`. The earlier schema-only probe showed the same missing-action pattern. This is **one Telugu action situation tested on two models and repeated**, not four independent workflows or proof that the feature never works.

Strict schema repaired the base model's translated JSON keys on the Hindi colour case. But the tiny aggregate strict-score change, 1/6→2/6, conceals worse actual action behavior. We reject this configuration. Equal caps also do not mean equal consumed compute: controls made two extra calls because they performed actions. The verification file reports actual tokens, calls, cost and request latency by arm.

## Review of the misses

**Action execution and reporting are different.** The controls performed the correct database operation. Base105B then translated output keys; Conversations returned a Telugu sentence instead of only the ISO date. Those were output-contract failures. The schema arms skipped the operation entirely. Base's first schema response falsely claimed confirmation, whereas its matched repeat said `not_performed`. Temperature0 did not yield identical repeated outputs.

**No address was invented.** The unknown-address label expects JSON `null` and `needs_clarification=true`. Base replies truthfully described the absence of an address but violated that representation. Conversations initially returned the string `"null"`; in the matched pair it returned real `null` but set clarification to false. Whether a read-only question about what is established requires a follow-up is a rubric choice. Preserve the strict grades, but do not call these factual hallucinations. A future protocol should explicitly define unknown and clarification behavior for both arms.

**Hindi formatting is not a broad language-quality result.** The schema responses copied `केसरिया`. Base control translated the keys and used `केसरीया`; it failed the exact-copy/format contract. Without independent native annotation, an orthographic variant cannot establish a broad Hindi-understanding defect. Conversations passed this case in both formats.

These observations concern our exact prompt, API route, settings and schema/tool combination. They do not establish a Sarvam production Voice Agents defect or explain the server implementation. We have not tested forced tool selection or other schemas. The interface change is a concrete lead, not a universal unsupported-feature claim.

## Verification and cost

- 20 new API requests across 18 excluded diagnostic episodes; zero API errors and automatic retries.
- All 20 provider response IDs are distinct. The prepared 288-episode comparison remains unattempted.
- New known token estimate: **INR0.406440**. New retained accounting: **INR0.508053**.
- Including the completed native study: known **INR58.686555**, retained **INR87.183984**, inside the founder's INR6,000 research ceiling. These are estimates/reservations, not a verified invoice.
- 25 focused tests passed: seven new checks plus 18 existing native-study checks. Ruff and diff checks passed.
- Network-blocked replay reproduced every request chain, tool call, final SQLite state, strict grade and ledger amount. A separate check confirmed that the failed gate blocks execution before reading credentials.

Same-code replay verifies integrity, not independent semantic truth. The first offline audit command lacked PYTHONPATH; it made no API calls or output report. The corrected invocation passed. Raw receipts and historical results are unchanged.

## Next phase

Separate action execution from final formatting. Keep tools available while handling the action, then use a bounded final-answer step with no executable tools and the actual tool history. Give both format arms the same evidence and final-call budget. First establish correct execution and reporting together. Then test authoritative current-state queries and server-enforced approval as separate controls.

Do not supply gold labels as backend facts, replace wrong model claims with correct values and call that model accuracy, or claim ordinary transaction controls are a novel memory system. The prepared 72 renderings are unattempted. Qualified native review and more independently authored workflows remain necessary for stronger public claims. No extra credential is needed to prepare the next step.

Sources checked October11: [Sarvam chat guide](https://docs.sarvam.ai/api/api-guides-tutorials/chat-completion/overview), [V1 reference](https://docs.sarvam.ai/api-reference/chat/chat-completions-v1), [pricing](https://docs.sarvam.ai/api/getting-started/pricing). Sarvam documents structured output and application validation; exact compatibility was tested here. Protocols, frozen code, labels, receipts, SQLite files and verification accompany this report.
