# Capped-actor recovery: one real failure recovered

Completed October 13, 2026. **The candidate succeeded on the sole eligible observed failure; the original control also succeeded.** This is one English scenario with two reporting variants, not two independent cases or a new full-study accuracy score. Original v3 remains failed and unchanged.

## What happened, plainly

A customer gave two possible booking dates: April14 if a morning slot was available, May14 otherwise. Nobody had checked availability. The existing actor used its 8,192-token output allowance and returned no answer or tool call. Our old pipeline then skipped both final reporters.

The new opt-in handoff stops that actor and gives the verified original source to the existing reporters. It does not repeat an action, infer a date, insert a model-written fact, change models or repair a model answer. Both reporters correctly returned `answer: null`, `needs_clarification: true`, `action_status: not_requested`. They signaled that clarification was needed; this experiment did not generate or test a natural-language follow-up conversation.

| Check | Original frozen episode | New recovery control | New recovery candidate (primary) |
|---|---|---|---|
| Final report reached | No | Yes | Yes |
| Date + clarification correct | Missing, failed | Yes | Yes |
| Raw action status correct | Missing | Yes | Yes |
| Receipt-owned action status correct | Missing | Yes | Yes |
| New backend actions |0|0|0|

## Scope and cost

All 240 parent routing records were audited. Actor terminals: 221 answer, 18 incomplete, 1 invalid_action_response. Only `confirm-fresh-13-en-m0` had both reporters unreached and passed the predeclared exact cap, unchanged-state and no-action eligibility checks. No successful case was substituted and no output was selected after execution. The candidate was primary before spending; the control is reported separately.

The two new requests used the original Sarvam-105B low profile, temperature 0, 8,192-token output cap, exact original reporting prompts/schema and identical original evidence. Both returned HTTP 200 with known usage through the normal provider transport. No native or tool call was repeated, no key was printed, and no paid request was retried.

| New reporter | Input tokens | Output tokens | Provider-call elapsed | Token-cost estimate |
|---|---:|---:|---:|---:|
| Control |835|5263|24.79s|₹0.328305|
| Candidate |843|3765|17.63s|₹0.238545|

New known estimate **₹0.566850**, retained with 25% headroom **₹0.708564**. Maximum reservation was ₹6.144 under a ₹8 phase cap and the standing total ₹6,000 cap. Native research cumulative: **7,037 calls, ₹271.224630 known, ₹371.289339 retained**. Older separate pilot accounting remains separate. Rates verified October 13 from [Sarvam pricing](https://docs.sarvam.ai/api/getting-started/pricing):₹15/₹60 per million input/output tokens, cached discounts ignored. These are estimates, not invoices or an account-balance check.

Latency matters: candidate recovery alone took 17.63 seconds; the earlier saved native failure took 42.76 seconds. These occurred in separate runs and are not a measured end-to-end latency or a distribution. Correct recovery here does not establish a fast voice-assistant path. Both reporter variants add compute relative to the old early exit; this is not an equal-actual-compute comparison.

## What this proves and what remains

The integration can recover a useful, source-correct final report from this exact capped/no-tool failure using the same model and evidence. The API output is saved verbatim and independently parsed; the answer was not hardcoded or repaired. This is a narrow observed engineering gain. The opt-in research path is not a default production policy and does not recover uncertain writes.

The original 240-episode scores remain base105B 90→105/120 and Conversations 95→96/120. Its failed acceptance gate, missing reports and Conversations wrong definite dates 18→22 remain in the historical record. Do not substitute an imputed 106/120 or a 100% headline. The blocked 108-call evidence screen remains unexecuted.

The bigger remaining issue is source interpretation: Conversations missed all 12 separate-booking preference questions and 7 of 12 unresolved-condition questions. That is not proved to be a persistent-memory problem. The next proposed comparison tests evidence organization and actor-prose influence over all 240 saved histories, with fresh matched reporter calls. See NEXT_STEP.md; it is not executed by this diagnostic.

## Verification and reproducibility

Implementation 035d45c; 240 focused offline Sarvam tests and Ruff pass. A separate GPT-6 Sol assistant reviewed the method/code and actual source/results. Independent human/native review is absent. Preparation and paid execution never parse the separate gold file. A durable exclusive claim and started marker prevent restarting this diagnostic; infrastructure or unknown usage stops subsequent provider dispatch.

The primary audit and a separate standard-library raw parser/accounting check agree on both new requests, receipts, answers and costs. The diagnostic snapshot replays all 240 routing decisions and the selected old native receipt. The full 844-call parent audit independently reproduced before this run; its entire HTTP archive remains separately published. External hashes verify the closed parent/screen and protected founder files. Archive replay commands and final publication checks are in the public README and completion receipt.

This is development-exposed, assistant-reviewed, one-case English recovery. No new English LongMemEval score, all-language memory support, industry-leading benchmark, product novelty or production reliability claim follows.
