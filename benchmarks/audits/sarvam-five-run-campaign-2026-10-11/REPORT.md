# Sarvam five-run campaign — October 11, 2026

We completed the five authorized run slots. Two stopped on infrastructure/output failures; three completed comparisons. There is **no replicated answer-accuracy advantage** for generated memory. There is a tested, narrower reliability improvement: unsupported validity dates can be kept out of memory without losing original evidence.

| Run | Change | Calls | Original-record control | Memory + originals | Outcome |
|---|---|---:|---:|---:|---|
| 001 | Frozen v3 strict schema | 2 | Incomplete | Incomplete | Provider rejected uniqueItems; one receipt has unknown usage |
| 002 | Remove only unsupported schema keyword | 26 | Incomplete | Incomplete | Telugu extraction hit output limit; one earlier invalid answer |
| 003 | JSON-object mode, same fact schema in prompt | 72 | 23/24 | 24/24 | One-answer development signal |
| 004 | Exact unchanged replication of 003 | 72 | 24/24 | 23/24 | Signal reversed; combined result 47/48 for each arm |
| 005 | Cited-text witness required for validity dates | 72 | 24/24 | 24/24 | Four additional bad-date augmentations rejected; answers preserved |

These are 12 short English/Hindi/Telugu development renderings of four families, repeated twice per run. They are not 24 independent tasks, a public held-out benchmark, the English 500-question baseline, or all-language performance. No model change: sarvam-105b, null reasoning, temperature 0, 4,096 output cap throughout. Models produced the saved responses; no gold values were injected into extraction or answering. Labels were used by the separate deterministic scorer. Provider receipts are saved evidence, not independently attested logs.

## What became more reliable

V4 isolated the unsupported schema keyword while retaining local duplicate-citation rejection. V5 allowed three later comparisons to finish, after v4's truncated malformed extraction. That sequence supports further testing of the contract path; it does not prove schema mode caused the truncation.

V6 adds one local check: a generated validity date must appear literally in a cited authorized source's text. Observation metadata, the question, uncited sources and other generated facts cannot supply the witness. A failed check rejects the entire augmentation, retains originals and retains extraction cost. It does not edit a date to make it pass.

In run005, four sets failed this new check and three failed the existing structural check. All seven source-only fallback answers were correct. The 17 admitted sets had no missing literal-date witnesses, but their full meaning is not certified. Across saved runs003/004, the same guard would newly reject 8/5 sets; that is offline admission evidence, not a counterfactual answer score.

Run005 answer-plus-required-source coverage was 23/24 for memory versus 17/24 for control. This secondary signal is unreplicated. Covering a required source does not prove every citation or statement is supported, and the primary answer score tied.

## Important flaws still open

1. **Conditions can become false events.** In run004, a Hindi conditional callback became a requested repair approval. The memory answer incorrectly said the call was authorized; control correctly said no. A literal date check would not reject this fact because the callback date appears in the source. The regression test preserves this limitation.
2. **The status field mixes different concepts.** It combines current/historical, requested/conditional and confirmed/withdrawn, but omits the legitimate backend outcome rejected. Outputs describing rejection can therefore fail the schema; other outputs substitute withdrawn, which means something different.
3. **Dates have different roles.** Observation time, scheduled service date, effective state date and interval endpoint cannot safely share loosely generated validity fields. Even a correctly cited date can occupy the wrong role. Inclusive/exclusive endpoints are unspecified.
4. **Originals in the prompt are not a sufficient safeguard.** The model can still follow a misleading generated fact. Some outputs mark current nonapproval historical, infer a language attribute, or place a condition only in free text while leaving its structured field null.
5. **The ordinary control is strong.** On these short records it nearly saturates. A memory system must justify its extra cost on harder, fresh cases and strong compute-matched controls. These results do not establish product novelty.

## Cost and execution

244 calls, zero retries, five slots consumed. Known current-rate estimate: **₹4.10268**, plus one HTTP500 call whose usage was not returned. Do not interpret this as an exact bill or say the error was free. Frozen-rate known estimate: ₹6.68358456. Retained attempt reservations: ₹1,009.57814784, inside the approved ₹1,489.54152960 campaign ceiling. No sixth run is authorized.

For run005, control cost estimate was ₹0.29154 and memory cost ₹0.82998, including extraction. The experiment fixes model and answer settings, **not total compute**. Current published prices were checked at [Sarvam pricing](https://docs.sarvam.ai/api/getting-started/pricing): input ₹15 and output ₹60 per million; no cache discount assumed. Prices are estimates, not invoice verification.

Every response, request, schema, checkpoint, approval, score and reopened store is preserved. Saved evidence can be replayed offline using verify.py; it blocks network calls and recomputes scores, accounting and store contents. Scorer output keeps semantic checks UNREVIEWED; separate assistant source reviews record actual issues without claiming independent native validation. 164 focused tests pass. Earlier test-development failures are documented, not hidden.

## Next work, before another paid run

Implement and audit an explicit distinction between source events and derived state: requests, conditional rules, assertions and backend outcomes should be separate record types, each with original source evidence and date roles. An unmet condition cannot become an approval event; rejected and withdrawn must remain distinct. Unknown interpretation should retain original evidence and abstain from a state update.

First test that contract on the saved counterexamples and paired positive cases, including explicit later approval, rejection, withdrawal and out-of-order receipts. Compare it against plain SQLite event/state handling, a stronger prompt/schema-only control and full-history with matched compute. Freeze fresh family-level development/confirmation splits before scoring. Add independent Hindi/Telugu review and broaden endpoint-specific language coverage before public claims.

This is a proposed architecture experiment, not a completed fix or a novel product claim. Further paid work needs a new bounded authorization after the next concrete package is ready. No credentials or account balances are published; no message was sent to Sarvam.
