# Native Sarvam investigation — complete first study

Date: October 11, 2026. Native model APIs and ordinary SQLite only; no AgentMem extraction or new memory architecture.

We completed **2,800 main task episodes plus 264 diagnostic episodes**, across two model IDs and two setups. The excluded setup adds 10 checks. There were **4,928 actual API requests**, no automatic retries, and 5 HTTP 429 responses. Every remaining planned episode was attempted. Raw requests, responses, actual test databases and failures are preserved.

The useful result is a set of reproducible, configuration-sensitive failures. It is **not an architecture gain, general Sarvam accuracy estimate, production vulnerability, or proven new product opportunity**.

## Main results: strict task success

A success requires the exact requested answer contract, correct answer, correct backend state, no forbidden mutation attempt and required actual operation receipt. These are custom synthetic tasks; their percentages are not comparable with published model benchmarks, our English 500 score or the earlier 12-example memory pilot.

Both models use null reasoning, temperature 0 and a 4,096-token output cap. Invalid output and provider errors remain in the denominator.

| Model | Setup | Discovery | Reserved confirmation |
| --- | --- | ---: | ---: |
| 105B | Full original context | 312/490 (63.7%) | 127/210 (60.5%) |
| 105B | Ordinary SQLite tool | 298/490 (60.8%) | 126/210 (60.0%) |
| Conversations | Full original context | 317/490 (64.7%) | 127/210 (60.5%) |
| Conversations | Ordinary SQLite tool | 300/490 (61.2%) | 126/210 (60.0%) |

Full context gets the original records directly. SQLite returns the same originals through read_records. Both action arms can call the real local update tool. SQLite adds a read roundtrip and uses prompt-based final JSON when tools are available; full-context read-only tasks use JSON-object mode. Thus this is an end-to-end setup comparison, not an isolated storage effect or matched-total-compute architecture comparison.

## Data quality correction and sensitivity

The bank contains 700 language renderings of 200 parameter rows across ten families. Exact visible-input auditing found **637 distinct inputs**: 451 in discovery, 192 in the reserved split, with six shared across splits. The English slice has 182 distinct inputs in 200 rows. Repeated conditional patterns and overlapping action variants caused this; group-ID separation was insufficient. Earlier wording that all 210 reserved inputs were unseen is corrected here.

Below, choose the lexicographically first case per visible-input hash; exclude any confirmation input seen in discovery. This selection never uses a model outcome. It leaves 451 discovery and 186 confirmation inputs per setup. It is a separately labeled post-discovery sensitivity, not a replacement for the original scores.

| Model | Setup | Discovery | Reserved confirmation |
| --- | --- | ---: | ---: |
| 105B | Full original context | 292/451 (64.7%) | 117/186 (62.9%) |
| 105B | Ordinary SQLite tool | 273/451 (60.5%) | 114/186 (61.3%) |
| Conversations | Full original context | 297/451 (65.9%) | 117/186 (62.9%) |
| Conversations | Ordinary SQLite tool | 278/451 (61.6%) | 114/186 (61.3%) |

These still share linguistic templates. No independent natural-workflow holdout, 700 independent situations, or qualified native-language validation is claimed. The confirmation began alongside remaining discovery after initial source review; the timing decision was recorded before its outputs. No configuration was selected from confirmation results.

## What reasoning changed

The failure-enriched diagnostic uses 33 renderings from ten parameterized situations, with four output-independent control bases. Null and low both have an 8,192-token ceiling; actual compute differs.

| Model/setup | Null | Low | Gains / losses |
| --- | ---: | ---: | ---: |
| 105B, Full original context | 14/33 | 20/33 | 10 / 4 |
| 105B, Ordinary SQLite tool | 15/33 | 15/33 | 5 / 5 |
| Conversations, Full original context | 15/33 | 15/33 | 1 / 1 |
| Conversations, Ordinary SQLite tool | 15/33 | 15/33 | 0 / 0 |

- Arithmetic example: seven items at INR 153, one cancellation and a single INR 20 discount gives **898**. Base 105B null answered 1071; low answered 898. This supports a settings-sensitive recovery, not a memory gain.
- Base 105B low still selected an old version-4 delivery date over a newer accepted version-5 date, including the English case. Other arithmetic mistakes remained.
- Conversations/full-context claimed an appointment change was already applied without any tool call; the database retained the old date. The Hindi and mixed-source examples share one underlying situation.
- Five original discovery episodes attempted an update after a visible negative approval statement. The synthetic backend intentionally delegates approval checking to the model. This is a model-policy test, not a flaw in Sarvam's production backend; real authorization should be enforced by application policy.
- We inspected all 31 failed base-105B-low diagnostic outputs: 20 primarily involve formatting/contract problems, and 11 contain substantive mismatches. The latter share only four underlying situations. This assistant review does not rescore the frozen results or replace independent native review.
- Base-105B-low backend state was correct 66/66 in this selected diagnostic. That does not establish broad reliability or erase earlier forbidden updates.

Base-105B-low reported 89,525 reasoning tokens in 109 calls. Conversations returned no reasoning-token counts in either profile; accepting the low parameter does not prove internal reasoning was enabled. Raw telemetry distinguishes unreported counts from reported zero.

## Actions, separately from words

Most read-only tasks cannot change the test database. Report action-only state results to avoid an inflated all-task backend percentage:

| Phase | Model/setup | Action episodes | Correct backend state | Forbidden update attempts |
| --- | --- | ---: | ---: | ---: |
| discovery | 105B, Full original context | 98 | 80/98 | 3 |
| discovery | 105B, Ordinary SQLite tool | 98 | 91/98 | 7 |
| discovery | Conversations, Full original context | 98 | 86/98 | 0 |
| discovery | Conversations, Ordinary SQLite tool | 98 | 98/98 | 0 |
| confirmation | 105B, Full original context | 42 | 37/42 | 1 |
| confirmation | 105B, Ordinary SQLite tool | 42 | 36/42 | 6 |
| confirmation | Conversations, Full original context | 42 | 37/42 | 0 |
| confirmation | Conversations, Ordinary SQLite tool | 42 | 42/42 | 0 |

## Spending and verification

Known usage estimate: **INR 58.280115** at the documented [INR 15/M input and INR 60/M output rates](https://docs.sarvam.ai/api/getting-started/pricing), with no cache discount. Retained accounting including 25% headroom and the full reservations for unknown-usage failures: **INR 86.675931**, within the approved INR 6,000 ceiling. These are estimates and reservations, not a verified invoice or account balance. Earlier campaigns are separate.

| Stage | Calls | Known usage estimate, INR | Unknown usage |
| --- | ---: | ---: | ---: |
| attribution | 426 | 9.610890 | 0 |
| confirmation | 1340 | 14.905440 | 0 |
| discovery | 3150 | 33.740595 | 5 |
| setup | 12 | 0.023190 | 0 |

Original discovery stopped on rate limits because our initial runner lacked aggregate pacing. The continuation attempted only previously unattempted episodes; all five original failed requests remain failed. Founder reported 120 requests/minute; the shared limiter was set to 90. No inference continues after completion of these frozen stages.

18 focused software tests passed. Network-blocked replay verifies 3,064 task episodes and 4,916 task calls against exact requests, raw receipts, tool traces and fresh SQLite replay. A separate whole-campaign audit includes setup, all 4,928 dispatches, 4,923 unique successful provider IDs, costs and retained unknown usage. These are mechanical checks, not independent semantic review or provider-signed attestations. API-call latency by model/profile is in receipt-audit.json; it excludes pacing and local tool time.

## What this means for the next decision

The earlier small memory comparisons tied 47/48 per arm across repeats. This new study uses different data, prompts and scoring; its percentages do not imply a decline from those small results. Neither study demonstrates a new memory advantage.

Next test ordinary stronger controls: supported structured output for format failures, authoritative current-state queries for version confusion, and application-enforced approval/action receipts for tool errors. Keep reasoning and each intervention separately attributable. A read-all-records SQL tool is not the strongest ordinary business integration. Do not inject evaluator answers into a purported backend control.

Then use genuinely new, varied workflows with visible-input/semantic grouping and qualified Hindi/Telugu review. Extend endpoint-supported languages in reviewed waves. No speech, Romanized-input, all-language, full-context-window, native Voice Agents platform, industry superiority or novelty claim follows from this text screen. Prior art already includes state-based task evaluation; see PRIOR_ART_NOTE.md and FOLLOWUP_CRITERIA.md.

Evidence and reproduction: analysis.json; receipt-audit.json; replay-verification.json; attribution/REPORT.md; source reviews; bank/; frozen manifests; calls/ and episode SQLite stores. Model IDs/settings are pinned, but an immutable provider model-weight revision is unavailable. Public code/evidence is tracked in [PR #51](https://github.com/Sahith59/AgentMem-OS/pull/51). English work and prior raw experiments are preserved.
