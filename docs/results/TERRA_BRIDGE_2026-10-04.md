# Terra saved-answer bridge — October 4, 2026

**500/500 judgments completed: 425/500 (85.0%) with Terra, versus 423/500 (84.6%) with the historical GPT-4o judge. The answers are identical. This is a judge-scale comparison, not improved answer accuracy.**

The founder approved exactly 500 saved-answer judgments at a $14.75 cap, no retries or answer generation. All 500 completed, with zero execution failures and zero retries. Nine historical negatives became positive; seven positives became negative. Judges agree on 484/500 (96.8%). GPT-4o was not called; it remains retired. Luna answers and the historical corpus, including its extraction ancestry, are preserved.

Terra settings: `gpt-5.6-terra`, low reasoning, default service tier, 2,048 completion-token cap, frozen per-type rubric and untrusted-data system instruction. Execution used isolated OpenAI SDK 2.54.0. Returned model aliases and provider receipts are recorded; an alias is not an immutable model snapshot. Code/execution baseline: `52ed8912656c3438568c02fe26a6811dbe0646cc`.

## What the review found

All 16 changed grades and the 20 prospectively selected agreements received assistant review against the frozen rubric and targeted scoped source excerpts. This is not independent human adjudication or an exhaustive review of every source conversation. Primary provider grades remain unchanged; no manually corrected headline is reported.

- Several preference gains and the sound-effects answer are credible grading corrections.
- Terra rejected three months against a two-month temporal reference, although the rubric explicitly tolerates an off-by-one month. This repeats a weakness seen in calibration.
- Two preference rejections look too strict: relaxing evening activities including the user's desired body scan, and medical-imaging publications matching the user's interests. These are review judgments, not independent gold labels.
- Commute advice, implicit abstention and an added guitar-shop name retain policy/source ambiguities.
- A correctly accepted abstention includes an incomplete explanation about documented education. Reference-grade acceptance does not certify every factual clause.
- In the agreement sample, the charity answer includes an extra source-described benefit concert, while the reference total omits it. That source also has a March/April chronology inconsistency. Preserve this as a source/reference ambiguity; do not call it a simple arithmetic defect.

**Decision: retain Terra v1 as a provisional internal comparison scale for controlled research, with source review of changed outcomes. It is not sufficient as the sole authority for final English closure.** Do not alter this frozen judge or relabel outputs to raise the score. A future evaluator revision would require a new version, fresh validation and a separate bridge; the current validation is now exposed.

## Breakdown

Abstention cases are separated from their nominal types, so these rows sum to 500.

| Stratum | Historical correct | Terra correct | Total |
| --- | ---: | ---: | ---: |
| single-session-user | 63 | 63 | 64 |
| multi-session | 85 | 84 | 121 |
| knowledge-update | 69 | 68 | 72 |
| abstention | 19 | 19 | 30 |
| single-session-preference | 20 | 24 | 30 |
| temporal-reasoning | 113 | 112 | 127 |
| single-session-assistant | 54 | 55 | 56 |

The largest remaining graded gaps are multi-session (37 misses), temporal (15), and abstention (11). These are priorities for diagnosis, not established counts of architecture defects: this audit demonstrates that some grades and references are ambiguous.

## Integrity and cost

Two accounting paths agree on all 500 results. The independent audit reconstructs requests, binds source/input/approval hashes, compares saved answers, verifies unique receipts and checks token/reservation arithmetic. All 143 selected excerpt references across 36 reviewed cases match authorized scopes, roles, original-turn hashes and byte-for-byte text slices. All 297 historical artifacts (146,593,430 bytes) remain hash-identical.

- Worst-case reservation: **$14.745885**, within the approved $14.75 cap.
- Successful-response usage upper estimate: **$0.2878635**, not an invoice or account balance.
- Tokens: 100,395 input; 3,073 output, including 983 reasoning tokens.
- New answers, retries and architecture changes in this run: **zero**.

The spending approval is consumed. No selector, answer-generation, repeat or full-run spending is included.

## Next bounded step

1. Prepare the existing 13 exposed semantic fixtures as a separately priced selector-quality package. Require all required evidence, no excluded evidence, valid provenance and explicit handling of focus blocks that do not fit. This evaluates selection only, not answer accuracy.
2. Only after that gate passes, freeze one matched Luna experiment: same answerer/settings/prompt/corpus/judge, with only the original-source focus block changed. Count all attempted cases, including unchanged non-fit treatments. Freeze gain, control, abstention, latency and cost gates before outputs. Review all changed verdicts, especially temporal and preference cases, with the same source/rubric policy.
3. Promote only if measured answer improvement survives those checks. Then prepare a full 500 and unchanged repeat. All 500 questions have influenced development; a separate benchmark is still needed for generalization claims.

The evidence-focus lever cannot recover facts missing from the delivered packet. If selected failures require absent source material, classify them as retrieval work instead of claiming this lever fixes them. A 90% point score would be at least 450/500 on a named judge scale; 425/500 is 25 short. This arithmetic is not a promise of 25 easy fixes or a closure date. Sarvam remains parked under the founder's direction.

## Changed-grade review

| Case | Old → Terra | Assessment | Reason |
| --- | --- | --- | --- |
| `09d032c9` | 0 → 1 | supported_correction | Portable power bank and battery-saving advice satisfy the core personalization. Source confirms ownership; the rubric does not require every preference point. |
| `1c0ddc50` | 0 → 1 | ambiguous | Known podcast and audiobooks personalize the response, but light reading conflicts with the reference preference and the podcast is not new. Partial personalization versus conflicting advice is a policy ambiguity. |
| `54026fce` | 0 → 1 | supported_correction | Weekly virtual coffee and coworking address the remote-work social need; user explicitly wanted virtual coffee breaks. |
| `57f827a0` | 0 → 1 | supported_correction | The mid-century modern dresser anchors the layout advice to the stated replacement plan. Extra generic tips do not remove that match. |
| `6e984302` | 0 → 1 | supported_correction | Enumerated modeling tools, wire cutter and sculpting mat are the source-described purchased sculpting set. |
| `8752c811` | 0 → 1 | supported_correction | Sound effects is the exact title of item 27 in the original assistant list; parenthetical examples are not required. |
| `a89d7624` | 0 → 1 | supported_correction | Live music, Red Rocks and dinner build on the recorded Denver experience. The rubric does not require repeating Brandon Flowers. Current venue availability was not researched and is not a claim of this audit. |
| `afdc33df` | 0 → 1 | supported_correction | Utensil holder and granite protection directly address both personalized concerns. |
| `gpt4_372c3eed_abs` | 0 → 1 | rubric_pass_source_caveat | The answer explicitly says the total is not determinable, satisfying abstention. Its explanation that only high school is documented is incomplete: source also gives a four-year UCLA degree and PCC history. Reference-grade acceptance is not complete factual fidelity. |
| `15745da0_abs` | 1 → 0 | ambiguous | Conditional correction to vintage cameras signals an object mismatch but never explicitly says the duration for films is unknown. Source confirms cameras for three months. Abstention explicitness is ambiguous. |
| `157a136e` | 1 → 0 | supported_correction | A 36–45 range does not identify the required 43-year difference. Source gives grandma 75; user asks whether 32 is young or old, an indirect age cue. Exact reference age inference is less explicit than the reference suggests. |
| `195a1a1b` | 1 → 0 | likely_false_rejection | Relaxing activities satisfy the preference rubric, and user specifically wanted the Insight Timer body scan that evening. Phone-use concern is in the reference, but reviewed user turns favor guided meditation; not every preference point is required. |
| `22d2cb42` | 1 → 0 | ambiguous | Answer includes the required Main St location, making rejection questionable under the contains-answer rubric. However Rhythm Central is named in an earlier plan, not explicitly identified in the later completed-service report. Source does not establish the extra name with certainty. |
| `75832dbd` | 1 → 0 | likely_false_rejection | Medical Image Analysis and explicit medical-imaging interest match the core AI-healthcare preference. The rubric permits partial personalization. Publication recency and every named venue were not independently validated. |
| `c9f37c46` | 1 → 0 | clear_rubric_false_rejection | Three versus two months is an off-by-one duration that the frozen temporal rubric explicitly accepts. Source implies two months at attendance (watching started three months ago; event last month). Answer has an event-time reasoning error that the benchmark tolerance accepts. |
| `gpt4_f49edff3` | 1 → 0 | supported_rejection_with_format_caveat | Dates alone do not explicitly identify which event is first, second and third, so the reference-only grader cannot verify the requested event order. Source supports Feb 5 nursery, Feb 10 shower and Feb 20 case. This is under-specified answer formatting, not proof of incorrect underlying dates. |

The [machine-readable aggregate and all 36 review decisions](terra-bridge-2026-10-04.json) include artifact hashes. Raw checkpoints, approvals, original-source excerpts and verifiers remain in the separate local memory directory `plans/2026-10-04-terra-evaluator/paid-bridge-001/`; they are not claimed to be Git-backed or publicly replayable. The unchanged historical 84.6% series is preserved.
