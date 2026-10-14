# Source interpretation comparison — completed October 13, 2026

**Some answers improved, but blanket removal of prior agent prose is not a reliable fix. The predeclared acceptance gate failed.** All 720 paid reporter calls completed once across 240 saved histories, with no infrastructure errors, unknown usage or retries. The primary replay and a separately implemented raw-response audit agree on every planned output, paired count and cost. English, Hindi and Telugu assistant reviews cover all 720 outputs; independent human/native review remains absent.

## What was tested

Each of 40 related synthetic scenarios has English, Hindi and Telugu versions and two saved Sarvam model histories. Each of those 240 histories received three new reporting calls:

- **A — chronological:** the ordinary original evidence sequence.
- **B — grouped:** the same evidence organized by source, without loss.
- **C — grouped, prose withheld:** B with the earlier agent's nonempty explanatory prose removed. Original customer messages, conditions, dates, booking references and actual tool receipts remain.

The model, reporter instruction, temperature, response schema and 8,192-token output ceiling were fixed within each comparison. Profiles were `sarvam-105b` with low reasoning and `sarvam-105b-conversations` with null reasoning. There were **no new actor calls, tool actions or backend changes**. This tests reporting on saved histories, not a fresh execution success rate. A is a contemporaneous control under the common instruction, not the older v3 control. Historical scores remain unchanged.

## Actual answer results

The primary score requires both the correct date and the correct clarification decision. Incomplete outputs are failures in the planned denominator.

| Model | A: chronological | B: grouped | C: prose withheld | C versus A, paired gains / losses |
| --- | ---: | ---: | ---: | ---: |
| Sarvam-105B | 104/120 (86.7%) | 99/120 (82.5%) | 105/120 (87.5%) | 11 / 10; net +1 |
| Sarvam-105B-Conversations | 96/120 (80.0%) | 100/120 (83.3%) | 103/120 (85.8%) | 14 / 7; net +7 |

Of the 11 base-model C-versus-A gains, five recover unusable A outputs and six fix usable but wrong answers. All 14 Conversations gains fix usable wrong answers. These are observed counts, not a demonstrated gain on unseen tasks. The descriptive 95% scenario-cluster intervals for C-minus-A are **−6.67 to +8.33 percentage points** for base and **−1.67 to +14.17 points** for Conversations; both include zero.

Only 184 histories actually contain prose that C removes. The predeclared primary contrast was C versus B on those treated histories:

| Model | B correct | C correct | Gains / losses | 95% descriptive cluster interval for the difference |
| --- | ---: | ---: | ---: | ---: |
| Base, 98 treated | 79/98 | 87/98 | 14 / 6; net +8 | −1.89 to +19.15 points |
| Conversations, 86 treated | 66/86 | 69/86 | 12 / 9; net +3 | −5.95 to +13.64 points |

Five of the 14 treated base gains recover an invalid B output; nine fix usable answers. All 12 treated Conversations gains fix usable answers. Both intervals include zero. All A→B, B→C and A→C contrasts, including no-treatment slices, are retained in [summary.json](summary.json) and [audit.json](audit.json).

## Why the candidate fails

| Reliability measure | Base A / B / C | Conversations A / B / C |
| --- | ---: | ---: |
| Wrong definite dates; lower is better | 3 / 4 / 1 | 22 / 18 / 8 |
| Unnecessary abstentions on known answers; lower is better | 0 / 1 / 2 | 2 / 2 / 9 |
| Incomplete finals; lower is better | 5 / 8 / 6 | 0 / 0 / 0 |
| Correct raw action status | 77 / 83 / 85 | 109 / 95 / 97 |
| Correct receipt-projected status | 115 / 112 / 114 | 120 / 120 / 120 |
| Entire raw report correct: answer, clarification and status | 70 / 71 / 78 | 86 / 78 / 80 |

C reduced wrong definite dates, but it sometimes refused to answer when the source clearly supplied the answer. Conversations' fully correct raw reports fell **86→80**, despite its date-plus-clarification count rising. The existing deterministic receipt projection repairs status on every usable final in all arms; this is the same pre-existing layer, not a new C benefit. It does not invent answers for capped outputs.

Four frozen checks fail: false abstentions increase on both models; base projected-status coverage falls versus A because C has one more incomplete final; and Conversations raw-status accuracy falls versus A. The candidate is **not promoted**, and neither the score definition nor the acceptance rules were relaxed after results. B alone also remains exploratory: its base score falls, and its Conversations raw-status count falls.

## What the failures tell us

There is a real source-interpretation problem in this integration. The required evidence is present. In Conversations, C improves separate-booking preferences **0→5/12** and unresolved conditions **4→10/12** versus A, but worsens explicit request corrections **12→8/12** and resolved conditions **10→8/12**. It does not solve the broader problem.

Three concrete source-checked examples are preserved in `analysis/root-case-review.json` inside the evidence archive:

1. `13-en-m1`: availability is explicitly unchecked. A/B give the existing March 14 booking date; C correctly asks for clarification instead of choosing April or May without evidence.
2. `03-en-m1`: the customer explicitly replaces April 4 with May 4. A/B correctly answer May 4; C incorrectly abstains.
3. `18-en-m0`: C restores the correct April 19 preference from B's wrong March date, but falsely reports that an action was already applied. A already answered correctly. The wrong March value matches both current backend state and another booking's old preference, so its exact causal source is unresolved.

The language slices disagree, which a pooled headline would hide. Each cell below has 40 cases:

| Model | English A / B / C | Hindi A / B / C | Telugu A / B / C |
| --- | ---: | ---: | ---: |
| Base | 36 / 32 / 36 | 33 / 33 / 36 | 35 / 34 / 33 |
| Conversations | 33 / 32 / 32 | 31 / 35 / 34 | 32 / 33 / 37 |

All four English Conversations request-correction cases become false abstentions under C; the corresponding Hindi corrections remain correct. Conversely, C falsely abstains on all four Hindi Conversations known-availability cases; A already abstained on two of these and B on one. These observations argue against a blanket rule that prior agent prose should always be removed.

The 56 no-treatment histories have byte-identical B and C requests. Base gives one gain and three losses under those identical inputs, including capped responses; Conversations gives 34/34 correct in both. That is observed run variability, not a prose-removal effect. The requests were single attempts, not repeated until successful. C also shortens B's input, so the treated contrast cannot distinguish an authority effect from a context-length effect.

## Cost and execution

New phase: **720 calls, ₹93.896460 known token-cost estimate, ₹117.370843 retained with headroom**, zero unknown-usage calls. Native research cumulative: **7,757 calls, ₹365.121090 known / ₹488.660182 retained**, under the existing total ₹6,000 ceiling. Older separate exploratory pilots remain separately accounted; these are estimates, not invoices or a dashboard balance.

| Model | A cost | B cost | C cost | Median reporter seconds A / B / C |
| --- | ---: | ---: | ---: | ---: |
| Base | ₹25.815420 | ₹29.891580 | ₹27.947355 | 12.83 / 14.84 / 13.58 |
| Conversations | ₹2.736045 | ₹3.870870 | ₹3.635190 | 1.00 / 1.00 / 1.00 |

Equal call counts and output ceilings do **not** mean equal actual compute: grouped inputs are larger than A, and output/reasoning token use varies. Full token counts and request p95 latencies are in summary.json. These are reporter timings, not end-to-end native workflow or voice performance. Local dispatch timestamps record pacer release, not provider arrival order.

## Decision and product relevance

This is **useful diagnostic progress with a rejected general fix**. We now have measurable gains and losses on preserved source, rather than only passing software tests. We have not established persistent memory as the missing component, an industry benchmark win, reliable 90% accuracy, or a unique product. The separate English LongMemEval result remains 84.6% from its historical single run.

Stop the blanket layout/prose-removal approach and further wording sweeps. The next offline task is to specify a question-specific evidence contract that keeps requested/preferred facts separate from confirmed backend state, preserves conditions, and always derives action status from actual receipts. Treat typed records and source-scoped retrieval as ordinary prior-art controls, not claimed inventions. A future candidate needs matched model/compute controls and fresh, independently reviewed cases before another product claim or broader rollout. See [NEXT_STEP.md](NEXT_STEP.md) and the [current primary-source prior-art check](PRIOR_ART_UPDATE.md).

The 40 scenarios share ten families and translated variants. They are synthetic, development-exposed and assistant-reviewed, not 720 independent real customers or an industry-wide Sarvam failure rate. No real customer data or live business system was changed. Local hashes support integrity and replay; they are not cryptographic provider signatures. The evidence publication package retains failures, source, gold, requests, receipts, cost meter, paired results and all language reviews.
