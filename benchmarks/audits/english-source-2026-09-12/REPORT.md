# English source audit — September 12, 2026 EDT

**The measured score is still 418/500 = 83.6%, reproduced twice.** This audit identifies concrete evidence gaps and implements an offline-tested candidate. It does not establish a new QA score. No paid calls occurred. Sarvam remains parked.

## What was reviewed

- Screened all 72 persistent misses using the question, both saved answers, original source turns, scoped retained facts and frozen packet. [Triage notes](triage.json) cover every ID. These are review priorities, not 72 finalized causal diagnoses.
- Built 72 hash-indexed dossiers containing 1,930 source turns and 15,304 scoped fact rows. Answerable dossiers contain annotated source sessions; abstention dossiers contain the whole permitted scope. The full texts are preserved for review. Saving or searching them is not a claim to have semantically read every sentence of every assistant turn.
- Recorded 16 detailed source witnesses across 13 cases, including retained-fact inspection. [Witness receipts](reviewed-witnesses.json). Other triage items remain provisional.
- Reconstructed six judge requests across three disputes and matched their saved request hashes. [Judge review](judge-request-review.json). Official labels are unchanged.
- Checked upstream-to-cache source lineage automatically across all 500 questions, using the pinned cleaned dataset revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`. Both dates and complete turn contents match. The audit also reproduced all 500 paid-run baseline packets byte-for-byte before comparing retrieval changes.

## Confirmed gaps and limits

| Layer | Source-backed example | Implication |
| --- | --- | --- |
| Retained memory detail + raw fallback | Nightingale page250, newer yoga three-times-weekly frequency, Tuesday/Thursday15-minute wake-up adjustment and12rarefigurines are absent from their source session's retained facts. Their necessary raw turns were also omitted. | Recover original detail before assuming a stronger answerer can solve the packet. The audit identifies persisted-data gaps; it does not yet distinguish original extractor omission from migration/filtering. |
| Raw retrieval | The512-term vocabulary removes `nightingale`, `speyer`, `tuesdays`, and `thursdays` in the corresponding histories. The Speyer phone reply is omitted despite delivery of adjacent user turns. | Rare query terms and their original replies need a fallback. A session hit is insufficient. |
| Evidence use | The packet contains37coins onMay27 and a later added quarter onMay29; both explanations still answer37. March26 work-related missed-run wording is also delivered but overlooked. | Some misses need better evidence interpretation. This candidate is not a complete reasoning fix. |
| Abstention policy | Questions swap iPad for iPhone, an undergrad project for a thesis, or ask to order Ferrari against an unsupported Porsche project. | The old prompt's bias toward an imperfect answer and refusal only when evidence is empty is a plausible contributor. Do not silently infer the missing entity or operand. Prompt causality still needs an isolated comparison. |
| Judge/output contract | “Sound effects” is rejected against “Sound effects”; Sephora answers explicitly include100more; sculpting answers enumerate the tools in source. | Likely grading false negatives. Preserve official labels and review separately; changing the judge is not a memory improvement. |
| Source/reference consistency | Pinned source datesJanuary19→April10 imply81days/11weeks4days, not reference15weeks. AMarch21“yesterday” baking class toApril15 implies26days, not21–22. | At least these date anchors conflict with references upstream. Do not distort chronology to chase those labels. A complete benchmark-label audit is not claimed. |

## Corrections to the previous diagnosis

The previous “all gold sessions present” metric required only any complete turn from each session. It did not establish that the answer-bearing information was delivered. Its “literal gold” proxy could accept a single number anywhere or partial word overlap. Both were overinterpreted in D-131/EI-101 and the earlier current-state paragraph.

Among64 answerable stable misses,33 have that session-hit proxy, but20 of those still lack at least one complete upstream-annotated answer turn. Only13/64 have every annotated turn intact. Even complete annotated turns are not proof of sufficient, unambiguous evidence; facts can also supply details when raw turns are absent.

The observed two-run union85.6% is a description of those two runs, not a mathematical ceiling for future samples. Another identical run still has little decision value. The earlier architecture-first conclusion and automatic event-ledger plan were premature. Historical structured-answerer trials already scored57.3% and65.3% with substantial regressions on their test population; new ledgers require a different, tested hypothesis.

## Two offline experiments

| Measure | Uncapped raw ranking | Preserve packet + source supplement |
| --- | ---: | ---: |
| Original packet unchanged as a prefix | No | 500/500 |
| Annotated turns recovered across all500 | 213 | 197 |
| Previously delivered annotated turns lost | 12 | 0 |
| Persistent-miss questions gaining annotated turns | 40/72 | 38/72 |
| Annotated turns gained within persistent misses | 50 | 47 |
| Stable-pass questions losing annotated turns | 10 | 0 |
| Prewritten offline gate | FAIL | PASS |

The first candidate improves coverage but is rejected for promotion because of losses. The second appends complete source turns ranked without the vocabulary cap, using at most4,000 additional characters and never exceeding the existing40,000-character cap. It retains every original byte. Its4,190 added turns all have verified source hashes, roles and packet offsets. The average addition is3,881.5characters, so input usage will increase; this is not an equal-token ranking-only experiment.

[First experiment](uncapped-summary.json), [supplement results](supplement-summary.json), [independent verifier](independent-verification.json).

The first harness attempt used the older110-builder session prefix and stopped immediately on a baseline mismatch. It is preserved in attempt001 in the outer memory. The corrected run reproduces all500. No failure was hidden and no paid result was overwritten.

## Next decision

The candidate remains opt-in; the measured default and model/prompt configuration are unchanged. Sixteen focused tests pass, including tampered source, speaker, offset, hash, prefix, rendering and budget checks. Independent verification covers all500 generated packets. QA accuracy and distraction regressions remain unmeasured.

A [150-pair screen design](next-screen-policy.json) is frozen: all72 persistent misses, all20 disagreements, and58 deterministically selected stable-pass controls, including all30 abstention questions overall. Gate: at least10 net new correct answers, at most2 stable-control losses, no net abstention regression, and no unresolved jobs. Keep the exact same answerer, judge and answer prompt. Preserve disputed official grades and separately disclose assistant review. This is a development-exposed diagnostic, not a held-out headline score.

Next engineering task: make that design an executable, priced, independently validated paired package. Obtain approval for its exact budget only after preparation is complete. A full500 run follows only if the controlled screen supports promotion. No provider execution is authorized by this audit or by the existence of credits.

## Reproduction and artifact storage

The full local source/cache/DB/packet artifacts remain under `codex-memory-2026-09-08/runs/english-improvement-2026-09-11/` in the outer workspace. This Git record backs up the review, candidate, verifier, compact results and provenance hashes; it is not a remote backup of the entire outer memory. The original baseline commit is `7135249fd86c03375e243fc82bcb042850c24c68`; candidate implementation commit is `b06e097`.

From the outer workspace, `python3 AgentMem-OS/benchmarks/audit_source_supplement.py codex-memory-2026-09-08/runs/english-improvement-2026-09-11 NEW_OUTPUT_DIRECTORY` builds a new supplement audit. `python3 AgentMem-OS/benchmarks/verify_source_supplement.py AUDIT_DIRECTORY NEW_VERIFICATION_FILE` independently verifies it. These commands require the frozen local inputs and an installed project; they perform no provider calls.
