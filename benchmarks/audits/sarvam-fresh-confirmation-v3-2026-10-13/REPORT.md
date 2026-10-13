# Corrected fresh-workflow comparison v3 — October 13, 2026

All 240 native episodes were attempted once. The table counts all 120 episodes per model, including failed or unreached final answers. This is a development comparison of one reporting sentence using fixed model profiles and shared native histories; it is not an English baseline, independent benchmark or new memory result.

The frozen mechanical gate is **FAIL**. The complete-pair requirement is **not met**: 478 reporter calls were issued across 480 planned reporter slots; 2 slots were not reached. No candidate is promoted. Conversations non-null wrong dates also increased from 18 to 22, violating a second frozen requirement. See [INTERPRETATION.md](INTERPRETATION.md) for the gain and failure breakdown.

| Model | Control: correct date + clarification | Candidate | Paired gains / losses |
|---|---:|---:|---:|
| sarvam-105b | 90/120 (75.0%) | 105/120 (87.5%) | 20 / 5 |
| sarvam-105b-conversations | 95/120 (79.2%) | 96/120 (80.0%) | 4 / 3 |

## What the scores do and do not mean

The reporters differ only in their first system sentence. Both share the same original customer records, native actor calls, tool receipts and backend state. The host projects execution status from actual receipts, but never changes the model’s answer or clarification flag. Empty/capped responses fail. Null with clarification=false can preserve the correct absence of a date while still failing the frozen date-plus-clarification contract; date-only totals are shown below.

| Model / arm | Usable final | Date only | Clarification | Raw status | Safe fact + state | Strict task | Non-null date mismatch |
|---|---:|---:|---:|---:|---:|---:|
| sarvam-105b / control | 100/120 | 94/120 | 96/120 | 76/120 | 90/120 | 90/120 | 5 |
| sarvam-105b / candidate | 116/120 | 114/120 | 107/120 | 87/120 | 105/120 | 105/120 | 2 |
| sarvam-105b-conversations / control | 120/120 | 95/120 | 106/120 | 106/120 | 94/120 | 85/120 | 18 |
| sarvam-105b-conversations / candidate | 120/120 | 96/120 | 108/120 | 107/120 | 95/120 | 84/120 | 22 |

## Actions, evidence and the blocked follow-up

Across 240 shared native episodes: 48 required new writes, 1 missed required writes, 11 unnecessary update attempts and 0 forbidden state changes. These are shared action outcomes, not gains caused by the reporter sentence.

`confirm-fresh-13-en-m0` returned HTTP 200 with finish_reason `length`, 8,192 completion tokens (8,191 reported reasoning tokens), null content and no tool calls on its first native call. The integration classified this as `invalid_action_response` and never issued either reporter. No state changed. This is not exhaustion of three actor calls. The original result remains unchanged.

The separately proposed 108-call evidence-layout/prose comparison includes that history and requires a complete parent comparison plus an existing candidate request. It cannot pass preparation. It remains **unexecuted**; no case was dropped or substituted, and no gate was weakened. An opt-in recovery route is engineering work only and must be tested prospectively before any accuracy claim.

## Language and failure review

Assistant reviewers checked all 240 native episodes and 480 reporter slots, including absent outputs. Actual outputs and source interpretations were cross-checked against saved records; no independent native-speaking human review was performed. The original v1 had 32 development-exposed episodes. V2 fixed 12 historical/current questions; v3 additionally fixed 12 preference/request questions before execution. Gold dates, sources, backend contracts, actor and reporter prompts were not revised after outcomes. This is 120 renderings of 40 related scenarios from ten inherited families, not 120 independent workflows.

| Model / language | Control correct | Candidate correct |
|---|---:|---:|
| sarvam-105b / en | 28/40 | 33/40 |
| sarvam-105b / hi | 31/40 | 36/40 |
| sarvam-105b / te | 31/40 | 36/40 |
| sarvam-105b-conversations / en | 34/40 | 34/40 |
| sarvam-105b-conversations / hi | 32/40 | 31/40 |
| sarvam-105b-conversations / te | 29/40 | 31/40 |

All family/language metrics are in `slice-summary.json`; every miss, changed output and raw-status mismatch is enumerated in [FAILURE_REVIEW.md](FAILURE_REVIEW.md). Full language judgments and receipt hashes are in the archive. Related cases limit generalization; no independent-binomial confidence interval is asserted.

## Cost and verification

This phase made 844 calls; known token-cost estimate ₹90.677925, retained amount with headroom ₹113.347725, unknown-usage calls 0. Native research cumulative: 7035 calls, ₹270.657780 known / ₹370.580775 retained under the original total ₹6,000 authority. These are usage estimates at published ₹15/₹60 input/output per million tokens, with cache discounts ignored and 25% headroom; they are not an invoice or balance check.

Primary audit replays all requests, receipts and local SQLite transitions without network calls. A separate raw parser and arithmetic check agrees. Runtime/gold files regenerate from the committed v3 builder. Archive checksums and clean-export verification are published with the evidence. The earlier stopped run and unexecuted v2 remain preserved.

See [NEXT_STEP.md](NEXT_STEP.md) for the bounded next task.
