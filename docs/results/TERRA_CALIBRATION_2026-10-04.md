# Terra replacement-judge calibration — completed

**Development32/32; internal validation118/120 (98.3%). The frozen calibration gate passes.** These are grading checks, not English-answer accuracy. The historical English result remains423/500=84.6% under GPT-4o grading, one run.

The founder approved at most152 Terra calls/$4.45 across both stages. The32 development judgments passed before validation began. All152 calls completed once with no provider failures or retries, using `gpt-5.6-terra`, low reasoning, a2,048-total-completion-token cap and the frozen preparation003 inputs. No prompts, labels or model settings changed after outputs. All saved requests, receipt identities, verdicts, source bindings and reservations passed independent reconstruction. Raw evidence remains in the local durable-memory directory; [the committed aggregate](terra-calibration-2026-10-04.json) records checkpoint/package/audit hashes.

| Frozen validation measure | Result | Gate |
| --- | --- | --- |
| Agreement with internal labels |118/120|At least114/120|
| False accepts against frozen labels |1/60|At most3|
| False rejects against frozen labels |1/60|At most3|
| Critical instruction-attack errors |0/2|Zero|
| Scenario families with any error |2/60|Reported separately|

All10 abstention judgments,20 preference judgments,18 update judgments,18 user-recall judgments and18 assistant-recall judgments matched their labels. Multi-session and temporal judgments each matched17/18.

## The two disagreements

1. **Extra answer / source-policy ambiguity:** the reference says the aquarium; the answer says zoo and aquarium. The authored source says the zoo remained a plan, so the internal label is negative. However, the judge receives the reference, not that source, and the rubric says to accept an answer containing the correct answer. Its acceptance may follow that permissive reference rubric. This is not an unambiguous judge defect. Preserve the frozen118/120 result; source faithfulness needs a separate audit.
2. **Clear temporal-rubric miss:** a response of five months against a four-month reference was rejected even though the rubric explicitly permits off-by-one month errors. Preserve the error; inspect temporal cases in the bridge audit. Do not tune on this validation and present a rerun as fresh.

The60 validation scenarios are short, internally authored and paired into120 judgments. Passing them does not certify performance on ambiguous real conversations or establish independent human calibration. The grading model and answerer are from the same provider/family. The result permits a bounded bridge experiment, not an English90% claim.

## Cost and next step

Total reservation was **$4.43249950**, within the approved$4.45 cap. The successful-response usage upper estimate was **$0.073447**, using frozen rates and no cache discount; this is not a billing invoice. Input/output usage was26,182/666 tokens, including46 reasoning tokens within output. No keys or account balance were exposed. Two environment checks stopped before dispatch; an isolated OpenAI SDK2.54.0 environment supplied the working client without modifying the repository venv.

The next prepared proposal is to regrade the **same500 saved Luna answers**, without generating new answers, for at most500 Terra calls and a conservative **$14.75 cap**, no retries. The actual validation receipt now satisfies its prerequisite. It is **not yet approved or executed**. Audit every changed grade and a prospectively defined stratified sample of20 agreements, keeping reference agreement separate from source faithfulness. This measures the effect of changing judges; it is not architecture improvement. A selector-quality check and a controlled Luna answer experiment remain later steps.
