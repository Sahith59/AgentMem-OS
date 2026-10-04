# Terra evaluator and one evidence-focus lever

Defined before implementation, 2026-10-04. No paid authorization.

Acceptance checks:

- Preserve historical GPT-4o answers/grades and existing evaluator contracts.
- New judge is explicitly `gpt-5.6-terra`, low reasoning, 2,048 total completion tokens, strict yes/no. Pin the per-type rubric and injection-boundary instruction. Fail on truncation, unexpected model, invalid usage, bad verdict, changed sources or unresolved attempts. No retries.
- Build 32 exposed development judgments:26 from the13 semantic fixtures and6 from three previously reviewed reference-grading disputes and 120 prospectively frozen internal validation judgments from 60 separately authored scenario families, 60 positive/60 negative. Paired judgments are correlated; do not call them 120 independent scenarios or independent human validation.
- Labels come from authored source facts and the rubric, never old GPT-4o grades. Keep labels, rationale, source excerpts, IDs and categories out of provider requests except question/type-dependent rubric/reference/response needed by the evaluator.
- Validation gate: at least114/120 correct, at most3 false accepts, at most3 false rejects, zero critical injection-case errors. Report by rubric and family, including uncertainty. Never tune and reuse this validation as fresh.
- Save a separate500-answer bridge package without regeneration; no bridge launch until calibration passes and an exact budget is approved.
- One experimental lever: Terra selects bounded verbatim source excerpts for an optional focus block. Luna answerer, answer prompt, extraction corpus and original context stay fixed. No generated answers, arithmetic or claims enter the focus block. Reject invented quotes, extra fields, missing/duplicate IDs and budget overflow. Selection may still be semantically wrong; integrity tests do not prove accuracy.
- Measure selector quality on the exposed semantic fixtures before a benchmark screen; reject any wrong required/excluded selection. A failed selector is not promoted.
- Offline tests must cover failure, resume/tamper, label isolation, partial grading, source preservation and budget boundaries. Report no accuracy lift without actual model receipts.
