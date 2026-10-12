# One clearer reporting instruction: a positive development result

Completed October 11, 2026 (America/New_York; execution receipts are October 12 UTC). All 288 calls finished with known usage, zero API errors and zero retries. The paid execution window was approximately 24.2 minutes. Primary offline replay and a separate raw-response/accounting calculation agree.

## What changed

We used every one of the 144 saved guarded histories from the prior source comparison: 72 English/Hindi/Telugu inputs across two model profiles. Each history received one fresh control and one fresh candidate answer. Only the first reporting sentence changed:

- Control: “Report the original service question from this complete source transcript.”
- Candidate: “Answer the original service question using the evidence in the transcript below; do not repeat the question.”

All original evidence, tools-as-data, remaining instructions, model profiles, temperature 0, strict JSON schema and 8,192-token caps stayed fixed. Base uses `sarvam-105b` with low reasoning; Conversations uses `sarvam-105b-conversations` with null reasoning. The saved requests and returned runtime fingerprints are preserved. A model alias is not a guarantee of permanently pinned provider weights.

No memory engine, answer key, answer repair, new tool action or database mutation was used. Expected answers remained exclusively in the offline audit. The existing application status layer derives execution status from trusted intent and actual historical receipts; it does not change dates or clarification.

## Results

| Model | Fresh control factual answers | Candidate factual answers | Paired gains / losses | Empty answers, control → candidate |
|---|---:|---:|---:|---:|
|105B, low reasoning|59/72 (81.9%)|72/72 (100%)|13 / 0|10 → 0|
|Conversations, null reasoning|70/72 (97.2%)|71/72 (98.6%)|2 / 1|0 → 0|

The **predeclared development screen passes** after source review. Neither model loses net factual answers, base improves materially on this set, and unsupported target-date assertions decline from one to zero. This is evidence that our reporting instruction was an important integration weakness in this configuration. It does not reveal the provider's internal mechanism or establish a new memory architecture.

The result is limited to final reporting on saved development histories. These 72 renderings come from 24 related synthetic scenarios; translations are correlated. There is no independent native-language review or fresh workflow/holdout result. **This is not a 100% end-to-end system, a general Sarvam score, or an improvement to the English 84.6% benchmark.**

## What still fails

The candidate Conversations answer for source-20/English unnecessarily abstains after an update is rejected. Both the read and rejection receipts establish current date 2026-11-20. The fresh control answered correctly. This is a real regression retained in the score.

Raw action-status fields remain unreliable. Fully correct raw reports are base 48/72 → 58/72 and Conversations 55/72 → 53/72. Raw status alone is correct base 51/72 → 58/72 and Conversations 57/72 → 54/72. Thus the candidate is **not** a reliable standalone structured reporter. With the pre-existing receipt-owned status layer, factual/application reports follow the main result table. We did not hide raw status regressions or count their correction as factual gains.

The original base/Hindi source-19 action is still missing. The candidate correctly reports what actually happened, but cannot perform an action retroactively. The diagnostic combining new answers with saved desired state is 71/72 for each candidate model, not a new end-to-end score.

The control has ten empty cap-exhausted base outputs, two base question echoes, one base missing clarification flag, one Conversations unsupported current-date substitution and one Conversations unnecessary abstention. The candidate has the one unnecessary abstention above. See FAILURE_REVIEW.md and assistant-review.json for all 16 incorrect outputs and all 55 raw-changed pairs.

## Compute, cost and repeat variation

| Model / arm | Input tokens | Completion tokens | Estimated cost | Median call latency |
|---|---:|---:|---:|---:|
|105B control|80,252|347,766|INR 22.069740|22.06 s|
|105B candidate|80,828|201,792|INR 13.319940|11.47 s|
|Conversations control|90,155|2,918|INR 1.527405|1.00 s|
|Conversations candidate|90,731|2,924|INR 1.536405|1.00 s|

The candidate uses eight extra prompt tokens per request. Caps are matched; actual compute is measured, not assumed equal. Base completion tokens fall by approximately 42.0%. Conversations has no meaningful latency/cost improvement in this run.

Historical guarded factual results were base 58/72 and Conversations 69/72. Fresh unchanged controls are 59/72 and 70/72. These repeated-control differences demonstrate residual variation at temperature 0; the new candidate is compared against the fresh matched controls, not selectively against the weaker historical numbers. Pairs were shuffled and first-arm order counterbalanced before execution. No independence-based significance claim is made.

New known estimate: **INR 38.453490**; retained including 25% headroom: **INR 48.066950**, below the INR 625 phase ceiling. All new usage is known. Native research cumulative: **6,080 calls; INR 169.754685 known; INR 244.451544 retained**, inside the existing total INR 6,000 allowance. Earlier separately authorized memory pilots add 247 calls and INR 4.400490 known estimates. Across the documented Sarvam series: **6,327 calls and INR 174.155175 known estimates**, with older unknown usage still unpriced. These are token-rate estimates at the recorded 15/60 microINR rates, not invoices or verified account balances.

## Verification and publication

Before execution, 154 focused tests and a complete synthetic-response simulation passed. After execution, all 288 raw requests/responses, source bindings, unique response IDs, usage charges, reservations and saved classifications replayed. Independent aggregation reproduces every factual grade and the execution meter. All 72 source renderings, 16 incorrect outputs and 55 raw-changed pairs are reviewed by the assistant. No independent native-language review is claimed.

Source code is committed separately as d64552a. Evidence publication, clean committed export and exact PR/main CI verification follow this report; final receipts in the dedicated memory record completion. Founder file inventory is excluded from the public archive. Original English work and older Sarvam results remain unchanged.

## Decision and next step

Keep this exact candidate for a fresh end-to-end confirmation; do not promote it yet. NEXT_PHASE.md specifies the preparation and acceptance conditions. No second paid phase was launched this turn. SARVAM_HISTORY.md explains the entire recorded Sarvam investigation, including our own integration/dataset mistakes and negative results. The genuine progress is a measurable reporting improvement with the same models, lower base completion usage, and a better ordinary-system baseline against which future memory work must prove its value.
