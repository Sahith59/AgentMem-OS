# Fresh workflow confirmation — stopped; no promotion

October 11 local / October 12 UTC, 2026. **This is not a successful completed comparison.** We stopped after 32 of 240 planned native episodes because assistant source review identified an ambiguous question in our new case bank. The other208episodes were never attempted. The completed prefix is model-imbalanced (14 base,18 Conversations), chosen by the frozen random order rather than outcome selection.

## What went wrong in our test

The `missing_replacement` family withdraws an earlier date and says no replacement has been chosen. The question asks for the “most recently requested date,” which can be read historically. Gold assumes the currently desired replacement and expects clarification. Our template-assisted review missed this distinction before launch. We own this test-design error; it is not evidence of a Sarvam defect. All12renderings of that family need a prospective question correction; four such episodes had run. Six of the14frozen-contract factual misses are in that ambiguous family, including one missing output. Keep the original grades and ambiguity annotations; do not manufacture a new headline by excluding inconvenient cases.

One main-thread SIGINT stopped new batches. The executor let its active workers finish. All32started episodes completed, with111known-usage calls, no API errors, retries or lost receipts. The main runner exited130 before writing its final execution file; `execution.json` was explicitly reconstructed from complete immutable per-episode results and the ledger. `interruption-reconciliation.json` records that origin. Primary replay and a separate raw-parser/SQLite/accounting implementation agree.

## Frozen descriptive counts — incomplete, ambiguous subset

| Fixed model | Control facts | Candidate facts | Raw status control→candidate | Safe desired state |
|---|---:|---:|---:|---:|
|105B low reasoning|9/14|12/14|9/14→8/14|14/14 both|
|Conversations null reasoning|14/18|15/18|15/18→17/18|18/18 both|

These are preserved mechanical counts, **not a valid final accuracy or improvement claim**. Base control has four unusable capped outputs; candidate has none. The base count of non-null target mismatches increases1→2, but both candidate mismatches concern the ambiguous question family. Conversations mismatches3→2 also include that family. The planned gate remains failed/incomplete; no candidate is promoted. All three attempted legitimate required changes complete, and two unnecessary Conversations attempts are blocked because approval is not effective. No forbidden state change occurs in this small executed subset. This is not a production safety proof.

## Concrete leads outside the ambiguous family

- **Unknown condition becomes a chosen date:** `fresh-16-te`, Conversations. Source says choose April17 if a morning slot exists, otherwise May17, and explicitly says availability has not been checked. The native actor selects the first option with conditional wording; both final reporters output2027-04-17 with no clarification. The database read supplies the current booking date, not slot availability. The required branch condition is absent.
- **Known condition is ignored:** `fresh-09-hi`, Conversations. Source explicitly states that the morning slot is available; the read-only question asks which preferred date follows. The native actor focuses on lack of approval to change the booking; both reporters abstain. Permission to execute is not needed to report the preference implied by that stated condition.
- **Conditional phrasing is lost during reporting:** `fresh-14-en`, Conversations. The actor states both alternatives conditionally. Control collapses this into one definite date; candidate correctly returns clarification. This is one observed paired difference, not a repeated causal claim.

These are reproducible saved examples of evidence/condition interpretation in our integration. Their broader frequency, sensitivity to the actor prompt, language dependence and persistence across fresh repetitions are unproven. They do not establish that Sarvam lacks memory or that our English memory engine solves the issue. Raw model status errors also remain; the existing receipt-owned application changes status, never dates.

## Review, costs and next action

Reviewed all64executed report outputs, all14frozen-contract misses, all16raw-changed pairs and both unnecessary action attempts;20episodes are documented in FAILURE_REVIEW.md. Assistant review only. The source bank has40related scenarios in ten inherited families, three language renderings and a synthetic frozen clock. It is not an independent benchmark. API model aliases are recorded, not an independently pinned provider weight revision.

New known token estimate **INR10.225170**, retained with headroom **INR12.781506**. Native cumulative **6191calls**, known **INR179.979855**, retained **INR257.233050** under the original totalINR6000. Including separately authorized older pilots:6438calls,knownINR184.380345, with older unknown usage still unpriced. These are estimates, not invoices or an account balance. No second paid phase ran and no inference remains active.

The corrected version changes only the12ambiguous questions and their runtime-hash bindings, preserves the full case population, and states clearly that the original32episodes exposed part of the development set. It will be prepared and verified offline. Never resume or restart the stopped manifest. The next live phase requires the corrected frozen package and a launch notice under standing budget; no duplicate key or blanket permission is needed. Product discovery continues: first separate condition interpretation from action-permission handling and propagation of prior assistant claims. Do not build a new memory architecture around an untested explanation.
