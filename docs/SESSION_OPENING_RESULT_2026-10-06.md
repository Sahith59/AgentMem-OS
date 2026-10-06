# Session-opening screen: completed, quality gate failed

October6,2026 America/New_York. Exactly ONE paid run this turn; no retries or second execution.

## Measured outcome

| Same32 development questions | Correct |
| --- | --- |
| Fixed Luna baseline | 29/32 |
| Fixed Luna with source-opening expansion | 28/32 |

There were zero gains, one loss and three shared misses. All128requests completed and all32pairs were scored. The raw status name `FAIL_OR_INCOMPLETE` means quality-gate failure here, not incomplete execution. The predeclared gate required at least3netgains and at most1loss. Do not promote this candidate or run a full benchmark on this evidence.

This small exposed sample is not the500-question English score. Historical423/500(84.6%) and separate provisional-Terra425/500(85.0%) remain unchanged. Neither29/32 nor28/32 establishes English closure, generalization or90%+ overall accuracy. The sample was not resampled based on observed outcomes.

## What the experiment revealed

- Kitchen count (`gpt4_ab202e7f`): the complete coffee-maker donation/upgrade turn reached the candidate, near the end of its context. Both arms still answered four rather than the reference five. This establishes failed answer use after delivery for this response. Attention, presentation and semantic admission of an upgrade as replacement are competing explanations; this run does not distinguish them.
- April attendance (`10d9b85a`): a matching workshop session was retrieved, but its opening omitted the later turn specifying a two-day April17/18 workshop. Together with the April10 lecture, those sources support three days. Both arms answered one. Returning a session opening is not equivalent to returning its relevant evidence.
- Current-role duration (`92a0aa75`): the candidate supplies28months until promotion but omits the later source stating45months total tenure. It answers28months in the current role; the reference uses45−28=17months. This is partial evidence and a wrong temporal relation, not demonstrated inability to subtract.
- Commute regression (`1c0ddc50`): both answers recommend history podcasts; the candidate additionally recommends reading on the bus. The reference prefers audio and cautions against visual attention, giving the rejection a plausible rubric basis. No explicit source prohibition on bus reading was identified, and one binary judgment cannot establish that appended context caused the loss. Frozen grades remain unchanged.

The expansion often appends unrelated opening topics because a later turn earned the session's lexical score. This is an observed limitation, not a successful general retrieval fix. Thirty packets expanded and two remained unchanged. All142appended original-source spans were checked before execution. Source-opening retrieval remains benchmark-only and unpromoted.

## Integrity, models and budget

Answerer remains gpt-5.6-luna; judge remains gpt-5.6-terra with its frozen contract. No extraction model/corpus change, planner call, prompt tuning, reference injection or benchmark answer hardcoding occurred. Sources were projected as date/role/text only, from the same question scope, with sessions later than question time filtered before ranking. The baseline packets were left unchanged; this does not newly certify their temporal cleanliness.

Package SHA256:207c5a4f3697be50534c3b6ac0a593c1d313334228b799bd631bbe43c96df215. Code/protocol commits81abd31/df01d66 merged via PR34 at a00804c7f77accb87016fd69334957522233fbeb; PR CI37416528088 and post-merge CI37416735423 passed.158focused tests and Ruff passed before execution. Independent GPT-6 Sol review first caught a future-source leak in the draft; it was repaired, revalidated and frozen before the sole paid execution.

128distinct provider response IDs and128distinct request IDs; complete source/code/package/approval bindings; independently reproduced summary; zero unresolved usage attempts. Reservation:$4.349651 under approved$4.35. Receipt-derived upper usage estimate:$0.20943265, not a provider invoice.

| Actual model tokens | Input | Output |
| --- | ---: | ---: |
| Baseline answers | 292531 | 5041 |
| Expansion answers | 301530 | 4741 |
| Baseline judgments | 8669 | 128 |
| Expansion judgments | 8329 | 429 |

Both arms used32answer+32judge calls and identical output/context ceilings. Actual tokens differ; this is not exact realized-compute equality.

## Decision and next work

Keep the failed result and all raw outputs. Do not promote broad session-opening expansion and do not launch another paid run this turn. The user's authorization was exactly one run; notify them before any proposed paid work in later turns.

Next offline hypothesis: retrieve the specific matching turns with bounded local context, retain source attribution and qualifiers, and check whether the evidence needed for the requested count or time relation is complete. Use gold-blind runtime inputs, diverse regressions and unchanged controls. No hardcoded kitchen rule, prefilled reference answer or oracle session IDs. Evidence delivery and answer use require separate checks. Freeze any future intervention separately; no future budget is approved by this report.

Local evidence: `plans/2026-10-06-session-opening/preparation-001/{package,approval,preflight}.json` and `paid-screen-001/{checkpoint,summary,post-run-audit}.json` in the new memory. The public summary and compact audit are under `benchmarks/audits/session-opening-2026-10-06/`. Original prior runs remain unchanged.
