# Indic memory research roadmap

Updated October 13, 2026. This is an evidence-first research direction, not a claim that a multilingual memory product is ready. The [current Sarvam research index](SARVAM_RESEARCH.md) links actual experiments, failures and the next comparison. The [older roadmap](history/INDIC_ROADMAP_BEFORE_2026-10-11.md) is preserved as history; its universal competitor and first-invention claims were not established and are withdrawn.

## Scope and product gate

The intended eventual coverage is the languages supported by the relevant Sarvam services, with service/model/version recorded. We start measurement with English, Hindi and Telugu because we can review those cases. Coverage is earned per language and task; one translation does not establish full support. Neither API integration nor startup-program acceptance demonstrates product quality.

Before designing new memory architecture, identify a reproducible failure in native Sarvam with a strong ordinary database, correct API use, clear instructions and verified tool permissions. Separate source interpretation, retrieval/persistence, action execution, reporting and evaluator errors. If an ordinary integration fix solves a case, publish that result. Do not label it a missing Sarvam memory capability.

Then test remaining persistent-information problems across sessions and languages with controlled identity, corrections, temporal scope and evidence provenance. A new component needs an equal-model/equal-budget comparison against strong ordinary controls. Product selection also requires a concrete user workflow and measurable benefit, not merely a difficult synthetic example.

## Public evaluation is required

Publish protocols, licensed or synthetic case sources, exclusions, separate generation/label files, code revisions, raw receipts, cost/latency, per-language metrics, paired gains/losses and repeats. Current development artifacts already do part of this. An independent benchmark additionally needs reviewed diverse cases, held-out families and independent language review. Translations and parameter variants must be grouped; they are not independent samples. Competitor comparisons require actual matched reproductions and appropriate prior-art review. No competitor score or novelty outcome is assumed.

## Historical implementation evidence

The repository contains experimental cross-lingual entity aliases (`ALIAS_OF`) using multilingual embeddings. An earlier hand-labelled EN/Hindi/Tamil set reported precision 0.762 / recall 0.533 at threshold 0.90. This is a historical draft-label measurement, not verified end-to-end multilingual memory quality. False aliases can contaminate retrieval and downstream answers even when original facts are not overwritten; non-destructive edges do not eliminate that risk.

The English record remains historical 423/500 once under its original judge; provisional Terra 425/500 grades the same answers differently. Neither is an Indic score. Read [RESULTS.md](RESULTS.md) for the English context and [Sarvam history](../benchmarks/audits/sarvam-reporter-wording-2026-10-11/SARVAM_HISTORY.md) for why the early multilingual memory advantage did not survive repetition.

## Sequence

1. Native confirmation, one-case recovery and the720-call source-interpretation comparison are complete. The latest candidate fails reliability checks; next specify a question-specific evidence contract offline before any new paid comparison.
2. Establish a strong ordinary-storage baseline, then investigate persistent, cross-session and cross-language evidence failures with newly reviewed cases. Do not assume a failure in advance.
3. Select the smallest product component only if it fixes a reproduced residual problem and improves an observable workflow outcome.
4. Publish a reviewed multilingual benchmark and matched baselines; expand language and domain coverage in measured stages.
5. Add speech-path testing only with a text baseline, separating recognition mistakes from memory and action failures.

No date for a novel product or industry-leading score is guaranteed. A result showing that ordinary controls suffice is also a valid research outcome.

## October 13 checkpoint

The corrected 240-episode comparison found a fixed-model base105B reporting gain (90→105/120), but the candidate failed acceptance: two reporter slots were never reached and Conversations wrong non-null dates increased 18→22. The separately proposed 108-call evidence-view screen is blocked and unexecuted; its selected capped history cannot be silently replaced. Source/condition interpretation and a skipped reporting handoff are concrete integration gaps. They do not yet establish a persistence problem or a novel memory product. See the [complete report](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/REPORT.md) and [bounded next task](../benchmarks/audits/sarvam-fresh-confirmation-v3-2026-10-13/NEXT_STEP.md).


A separate two-call recovery diagnostic subsequently produced a correct final report in both original variants for the sole capped English case, with no repeated action. The old v3 scores/gate and blocked108-call proposal stay unchanged. This removes one observed reporting failure; it does not establish persistence as the remaining cause. The subsequently completed all240-history source-interpretation comparison is described below. [Recovery result and next step](../benchmarks/audits/sarvam-capped-recovery-2026-10-13/REPORT.md).

## Latest source-interpretation decision

The720-call comparison measured base104→105/120 and Conversations96→103/120 date-plus-clarification versus chronology, but failed predeclared reliability checks: more false abstentions on both models and worse status reporting/coverage. No candidate promotion or novel memory claim. Stop blanket prose removal; next investigate question-specific source interpretation against strong ordinary controls, with actual compute and fresh-case evidence. The historical English score is unchanged. [Report](../benchmarks/audits/sarvam-source-interpretation-2026-10-13/REPORT.md) and [next proposal](../benchmarks/audits/sarvam-source-interpretation-2026-10-13/NEXT_STEP.md).
