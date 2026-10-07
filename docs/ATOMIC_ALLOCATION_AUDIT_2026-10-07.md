# Complete-bundle allocation: implementation works, readiness fails

October7,2026 (America/New_York). **The requested packing improvement is implemented and audited across all500 questions. It does not justify another paid answer run.** Historical full-set accuracy remains423/500(84.6%,once), with425/500 a Terra regrade of identical answers. No new answers, paid LLM calls or model changes occurred.

## What changed and why

Greedy admission can spend the remaining context budget on one highly ranked source bundle and crowd out several other complete bundles. The new opt-in `allocation="joint"` examines every subset of the same eight novel anchors, at most256 combinations. It chooses the fitting combination with the highest sum of existing retrieval scores for wholly covered anchors.

Every admitted anchor retains its whole original-session +/-1 neighborhood. Shared source turns cost space only once, and complete sources already certified in the baseline are reused. An indirectly completed anchor contributes its score once even if it was not a seed of the chosen subset. Exact chronological rendering, source IDs, roles, dates, hashes, offsets and the4,000extra/40,000total character limits remain unchanged. Future or gapped neighborhoods are rejected. No original text is clipped or rewritten. All500 original baseline prefixes remain intact.

A higher sum of retrieval scores is an optimization objective, not a promise of more correct answers. Several shorter sources can displace a larger source containing a crucial fact. This tradeoff was frozen before outputs. Normal assembly and greedy selection remain the defaults; the new policy is not promoted.

## Frozen all500 result

The control reproduces the corrected October6 source-aware v2 candidate byte-for-byte, including its receipts and report. Ranking, top-eight anchors, source eligibility, presence ledger, models and budgets are identical between arms; only admission differs.

| Measure | Greedy control | Joint allocation |
| --- | ---: | ---: |
| Admitted complete neighborhoods | 876 | 1,017 |
| Appended source turns | 1,733 | 1,958 |
| Questions with all annotated turns present, of479 | 365 | 367 |
| Historical misses gaining annotated text vs original baseline | 7/77 | 7/77 |
| Historical-miss evidence changes vs control | — | 2gains / 2losses |

There are131 changed packets;130 have higher retrieval-score utility. Across all500 questions, six cases gain and five lose annotated text relative to control. Among423 historically correct cases, four gain and three lose. Eight abstention packets change without changing annotated-turn presence. Original baseline text is never lost; losing a control-added source still matters.

The prospective gate required at least10 historical misses gaining annotated text vs original baseline and at least5 net gains vs matched control, complete integrity and source review. Actual values are7 and0. **FAIL.** More admitted neighborhoods did not produce more useful annotated evidence in the historical-miss cohort. No objective weights were retuned after this result.

## Source-level interpretation

The new phone-accessory source names a power bank and wireless charging pad, but the saved answer already mentions a power bank. The new clinic source says a previous journey took two hours; this does not establish that Monday's journey took exactly two hours. Neither is an established repair.

The candidate loses the original commuter-bike tire replacement plan for March, which was a plausible missing contribution to the saved one-bike answer. It also loses an EP purchase turn whose underlying purchase was already counted. These four changed historical misses show why a larger topic-relevance score cannot be treated as answer-fact coverage. No saved answer or grade was changed.

Independent qualitative review of all15 required cases is complete: these four historical misses, three historically correct coverage losses and eight changed abstentions. In the bike case, utility rises from0.04993 to0.06050 while the March commuter-bike repair plan and clarifying hybrid-bike turn are replaced by road-bike routes, shops and service advice. A historically correct museum question also loses raw disambiguating evidence mentioning the user's father. Other lost raw turns retain corresponding facts in the baseline. Changed abstentions retain unresolved person, role, location, price or frequency evidence; no answer stability is established. See `source-review.json`; this is assistant source review, not independent human calibration.

## Why no automatic smaller-span compression was added

The previous source review found26 blocked user turns across21 historical misses. Fourteen full neighborhoods exceed even an empty addition budget, so an allocator cannot fit them without omitting text. Twelve other turns were blocked by competition for remaining space. This experiment tested that competition mechanism only.

A separate exact-prefix reuse diagnosis inspected44 required partner source pairs. Three complete neighbors were already handled by the ledger; just one had a plausible new partial-prefix reuse opportunity. That case still has unresolved project identity. It does not justify building a general partial-source parser or claiming safe semantic compression. Arbitrary query-similarity clipping can remove a date, negation, condition, antecedent or assistant-authored comparison. Exact offsets make omitted text traceable; they do not prove its omission is harmless.

## Verification

- 121 focused offline tests passed, including14 new allocation cases. Tests exercise complete qualifications, shared context, incidental anchor coverage, exact budget accounting, deterministic ties, future/gap rejection, fixed candidate bounds, numeric overflow and the real assembler entrypoint with an injected encoder. Ruff and diff checks pass. An initial new-test bracket typo and a synthetic tie fixture that accidentally fit both sources were corrected before benchmark output; the policy was unchanged.
- Independent verification reconstructs127,505 subsets across500 questions and confirms every exact optimum and tie. It verifies500 reproduced greedy controls, identical top-eight candidates,3,691 appended source receipts and30,336 counted baseline-presence certificates across both arms, plus source metadata, scope, time, hashes, offsets, packet costs and full admitted-neighborhood coverage.
- The frozen candidate/control packet file is SHA256 `d6011cbbc231d788ef99eea28369de5f0b2e553dbd81414750bf29d4250b12bc`. Code, inputs, cached vectors, script and policy hashes are retained. Construction took28seconds locally; there is no mandatory multi-day wait between offline checks.
- The cached NumPy environment still emits the previously recorded matrix warnings. Nonfinite checks pass; previous direct float64 replay found no top-eight differences. Their root cause remains unproved. All500 cases are repeatedly exposed development data, not a hidden validation set. Baseline same-day timestamp-contract issues remain documented and unchanged.

## Next paid500 and next decision

**The full paid500 is unscheduled.** The earlier October8–9 planning window was conditional on a successful offline gate and a successful fixed-Luna/Terra answer comparison. This candidate failed the offline gate, so it supplies no evidence for scheduling that run. Credits and the founder's availability do not change this conclusion.

Stop variations of this complete-bundle score objective; do not run an unchanged paid package or tune weights until the development numbers look favorable. The next design must distinguish the facts needed to answer the question from passages that merely match its topic, and preserve their explicit qualifications. That contract and its validation are not yet implemented. Any semantic compression or a changed candidate-retrieval policy needs a separate frozen comparison, with the established whole-source control and all observed failures retained.

The required sequence remains: a source-reviewed positive offline candidate; a concrete, separately approved paired answer screen with Luna/Terra fixed and matched call/context budgets; only if actual answer gains survive controls, a full500 measurement and repeat under approved limits. No90% deadline, promotion, novel research claim or automatic Sarvam transition follows this work.

Full evidence: `POLICY.md`, `freeze.json`, `audit.json`, `verification.json`, `source-review.json`, `validation.json`, and archived `audit.py`. Large packets, runtime data and cached vectors remain local and hash-bound. Public reproduction instructions state the required historical inputs and original workspace layout.
