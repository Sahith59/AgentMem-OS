# Sarvam work, from the first plan to the current experiment

Prepared October 11, 2026, America/New_York. This is a plain-language account of the recorded Sarvam work, including failed runs and changes of direction. It is not a claim that we have audited every file, every language or every competitor. The current experiment is complete and its outcome is included below after replay and source review.

## What we are trying to achieve

Your long-term aim is a useful, credible contribution to the Indian AI ecosystem: reliable information that survives separate conversations, language changes, corrections and real business actions, with public evidence that others can inspect. Sarvam Startup Program acceptance gives us an opportunity and credits; it is not evidence that our design is already superior or that Sarvam has endorsed this research.

The intended language scope is **all languages supported by the relevant Sarvam services**. Hindi and Telugu are early investigation slices, with English controls. We have not validated all those services or languages. Public reproducibility is required; a convincing-looking demo or a single high percentage is insufficient.

Your later direction is the active one: first measure Sarvam plus ordinary tools and storage, then choose what to build from the remaining real failures. We are following that direction. The English AgentMem memory engine has not been used as the architecture in the larger native Sarvam studies. The English benchmark is still a separate unfinished track at its recorded 84.6%; no Sarvam percentage changes it.

## A few terms in everyday language

- An **input** is one question and its supplied information. A Hindi translation and its English original are related inputs, not independent real-world situations.
- An **episode** is one complete task attempt. It may call the model several times to read data, take an action and answer.
- An **API call** is one request to Sarvam. More calls do not mean more independently tested customers.
- A **control** is the ordinary version we compare against. A **guard** is a server-side permission check that can reject an unsafe database update.
- A **finalizer** is the extra model call that turns the task history into the final structured answer.
- **Correct fact**, **correct stored state**, and **successful task** differ. A model can truthfully say “I changed it” after a forbidden change, or truthfully report the old date after failing to perform an approved change.

## 1. September planning: a direction, not a completed product

The initial plan joined English closure, cross-language identity work, Sarvam integration and an eventual Indian-language memory evaluation. A September 8 revision proposed a solo 28-day text-memory pilot: remembering preferences/appointments after restart, accepting corrections, avoiding wrong-person merges and forgetting information. It proposed 40 families/400 pairedinstances and a demo/cookbook. Those were targets, not achieved deliverables or guaranteed dates.

English remained the priority under your earlier instructions. On October 7 you explicitly opened this separate Sarvam chat, allowing the tracks to proceed independently. Therefore we must not describe the oldSeptember calendar as fulfilled or imply that Sarvam waited because an API run inherently needs days. The substantial API comparison later took about 40 minutes; design, debugging and review were separate work.

Local sources: `plans/2026-09-08-english-sarvam/PLAN.md` in the outer memory, `FOUNDATION_BRIEF.md`, decisions S-001–S-005. These old proposals are retained, including their superseded assumptions.

## 2. October 7–8: you corrected the product scope and research order

We initially proposed Hindi/Telugu personal and small-business assistants. You correctly required wider eventual language coverage, deeper research before choosing a product, and a public benchmark from the outset. Those corrections superseded the earlier integration-first recommendation.

We examined Sarvam documentation/product material, competitor memory systems and evaluation work. The result was that generic language switching, customer context, temporal records, provenance and memory APIs already have substantial prior art. Sarvam also describes CRM/context/tool integrations. We cannot claim “Sarvam has no memory” or “we invented multilingual memory.” Public reports of tool/backend failures supplied hypotheses; they did not prove memory was their cause.

The documented language sets differed by endpoint, so one universal “all Sarvam languages” claim would be misleading. The October 8 capability registry records those dated distinctions; it is not a live certification of today's product catalog. We did not deploy or benchmark competitors, test Voice Agents/Cowork directly, or establish customer willingness to pay.

**Outcome:** useful research and clearer requirements; no product advantage or live score. Source: `research/2026-10-08/RESEARCH_REPORT.md`, its source ledgers/language registry, decisions S-006–S-009.

## 3. October 9: check whether simple storage already solves the example

An offline SQLite test handled delivery-update ordering and duplicate operations. That means the generic delivery example was not a new memory invention.

We also audited an older identity-matching test. Its 344 pairs reconstruct, but labels sometimes required missing identity context or mixed identity with sentence similarity. A needed fixture was ignored/untracked, so a clean checkout could not reproduce it. The old 86.51% precision/87.20% recall figures stay historical draft-label metrics, not Sarvam memory accuracy. A separate earlier code inspection found Latin-only matching in a forgetting path; that remains a static issue, not a fixed multilingual deletion system.

**Outcome:** evaluation weaknesses identified before relying on their scores. Source: `experiments/2026-10-09-discovery/RESULTS.md`; decisions S-010–S-012. No paid calls in this offline audit.

## 4. First runnable comparison: preserve original evidence and meter everything

We prepared 38 language renderings from 12 templates/9 families, two repeats and a maximum 228-call pilot. One arm received full original history; another added model-extracted facts in SQLite while retaining the same originals. This prevented an unfairly weak fact-only control. Extraction added compute, so this was not an equal-total-cost architecture comparison.

We added frozen requests, separate expected answers, parser checks, real isolated stores, complete call accounting and no automatic retries. 46 offline tests and clean-export verification passed. The cases were assistant reviewed; independent native-language review was absent.

**Outcome:** a runnable research setup, not proven answer improvement. [Preparation/public record](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-discovery-2026-10-10/REPORT.md); PR 44 covers initial preparation.

## 5. First live pilot failed before an answer could be measured

Only 1 of 228 plannedcalls ran. Base 105B used 4095 of 4096 completiontokens for reasoning and returned no final content. There was no accepted extraction or answer. We stopped and preserved the failure; accuracy was undefined, not 0%.

A separately approved 1-call check changed only reasoning effort from low to null. It returned usable JSON and preserved the pickup instruction, but produced six copied metadata items plus one copied sentence. It was verbose extraction, not an answer-score or useful-compaction breakthrough. An offline reopen check verified those returned records persisted; no live answer comparison occurred.

**Outcome:** one configuration symptom improved on one example; reliability and memory benefit unproved. [Failed pilot](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-discovery-2026-10-10/REPORT.md); [one-call diagnostic](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-discovery-2026-10-10/thinking-off-probe/REPORT.md); PR 45/46.

## 6. The smaller comparison exposed another problem in our setup

A new 72-call comparison stopped after 1 extraction. The model placed a full observation timestamp in a date-only field. The parser correctly rejected it. No answer calls ran. This exact request had succeeded in the previous probe, showing why one success at temperature 0 was not enough.

The run also exposed a design weakness: an extraction failure prevented the independent full-history answer from being tested. We then versioned the setup so rejected generated facts could be discarded while original evidence and the independent control remained available. We did not silently trim dates or overwrite old failures.

**Outcome:** safer evidence admission and a better experiment structure, not a measured memory gain. [Stopped comparison](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-small-comparison-2026-10-11/live-run-001/REPORT.md); [admission repair](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-admission-v3-2026-10-11/REPORT.md); PR 47–49.

## 7. Five approved small runs: the apparent gain did not repeat

These used 12 development renderings and two repeats. They were too small and correlated to define broad language accuracy.

| Run | What changed | Actual calls | Result |
|---|---|---:|---|
|001|Strict-schema extraction|2|Stopped on unsupported schema keyword; one control answer had completed.|
|002|Removed only that provider-incompatible keyword; local validation remained|26|Stopped on a truncated extraction.|
|003|JSON-object transport with the schema described in the prompt|72|Memory 24/24; original-history control 23/24. Small positive signal.|
|004|Exact repeat of 003|72|Memory 23/24; control 24/24. The advantage reversed.|
|005|Reject generated validity dates without a literal date in the cited source|72|Both 24/24. Four additional problematic date augmentations rejected.|

Runs 003+004 pooled to **47/48 for each arm**. We therefore did not establish an accuracy gain. Run 005 improved the secondary answer-plus-required-source count 23/24 versus 17/24, but that signal was unreplicated and was not proof every claim/citation was supported.

A concrete semantic failure remained: a conditional Hindi callback statement became an affirmative repair approval. A date appearing in the source does not prove the condition or approval is true. The date check could not solve this. In run 005, all seven source-only fallback answers were correct; a complex augmentation was not always needed.

**Outcome:** a useful rejection rule and reproducible failure evidence; **no replicated memory-answer advantage**.244 calls total, zero retries, one unknown-usage provider failure. [Full report](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-five-run-campaign-2026-10-11/REPORT.md); PR 50.

## 8. Your evidence-first reset: test native Sarvam before adding architecture

You explicitly asked us to stop treating the English memory system as the starting solution, use native Sarvam APIs and ordinary storage, and investigate on a larger sample under an INR 6000 research ceiling. That superseded the proposed next memory-component implementation.

The larger study used base 105B and 105B Conversations, full original context and ordinary SQLite tools. It completed **2800 main episodes plus 264 diagnostic episodes**, plus excluded setup, in **4928 API calls**. There were five original rate-limit errors; our runner initially lacked shared pacing. We added pacing, used your reported 120 RPMlimit to set 90 RPM, and attempted only previously unattempted jobs. The five failed jobs stayed failed; no automatic retry hid them.

| Model/setup | Main discovery strict success | Reserved confirmation strict success |
|---|---:|---:|
|105B/full history|312/490|127/210|
|105B/SQLite|298/490|126/210|
|Conversations/full history|317/490|127/210|
|Conversations/SQLite|300/490|126/210|

“Strict” combined answer shape, factual answer, correct stored state and operation requirements. These roughly 60–65% results are **not a decline from the earlier tiny 24/24 tests**: different inputs, formats, difficulty and scoring were used. They are also not general Sarvam model accuracy.

We found a dataset-design weakness: 700 rendered rows contained 637 distinct visible inputs, and six appeared in both splits. A separate outcome-independent view removed duplicates/overlap: confirmation 117/186 full context and 114/186 SQLite for either model. We disclosed the flaw instead of advertising an independent held-out benchmark.

Important action result: Conversations/SQLite kept correct state in 140/140 main action episodes. Base 105B made 17 forbidden update attempts across main episodes. Correct database state did not imply every answer was correct.

A selected reasoning diagnostic held the 8192-token cap constant. Base/full history improved 14→20/33, with 10 gains and 4 losses; base/SQLite stayed 15/33 and Conversations had no net gain. One arithmetic answer improved 1071→898. Old-date and other arithmetic mistakes remained. Of 31 failed base-low outputs reviewed, 20 primarily concerned output contracts and 11 were substantive mismatches across four underlying situations. More reasoning was not a universal fix or a memory contribution.

**Outcome:** a much stronger failure map, plus explicit flaws in our test design; no new memory architecture or novelty established. [Study and complete limitations](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-native-study-2026-10-11/REPORT.md); PR 51.

## 9. Strict output formatting helped appearance but harmed action execution

We next tested strict JSON together with tools. Six setup responses had valid JSON, but only 2/6 passed the full contract. Neither authorized action was executed. A separate matched diagnostic showed both ordinary controls executing the action while both schema arms skipped it. That was one Telugu action scenario across two models, not a universal claim about Sarvam's schema support.

We closed the planned 288-episode expansion without running it. Valid JSON was clearly insufficient if the promised action did not happen.

**Outcome:** candidate rejected; 20 new API calls. [Report](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-structured-control-2026-10-11/REPORT.md); PR 52.

## 10. Separating actions from answers exposed our own API integration bug

We allowed native tool actions first, then tried final reporting separately. Our request omitted tool definitions while retaining native tool messages, causing six HTTP 400 errors. That was our request-construction bug, not model answer inaccuracy.

A four-call repair retained definitions and prevented new tools. Conversations answered correctly 2/2; base returned empty content 2/2 below its token cap. That fixed request admission but did not solve reporting. We also preserved a real pending-approval update by base in our deliberately unguarded test backend. Some other “strict failures” were awkward status labels or an artificial demand to attempt a rejected operation, not dangerous behavior.

**Outcome:** original gate remained failed; 26 calls total in this phase. [Report](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-separated-finalization-2026-10-11/REPORT.md); PR 53.

## 11. Complete transcript reporting and ordinary server checks

We represented the complete original history and actual tool receipts as JSON data, with no executable tools in the reporting request. Both models then answered correctly in both output formats: **4/4 new calls** on two saved successful histories. This was a limited compatibility success, not proof the finalizer would work on every case.

Separately, we built ordinary SQLite permission checks. In 44 offline database variants, the guard blocked 15 deliberately unapproved writes and preserved three legitimate first writes. Tests covered duplicate requests, conflicting request keys, scope, stale versions, concurrent calls and rollback. This is standard application authorization, not a novel memory algorithm or deployed production authentication.

**Outcome:** two bounded engineering successes worth integrating. [Report](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-transcript-approval-2026-10-11/REPORT.md); PR 54.

## 12. Integrated setup: correct state, imperfect reporting

Six situations × two models × two approval settings gave 24 live episodes/61 calls. All 24 current dates/states were correct and all four approved new writes succeeded. Raw reports were 21/24 correct; strict tasks 19/24 because of three status mismatches and two unnecessary closed-booking attempts. The frozen gate failed and stayed failed.

A separate offline application layer derived action_status from actual receipts and intent. It changed only three labels, giving 24/24 application reports and 22/24 strict tasks on those saved histories. It did not fix dates or change model outputs.

A fresh version then passed 12/12 application reports/states and 2/2 approved writes, using 30 calls. Raw reports were 8/12; four statuses were corrected; one unnecessary safely blocked attempt remained, so strict success 11/12. This passed the newly defined setup gate, not the old failed gate retroactively.

**Outcome:** verified ordinary integration with a clear remaining limitation. [First setup](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-integrated-ordinary-2026-10-11/REPORT.md); [new version](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-ordinary-source-2026-10-11/REPORT.md); PR 55/56.

## 13. The 288-episode source challenge: what we knew before today's run

Seventy-two English/Hindi/Telugu inputs (24 related scenarios) × two models × two approval settings, 723 API calls, about 40 minutes of API execution. No API/backend errors or retries. Full source, current state and action arguments were provided; this was not a retrieval or long-term-memory test.

| Model | Safe tasks, approval off | Safe tasks, approval on |
|---|---:|---:|
|105B|66/72|57/72|
|Conversations|67/72|69/72|

Conversations/guarded achieved 95.8% on these selected synthetic tasks. That is not our English benchmark or all-language reliability. Strict Conversations tasks were 67/72 in both arms because unnecessary attempted actions were also penalized.

What we found:

- Four unauthorized Conversations writes when enforcement was off; none with enforcement. Three live guarded attempts were blocked. 23 of 24 required legitimate writes completed; base omitted one Hindi update.
- 18 empty base final outputs: 16 exhausted 8192 tokens; two stopped with null content below the cap. Fifteen had already stated the correct date in native prose; three had already performed the correct approved update.
- Six valid answer/clarification mismatches: a copied question, one unsupported current-date substitution, a missing clarification flag, two over-abstentions and an omitted known date. These are not six invented dates.
- Receipt-owned status corrected 59 labels without changing dates/clarification. Complete desired reports were 201 raw→259 application. This is contract engineering, not 59 new factual answers.
- All 89 flagged episodes across 51 rendered cases received assistant review. All 288 request/state/charge chains reproduced. 143 focused tests and clean-export checks passed; 3,083 archived files verified. Five commits merged in PR 57.

Read-only pairs sometimes differed although the guard was inactive: 10 base pairs and 3 Conversations pairs had only one correct answer. Do not credit those variations to approval enforcement. A separate offline replay with exactly the same tool calls isolated the guard mechanism; it did not invent new model answers.

**Outcome:** concrete evidence that the final reporting stage loses useful results, and ordinary permission checks protect state. [Full comparison](https://github.com/Sahith59/AgentMem-OS/blob/main/benchmarks/audits/sarvam-source-comparison-2026-10-11/REPORT.md); PR 57.

## 14. Today's completed experiment: one sentence, fixed models

We tested all 144 saved guarded histories with a fresh control and fresh candidate: 288 completed finalization calls. Candidate changes only “Report the original service question...” to “Answer the original service question using the evidence in the transcript below; do not repeat the question.” All remaining source/history, model profiles, schema, temperature and 8192-token cap are unchanged. Both read-only and action histories are included.

This tested whether wording contributes to the observed failure, without assuming beforehand that it would improve accuracy. No tool action was rerun and no wrong output was corrected using an expected answer. 154 focused tests and a complete 288 synthetic-response simulation passed before the live run. A synthetic simulation is a software check, not a score.

The earlier ₹300 phase cap was provisional. The full-context reservation calculation is ₹619.3152, so the phase ceiling is ₹625 inside the existing total ₹6000 authority. Actual usage settled below this conservative bound; every new call has known usage.

**Completed outcome:** base 105B improved from **59/72 to 72/72** factual answers (13 paired gains, zero losses); Conversations improved from **70/72 to 71/72** (two gains, one loss). Base empty answers fell from ten to zero. The new instruction used about 42% fewer base completion tokens. No new API failures or retries occurred. Independent replay and accounting agree.

The predefined factual screen passes after reviewing all 16 incorrect outputs and 55 changed raw pairs. This is our clearest current reporting improvement with fixed models, but **not a new memory advantage or a general 100% system score**. The candidate still unnecessarily abstains in one Conversations case. Raw execution-status guesses remain wrong in 14 base and 18 Conversations answers; the existing application layer must derive status from receipts. Base's old missed Hindi action remains missed, even when the new answer truthfully reports the old date. No actions were rerun.

**Decision:** prepare fresh end-to-end confirmation with the unchanged candidate. Do not promote it from a saved-history screen alone. The run used INR 38.453490 known estimated usage (INR 48.066950 retained with headroom). See REPORT.md, FAILURE_REVIEW.md, gate.json and NEXT_PHASE.md for the exact evidence and proposed next work.

## What is solid, what is unresolved

**Solid within the tested scope:** reproducible request/receipt/accounting tools; ordinary persistence and source preservation; permission/idempotency/transaction checks; explicit failures retained; application-owned execution status; evidence locating reporting losses. Passing software tests proves those checks work on their test cases, not that the models are always accurate.

**Unresolved:** a repeatable answer gain from a new memory architecture; the exact cause and reliable repair of all empty final outputs; broader natural-language source interpretation; independently reviewed Hindi/Telugu data; genuinely new workflow holdouts; other Sarvam languages/routes; real Voice Agents/voice behavior; production identity/approval provisioning; customer adoption; a publication-quality novelty claim.

Public code and reproducible research artifacts already exist. A trusted industry benchmark needs stronger independent data/review and broader coverage; we should not rename this development suite to imply that work is complete.

## What we should acknowledge about our own process

We initially moved toward memory augmentation before proving the remaining need; your evidence-first correction improved the research direction. Tiny checks were necessary for API compatibility but insufficient for general claims. Some costs in time came from our own request construction, strict output conventions and duplicate-input design. We have preserved those errors and corrected their interpretations. A sequence of green unit tests is not a sequence of better model results.

The way to avoid another vague loop is to give each experiment one question, a bounded manifest and a stop rule. Stop a failed candidate; publish negative results. Do not add complexity or stronger models merely to obtain a higher number. Establish the ordinary integration first, then identify a real persistent-information limit and test a memory intervention against strong controls on new data.

## Spending: do not confuse the histories

Before today's new calls, the **native evidence-first campaign** had 5792 API calls and known usage estimate ₹131.301195, with ₹196.384594 retained including headroom and old unknown costs. This is the active ₹6000 ledger.

Earlier, separately authorized pilot/memory experiments used 247 calls: three initial single calls plus 244 five-campaign calls. Their known estimates at the recorded 15/60 rates total ₹4.40049, with one additional unknown-usage error. They are not secretly included in 5792. Thus the documented series before today's run has 6039 calls and ₹135.701685 known estimated usage across both eras, with unknown usage remaining unpriced. This is not lifetime project spending, an invoice or verified account balance. Do not add old worst-case reservations to actual spend. After today's 288 calls, native totals are 6,080 calls / INR 169.754685 known / INR 244.451544 retained. Across both documented eras, totals are 6,327 calls / INR 174.155175 known estimates, with older unknown usage still unpriced. These totals are calculated from receipts, not the projected budget.

## Audit trail and coverage

The history is assembled from the September plan, Sarvam decisions S-001–S-063, research report, original experiment reports, completion/verification receipts and published PR 44–57 artifacts. Historical quantities remain on their original tasks and scales. This turn replays the prior 288 episode comparison and its attribution, verifies founder preservation, and tests then audits the new reporter experiment. It does not rerun every historical API answer or replace independent language review. Original failures, labels, code snapshots and obsolete plans remain available.
