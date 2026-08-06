
## 2026-08-06 — Run a flagship "proof" test N times before believing it (Critics, Stage 1 R4)
Evidence: the round-3 N1 proof test passed for the team and was written into the build log as
"Proof: 3-OS-process test — mention_count exactly 60". On re-run it failed 6/10 in isolation and
3/5 in the full file. A concurrency test that passes once proves nothing; it must be looped.
**Enforced check:** any test offered as proof of a CONCURRENCY fix must be run >=10x and the pass
rate reported. A single green run is not evidence.

## 2026-08-06 — Optimistic CAS across two transactions is not atomic (Critics, Stage 1 R4)
Evidence: read via autocommit SELECT + write via a new BEGIN leaves a lost-update window; measured
1-3% retry exhaustion on a hot row at only 3-8 writers. **Enforced check:** a compare-and-set must
occur inside ONE write transaction (BEGIN IMMEDIATE) or be expressed as a single SQL statement.

## 2026-08-06 — Coverage percentage without characterization hides the fired branches (Critics, R4)
Evidence: third repeat (R2 M1, R3 N6, now R4-2). "98%" was accurate; the 6 uncovered lines were the
error paths that actually fire. **Enforced check:** every uncovered line must be named and justified
in the build log, and any branch demonstrated to fire under test must be covered.

## 2026-08-06 — A broad `except` that swallows the cause misdiagnoses ops failures (Critics, R4)
Evidence: read-only DB reported as "lost 8 version races", __cause__ and __context__ both None.
**Enforced check:** narrow the exception to the specific recoverable condition, and always
`raise ... from e` when converting an exception at the end of a retry loop.

## 2026-08-06 — A store must never call expire_all() on a session it does not own (Critics, Stage 1 R5)
Evidence: `_reaffirm`'s `session.expire_all()` silently discarded a caller's unflushed edits
(turn.content, fact.extraction_model) in a caller-owned batch on the production session config
(autoflush=False), and the store returned success. A targeted `session.expire(obj)` was measured
to give the identical fresh re-read with no collateral damage. **Enforced check:** in any code
that accepts a caller-supplied session, session-WIDE operations (expire_all, rollback, commit,
close, flush-of-everything) are forbidden on the not-owned path; only object-targeted operations
are allowed. Any `db=`/caller-session API needs at least one test that stages caller state
BEFORE the call and asserts it survives.

## 2026-08-06 — A structural removal must delete the mechanism's VOCABULARY too (Critics, R5)
Evidence: 4th repeat of the stale-doc class (N6-N8, R4-5, now R5-2/R5-4). After the retry/CAS
removal, the rewritten function's own docstring still promised "version-guarded", "retry up to
_REAFFIRM_RETRIES" (constant deleted), and a test-file comment still credited "NullPool" two
rounds after the QueuePool switch. **Enforced check:** when a mechanism is removed, grep the WHOLE
repo for every identifier and every noun that named it (constant, class, pool name, "retry",
"version") and show the grep as evidence in the build log; a rewritten function's docstring is
part of the diff, not documentation debt.

## 2026-08-06 — The evidence doc must record the round that falsified it (Critics, R5)
Evidence: the build log still presents the round-3 optimistic-CAS resolution and its falsified
proof ("mention_count exactly 60") as current, with no round-4 record and stale "98% coverage"
artifacts. **Enforced check:** a fix round is not complete until the log entry that the round
falsified is annotated in place (struck/marked superseded) and the new round's findings +
resolutions + freshly measured artifacts are recorded — the log is the founder's evidence, and a
stale row in it is an over-claim.

## 2026-08-06 — Mutation-test the fix, not just the feature (Critics, Stage 1 R6)
Evidence: two fixes shipped in R5. Swapping `expire(fact)`→`expire_all()` flipped its regression
test red (real tripwire), but deleting the `coalesce(mention_count,1)` left the suite 51/51 GREEN
while a legacy NULL row silently returned to mention_count=None forever. A green suite proves the
fix is present, never that it is DEFENDED. **Enforced check:** for every fix, mutate the fixed line
away and re-run; if the suite stays green, the fix has no tripwire and needs one before the gate.

## 2026-08-06 — Model-level column constraints never reach an existing table (Critics, R6)
Evidence: `mention_count` gained `nullable=False, server_default=1`, but `create_all` skips
existing tables and `_migrate_semantic_tier` verifies indexes/constraints only — the founder's real
DB still reads `mention_count INTEGER` and the verifier still reports "verified". The defense that
actually holds on live data was the SQL-side `coalesce`, not the model change. **Enforced check:**
when a column definition changes, state explicitly which DBs it reaches (fresh only vs migrated),
verify the claim against a real existing DB read-only, and keep a runtime defense for the old shape.

## 2026-08-06 — Say PASS plainly when it is earned (Critics, R6)
Evidence: rounds 1-5 all blocked; round 6 found no blocker and no major. Manufacturing a finding to
justify the round would have cost the team a day and taught them that the gate never opens.
**Enforced check:** a round that finds only notes reports PASS-WITH-NOTES with the notes named and
the verified fixes stated plainly — the verdict follows the evidence, not the round number.

## 2026-08-06 — "0 rejected" is not quality evidence when the validator is nearly inert (Critics, Stage 2 R1)
Evidence: the G2 smoke's headline "42 facts created, 0 rejected" was offered as extraction-quality
evidence. Measured: 9 of the 42 (21%) are assistant knowledge ("Bat Wings will be available during
Mickey's Halloween Party"), and a pure tool-noise session produced 4 junk facts with 0 rejections.
The validator only checks text length, an enum the JSON schema already enforces, calendar-parseability,
and one 6x5 phrase regex — it cannot reject content junk at all. **Enforced check:** a rejection rate
may never be cited as evidence of extraction quality without a stated denominator of what the validator
is CAPABLE of rejecting, plus a run on a session designed to produce the failure the validator claims
to catch. Zero rejections on a happy corpus is a null result, not a pass.

## 2026-08-06 — Reading a partial printout and recording a defect is the same error as reading a lookalike (Critics, Stage 2 R1)
Evidence: the build log records "'in July' emitted as 2023/07/01 point rather than month interval"
as a known extractor wart. The store actually holds t_occurred=2023/07/01 AND t_occurred_end=2023/07/31
— a correct interval. The smoke's print statement shows only `f.t_occurred`, so the author inferred a
defect from an incomplete view and put it in the permanent record; anyone acting on it would have
"fixed" working code. This is R2 B5 (EXPLAINing a re-declared lookalike) in a new costume: the
conclusion was drawn from a rendering of the artifact, not the artifact. **Enforced check:** before a
defect goes into the build log, dump the STORED ROW (all columns), not the harness's printout of it.
Any smoke that prints a value with a companion column (t_occurred/t_occurred_end, start/end, id/hash)
must print both or print neither.

## 2026-08-06 — A citation/integrity flag that returns True for an EMPTY set is a false clean (Critics, Stage 2 R1)
Evidence: `provenance()` computes `citations_intact = set(found) == set(cited)`; for a fact with zero
cited turns both sides are empty, so an uncited fact reports intact=True — and the flagship smoke
printed exactly that for "The user rode the Mako rollercoaster." 2/42 flagship facts and 11/24
adversarial facts cite nothing. Separately, 19.8% of citation edges point at turns that do not contain
the fact, also reported intact. **Enforced check:** any boolean named *_intact / *_ok / *_verified must
be false (or a distinct third state) when there is nothing to verify; write the empty-input test first.
Provenance claims additionally need a support test — does the cited turn's text actually contain the
fact's distinctive content — not just a foreign-key existence check.

## 2026-08-06 — A session-global feature flag makes a per-fact rule fire on unrelated evidence (Critics, Stage 2 R1)
Evidence: `source_had_digits` is computed once over ALL user turns concatenated, then applied to every
candidate fact. On LongMemEval every user turn carries a "[2023/11/04 (Sat) 03:27]" prefix, so the flag
is armed on 19,143/19,195 sessions (99.73%) — the "vague quantifier where source had numbers" rule
becomes an unconditional ban. Measured: a user who genuinely said "several times this month, I honestly
lost count" had that true fact destroyed because a different turn mentioned "18 dollars". **Enforced
check:** a validator predicate that is a function of (fact, source) must be computed against the SOURCE
SPAN the fact came from, never a session-wide aggregate; and any near-100% trigger rate on the target
corpus means the guard is not measuring what its name says. Measure the arming rate on the real corpus
before shipping the guard.

## 2026-08-06 — A prior stage's "next stage will handle it" note must be carried into the next stage's record (Critics, Stage 2 R1)
Evidence: Stage-1 F7 recorded "Store DELIBERATELY permits future t_occurred; Stage 2 extractor gets
explicit planned-event handling. Founder may veto." Stage 2 shipped with no planned-event handling
(measured: "The user plans to run the Boston Marathon in April 2024" stored as fact_type=event,
t_occurred=2024/04, indistinguishable from an occurred event) and the Stage-2 record does not mention
the omission — so a founder-flagged open decision silently disappeared. **Enforced check:** every
"Stage N+1 will do X" written into a stage record becomes a mandatory line item in Stage N+1's record,
answered DONE / DEFERRED-with-reason / DROPPED-with-founder-approval. The critic's first pass on any
stage greps prior records for that stage's number.

## 2026-08-06 — A gate test whose fixture cannot produce the failure is a rubber stamp (Critics, Stage 2 R2)
Evidence: `test_assistant_knowledge_rejected_by_support_gate` asserts "The user's build succeeded in 4213
milliseconds." is rejected. Run against the tool-noise session that MOTIVATED the finding — where the user
says "run the build" — the identical fact is ACCEPTED (4 tokens => threshold 1, "build" is the shared
token). The test passes only because its fixture session contains no user turn with that word. **Enforced
check:** a test written for an adversarial finding must run against the adversarial INPUT that produced the
finding (or a faithful reduction of it), and the reviewer must re-run the original probe — a green unit test
on a hand-made fixture is never evidence that a content gate works on real data.

## 2026-08-06 — Truncating an evidence list by ID order throws away the best evidence (Critics, Stage 2 R2)
Evidence: `cited[:8]` keeps the 8 lowest turn ids. Measured on the flagship smoke: a fact whose true source
turn shares 7 tokens was cited to turns sharing 3, because the source turn came later in the session; 13/22
facts hit the cap and 2 lost their strongest evidence outright — while `citations_intact` still reported
True. **Enforced check:** any capped evidence/citation list must be RANKED by support strength before the
cap, and the cap must be recorded in the provenance output ("8 of 11 supporting turns") so a truncated
citation set can never be read as a complete one.

## 2026-08-06 — A char-based input cap does not bound a token-based context window (Critics, Stage 2 R2)
Evidence: TRANSCRIPT_CAP=36000 chars with Ollama num_ctx=10240. English measured 8,120 tokens (79% of the
ceiling); the same 36,000 chars of Devanagari measured prompt_eval_count=10240 — silently clamped by the
server — while the engine reported truncated_chars=0 and logged nothing. The one path this breaks first is
the cross-lingual path the roadmap depends on. **Enforced check:** when an LLM boundary has a token limit,
measure the actual prompt token count on the WORST-CASE script/content the product claims to support, and
read the server's own prompt_eval_count back to detect clamping — a character budget is not a token budget.

## 2026-08-06 — Fixing "read only turn 0" by "read every turn" hands the field to untrusted content (Critics, Stage 2 R2)
Evidence: the R1 fix made `_session_date` scan ALL turns for "Session dated <date>". A user turn containing
"Please note: Session dated 2099/01/01." now sets t_mentioned for every fact in the session AND disables the
future-dated-event guard (measured end to end). The stamp is role-blind, unauthenticated, and never
sanity-checked against turn.created_at. **Enforced check:** when broadening where a trusted value may be
read from, state which ROLES may supply it and bound it by an independent source (row timestamps); any
value parsed out of user-authored text is untrusted input, and widening the search is a widening of the
attack surface, not just a bug fix.

## 2026-08-06 — A precision fix must be measured for RECALL on the same corpus before it ships (Critics, Stage 2 R3)
Evidence: R2 blocked because tool output ("4213 ms") rode into the store. The fix — every number in a fact
must appear in user evidence — does reject 4213. It also rejected 15 of 18 true user facts on plan-heavy
sessions, 6 of 48 on 10 random real LongMemEval sessions (~5 of them true), and emitted the FALSE reason
"tool/assistant output is not a user fact" for facts the user had typed verbatim. The team measured the
false-accept it was fixing and never measured the false-reject it created; the previous round's own
false-rejection metric (2.1% morphology) was not re-run. **Enforced check:** when a gate is tightened to
kill a measured false-accept, re-run the SAME corpus probe that produced the original number and report
both directions (what it now blocks AND what it now loses) in the same table. A gate reported only by the
attack it stops is a half-measurement.

## 2026-08-06 — A validator that reads surface form is a lottery, not a rule (Critics, Stage 2 R3)
Evidence: the numbers gate accepted "three times" and rejected "3 times" — while the extraction prompt
ORDERS the numeral ("If the user did something N times, the fact states N"). It accepted "$1,200" and
rejected "1200 dollars"; it accepted "October 15th, 2023" only because `\b\d[\d,.]*\b` cannot see the
ordinal "15th", and rejected the identical claim written "2023/10/15". Substring matching also accepted
"ran 42 marathons" against a turn saying "42195 metres". The flagship smoke passed by that ordinal
accident. **Enforced check:** before a string/regex rule becomes a gate that DESTROYS data, enumerate the
surface forms of the thing it matches (spelled-out numerals, ordinals, currency/thousands separators,
percentages, ISO vs prose dates) and test each; and never let a gate's prompt and its validator disagree
about the required output format.

## 2026-08-06 — Narrowing an attack surface is not closing it, and the comment must not say otherwise (Critics, Stage 2 R3)
Evidence: R2 found user text could hijack the session date. The fix limited the stamp scan to the first 3
turns and the comment says the hijack is fixed — but the scan stayed ROLE-BLIND, so a stamp in the user's
first message still sets the date for the whole session (measured: 2099/01/01). The repo's test planted
the hijack at turn 4, outside the new window. Separately, a corpus scan showed the real fix was free: all
19,195 sessions carry the stamp on a SYSTEM line at index 0, so a role check would have cost nothing.
**Enforced check:** when a fix narrows rather than eliminates a class, the comment/docstring must say
which cases remain open, and the test must plant the attack in the WORST remaining position, not a
position the fix already excludes.

## 2026-08-06 — Changing a row's TYPE changes which dedup key it lands in (Critics, Stage 2 R3)
Evidence: Stage-1 F3 (a blocker) put t_occurred into the hash for events so two occurrences of the same
sentence stay two rows. Stage 2's F7 fix retypes future-dated events to "state" — and states hash on text
alone. Measured: two plans with the same sentence on 2024/04/15 and 2025/04/21 collapsed to one row
holding only the 2024 date; the identical input with no retype produced two rows. A fix in one file
silently reverted a blocker-grade guarantee in another. **Enforced check:** any code that mutates a
record's type/kind before storage must be tested against the store's dedup identity for BOTH types, and
the reviewer must re-run the original blocker's probe for the type being moved into.

## 2026-08-06 — An assertion that matches the mutant is not a tripwire (Critics, Stage 2 R3)
Evidence: `test_truncation_loud_and_persisted` asserts `row.rejected_count == 0` on a session with zero
rejections. Deleting the persistence entirely (writing a literal 0) leaves the suite green — the claimed
fix "rejected_count PERSISTED" had no tripwire while appearing covered. Same round: the session-year
exemption and the vague guard's date-strip half were also mutation-green. **Enforced check:** a
persistence assertion must use a value the mutant cannot produce — assert a NON-DEFAULT count from a
fixture that actually rejects — and every claimed fix goes through the mutation sweep, not just the ones
a previous round named.

## 2026-08-06 — Substring matching is the surface-form lottery wearing a rule's clothes (Critics, Stage 2 R4)

R3 blocked a numbers gate for judging surface form ("3 times" rejected, "three times" accepted). The fix
added a word-numeral map — matched with `if word in user_low`. Substrings, not words. Whole-corpus
measurement: 18.9% of user turns and 51.5% of sessions contain a carrier word that manufactures a number
the user never said ("content"/"attended"/"often" -> 10; "someone"/"phone"/"money" -> 1; "network" -> 2;
"height" -> 8), and a tool-output fact rode it into the store end-to-end. The rule was still reading
surface form; only the surface changed. **Check: when a fix replaces a string comparison with a lookup,
verify the lookup's MATCH SEMANTICS (word boundary? case? token? value?) and measure the false-match rate
on the real corpus, not on the fixture.** The strict form here (`\bword\b`) was free.

## 2026-08-06 — An exemption written for the true case licenses the false case (Critics, Stage 2 R4)

The "session-year exemption" was removed as exploitable and replaced by "exclude the fact's own
t_occurred digits" — which reinstates it per fact for every fact the model dates, plus every number equal
to the day or month. A tool number ("2024 warnings", "took 15 seconds") walks in whenever the model
stamps the fact with the session date, and whether it walks in depends on zero-padding ("5" vs "05").
The exclusion was genuinely load-bearing (2 of 11 real numeric facts needed it), so deleting it re-breaks
recall. **Check: an exemption must be scoped to the EVIDENCE that justifies it — here, the digits of the
date literal actually present in the fact text — never to a set of VALUES that any other number can
match. And a comment calling model output "engine-derived" is a false-provenance claim; trace where the
field is actually filled.**

## 2026-08-06 — A validator built out of an English regex silently deletes every other language (Critics, Stage 2 R4)

`_TOKEN = [a-z0-9]{3,}` makes the support gate return the empty token set for Devanagari, Cyrillic and
CJK, so every fact from a non-Latin-script session is rejected — with the message "no supporting USER
turn — assistant/system knowledge is not a user fact", i.e. the audit trail calls the user's own words
assistant knowledge. The project's contract names cross-lingual canonical facts a target distinctive.
**Check: for any gate built on a character-class regex, run one real non-Latin session through it before
claiming the gate works, and put the limitation in the NOT-BUILT list the moment it is known.**

## 2026-08-06 — A one-sided guard reports as if it were two-sided (Critics, Stage 2 R4)

The numbers gate inspects digits in the fact text only. A fabricated word count ("seventeen times"
against a user who said "twice") is never checked, and the flagship fact of the whole aggregation thesis
("rode ... three times in a row") passes the guard without ever being tested by it — while the docstring
claims "every NUMBER in the fact must appear in user-cited content". **Check: state the guard's domain in
the same sentence as its promise, and test the direction you did NOT implement so the gap is visible.**

## 2026-08-06 — A regex written to EXEMPT a pattern can MANUFACTURE data (Critics, Stage 2 R5)

`_DATE_LITERAL_RE` was added to exempt date digits from the numbers gate. On "in February 2023" its
`\d{1,2}` matched "20" and left "23" behind — a number present in no source, which the gate then demanded
the user had stated. Two true facts (one count-bearing) died on the same real sample where the previous
round had zero false rejections; the rejection rate doubled 4.7% -> 9.3%.
Rule: any regex that DELETES text before a check must be tested on what it LEAVES, not on what it
matches. Enumerate the shape family (all 12 month names, day-first, day-suffix, ranges, bare years) and
assert the residue is empty or only genuine values. Better: never leave a residue — parse the date
expression and remove it as a unit, or don't remove anything.
Evidence: scratchpad/s2r5/monthyear.py, repro23.py; review log R5-B1.

## 2026-08-06 — A number gate that needs a word boundary cannot see "10s" (Critics, Stage 2 R5)

`\b\d[\d,.]*\b` matches nothing in 10s, 4213ms, 3x, 16GB, $20K — the boundary after the digits never
exists. Given the same tool output, the real model wrote both "4213 milliseconds" (rejected) and "16GB"
(accepted and stored). Three rounds of fixes to this gate all assumed numbers arrive space-separated.
Rule: a validator over model-generated text must be tested against the model's OWN formatting variety,
not the fixture author's. Run the real extractor on the adversarial session and grep its output for the
forms your regex cannot see (\d+[A-Za-z], \d+[.,]\d+, word numerals, unicode digits) before claiming the
class is closed.
Evidence: scratchpad/s2r5/glue_e2e.py (real llama3.1 run); review log R5-B2.

## 2026-08-06 — Fixture transcripts are cleaner than production transcripts (Critics, Stage 2 R5)

The numbers gate treats raw user text as ground truth for values. In the unit fixtures a user turn is
"I rode rollercoasters three times." In the real corpus EVERY user line is
"User: [2023/05/20 (Sat) 14:05] ..." — 20,452/20,452 measured — so every session silently pre-licenses
~5 numeric values (year, month, day, hour, minute) that no human ever asserted, and tool numbers equal to
any of them ride in.
Rule: before trusting an evidence-derived allowlist, print the allowlist for a REAL session, not a
fixture one. Ask "what does this set contain that the user never said?"
Evidence: scratchpad/s2r5/userdate.py, leakvol.py; review log R5-B3.

## 2026-08-06 — Making a gate unicode-aware is not making it work (Critics, Stage 2 R5)

Switching `[a-z0-9]{3,}` to `[^\W_]{3,}` stopped Devanagari sessions auto-rejecting, and the test that
pinned it only asserted acceptance. Python's \w excludes combining marks, so a whole Hindi sentence
yields ONE 3-char fragment; with need=1 an unrelated assistant-knowledge fact sharing that fragment is
ACCEPTED. The failure mode flipped from false-reject to false-accept and the record read as "fixed".
Rule: a support/quality gate needs a NEGATIVE test in every language path it claims to serve — feed it
junk in that script and require rejection. An accept-only test cannot tell "works" from "inert".
Evidence: scratchpad/s2r5/hindi5.py; review log R5-M2.

## 2026-08-06 — A critic's own harness must force the scratch DB path (Critics, Stage 2 R5)

My mutation sweep injected a mutated `agentmem_os.db.engine` into subprocesses BEFORE pytest could load
tests/conftest.py, so import-time `init_db()` ran against the resolved production path
(/Volumes/Sahith_SSD/AgentMem-OS/db/agentmem_os.db) instead of a scratch file. Nothing was damaged (the
migration is idempotent and reported "verified"), but this is the exact M4 finding I raised against the
team in Stage 1 R2, committed by my own tooling.
Rule: every critic harness sets AGENTMEM_OS_DB_PATH to a fresh temp file as its FIRST statement, before
any agentmem_os import — conftest only protects code paths that pytest loads first.

## 2026-08-06 — A test can pass for the wrong reason when the FIX and the FALLBACK both produce green (Critics, Stage 2 R6)

`test_devanagari_gate_functions_both_directions` was written to pin the R5-M2 tokenizer rewrite. Reverting
`_TOKEN` to the broken `\w{3,}` left all 44 tests passing. Mechanism: `\w{3,}` shreds the Hindi fact into 3
garbage fragments, which trips a DIFFERENT rule — `need = 1 if len(ftoks) <= 4` — and one shared fragment
('जयप') then satisfies the loosened threshold. The assertion (`created == 1, rejected == 1`) was true under
both the fix and the bug. **Check: when a fix and an adaptive fallback can both produce the asserted
outcome, assert on the fix's OWN observable (here `_tokens()` returning whole words, or the overlap count),
never on the downstream accept/reject.** Generalizes: any test asserting a final verdict over a pipeline
with an adaptive threshold is a candidate for this failure. Evidence: scratchpad/s2r6/mut6.py mutation 16
survived a 44-test suite; direct comparison showed 8 whole words vs ['उपय','गकर','लरक'].

## 2026-08-06 — Measure corpus prevalence BEFORE elevating a constructed shape to a blocker (Critics, Stage 2 R6)

I constructed seven live holes in the new numeric parser: date-expression exemption hiding a fabricated
count ("The user's suite emitted 20 January warnings" → 20 exempted, unchecked), zero-pad collisions
("007" licenses "7"), abbreviated-month blindness ("Dec 2023"), US m/d/y blindness, copied inline stamps
leaking "14:05" digits, trailing-zero decimals ("4.5" vs "4.50"), and "once" claiming the value 1. All are
real and reachable. All measured **0.0% in 193 real stored facts** from live llama3.1 on the target corpus.
Four earlier rounds blocked on shapes that DID have prevalence (glued units: real model wrote 16GB; orphan
digits: destroyed 2 true facts on a 43-fact sample). **Check: a shape becomes a blocker when it is measured
in real output or real corpus text, not when it can be constructed. Record the measured prevalence either
way, so the next round does not re-litigate a shape already shown to be corpus-absent.**

## 2026-08-06 — Verify a tuned fix on a DIFFERENT sample than the one it was tuned against (Critics, Stage 2 R6)

The team's required recall artifact was 10 random sessions at seed 20260806 — the same sample the R5 fix
was diagnosed and validated on. Reproducing it exactly (43 candidates / 2 rejected / both true) proves the
record honest but proves nothing about generalization. A fresh probe at seed 31337 with a different length
window and 3x the sessions (156 candidates) is what actually tested the fix: 4 rejections, all true, zero
false rejections, and it surfaced the value-vs-predicate residual ("The user has asked 38 questions."
licensed by the user typing "19 Question (19/38)") that the original sample never contained. **Check: when
a gate was tuned against sample S, re-run S to audit the record's honesty AND run an independent S' to
judge the fix. Passing only on S is indistinguishable from overfitting.**
