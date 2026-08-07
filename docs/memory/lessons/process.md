
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

## 2026-08-06 — A default-valued column makes `IS NULL` a dead branch (Critics, Stage 3 R1)

`_merge_duplicate_kg_nodes` merges duplicate co-occurrence edges `WHERE relation_type IS NULL`. The ORM
column carries `default="CO_OCCURS"` and the ALTER TABLE carries `DEFAULT 'CO_OCCURS'` (SQLite backfills
existing rows with it), so **34,905 of 34,905 real edges are 'CO_OCCURS' and none are NULL** — the
branch cannot fire in production. The test passed because its hand-written `_OLD_SCHEMA` omits the
default and its INSERT omits the column. **Enforced check:** before writing a predicate over a column,
run `SELECT <col>, count(*) … GROUP BY 1` against the REAL database and put the distribution in the
build log; a fixture's DDL is not the product's DDL, and a Python-side `default=` never reaches raw SQL.

## 2026-08-06 — Moving the cheap compute out of the lock while the expensive compute stays in (Critics, S3 R1)

Stage 3 deliberately moved spaCy NER before the batch transaction ("no model compute while any write
lock is held") and then called the sentence-transformers ALIAS RESOLVER inside it. Measured: model load
87.1–89.2 s per fresh process (with network calls to HF), all of it while the SQLite write lock is held,
and a competing writer FAILED with "database is locked" (busy_timeout 30 s). Embedding was the cheap
part (0.7 ms/text). The code comment even acknowledged "every ms of compute extends the hold".
**Enforced check:** when a rule says "no X inside the lock", enumerate EVERY lazy initializer reachable
from inside the lock (model loads, network clients, import-time singletons) and measure the lock hold
with a competing writer — an unloaded singleton is compute that has not happened YET, not compute that
was moved out.

## 2026-08-06 — A recovery sweep ordered by id with a LIMIT starves behind its own permanent residue (Critics, S3 R1)

`link_missing` selects facts with zero join rows `ORDER BY id LIMIT 500`. A fact whose NER finds nothing
never gets a join row, so it is a permanent candidate — **22% of real facts (5/23 in the reproduced G2
run)**. Measured: 12 entity-less facts at limit=10 make a linkable fact permanently unreachable across
unlimited reruns, and the docstring's "rerun until swept=0 to drain" never terminates. **Enforced
check:** any sweep whose candidate predicate can be permanently true for a row must either advance a
cursor (id > last_seen) or record the attempt; and any "repeat until N=0" instruction must be proven to
reach 0 on a corpus containing the no-op case.

## 2026-08-06 — One-hop graph traversal is not entity unity (Critics, Stage 3 R1)

The cross-lingual claim rests on expanding ALIAS_OF one hop from the seed node. Because the alias anchor
is chosen by BEST similarity, a second Indic variant aliases to the FIRST variant (0.9715) rather than
to the English node (0.9324), forming a chain — and the English query then silently returns 1 of 2
facts. The second variant existed only because the tokenizer keeps the U+0964 danda ("चेन्नई।").
**Enforced check:** when unity is implemented as N-hop traversal, test with THREE nodes in a chain, not
two; and state the hop count in every sentence that promises unity.

## 2026-08-06 — A feature can be schema-complete and prompt-unreachable (Critics, Stage 3 R1)

F7 shipped exactly as the founder specified (enum column, single-source detector, merge matrix, 7 unit
tests) — and fired **0 times in 23 real facts**, because the extraction prompt orders plans to be typed
"state", its own example carrying a DATE. The disclosed boundary mentioned only UNDATED plans. The
marker now depends on the model disobeying the prompt. **Enforced check:** for any new field whose value
depends on upstream classification, grep the PROMPT (or the upstream rule) for the class that would
produce it and report the real-corpus fire rate in the same paragraph as the "built" claim. A feature
with a 0% fire rate on the target corpus is DORMANT, not DONE.

## 2026-08-06 — A case-folding fix that REPLACES the exact predicate regresses exact matches (Critics, S3 R2)

Fixing "reads go blind over casing" by swapping `col == q` for `lower(col) == q.lower()` looks strictly
wider. It is not: SQLite's `lower()` folds **ASCII only**, Python's folds Unicode, so for any text
carrying a non-ASCII uppercase letter the two sides can never meet — `facts_for_entity('Übermensch')`
returned **[] on a node stored byte-identically**, measured 4/4, with **11 such entity_texts in the real
KG** (`Συγνώμη`, `IDÅSEN`, `Ruben Östlund`, `the Champs-Élysées`, …). The fix converted a partial miss
into a false-empty on a path that used to work. **Enforced check:** widen a match predicate with OR,
never by replacement; and whenever a fix moves a comparison from Python into SQL (or back), test one
input where the two languages' semantics differ — casing, collation, NULL ordering, integer division.

## 2026-08-06 — A repair that rewrites keys must normalize them, or it manufactures what it cannot see (Critics, S3 R2)

The node-dedup migration re-points `kg_edges.source_id/target_id` to the keeper without restoring the
src<tgt convention every writer assumes. Because keeper = min(id), the common case INVERTS the pair, and
both the in-group merge and the new global duplicate repair — which group by `(source_id, target_id)` —
then report `merged 0` / `clean` while the undirected loader still keeps one row's weight. The test
fixture's ids were arranged so re-pointing never inverts: **third instance in this arc of "the fixture
cannot produce the failure"**. **Enforced check:** any UPDATE that rewrites a column participating in a
canonical ordering must re-apply that ordering in the same statement, and its fixture must contain at
least one row where the rewrite BREAKS the convention.

## 2026-08-06 — A "zero rows" recovery sweep cannot recover a PARTIAL failure (Critics, Stage 3 R3)

`link_missing` is the compensating control that justified moving alias planning outside the write lock:
skipped surfaces "land in skipped and the sweep recovers them" (said in four places). Its candidate
predicate is `NOT EXISTS(any join row)`. Measured: a fact linking Microsoft and skipping गूगल is not a
candidate at all — the anchor arrives, the sweep reports `candidates=1` (the Indic-ONLY fact), and the
mixed fact is orphaned permanently; `facts_for_entity('Google')` never returns it. The all-or-nothing
predicate covers the failure the sweep was BUILT for (crash before linking) and misses the failure the
new contract CREATES (per-surface skips). **Enforced check:** when a mechanism is offered as the
compensating control for a blocker fix, run it against the exact residue THAT fix produces, not only
against the failure it was originally written for — and state the predicate ("zero links", not "missing
links") in every sentence that promises recovery.

## 2026-08-06 — Importing a module to read a constant ran a destructive migration on the live DB (Critics, Stage 3 R3)

`db/engine.py` ends with a bare `init_db()`, so `from agentmem_os.db.engine import DB_PATH` — typed by a
READ-ONLY reviewer to find the DB path — executed the Stage-3 migration against the founder's production
database: 2,711 CO_OCCURS rows deleted, 1,979 weights and timestamps rewritten, an index built. No
backup, no dry-run, no confirmation; the only trace is one INFO log line. Outcome was benign (byte-
identical to the copy verified the round before, sum(weight) conserved exactly) — by luck of prior
verification, not by design. **Enforced check (reviewers):** never import the product's engine module
in-process; resolve DB_PATH from config.yaml and open production DBs with `file:...?mode=ro` URIs only.
**Enforced check (builders):** a migration that DELETEs rows or rewrites values must not run as an import
side effect without first writing a recoverable pre-state.

## 2026-08-06 — Log writes must target the ABSOLUTE repo path (stray-file incident)
During Stage 3, the shell's working directory silently reset mid-session
and two relative-path appends (`cat >> CONSOLIDATION_V2_BUILD_LOG.md`)
CREATED a stray file one directory above the repo. The paired
grep-verify passed because it checked the same wrong file — the
verification inherited the defect it existed to catch. Caught only when
the R2 critic found the repo log "pointing at nothing". Rule, standing:
every log/notes write AND its grep-verify use the absolute repo path;
a verify that shares the failure mode of the thing it verifies is not a
verify. (Same family as Stage-2 R2's silently no-op'd in-place edits —
the third member of the "verify the real artifact" lesson class.)

## 2026-08-06 — A mock that raises BEFORE the body never exercises the fix (Critics, Stage 3 R4)

`test_retry_does_not_double_append_skipped` was written to pin a real fix (copy the `skipped` list before
apply, so a store-owned retry cannot double-append). Its fake `_link_in_session` raises the race error on
call 1 *instead of* running the real body — so attempt 1 never appends anything and the copy is never
needed. Measured: reverting `skipped = list(skipped)` leaves the test green in 0.37s, while a realistic
harness (raise inside `_get_or_create_node` AFTER the anchor-miss append) reproduces the duplicate entry
immediately. **Enforced check:** when the bug is "state carried from attempt 1 into attempt 2", the mock
must fail LATE — after the state-producing step — or the test proves nothing. Every "fix + test" claim
gets the revert run, not the green run.

## 2026-08-06 — `CREATE TABLE AS SELECT * … WHERE 0` freezes a backup's schema forever (Critics, Stage 3 R4)

A pre-state backup created as `CREATE TABLE IF NOT EXISTS x_backup AS SELECT * FROM t WHERE 0` and filled
with `INSERT INTO x_backup SELECT * FROM t WHERE …` breaks permanently the first time `t` gains a column:
"table x_backup has 11 columns but 12 values were supplied". When the migration runs as an import side
effect, that OperationalError becomes a RuntimeError at `import`, i.e. the whole product stops starting —
and only on databases that actually have rows to back up, so CI on a fresh DB stays green. **Enforced
check:** backup/copy statements name their columns, or the table is rebuilt when its shape no longer
matches the source. Reproduced, Stage 3 R4 (scratchpad p_drift).

## 2026-08-06 — A compensating control needs a CALLER, not just a method (Critics, Stage 3 R4)

A blocker fix was accepted because "the sweep recovers those later", and the sweep is real, correct, and
tested — but `link_missing` is invoked nowhere outside tests: not by the engine after a link failure, not
by any endpoint or CLI command, not by the smoke. Recovery therefore requires a human to open a REPL and
write the documented drain loop. **Enforced check:** when a mechanism is offered as the compensating
control for a risk, grep for its CALLERS in product code before accepting it; if there are none, the honest
claim is "recoverable by an operator", never "recovered". (Sibling of the R1 lesson "a feature can be
schema-complete and prompt-unreachable" — same defect, one layer up.)

## 2026-08-06 — An auto-invoked compensating control inherits a blast radius nobody sized (Critics, S3 R5)

R4 blocked because the recovery sweep had no caller. The fix auto-invoked it inside consolidate_session
after a link_failure commit — correct, tested, mutation-caught. But the drain is scoped to
(agent_id, user_id), not to the failed batch: measured, 300 pre-existing unlinked facts in scope turned one
2-fact consolidation into a 302-fact / 673-link backfill inside the same call, bounded only at
max_rounds×limit = 100,000 facts. Under a PERSISTENT linker fault it becomes O(sessions × scope): 8 sessions
× 400 facts = 3,216 link attempts, 661 KB of log, and 11.9 KB single log lines because the whole failures
list is embedded in the report that gets logged. And it is not best-effort — when the drain's own query
raises (probe: OperationalError), the exception escapes consolidate_session AFTER the facts and log row
committed, discarding the entire report of a run that succeeded. **Enforced check: when wiring a
compensating control into an automatic path, state its worst-case work in the docstring, bound it to the
unit that failed unless a wider sweep is intended, wrap it best-effort so a recovery fault cannot fail a
committed run, and truncate any failure list that lands in a log line.**

## 2026-08-06 — Code written to close a review finding must go through the same mutation sweep (Critics, S3 R5)

The R4 fixes for the findings a critic NAMED were all mutation-proven (19/23 caught). The survivors were
all in code the fix itself introduced: `"complete": rounds < max_rounds` → `True` and
`while rounds < max_rounds` → `while True` both leave the suite green, and coverage confirms the LOUD
runaway warning never executes. Same round, the tightened _validate_plan surface rule (stripped, len>=2)
had no tripwire either — reverting it to the old rule left 14 selected tests green. This is the Stage-2 R3
enforced check ("every claimed fix goes through the mutation sweep, not just the ones a previous round
named") recurring one level up: **new code written to satisfy a reviewer is unreviewed code — mutate its
own safety claims and honesty flags before calling the round closed.**

## 2026-08-06 — A candidate-selection threshold reused as the DECISION gate is not a gate (Critics, Stage 4 R1)

Stage 4's defensibility argument for an 8B judge is "the LLM only proposes; a deterministic co-signal
must agree". The shortlist's lexical pool admits candidates on `_tfidf_cosine(new, cand) >= 0.25`, and
the co-signal then checks `cosine >= 0.25` — same function, same text pair, same constant. Every
lexical-pool candidate passes the gate by construction, so for that pool the verdict is LLM-only.
Measured: "The user is allergic to peanuts." invalidated by "The user is allergic to shellfish."
(cosine 0.6311, no polarity flip, no entity link, dropped=[]) — the exact dangerous class the design
cites Mem0 #1674 for. The reuse was invisible because the pool was added at a LATER gate than the gate
it silently disarmed, and the justifying comment still described the pre-pool world ("our shortlist
already guarantees entity overlap"). **Enforced check: when a new candidate source is added to a
pipeline, re-derive every downstream gate's independence for that source — a gate whose predicate is
implied by the selector must be raised, replaced, or declared inert in the record — and never source a
threshold from a docstring (0.25 was cited to `forget_about`, whose code uses keyword overlap and no
cosine at all).**

## 2026-08-06 — A shortlist admitted for one ACTION is reachable by every other action (Critics, S4 R1)

Planned events were added to the judgment shortlist in a reserved pool so they could be CANCELLED.
The cancellation loop guards `ctype == "event" and cstatus == "planned"`; the supersession loop, which
runs FIRST over the same shortlist, has no type guard at all — so the model naming a planned event in
`superseded_ids` supersedes it. Worse, 'planned' means future-dated, so the domain-time direction rule
always makes the event the WINNER: the user's true current state fact is the one invalidated, and a
candidate double-listed in both arrays ends up cancelled AND the superseder of a live fact. The record
claimed that case was "dropped by the status guard"; reproduced, it drops nothing. **Enforced check:
every candidate that enters a shortlist must be type-checked at EVERY action that consumes the
shortlist, not only at the action it was admitted for — and a direction rule that reads a date must ask
whether one class of candidate always wins that comparison.**

## 2026-08-06 — Committing the action before the audit row makes the recovery marker lie (Critics, S4 R1)

The judge applies supersessions through the store (each its own committed transaction) and writes the
judgment row afterwards in a separate session. A crash in that window leaves an applied, durable
supersession with no audit row — and because the sweep's candidate predicate is "live fact with no
judgment row", the loser is now invisible to recovery: permanently unaudited, never re-judged, against
a documented contract that "a fact with no judgment row has never been judged". The mirror case is
worse for honesty: the re-judgment persists "store refused: already superseded" for an action this same
judge performed. **Enforced check: an audit row that doubles as a recovery marker must be written in
the SAME transaction as the actions it records; if it cannot be, the docstring must say which states
the marker cannot distinguish.** (Sibling of Stage-3 R3's "a zero-rows sweep cannot recover a PARTIAL
failure" — the predicate again covers the failure it was written for and misses the one the new code
creates.)

## 2026-08-06 — A reviewer's own read-only rule dies at the first module-level import (Critics, S4 R1)

I ran the review's named script (`python3 tests/test_conflict_detection.py`) and later imported
`agentmem_os.llm.supersession` to check a scratch path. Both ran `init_db()` — every migration — against
the founder's production DB, because Stage 4 added a MODULE-level `from agentmem_os.memory.conflict_detector
import ...` and that module module-imports `db.engine`, whose tail calls `init_db()`. Measured:
`import agentmem_os.llm.consolidation_v2` does NOT pull the engine; `import agentmem_os.llm.supersession`
does. Outcome was benign (verified read-only: 0 leftover rows, CO_OCCURS 32,194 / 35,051.0 unchanged),
but this is the SECOND occurrence of the R3 incident, and the first one produced a standing rule I
believed I was following. **Enforced check (reviewers): export AGENTMEM_OS_DB_PATH to a scratch file in
EVERY shell and EVERY harness before any import or any script run — not only when a DB is obviously
involved; a task instruction to "run this directly" is not an exemption. (Builders): a new module-level
import edge into a package whose import runs migrations is a defect on its own — keep DB imports
function-local, as the rest of that module already does.**

## 2026-08-07 — A "stripped" signal that strips what the metric already ignores is the same signal twice (Critics, S4 R2)

R1-B2 blocked a vacuous gate (admission cosine == agreement cosine). The fix added an "independent"
metric-update signal: numbers differ AND the NUMBERS-STRIPPED texts are near-identical (cosine >= 0.7).
But the repo's `_tfidf_cosine` tokenises with `re.findall(r"[a-z]+")` — digits were never in the vector.
Measured: `stripped_cosine == cosine` on 20,000 randomized adversarial pairs, zero exceptions. The new
gate is `cosine >= 0.7` wearing a different name, and it is number-BLIND in the worst place: "personal
best time in the charity 5K run is 25:31" vs "...10K run is 55:12" scores cosine 1.000, so the 5K record
is invalidated by the 10K record with dropped=[]. **Enforced check: before claiming a second signal is
INDEPENDENT of the first, compute both on the same inputs and prove they differ — read the tokenizer, not
the function name. A transformation that removes information the measure never had is a no-op, and a
threshold on a no-op is the original threshold.**

## 2026-08-07 — The action nobody attacked is the one with no gate (Critics, S4 R2)

Every gate in the supersession judge sits on the supersede path, because that is the path two review
rounds attacked. The cancellation path — same shortlist, same model output, same transaction — checks
only "is this candidate a planned event". No co-signal, no topical test. The co-signal IS computed for
those candidates and IS persisted; it is simply never read, so an audit row exists that records
`"agrees": false` next to an applied cancellation. And the same round's reader-side fix (cancelled facts
excluded from current_facts and facts_overlapping) silently RAISED the blast radius of that ungated
action from cosmetic to "the plan disappears from the user's memory". **Enforced check: when a module has
more than one write action, enumerate the gates per ACTION, not per module — and when a change makes an
existing action's effect more visible or more permanent, re-derive that action's gate before shipping the
visibility change.**

## 2026-08-07 — A guard pasted after the opening triple-quote is a string, not a guard (Critics, S4 R2)

The Ma8 fix for the reviewer-safety finding ("this script must pin a scratch DB path before any import")
was inserted immediately after the module docstring's opening `"""` — nine lines of import-and-assign that
Python stores as prose. The record says the guard landed; running the script with an inherited
AGENTMEM_OS_DB_PATH still created and migrated 17 tables at the inherited path (verified against a scratch
"PRETEND_PROD" file). Third occurrence of the import-time-migration class in this project, and the first
where the FIX was the failure. **Enforced check: a guard that must run before imports is verified by
OBSERVING it run (print/assert the value it sets, or point it at a decoy path and confirm the decoy stays
untouched) — never by reading the diff. Reviewers: keep exporting the scratch path in every shell anyway;
the guard you were told about may not exist.**

## 2026-08-07 — A gate whose test is the pool's own admission test is the pool, twice (Critics, S4 R3)
**Evidence:** llm/supersession.py:408-416 gates cancellation on `shared_nodes or _content_word_overlap(new,
cand)`. Cancellation candidates can only reach the shortlist through the planned pool (:544-558), whose
admission test is *the same two properties*: entity-linked facts are drawn from `peer_fact_ids` (shared
node -> `shared_nodes` True), entity-less facts are filtered by `_content_word_overlap(fact.fact_text,
c.fact_text)` — the identical call the gate then repeats. Measured on every reachable candidate: the gate
passes 100% of the time. The only test that turns "delete the gate" red builds a snapshot by hand that the
shortlist cannot produce.
**Rule:** before believing a gate, ask what SELECTED the thing being gated. If the selector and the gate
test the same property, the gate's true rejection rate is zero and its test must be written through the
product entry point, not through a hand-built snapshot. Third occurrence of lessons:504 in one stage.

## 2026-08-07 — "X implies Y, so the check is dead" needs a fuzz over the ALPHABET, not the semantics (Critics, S4 R3)
**Evidence:** `_metric_update` dropped its length check because "equal masks imply equal token counts (each
'#' is one token)". True for every digit, unicode digit and whitespace input — 200k fuzz, zero
counterexamples. False the moment a literal `#` appears in the source text: "ticket is 7 and it is 3 days
old" vs "ticket is # and it is 3 days old" mask identically with 2 vs 1 numbers, `zip` silently truncates,
and the function returns True on a misaligned comparison (7 against 3). My first fuzz missed it because my
alphabet did not contain the mask character itself.
**Rule:** when a lemma justifies deleting a guard, fuzz the alphabet including the SENTINEL the transform
emits. The output character of a normalizer is always a legal input character.

## 2026-08-07 — A rebuilt function whose docstring still describes the version it replaced (Critics, S4 R3)
**Evidence:** `_cosignal`'s docstring (:576-577) still says the metric signal is "numbers-stripped texts
near-identical (cosine >= 0.7)" — the exact arithmetic the previous round proved vacuous and which no
longer exists anywhere in the function; and (:572-573) still calls the flip subject guard
"entity/content-word overlap" while the code twenty lines down requires an entity node only, contradicted
by its own inline comment. Both survived a round in which the function was rewritten and the build log was
corrected in five places.
**Rule:** a rewrite's diff must include every self-description ABOVE the changed lines, not only the
build-log record. Grep the changed function for the falsified constant/phrase before calling the fix done.

## 2026-08-07 — Raising a threshold is not the same as changing what it measures (Critics, S4 R4)
**Evidence:** R3 blocked a cancellation gate that required ONE shared content word ("climbing") between
the cancelling text and the plan. The fix raised it to TWO and added a clause test. Both the old and the
new rule measure the same thing — how many >=5-char words the two strings happen to share — so the
failure just moved up one word: "cancelled his weekend TRAINING session with the physiotherapist"
cancels "plans to join the weekend TRAINING camp for the marathon", real llama3.1, 3/3 deterministic.
The calibration is inverted at the same time: verbatim TRUE cancellations of short-named plans ("Rome
marathon", "yoga class", "gym trial") all REFUSE at 1 shared word. Generic words are frequent, so a
count-based rule admits the coincidences and rejects the identities.
**Rule:** when a gate is defeated by a coincidence, ask whether the fix changes the MEASURE or only its
threshold. A threshold change must be validated with a fresh adversarial set built from the SAME
coincidence mechanism at the new setting (here: two generic words, not one) — re-running the old
counterexample only proves the old counterexample is dead. And measure the true-positive side too: a
gate that rejects real cancellations while admitting fake ones is not conservative, it is miscalibrated.

## 2026-08-07 — An assertion that cannot fail is how a survivor mutation comes back green (Critics, S4 R4)
**Evidence:** R1-m3/R2/R3-m3 all reported the same survivor — the recovery sweep's `superseded_by IS
NULL` filter has no tripwire. The R3 fix round added
`assert all(r2.skipped != None or True for r2 in rows_a)` to test_shortlist_and_sweep_exclude_superseded,
and the build log recorded "m2/m3 pool and sweep live-filters pinned". `X != None or True` is a
tautology; the neighbouring `assert not any(r2.raw_output_json ...)` also cannot fail because the code
under test skips superseded facts before the LLM call. Deleting the filter still passes all 228 tests,
fourth round running.
**Rule (builders):** a test written to kill a surviving mutant must be validated by RUNNING that
mutation and watching it go red — a pin is proven by the red, never by the green. **(Reviewers):** grep
new assertions for `or True`, `!= None or`, `assert x or`, and any `assert` whose right side is a
constant, before believing a "pinned" claim; then re-run the exact mutation the pin names.

## 2026-08-07 — A position-accurate check applied to a string-matched context checks the wrong span (Critics, S4 R5)
**Evidence:** `_cancellation_binds` v3 tests negation at the exact match offset
(`new_text[m.start()-40:m.start()]`) but then picks the clause with
`next(c for c in split(new_text) if m.group(0).lower() in c.lower())` — the FIRST clause containing the
cue STRING, not the clause containing THIS match. With the same cue form in two clauses, a non-negated
occurrence is judged on a NEGATED clause's words: "The user did not cancel the pottery workshop weekend,
but did cancel the pottery class." marks "The user plans a pottery workshop weekend." CANCELLED
(measured through judge_fact, event_status='cancelled'). The pinned test could not catch it because its
two cue forms differ ("cancelled" vs "cancel"). The mirror image is a silent miss: a true cancellation in
the second clause is judged on the first clause and refuses.
**Rule:** when one part of a check uses match POSITIONS and another part re-finds the same token by
string search, they can disagree — carry the span through (offsets) instead of re-searching. Reviewers:
for any per-occurrence loop, build a case where the SAME surface form appears twice with different
polarity and assert the gate reads the occurrence it claims to read.

## 2026-08-07 — A symmetric rule cannot be pinned by a positive example (Critics, S4 R5)
**Evidence:** R4-B1's fix hinged on lowering the content-word floor from 5 to 4 chars "to recover the
distinctive short words the 5-char rule destroyed ('yoga', 'rome')". The pin added for it asserts the
Rome cancellation BINDS its Rome plan. Because containment compares two sets built by the SAME extractor,
raising the floor back to 5 drops 'rome' from BOTH sides and leaves {marathon} ⊆ {marathon} — the
assertion still passes, and the whole 233-test suite stays green while "cancelled the Rome marathon"
starts binding a BOSTON marathon plan and "cancelled the yoga class" starts binding a pottery class.
**Rule:** a constant that controls DISCRIMINATION is only observable on a NEGATIVE case. When a rule is
symmetric (same transform applied to both sides), a positive-example test pins nothing about the
transform — write the negative (a distinctive word present on one side only) and prove it by running the
mutation red.

## 2026-08-07 — A text splitter feeding a containment rule has all its errors in the unsafe direction (Critics, S4 R6)
**Evidence:** `_cancellation_binds` decides "everything the cancelling CLAUSE names must be in the plan",
and the clause comes from `_CLAUSE_SPLIT_RE = [.;!?]|\band\b|\bbut\b`. An abbreviation period ends the
clause early: "The user cancelled the appointment with Dr. Meyer." → clause "The user cancelled the
appointment with Dr" → named={appointment}, which IS a subset of "plans an appointment at the downtown
clinic" → BINDS, and judge_fact writes event_status='cancelled' with dropped=[]. The control without the
period ("with Doctor Meyer") refuses. 5 of 6 abbreviation shapes (Dr./St./Mr./Mrs./Ave.) false-bind.
The asymmetry is structural: a splitter false positive can only SHRINK the named set, and a smaller set
is strictly more likely to be a subset — so every parsing error pushes toward BINDING, never toward
refusing. The defect survived rounds 3-6 because each round tested the RULE and never the PARSER feeding
it, and because the round-4 record wrote "abbreviation clause-splitting verified harmless for cue-first
phrasings" from a probe that only exercised the cue-LAST (safe) direction.
**Rule (builders):** when a gate's decision is a set relation over words a parser extracted, ask which
direction each parser error moves the decision. If all errors move toward the unsafe verdict, the parser
is part of the gate and must be adversarially tested with the gate — real punctuation (abbreviations,
decimals, ellipses, quotes), not just clean sentences. **(Reviewers):** never accept "class X verified
harmless" unless the record names the probes; re-run the class in the direction the probe did NOT cover.
A finding raised in the safe direction does not license a "harmless" verdict on its mirror.

## 2026-08-07 — Fixing the context selector without fixing the context WINDOW leaves the fix half-applied (Critics, S4 R6)
**Evidence:** R5-Ma1 was fixed correctly — clause boundaries now come from the splitter's spans by match
position — but the negation test next to it still reads a raw `new_text[m.start()-40:m.start()]`, which
crosses the very boundaries the fix computes. Result, measured: "The user did not cancel the pottery
class but cancelled the pottery workshop weekend." reports BOTH occurrences NEGATED and refuses a true
cancellation. Safe direction here, so no test noticed, and the build log claims "span-accurate negation
demands span-accurate clauses" as if both were now span-accurate.
**Rule:** when a fix replaces one notion of "the surrounding context" with a better one, grep the same
function for every OTHER place that computes context by the old notion and either convert them or state
in the record that they were deliberately left, with the measured consequence and its direction.

## 2026-08-07 — The REPLACEMENT disclosure is an unverified claim too (Critics, S4 R7)
**Evidence:** R6 blocked Stage 4 because BUILD_LOG said "abbreviation clause-splitting verified harmless
for cue-first phrasings" and measurement showed 5/6 false BINDS. The fix struck that line — and wrote a
new one: "single-letter abbreviations ('p.m.') still split, in the safe cue-first direction only." Nobody
probed it. R7 probed it: **5 of 6 cue-first phrasings falsely bind** — "cancelled the Friday 6 p.m.
dinner reservation" → clause truncates to "…the Friday 6 p" → named={friday} ⊆ "plans a Friday lunch
reservation" → BINDS. Identical mechanism, identical direction, identical 5/6 rate as the line it
replaced. The code fix was scoped to CAPITALIZED 1-3-letter tokens (`(?<![A-Z])…`); the record described
the residual as safe without measuring the residual.
**Rule (builders):** when a fix covers part of a class, the disclosure of what is LEFT is a new claim and
needs its own probes before it is written. Never carry a "the rest is safe" sentence out of a round that
was blocked for exactly that sentence. **(Reviewers):** after a fix lands, probe the COMPLEMENT of what
the fix covers — read the guard's own scope words ("capitalized", "1-3 letter") and build the probe set
from what they exclude.

## 2026-08-07 — Measure the direction of the FIX, not only the direction of the bug (Critics, S4 R7)
**Evidence:** R6-m1 narrowed the negation lookback to the clause (`max(c_start, m.start()-40)`). The BUG
it fixed was refuse-direction (a prior clause's negation suppressed a true cancellation). The FIX is
binding-direction: a narrower window sees FEWER negations, so more cues bind. Measured 5/5 new false
binds the old code refused — "The user did **not** cancel the pottery class **and** cancel the pottery
workshop weekend." now binds the workshop plan, because the elided negator sits in the prior clause.
No test noticed: reverting the change leaves 54/54 green.
**Rule:** any change to a suppression/veto window is a change to how often the unsafe verdict is reached.
State which direction the fix moves the gate, measure the newly-admitted cases, disclose them, and pin
the fix by a NEGATIVE (a case that must still be suppressed) — a fix whose reversion leaves the suite
green is unpinned, whatever its round number.

## 2026-08-07 — Fix ONE alternative of a regex, and the other alternatives keep the bug (Critics, S4 R8)
**Evidence:** `_CLAUSE_SPLIT_RE` has four alternatives: `\.` , `[;!?]` , `\band\b` , `\bbut\b`. R6-Ma1
found the period alternative truncating the cue clause and false-binding 5/6; R7-B1 found the residual
LOWERCASE half of the same alternative false-binding 5/6; both were fixed. Nobody ever probed `and`/`but`
— and they truncate identically, because a conjunction inside a coordinated noun phrase splits BEFORE the
shared head noun: "cancelled the Friday **and Saturday dinner reservations**" → clause "…cancelled the
Friday and" → named={friday} ⊆ "plans a Friday **lunch** reservation" → BINDS. 5 of 6 measured, and it
re-reaches the EXACT sentinel pairs (Friday dinner→lunch, German lesson→exam) that the R7 pin certifies
as refusing. Two rounds of fixes hardened one branch of a four-branch regex and the record moved on.
**Rule (builders):** when the defect is in one alternative of an alternation (regex `|`, a dispatch
table, an if/elif chain), the unit of repair is the ALTERNATION, not the alternative. Enumerate every
branch, run the same probe set through each, and record per-branch results — "fixed" means the class is
closed on all branches or the open ones are named in the disclosure. **(Reviewers):** when a fix lands on
one branch, build your next probe set by substituting the OTHER branches into the round's own pinned test
fixtures. If the pin's sentinel pair is still reachable through a sibling branch, the class is not closed.

## 2026-08-07 — A "safe direction" argument breaks at the boundary case the rule short-circuits on (Critics, S4 R8)
**Evidence:** The record justifies a deliberate under-splitting trade with "missed sentence splits GROW
the clause, which is the refusing direction" — true for the `outside = named - plan_words` test, since a
bigger `named` can only add words outside the plan. But the gate short-circuits FIRST on `if not named:
refuse` ("cue clause names nothing"). Growth can turn an EMPTY named set into a non-empty SUBSET, which
flips refuse → BIND. Measured once (contrived text: "The user cancelled it. The pottery workshop weekend
was fun." binds, because "it." is a 2-letter token so the period no longer splits).
**Rule:** a monotonicity argument ("more input can only move the verdict toward safe") is only valid over
the branch it reasons about. Before writing it into a record, list every early-return/short-circuit the
value passes through and check the argument at each one — empty-set and zero-length guards are where
monotonicity usually inverts.

## 2026-08-07 — A read-only reviewer's coverage run is a WRITE (Critics, S4 R8)
**Evidence:** Running the dispatched `pytest --cov` inside the repo made pytest-cov write and combine
`.coverage*` data files in the repo root; the untracked parallel data file present at session start was
consumed and replaced. Source, test and doc hashes were all byte-identical, so the review's read-only
claim held where it mattered — but the working tree was not untouched, and the entry would have been
technically false if written as "nothing changed".
**Rule (reviewers):** export `COVERAGE_FILE=$SCRATCH/.coverage` (or run coverage against the shadow copy)
before any `--cov` invocation, alongside the existing `AGENTMEM_OS_DB_PATH` rule. Tools that look
read-only — coverage, profilers, caches, `--lf`/`.pytest_cache`, `__pycache__` — all write. Verify
read-only by HASHING the files you care about and disclosing anything else you touched, rather than by
asserting a clean session.

## 2026-08-07 — Rank-based selection is undone by head-truncation of a chronologically-ordered block (Critics, S5 R1)
**Evidence:** `FactRetriever.build_block` selects by RANK ("the best evidence must survive the budget",
fact_retrieval.py:166), then sorts the survivors chronologically ASCENDING, and the assembler applies
`_fit_to_budget(..., keep="head")`. Head-keeping on an ascending timeline keeps the OLDEST and cuts the
NEWEST — which is where the rank-0 fact usually is on a knowledge-update question. Measured: 30 low-
relevance 2020 facts + 1 rank-0 current fact at a 120-token allocation rendered three 2020 workshop lines
in full and the current fact as `"Rachel is cu"`. The repo had ALREADY measured this exact class for the
chunk path — qa_accuracy_eval.py:277-284, "Rank order made that accidentally safe; chronological
presentation made it catastrophic (earliest-dated survived, gold evidence was cut ... collapsed to 0.13)"
— and the new tier reintroduced it. Compounding: transition lines (`[change history: ...]`) are added
AFTER the budget fill and never counted, so overflow is the default (1072 chars rendered for
char_budget=200), and `build_block` always admits the first fact regardless of budget.
**Rule (builders):** whenever selection order and presentation order differ, the truncation end must
follow the PRESENTATION order, not the selection order — or the block must be built so it cannot
overflow (count every character you will render, transition lines included). **(Reviewers):** for any
"rank decides what survives" claim, build a fixture where the top-ranked item is LAST in presentation
order, squeeze the budget, and assert on the surviving TEXT. A test that only asserts the section exists
(`test_full_facts_block_starves_chunks_loudly`) cannot see this.

## 2026-08-07 — A cache-depth guard that can never be satisfied turns the cache into a write amplifier (Critics, S5 R1)
**Evidence:** `get_history` was fixed to `if cached and len(cached) >= last_n` so a 10-turn cache cannot
answer `last_n=20`. Correct — but `RedisCache.max_turns` is 10 and the assembler's only call is
`last_n=20`, so the hit is now IMPOSSIBLE, every call falls through to SQLite, and the unchanged
repopulate loop (store.py:211-217) re-pushes turns into an already-warm list. Modeled on the real
lpush/ltrim semantics: a 5-turn session's cache becomes `[t1..t5,t1..t5]`, and `mcp_server/server.py:377`
(`last_n=10`) then serves ten turns with every turn duplicated. The fix's own test cannot see it: its
`_FakeRedis.push_turn` is `pass` and `get_history` returns a constant list, so accumulation is
unmodelable — and `tests/conftest.py` forces `AGENTMEM_OS_DISABLE_REDIS=1`, so no test in the suite ever
exercises the real cache path.
**Rule (builders):** when you add a precondition to a cache hit, check it against the cache's OWN
capacity and the real callers' parameters — if capacity < the dominant caller's request, you have
disabled the cache, and you must then also fix the write path it still triggers. **(Reviewers):** two
fixes landing together where one (a kill-switch forced in conftest) removes the test suite's ability to
observe the other is a structural blind spot, not two independent fixes — say so, and demand a fake that
models the real data structure (lpush/ltrim/lrange), not one that returns a constant.

## 2026-08-07 — A generosity fix that appends its candidates last is deleted first by the cap (Critics, S5 R1)
**Evidence:** `_query_surfaces` was widened during G1 because bare 'Rachel' yields no NER surface and
adjacent names merge into one 'Rachel Priya' span "that matches no node". The fix appends capitalized
sub-words at the END of the list, and `retrieve` then takes `uniq_surfaces[:_QUERY_SURFACE_CAP]` (8).
Measured on a 7-entity question: 17 surfaces, the 8 kept are the 5 merged multi-word spans (exactly the
ones known not to resolve) + 2 single names + 'Rachel'; 9 of the 10 sub-word surfaces the fix exists to
produce are dropped. No test touches the cap.
**Rule:** a cap and an ordering are one decision. When you add a candidate source to fix a recall gap,
place it in the ordering by its VALUE, not by where it was convenient to append, and pin the cap with a
fixture that overflows it.

## 2026-08-07 — Fixing a budget in the wrong UNIT moves the failure, it does not close it (Critics, S5 R2)
**Evidence:** R1-B1 (head-truncation deleting the rank-0 fact) was fixed by making `build_block` fill
against the full rendered line and never exceed `char_budget`. But the caller's real constraint is
TOKENS: the assembler passes `char_budget = sem_budget * 4` and then calls
`_fit_to_budget(block, sem_budget, ...)`, which token-counts and binary-search-truncates. Measured
density of a rendered fact block (dates, brackets, type parens) is 3.68-3.84 chars/token — ALWAYS below
the 4.0 the proxy assumes — so a block that exactly satisfies the char budget always overshoots the
token budget. Swept semantic budgets 60..1200 step 10: 95/115 (83%) still lost the rank-0 fact, section
ending mid-word (`"[2024/12/31] (state) Rac"`). The producer and the consumer of a budget must agree on
the unit; a 4-chars-per-token proxy is a fast path, never a contract.
**Rule (builders):** when two components share a budget, both must measure it with the SAME function.
If the enforcing side counts tokens, the filling side must count tokens. **(Reviewers):** for any
"never exceeds the budget" claim, sweep the budget across a wide range rather than testing one value —
threshold bugs are intermittent by budget, and one passing value proves nothing.

## 2026-08-07 — Write the pin's needle so it exists ONLY in the thing you are protecting (Critics, S5 R2)
**Evidence:** `test_truncation_cannot_delete_top_ranked_fact` was written to prove the rank-0 CURRENT
fact survives. Its fixture is 1 current fact ("Rachel is currently working at TechCorp.") + 30 fillers
("Rachel attended **TechCorp** workshop session number i downtown."), and the assembler-half assertion
is `assert "TechCorp" in out`. Every filler contains the needle. Ran the exact fixture at the exact
budget: assertion True (green) while `current.fact_text in out` was False and the section ended
`"[2024/12/31] (state) Rac"` — the pin is satisfied BY the failure state. This is the second time in one
stage that a blocker's replacement pin was green while the blocker fired (cf. 2026-08-06 Stage 2 R3,
"An assertion that matches the mutant is not a tripwire").
**Rule (builders):** the needle in a survival test must be a string that appears in the protected item
and NOWHERE else in the fixture — assert `subject.fact_text`, not a shared brand/entity token.
**(Reviewers):** for every new pin, grep its needle against the rest of its own fixture before believing
it; if any distractor contains the needle, the pin is decorative. Then run the fixture and assert the
property directly, independent of the test's own assertion.

## 2026-08-07 — A pin whose fixture is excluded by the BUDGET, not by the mechanism, pins nothing (Critics, S5 R3)
**Evidence:** `test_fill_stops_at_first_nonfit_no_leapfrogging` was written to pin `break`-not-`continue`
in build_block's fill: facts [alpha (short), beta (oversized), gamma (short)] at `token_budget=30`,
asserting `"gamma" not in block`. Measured the fixture's real costs: alpha=19 tokens, gamma=19,
alpha+gamma joined = 38. At budget 30 gamma cannot fit under EITHER branch, so `continue` produces the
same block as `break` and the mutation survives with 35 passed. Discriminating budgets are 38..168; the
test picked 30. Third tautological pin in one stage (cf. the "TechCorp" needle, S5 R2 B1a).
**Rule (builders):** for any test asserting "X is excluded", first compute what X costs and prove the
budget ADMITS it — the exclusion must come from the mechanism under test, never from the limit. Assert
the positive control too ("under the reverted behaviour X WOULD appear").
**(Reviewers):** never accept "now PINNED" on inspection. Run the mutation. If you cannot, compute the
fixture's own arithmetic against its own threshold and show the assertion holds on both sides.

## 2026-08-07 — Enforcing a budget in the right unit can cost you an order of magnitude (Critics, S5 R3)
**Evidence:** Fixing a char-vs-token budget mismatch correctly (fill against `TokenCounter.count` of the
accumulating block) made the fill O(n²) in tokenization: each candidate re-tokenizes the entire block
built so far. Measured at the product default budget — 100 facts 53 ms, 200 facts 198 ms, 350 facts
586 ms, 500 facts (the module's OWN `_LEXICAL_SCAN_CAP`) 1150 ms per call, against 4.5 ms for a single
count of the final block. That sits inside a synchronous product read path, and 500 facts is the
designed ceiling, not an outlier.
**Rule (builders):** when you replace a cheap proxy with an exact measure inside a loop, make it
INCREMENTAL (cost of the new item + join, one exact count at the end) and measure at your own declared
cap before calling it done. **(Reviewers):** every correctness fix that moves work inside a loop earns a
complexity question and a timing run at the component's own documented limit — a fix that is right and
20x slower is a finding, not a footnote.

## 2026-08-07 — A fixture's chars/token ratio is a hidden parameter of every budget sweep (Critics, S5 R4)
**Evidence:** `_fit_to_budget` cuts TWICE — a char fast path (`len(text) > token_budget*4` → head-cut)
and a token binary search. Three rounds of B1 fixes hardened the token side; nothing ever constrained
chars. In R3 I probed the char gate explicitly and measured "0 of 115 budgets" — true, but only because
that fixture's rendered lines measured 3.7-3.88 chars/token, just under the 4.0 threshold. Re-run with
ordinary long-common-word English prose (5.87 chars/token, no rare tokens, no adversarial input), the
gate fires at 9 of 9 budgets including the product default 15360 and qa_accuracy_eval's 4740, deleting
the rank-0 current answer every time while build_block's token contract held exactly (4738 ≤ 4740).
Every test in the suite sits on the safe side of a threshold nobody knew was a parameter.
**Rule (builders):** when a budget is enforced through a proxy ratio, the fixture must straddle the
ratio — include content ABOVE and BELOW it — or the sweep only proves the ratio you happened to pick.
**(Reviewers):** before reporting "N of M budgets clean", measure the fixture's position relative to
every threshold in the path and say what regime the number covers. A sweep over the wrong axis is a
precise-looking zero. I reported one; it was true of the fixture and false of the class.

## 2026-08-07 — Prove equivalence by measurement, not by argument, before excusing a live mutant (Critics, S5 R4)
**Evidence:** Four mutations of the new incremental token estimate (drop the +1 join term, never reject
at the boundary, deliberately under-count, skip the drift reset) all survived the suite. The tempting
calls are both wrong: "unpinned, therefore a finding" and "the docstring says it's a fast path,
therefore fine". Measured instead across 58 budgets × 201 facts: 0 output differences, 0 budget
violations, 0 rank-0 losses, and identical fact counts versus a true exact fill. They are genuine
equivalent mutants because the post-sort exact trim is the real guarantee — and that is now a measured
claim with a harness behind it, not a reading of the comment.
**Rule:** an equivalent-mutant excuse is only admissible with a differential harness that ran both
variants over a real input range and found no observable difference. Otherwise it is an untested
assumption wearing a design rationale.

## 2026-08-07 — A new guard can MASK the mechanism an older test was written to observe (Critics, S5 R5)
**Evidence:** `test_count_calls_stay_linear` was added to guard the O(n) tokenization property by
killing the P2 mutant (boundary exact-count never rejects). In the same round a char break was added
AHEAD of the token logic in the same loop. On the test's fixture (3.83 chars/token) the char break
terminates the fill at the same point the token break would, so correct code and P2 produce IDENTICAL
call counts — 5 vs 5 at token_budget=60, 23 vs 23 at 600 — against an assertion of `<= 30`. The mutant
is not equivalent: on 1.76-chars/token content it is obvious (7 vs 16, 11 vs 28, 19 vs 56, 35 vs 89).
The guard was written correctly and then blinded by a sibling fix landing in the same commit.
**Rule (builders):** when you add an early-exit ahead of existing logic, re-run every mutation that the
downstream logic's tests were written to kill — a guard that passes because something upstream now
short-circuits is not passing. **(Reviewers):** on any round that adds a new gate to an existing loop,
replay the PRIOR rounds' mutants, not just the current ones. This is the second time in one stage that
the chars/token ratio decided whether a test could see anything at all; when a hidden parameter is
found once, sweep every existing fixture against it rather than only the new one.

## 2026-08-07 — A reviewer's own error propagates into code and record faster than its correction (Critics, S5 R6)
**Evidence:** I asserted in R3 that `TokenCounter("gpt-4o")` resolves to cl100k_base; it resolves to
o200k_base. By the time I caught it in R5, the wrong name had been copied into three live places —
the build log's R3 fix record, `_trim_to_budget`'s docstring in shipping code, and a test docstring —
because the team quotes reviewer measurements verbatim (correctly: that is what makes a record
auditable). The R5 note recorded the correction in one place; the three original instances stayed.
**Rule (reviewers):** when you state a fact the team will quote — an encoding name, a constant, a
threshold, a measured ratio — verify it with a one-line probe before writing it, not after. When you
later correct yourself, GREP the correction: list every file and line that carries the wrong version
and name them individually, because a general "I was wrong about X" note does not reach the copies.
**(Builders):** attribute quoted measurements to the round that produced them, so a later correction
has a search key.
