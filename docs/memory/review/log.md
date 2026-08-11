
## 2026-08-06 — Critics — Consolidation v2 Stage 1, G3 ROUND 4 — **BLOCK**

**Claim reviewed:** "49/49 tests, 98% coverage, smoke green; round-3 items N1-N13 resolved."
**Method:** read db/semantic_facts.py, db/engine.py, models.py SemanticFact, tests, conftest, smoke;
re-ran r3 harnesses (two_process2, lock_order, type_collision, partial_uq_clean, a3_real_engine,
perf_churn, queuepool_alt) + 4 new r4 harnesses; pytest scoped to tests/test_semantic_facts.py only.

**Verdict: BLOCK — 1 blocker, 3 majors, 2 minors.**
- R4-1 BLOCKER: `test_cross_process_reaffirmation_loses_nothing` fails 3/5 full-file runs, 6/10 isolated.
  G1 not met; "49/49" and "mention_count exactly 60" are unreproducible. Root cause: the optimistic CAS
  reads (autocommit SELECT) and writes (new write txn) in SEPARATE transactions. Measured exhaustion of
  _REAFFIRM_RETRIES on a contested row: 1.7% (3p x20), 2.7% (3p x50), 2.0% (5p x20), 3.1% (8p x20),
  1.5% (8 threads x25), 0.8% (16 threads x25). LOUD, not silent — no anthem-invariant violation — but a
  hard failure of the store's core op, in-process as well as cross-process.
- R4-2 MAJOR: 4 of 6 uncovered lines (328-333, 335, 343, 345) are exactly the N1 error paths; 345 fires
  in production shape. Build log states bare "98%" with no characterization — R2 M1 / R3 N6 repeat.
- R4-3 MAJOR: `except OperationalError` too broad + destroys root cause. Proven: read-only DB surfaces as
  "lost 8 version races", __cause__=None, __context__=None.
- R4-4 MAJOR: QueuePool ceiling (5+10=15) undisclosed in the engine's own decision comment. 25 live
  sessions -> 25 TimeoutErrors after a 30s stall. Partly my R3 N11 omission; must be documented before
  Stage 2 fans out caller-owned batch sessions.
- R4-5 MINOR: engine.py:132 comment still says "(NullPool)" after the QueuePool switch (N6-N8 repeat).
- R4-6 MINOR: build log stage table still "42/42, 99%" / "R3 pending" vs body's 49/49 / 98% (R2 M5 repeat).

**Verified genuinely fixed:** N2 (lock gone, 0.2s vs 31s freeze), lock-removal safety (2/4/8/16 threads on
one NEW fact -> 1 row, 1 created, exact counts, 0 errors), N3, N4 (real-module binding + pool tripwire;
a3 100 rows/400 mentions/300 turns/0 errors), N5, N8, N9/N12, N11 throughput (8.4k vs 1.886k ~ claim),
version-guard soundness, connection-state cleanliness, refresh() correctness, coverage number, G2 smoke
(exact repro), and N10 does NOT brick the founder's real DB (verified read-only).

**Who needs to know:** Dev-Head (owns fix), bosses (gate held). Re-review on fix.

## 2026-08-06 — Critics — Consolidation v2 Stage 1, G3 ROUND 5 — **BLOCK**

**Claim reviewed:** "R4-1..R4-6 resolved; re-affirmation rewritten deterministic (relative
increment acquires the SQLite write lock, re-read + merge in the same txn); N1 test 10/10."
**Method:** read db/semantic_facts.py, db/engine.py, models.py, tests, conftest, build log,
design doc; re-ran r3/r4 harnesses (two_process2, contention_sweep, operr_breadth) + 6 new r5
harnesses (lockhold, expire_all_probe, expire_all_cost, caller_batch2, mixed_hammer,
targeted_expire, edge_probes) in scratchpad/r5; pytest scoped to tests/test_semantic_facts.py only.

**Verdict: BLOCK — 1 blocker (NEW), 2 majors, 4 minors. Every round-4 item verified genuinely fixed.**

- R5-1 BLOCKER (NEW): `_reaffirm` line 311 `session.expire_all()` SILENTLY DISCARDS every
  unflushed edit a caller staged in its own batch session, and the store reports success.
  Measured on the exact production session config (SessionLocal autoflush=False): caller set
  turn.content="MARKED-BY-CALLER" and fact.extraction_model="CALLER-EDIT", called
  add_fact(db=session) on an existing fact, both reverted in-session and `db.commit()` persisted
  nothing. autoflush=True hides it (Query.update autoflushes) — production is autoflush=False.
  Pending (unflushed NEW) objects survive, so it is selective and invisible. Blast radius: the
  documented `db=` batch API Stage 2 is designed around. expire_all() buys ZERO correctness:
  targeted `session.expire(fact)` gives the identical fresh re-read (sees another writer's
  committed merge) and keeps the caller edit — verified. Also costs: 599 extra SELECTs (46ms)
  when a caller re-touches 600 loaded objects. On owns=True the preceding rollback() already
  expires everything, so 100% of expire_all's effect is the harmful one. No test stages a caller
  edit — the suite is blind to it.
- R5-2 MAJOR: `_reaffirm`'s docstring (lines 277-287) still describes the DELETED mechanism —
  "Version-guarded", "UPDATE only lands if the row is unchanged since we read it", "re-read on a
  fresh snapshot and retry", "retry up to _REAFFIRM_RETRIES" (constant no longer exists),
  "version conflict raised LOUDLY". 4th repeat of the stale-doc class (N6-N8, R4-5), on the one
  function rewritten this round.
- R5-3 MAJOR: build log has NO round-4 record; the round-3 N1 row still presents the optimistic
  CAS + retries design and its falsified proof ("mention_count exactly 60") as the resolution;
  "CURRENT artifacts: 49/49 tests, 98% coverage" is stale (measured 99%, 1 missed line) and
  pre-dates the rewrite. A reader of the evidence doc today concludes the shipped design is the
  one R4 measured broken.
- R5-4 MINOR: tests/test_semantic_facts.py:334 still says "Fixed at the engine level (NullPool
  per-session connections + WAL)" — R4-5 fixed engine.py only; R3 claimed "both files".
- R5-5 MINOR: line 310 ("fact vanished during re-affirmation") is the only uncovered line; I
  PROVED it fires (monkeypatched concurrent DELETE -> correct loud ValueError) and it is testable
  in ~10 lines. Cover it or name+justify it in the log (R4-2 enforced check).
- R5-6 MINOR: `mention_count` has a Python-side default only (no server_default, nullable). A row
  with NULL mention_count absorbs every affirmation forever: NULL+1=NULL, no error, provenance
  reports None. Latent (no non-ORM writer today) but it defeats the tier's count thesis; a
  Python-side read-modify-write would have raised, the SQL relative increment cannot.
- R5-7 MINOR: `_reaffirm_worker` (test line 601) + its sessionmaker omit production
  `autoflush=False, autocommit=False` — the flagship cross-process proof does not run the
  production session config (R2 M6 class; the one live bug found is autoflush-sensitive).
- Notes: comment line 301 ("surfaces as a loud OperationalError and the batch retries") —
  measured: SELECT-only sessions do NOT pin a DBAPI snapshot here (in_transaction False), so that
  path is near-unreachable, and nothing in the repo retries a batch. Forward-looking: a caller
  batch holds SQLite's single write lock from its first write to commit (measured another process
  blocked 1.8s); with busy_timeout=30s, Stage-2 batches longer than 30s will fail every other
  writer in the product.

**Verified genuinely fixed (say it plainly):** R4-1 deterministic re-affirmation holds under
everything I could throw at it — N1 test 10/10 isolated, full file 5/5 at 49/49, contention sweep
3p/3p50/5p/8p and 4/8/16 threads all 0% raised with exact mention_count and all langs,
two_process2 3 procs x 150 facts -> 450/450 zero loss zero errors, mixed hammer (8 affirmers + 2
supersede loops + 2 caller batches + 2 readers) -> 0 errors and 341/341 exact. DECISIVE probe:
stalling 3s between the increment and the re-read while two OS processes fired at the same row —
they blocked ~2.9s and merged cleanly, final state exact; the write-lock claim is true, not
asserted. R4-2/R4-3: retry loop, broad OperationalError catch, caller-conflict raise and
retries-exhausted raise are structurally GONE; read-only-DB probe now surfaces the real
OperationalError with __cause__/__context__ and the real message. Uncovered lines 6 -> 1. R4-4
pool ceiling explicit (engine.py:115-118). R4-5 engine comment fixed. R4-6 stage table refreshed
(body not). G2 smoke reproduces exactly (93 created / 0 created+95 re-affirmed / both index
plans). Throughput: insert 2,720 ops/s, store-owned re-affirm 1,703 ops/s, caller-batch 2,033
ops/s — no cliff from the wider write txn.

**Who needs to know:** Dev-Head (owns fix: one-word expire fix + regression test + 2 doc items),
bosses (gate held on a NEW hole, not on a round-4 regression). Re-review on fix.

## 2026-08-06 — Critics — Consolidation v2 Stage 1, G3 ROUND 6 — **PASS-WITH-NOTES**

**Claim reviewed:** "R5-1..R5-7 resolved; targeted `session.expire(fact)` + caller-staged-edit
regression test; docstring rewritten to the real mechanism; build log records R4+R5 and supersedes
the falsified R3 N1 row; vanished-row guard covered (100%); mention_count NULL-proofed; worker
session production parity. CURRENT: 51/51, 100% line coverage, flagship 10/10, smoke green."
**Method:** read db/semantic_facts.py, db/models.py SemanticFact, db/engine.py, the full test file,
conftest, smoke, build log; re-ran every r5 harness (expire_all_probe, targeted_expire,
caller_batch2, mixed_hammer x3, lockhold, edge_probes, expire_all_cost) against the fix; scoped
pytest ONLY to tests/test_semantic_facts.py (test_e2e_claude.py never touched, $0 spent); flagship
12x isolated + full file 6x; line AND branch coverage; 5 new r6 probes in scratchpad/r6
(expire_sufficiency, legacy_null, sweep/sweep_heavy with a synchronised start barrier,
mutation_test, mutation_coalesce). Repo left byte-identical; founder's real DB read-only only
(mtime unchanged, 14,505 turns / 34,905 kg_edges / 0 semantic_facts intact).

**Verdict: PASS-WITH-NOTES — no blocker, no major. All seven round-5 minimums genuinely fixed.**

**Verified fixed, by execution:**
- R5-1: `expire_all()` is GONE from the store (repo-wide grep: the only remaining call is
  api/app.py:821 on the app's OWN long-lived session — not a caller-supplied one, out of lane).
  Blast radius measured closed: an edit staged on a Turn AND on a different SemanticFact both
  survive the re-affirm and persist. Sufficiency proven, not asserted: caller loads a stale copy →
  another OS PROCESS commits a merge → caller re-affirms via `db=` → final row langs A,B,C,
  mention 3/3, sessions s-A/s-B/s-C — the re-read sees the other process's committed merge through
  the identity map. Regression test is a REAL tripwire, mutation-tested: swapping
  `expire(fact)`→`expire_all()` in a loaded copy of the module flips
  test_reaffirm_preserves_caller_staged_edits from PASS to FAIL ('MARKED-BY-CALLER'→'ORIGINAL').
- R4-1 still holds after the change — the decisive probe re-run: 3s stall between the increment and
  the re-read while two OS processes fire at the same row → they blocked 2.86/2.87s, merged
  cleanly, final mention 4/4, all langs, all sessions. New r6 sweep with a synchronised start
  barrier: 3x50, 5x30, 8x20, 6x25 mixed store-owned/caller-batch, 4x40 all-caller-batch, 8x100
  mixed (801/801), 12x40 (481/481) — every configuration exact, 0 errors, all langs, all sessions.
  mixed_hammer 3/3 runs at 341/341, 0 errors (its "sessions 10 vs expect 11" line is a harness
  arithmetic slip — the seed reuses s-0 — not a loss; 10 distinct is correct).
- R5-2: docstring now describes the shipped mechanism and every sentence is true.
- R5-3: build log carries round-4 and round-5 records; R4's paragraph names and voids the R3 N1
  row and its "exactly 60" proof.
- R5-4: the NullPool credit is gone from tests/test_semantic_facts.py:334.
- R5-5: test_reaffirm_vanished_fact_raises covers line 309; **100% line coverage independently
  measured (257 stmts, 0 missed)** — the 51/51 and 100% claims are exact.
- R5-6: coalesce verified on the REAL legacy shape (rebuilt the table as `mention_count INTEGER`,
  inserted NULL): NULL→2→5, provenance 5. New fresh DBs get NOT NULL DEFAULT 1.
- R5-7: worker sessionmaker is autocommit=False/autoflush=False/expire_on_commit=False.
- G2 smoke reproduces the pasted artifacts exactly (93 created, 2 re-affirmed, 6 dated; pass 2
  0 created/95 re-affirmed; 93 of 93; both hot paths on captured store SQL, no temp sort).
- Import-time migration does NOT brick the founder's real DB (verifier logic replayed read-only:
  dedup unique present and non-partial, 0 required indexes missing, idx_kg_edges_active present).
- Stage 1 is self-contained: nothing in the product imports SemanticFactStore yet; models.py diff
  is confined to SemanticFact; no scope drift.

**Notes (none hold the gate):**
- N1 (docstring precision): line 284 "Only the affected fact is expired — never the caller's
  identity map" is true but incomplete. Measured residual: a caller's unflushed edit to THE SAME
  fact is still silently dropped (`extraction_model` 'CALLER-EDIT' → 'm') and the store returns
  success. That row is the one the store is contractually merging, so the behaviour is defensible —
  but say it in the docstring so Stage 2 does not assume otherwise. (A loud `session.is_modified`
  check is an option, not a demand.)
- N2 (missing tripwire): R5-6's coalesce has NO test. Mutation-tested: replacing
  `coalesce(mention_count,1)+1` with `mention_count+1` leaves the suite 51/51 GREEN while a legacy
  NULL row goes back to mention_count=None forever — and legacy IS the founder's real schema
  (`mention_count INTEGER`, nullable, no server default; confirmed read-only). ~15 lines; exact
  scenario in scratchpad/r6/legacy_null.py.
- N3 (build-log drift, 3rd of this class after R2 M5 / R4-6): the stage table row still reads
  "✅ 49/49" and "R5 pending" while the body reads 51/51 / round 6 dispatched; line 170's N1 row
  still bolds the falsified "mention_count exactly 60" without an in-place superseded marker.
- FYI, no action asked: branch coverage is 99% (5 partial arcs — 96->102, 245->247, 400->402,
  465->467, 501->503; two of them are the caller-batch error paths Stage 2 will exercise);
  `_migrate_semantic_tier` reports "verified" for a table whose mention_count column no longer
  matches the model (latent — 0 rows in the real DB, coalesce is the working defense); the two
  known Stage-2 constraints stand (15 live-session pool ceiling; a caller batch holds SQLite's
  single write lock from its first write to commit against busy_timeout=30s).

**Who needs to know:** Dev-Head (N1/N2/N3 are cheap and should land before the founder report),
bosses (Stage 1 G3 is PASSED; the gate is open). Six rounds converged: the store's correctness
claims are now proven by execution, not asserted.

---

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 1: **BLOCKED** (5 blockers, 8 majors, 8 minors)

**What I did:** independent adversarial review of the Stage-2 consolidation engine
(`llm/consolidation_v2.py`, `tests/test_consolidation_v2.py`,
`benchmarks/consolidation_v2_stage2_smoke.py`) against `CONSOLIDATION_V2_DESIGN.md` §5.2,
`CONSOLIDATION_V2_STAGE0_RESEARCH.md` §1.1/§2.2/§4/§5 and the build log's Stage-2 record.
Verified by execution, not reading: reproduced the G2 smoke exactly (42/0/6, deterministic),
dumped all 42 facts + every cited turn, ran 8 adversarial real-llama3.1 sessions, a
36k-truncation canary probe, a SIGKILL crash probe, a 2-process same-session race, and a
16-way parallel write-lock probe. 13 real Ollama sessions total, $0.

**Result: BLOCK.**

Blockers:
- B1 The Mem0 #4573 junk defense claimed in the build log ("junk rejected (Mem0 #4573 class)")
  does not exist. Pure tool-noise session → 4 facts created, 0 rejected. Flagship run: 9/42
  (21%) are assistant knowledge ("Bat Wings will be available during Mickey's Halloween
  Party"). The G1 test only exercises validator syntax rules on hand-written candidates.
- B2 `provenance()` returns `citations_intact=True` for false and EMPTY citation sets. 8
  flagship facts cite turns 84/97/105; content verifiably lives in turns 95/100-104. 19/96
  citation edges (19.8%) share no distinctive token with the cited turn; 2/42 flagship and
  11/24 adversarial facts cite nothing. Root cause: `_cited_turns` is plain >=5-char token
  overlap>=2 over USER turns only — short atomic facts have <2 such tokens and can never be
  cited. False-clean on the product's headline transparency claim.
- B3 Module docstring (consolidation_v2.py:13-24) makes 5 claims the code does not implement:
  "dense few-shot" (zero examples in `_prompt`), "every fact dated" (32/42 undated, passed),
  "digits present when the source turn had them" (no such check), "relative dates resolved
  against the SESSION date" (never validated), "Tier 2-3 embedding shortlist + batched LLM
  adjudication" (zero such code in the repo). Repeat of the R5 comments-that-lie class.
- B4 The build log's admitted wart ("'in July' emitted as 2023/07/01 point rather than month
  interval") is FALSE — the store holds t_occurred=2023/07/01 AND t_occurred_end=2023/07/31.
  The smoke prints only `f.t_occurred` (line 63), so the author read a lookalike and recorded
  a non-existent defect. Exact repeat of R2 B5.
- B5 "= the gold answer 10, enumerable by counting rows" is false: counting the 6 rows gives
  6; reaching 10 requires parsing "three times" out of two fact texts and summing. Sessions
  were oracle-selected via `q["gold_keys"]`; no noise, no retrieval, no answerer.

Majors: planned/future events stored as ordinary events (Stage-1 F7 commitment undelivered
and undisclosed; validator accepts 2099/01/01); vague-quantifier guard armed on 99.73% of
LongMemEval and measured destroying a true fact; relative-date resolution ~50% with silent
NULLs and off-by-3-days; `_session_date` reads only turn[0] (measured t_occurred 14 months
BEFORE t_mentioned); unparseable date discards the whole fact ("went bouldering twice" lost to
a date RANGE); silent 36k truncation while ConsolidationLog claims all 463 turns processed;
rejections never persisted; no dedup past exact hash (near-dup pair in the flagship itself).

**Verified genuinely good (attacked, held):** atomicity under real SIGKILL mid-batch (0 facts,
0 logs, clean re-run 12/12/1); 2-process same-session race (one wins 12 facts, loser fails
loudly, zero duplicates); 16 parallel consolidations x 30 facts = 480/480, 0 errors, slowest
0.50s — the 15-session pool ceiling and 30s busy_timeout hold with ~60x margin; extraction
genuinely outside the txn; schema-constrained decoding never malformed across 13 real sessions;
prompt injection resisted; month-only dates DO become correct intervals; 0/19,195 corpus
sessions hit the 36k cut (max 13,732).

**Refs:** llm/consolidation_v2.py:13-24,96-119,143-158,201-219; tests/test_consolidation_v2.py:99-118;
benchmarks/consolidation_v2_stage2_smoke.py:52,63; db/semantic_facts.py:624;
CONSOLIDATION_V2_BUILD_LOG.md:39,58-72; harnesses in session scratchpad (probe_smoke_full.py,
adversarial.py, trunc_probe.py, crash_conc.py, par2.py).

**Who needs to know:** Dev-Head (owns B1-B5 + M1-M8), bosses (Stage 2 G3 is BLOCKED — the stage
table row at line 39 still reads all-☐/STARTED while the record header claims G1✅G2✅, 4th
repeat of the build-log-drift class). Founder decision needed on Stage-1 F7 (planned events)
which Stage 2 silently skipped.

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 2: **BLOCK** (4 blockers, 8 majors, 9 minors)

**Claim reviewed:** "R1's B1-B5/M1-M8 resolved — USER-turn support gate rejects assistant/tool junk;
all-roles citations with adaptive thresholds; citations_intact three-state; planned events rejected;
unparseable dates drop-date-keep-fact; loud truncation; any-turn session-date stamps; per-fact digit
guard excluding date-stamps; _llm boundary test; lang_source passthrough; docstring claims only what
exists; build-log record corrected." Suites 66/66.

**Method:** read llm/consolidation_v2.py, tests/test_consolidation_v2.py, the semantic_facts diff, the
stage-2 smoke, design §5.1/§5.2, the build log. Ran: scoped pytest (66 passed, test_e2e_claude.py never
touched, $0); R1's adversarial.py (8 real llama3.1 sessions); R1's probe_smoke_full.py (4 flagship
sessions) + cite_audit.py + 2 new citation audits; trunc_probe.py; crash_conc.py; par2.py (found it now
vacuous — rewrote as par3.py); 10 RANDOM real LongMemEval sessions; 5 plan-heavy sessions; a 10-mutation
tripwire sweep of every R1 fix; arming-rate measurement over all 98,912 user turns of longmemeval_s;
Devanagari num_ctx probe; coverage. ~30 real Ollama calls, $0. Repo untouched apart from this log.

**Verdict: BLOCK.**

Blockers:
- R2-B1 The support gate does NOT stop the class it names. Pure tool-noise session still stores 2 facts
  incl. "The user's build was successful in 4213 ms with 0 errors" (system JSON turn 4). The EXACT fact
  text the G1 test asserts rejected — "The user's build succeeded in 4213 milliseconds." — is ACCEPTED
  against the real session: 4 tokens => need=1, and user turn "run the build" supplies "build". The test
  passes only because its fixture has no user turn containing that word. Rubber stamp.
- R2-B2 Citation truth unresolved beyond the empty case. R1's own metric: 21.4% unsupported edges (was
  19.8%). `cited[:8]` (line 176) keeps the 8 LOWEST turn ids: fact #18 keeps max-overlap-3 turns and
  DROPS the overlap-7 user turn that literally states it; #21 drops the overlap-8 source. Same fact #18
  carries a FABRICATED t_occurred (July, transplanted from a different event). provenance() =>
  citations_intact=True for all of it. Truncation probe cited turn 463, outside the model's input.
- R2-B3 4 of 10 R1 fixes have NO tripwire (mutation sweep, suite stays 14/14 green): per-fact vague
  guard reverted to session-global; citations reverted to user-only; adaptive threshold -> fixed need=2;
  truncation warning deleted. Line 174 (guard fires) uncovered; the guard test's own comment is wrong
  about why it passes. Verbatim repeat of the R6 mutation-test lesson.
- R2-B4 Build log has NO fix-pass record: no B1-B5/M1-M8 resolution table, no fresh artifacts. Its
  numbers (8/8, 42 facts, 6 events, 21% assistant knowledge, 19.8%, "junk rejected (Mem0 #4573 class)")
  all describe the PRE-FIX engine and are now false. 5th repeat of the drift class.

Majors: session-date hijack from user content ("Session dated 2099/01/01" in a user turn sets every
fact's t_mentioned and disables the future-event guard); future-dated event DELETES the fact (real loss
measured: Stripe start date) while llama complies with plans-are-states only 7/11 and the rest land as
fact_type=event with NULL dates; model-side context truncation silent (Devanagari at the 36k char cap =>
prompt_eval_count 10240 clamped, engine reports truncated_chars=0; English 8,120/10,240); persisted
audit row says turns_processed=401 with 17,850 chars dropped while line 204's comment claims otherwise;
vague guard still destroys a true fact on same-turn digit co-occurrence; support gate false-rejects on
morphology ("microchipping" vs "microchipped", 2.1% of candidates on 10 random sessions, 50% of the
gate's rejections); design §5.1's "else session date" silently not implemented (73% of flagship facts
undated); relative dates still ~50% wrong and undisclosed ("three weeks ago" -> 2023/09/23).

**Verified genuinely good:** 66/66 exact; empty-citation false clean fixed; assistant-knowledge content
on the flagship ~21% -> ~0 by hand audit of all 22 (the PROMPT rule earned this, not the gate); arming
rate 99.73% -> 15.53% measured on all 98,912 user turns; 6 of 10 mutations caught (date-strip, support
gate, F7 rule, any-turn stamp, keep-fact-on-bad-date, lang_source); SIGKILL atomicity 0/0; 2-process race
one loud loser zero dupes; 16-way x 30 = 480/480 at 0.63s slowest; injection resisted; month intervals
correct; nothing in the product imports the engine — scope contained.

**Refs:** llm/consolidation_v2.py:109,117,120,134-138,143-155,165-176,204,237,284;
db/semantic_facts.py:626-627; tests/test_consolidation_v2.py:81-96,99-113,142-153,155-167;
benchmarks/consolidation_v2_stage2_smoke.py:52,63; CONSOLIDATION_V2_BUILD_LOG.md:39,58-85;
CONSOLIDATION_V2_DESIGN.md:118. Harnesses: session scratchpad/s2r2/ (gate_probe, cite2, cite3,
cap_probe, mutate, randsess, plans, arming, regex_probe, ctx_hi, logrow, edge, par3+wpar).

**Who needs to know:** Dev-Head (owns all four blockers), bosses (Stage 2 G3 stays BLOCKED; the record
still shows pre-fix artifacts as current). Founder decision still open on F7 (planned events) and now
also on design §5.1's undated-fact policy.

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 3: **BLOCK** (4 blockers, 7 majors, 5 minors)

**Claim reviewed:** R2's 8 must-change items resolved — (1) numbers-in-facts support gate + faithful
tool-noise test, (2) strength-ranked citations + cap disclosure, (3) tripwires for the 4 untripwired
fixes + vague-guard branch (97% cov), (4) header-only session stamps, (5) planned events retyped not
deleted, (6) prompt_eval_count/ctx clamp reported, (7) truncated_chars + rejected_count persisted via
additive migration, (8) full R2 resolution table + fresh artifacts in the build log. Suites 75/75.

**Method:** read llm/consolidation_v2.py (full), tests/test_consolidation_v2.py (full), the
semantic_facts/models/engine diffs, the stage-2 smoke, DESIGN §5.1, the build log. Ran: scoped pytest
(75 passed, test_e2e_claude.py never touched, $0); coverage (97%, 5 lines); a 20-mutation tripwire sweep
rewritten for the current source; R1's adversarial.py (8 real llama3.1 sessions); R2's plans.py (5 real
sessions), randsess.py (10 RANDOM real LongMemEval sessions), ctx_hi, logrow, edge, cap_probe (adapted),
gate_probe (adapted); the real Stage-2 G2 smoke (4 flagship sessions); a whole-corpus stamp-position scan
(19,195 sessions, $0); a new deterministic numbers-rule repro; a new retype-vs-hash collision probe; a
restored 16-way parallel-write probe. ~30 real Ollama calls, $0. Repo untouched apart from this log.

**Verdict: BLOCK.**

Blockers:
- R3-B1 The numbers rule destroys true user facts at scale. `_STAMP_RE` (line 189) deletes EVERY ISO date
  from the user evidence before the check, so any fact repeating a date the user typed is rejected as
  "not stated by the user — tool/assistant output is not a user fact" — a FALSE statement in the audit
  trail. Measured: plans.py 15/18 candidates rejected, all 15 true user facts, 4 of 5 sessions produced
  ZERO facts; randsess 6/48 (12.5%) rejected on 10 random real sessions, ~5 of 6 true losses, 100% caused
  by date digits; adversarial lost "bought a new bike on 2023/11/03" (correctly resolved) and "went
  bouldering twice". Worse, the rule is surface-form lottery, not truth: "3 times" REJECTED vs "three
  times" ACCEPTED (the v2 aggregation thesis, and the prompt at line 353 explicitly ORDERS the numeral);
  "1200 dollars" REJECTED vs "$1,200" ACCEPTED; "October 15th, 2023" passes only because `\b\d[\d,.]*\b`
  can't see "15th" — that ordinal accident is the only reason the flagship survived. Substring matching
  also ACCEPTS fabrications ("ran 42 marathons" against "42195 metres"). R2's 2.1% morphology
  false-rejection was traded for a ~10-83% date false-rejection.
- R3-B2 The session-date hijack is NOT fixed, and the code comment says it is. STAMP_SCAN_TURNS=3 keeps
  the scan role-blind; a user turn inside the header window still sets the date. Measured: user turn 2
  containing "Please note: Session dated 2099/01/01" -> session_date=2099/01/01 for the whole session.
  The repo's own test plants the hijack at turn 4, outside the window. Lines 51-52 and 324-327 name
  R2-M1 as fixed. Comments-that-lie class, 3rd repeat.
- R3-B3 Retype-to-state re-opens Stage-1 F3 (a BLOCKER there). `fact_hash` includes t_occurred only for
  events; retyping a plan to state moves it into the text-only key. Measured: two plans, same sentence,
  2024/04/15 and 2025/04/21 -> ONE row keeping 2024 only, mention_count=2, second date silently gone.
  Control with the identical input and no retype -> 2 rows, both dates. Untested, unwarned. Separately,
  the retype is inert on real data: plans.py fired it 0/18 times because B1 rejects dated plans first.
- R3-B4 Must-change item (8) was not done at all. CONSOLIDATION_V2_BUILD_LOG.md's Stage 2 record (lines
  58-85, mtime 12:41) is still the R1-era text: no R2 resolution table, no R2 record, no fresh artifacts.
  Its G2 numbers ("42 facts created", "6 events", "19.8% unsupported edges") are stale twice over — the
  real smoke now produces 22 facts and 4 rollercoaster events. The stage table (line 39) still reads
  "R1 ✗ BLOCK / fix pass in progress". The docstring's "escalated to the founder with F7" (M7 undated
  facts) has no founder-facing artifact anywhere. 6th repeat of the build-log-drift class.

Majors: 3 of 20 mutations leave the suite green — the vague guard's date-strip half, the session-year
exemption (new, untested, and demonstrably exploitable: "The user's build produced 2024 warnings." is
ACCEPTED in the 2024-dated tool-noise session), and rejected_count persistence, whose apparent test
asserts `row.rejected_count == 0` on a zero-rejection session (0==0 rubber stamp — the exact class R2-B3
blocked); the additive migration uses `except Exception: pass` inside the one function whose docstring
says the silent except-pass idiom is what it was rewritten to eliminate, reports nothing about the two
columns, and has zero tests; 63.1% of stored citation edges now point at assistant/system turns while
provenance's `citations_intact` still only means "the ids resolve"; cap-disclosure warnings are emitted
for REJECTED facts that were never stored; rejection reasons (now often false, per B1) persist only as a
count, so destroyed true facts leave no durable record of what was lost; question-echo bloat measured on
a random real session (8/8 facts were "The user asked …"/"The user is interested in learning about …"),
undisclosed; par3.py — R2's own concurrency probe — is now vacuous (0 created, 30 rejected per worker)
because B1 rejects its synthetic facts.

**Verified genuinely good (attacked, held):** 75/75 exact, 97% engine coverage (5 lines: 145, 147,
335-336, 344); the named R2-B1 fact ("build succeeded in 4213 milliseconds") now REJECTS against the real
tool-noise session and the p1 run stores zero tool-output facts (4 candidates -> 3 rejected, 1 kept, and
that one is user-stated); citation ranking works — 16/22 flagship facts hit the cap, 0 lost their
strongest evidence (was 2), 0 facts cite a set with no user turn; cap disclosed per fact; header-only
stamps cost NOTHING on real data — all 19,195 longmemeval_s sessions carry the stamp at line index 0 on a
system line, 0 would lose it, 0 user lines in the window carry a stamp; ctx clamp surfaces end-to-end on
real Devanagari (prompt_tokens=10240, ctx_clamped=True, truncated_chars=4165); truncated_chars=17850
persisted on the audit row alongside turns_processed=401; vague guard warns and never rejects (tripwired
both ways); unparseable/non-string dates drop the date and keep the fact; 17/20 mutations caught;
16 parallel consolidations x 30 facts = 480/480 with 16 log rows at <0.2s once the probe is un-vacuumed;
malformed candidate shapes (None text, missing keys, list/int dates) never raise.

**Refs:** llm/consolidation_v2.py:51-52,75-76,163-166,172-219,187-207,189,324-346,353;
db/engine.py:268-274; db/semantic_facts.py:112-129,624-626; tests/test_consolidation_v2.py:126-152,
252-273,288-309; benchmarks/consolidation_v2_stage2_smoke.py; CONSOLIDATION_V2_BUILD_LOG.md:39,58-85;
CONSOLIDATION_V2_DESIGN.md:118. Harnesses: session scratchpad/s2r3/ (mutate 20-mutation sweep, numrule,
retype_hash, stamp_corpus, gate_probe, cap_probe, ctx_e2e, migr, why15, randsess, par4+wpar3).

**Who needs to know:** Dev-Head (owns B1-B3 + all majors), bosses (Stage 2 G3 stays BLOCKED for a third
round; the record still shows R1-era artifacts as current). Founder decisions still open: Stage-1 F7
(planned events), DESIGN §5.1's undated-fact policy — neither has reached a founder-facing document.

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 4: **BLOCK** (4 blockers, 6 majors, 6 minors)

**Claim reviewed:** R3's 4 blockers + 7 majors resolved — (B1) numbers gate rebuilt on VALUES against
raw user evidence with comma normalization, word-numeral map, own-t_occurred exclusion, session-year
exemption REMOVED, truthful rejection message; (B2) stamps accepted ONLY from system-role header lines,
user stamps skipped with a note, hijack test inside the window; (B3) no retype, two plan dates = two
rows; (B4) rounds record appended + grep-verified; (M1) 3 tripwires incl. "strip-half covered";
(M2) narrow migration except + report field; (M3) provenance user_turns_resolved; (M4) cap disclosure
only for accepted; (M5) rejections_json persisted; (M6) no-question-echo prompt rule.

**Method:** read llm/consolidation_v2.py + tests/test_consolidation_v2.py in full, the engine/models/
semantic_facts diffs, DESIGN §5.1, the whole appended rounds record. Ran (repo untouched except this log
and the lessons file; $0, ~35 local Ollama calls): scoped pytest 78/78, coverage 97% (5 lines: 149, 151,
357-358, 366); a 23-mutation sweep rewritten for the current source + a 2-mutation mini-sweep; r3's
numrule, retype_hash, gate_probe, stamp_corpus (19,195-session rescan), migr, ctx_e2e, randsess (10
random real sessions), par4/wpar3 (16 parallel), r2's plans.py (5 real sessions); the real G2 smoke;
and new probes: numhole (surface-form/exemption matrix), wordmap_corpus (whole-corpus false-satisfaction
rate), ride_e2e (tool numbers through the real pipeline into the store), depend (exemption dependency of
real accepted facts), stamp4 (role gate + note honesty), crosslingual, wordfact, reason.

**Verdict: BLOCK.**

Blockers:
- R4-B1 The word-numeral map matches SUBSTRINGS (line 207 `if word in user_low`), so ordinary English
  manufactures user-stated numbers and R2-B1 is reopened. Whole-corpus measurement: 18,673 of 98,912
  user turns (18.9%) and 9,895 of 19,195 sessions (51.5%) contain a carrier word; manufactured values
  "10" x9,940, "1" x8,939, "2" x764, "8" x551 (content/attended/potential/listening/often -> ten;
  mentioned/someone/phone/done/money -> one; network -> two; concert/concept/concerned -> once).
  End-to-end through the real pipeline (ride_e2e): user says "run the build and show me the bundle
  content", tool prints "compiled in 10 s", model emits "The user's build compiled in 10 seconds."
  -> ACCEPTED and STORED. Repeat of the logged lesson "A validator that reads surface form is a lottery,
  not a rule" (Stage 2 R3). The strict form costs nothing: every legitimate case in the suite and in the
  probes ("three times", "twice") is a whole word.
- R4-B2 The own-t_occurred-digits exclusion (line 202) reinstates the "removed" session-year exemption
  per fact, and the comment (lines 195-197) claiming those digits are "engine-derived calendar framing,
  never tool output" is false — t_occurred comes from the model's own JSON for that fact. Demonstrated
  in a 2024/01/15 session (ride_e2e): "The user's build emitted 2024 warnings." and "The user's build
  took 15 seconds." are ACCEPTED and STORED from tool output; the same facts with t_occurred=None are
  rejected. It is also a zero-padding lottery: "5 errors" is rejected on 2024/05/05 but "20 minutes" is
  accepted on 2024/05/20. NOT a delete-it fix — the exclusion is load-bearing for true facts (2 of 11
  numeric facts in the random run pass only because of it); it must be narrowed to digits belonging to a
  date-shaped substring of the fact text that normalizes to t_occurred.
- R4-B3 The stamp role gate is not what the record says. The resolution claims "system-role header lines
  only"; the code (line 345) skips only `role == "user"`, so an assistant/tool turn inside the 3-turn
  window sets the session date. Measured: assistant turn 1 = "Session dated 2099/01/01" -> session_date
  2099/01/01, note None (silent). Untripwired both ways — tightening to `role != "system"` leaves the
  suite green, and so does widening the window to the whole session. Their own corpus rescan (which I
  reproduced: 19,195 sessions, stamp at line index 0 in 100%, system role in 100%, 0 user-line stamps in
  the window) proves the strict fix is free. Repeat of "Narrowing an attack surface is not closing it,
  and the comment must not say otherwise" (Stage 2 R3).
- R4-B4 R3-B4 is only half resolved and is reported as done. The appended rounds record is real, honest
  about the two no-op writes, and matches the round history — but it contains no current artifacts, and
  the Stage 2 block above it (lines 58-85) still presents the R1-era G2 numbers as the record ("42 facts
  created", "6 events", "~21% assistant knowledge", "19.8% unsupported edges", heading "G1 ✅ 8/8, G3
  pending") with no supersession marker, while the smoke I ran produces 20 created / 4 rollercoaster
  events. Stage table line 39 still reads "75/75, 97% cov" (now 78/78). The operating contract (line 13)
  requires G2 artifacts pasted in this log; the only ones present are false. 7th repeat of the
  build-log-drift class.

Majors:
- R4-M1 The numbers gate only inspects DIGITS in the fact, so a fabricated WORD count is unchecked:
  against "I rode the coaster twice", "The user rode the coaster three times." and "... seventeen
  times." are both ACCEPTED, only "3 times" is rejected. The flagship smoke fact ("rode the Revenge of
  the Mummy rollercoaster three times in a row") therefore passes the guard untested — the count the
  aggregation thesis rests on is unverified. The module docstring (lines 19-21) claims "every NUMBER in
  the fact must appear in user-cited content"; false in both directions after B1.
- R4-M2 The support gate is ASCII-only (`_TOKEN = [a-z0-9]{3,}`, line 81), so every fact from a
  non-Latin-script session is rejected — `_tokens()` returns the empty set for Hindi/Russian/Chinese, and
  the real Devanagari e2e run produced candidates=1, created=0, rejected=1 with the reason "no supporting
  USER turn — assistant/system knowledge is not a user fact", a false statement about the user's own
  words. Canonical-English facts over a Hindi transcript fail identically. Undisclosed in the docstring's
  NOT-built list, while the operating contract names "cross-lingual canonical facts" a target distinctive
  and the roadmap has Gate E. Disclose + pin with a test now; fix later.
- R4-M3 The rejection reason is untruthful when a numeric fact has NO user turn at all: the numbers
  message fires first and line 215 suppresses the real cause. Live example from the random run: "The user
  has read about Custom Body Parts In-vivo on 2023/05/29." -> "numbers ['05','2023','29'] not found",
  when the truth is that no user turn supports it (drop the date and the same fact reports the correct
  reason). rejections_json is now the durable audit record, so this mislabels durably.
- R4-M4 The migration's swallow is not narrow and the report can lie: `except sqlite3.OperationalError:
  pass` also swallows "no such table" and "database is locked", after which the report says
  consolidation_log_columns = "verified". Measured: a DB with no consolidation_log returns "verified".
  The inline comment "column exists — the ONLY swallowed case (R3-M2)" is false.
- R4-M5 Three claimed fixes have no tripwire (23-mutation sweep, 20/23 caught): the vague guard's
  stamp-strip half — explicitly claimed as "strip-half covered" in M1 and it is NOT (the s1 fixture's
  stamp is on a system turn, so the strip is unreachable); user-side comma normalization (claimed in B1);
  cap-disclosure-only-for-accepted (claimed in M4). Plus the two in B3.
- R4-M6 provenance's `user_turns_resolved` (R3-M3) has no assertion anywhere in the suites — a revert is
  silent.

Minors: the "never silent" skip note is overwritten by the multi-stamp note when both fire (line 366);
a lowercase user "session dated" is neither used nor noted; word numerals >12 are unknown to the map, so
"fifteen"->"15" is false-rejected (rare here: ~15 corpus turns each for fifteen/hundred/million, so
minor, but the same lottery); 2 of the 5 uncovered lines (149, 151) are rejection branches (too-short
text, bad fact_type) never exercised; question-echo bloat is down but alive — 3 of 41 facts (7.3%) in the
fresh random run, 2 of them numeric dated events ("The user asked for 20/100 polyvagal-themed quotes");
datetime.utcnow() deprecation noise throughout.

**Verified genuinely good (attacked, held):** R3-B1's recall damage is really fixed — plans.py 0/19
rejections (was 15/18), randsess 2/43 (4.7%, was 12.5%) with both rejections true rejections and no false
rejection found; user-typed dates, "3 times"/"three times", "$1,200"/"1200 dollars"/"15%" all accepted;
the fabrication direction still blocked ("42 marathons" and "126585 metres" rejected against "42195
metres"). R3-B3 fixed decisively: two plan dates = two rows, no retype, per-fact "pending F7" warning,
tripwired. The user-role hijack inside the window is blocked and noted, and their corpus claim
(0/19,195 stamps outside a system-role line 0) reproduces exactly. Adversarial p1 still stores zero
tool-number facts in its own shape (4213 ms, 0 errors, and "2024 warnings" with t_occurred=None all
rejected). rejected_count/rejections_json/truncated_chars persisted and tripwired; migration adds the
three columns on a legacy DB; ctx clamp surfaces end-to-end on real Devanagari (truncated_chars=4165,
prompt_tokens=10240, ctx_clamped=True); 16 parallel consolidations x 30 facts = 480/480 with 16 log rows
in 0.92s (R3's vacuous concurrency probe is live again); smoke reproduces 20 facts / 4 rollercoaster
events, counts and citations intact; 78/78 tests, 97% coverage, 20/23 mutations caught.

**Refs:** llm/consolidation_v2.py:18-23,75-81,193-218,202,207,215,345,359-367;
db/engine.py:268-280; db/semantic_facts.py:609-631; tests/test_consolidation_v2.py:221-243,262-283;
CONSOLIDATION_V2_BUILD_LOG.md:39,58-85 + appended Stage 2 rounds record;
CONSOLIDATION_V2_DESIGN.md:118. Harnesses: session scratchpad/s2r4/ (numhole, wordmap_corpus, ride_e2e,
depend, stamp4, crosslingual, wordfact, reason, mutate4, mutate4b) plus s2r2/s2r3 re-runs.

**Who needs to know:** Dev-Head (owns B1-B4 + all majors), bosses (Stage 2 G3 stays BLOCKED for a fourth
round; the numbers gate has been the blocker in R2, R3 and R4 — the class is "surface form instead of
value", and each fix has traded one direction of error for the other). Founder decisions still open:
Stage-1 F7 (planned events) and DESIGN §5.1's undated-fact default — both now named in the build log,
neither yet in a founder-facing artifact.

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 5: **BLOCK** (3 blockers, 5 majors, 5 minors)

**Claim reviewed:** R4's 4 blockers + 6 majors resolved — (B1) whole-word numeral regexes on BOTH sides;
(M1) fact-side word numerals checked; (B2) date exemption SHAPE-scoped to date literals in the fact text;
(B3) stamps from SYSTEM-role header lines only; (M3) zero-evidence numeric facts report the true cause;
(M4) migration reports "table absent" honestly and swallows ONLY duplicate-column; (M2) unicode-aware
tokens; (M5) strip-half / comma / cap-disclosure tripwires; (M6) user_turns_resolved asserted;
(B4) supersession marker + fresh artifacts (88/88, smoke 20 facts / 4 events).

**Method:** read llm/consolidation_v2.py + tests/test_consolidation_v2.py in full, db/engine.py's
migration, db/semantic_facts.py provenance, the appended rounds record + stage table. Ran ($0 local;
~60 Ollama calls): scoped pytest 88/88 (1.8s), coverage 97% (missing 183, 185, 391-392, 400); a
31-mutation tripwire sweep (27 caught, 4 untripwired); s2r4's decisive probes re-run against the fixed
code (numhole, ride_e2e, stamp4, wordfact, reason, crosslingual, wordmap corpus rescan of all 19,195
sessions / 98,912 user turns); s2r3's numrule + randsess (SAME seed, same 10 real sessions, same 43
candidates as R4) and s2r2's plans (5 sessions); the real G2 smoke; par4 16-way parallel; and new probes:
shapes/monthyear (date-literal shape matrix), glue_e2e (glued-unit numbers, stub AND real llama3.1),
userdate + leakvol (inline-timestamp leakage on the real corpus), hindi5 (Devanagari/CJK gate behaviour),
migr5 (absent/duplicate/locked/view migration paths), repro23 (full-text reproduction of the new
false rejections).

**Verdict: BLOCK.** R4's four blockers are genuinely fixed and tripwired. The NUMBERS GATE CLASS IS NOT
FIXED — it still decides on English surface shape rather than parsed values, and this round that produced
one measured RECALL REGRESSION plus two live false-accept paths, none of them disclosed.

Blockers (all one root cause: value extraction by ad-hoc regex, no provenance):
- R5-B1 `_DATE_LITERAL_RE` (lines 90-93) MANUFACTURES a number for every "Month YYYY" (no day) date:
  `\d{1,2}` eats "20" of "2023", leaving orphan "23" that appears nowhere in the text and can never be
  user-stated. Measured on the identical randsess sample (same seed, same 43 candidates as R4):
  rejections 2/43 (both correct, R4) -> 4/43 (R5), the two new ones FALSE — "The user has been attending
  masters swimming sessions ... since mid-January 2023." and "The user completed 30 miles in 2 hours and
  15 minutes during the charity cycling event 'Pedal for a Purpose' in February 2023." Both are true user
  facts; the second is count-bearing (the aggregation thesis). Shape matrix: January/February/March/
  April/May/December + YYYY, "5 January 2023", "the 5th of October 2023", "between March 2022 and June
  2023" all manufacture. R4's prescription was "digits belonging to a date-shaped substring of the fact
  text THAT NORMALIZES TO t_occurred"; the shipped fix ignores t_occurred entirely. Repeat of the logged
  lesson "A precision fix must be measured for RECALL on the same corpus before it ships" (R3).
- R5-B2 `_NUM_TOKEN = \b\d[\d,.]*\b` (line 80) requires a word boundary AFTER the number, so any number
  glued to a unit is invisible on BOTH sides: "10s", "4213ms", "3x", "$20K", "16GB" yield NO value.
  End-to-end with the REAL model (no stubbing): tool output `{"stdout":"webpack compiled in 10s; bundle
  16GB; suite 4213ms; 3x retries"}` -> llama3.1 wrote "The bundle size is 16GB for the user's project" ->
  ACCEPTED and STORED, while its own "4213 milliseconds"/"10 seconds" phrasings were correctly rejected.
  Stub e2e stores "The user's test suite ran in 4213ms." — the R2-B1 attack fact with the space removed.
  Formatting decides; that is the surface-form lottery R2/R3/R4 each closed in one spelling only.
- R5-B3 The shape-scoping was applied to the FACT side only. `_user_number_values` (111-118) strips
  nothing, and in the REAL corpus 20,452/20,452 (100%) of user lines carry an inline
  "[YYYY/MM/DD (Day) HH:MM]" stamp, so every session pre-licenses ~5 distinct values (mean 4.9, ~3.8 of
  them in 1..60). Demonstrated: with the user turn "User: [2023/05/20 (Sat) 14:05] Please run the build",
  tool facts "20 warnings", "14 chunks" and "took 2023 ms" are ACCEPTED; the control "7 warnings" is
  rejected. This reinstates in practice the session-year exemption R3 deliberately REMOVED and adds
  day/hour/minute. Repeat of "A one-sided guard reports as if it were two-sided" (R4).

Majors:
- R5-M1 The word map stops at twelve, so word counts above it are silently UNCHECKED: against "I rode it
  twice", "... seventeen times." is ACCEPTED (measured) while "three times" is correctly rejected. The
  module docstring (18-21) still claims every number VALUE in the fact must be user-stated; the ceiling is
  disclosed nowhere (build log, design doc, docstring). Corpus prevalence is low (~200/98,912 user turns
  contain a >12 word numeral) — this is a DISCLOSURE + docstring fix, not necessarily a code fix.
- R5-M2 The unicode tokenizer does not make the gate work for Devanagari — it makes it INERT. Python's
  \w excludes combining marks, so "मैंने कल जयपुर में संगीत समारोह देखा और बहुत आनंद आया।" yields ONE
  token ('जयप'), need=1, and a pure assistant-knowledge fact ("जयपुर भारत के राजस्थान राज्य की राजधानी है...")
  is ACCEPTED on that coincidence — a false accept of the Mem0 #4573 class the gate exists to stop. CJK is
  still all-or-nothing (one 12-char token per sentence; two near-identical Chinese facts split ACCEPT/
  REJECT on punctuation). The record's "same-script Hindi sessions accept" is true and misleading;
  test_unicode_sessions_not_auto_rejected tests only the accept direction (rubber-stamp shape).
- R5-M3 4 of 31 mutations survive. Two are claimed as done in the record: R4-M4's "swallows ONLY
  duplicate-column, raises otherwise" (reverting the guard to bare `except: pass` leaves both migration
  tests green — the honest-absent half IS tripwired) and R4-M6's "user_turns_resolved asserted"
  (rewriting it to ALL cited turns passes: the test asserts `1 in ...` and a subset relation, both true
  of the mutant — direct repeat of "An assertion that matches the mutant is not a tripwire", R3). The
  other two: R4-M3's no-evidence reason has no test at all, and widening the stamp window to the whole
  session is still green (repeat from R4).
- R5-M4 Untruthful reason survives in a second shape: when a numeric fact is supported ONLY by
  assistant/system turns, line 249's suppression still reports "numbers [...] not found in user-stated
  content" instead of the real cause. rejections_json makes this durable.
- R5-M5 The 4-5 pre-licensed timestamp values and the date-literal strip are silent: no report field says
  a fact's digits were exempted or which user turn licensed them. Provenance exists for citations but not
  for the numbers decision, which is the decision this stage keeps failing.

Minors: whole-word "one" idioms still yield value 1 (1,456/98,912 user turns = 1.5%: "one of" x667,
"the one" x370, "one thing" x209) — far smaller than R4's 18.9% substring rate, but the same shape;
"May 15"/"March 5" inside a fact are stripped as date literals, so a count after a month word is
unchecked; a `consolidation_log` VIEW is reported as "table absent (create_all creates it)" (create_all
will not repair it); coverage misses 183/185 (rejection branches, R4 repeat), 391-392, 400 (the
multiple-stamp note my probe exercises but no test does); question-echo bloat persists at 2/39 stored
facts, both numeric ("The user asked for 20/100 polyvagal-themed quotes"); RUNNING_NOTES.md (the
founder-facing status doc) still carries no Stage-2 status and neither open founder decision (F7,
DESIGN 5.1) — R4 carry-over; datetime.utcnow() deprecation noise throughout.

**Verified genuinely good (attacked, held):** R4-B1 dead — the whole-corpus rescan shows the 17,632 user
turns where a substring map manufactured a value now yield nothing, and ride_e2e's 7 tool facts
(10/1/8/2/4213/2024/15) are ALL rejected with 0 stored. R4-B2 dead in the direction it was raised —
"2024 warnings" rejects with and without the model's stamp, "20 minutes"/"5 errors" reject on matching
session dates, the zero-padding lottery in that direction is gone, and a real date-literal fact still
accepts. R4-B3 decisive — assistant, tool and user stamps are all refused inside the window and noted
(notes now combine; lowercase noted). R4-M3's headline case now reports "no supporting USER turn".
R4-M4's absent/duplicate/locked paths are honest (locked DB raises RuntimeError, never "verified").
R4-B4 real — the R1-era block carries a supersession marker, the stage table reads 88/88, and both fresh
artifacts reproduce exactly: 88/88 tests in 1.8s, smoke 20 created / 4 rollercoaster events
(3+1+3+3 = the gold 10) with citations intact=True. plans 0/19 rejections, no recall damage there.
27/31 mutations caught. 16 parallel consolidations x 30 facts = 480/480, 16 log rows, 0.93s.

**Refs:** llm/consolidation_v2.py:18-23,75-78,80,90-93,97-118,236-252,370-404;
db/engine.py:270-288; db/semantic_facts.py:625; tests/test_consolidation_v2.py:262-294,296-318,
320-339,398-413,416-430,433-451; CONSOLIDATION_V2_BUILD_LOG.md:39,58,320-334.
Harnesses: session scratchpad/s2r5/ (shapes, monthyear, glue_e2e, userdate, leakvol, hindi5, migr5,
repro23, corpus5, mutate5/5b/5c, mini_mut) + s2r4/s2r3/s2r2 re-runs.

**Who needs to know:** Dev-Head (owns B1-B3 + all majors), bosses (Stage 2 stays BLOCKED for a fifth
round; the numbers gate has now been the blocker in R2, R3, R4 and R5 — five rounds of spelling patches
on the same decision. The exit is to parse numeric MENTIONS once, on both sides, with span provenance:
one tokenizer that sees digits, glued units, decimals and word numerals; normalization (commas, padding);
classification of each mention as date-component vs quantity by parsing the enclosing date expression and
matching it against t_occurred; then compare VALUE sets with date components excluded on BOTH sides. And
the recall measurement on the same 10-session sample must be a REQUIRED artifact in the record before any
future gate change ships — it is what would have caught B1 today). Founder decisions still open: Stage-1
F7 and DESIGN 5.1's undated-fact default.

## 2026-08-06 — Critics — Consolidation v2 STAGE 2 G3 ROUND 6: **PASS-WITH-NOTES** (0 blockers, 2 majors, 11 minors)

**Claim reviewed:** R5's prescribed CLASS exit (ONE numeric-mention parser, `_quantity_values`, with span
provenance on both sides) is implemented as specified; recall artifact honored; 96/96.

**Verdict: the class is genuinely exited.** R2-R5 all blocked on the numbers gate; R6 does not. Verified by
execution, not by reading the record:

- **R5's three blockers are dead.** Orphan digits: the two facts R5 measured as FALSE rejections ("masters
  swimming …", "30 miles in 2 hours 15 minutes") are now ACCEPTED on the same sample. Glued units: the
  real-model 16GB attack, 4213ms, 10s and 3x are all rejected with 0 stored (part 2 run against live
  llama3.1, not a stub). Stamp licensing: inline `[YYYY/MM/DD …]` stripped before licensing; "20 warnings"/
  "14 chunks" class rejects.
- **Required recall artifact reproduced INDEPENDENTLY by me, exactly: 43 candidates, 2 rejected (4.7%),
  both TRUE rejections** (assistant knowledge + a prompt-meta artifact).
- **New independent generalization probe (the highest-value check this round): 30 real sessions, NEW seed
  31337, NEW length window — off the sample the team tuned and disclosed against. 156 candidates, 4
  rejected (2.6%), ALL FOUR TRUE**, including the textbook case the gate exists for: "The user got 5 GB of
  free storage with iCloud" (vendor number the user never stated) and "considering a 50 GB, 200 GB, or 1 TB
  plan". Zero false rejections. The fix generalizes; it was not tuned to R5's probe.
- **Licensing audit is truthful at scale:** every one of 37 licensed values across both samples resolves to
  a real, same-session, role=user turn that genuinely contains the value after stamp-stripping — 0
  violations, 0 ghost ids, 0 empty turn lists. All 22 accepted numeric facts read as user-grounded.
- **Suites/artifacts reproduce:** 96/96 in 3.4s; 98% line coverage on llm/consolidation_v2.py; smoke 20
  created / 4 rollercoaster events (3+1+3+3) citations intact=True; plans 0/19 (no recall regression);
  migration honest on absent/duplicate/view/locked (locked RAISES RuntimeError); Hindi gate functions both
  directions; ride_e2e 7/7 tool facts rejected.
- **Mutation sweep rebuilt for the NEW subsystem: 17/20 caught** (trailing-\b revert, month-year alternative
  removal, alternation reorder, comma normalization, unconditional exemption, word-numeral removal,
  substring revert, stamp-strip revert, gate disabled, all-roles licensing, true-reason revert, support
  requirement removed, audit removal, stamp window, system-role-only, citation ranking, adaptive threshold).
- **Every adversarial shape I invented measured ZERO prevalence in 193 real stored facts** — abbreviated
  months, US m/d/y, copied inline stamps, clock times, digit-adjacent-to-month, >twelve word numerals, and
  "one"/"once" in numeric facts are all 0.0%. I am NOT elevating any of them to a blocker; they are
  constructively reachable but corpus-absent. Recorded here so a future round does not re-litigate them.

**M1 (major, test quality) — the R5-M2 tokenizer fix is UNTRIPWIRED; a revert is silently green.** Reverting
`_TOKEN` to `\w{3,}` leaves all 44 tests passing, including
`test_devanagari_gate_functions_both_directions`. The fix is real and load-bearing (`\w{3,}` shreds
"उपयोगकर्ता ने पिछले हफ्ते तीन बार रोलरकोस्टर की सवारी की।" into 3 garbage fragments
['उपय','गकर','लरक'] vs 8 whole words). The test passes for the WRONG reason: under the broken tokenizer the
fact drops to 3 tokens, which trips `need = 1 if len(ftoks) <= 4`, and a single garbage fragment ('जयप')
satisfies overlap. This is the R2 "fixes untripwired" class and R5-M3's own tripwire requirement. Needs a
discriminating assertion on `_tokens()` output (whole words present) or on the overlap count, not on
created/rejected.

**M2 (major, undisclosed failure mode) — a concurrent same-text consolidation aborts an ENTIRE session's
batch.** By deliberate Stage-1 design (db/semantic_facts.py:248-251, "Caller owns the transaction … Surface
it"), a dedup race re-raises when the caller owns the txn — which is exactly Stage 2's batch path. Measured:
6 threads consolidating identical fact text → 3 IntegrityErrors, and each loser's whole consolidation rolls
back (0 facts, 0 ConsolidationLog row). Data is NOT corrupted and it is LOUD (1 row, mention_count=3), so
this is not a blocker — but the module docstring says "one atomic caller-owned batch" and Stage 2's record
says nothing about it. Disclose in the Stage-2 record; recommended follow-up is an orchestrator-level catch
of the dedup IntegrityError with one batch retry.

**Minors:** (m1) FALSE COMMENT, repeat of R3-N6 — consolidation_v2.py:230 "Now: retype to state." is
directly contradicted by lines 233-236 which say retyping was reverted; it sits above the code it
misdescribes. (m2) FALSE DOCSTRING, repeat of R1 — `evaluate_fact`'s docstring documents a 6-tuple and it
returns 7 (number_audit); module docstring header still reads "G3 R1+R2 corrected" at R5. (m3) DEAD CODE,
repeat of R3-N9 — `_NUM_TOKEN` (line 82) is defined and never used, the pre-R5 parser's remains sitting
beside the live one. (m4) two more mutations survived: removing zero-pad normalization, and loosening the
date-span CONTAINMENT test (`a <= start and end <= b`) to overlap — containment is the parser's soundness
core and has no tripwire. (m5) UNDISCLOSED RESIDUAL: the gate grounds VALUES, not PREDICATES — "The user has
asked 38 questions." was accepted because the user typed "19 Question (19/38)"; the number is user-stated,
the claim is wrong (1/22 accepted numeric facts). The count thesis must not be read as predicate-level.
(m6) residual mis-stated: '"one" idiom … accept-direction only' is wrong — the reject direction exists
("The user owns one dog named Max" ← "my dog Max" rejects); magnitude fine (0/193), drop the qualifier.
(m7) UNDISCLOSED: the support gate is structurally inert for non-space-delimited scripts — Chinese collapses
to ONE token; reject-direction so safe, but "cross-lingual canonical facts" is a claimed target distinctive,
so disclose before any cross-lingual claim. (m8) SILENT FALLBACK in a function built to be non-silent —
`_session_date` swallows a malformed system stamp (lines 436-437 `except ValueError: pass`) and falls back
to the DB timestamp with NO note; the `late` scan also lacks the re.IGNORECASE the user-stamp scan has.
(m9) 5 uncovered lines: two rejection reasons ("text too short", "bad fact_type") ship untested, plus the
malformed-stamp swallow and the multiple-header-stamp note. (m10) malformed model payloads (`facts` as
str/dict, item null/str) raise AttributeError — verified loud, pre-transaction, 0 rows and 0 logs, so
correct, but the error is ugly. (m11) two overlapping stamp regexes (`_INLINE_STAMP_RE`, `_STAMP_RE`) with
different shapes used in different places — a future divergence trap.

**Confirmed live, already disclosed:** relative-date resolution is wrong in real output — user said "on the
15th … last month" with session date 2023/11/04; stored t_occurred=2023/11/15, truth 2023/10/15. Also
confirmed accurate: word map ends at twelve (0.17% of sessions carry a >twelve word numeral — the disclosed
~0.2% is honest), and date digits in fact text are never grounded against user evidence (by design).
Founder decisions still open and unresolved: Stage-1 F7 (planned-event storage) and DESIGN §5.1
(undated-fact default).

**Refs:** llm/consolidation_v2.py:82,86,98-115,118-146,228-236,264-292,415-449;
db/semantic_facts.py:232-266; tests/test_consolidation_v2.py:398-413,508-530;
CONSOLIDATION_V2_BUILD_LOG.md (Stage 2 G3 rounds record, R5 entry + residuals list).
Harnesses: session scratchpad/s2r6/ (holehunt6, prev6, prev6b, fresh30, audit_check, mut6, robust6) +
re-runs of s2r5/s2r4/s2r3 (glue_e2e, ride_e2e, randsess, plans, depend, numhole, stamp4, reason, wordfact,
crosslingual, hindi5, migr5, wordmap_corpus).

**Who needs to know:** Dev-Head (M1 + M2 + m1-m11; M1 and m1/m2/m3 are REPEATS of already-logged classes —
untripwired fix, false comment, false docstring, dead code — fix before Stage 3 opens). Bosses: **Stage 2
G3 PASSES with notes; the numbers gate is closed as a class after five rounds.** None of the notes is a
correctness hole in shipped behavior; M1 is test quality, M2 is disclosure. Founder: F7 and DESIGN §5.1
still need decisions, and the two undisclosed residuals (value-not-predicate grounding; CJK inertness)
should reach the Stage-2 record before any claim is written from it.

## 2026-08-06 — Critics — Consolidation v2 Stage 3 (KG integration), G3 ROUND 1 — **BLOCK**

**Claim reviewed:** "Stage 3 shipped: fact→entity linking, event_status (F7), NER refactor, migration;
G1 146/146 across 4 files, 100% coverage on db/fact_entities.py + db/semantic_facts.py; G2 smoke green."
**Method:** read every changed file + the new one; ran the scoped suite (test_fact_entities,
test_consolidation_v2, test_semantic_facts, test_temporal_kg — NEVER a broad pytest); line + branch
coverage; 16-mutation sweep against a scratch COPY of the repo (repo never touched); 10 probes in
scratchpad/s3r1 (p1_sweep, p2_index, p3_merge, p4_lock, p5_danda, p6/p6b_scope, p8/p8b_identity,
p10_reaffirm, pa_lock, pd_status, ppoison, prace, pprev, ph_cost); read-only inspection of the
founder's dev DB; RE-RAN the real G2 smoke end to end (llama3.1 + spaCy + e5-small).

**Reproduced honestly (state plainly):** 146/146 (33/48/60/5 — exactly the claimed split); 100% LINE
coverage on both files; G2 smoke reproduces to the digit — 20 facts, 25 links, 18 nodes, **25 real join
rows**, log rows 25, link_failure None, गूगल admitted at cosine 0.9506, facts_for_entity('गूगल') ==
facts_for_entity('Google') on that fixture. The caller-batch precondition is REAL: measured that a
RE-AFFIRMED add_fact takes the write lock before linking (outside writer free before, blocked after).
Migration backfill CASE vs `_event_status` agree on 10/10 interval forms (0 mismatches). Mutation sweep
14/16 caught. The smoke does not import db/engine (no production-DB side effect) — checked explicitly.

**Verdict: BLOCK — 3 blockers, 5 majors, 13 minors.**

- **B1 (blocker) — ~87–89 s of DB-WIDE write lock inside the caller batch, violating the file's own
  stated rule.** `_link_in_session` calls `_plan_surfaces` (fact_entities.py:181) AFTER add_fact has
  taken the SQLite write lock; the first Indic surface hits `resolver.enabled` (fact_entities.py:247),
  which LOADS sentence-transformers (network calls to HF included). Measured, fresh process: model load
  87.29 s / 87.07 s; p4_lock at 1,001 nodes = **89.09 s of held lock**, and the competing writer FAILED
  with "database is locked". busy_timeout is 30 s, so every other writer in the system dies.
  consolidation_v2.py:389-391 states the rule ("no model compute while any write lock is held that this
  loop controls") and fact_entities.py:178-180 acknowledges the hazard and then does it anyway.
  Embedding itself is cheap (0.7 ms/text; ~8 s at the founder's 10,911 nodes) — the load is the killer.
  The store-owned path is CLEAN (planning happens before any write); only the caller-batch path is hit.
- **B2 (blocker) — the CO_OCCURS weight merge is inert on production-shaped rows; its test is a rubber
  stamp.** engine.py:405-418 filters `relation_type IS NULL`. Real edges carry 'CO_OCCURS':
  **34,905/34,905 in the founder's dev DB, 0 NULL.** p3_merge, same duplicate group, both shapes:
  production shape → `co_occurs_edges_merged: 0`, two rows survive, the undirected loader keeps 4.0
  instead of 6.0 (silent weight loss); NULL shape → merged to 6.0. The only test
  (tests/test_fact_entities.py:507-511) inserts NULL because its `_OLD_SCHEMA` kg_edges has no default.
  The build log records "CO_OCCURS weights SUMMED … leaving two rows would silently drop weight" as a
  PINNED failure path — that protection does not exist for real data. (Context: the dev DB already holds
  1,979 duplicate CO_OCCURS pairs.) Stage-2 R2 lesson repeat: fixture cannot produce the failure.
- **B3 (blocker) — link_missing starves permanently and its documented drain loop never terminates.**
  fact_entities.py:436-443 selects zero-join facts `ORDER BY id LIMIT n`; a fact whose NER finds nothing
  never gains a join row, so it is a permanent candidate. Measured on REAL G2 output: **5 of 23 facts
  (22%) have zero extractable surfaces.** p1_sweep with 12 such facts + 1 linkable one at limit=10:
  4 consecutive runs return swept=10, links_created=0, and the linkable fact is NEVER reached.
  Docstring line 428 says "rerun until swept=0 to drain" — swept never reaches 0. The module's claim
  (lines 32-35) that link_missing "makes the recoverability claim true" is false at scale.

- **M1 (major) — the migration verifies kg_nodes uniqueness by INDEX NAME ONLY**, contradicting its own
  docstring ("verify by INSPECTION (PRAGMA index columns, never index names alone)", engine.py:437-440).
  engine.py:525-526 matches r[1] == "uq_kg_nodes_scope_text". p2_index: a DB carrying
  `CREATE UNIQUE INDEX uq_kg_nodes_scope_text ON kg_nodes(entity_text)` (scope-BLIND) reports
  `kg_nodes_unique: verified` while agent-B's "Google" is permanently rejected. Same class as G3 R2-B2,
  which this very docstring cites. Fix: read `sqlite_master.sql` (functional indexes have NULL columns
  in PRAGMA index_info, which is exactly why name-matching was chosen — say so and check the SQL text).
- **M2 (major) — the persisted audit row cannot distinguish "linking suspended by failure" from a clean
  re-affirmation.** `entities_linked` counts only NEW join rows; link_failure is NOT a column. p10:
  run 2 of the same session logs `summaries_generated=0, entities_linked=0, link_failure=None` — byte
  identical to a run where linking blew up on fact 1. consolidation_v2.py:418-419 claims "the count
  mismatch is visible in the log row"; there is no expected count in the row to mismatch against.
- **M3 (major) — one-hop ALIAS_OF traversal breaks the cross-lingual unity claim with 2+ Indic
  variants.** Real resolver, p8b: node "Chennai"; the tokenizer produces both "चेन्नई।" (sentence-final)
  and "चेन्नई"; the second aliases to the FIRST (cos 0.9715 > 0.9324 to Chennai), forming a CHAIN.
  `facts_for_entity('Chennai')` then returns 1 of the 2 Hindi facts — silently. fact_entities.py:361-372
  expands exactly one hop. The build log's "facts_for_entity('गूगल') == facts_for_entity('Google') ==
  all 3 facts — the cross-lingual unity claim, demonstrated" holds only for the single-variant fixture.
- **M4 (major) — node identity is case-SENSITIVE while surface dedup is casefold; undisclosed in the new
  read API.** Measured on the founder's real KG: **164 of 10,747 groups (1.5%) are case variants**
  ("The Big Island"/"the Big Island", "Turkey"/"turkey"). p8: two facts, surfaces "Google"/"google" →
  two nodes, `facts_for_entity('Google')` returns 1 of 2; `facts_for_entity('google')` returns [].
- **M5 (major) — the F7 'planned' marker is materially unreachable through the real pipeline, and the
  disclosed boundary understates it.** The extraction prompt (consolidation_v2.py:519) orders PLANS to
  be typed "state", **and its own example carries a DATE** ("The user plans to attend X on DATE" is a
  state, never an event). The record discloses only "…with no date extracts as an undated state". In the
  reproduced G2 run: **0 of 23 facts carry 'planned'** (7 events, all 'occurred'), while 3 plan-facts
  ("planning to serve", "plans to use", "wants to buy") were typed state → no marker. The marker fires
  only when the model DISOBEYS the prompt. The founder must be told this plainly.

**Minors:** (m1) U+0964/0965 danda and Devanagari digits are inside `_INDIC_TOKEN_RE` (0900–0D7F), so
"चेन्नई।"/"है।"/"२०२३" are surfaces; measured cosine cost 0.9506→0.9398 (Google) and 0.9543→0.9324
(Chennai) — 22–43% of the τ margin — and it manufactures the duplicate node behind M3. Turn-path parity
is genuine (both share the defect). (m2) FALSE PARITY COMMENT: fact_entities.py:254-256 says same-batch
surfaces never anchor "same as the turn path", but knowledge_graph.py:528 DOES add same-turn non-Indic
NER texts to the candidate pool — the fact path is strictly narrower (recall loss in the first
cross-lingual session; the G2 smoke had to consolidate an English anchor session first). (m3) TWO
UNTRIPWIRED behaviors: deleting `.nullslast()` (fact_entities.py:391) and flipping `if link_failure is
None` (consolidation_v2.py:422) to `if True` both leave the suite GREEN — the failure POLICY of record
has no test. (m4) the cross-process race test does not observe a collision: with a TIGHTER two-phase
barrier the retry path fired 1/10 (prace); the shipped test's barrier is looser and its assertions hold
whether or not a race occurs. End state was correct 10/10 — the mechanism is fine, the record's "real
2-OS-process race … pinned" is not what the test pins. (m5) dead `self._nlp = None`
(knowledge_graph.py:264) after the module-level refactor — REPEAT of the logged "delete the mechanism's
vocabulary too" class. (m6) the DB-poisoned batch raises PendingRollbackError at the NEXT add_fact, not
"at commit" as the comment says, and the captured link_failure string is discarded on that path (0
facts, 0 links, 0 log rows — verified loud, ppoison). (m7) `_merge_duplicate_kg_nodes` never re-points
semantic_fact_entities and runs on a raw sqlite3 conn with FK enforcement OFF → orphan link rows if ever
reached with links present; low reachability, undisclosed in the docstring's enumeration. (m8) reversed
(a,b)/(b,a) CO_OCCURS pairs are never merged although the loader is undirected — **0 reversed pairs in
the real DB, corpus-absent, recorded so a future round does not re-litigate it.** (m9) "100% coverage"
is LINE coverage; branch coverage is 99% (1 partial in fact_entities: 116->115; 4 in semantic_facts) —
say which. (m10) `SemanticFact.entities` display cache stores SKIPPED surfaces that were never linked
(उपयोगकर्ता/में/काम), so the JSON cache and the join table disagree. (m11) facts_for_entity materializes
every fact_id into an IN(...) list before applying `limit`; this build tolerates ≥200k variables, a
stock SQLITE_MAX_VARIABLE_NUMBER=999 build breaks at 999 facts per entity. (m12) "log rows agree 25=25"
is tautological (both sides are the same in-process variable); the load-bearing number printed on the
same line — 25 REAL join rows — is the one to cite. (m13) the smoke uses legacy `Query.get()` (two
deprecation warnings inside the evidence artifact).

**Must change to pass:** B1 (move resolver warm-up/planning outside the held write lock — or refuse the
Indic path inside a caller batch), B2 (match CO_OCCURS by value, not NULL; rebuild the test fixture on
production-shaped rows), B3 (bound/advance the sweep — cursor by id, or mark swept facts — and fix the
"rerun until swept=0" instruction), M1 (verify the index by its SQL text), M2 (persist link_failure or
an expected count), M3 (transitive closure or disclose one-hop in all three places that claim
cross-lingual unity: fact_entities.py:13-17, models.py:253-255, DESIGN correction 3), M4 (normalize or
disclose), M5 (correct the founder-facing boundary + report 0/23). Minors m2/m3/m5/m9/m12 are repeats of
logged classes and should land in the same round.

**Refs:** db/fact_entities.py:4-5,13-17,32-35,178-181,231,247,250-258,254-256,361-372,391,428,436-443;
db/engine.py:383-386,405-418,437-440,520-542; llm/consolidation_v2.py:389-391,414-422,519;
db/knowledge_graph.py:264,528; db/models.py:253-255,337-350;
tests/test_fact_entities.py:396-425,436-440,499-536; CONSOLIDATION_V2_BUILD_LOG.md:411-455.
Harnesses: scratchpad/s3r1/ (boot, p1_sweep, p2_index, p3_merge, p4_lock, p5_danda, p6_scope, p6b_scope,
p8_identity, p8b_chain, p10_reaffirm, pa_lock, pd_status, ppoison, prace, pprev, ph_cost, sweep/1/3,
runmut, smoke_rerun.log).

**Who needs to know:** Dev-Head (all findings; B1-B3 + M1-M5 before the gate reopens; m2/m3/m5 are
repeats of already-logged classes). Bosses: **Stage 3 G3 round 1 BLOCKS** — the G1/G2 numbers are honest
and reproduce exactly, but three shipped mechanisms do not do what the record says they do. Founder:
F7's 'planned' marker fired 0/23 on the real corpus because the prompt types plans as states — a
founder-requested feature is effectively dormant; and the cross-lingual unity claim needs the one-hop
caveat before it appears in any public claim.

## 2026-08-06 — Critics — Consolidation v2 Stage 3 (KG integration), G3 ROUND 2 — **BLOCK**

**Claim reviewed:** "R1's 3 blockers + 5 majors + minors all fixed; 156 tests green across the 4 scoped
files; the migration now REPAIRS data (global CO_OCCURS pair merge, weights summed, no information
lost)."
**Method (mutate the fixes, not the feature):** re-read every changed file; reran the scoped suite and
line+branch coverage; 6 probe harnesses in scratchpad/s3r2 (p_lock, p_read, p_fk, p_mig, p_rev, p_misc,
p_loader) including an instrumented WRITE-LOCK probe that attempts a real competing write from an
independent connection at every resolver touch; ran the migration against a COPY of the founder's dev DB
(source opened read-only, never mutated); mentally reverted each R1-repro test.

**Reproduced honestly — the fixes that hold.** B1 is CLOSED and measured: on the engine path all 6
resolver touches report the competing writer FREE, the store-owned sweep the same, and the in-batch
control reports BLOCKED (probe is sensitive). B2 is CLOSED and the repair is real on the founder's data:
dev-DB copy 34,905 → 32,194 CO_OCCURS rows, sum(weight) 35,051.0 → 35,051.0 (exact), loader-visible
total 32,340 → 35,051 (**2,711 units of weight recovered**), idempotent (rerun 0), 3.99 s. B3 CLOSED
(cursor terminates; the repro genuinely fails if reverted). M2 CLOSED (persisted + a real suspension
tripwire). M3 CLOSED inside the depth cap; no cross-agent fact leak (scope_key backstop measured). M5
disclosed plainly. **156/156 reproduced**; coverage reproduced: 100% LINE / 99% branch (fact_entities
partials 125->124, 436->451) — the restated claim is accurate. Minors verified: danda excluded,
false-parity comment corrected, nullslast + suspension tripwires real, dead `self._nlp` gone, smoke's
three-way assert present.

**Verdict: BLOCK — 1 blocker, 5 majors, 14 minors.**

- **B1 (blocker) — the M4 fix REGRESSED exact-text reads.** fact_entities.py:416-422 replaced the exact
  seed predicate with `func.lower(entity_text) == entity_text.lower()`. SQLite's `lower()` folds ASCII
  ONLY; Python's folds Unicode. Measured on the founder's KG: **11 distinct entity_texts** where the two
  disagree (`Übermensch`, `Συγνώμη`, `IDÅSEN`, `Ruben Östlund`, `the Champs-Élysées`, …). Probe p_read
  P1: node stored EXACTLY as the query text, real link row → `facts_for_entity('Übermensch')` returns
  **[] on 4/4**. A read that previously worked now silently answers "no facts about this entity" — a
  false-empty in the deliverable's read API. Fix: OR the exact predicate with the folded one (keeps the
  index usable for the exact leg) and disclose that folding is ASCII-only.

- **M1 (major) — plan→apply admits an UNGATED Indic node.** `_link_in_session` (fact_entities.py:238-257)
  creates the surface node FIRST and only then looks for the anchor. p_read P3: anchor deleted between
  plan and apply → node created, no ALIAS_OF edge, no skip recorded. That falsifies the invariant
  `_plan_surfaces` relies on (lines 310-313: "every stored Indic node already passed this gate") — the
  junk node then becomes an anchor candidate. Reachable: `POST /demo/reset` (api/app.py:813) purges every
  agent_id-NULL node. P4: the same happens when a plan computed for agent A is applied with
  `agent_id='Z'` — the plan carries no scope and apply validates nothing. Fix: resolve the anchor before
  creating the node for `via=='alias'`; skip + record when absent.
- **M2 (major) — the index check is a SUBSTRING test, not a shape test** (engine.py:583-593). p_mig:
  `…ON kg_nodes(coalesce(agent_id,''), entity_text) WHERE agent_id IS NOT NULL` → report **"verified"**,
  two NULL-scope duplicates INSERT fine; `…(coalesce(agent_id,''), entity_text, id)` → **"verified"**,
  duplicates allowed in every scope. Same class as R1-M1, and the new test pins only the one impostor I
  named. Fix: reuse the partial-index flag this file already reads (PRAGMA index_list r[4], line 549) +
  index_info column count, or compare the normalized DDL for EQUALITY.
- **M3 (major) — `_merge_duplicate_kg_nodes` manufactures the shape its own repair cannot see.**
  engine.py:399-402 re-points edges without re-ordering src<tgt; since keeper = min(id), an edge
  (X, loser) with X > keeper becomes REVERSED — the common case. p_rev: nodes 1/7 dup + edges (1,3,2.0),
  (3,7,5.0) → after migration (1,3,2.0) and (3,1,5.0), report `co_occurs_edges_merged: 0` and
  `co_occurs_dedup: 'clean'`, loader keeps 5.0 of a true 7.0. The fixture (tests/test_fact_entities.py:
  548-563) is arranged so re-pointing never inverts — third instance of "the fixture cannot produce the
  failure" in this arc. Fix: normalize orientation on re-point; group both merges by (min,max).
- **M4 (major) — Stage 3's new FK breaks an existing product endpoint.**
  `semantic_fact_entities.node_id → kg_nodes.id` with foreign_keys=ON: p_fk reproduces `POST /demo/reset`
  (api/app.py:807-813) failing with `IntegrityError: FOREIGN KEY constraint failed` → HTTP 500, once any
  global-scope fact is linked. Latent only because nothing wires Stage 2/3 into api/cli/mcp yet —
  certain the moment it is wired.
- **M5 (major) — the B1 guard is one-sided.** `plan_surfaces(…, db=<caller session>)`
  (fact_entities.py:131-154) will load the model under a caller's write lock; only its docstring says
  don't. `link_fact` raises a loud ValueError for the same mistake. Zero in-repo callers pass db= —
  close the door or make it loud; no test would catch the regression.

**Minors:** (m1) closure depth cap truncates SILENTLY — 14-node chain returns 11/14 facts, no log
(fact_entities.py:436). (m2) `_merge_duplicate_kg_nodes` still never re-points semantic_fact_entities and
runs with FK enforcement OFF (R1 m7 unfixed, still not in the docstring's enumeration). (m3) "no
information lost" is over-stated: the keeper keeps the EARLIEST last_updated (all 1,979 dev groups
differ) and drops later rows' session_id — verified inert (neither column has any reader), so say that
instead. (m4) Devanagari digits (U+0966-096F) are still surfaces (entity_aliases.py:53) — R1 m1 named
them alongside the danda. (m5) the build log's last line promises "Smoke re-run against the fixed engine
below"; the file ENDS there (527 lines) and no smoke artifact exists in the repo — paste it or drop the
claim. (m6) the 3.99 s repair runs inside ONE write transaction at IMPORT time on a raw sqlite3
connection with the 5.0 s DEFAULT timeout while app connections use busy_timeout=30000 — a competing
writer holding the lock >5 s turns `import agentmem_os.db.engine` into a RuntimeError. (m7) the in-memory
loader collapses CO_OCCURS and ALIAS_OF on the same pair (measured: w=12.0 replaced by w=1.0) and
ingest_turn creates BOTH for every alias pair — same "add_edge overwrites" mechanism the repair exists
for. (m8) `linked_via` accepts any string ('TYPO' persisted) though models.py documents 'ner'|'alias'.
(m9) "visits every unlinked fact exactly once" is per DRAIN: SemanticFact.id is INTEGER PRIMARY KEY
without AUTOINCREMENT, so a reused rowid below the cursor waits for the next drain. (m10) R1 m6 unfixed
— consolidation_v2.py:440-442 still says a poisoned batch raises "at commit"; it raises at the next
add_fact. (m11) R1 m10 unfixed — the display cache still stores SKIPPED surfaces (उपयोगकर्ता/में/काम/
करता/है measured again). (m12) R1 m11 now also applies to the closure's node_ids IN(...) list. (m13)
RUNNING_NOTES.md (the founder-facing status doc) still has no Stage-3 entry — the two things the founder
must know before the next product start (the 1,979-pair repair fires at import; 'planned' fired 0/23)
live only in the build log. (m14) nothing in cli/, api/, mcp_server/ calls link_missing or
facts_for_entity — the recovery drain and read API are Python-only entry points today.

**Test quality:** the R1-repro tests are genuine, not rubber stamps — reverting the cursor makes
test_sweep_cursor… hit its `rounds < 10` guard; reverting closure→one-hop makes the चेन्नई chain test
fail on the middle node's fact; reverting the merge to IS-NULL-only makes the repair report "clean"
instead of "merged 1 duplicate pairs". The two new tripwires (nullslast, suspension `calls["n"] == 1`)
both fail when their mechanism is removed.

**Must change to pass:** B1 (restore exact match — OR it with the fold — and disclose ASCII-only
folding), M1 (anchor-before-node for 'alias' plan entries; validate/stamp plan scope), M2 (verify the
index by shape: partial flag + column count, or DDL equality), M3 (orient re-pointed edges; group merges
by unordered pair; fixture where re-pointing inverts), M4 (clean semantic_fact_entities in
/demo/reset — or any kg_nodes purge — before deleting nodes), M5 (guard or remove `plan_surfaces(db=)`).
Minors m2/m3/m5/m10/m11/m12 are repeats or carried-over R1 items and should land in the same round.

**Refs:** db/fact_entities.py:131-154,238-257,310-313,416-422,436; db/engine.py:399-425,441-469,
583-593; db/entity_aliases.py:53; llm/consolidation_v2.py:409-411,440-442; api/app.py:807-813;
tests/test_fact_entities.py:548-563; CONSOLIDATION_V2_BUILD_LOG.md:457-527. Harnesses:
scratchpad/s3r2/ (p_lock, p_read, p_fk, p_mig, p_rev, p_misc, p_loader, suite.log, cov.log).

**Who needs to know:** Dev-Head (B1 + M1-M5 before the gate reopens; the repeats are cited). Bosses:
**Stage 3 G3 round 2 BLOCKS**, but the shape of the round changed — R1's three blockers are genuinely
closed and the CO_OCCURS repair is measurably real on the founder's data (2,711 units of weight
recovered). The single blocker is a REGRESSION introduced by an R1 fix, not an unfixed R1 finding.
Founder: nothing new you must decide; the next product start still repairs 1,979 duplicate pairs (~4 s,
verified on a copy, weight-conserving), and that note belongs in RUNNING_NOTES.md, not only the build log.

## 2026-08-06 — Critics — Consolidation v2 Stage 3 (KG integration), G3 ROUND 3 — **BLOCK**

**Claim reviewed:** "R2's blocker + 5 majors + 14 minors all fixed; 167 tests green across the 4 scoped
files; db/fact_entities.py 100% line / 99% branch; post-R2 smoke identical to the digit; the stray-file
process incident is repaired and the build log reads coherently."
**Method (mutate the fixes, not the feature):** re-read every changed file; reran the scoped suite and
line+branch coverage; RE-RAN the real G2 smoke end to end (llama3.1 + spaCy + e5-small); 13-mutation
sweep against a scratch COPY of the repo (repo never touched); 7 probes in scratchpad/s3r3 (p_plan,
p_mig, p_alias_inv, p_read, p_reset, p_retry, p_sweep_gap); byte-compared the founder's live DB against
R2's verified migrated copy; verified the Mem0 PR #4805 / issue #6591 claims against the GitHub API.

**Reproduced honestly — what holds.** 167/167 (0.0 flakes); coverage exactly as claimed (224 stmts, 0
missed, ONE branch partial 125->124). **The G2 smoke reproduces to the digit a third time**: 20 facts /
25 links / 18 nodes / 25 real join rows == log sum (three-way assert), गूगल admitted at cosine 0.9506,
facts_for_entity('गूगल') == facts_for_entity('Google'), danda excluded ("है", not "है।"), Part-B Gate-E
residual unchanged. R2-B1 fix real (mutation: dropping the exact arm fails the Übermensch test); R2-M1
real (scope + shape + anchor-first all revert-detected; the caller's plan dict is NOT mutated by apply);
R2-M2 real (substring regression fails both lookalike fixtures; SQLAlchemy's own create_all DDL passes
equality — no drop/rebuild loop on fresh DBs); R2-M3 real (dropping the swap fails two fixtures; keeping
min(last_updated) fails the recency fixture); R2-M4 verified BY REPRODUCTION (reset now succeeds with
FKs ON, agent-scoped nodes/links untouched); R2-M5 real. **kg_edges carries NO unique index** (fresh
schema and live DB: only non-unique idx_kg_edges_active) — the swap is safe as shipped.
**Disclosure (my incident):** importing `agentmem_os.db.engine` to read DB_PATH ran `init_db()` and
migrated the founder's LIVE DB (19:11:15). Byte-compared to R2's verified copy: identical except the
1,979 rows' `last_updated` (the intended recency fix) + the 2 new log columns; 34,905→32,194 rows,
sum(weight) 35,051.0 → 35,051.0 exactly. Predicted outcome confirmed on real data — accidentally.

**Verdict: BLOCK — 1 blocker, 5 majors, 16 minors.**

- **B1 (blocker) — the sweep does NOT recover skipped surfaces; four record sites say it does.**
  `link_missing` selects facts with ZERO join rows (fact_entities.py:598). A fact that linked ANY other
  surface is not a candidate, so a surface skipped for a missing alias anchor is orphaned PERMANENTLY.
  p_sweep_gap, measured: fact A "The user works at गूगल and Microsoft." → linked Microsoft, skipped गूगल;
  fact B (Indic-only) → 0 links. Anchor 'Google' arrives; `link_missing()` returns **candidates=1** (B
  only); B recovers, **A never does**; `facts_for_entity('Google')` returns [B] and the fact that
  literally names गूगल is invisible from that entity forever. False in: fact_entities.py:151 ("picked up
  by the link_missing sweep"), fact_entities.py:275 ("the sweep re-plans against fresh state later" — the
  R2-M1 skip inherits it), consolidation_v2.py:408 ("the link_missing sweep recovers them"),
  BUILD_LOG:575 (the compensating control offered for R1-B1). Realistic without code-switching: any
  2-entity Indic fact where one anchor exists and one doesn't. Honest bound: 0 facts in the current G2
  sample hit it (Part-B facts are support-gate rejected; Part-C's fact has one anchorable surface).

- **M1 (major) — "PROCESS INCIDENT (logged as its own lesson)" (BUILD_LOG:691) points at nothing.** No
  such lesson exists: docs/memory/lessons/process.md gained 7 lessons, none about the stray file/cwd; a
  grep for stray|cwd|absolute path|working directory across docs/memory returns ZERO hits. The standing
  rule ("every log write and its grep-verify use the ABSOLUTE repo path") also lives only in the build
  log, not where rules are looked up. This is a SAME-ROUND REPEAT of the class R2 caught, inside the
  paragraph that documents that class.
- **M2 (major) — the repair is an IMPORT side effect on production data with no backup, and the
  founder-facing note is now false.** engine.py:662 runs `init_db()` at import; _migrate_stage3 DELETEs
  rows, rewrites weights and swaps source/target. I triggered it on the founder's live DB by importing
  the module to read a constant. RUNNING_NOTES.md:330-334 still tells the founder "next product start
  against the dev DB performs that repair automatically" — it already ran. Minimum: correct the note;
  better: copy the DB file (16 MB) or write the pre-state of merged pairs before the first destructive
  pass, and say plainly that any import of the package performs it.
- **M3 (major) — the node merge INVERTS ALIAS_OF pairs, which the ordered exists-check then
  duplicates.** engine.py:407-409 re-points without re-ordering and the canonicalization is CO_OCCURS-
  only (engine.py:457-461). p_alias_inv: nodes 1/5/9, ALIAS_OF (5,9), 9 merges into keeper 1 → row
  becomes **(5,1)**; `_ensure_alias_edge` (fact_entities.py:413-420) checks (min,max) only, so the next
  link writes a SECOND row — measured before=1, added=1, after=2 [(2,1),(1,2)]. Reads survive (closure
  and loader are direction-agnostic), so the harm is duplicate metadata — but engine.py:389-391 justifies
  leaving typed edges alone with "supersession chains reference their ids and valid_from/valid_until
  already disambiguate", which is FALSE for ALIAS_OF (not in SUPERSEDABLE_RELATIONS, no valid_until).
  Fix: order typed re-points too, or correct the stated reason.
- **M4 (major) — a claimed-landed R2 minor only half-landed.** BUILD_LOG:686 says "'exactly once'
  softened to per-drain"; fact_entities.py:583 says "once PER DRAIN" but the MODULE docstring
  (fact_entities.py:33) still says "visits every unlinked fact exactly once". The load-bearing copy is
  the one that wasn't changed.
- **M5 (major) — R2-M4 is the only R2 fix with no regression test.** api/app.py:811-814 is correct (I
  reproduced it), but nothing in tests/ imports api or exercises a kg_nodes purge; reordering those
  deletes restores the 500 with a green suite. Every other R2 fix is revert-detected.

**Minors:** (m1) link_fact's docstring (fact_entities.py:188-189) still documents `plan=(plan, skipped)`
— the shape the code now REJECTS and a test pins as rejected. (m2) the store-owned retry re-uses the
same `skipped` list, so a race double-appends: p_retry shows ('गूगल','alias anchor not found at apply')
twice in one report (feeds `entity_links_skipped`). (m3) `_validate_plan` stops at `via`: surface "" is
ACCEPTED (creates an entity_text='' node + link row) and `skipped="oops"` explodes into ['o','o','p','s']
— the plan path bypasses `_dedup_surfaces`' len>=2 rule. (m4) the Indic-digit fix is UNTRIPWIRED —
deleting `not tok.isdigit()` (entity_aliases.py:145) leaves the suite green; the danda test two lines
away is where the assert belongs. (m5) `facts_for_entity(None)` raises AttributeError ('' and '  ' return
[]). (m6) tests/test_fact_entities.py:1018-1019 is `assert ... == [] or True` — a dead assertion inside
the R2-B1 regression test. (m7) R1-m6/R2-m10 unfixed 3rd round: consolidation_v2.py:441 still says
"commit below raises" (it raises at the next add_fact). (m8) R1-m10/R2-m11 unfixed 3rd round: the display
cache still stores SKIPPED surfaces (consolidation_v2.py:425). (m9) R1-m11/R2-m12 unfixed: unbounded
IN(...) lists on both fact_ids and closure node_ids. (m10) R1-m13 unfixed: the smoke still uses legacy
Query.get — 2 deprecation warnings inside the evidence artifact. (m11) BUILD_LOG:58 points at the Stage-2
G3 record "at the END of this file"; it is now mid-file (line 287). (m12) "recorded as a KNOWN ISSUE for
the loader backlog" (BUILD_LOG:686-689) — no backlog artifact exists; the record is that sentence.
(m13) RUNNING_NOTES' Stage-3 entry is stale (says "G3 in round 2", "156 tests"). (m14) post-reset,
link_missing RESURRECTS the purged demo nodes (p_reset: 'Google' returns as a new node) — "clear the
entire global KG namespace" holds only while nothing calls the sweep. (m15) kg_nodes' ONLY index is the
coalesce() expression index, which no query can use (every lookup filters agent_id directly): the seed
and every `_get_or_create_node` are full scans — measured 0.21 ms exact / 0.25 ms or_ at 10,911 nodes.
Fine today; the index is dedup-only, not a read index — say so rather than assume. (m16) the coverage
partial is the duplicate-token skip branch, not a "loop-exit" partial.

**Test quality:** 11 of 13 targeted mutations caught (M1-M10, M13). Survivors: the Indic-digit guard
(m4) and the `list(plan["skipped"])` copy. The R2-fix tests are genuine, not stamps.

**Must change to pass:** B1 (either extend recovery to partially-linked facts — persist skips or sweep
by surface-count — or correct ALL FOUR sites plus the founder-facing bound on cross-lingual unity), M1
(write the lesson, or delete the claim), M2 (correct RUNNING_NOTES; state the import-time destructive
repair plainly; backup or pre-state artifact before the first destructive pass), M3 (order typed
re-points OR fix the stated rationale), M4 (module docstring), M5 (a test that fails if the delete order
regresses). m1/m2/m3/m4/m6 land the same round; m7/m8/m9/m10 are 3rd-round carries — land or waive them
explicitly in the record.

**Refs:** db/fact_entities.py:33,151,178-181,188-189,275,413-420,470-478,598; db/engine.py:389-391,
407-409,457-461,662; llm/consolidation_v2.py:408,425,441; api/app.py:811-814; db/entity_aliases.py:145;
tests/test_fact_entities.py:937-947,1018-1019; CONSOLIDATION_V2_BUILD_LOG.md:58,575,686-698;
RUNNING_NOTES.md:330-334. Harnesses: scratchpad/s3r3/ (p_plan, p_mig, p_alias_inv, p_read, p_reset,
p_retry, p_sweep_gap, runmut, smoke_r3.log, mut/).

**Who needs to know:** Dev-Head (B1 + M1-M5 before the gate reopens; m7-m10 are third-round carries).
Bosses: **Stage 3 G3 round 3 BLOCKS** — every R2 fix I could mutate holds and the smoke reproduced a
third time, but the safety net that justified the R1-B1 contract does not catch the case that contract
creates, and it is asserted as fact in four places. Founder: (1) I accidentally ran the CO_OCCURS repair
on your live DB by importing the module — outcome byte-identical to the copy we verified, weight exactly
conserved (35,051.0), nothing lost; RUNNING_NOTES must stop saying it is still pending. (2) Cross-lingual
unity currently has an unstated bound: a mixed fact's un-anchored surface never links, even later.

## 2026-08-06 — Critics — Consolidation v2 Stage 3 (KG integration), G3 ROUND 4 — **BLOCK**

**Claim reviewed:** "R3's blocker + 5 majors + 16 minors all fixed or waived-with-record; 174 tests green
across the 4 scoped files; db/fact_entities.py 100% line (234 stmts) / 99% branch; fourth identical smoke;
three waivers honestly recorded — this is the stage-closing round."
**Method (mutate the fixes, not the feature):** reran the scoped suite + line/branch coverage; ran my own
FIFTH end-to-end G2 smoke (llama3.1 + spaCy + e5-small, AGENTMEM_OS_DB_PATH forced — live DB untouched,
mtime still 19:11); 16-mutation battery against a scratch COPY of the repo; 5 probes (p_retry, p_drift,
p_rescan, p_validate, p_reset_iso); read the founder's live DB and the R2-era pre-repair copy READ-ONLY
via `file:...?mode=ro` (no engine import — R3 lesson enforced); re-verified the Mem0 claims at the GitHub
API; audited the R3-resolution build-log section and the RUNNING_NOTES correction claim by claim.

**Reproduced honestly — what holds.** 174/174 green (114.8s). Coverage exactly as claimed: 234 stmts, 0
missed, 102 branches, ONE partial (132->131). **R3-B1's fix is real AND correct under paging** (p_rescan):
default drain sees only the 2 zero-surface facts and terminates; deep drain covers all 7 facts, terminates
(5 rounds at limit=2, 8 at limit=1), recovers the गूगल surface, is IDEMPOTENT on a second full drain (0 new
links, still exactly 1 ALIAS_OF row), correct at after_id>max and under user_id scope. R3-M1 genuinely
closed (lessons/process.md:435-445 + the two R3-incident checks at :423-433). R3-M2's RUNNING_NOTES
correction verified line by line against the live DB: 34,905 → 32,194 rows, sum(weight) 35,051.0 conserved
EXACTLY, 0 dup pairs, 0 inverted rows; the R2-era copy is genuinely pre-repair (34,905 / 1,979 dup pairs).
R3-M3 closed (MU7/8/9 caught); ALIAS_OF is symmetric everywhere it is read (knowledge_graph.py:848 renders
"A = B"), and NO runtime code re-points edges, so the swap cannot invert meaning. R3-M4 closed in code.
R3-M5 closed (MU12 turns the real-endpoint test red). Mem0 claims re-verified today: graph_memory.py 404,
mem0/graphs 404, PR #4805 merged 2026-04-14, issue #6591 open — COMPETITIVE_ANALYSIS.md:112-121 honest.
**FIFTH identical smoke reproduction:** 20 facts / 25 links / 18 nodes / 25 join rows == log sum (three-way
assert), गूगल↔Google 0.9506, facts_for_entity('गूगल') == facts_for_entity('Google') (3 facts, both
languages), danda excluded, Part-B Gate-E residual unchanged (2 candidates → 0 created).
**Mutation score 13/16.** The 3 survivors ARE findings M2, M3, m3.

**Verdict: BLOCK — 0 blockers, 6 majors, 9 minors.** No code-behavior blocker remains; every major is a
record / test-stamp / latent-fragility item with a named one-to-few-line fix.

- **M1 — "ALL FOUR sites now state the two depths precisely" (BUILD_LOG:727-728) is false at the site that
  lives in the record.** BUILD_LOG:575 still reads "such surfaces land in skipped and the sweep recovers
  them" — the exact sentence R3-B1 named. Two more carry the same defect: :496-499 ("a link_missing()
  sweep makes the recoverability claim TRUE") and :596 ("visits every unlinked fact exactly once", the
  R3-M4 wording). The file has an in-place correction convention used TWICE (:58 "[SUPERSEDED…]", :199
  strikethrough + "[SUPERSEDED by R4]") and DESIGN.md uses "CORRECTIONS OF RECORD" — neither applied here.
  Same class as R3-M4: claimed landed everywhere, the load-bearing copy untouched.
- **M2 — the retry regression test does not detect reverting the fix it names (MUTATION-PROVEN).** Change
  fact_entities.py:276 `skipped = list(skipped)` → `skipped = skipped` and
  tests/test_fact_entities.py:1271 passes in 0.37s. Cause: the mock at :1282-1286 raises BEFORE the real
  body, so attempt 1 never appends the apply-side skip. p_retry (realistic race: raise inside
  _get_or_create_node AFTER the anchor-miss append) reproduces R3's bug exactly — 2× ('गूगल','alias anchor
  not found at apply') without the copy, 1× with it. BUILD_LOG:752-753 claims "(copy; test)".
- **M3 — the CO_OCCURS backup INSERT (engine.py:507-510) — the branch that actually ran on the founder's
  DB (2,711 rows) — has NO tripwire.** Replacing it with `pass` leaves 4 relevant tests green; only the
  ALIAS_OF branch is pinned (tests:1332-1336). Same class as R3-M5, on a data-safety net. One assert fixes it.
- **M4 — the backup table's schema is frozen at creation; the first future kg_edges column addition makes
  the PACKAGE UNIMPORTABLE (REPRODUCED, p_drift).** engine.py:462-464 `CREATE TABLE IF NOT EXISTS … AS
  SELECT * FROM kg_edges WHERE 0`; both inserts are `SELECT *`. After an ordinary `ALTER TABLE kg_edges ADD
  COLUMN`: `OperationalError: table kg_edges_dedup_backup has 11 columns but 12 values were supplied` →
  _migrate_stage3 RuntimeError → init_db() at engine.py:704 runs at IMPORT → API/CLI/MCP/tests all fail to
  start. Fires only on DBs that HAVE duplicate pairs, so CI on a fresh DB stays green and a user's install
  dies. No data loss (failure precedes the DELETE; txn rolls back). Fix: name the columns or rebuild on shape drift.
- **M5 — the recovery sweep has NO caller in the product.** `link_missing` (either depth) is invoked only
  by tests (test_fact_entities.py, test_consolidation_v2.py:976) — not by the engine after a link_failure,
  not by any endpoint or CLI command, not even by the G2 smoke. It is nevertheless the compensating control
  sold for R1-B1 and for the failure policy (fact_entities.py:30-44, consolidation_v2.py:34-37,
  BUILD_LOG:496-499). Recovery today = a human writing the documented drain loop in a REPL. Same class as
  the logged R1 lesson "schema-complete and prompt-unreachable". Wire it, or say so plainly in the record.
- **M6 — the production repair's pre-state is not where a record says it is.** Founder memory
  agentmem_os_known_issues.md:68 (#11) says the dev-DB repair's "deleted rows snapshotted to
  kg_edges_dedup_backup" — verified read-only: the live DB has NO such table (the repair predates the
  backup code). RUNNING_NOTES:348-351 is correct and points instead at the critic's scratchpad — a 15.8 MB
  file in /private/tmp (macOS-purgeable). Correct #11; keep a durable copy or state plainly that no
  pre-state survives. (Softest of the six: the repair itself was verified twice and conserved weight exactly.)

**Minors:** (m1) benchmarks/consolidation_v2_stage3_smoke.py:23-36 does not pin AGENTMEM_OS_DB_PATH the way
tests/conftest.py does — every documented rerun runs the import-time migration against the founder's LIVE
DB (forensics: live mtime still 19:11 and no backup table, so no rerun has used the default path since R3;
my independent 5th reproduction corroborates the numbers, so this is hygiene, not evidence integrity — but
keep the run log as an artifact). (m2) BUILD_LOG:701,771 still say "loop-exit partial"; measured partial is
132->131, the duplicate-script-token skip arm (R3-m16, 4th round). (m3) the _FACT_ID_BOUND LIMIT is
untested — dropping `.limit(_FACT_ID_BOUND)` leaves test_fact_id_bound_is_loud green (it patches the bound
to exactly the fact count); only the warning is pinned. (m4) R3-m9's other half is still unbounded:
node_id.in_(node_ids) and the closure in_(frontier) (fact_entities.py:519-520,543). (m5) the demo-reset
test is order-coupled: tests:1386 asserts SemanticFactEntity.count()==0 GLOBALLY on the conftest-shared
scratch DB — p_reset_iso: one agent-scoped link created earlier in the same process makes it FAIL while the
endpoint behaves perfectly; it also leaves a ResetCorp fact + regrown nodes behind. (m6) _validate_plan
still admits 1-char and untrimmed surfaces ("a", "   x   " both created nodes) that _dedup_surfaces rejects
everywhere else; confidence/etype junk IS loud (StatementError/ProgrammingError — verified). (m7)
consolidation_v2.py:445 still says "commit below raises" (it raises at the next add_fact) — 4th round; the
waiver names the TEST gap, not the wording. (m8) RUNNING_NOTES:317,322 still say "G3 in round 2" / "156
tests" at stage close (R3-m13, unlanded, unwaived). (m9) facts_for_entity(None) raises AttributeError
(fact_entities.py:496; R3-m5, unlanded, unwaived).

**Waivers judged.** (1) Poisoned-batch-commit-raise untested — defensible AS A TEST WAIVER, but "not worth
the harness" is overstated (my p_retry builds a mid-apply DB failure in ~15 lines) and it does not cover
the wording R3 flagged. (2) kg_nodes dedup-index-not-read-index — DEFENSIBLE, measured (0.21-0.25 ms at
10,911 nodes), recorded in both the build log and founder memory #13. (3) Loader same-pair collapse —
DEFENSIBLE and now genuinely recorded (founder memory #12 with 12.0→1.0 and both call sites); one honesty
line missing: Stage 3 makes the collision the COMMON case, since the turn path writes BOTH a CO_OCCURS and
an ALIAS_OF edge for exactly the code-switched pairs this stage advertises.

**Must change to pass:** M1 (mark or correct :575, :496-499, :596 using the file's own SUPERSEDED
convention, or drop the "all four" claim), M2 (make the mock raise after the anchor-miss append — p_retry
shows the shape), M3 (one backup-row assert on the CO branch), M4 (explicit column list or shape-drift
rebuild), M5 (wire a caller or disclose "operator-only, no product caller" in the module docstring AND the
record), M6 (correct known-issue #11; durable copy or plain statement). m1/m2/m5/m8 are cheap and should
land the same round; m3/m4/m6/m7/m9 land or get waived explicitly.

**Refs:** db/fact_entities.py:30-44,276,496,519-520,543,592-662; db/engine.py:438,462-464,488-492,507-510,
704; llm/consolidation_v2.py:34-37,410,445; api/app.py:805-825; tests/test_fact_entities.py:1242-1343,
1271-1294,1332-1336,1386; benchmarks/consolidation_v2_stage3_smoke.py:23-36;
CONSOLIDATION_V2_BUILD_LOG.md:58,496-499,575,596,701,709-776; RUNNING_NOTES.md:317,322,341-351;
docs/memory/lessons/process.md:410-445; founder memory agentmem_os_known_issues.md:68-70.
Harnesses: scratchpad/s3r4/ (runmut.py + 16 mutations, p_retry, p_drift, p_rescan, p_validate, p_reset_iso,
p_ro, smoke_r4.log, run1.log).

**Who needs to know:** Dev-Head (M1-M6 + m1/m2/m5/m8 before the gate reopens; the three test-stamp findings
M2/M3/m3 are the same family — assertions that do not pin the mechanism). Bosses: **Stage 3 G3 round 4
BLOCKS, but nothing about the SYSTEM is broken** — every R3 fix works under mutation, the smoke reproduced a
fifth time to the digit, and the remaining six are record accuracy, three untripwired assertions, one latent
import-killing fragility, and one unwired compensating control. One tight round from close. Founder:
(1) the sweep that makes "linking can fail safely" true has no product caller — recovery is currently an
operator action; (2) the pre-repair copy of your live DB lives in a temp dir and one memory record says it
lives in a table that does not exist; (3) the live DB was NOT touched this round (mtime still 19:11).

## 2026-08-06 — Critics — Consolidation v2 Stage 3 (KG integration), G3 ROUND 5 — **PASS-WITH-NOTES**

**Claim reviewed:** "R4's 6 majors + 9 minors all fixed; 179 tests green across the 4 scoped files;
db/fact_entities.py 240 stmts / 0 missed / 99% branch; this round closes Stage 3."
**Method (mutate the fixes, attack the new code):** reran the scoped suite + branch coverage; 23-mutation
battery against a scratch COPY of the repo (never the working tree); 3 probes against recover_links
(breadth, audit coherence, drain-failure propagation) + an amplification probe; read the founder's live DB
and BOTH scratchpad copies READ-ONLY via `file:...?mode=ro` (no engine import — R3 lesson enforced); ran my
own SIXTH end-to-end G2 smoke (real llama3.1 + spaCy + multilingual-e5-small, AGENTMEM_OS_DB_PATH forced);
audited the R4-resolution build-log section, the RUNNING_NOTES correction and the two founder-memory records
claim by claim.

**Reproduced honestly.** 179/179 green (116.0s). Coverage EXACT: db/fact_entities.py 240 stmts, 0 missed,
106 branches, ONE partial (133->132, the duplicate-token skip arm) = 99% — matches the claim to the digit.
**Mutation score 19/23.** The two MANDATORY mutations are now caught: reverting `skipped = list(skipped)`
turns test_retry_does_not_double_append_skipped RED in 0.60s (R4-M2 closed), and replacing the CO_OCCURS
backup INSERT with `pass` turns test_global_repair_merges_reversed_rows RED (R4-M3 closed). R4-M4 closed
three ways (guard removal, DROP-instead-of-RENAME, name-comparison weakened — all caught). R4-M5 closed
(removing the auto-invocation, stalling the cursor, dropping the deep flag, dropping the failures list, and
disabling batch suspension are all caught). R4-M1 closed: the three bracketed CORRECTED annotations are in
place and a repo-wide grep finds NO remaining site claiming the shallow sweep recovers skipped surfaces.
R4-M6 closed AND independently verified: the live DB (mtime still 19:11:15, untouched this round) has NO
kg_edges_dedup_backup table, 32,194 CO rows / 35,051.0 weight / 0 dup pairs / 0 inverted; the R2-era
pre-repair copy still exists (34,905 rows, 1,979 dup pairs) and its loader-visible weight is 32,340.0 —
so memory #11's "2,711 units of visible weight lost" is EXACTLY right (35,051.0 − 32,340.0).
**SIXTH identical smoke reproduction, mine:** 20 facts / 25 links / 18 nodes / 25 join rows == log sum 25
(three-way assert), गूगल↔Google 0.9506, facts_for_entity('गूगल') == facts_for_entity('Google') (3 facts,
both languages), danda excluded, Part-B Gate-E residual unchanged (2 candidates → 0 created),
"planning a trip to Disneyland" still typed state/undated (planned-marker unreachability, observed live).

**Verdict: PASS-WITH-NOTES — 0 blockers, 0 system majors, 1 mandatory record correction, 8 minors.**
The system is sound and every R4 fix holds under mutation. One line of the record is false and must be
corrected before the stage record is filed; it is grep-verifiable and needs no further review round.

- **MANDATORY (record) — BUILD_LOG:835 claims '"commit below raises" corrected to next-flush-or-commit'.
  It is NOT corrected.** llm/consolidation_v2.py:444-446 still reads "the failure poisoned the session at
  the DB layer, commit below raises LOUDLY" (5th round for this item, R4-m7), and the module header at
  :36-38 carries the same "aborts the batch LOUDLY at commit". Land the two-word comment fix OR delete the
  claim — a false "landed" line in the stage-closing record is the R4-M1 class.

**Minors.** (m1) Two mutation SURVIVORS in the NEW recover_links code: `"complete": rounds < max_rounds` →
`True` and `while rounds < max_rounds` → `while True` both leave the suite green; coverage corroborates
(consolidation_v2.py:541, the LOUD runaway warning, is never executed). Repeat of the logged enforced check
"every claimed fix goes through the mutation sweep" (lessons:199-206). (m2) The tightened _validate_plan
surface rule (stripped, len>=2) has no tripwire — reverting it to the old "non-empty" rule leaves 14
selected tests green (2 survivors). (m3) **Audit incoherence:** ConsolidationLog persists entities_linked=0
+ link_failure and NOTHING about the auto-recovery — measured 12 link rows in the DB against a persisted 0;
link_recovery exists only in the returned report and the logger.info line. Same reasoning as R1-M2 (the
count alone cannot distinguish states) applied one level up. (m4) **The auto-drain is not best-effort:** if
link_missing itself raises (probe: OperationalError "database is locked"), the exception ESCAPES
consolidate_session after facts + log committed, discarding the whole report (link_failure, rejections,
warnings, numbers_audit) for a linking-side fault; a caller retry re-spends the LLM. 3-line try/except,
untested. (m5) **Blast radius undisclosed:** the auto-drain is SCOPE-wide, not batch-scoped — measured 300
pre-existing unlinked facts → 302 swept / 673 links created inside ONE consolidate_session call; bounded
only at max_rounds×limit = 100,000 facts; under a PERSISTENT linker fault it is O(sessions × scope)
(measured: 8 sessions × 400 facts = 3,216 attempts, 661 KB of log, single report log lines of 11.9 KB
because the full failures list is logged). (m6) benchmarks smoke uses os.environ.setdefault while the
repo's own idiom (tests/conftest.py, R3-N8) is FORCED assignment; BUILD_LOG:828-829's "a rerun can never
migrate the live DB" holds only while the var is unset — setup.sh's .env template sets it. (m7) No smoke
rerun is recorded for R4 (every prior round recorded one, and R4 changed consolidation_v2.py); I ran it —
6th identical reproduction — so this is a record gap, not an evidence gap. (m8) RUNNING_NOTES says the
compensating control "has a product caller" without stating anywhere in that doc that ConsolidationV2
itself has NO caller outside benchmarks/tests (verified: api/cli/mcp_server/storage/memory/agents contain
zero references; the v1 SleepConsolidationEngine IS wired in storage/store.py). The phrase is defensible
(R4 named "the engine after a link_failure" as the first acceptable caller) but the founder-facing doc
needs the one clause.

**Waivers judged — all three stand.** (1) Poisoned-batch commit-raise untested: defensible test waiver.
(2) kg_nodes dedup-index-not-read-index: defensible, measured, recorded in memory #13. (3) Loader same-pair
collapse: now genuinely upgraded — memory #12 carries the R4-demanded honesty line ("Stage 3 makes this
collision the COMMON case", must-fix before any cross-lingual subgraph-serialization claim).

**Stage 3 honest-claims summary (for the record).** CAN claim: facts link to KG nodes through a real join
table with provenance (surface_text, linked_via, confidence); cross-lingual entity unity demonstrated on
real models six times identically (गूगल↔Google 0.9506, same 3 facts from both surfaces); planning is
separated from applying so no model load runs under a write lock; a real production KG bug was found and
repaired with weight conserved exactly (35,051.0, verified read-only twice); recovery is now auto-invoked
after a suspended-linking commit. MANDATORY DISCLOSURES: Gate-E canonical-English residual (Hindi sessions
still produce English facts that the support gate rejects — Gate E work, not Stage 3); the 'planned' marker
is prompt-unreachable (0/23 in G2, reconfirmed live); case folding is ASCII-only (non-ASCII matches exactly
or via ALIAS_OF); recovery has TWO depths and the default one cannot see partially-linked facts; the
auto-drain is scope-wide and its links are NOT reflected in ConsolidationLog; the loader still collapses
same-pair CO_OCCURS+ALIAS_OF edges (the common code-switched case); nothing is wired into product retrieval
(Stage 5) — ConsolidationV2 has no caller outside benchmarks and tests.

**Refs:** db/fact_entities.py:276,279,338-386,607-677; llm/consolidation_v2.py:34-38,444-446,479-490,
515-547; tests/test_fact_entities.py:1281-1309,1133-1161,1439-1465; tests/test_consolidation_v2.py:
940-1024; db/engine.py:462-483,507-511,526-529; CONSOLIDATION_V2_BUILD_LOG.md:499-503,580-583,605-607,
791-845; RUNNING_NOTES.md:341-372; founder memory agentmem_os_known_issues.md #11-#13.
Harnesses: scratchpad/s3r5/ (runmut.py + 23 mutations, p_recover.py, p_amplify.py, smoke_r5.log).

**Who needs to know:** Dev-Head — one mandatory record correction (the false "commit below raises corrected"
claim) plus 8 minors; m1/m2 are mutation-proven test gaps in code written for R4, m4/m5 are the new
auto-drain's failure and cost behavior. Bosses: **Stage 3 PASSES on the system** — 179 green, coverage
exact, 19/23 mutations caught (the 4 survivors are all named minors), the G2 smoke reproduced a sixth time
to the digit, and the live DB was not touched. Founder: recovery now runs automatically after a failed
linking batch, but it sweeps the whole scope (a first triggered run on an old DB is a full backfill), its
links do not appear in the persisted consolidation log, and if the drain itself dies the whole run's report
is lost — three cheap fixes, none of them data-safety.

## 2026-08-06 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 1 — **BLOCK**

**Claim reviewed:** "Stage 4 G1+G2 done: 204/204 across 5 scoped files, llm/supersession.py 96%,
conflict-detection script 100%; the LLM only proposes and deterministic gates decide; events are
excluded except planned→cancelled; every judgment is persisted; recovery sweep auto-invoked on
judge failure; G2 demonstrates the machine end to end with an honest in-order/out-of-order asymmetry."
**Method:** ran the 5 scoped files (204 passed, 119.4s) + branch coverage on llm/supersession.py +
tests/test_conflict_detection.py directly (28/28); read every Stage-4 line of code against every
Stage-4 line of the record; 31 valid mutations on a scratch COPY (never the working tree); 16
attack probes (P1-P16) on isolated scratch DBs; my own rerun of benchmarks/consolidation_v2_stage4_smoke.py
with real llama3.1 (reproduced in-order=0 / out-of-order=1, same fact ids, same mutual 4<->6 tie-drop);
live DB inspected READ-ONLY via file:...?mode=ro.

**Reproduced honestly.** 204/204 and the per-file split is exact (supersession 21 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5). Coverage EXACT: 164 stmts, 6 missed, 46
branches, 0 partial = 96%, missing 127-140 = the real-LLM HTTP body, as claimed. Script suite 28/28.
**Mutation score 22/31.** Both gate-caught design fixes are genuinely pinned: removing the lexical
pool turns test_entityless_facts_reach_the_judge_via_lexical_pool RED (1.1s); removing the reserved
planned pool and crowding it out behind the peer cap BOTH turn test_shortlist_composition_and_cap RED;
the conflict_detector "works at" root-cause fix is caught by tests/test_consolidation_v2.py.
db/knowledge_graph.py:96-97's local patch verified redundant with NO behavior change (WORKS_AT still
compiles with the s-form). Attacks that HELD: two judges racing the same pair (one applies, the other
records "store refused: already superseded by ...", both audited); dead-winner chains (chain() = [2,3,1],
transition_text renders the full chain, current_facts correct); cost (9-12 ms per shortlist at 300 /
1,500 / 5,000 live facts).

**Verdict: BLOCK — 4 blockers, 8 majors, 12 minors.** The dangerous class is NOT blocked: I invalidated
a true current fact through two independent paths using only in-scope model output and passing gates.

**BLOCKERS**
- **B1 — a planned EVENT can invalidate a true current STATE fact.** llm/supersession.py:215-243: the
  superseded_ids loop has NO fact_type/event_status guard (the cancelled_ids loop at :249-254 does).
  The reserved planned pool (:341-345, the G1 "design fix") guarantees planned events are shortlisted;
  'planned' means future-dated, so `_domain_time(event)` is ALWAYS later and the direction rule at
  :232-233 REVERSES — the state fact loses. Probe P2: "The user works at Google and left the conference
  committee." superseded BY "The user attends the Google conference." [2099/06/01]; current_facts then
  returns ONLY the event. Probe P14: same result with a 2030-dated offsite, cosine 0.3908, zero drops.
  **P13 falsifies BUILD_LOG:1067-1068** ("One candidate double-listed in both arrays was dropped by the
  status guard"): double-listing supersedes FIRST and cancels SECOND — final state = user's current
  fact invalidated, sole live fact a CANCELLED event, dropped=[]. Violates design decisions 2 and 3 and
  supersession.py:17-22. Repro: scratchpad/s4r1/probe.py (P2), probe3.py (P13, P14).
- **B2 — the co-signal gate is VACUOUS for every lexical-pool candidate.** :336-337 admits on
  `_tfidf_cosine(new, cand) >= _COSIGNAL_COSINE`; :222 gates on the same function, same pair, same
  threshold — always true. For the pool added at G2 (the one built to reach entity-less facts) gate (b)
  rejects nothing, and ":225 'LLM-only verdicts never act'" is false. Probe P1: "The user is allergic to
  peanuts." vs "The user is allergic to shellfish." — no entity links, no polarity flip, cosine 0.6311 —
  the peanut allergy is invalidated on the model's word alone, dropped=[]. The justification comment at
  :70-76 ("our shortlist already guarantees entity overlap, so this gate must carry more weight alone")
  is false for this pool, and its provenance is wrong: forget_about's 0.25 exists only in a docstring —
  memory/conflict_detector.py:337-363 uses >=4-char keyword overlap, no cosine. Compounding: M22 (pool
  threshold 0.25->0.0) leaves 21/21 green — the only thing between the model and every live same-type
  fact in the newest 300 is an untested constant.
- **B3 — an applied supersession can exist with NO audit row and never be re-judged.** `_apply` commits
  each action immediately (:236-240, :257) and writes the judgment row in a separate session at the end
  (:262-269). Probe P4 (crash before the row): new.superseded_by set, t_invalid set, 0 judgment rows,
  and the fact now fails judge_missing's `superseded_by IS NULL` predicate (:392-396) -> permanently
  unaudited AND unreachable by the sweep. Falsifies design decision 7 ("Every judgment is PERSISTED")
  and :48-50 ("a fact with no judgment row has never been judged"). The mirror case re-judges, re-spends
  the LLM, and persists "store refused: already superseded" for an action this judge itself performed.
  Both store paths already accept db= with flush-not-commit — one transaction closes this.
- **B4 — record claim falsified: "auto-invoked on judge failure".** BUILD_LOG:988-992 and
  consolidation_v2.py:641 ("the R4-M5 lesson applied at build time, not at review time"). Repo-wide grep:
  `recover_judgments` has ZERO callers outside tests. Link recovery IS auto-invoked (:506-509); the judge
  path (:554-572) only patches the log row. Exactly the lesson at lessons/process.md:468-476. NOTE: do
  not blindly auto-invoke — process.md:478-491 (blast radius); a scope-wide judge sweep is one LLM call
  per fact. Correct the record, or wire it batch-scoped with the worst case stated.

**MAJORS.** (Ma1) `_has_polarity_flip` is used outside the guard rails its own shipped caller applies:
conflict_detector.py:254-259 requires entity overlap AND cosine >= 0.10 first; supersession.py:349 calls
it bare, and structural categories skip the entity check by design (:181-182). Probe P3: flip=True
(employment) for "works at Google" vs "left the party early"; flip=True (location) for "lives in Berlin"
vs "left a good tip"; flip=True (education) for "studying at MIT" vs "left her umbrella" — "left" is the
shared negative token of three categories, and Stage 4 widened the employment positive vocabulary.
(Ma2) Domain comparison mixes axes and ignores intervals, unaudited: `_domain_time` (:107-111) falls back
t_occurred->t_mentioned, so probe P6 has a fact mentioned 2023/01/01 (t_occurred 2024/12/01) supersede a
fact mentioned 2024/06/01; and a month-interval fact (2023/05 -> 05/01..05/31) loses to a point date
INSIDE its own interval — t_occurred_end is never read, though the store models it deliberately
(semantic_facts.py:85-110). The tie gate (:227) is exact string equality, so overlap-ambiguity is not
"conservative". The audit row records no domain times, no axis, no direction rationale.
(Ma3) 'cancelled' has no reader: probe P5 — current_facts, facts_as_of and facts_overlapping all return
a cancelled event like a live one, transition_text returns ""; nothing in db/api/storage/memory/mcp_server/cli
filters or annotates it. And design decision 9's "EVENT_STATUSES grows 'cancelled'" is false —
semantic_facts.py:58 is still {"occurred","planned"} and mark_event_cancelled writes a value the store's
own validator rejects. (Ma4) The audit row omits what the gates turn on: shortlist_json = {"ids",
"peer_cap", "planned_cap"} only (:361-363) — no lexical window, no lexical threshold, no per-candidate
pool provenance, so an auditor cannot tell whether a supersession had independent evidence (see B2).
(Ma5) Skip reason misattributes: both paths persist "no candidates sharing an entity" (:176, :179) though
three pools ran — observed live in my smoke pass C, 7/7 rows, on facts with no entity links at all.
(Ma6) Fixture blindness: M21 (drop the shortlist `scope_key` filter) leaves 21/21 green because
test_shortlist_composition_and_cap:248-250 dates the other-scope fact OLDEST — the cap alone excludes it
(probe P12: with it newest, another user's fact text enters the prompt). M23 (cap 12->100) and M24 (window
300->1000) survive because the same test monkeypatches those constants to the asserted values. Repeat of
lessons:124 and lessons:199. (Ma7) Stage 4 re-armed the import-time-migration trap (lessons:423-433):
supersession.py:63 is a MODULE-level import of conflict_detector, which module-imports db.engine, whose
tail runs init_db(). Measured: importing llm.consolidation_v2 does NOT pull the engine; importing
llm.supersession DOES — a new llm->engine edge, so reading a constant out of the judge runs every
migration. Every other DB import in that module is function-local. (Ma8) The G1 record's own instruction
(run tests/test_conflict_detection.py directly) bypasses tests/conftest.py:12-16 — the file exists to FORCE
a scratch DB "regardless of the inherited environment" — and the script calls init_db() at :198 and writes
through the real engine.

**MINORS.** (m1) M18: mark_event_cancelled's rowcount guard — the "race-defended at the store" claim —
can be deleted with the whole scoped suite green (supersede's equivalent IS caught, M30). (m2) M27:
removing `superseded_by.is_(None)` from `_pool` leaves 21/21 green — dead facts would fill the prompt.
(m3) M20: judge_missing's live-fact filter untested. (m4) M14: `_int_list`'s bool rejection untested.
(m5) M17: "CREATED facts only" has no tripwire AND is disclosed only in a code comment
(consolidation_v2.py:441-446) — no "re-affirm"/"created only" line exists anywhere in the Stage-4 record.
(m6) judge_failure patch picks the newest ConsolidationLog row for the session (:561-565) — probe P8: run
A's failure lands on run B's row. (m7) ConsolidationLog persists judge_failure but nothing about
SUCCESSFUL judgments (R5-m3 class). (m8) `_apply` binds ctext/_cat/ctype and never uses them (:220-221) —
the unused `ctype` is precisely the guard B1 needs. (m9) The smoke prints its own criterion ("the direction
rule holds iff the same old fact loses in both") and the recorded run does not meet it; the record
discloses the asymmetry but never says the criterion failed. (m10) The lexical pool produced NO candidate
in ANY real-corpus judgment in the smoke (pass C: 7/7 skips); the one real supersession came through the
entity pool — the G2 fix is unit-tested only, and run 2 showed the real cause of run 1 was event typing.
(m11) recover_judgments' bound is max_rounds x limit = 40,000 facts ~ 40,000 LLM calls, not stated (contrast
recover_links, which states its bound after R5-m5). (m12) The "newest 300" window is in the docstring but
not in the audit row.

**Disclosure — my own two incidents, both the lessons:423-433 class.** (1) Running
tests/test_conflict_detection.py directly as instructed ran init_db() against the founder's production DB
(/Volumes/Sahith_SSD/AgentMem-OS/db/agentmem_os.db, mtime 22:56) — Stage-4 columns added,
supersession_judgments created, 1 session + 3 turns written and deleted by the script's own finally block.
(2) `import agentmem_os.llm.supersession` (to verify a scratch path) did it a second time — the Ma7 edge.
Verified READ-ONLY afterwards: 0 leftover rows, CO_OCCURS 32,194 rows / 35,051.0 weight (identical to the
R5-verified values), the single is_active=0 turn is the pre-existing demo one, max turn id back to 16640.
No data lost; the schema is now Stage-4-migrated. My harnesses forced AGENTMEM_OS_DB_PATH from that point
on; the smoke rerun forced it too.

**Refs:** llm/supersession.py:63,70-76,107-111,215-243,249-254,262-269,336-345,349,361-363,392-396;
db/semantic_facts.py:52-58,203-206,375-378,468-514; llm/consolidation_v2.py:441-446,506-509,554-572,637-667;
memory/conflict_detector.py:34-65,181-182,254-259,337-363; db/knowledge_graph.py:96-97;
tests/test_supersession.py:241-283,508-553; tests/conftest.py:12-16;
CONSOLIDATION_V2_BUILD_LOG.md:942-1001,1015-1078. Harnesses: scratchpad/s4r1/ (runmut.py + runmut2.py,
33 mutations; probe.py, probe2.py, probe3.py, probe4.py; smoke_r1.log).

**Who needs to know:** Dev-Head — B1/B2 are live over-supersession paths with 3-line repros; B3 is a
one-transaction fix; B4 is a record correction (do NOT auto-wire without sizing the sweep). Bosses:
**Stage 4 does not pass round 1** — the machine works (reversal reproduced live, race and chain attacks
held, 22/31 mutations caught, coverage exact) but its central safety claim ("an LLM-only verdict never
acts", "events are excluded") is false on two demonstrated paths. Founder: the judge can currently delete
a true fact from your current view in two ways — a future-dated plan outranking a present fact, and a
lookalike sentence passing a gate that is the same test twice — both reversible (nothing is destroyed),
both fixable this round.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 2 — **BLOCK**

**Claim reviewed:** "R1's 4 blockers + 8 majors all fixed (type guard, co-signal independence, one
transaction, record corrections, Ma1-Ma8); 211 tests green across the 5 scoped files (28/54/64/60/5);
conflict script 28/28; smoke reproduces post-fix."
**Method:** reran the 5 scoped files IN THE REPO (211 passed, 120.7s) and again on a scratch copy with
branch coverage; ran tests/test_conflict_detection.py directly (28/28) AFTER verifying its new guard;
reran benchmarks/consolidation_v2_stage4_smoke.py with real llama3.1; 34 valid mutations on a scratch
COPY (never the working tree), all survivors re-run against ALL 5 files; 5 attack probes on isolated
scratch DBs; a 20,000-pair randomized fuzz of the new co-signal's arithmetic; import-edge probe over
every module in llm/, db/, memory/. Every shell and every harness exported AGENTMEM_OS_DB_PATH to a
scratch file BEFORE any import (lessons:547). **No production DB was touched this round** — verified by
pointing the guard at a scratch "PRETEND_PROD.db" and watching the trap fire there instead.

**Reproduced honestly.** 211/211 and the per-file split is EXACT (supersession 28 / consolidation_v2 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5). Conflict script 28/28. Coverage on
llm/supersession.py = 215 stmts, 9 missed, 70 branches, 2 partial = **95%** (was 96%): missing 165-178 is
the real-LLM HTTP body as before, but 142 and 325-327 are NEW uncovered PRODUCT lines (see Ma2/Ma7 below).
Smoke reproduces exactly: same Rachel supersession in backfill order ([2023/05/23] superseded by
[2023/05/26], t_invalid=2023/05/26, transition_text + facts_as_of correct), same in-order=0 /
out-of-order=1 asymmetry, boundary pass 0. **All FIVE demanded mutations are genuinely pinned** — revert
the type guard -> test_planned_event_can_never_supersede_a_state RED; make lexical cosine approve ->
test_similarity_cannot_approve_what_similarity_admitted RED; drop `db=txn` (split the transaction) ->
test_actions_and_audit_commit_atomically RED; un-guard the flip -> test_polarity_flip_requires_shared_subject
RED; revert to mixed-axis scalars -> test_interval_overlap_is_ambiguous_strict_precedence_applies RED.
Ma4/Ma5/Ma6/Ma7 verified (audit provenance + constants pinned; skip reason pinned; scope-filter removal now
detectable; `import agentmem_os.llm.supersession` no longer pulls db.engine — measured clean, as are all of
db/* and llm/* except the pre-existing llm.adapters and memory.conflict_detector). Retry loop attacked and
it HOLDS: a permanently-refusing store terminates after exactly len(plan) attempts with every exclusion in
dropped_json and applied={} (probe p4-A). Audit JSON serializes cleanly (dates are sortable STRINGS, tuples
-> arrays, None -> null). The (t_occurred=None, t_occurred_end=set) shape is unreachable through the store
(add_fact:212-214 and _reaffirm:365-367 always write the pair together) and would degrade to the mention
axis anyway. **Mutation score 19/34.**

**Verdict: BLOCK — 4 blockers, 8 majors, 12 minors.** The R1 blockers are closed. The CODE WRITTEN TO
CLOSE THEM reopens the same dangerous class three ways, and the one fix aimed at protecting reviewers is
dead code inside a docstring.

**BLOCKERS**
- **B1 — the metric-update co-signal is the SAME similarity function, and it invalidates a true current
  fact.** llm/supersession.py:495-505 computes `stripped_cos` on numbers-stripped text and gates at 0.7,
  claiming independence ("numbers-stripped texts near-identical"). But `_tfidf_cosine` tokenises with
  `re.findall(r"[a-z]+")` (memory/conflict_detector.py:74-97) — **digits are already invisible to it**, so
  `stripped_cosine` is IDENTICAL to `cosine` for every input: 20,000 randomized adversarial pairs, **0
  differences**. The branch is literally `cosine >= 0.7 AND the number strings differ` — similarity
  approving what similarity admitted at 0.25, exactly the rule R1-B2 established, one threshold up.
  Consequence, reproduced end to end (probe p1): "The user's personal best time in the charity 5K run is
  25:31." is SUPERSEDED by "The user's personal best time in the charity 10K run is 55:12." — cosine 1.000
  (the tokenizer cannot see 5 vs 10), metric_update=True, **dropped=[]**, t_invalid=2023/06/01, and
  current_facts returns ONLY the 10K fact. The archetypal class the lexical pool was BUILT for is the class
  it destroys. The same gate approves "flight at 6:15 on Friday" vs "flight at 9:40 on Monday" (0.755),
  "$1,200 for the laptop" vs "$2,400 for the desk" (0.706), "won 3 games in March" vs "won 5 games in
  April" (0.706). Repro: scratchpad/s4r2/probes/p1_metric.py + the fuzz harness.
- **B2 — CANCELLATION is a pure LLM verdict: no co-signal, no topical gate — and Ma3 just multiplied its
  blast radius.** The cancel loop (:313-328) checks only "id in shortlist" + "candidate is event/planned" +
  "not double-consumed". `cosignals` IS computed for planned candidates and PERSISTED, and never consulted.
  Worse, the planned pool passes `id_filter=None` when the new fact has no entity links (:451-455), so
  EVERY entity-less state/preference fact puts up to 4 arbitrary live plans in front of an 8B model that is
  told to name plans that were "called off". Probe p5: new fact "The user prefers oat milk in coffee."
  cancels "The user is flying to Tokyo for his sister's wedding." [2099/09/14] — dropped=[], and **the
  audit row for that very action records "agrees": false**. Post-Ma3 the event now vanishes from
  current_facts AND facts_overlapping, and the store makes cancelled TERMINAL against re-affirmation
  (semantic_facts.py:470-478) with no un-cancel API. Falsifies supersession.py:24-25/:46 and design
  decision 3 ("the LLM only PROPOSES; deterministic gates DECIDE ... an LLM-only verdict never acts") for
  the second of the module's two write actions. R1 blocked this shape on supersede; it was never applied to
  cancel.
- **B3 — the Ma8 reviewer-safety fix is INSIDE the module docstring: it has never executed.**
  tests/test_conflict_detection.py:1-18 — the file opens its docstring on line 1 and the nine-line guard
  (`_os.environ["AGENTMEM_OS_DB_PATH"] = ...`) sits on lines 3-10, INSIDE that string, above the title.
  Proven, not inferred: with an inherited AGENTMEM_OS_DB_PATH the script created and migrated 17 tables
  (incl. semantic_facts) at the INHERITED path. conftest.py:12-16 states the founder's own .env/setup.sh
  export that variable at a real DB, and the record's own instruction is "run this file directly" — so the
  documented procedure still migrates the production DB. **Third occurrence of the class** (lessons:423,
  lessons:547); BUILD_LOG:1140-1141 claims it landed.
- **B4 — the "subject-guarded" flip is not subject-guarded, and it deletes true current state facts.**
  :492 `flip and _content_word_overlap(cand_text, new_text)` — one shared 5+-char non-stopword. Probe p3,
  three reproductions where the user's TRUE current fact is the one that disappears, dropped=[] each time:
  "The user is studying at Stanford." LOST to "The user left Stanford Stadium before the encore."
  (education); "The user lives in Berlin." LOST to "The user moved from Berlin Hauptbahnhof to the hotel on
  foot." (location); "The user's personal best time in the charity 5K run is 25:31." LOST to "The user had
  a terrible time at the charity 5K run." (sentiment — "best" is in the positive vocabulary). A venue, a
  station and a topic word are not subjects. R1-Ma1 narrowed the surface; the record (:486-491 and
  BUILD_LOG:1119-1123) discloses only the MISS direction of the cost and calls the result a subject guard —
  lessons:180 and lessons:240 verbatim.

**MAJORS.** (Ma1) **Ma3's entire reader-side fix has NO test:** removing the cancelled filter from
current_facts, removing it from facts_overlapping, and flipping `include_cancelled` to True each leave all
211 green (M25/M26/M27); `include_cancelled` has ZERO callers anywhere in the repo. Code written to close a
review finding, shipped untested — lessons:493.
(Ma2) **"pinned both directions" (Ma2 claim, BUILD_LOG:1126) is false.** Coverage shows
llm/supersession.py:142 — the interval-axis REVERSAL return, the Josh case on the occurrence axis — never
executes in the whole suite; and M16 (`c_end < n_start` -> `<=`) survives, so the docstring's "Overlap or
touch = ambiguous" (:128) has no tripwire either.
(Ma3) **"12 minors -> all fixed" (BUILD_LOG:1080) is unsupported and false.** The R1 section contains no
minors record at all. Measured: m1 (M28 mark_event_cancelled rowcount guard), m2 (M14 `_pool` live filter),
m3 (M24 sweep live filter), m4 (M19 `_int_list` bool rejection) and m5 (M29 "CREATED facts only") all still
survive; m6 (judge_failure picks the newest log row, consolidation_v2.py:561-565) and m9 (the smoke's own
criterion) are untouched code. Two are genuinely fixed (m8 unused binds, m12 window in the audit row).
(Ma4) **The "Design decisions of record" section still states five falsified claims** — dec 1 (:943-947,
entity-only shortlist), dec 3(b) (:952-957, the pre-independence co-signal), dec 3(c) (:957-960, "t_occurred
start, fallback t_mentioned" = the mixed-axis rule Ma2 killed), dec 8 (:986-992, "auto-invoked on judge
failure"), dec 9 (:993-997, "EVENT_STATUSES grows 'cancelled'"), plus Build surface :1006. Both CORRECTED
annotations landed, but 120 lines downstream in the round section; the G2 line :1067-1068 still asserts the
falsified "dropped by the status guard" verbatim. The file's own convention is INLINE annotation (see :58).
(Ma5) **The new signal has zero real-corpus exercise.** My smoke rerun reproduces pass C at 7/7 "no
candidates from any pool" — the lexical pool STILL never fires on the real corpus (R1-m10 unchanged), so
the metric-update branch is unit-test-only, and B1 shows what it does when it fires.
(Ma6) **The new gate's only quantitative content is an untested constant:** M6 (0.7 -> 0.0) and M7 (drop
"numbers must differ") both survive all 211 — the exact shape of R1-B2 and lessons:504, recurring inside
the fix for R1-B2.
(Ma7) **The double-consumption guard is unreachable in tests and its test is misnamed.** M18 survives and
coverage confirms :325-327 never executes; test_double_listed_candidate_superseded_not_double_consumed
exercises the TYPE guard (its own assertion is `superseded == []`), so the behavior in its name is untested
and does not occur. lessons:124 + lessons:302.
(Ma8) **The caps are still unpinned:** M33 (_SHORTLIST_CAP 12->100) and M34 (_LEXICAL_SCAN_CAP 300->5)
survive because the tests monkeypatch those constants to the asserted values — the other half of R1-Ma6,
not addressed. They are audited now (Ma4) but not enforced.

**MINORS.** (m1) "Numbers differ" is STRING inequality: "$2,000"/"$2000", "70 kg"/"70.0 kg", "7 am"/"07:00
am" all register as metric updates — the Mem0 #1674 restatement class. (m2) Multiple reversals in one
judgment cost a full rollback per extra candidate and audit "store refused: fact N is already superseded by
M; refusing to rewrite history" for THIS judgment's own action (probe p4-B). (m3) Duplicate ids in the
model's arrays cost the same rollback and audit "changed concurrently" for our own duplicate. (m4)
recover_judgments states no numeric bound (contrast recover_links: "the bound is max_rounds x limit") and
has no product-surface caller — one test calls it, while BUILD_LOG:1116-1117 says "operators/schedulers
invoke the drain". (m5) The smoke still prints "the direction rule holds iff the same old fact loses in
both" and the reproduced run does not meet it; the record still never says the criterion failed (R1-m9).
(m6) judge_failure still patches the newest ConsolidationLog row for the session (R1-m6). (m7) Still
nothing persisted about SUCCESSFUL judgment counts on the log row (R1-m7). (m8) The `_COSIGNAL_COSINE`
comment (:82-88) still claims "our shortlist already guarantees entity overlap" — false for the lexical
pool this same constant admits — and still sources 0.25 to forget_about, which uses 4+-char keyword overlap
and no cosine (R1-B2 named this; not corrected). (m9) Every audit row carries `stripped_cosine` as a
separate field that is always identical to `cosine` — two numbers that are one. (m10) M9 (drop
`txn.rollback()` before retry) survives, but it is an EQUIVALENT mutant: `finally: txn.close()` already
discards the transaction — noted so the next round does not chase it. (m11) facts_as_of includes cancelled
while current_facts excludes it; disclosed in the docstring (:573-577), acknowledged not blocked. (m12)
95% coverage is now 1 point below R1's 96% with two non-HTTP product lines uncovered; if coverage is
reported again, report the uncovered SET, not just the number.

**Refs:** llm/supersession.py:24-25,44-46,82-88,119-149(142),313-328,325-327,451-455,486-492,495-505;
memory/conflict_detector.py:34-65,74-97,119-130; db/semantic_facts.py:468-514,526,545-551,605-641;
llm/consolidation_v2.py:441-446,561-565,637-667; tests/test_conflict_detection.py:1-18;
tests/test_supersession.py:665-695,730-751; CONSOLIDATION_V2_BUILD_LOG.md:943-1001,1006,1067-1068,
1080,1100-1105,1119-1141. Harnesses: scratchpad/s4r2/ (runmut.py 28 + runmut2.py 6 mutations,
runmut_survivors.py; probes/p1_metric.py, p2_pairs.py, p3_reversed.py, p4_retry.py, p5_cancel.py;
import_probe.sh; smoke_r2.log).

**Who needs to know:** Dev-Head — B1 is arithmetic (a cosine that cannot see digits cannot be an
independent numbers signal: compare the numbers themselves, and require a same-metric check that survives
"5K" vs "10K"); B2 needs a deterministic gate on cancellation and a topical restriction on the unrestricted
planned pool; B3 is a two-line move above the docstring; B4 needs a real subject (the entity join table
exists — Stage 3 built it). Bosses: **Stage 4 does not pass round 2.** The R1 fixes are real and pinned
(5/5 demanded mutations caught, smoke reproduces, imports clean, retry loop holds) — but three of the four
new blockers live INSIDE the R1 fixes, which is process.md:493 recurring at full scale. Founder: the judge
can still remove a true fact from your current view in three ways — a "different metric" it cannot see the
difference in (5K vs 10K), a shared word in two unrelated sentences (Stanford the school vs Stanford
Stadium), and a plan cancelled on the model's word alone with the audit row itself recording that the
evidence disagreed. All three are reversible (nothing is deleted) and all three are fixable.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 3 — **BLOCK**

**Claim reviewed:** "R2's 4 blockers + majors + every minor resolved; 220 tests green (37/54/64/60/5);
conflict script 28/28; llm/supersession.py 98% with uncovered set exactly 222-235; smoke reproduces
(demonstration + boundary) plus a NEW Part D live metric supersession."
**Method:** reran the 5 scoped files in the repo (220 passed, 120.0s) and again on a scratch COPY
(shadowing the editable install; 220 passed, 121.5s); per-file collection counts; branch coverage;
tests/test_conflict_detection.py run DIRECTLY against a decoy inherited AGENTMEM_OS_DB_PATH
("PRETEND_PROD.db"); full smoke with real llama3.1; **38 mutations on the scratch copy (never the
working tree)**, survivors re-run against all 5 files; 7 attack probes on isolated scratch DBs; a
500k-string fuzz of the new number arithmetic; 5 adversarial cases against the REAL llama3.1 (3 samples
each). Every shell and harness exported AGENTMEM_OS_DB_PATH to a scratch file BEFORE any import
(lessons:547). **Production DB untouched** — mtime unchanged at Aug 6 23:26 across the whole round;
the decoy path was never created.

**Reproduced honestly — every headline number is exact.** 220/220 with the split EXACT (supersession 37
/ consolidation_v2 54 / fact_entities 64 / semantic_facts 60 / temporal_kg 5). Conflict script 28/28 and
**B3 is genuinely closed**: the guard now sits above the docstring, and with a decoy inherited path the
decoy file was never created (verified by execution, not by diff — third-incident class finally shut).
Coverage llm/supersession.py = 251 stmts, 6 missed, 88 branches, **0 partial = 98%, missing exactly
222-235** (the real-LLM HTTP body) — the claim is precisely true. Smoke reproduces exactly: Rachel
[2023/05/23] superseded by [2023/05/26] in backfill order, t_invalid=2023/05/26, transition + facts_as_of
correct, in-order=0 / out-of-order=1, boundary pass 0; **Part D fires — superseded=[(1,2)],
t_invalid=2023/05/30**, as claimed. **ALL SEVEN demanded mutations turn a NAMED test red**: weaken masked
equality (M1/M1b) and allow two differing values (M2/M2b) -> test_metric_signal_is_digit_aware; drop the
cue gate (M3) and un-filter the entity-less planned pool (M6) -> test_oat_milk_cannot_cancel_the_tokyo_flight;
drop the shared-subject half (M4) -> test_apply_defends_stale_plan_without_shared_subject; revert the flip
guard to content words (M5) -> test_polarity_flip_requires_shared_entity_node; drop the cancelled reader
filters (M26/M27/M28, all three) -> test_cancelled_readers_and_opt_in. R2-Ma1 (untested reader fix) and
R2-Ma6 (untested constants: M11 0.25->0.0 now caught) are CLOSED; caps pinned without monkeypatching
(M9, M10 caught). The **"equal masks imply equal counts" lemma is TRUE for all digit/whitespace/unicode
inputs** (200k fuzz, 0 counterexamples) — and FALSE for one character it forgot (Ma1 below).
**Mutation score 29/37 valid (2 survivors are equivalent mutants) — the best of the three rounds.**

**Verdict: BLOCK — 1 blocker, 8 majors, 12 minors.** Three of R2's four blockers are properly dead. The
fourth was half-fixed: the cue is a real gate; the shared-subject half is a tautology on every path the
product can actually produce, so nothing deterministic constrains WHICH plan a cancellation kills.

**BLOCKER**
- **B1 — the cancellation "shared subject" gate cannot reject anything reachable, and a real llama3.1 run
  deletes a true live plan through it.** llm/supersession.py:408-416 gates on
  `cand["shared_nodes"] or _content_word_overlap(new_text, cand_text)`. Every cancellation candidate can
  only enter the shortlist through the planned pool (:544-558), and that pool's own admission test IS the
  gate: entity-linked facts draw planned candidates from `peer_fact_ids` (shares >=1 node) -> `shared_nodes`
  is True by construction; entity-less facts filter the pool with `_content_word_overlap(fact.fact_text,
  c.fact_text)` (:557) and the gate then re-runs `_content_word_overlap` on the same pair with the same
  argument order (:411). Measured on every reachable candidate (probe p2-C): SUBJECT-GATE-PASSES=True,
  both branches. lessons:504 ("a candidate-selection threshold reused as the DECISION gate is not a
  gate"), third occurrence, and the module's own INDEPENDENCE RULE (:565-570) violated on the second of
  its two write actions. **Live repro, REAL llama3.1, default config, 3/3 deterministic (probes p1+p3):**
  new fact "The user cancelled his climbing gym membership." cancels the live plan "The user plans to
  enter the climbing competition in October." — `{'cancelled': [1], 'dropped': []}`, plan gone from
  current_facts, **while the model's own reasoning string says "which is unrelated to entering a
  competition"**. The cue constrains WHETHER a cancellation may fire; the choice of WHICH plan dies is
  still LLM-only. Aggravating: `mark_event_cancelled` is one-way — the store has no un-cancel/restore
  method (probe p6-C) and cancelled is terminal against re-affirmation, so unlike a supersession this is
  not operator-reversible through any shipped API. Falsifies BUILD_LOG:1200-1204 ("deterministic gate,
  both halves required") and supersession.py:397-402. Corroborating: **M4c (delete the entity-node half
  entirely, keep only content words) SURVIVES all 220** — the node half has no tripwire at all; the only
  test that turns M4 red (test_apply_defends_stale_plan_without_shared_subject:980-1005) calls `_apply`
  with a hand-built snapshot the shortlist cannot produce. Repro: scratchpad/s4r3/probes/p1_cancel_tautology.py,
  p2_cue.py, p3_real_llm.py.

**MAJORS.**
(Ma1) **The removed length check was NOT dead: the lemma is false for a literal '#'.** :141-142 records
"equal masks imply equal token counts (each '#' is one token), so no length check". A '#' already present
in the source text also masks to '#'. Reproduced (p7): `_metric_update("The user's ticket is 7 and it is
3 days old.", "The user's ticket is # and it is 3 days old.")` -> masks equal, `['7.0','3.0']` vs `['3.0']`,
**zip misaligns and returns True** — comparing 7 against 3. Fuzz with '#' in the alphabet: 401/300,000
mask-equal-but-count-unequal, 53 of them return True on a misaligned comparison. This falsifies the
docstring's "the numeric token counts must align" (:129-130), the comment at :141-142, and
BUILD_LOG:1192-1193 ("numeric positions align"); and
test_norm_nums_unparseable_and_len_mismatch_and_shared_subject_drop:940-941 is named for a len-mismatch it
never exercises (the mask check catches that case) — lessons:302. One-line fix: restore `if len(a) != len(b):
return False`.
(Ma2) **Interval touch is pinned in ONE direction only.** M8 (`c_end < n_start` -> `<=`, :196) is caught;
**M8b (`n_end < c_start` -> `<=`, :198) SURVIVES all 220.** The docstring's "Overlap or touch = ambiguous"
has a tripwire on the forward branch and none on the reversal branch. lessons:240, second occurrence, on
the very rule R2 asked to be pinned "both directions".
(Ma3) **The entity-node half of the cancellation gate is untested** (M4c survives) — see B1.
(Ma4) **The record still asserts a falsified line R2 named by reference.** Five INLINE CORRECTED
annotations did land (decisions 1/3/8/9 + the G2 status-guard line, BUILD_LOG:948-952, 968-977, 1008-1012,
1018-1024, 1095-1099) — but the sixth item R2 listed, the **Build surface line (now :1033)**, still reads
"db/models.py (t_invalid, supersession_judgments table, **EVENT_STATUSES+cancelled**)". Doubly false:
`EVENT_STATUSES = frozenset({"occurred","planned"})` (db/semantic_facts.py:58, deliberately unchanged),
and it is not in db/models.py at all.
(Ma5) **The metric-update signal still has ZERO real-corpus exercise.** My smoke rerun: pass C = 7/7
"no candidates from any pool (entity, lexical, planned)" and 0 entity links — the lexical pool has produced
no candidate in ANY judgment across all three real-corpus passes, in three consecutive rounds (R1-m10,
R2-Ma5, now). Part D is real-judge-LLM but STORE-INJECTED, hand-authored facts. So the digit-aware signal
has never fired on an EXTRACTED fact, and the smoke's RESULT line (:170-174) says "store-level metric pass
superseded=1" without saying those two facts were hand-written.
(Ma6) **The cue vocabulary is negation-blind, and the repo already ships the fix.** `_CANCEL_CUE_RE`
(:93-97) fires on "The user did not cancel the pottery workshop.", "The user has not cancelled his trip to
Paris.", "The user refuses to cancel the workshop.", "The user's flight was almost cancelled but went
ahead." (probe p2-A) — and an end-to-end run with the negated text cancels the plan (p2-B).
memory/conflict_detector.py:144-149 already implements `_negated(text, pattern)` with a 40-char preceding
window over `_NEGATIONS` and is used by the polarity path; the cue path does not call it. Under-reach, not
a missing capability.
(Ma7) **`_metric_update` cannot tell a value from an identity when there is only ONE number** — and the
docstring's stated rationale only covers the two-number case. Reproduced True (p4): "apartment 12B"/"14B",
"Route 66"/"Route 95", "flight to Tokyo is JAL 456"/"JAL 789", "The user's child is 7 years old."/"is 4
years old.". End-to-end (p6-A): the 7-year-old fact is superseded, t_invalid set, current_facts returns
only the 4-year-old one — and **real llama3.1 proposes exactly that supersession 3/3** (p3). Measured
prevalence on the real corpus is 0 (Ma5), which is why this is a major and not a blocker (lessons:314) —
but it must be gated or disclosed, not left implied-safe by a rationale that does not cover it.
(Ma8) **The rebuilt `_cosignal` still describes the design it replaced, in its own docstring.** :576-577
says the metric signal is "numbers-stripped texts near-identical (**cosine >= 0.7**)" — the exact
arithmetic R2-B1 killed; there is no cosine in `_metric_update` at all. :572-573 says the flip subject
guard is "**entity/content-word overlap**" while :595 requires `shared_nodes` only, contradicted 20 lines
later by its own comment (:587-594). lessons:180 and lessons:240: the comment must not say more, or other,
than the code does.

**MINORS.** (m1) Notation restatements register as metric updates: '25:31' vs '25.31' -> True, '07:00 am'
vs '7 am' -> True, and '1,5 percent' comma-strips to **15.0** (European decimal). Reproduced end to end
(p6-B: the older fact is superseded by its own restatement). Low harm, but it contradicts the prompt's own
"a mere restatement = NOT superseded". (m2) M16 survives — `_pool` still not pinned to live facts (R1-m2,
R2, now third round). (m3) M19 survives — the sweep's `superseded_by IS NULL` filter untested (R1-m3,
third round). (m4) M20 survives — `_int_list`'s bool rejection untested (R1-m4, third round). (m5) M29
survives — `mark_event_cancelled`'s rowcount race guard, the "race-defended at the store" claim, still has
no tripwire (R1-m1, third round). (m6) M15 (`if not a: return False`) and M30 (the `!= "cancelled"` half
of the terminal-merge guard, db/semantic_facts.py:378) are **equivalent mutants** — both are unreachable
as differences given the conditions that follow; noted so R4 does not chase them, but the terminal-merge
comment (:375-377) credits a check that can never be the deciding one. (m7) `recover_judgments`'
docstring (llm/consolidation_v2.py:643-645) states the runaway guard and the per-fact LLM cost but not the
numeric bound; only BUILD_LOG:1012 states "max_rounds × limit" (R2-m4, partially landed). (m8)
test_double_listed_candidate_superseded_not_double_consumed is still named for a guard that was correctly
deleted. (m9) Cue vocabulary misses "cancellation", "cancels", "calls off", "backed out", "pulled out",
"shelved", "will not be attending" — the mild direction, undisclosed. (m10) `include_cancelled` still has
zero non-test callers; repo-wide, NOTHING outside benchmarks/ and llm/consolidation_v2.py reads semantic
facts at all. (m11) The entity-pool bare-cosine co-signal (0.25) remains the weakest surviving gate; I
probed three realistic coexisting same-entity pairs with real llama3.1 (Emma/MIT+piano at cosine 0.4346,
Google job+volunteering, Rachel colleague+gym) and **the model declined all 3/3** — residual risk, not a
finding. (m12) 98% coverage with 0 partial branches is a real improvement over R2's 95%.

**Refs:** llm/supersession.py:88-97,123-145(141-142),176-206(196,198),397-416(403,408-416),
540-558(557),562-606(572-577,595,599-600); db/semantic_facts.py:58,375-378,468-514,526,545-551,632-634;
llm/consolidation_v2.py:179-203,637-667; tests/test_supersession.py:934-941,945-961,980-1005;
benchmarks/consolidation_v2_stage4_smoke.py:141-174; CONSOLIDATION_V2_BUILD_LOG.md:1033,1182-1249.
Harnesses: scratchpad/s4r3/ (runmut.py 30 + runmut2.py 8 mutations; probes/p1_cancel_tautology.py,
p2_cue.py, p3_real_llm.py, p4_metric_edges.py, p5_entity_cosine.py, p6_misc.py, p7_hash.py; smoke_r3.log).

**Who needs to know:** Dev-Head — B1 needs a cancellation signal that is INDEPENDENT of the pool's
admission test and that binds the cue to THIS plan (e.g. the cue clause must contain the plan's own
distinctive words, or require >=2 shared content words plus a negation check reusing
conflict_detector._negated); Ma1 is one restored line; Ma2 is one mutation-driven assertion; Ma4/Ma8 are
record/docstring corrections at named lines; Ma6 reuses code that already exists. Bosses: **Stage 4 does
not pass round 3, but it is close** — 29/37 mutations caught (best of the arc), every headline number
exact, B3 finally proven by execution, and the reader-side and constant-pinning majors genuinely closed.
The single blocker is the same shape as the last two rounds' blockers: a gate that re-runs the test that
admitted the candidate. Founder: the judge can still cancel a plan you never cancelled — say "I cancelled
my climbing gym membership" and a planned climbing competition is marked off, with the model's own written
reasoning saying the two are unrelated — and cancellation, unlike supersession, has no undo in the code.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 4 (final-intent) — **BLOCK**

**Claim reviewed:** "R3's blocker + 8 majors + 12 minors all resolved; 228 tests green (45/54/64/60/5);
conflict 28/28; llm/supersession.py 98% with the uncovered set exactly 287-300; smoke reproduces
(Rachel demonstration + boundary + Part D store-level metric supersession); the cancellation binding
gate is independent of pool admission and binds the cue to THIS plan."
**Method:** reran the 5 scoped files in the repo (228 passed, 127.3s) and on a scratch COPY shadowing
the editable install (228 passed, 119.9s; `agentmem_os.__path__` asserted to the copy); per-file
collection counts; branch coverage; tests/test_conflict_detection.py run DIRECTLY twice — once on a
scratch path, once with a DECOY inherited AGENTMEM_OS_DB_PATH ("PRETEND_PROD_R4.db", never created);
full smoke with real llama3.1; **22 mutations on the scratch copy only**; 17 deterministic binding
probes; **6 end-to-end cancellation cases against the REAL llama3.1, 3 trials each (18 live runs)**;
one real-pipeline extraction probe to settle reachability. Every shell exported AGENTMEM_OS_DB_PATH to
a scratch file BEFORE any import (lessons:547). **Production DB untouched** — mtime still Aug 6 23:26.

**Every headline number is exact.** 228/228 with the split EXACT (45/54/64/60/5). Conflict 28/28
directly, and 28/28 again with a decoy inherited path that was never created — B3 stays closed by
execution. Coverage llm/supersession.py = **272 stmts, 6 missed, 98 branches, 0 partial = 98%, missing
exactly 287-300** (the real-LLM HTTP body); the module GREW 251→272 statements and held 98% with zero
partial branches. Smoke reproduces exactly: Rachel [2023/05/23] superseded by [2023/05/26],
t_invalid=2023/05/26, in-order=0 / out-of-order=1, boundary=0, **Part D superseded=[(1,2)],
t_invalid=2023/05/30**. **All 12 demanded mutations turn the NAMED test red** — drop the cue (M1) →
test_oat_milk_cannot_cancel_the_tokyo_flight; remove the negation window (M2) and shrink it to 0 chars
(M2b) → test_negated_cue_never_cancels; weaken >=2 shared words to >=1 (M3) / delete it (M3b) →
test_gym_membership_cannot_cancel_climbing_competition; drop the clause requirement (M4) / neutralize
clause extraction (M4b) → test_cue_clause_must_name_the_plan; drop the restored length check (M5) and
re-allow single-number pairs (M6) → test_metric_hash_literal_and_single_number_and_decimals; revert
grouping-only comma stripping (M7) → same test; both interval-touch directions (M8/M8b) → their two
tests (**R3-Ma2 CLOSED**); pool live filter (M9) → test_shortlist_and_sweep_exclude_superseded; bool
ids (M11) → test_int_list_rejects_booleans; weaken the reinstate guard (M12) →
test_reinstate_cancelled_event. **reinstate_cancelled_event is NOT reachable from the judge BY CODE
PATH** (verified, not just by test): the judge calls only supersede() and mark_event_cancelled(), has
no getattr/eval/dynamic dispatch, and the method has zero non-test callers repo-wide.
**Mutation score 19/22; 2 survivors are equivalent/waived, 1 is real (Ma1).**

**Verdict: BLOCK — 1 blocker, 5 majors, 12 minors.** The R3 blocker's CORE defect is genuinely fixed:
the gate is independent of pool admission and rejects real reachable candidates. But its binding
criterion is satisfied by two GENERIC shared words, and a realistic pair kills a true live plan 3/3
with the default model.

**BLOCKER**
- **B1 — two generic shared content words are enough to bind a cancellation, and real llama3.1
  destroys a true live plan 3/3.** llm/supersession.py:137-147. Measured (probe p_e2e.py E2, real
  llama3.1:latest, default config, temperature 0, **3/3 deterministic**): live plan (event/planned)
  "The user plans to join the weekend training camp for the marathon." + new state fact "The user
  cancelled his weekend training session with the physiotherapist." → shared={training, weekend} (2,
  both generic), cue clause contains both → binds=True → model proposes cancelled_ids=[1] → **applied,
  event_status='cancelled'**, the marathon camp gone from current_facts. Second case E1 fires 1/3
  ("cancelled the yoga workshop weekend in Big Sur" cancels "plans to attend the pottery workshop
  weekend in Sonoma"). The calibration is INVERTED: TRUE cancellations of short-named plans REFUSE —
  "The user cancelled the Rome marathon." vs "The user plans to run the Rome marathon." refuses (1
  shared word), same for "yoga class" and "gym trial" — so the discrimination axis (count of >=5-char
  shared words) is orthogonal to correctness. Falsifies supersession.py:117-127 ("one shared word is
  topical coincidence, not identity" — two generic ones are too) and BUILD_LOG:1264-1269 ("binds the
  cue to THIS plan"). **Reachability, measured honestly (p_extract.py, full real pipeline):** today
  this is UNREACHABLE end to end — "I had to cancel my weekend training session with the physio"
  extracts as fact_type=event/status=occurred and judge_fact SKIPS it; "I signed up for the weekend
  training camp" extracts event/occurred, not planned (consistent with the Stage-3 disclosure that the
  'planned' marker is prompt-unreachable). Blast radius today = zero. **The gate IS the entire safety
  story for the parked plans-as-events prompt decision**, which is why it must be right before that
  decision is unparked. Two remedies, either closes it: (a) make the binding discriminative on the
  plan's distinctive word(s), pinned by this repro through judge_fact (a coverage RATIO does not
  separate — measured: false A2 = 2/4 = 0.50, true E1b = 3/6 = 0.50); (b) keep the gate, DOWNGRADE the
  claim, and put the verbatim 3/3 repro in the stage's mandatory disclosures.

**MAJORS.**
(Ma1) **The sweep live-filter "pin" is a tautology and M10 survives all 228.**
tests/test_supersession.py:1136 `assert all(r2.skipped != None or True for r2 in rows_a)` — `X != None
or True` is ALWAYS True; :1137 also cannot fail because judge_fact skips superseded facts before the
LLM call. Deleting `SemanticFact.superseded_by.is_(None)` from judge_missing (llm/supersession.py:712)
survives all 5 files. BUILD_LOG:1294 "m2/m3 pool and sweep live-filters pinned ... never sweep" is half
false — the pool half IS pinned (M9 caught). FOURTH round (R1-m3, R2, R3-m3), and this round the team
answered a survivor with a no-op assertion. Behavioral risk is low (one wasted judge_fact writing a
skip row, no wrong write); the finding is the false pin plus the rubber stamp (lessons:124).
(Ma2) **The R3-Ma5 smoke-label fix did not land.** BUILD_LOG:1281-1282 claims "the smoke RESULT line
now states Part D is store-injected/hand-authored". benchmarks/consolidation_v2_stage4_smoke.py:171-175
still prints "...store-level metric pass superseded={n}" — the exact wording R3 quoted and rejected (my
rerun printed "store-level metric pass superseded=1"); the file contains no "hand-authored"/
"hand-written"/"not extracted" anywhere. The other half of Ma5 (the never-fired-on-an-extracted-fact
disclosure) DID land at BUILD_LOG:1282-1285.
(Ma3) **There is no STAGE 4 HONEST CLAIMS OF RECORD block.** Stage 3 has one (BUILD_LOG:871-896,
"critic-approved wording", with MANDATORY DISCLOSURES); the Stage 4 record ends at :1314 without an
equivalent. A stage cannot close without it, and lessons:118-122 requires the prior stage's "Stage N+1
will do X" items to be answered as line items here.
(Ma4) **The module's design-of-record docstring still describes the KILLED cosine design.**
llm/supersession.py:33-34 — "the metric-update signal (numbers differ, numbers-stripped texts
near-identical)". The code requires masked texts EXACTLY identical, >=2 numeric tokens and EXACTLY ONE
differing value; two-differ and single-number pairs both REFUSE. Both phrases overstate breadth toward
MORE supersession. Direct repeat of lessons:624-630 — Ma8 was fixed inside `_cosignal` (:631-634 now
correct) and the same phrase survived one level up in the same file.
(Ma5) **The m5 waiver's downgrade was not applied where the claim lives.** BUILD_LOG:1298-1303 says the
claim is downgraded to "guarded by the same rowcount pattern", not "race-proven"; BUILD_LOG:1060 still
reads, under "Failure paths pinned:", "cancellation only reaches live planned events, race-defended at
the store". M14 (delete mark_event_cancelled's `updated != 1` guard, db/semantic_facts.py:506-510)
survives all 228 — nothing pins it. Same shape as R3-Ma4. **The waiver itself is HONEST and I accept
it** (supersede()'s 2-process test proves the pattern; an event-listener interleave harness for
mark_event_cancelled is disproportionate) — only the downgrade text is missing at :1060.

**MINORS.** (m1) The negation check reads only the FIRST cue occurrence (:131-136) while the "reused"
conflict_detector._negated (:144-149) checks EVERY occurrence — "The user cancelled his climbing gym
membership; he did not cancel the climbing competition." BINDS to the climbing-competition plan (the
negated clause supplies the second shared word); the reversed sentence refuses. Real model declined
3/3. (m2) The m9 vocabulary extension added "cancellation", creating a new false-cue class: "travel
insurance now includes trip cancellation coverage for the Tokyo conference" binds to a Tokyo-conference
plan; model declined 3/3. (m3) Hypothetical ("may cancel ... if it rains"), third-party ("The organiser
cancelled ... refund policy") and question forms all bind; model declined 3/3; extractor realism makes
the question form unlikely, agreed. (m4) Negation beyond the 40-char window binds — inherited bound,
not disclosed in the docstring that says only "40-char negation window, reused". (m5) Clause splitting
on `[.;!?]` (:99) breaks on abbreviations/anaphora: "The pottery workshop is at 5 p.m. It was
cancelled." selects "It was cancelled" and REFUSES a true cancellation — mild direction, undisclosed.
(m6) Plans whose distinctive words are <5 chars are structurally uncancellable (Rome marathon, yoga
class, gym trial all refuse a verbatim true cancel) — measured, undisclosed. (m7) `_content_words`
counts structural verbs — "plans"/"attend" are content words of every plan text and inflate
len(shared). (m8) M13 (drop the `!= "cancelled"` half of the terminal-merge guard,
db/semantic_facts.py:381) is an EQUIVALENT mutant by analysis — consistent with R3-m6, not chased.
(m9) Ma7's single-number refusal also CLOSED R3-m1's notation residuals ('25:31'/'25.31', '07:00 am'/
'7 am', '1,5'/'15' all now False, measured); BUILD_LOG:1306-1307 still lists them open — inaccurate in
the safe direction. (m10) `judge_fact` assumes the model reply is a dict (`raw.get` at :416/:451); a
JSON list or string raises AttributeError — contained by the per-fact `except Exception` in
judge_missing:729 and consolidate_session:550 and audited, but untested. (m11) Unchanged: `include_
cancelled` has zero non-test callers and nothing outside benchmarks/ and llm/consolidation_v2.py reads
semantic facts at all. (m12) Coverage grew 251→272 statements and held 98% with 0 partial branches and
the same uncovered set — the new code is genuinely covered, not diluted.

**Refs:** llm/supersession.py:33-34,99,111-148(131-136,137-147),151-210,241-271,397-473(467-472),
530-616,618-662,692-736(712); db/semantic_facts.py:378-385,471-517(506-510),519-560;
llm/consolidation_v2.py:539-557,596-616,638-670; tests/test_supersession.py:937-1010,1015-1099,
1110-1137(1136),1140-1169; benchmarks/consolidation_v2_stage4_smoke.py:120-175;
CONSOLIDATION_V2_BUILD_LOG.md:871-896,1033-1039,1060,1254-1314(1264-1269,1281-1285,1294,1298-1307).
Harnesses: scratchpad/s4r4/ (runmut.py 22 mutations + mut_results.json; p_binds.py 17 deterministic
cases; p_e2e.py 6 cases x 3 real-llama3.1 trials; p_extract.py real-pipeline typing probe; p_ratio.py;
p_misc.py; smoke_r4.log).

**Who needs to know:** Dev-Head — B1 needs the binding to discriminate on the plan's DISTINCTIVE
word(s) (a plain coverage ratio does NOT separate: false 0.50 vs true 0.50 — measured), pinned by the
physio/marathon repro through judge_fact, OR an honest downgrade + disclosure; Ma1 is one real
assertion replacing a tautology; Ma2/Ma4/Ma5 are three text edits at named lines; Ma3 is the stage's
missing honest-claims block. Bosses: **Stage 4 does not pass round 4.** Everything measurable is exact
(228 with the exact split, 98% with the exact uncovered set, conflict 28/28 with a decoy path, smoke
reproduces including Part D), the mutation score is the best of the arc (19/22, all 12 demanded
mutations red on the NAMED test), reinstate is proven operator-only by code path, and R3-Ma1/Ma2/Ma7/
m1/m4 are genuinely closed. What blocks is one measured wrong write and four record/description
over-claims. Founder: the cancellation gate got much stronger but is still guessing — say "I cancelled
my weekend training session with the physio" and a planned marathon training camp is marked off, three
times out of three. It cannot happen in the product today (the extractor types cancellations as past
events, so the judge never sees them) — it becomes reachable the moment you unpark the plans-as-events
prompt decision. And there is now an undo (an operator-only reinstate), which there was not last round.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 5 (closing-intent) — **BLOCK**

**Claim reviewed:** "R4's blocker + 5 majors + 12 minors all resolved; _cancellation_binds v3 =
CONTAINMENT (per non-negated cue occurrence, ALL occurrences checked, the cue clause's object words —
4-char floor, cue spans removed, clause split on enders + and/but — must be non-empty and a SUBSET of
the plan's words); 233 tests green (50/54/64/60/5); conflict 28/28; coverage llm/supersession.py 98%
uncovered EXACTLY 312-325; smoke reproduces with the honest Part-D label."
**Method:** reran the 5 scoped files in the repo (233 passed, 123.8s) and on a scratch COPY shadowing
the editable install (233 passed, 124.6s; `agentmem_os.__path__` asserted to the copy); per-file
collection counts; branch coverage; tests/test_conflict_detection.py DIRECTLY twice — once on a scratch
path, once with a DECOY inherited AGENTMEM_OS_DB_PATH ("PRETEND_PROD_R5.db", never created); full smoke
with real llama3.1; **10 mutations on the scratch copy only** (the 6 demanded + M10 re-verify + 3
survivor hunts); 26 deterministic binding probes; **4 cancellation cases x 3 real-llama3.1 trials (12
live runs)** plus 3 mocked-proposal end-to-end writes. Every shell exported AGENTMEM_OS_DB_PATH to a
scratch file BEFORE any import (lessons:547). **Production DB untouched** — /Volumes/Sahith_SSD/
AgentMem-OS/db/agentmem_os.db mtime still Aug 6 23:26; repo file set unchanged.

**Every headline number is exact.** 233/233 with the split EXACT (50/54/64/60/5), in the repo and on
the shadow copy. Conflict 28/28 twice, decoy path never created. Coverage llm/supersession.py = **284
stmts, 6 missed, 102 branches, 0 partial = 98%, missing exactly 312-325** (the real-LLM HTTP body); the
module grew 272→284 and held 98% with zero partial branches. Smoke reproduces: in-order=0,
out-of-order=1 (Rachel [2023/05/23] superseded BY [2023/05/26], t_invalid=2023/05/26, superseded_at
set), boundary=0, Part D superseded=1 with the label "STORE-INJECTED, HAND-AUTHORED facts, real judge
LLM: the metric signal has never fired on an EXTRACTED fact". **All 6 demanded v3 mutations turn a
NAMED test red** — M1 drop subset → 6 tests incl. all four R4 pins; M2 drop non-empty →
test_pronoun_only_cancellation_names_nothing; M3 first-occurrence-only negation →
test_negation_checked_at_every_cue_occurrence; M4 re-admit cue leakage → 8 tests incl.
test_oat_milk_cannot_cancel_the_tokyo_flight; M6 remove and/but split →
test_double_listed_candidate_type_guard_then_cancellation. **M5 (4-char floor → 5) SURVIVES all 233.**
Extra hunts: M12 reverse containment → 3 red, M13 intersection-only (the R3 rule) → 4 red (direction and
strength are genuinely pinned); M11 empty _S4_EXTRA_STOP survives. **Mutation score 8/10.**
**R4-Ma1 CLOSED BY EXECUTION** (M10, deleting the sweep's live filter, now turns
test_shortlist_and_sweep_exclude_superseded red — it survived all 228 last round). **R4-Ma2 CLOSED**
(smoke label landed, grep + live run). **R4-Ma4 CLOSED** (module docstring bullet 3 states the
digit-aware truth). **R4-Ma5 CLOSED** (BUILD_LOG:1060-1062 carries the downgrade + waiver inline at the
claim). R4-B1's core defect is genuinely dead: the physio/marathon repro refuses, gym/climbing refuses,
insurance-noun refuses, pronoun-only refuses, Rome binds.

**Verdict: BLOCK — 0 blockers, 3 majors, 10 minors.** Nothing here is a wrong write reachable in the
product. What blocks the CLOSE is that two of the v3 gate's own load-bearing pieces are not what the
record says they are, and one is unpinned.

**MAJORS**
- **(Ma1) Negation is defeated by clause-selection-by-STRING — measured wrong write.**
  llm/supersession.py:160-161. The negation test is position-accurate (`m.start()`, :156-158) but the
  clause is chosen by `next((c for c in _CLAUSE_SPLIT_RE.split(new_text) if m.group(0).lower() in
  c.lower()), new_text)` — the FIRST clause containing the cue STRING, not the clause containing THIS
  match. When the same cue surface form occurs twice, a non-negated occurrence is evaluated against a
  NEGATED clause's words. Repro (measured, deterministic): new state fact "The user did not cancel the
  pottery workshop weekend, but did cancel the pottery class." + planned event "The user plans a pottery
  workshop weekend." → clause examined = "The user did not cancel the pottery workshop weekend" →
  named={pottery,workshop,weekend} ⊆ plan → binds → **judge_fact applies it, event_status='cancelled'**
  on a text that says the user did NOT cancel it. (The 40-char window saves the semicolon variant; ", but
  did " pushes the "not" out of range.) Mirror-image miss, safe direction: "cancelled the newsletter
  subscription and cancelled the Rome marathon" REFUSES a true cancellation. Falsifies :133-135 ("Per
  non-NEGATED cue occurrence (every occurrence checked — R4-m1: checking only the first let a negated
  second mention through)") and BUILD_LOG:1339/1363 ("multi-occurrence negation ... pinned"). The pin at
  tests:1214-1224 cannot catch it (its two cue forms differ: "cancelled" vs "cancel"). Reachability,
  honest: real llama3.1 declined 3/3 (it proposed supersession, killed by the type guard), extraction
  types cancellations as past events, plans-as-events is parked — blast radius today ZERO. Remedy (a)
  select the clause by match POSITION and pin with this exact text; or (b) correct :133-135, BUILD_LOG,
  and the claims block to say negation holds per occurrence ONLY when the cue form does not appear in an
  earlier clause, with this repro verbatim.
- **(Ma2) The 4-char floor — the R4-B1 fix's own key ingredient — is UNPINNED; M5 survives all 233.**
  llm/supersession.py:117 (floor), :101-107 (rationale), BUILD_LOG:1334-1336 ("4-char floor, recovering
  the distinctive short words the 5-char rule destroyed: 'yoga', 'rome'"). Setting the floor back to 5
  leaves the suite 233/233 GREEN while measurably breaking discrimination: "The user cancelled the Rome
  marathon." then BINDS to "The user plans to run the Boston marathon in April.", and "cancelled the yoga
  class" BINDS to "plans to take the pottery class" (both refuse at floor 4). The pin the team wrote
  (tests:1205-1211, the Rome POSITIVE) cannot fail at either floor — containment is symmetric, so
  dropping 'rome' from BOTH sides leaves {marathon} ⊆ {marathon}. The floor is only visible on a
  NEGATIVE. Remedy: one assertion — Rome-cancel vs BOSTON-marathon plan must be False — then re-run the
  floor mutation and watch it go red (lessons:655-658, the team's own rule).
- **(Ma3) The rewritten gate's KILLED self-description survives at the CALL SITE and in a test comment —
  third round of this exact class.** llm/supersession.py:490-494 still reads "(negation-aware cue + >=2
  shared content words + the cue clause naming the plan)"; tests/test_supersession.py:1165 still reads
  "The third binding half: >=2 shared words overall". Both describe the v2 rule R4-B1 killed. Direction is
  safe (understates strictness), but this is a direct repeat of lessons:622-630 ("grep the changed
  function for the falsified phrase") after R3-Ma8 and R4-Ma4 fixed the same phrase one and two levels up
  in the same file.

**CLOSE-OUT CONDITION (not counted as a major — I am the artifact's source).** There is still no STAGE 4
HONEST CLAIMS OF RECORD block in CONSOLIDATION_V2_BUILD_LOG.md (the file ends at :1381; BUILD_LOG:1355
defers it to this verdict). The critic-approved wording is delivered with this verdict and must be pasted
verbatim, including the Stage-3 forward items answered as line items (lessons:114-122).

**MINORS.** (m1) M11 (empty `_S4_EXTRA_STOP`) also survives all 233 — R4-m7's "'plans'/'attend' added to
the structural stop extension" has no tripwire; mixed direction, low stakes. (m2) Containment does NOT
require the plan's DISTINCTIVE word, only the absence of foreign words: "The user cancelled his weekend
training." BINDS the marathon-camp plan (the R4-B1 pair under a shorter, natural phrasing); "The user
cancelled dinner." BINDS "plans to take the family to the Tokyo conference dinner"; "cancelled the annual
retreat" binds a 7-word plan. Real llama3.1 declined to propose all of these 3/3 (12 live runs); mocked
proposals land the write. Must be disclosed. (m3) Composite/verbose plans bind by word UNION with no
adjacency: "The user cancelled the Tokyo trip." binds "plans to attend a wedding in Tokyo and a separate
trip to Kyoto." (m4) Measured true-positive cost: 5 of 8 natural TRUE-cancellation phrasings REFUSE
("...in April", "...because of an injury", "his Rome marathon entry", "The Rome marathon trip fell
through", "is not going to the Rome marathon anymore") — safe direction, but the record's "Rome binds"
reads as if true cancellations work generally. (m5) Stemless matching: "marathons" ≠ "marathon" → refuse
(miss); possessives are fine ("Tokyo's" → tokyo). (m6) Question form binds ("Did the user cancel the Rome
marathon?") — R4-m3, re-measured. (m7) BUILD_LOG:1309-1310 (R3 section) still says notation restatements
"remain a disclosed residual" with no inline correction, while :1365-1367 says they are CLOSED — same
shape as R4-Ma5, safe direction. (m8) tests:1231-1234 binds `why` and never asserts on it, weaker than
its three siblings. (m9) Harness honesty: my first M5/M6 patches inserted a comment inside a call and
produced a syntax error (bogus "7 failed / 92 errors"); both were re-run clean and only the clean numbers
appear above. (m10) Standing: `include_cancelled` still has zero non-test callers; nothing outside
benchmarks/ and llm/consolidation_v2.py reads semantic facts at all.

**Refs:** llm/supersession.py:100-107,110-118,121-173(133-135,156-158,160-161,166-170),176-235,
299-325,329-420,422-554(485-501,490-494),556-644,646-690,692-764(740); db/semantic_facts.py:471-517;
tests/test_supersession.py:1110-1145,1164-1178(1165),1181-1211(1205-1211),1214-1224,1227-1234,1237-1245,
1248-1265; benchmarks/consolidation_v2_stage4_smoke.py:141-176; CONSOLIDATION_V2_BUILD_LOG.md:871-896,
1045-1066(1060-1062),1257-1317(1309-1310),1319-1381(1330-1343,1355,1363-1375).
Harnesses: scratchpad/s4r5/ (runmut.py + mut_results{,2,3}.json = 10 mutations; p_e2e.py 4 cases x 3
real-llama3.1 trials; probe batches 1-3 = 26 deterministic binding cases + the mocked-proposal write
repro; pkg/ = shadow copy of the repo).

**Who needs to know:** Dev-Head — Ma1 is a 2-line change (select the clause by match position) plus one
test, or an honest wording correction in three places; Ma2 is ONE assertion plus re-running the floor
mutation; Ma3 is two comment edits found by grepping ">=2 shared". Then paste the claims block.
Bosses: **Stage 4 does not close on round 5.** Everything measurable is exact for the third round running
(233 with the exact split on repo AND shadow copy, 98% with the exact uncovered set, conflict 28/28 with
a decoy path, smoke reproduces including the corrected Part-D label), R4-Ma1/Ma2/Ma4/Ma5 are closed and
R4-B1's core defect is genuinely dead — the gate now refuses every false pair R3 and R4 measured. What
blocks is one incomplete fix with a measured (unreachable) wrong write, one unpinned constant that the
fix's headline depends on, and a stale description of the killed rule at the call site. Founder: the
cancellation gate is much better — it now refuses the physio/marathon case and the gym/climbing case that
killed real plans in earlier rounds — but a sentence that says "I did NOT cancel the pottery workshop, I
cancelled the pottery class" can still mark the workshop off, and the setting that makes "Rome" work can
be reverted without a single test noticing. Neither can happen in the product today (cancellations never
reach the judge), and both are short fixes.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 6 (focused confirmation) — **BLOCK**

**Claim reviewed:** "R5's 3 majors + m1 fixed (clause boundaries by POSITION; the 4-char floor pinned by
its NEGATIVE; the killed '>=2 shared' self-description dead at the call site and in the test comment;
_S4_EXTRA_STOP pinned); 236 tests green (53/54/64/60/5); conflict 28/28; coverage llm/supersession.py 98%
uncovered exactly the real-LLM HTTP body (320-333); the R5-resolution section + STAGE 4 HONEST CLAIMS OF
RECORD block appended to the build log."
**Method:** reran the 5 scoped files (236 passed, 122.0s) + per-file collection counts; branch coverage;
tests/test_conflict_detection.py DIRECTLY; **6 mutations on a scratch shadow copy** ($SC/pkg, asserted
`agentmem_os.__path__` AND `tests.*.__file__` both resolve to the copy) — the 4 the round demanded plus 2
containment re-verifies; 26 deterministic binding probes (10 regression + 9 adversarial + 7 abbreviation);
1 mocked-proposal end-to-end WRITE repro; 6 real-llama3.1 live trials (local, free). Every shell exported
AGENTMEM_OS_DB_PATH to a scratch file BEFORE any import (lessons:547). **Read-only proven:** all four
Stage-4 file hashes byte-identical before and after; prod DB mtime still Aug 6 23:26; repo file set
unchanged. **Smoke NOT re-run** (disclosed below).

**Everything the round was dispatched to verify is CLOSED, and every number is exact.**
236/236 with the split EXACT (53/54/64/60/5). Conflict 28/28. Coverage llm/supersession.py = **287 stmts,
6 missed, 102 branches, 0 partial = 98%, missing exactly 320-333**, which I read line-by-line: it is
precisely the `_llm` HTTP body (`urllib.request.Request` → `return json.loads(body["response"])`) and
nothing else. **All 4 demanded mutations turn the NAMED test red, each alone:** M1 revert to string-based
clause selection → `test_position_accurate_clause_selection` (1 failed / 52 passed); M2
first-occurrence-only negation → `test_negation_checked_at_every_cue_occurrence`; M3 floor 4→5 →
`test_four_char_floor_is_load_bearing` (**the R5 survivor is dead**); M4 empty `_S4_EXTRA_STOP` → 6 red
incl. `test_structural_stop_extension_is_load_bearing` (**the second R5 survivor is dead**). Re-verifies:
M5 drop the subset check → 8 red, M6 drop the non-empty check → 2 red. **Mutation score 6/6.** R5-Ma1's
exact wrong-write text now REFUSES with the right reason ("cue is NEGATED; cue clause names things
outside the plan (class)") — the second occurrence was judged on ITS OWN clause — and the mirror binds.
R5-Ma3: `>=2 shared` returns NOTHING in llm/supersession.py or tests/test_supersession.py; the call site
(:498-504) now describes containment. R5-m7: the inline bracketed correction is in place at
BUILD_LOG:1310-1312. AST scan of the test file: **zero tautological asserts** (the `or True` class stays
dead). Claims-block audit: every headline number matches my own reruns; disclosures 1-8 and the Stage-3
forward items match the measured reality of R5's findings — except one measured class that is missing and
one record line that is affirmatively false (Ma1 below).

**Verdict: BLOCK — 0 blockers, 1 major, 3 minors.** This is a ONE-ITEM block on a pre-existing defect the
spot-check surfaced, not a regression: nothing this round broke, and all four R5 items are genuinely dead.

**MAJOR**
- **(Ma1) The clause splitter breaks on abbreviation periods, and every splitter false positive is in
  the UNSAFE direction — measured wrong write, end to end; and BUILD_LOG:1369-1371 states this exact
  class was "verified harmless for cue-first phrasings", which is false.** llm/supersession.py:100
  (`_CLAUSE_SPLIT_RE = [.;!?]|\band\b|\bbut\b`) with :167-170 (position-based clause). A `.` inside an
  abbreviation ends the clause early, DELETING the words that name the specific thing; a smaller named
  set is strictly more likely to be a subset, so truncation always moves toward BINDING. Measured
  (deterministic), cue-FIRST phrasings, 5 of 6 FALSE BIND: "The user cancelled the appointment with Dr.
  Meyer." → clause = "The user cancelled the appointment with Dr" → named={appointment} ⊆ plan → BINDS
  "The user plans an appointment at the downtown clinic."; same for "trip to St. Louis" → "plans a trip
  to Chicago"; "dinner with Mr. Alvarez" → "plans a dinner with the Tokyo team"; "lesson with Mrs.
  Kowalski"; "viewing on Oak Ave. downtown". **Control proves the mechanism:** "with Doctor Meyer" (no
  period) REFUSES with named={doctor, meyer}. **End-to-end write repro** (mocked proposal, scratch copy):
  judge_fact writes `event_status='cancelled'` on the downtown-clinic appointment, `dropped=[]` — the
  gate is the last defense and it passed. NOT a regression: I ran the same probes against the R5
  string-based mutant and the results are IDENTICAL, so this predates the R5-Ma1 fix. Blast radius today
  ZERO, measured honestly: real llama3.1 proposed cancellation 0/3 on both cases (it proposed
  supersession for Dr. Meyer, killed by the type guard), and cancellation is prompt-unreachable
  (disclosure 4). What makes it blocking is the RECORD: BUILD_LOG:1370 says the cue-first abbreviation
  family was verified harmless — my R4-m5 raised only the cue-LAST case, which is the safe direction —
  and the claims block's disclosure 5 enumerates the gate's misses as "all measured" while implying the
  gate is safe whenever the sentence names something distinctive. Here the sentence names Meyer/Louis/
  Alvarez and the gate binds anyway. Remedy (a) code: do not split on a `.` that terminates a short
  capitalized token (or require whitespace + a following lowercase word), then pin with the Dr. Meyer
  text and re-run the split mutation; or (b) record: strike "verified harmless for cue-first phrasings"
  from BUILD_LOG:1369-1371 with an inline bracketed correction, and add to disclosure 5, verbatim: "an
  abbreviation period ('Dr.', 'St.', 'Mr.', 'Ave.') ends the cue clause early and DELETES the words that
  name the specific thing, so 'cancelled the appointment with Dr. Meyer' binds a different planned
  appointment; every clause-splitter false positive is in the binding direction."

**MINORS.** (m1) The negation window is still NOT clause-bounded — llm/supersession.py:163 keeps a raw
40-char lookback that crosses the boundaries :159-160 now computes, so a negation in the PRIOR clause
suppresses a TRUE cancellation in this one: "The user did not cancel the pottery class but cancelled the
pottery workshop weekend." vs "The user plans a pottery workshop weekend." → refuses, why="cue is
NEGATED; cue is NEGATED". Safe direction, but it half-applies the fix's own stated principle
(BUILD_LOG:1394-1396 "span-accurate negation demands span-accurate clauses") and disclosure 5 discloses
only the opposite direction (">40 characters ... invisible"). (m2) BUILD_LOG:1408-1409 "the
positive-control reason asserted" does not close R5-m8 as written: the insurance test still binds `why`
and never asserts on it (tests/test_supersession.py:1230-1233), and the NEW pin repeats the pattern —
`test_position_accurate_clause_selection` (tests:1275) binds `why` unused, so it asserts only THAT the
gate refused, never that the SECOND clause was the one examined; add `assert "class" in why`. AST scan
found exactly these two unused reason bindings. (m3) Method disclosure, mine: I did NOT re-run the smoke
this round — I grep-verified the honest Part-D label at benchmarks/consolidation_v2_stage4_smoke.py:174-176
and reasoned it unchanged by construction (statement count 284→287 = +3, exactly the clause-boundary
edit; branches unchanged at 102; the smoke's cancellation path measured 0 in R5). The claims block's
"demonstrated end to end ... Rachel [2023/05/23] superseded by [2023/05/26]" rests on R5's run, not mine.

**Refs:** llm/supersession.py:100,104-107,110-118,121-181(133-135,159-160,163-170),318-333,498-511;
tests/test_supersession.py:1164-1177,1226-1233(1230-1233),1236-1244,1268-1284(1275),1287-1296,1299-1308;
CONSOLIDATION_V2_BUILD_LOG.md:1310-1312,1366-1374(1369-1371),1385-1417,1419-1494(disclosure 5).
Harnesses: scratchpad/s4r6/ (pkg/ = shadow copy; runmut.py + mut_results.json = 6 mutations;
abbrev_probe.py = fixed-vs-mutant comparison; pkg/agentmem_os/tests/test_r6_critic_probe.py = the
end-to-end write repro, scratch copy ONLY).

**Who needs to know:** Dev-Head — one item. Either one regex guard in `_CLAUSE_SPLIT_RE` plus the Dr.
Meyer pin, or two record edits (strike BUILD_LOG:1370's false "verified harmless", extend disclosure 5).
Then m1's one-line disclosure and m2's one assertion. Bosses: **everything R6 was dispatched to confirm
is confirmed** — 236 with the exact split, 98% with the exact uncovered set, conflict 28/28, all four
demanded mutations red on the named test (both R5 survivors killed), the grep clean, the record's numbers
exact. Stage 4 does not close only because the spot-check found a fourth-round-running instance of the
same class: the record says a class was verified and the measurement says the opposite. Founder: the
cancellation gate is genuinely much better than three rounds ago, but "I cancelled my appointment with
Dr. Meyer" can still mark a different planned appointment cancelled, because the code treats the dot in
"Dr." as the end of the sentence and throws away the name. It cannot happen in the product today, and
the fix is small.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 7 (final minimal) — **BLOCK**

**Claim reviewed:** "R6-Ma1 fixed in code (lookbehinds refuse a period after a 1-3-letter capitalized
token) + pinned; R6-m1 negation window clause-bounded; R6-m2 both reason assertions prove the mechanism;
the R6 record audited ('verified harmless' struck inline, disclosure-5 addition about abbreviation
periods, smoke rerun); 237 tests (54/54/64/60/5); conflict 28/28; coverage 98% uncovered exactly 332-345;
smoke reproduces incl. Part D — render the FINAL STAGE VERDICT."
**Method:** reran the 5 scoped files (**237 passed, 124.8s**) + per-file collect counts; branch coverage;
tests/test_conflict_detection.py DIRECTLY; **5 mutations on a scratch shadow copy** ($SC/pkg, containment
asserted: `agentmem_os.__path__`, `llm.supersession.__file__` AND `tests.test_supersession.__file__` all
resolve to the copy); a **3-variant A/B** (current vs pre-R6-m1 vs pre-R6-Ma1) to establish causation on
every mismatch; 21 + 14 deterministic binding probes; AST scan of the test file; **smoke RERUN myself**
(local llama3.1, free). Every shell exported AGENTMEM_OS_DB_PATH to a scratch file BEFORE any import.
**Read-only proven:** all six file hashes byte-identical before and after (`shasum -c` all OK); prod DB
mtime still Aug 6 23:26; `git status` byte-identical to session start.

**Everything R7 was dispatched to confirm about the FIXES is confirmed, and every number is exact.**
237/237 with the split EXACT (54/54/64/60/5). Conflict 28/28. Coverage llm/supersession.py = **287 stmts,
6 missed, 102 branches, 0 partial = 98%, missing exactly 332-345**, read line-by-line: precisely the
`_llm` HTTP body (`urllib.request.Request` → `return json.loads(body["response"])`) and nothing else.
**M1 (revert the lookbehinds) turns `test_abbreviation_periods_do_not_truncate_the_clause` red ALONE**
(1 failed / 53 passed) — the R6-Ma1 pin is real and load-bearing. **M2 (drop `and|but`) → 2 red**
(`test_position_accurate_clause_selection`, `test_double_listed_candidate_type_guard_then_cancellation`)
— conjunction splitting still works and is pinned. **M5 (reason drops the named-word list) → 2 red**
while the refusals stay green — R6-m2 CLOSED: both assertions prove the MECHANISM, not the refusal. AST
scan: **zero tautological asserts, zero unused why/reason bindings**. All 7 honorific probes (Dr./St./Mr./
Mrs./Ave./Rev./Sgt.) now REFUSE with the name surviving in the reason; the true-cancellation-with-
honorific control BINDS. R6-m1's own crossing-boundary probe behaves as claimed (pre-R6-m1 refused, now
binds). BUILD_LOG:1370-1374 does carry the inline `[FALSIFIED by R6-Ma1 …]` strike. **Smoke rerun by me,
reproduces exactly:** in-order=0, out-of-order=1 (Rachel [2023/05/23] superseded by [2023/05/26],
t_invalid=2023/05/26, superseded_at set, facts_as_of still shows the old fact at its own date),
boundary=0, Part D superseded=1 under the honest STORE-INJECTED/HAND-AUTHORED label.

**Verdict: BLOCK — 1 blocker, 1 major, 4 minors.** The code fix R6 demanded is real and correctly pinned.
The stage cannot close because the *replacement record line* R6 wrote is itself measurably false, at the
same 5/6 rate, about the residual half of the same regex — and because R6-m1 shipped an undisclosed,
unpinned change in the BINDING direction.

**BLOCKER**
- **(B1) The R6 disclosure-5 addition — the exact wording this round asks me to certify as part of the
  definitive record — is affirmatively FALSE, measured 5/6 in the direction it calls safe.**
  BUILD_LOG:1530-1533 states: *"abbreviation periods no longer truncate clauses (pinned); single-letter
  abbreviations ('p.m.') still split, in the safe cue-first direction only."* Both halves are false. The
  fix (llm/supersession.py:107-109) is scoped to **capitalized** 1-3-letter tokens; a period after a
  **lowercase** abbreviation still truncates the clause, still deletes the naming words, and still moves
  every error toward BINDING. Measured, all cue-FIRST, all wrong writes onto a DIFFERENT plan — **5 of 6**:
  "cancelled the **Friday 6 p.m.** dinner reservation" → clause `"…cancelled the Friday 6 p"` → named
  {friday} ⊆ plan → BINDS "plans a Friday **lunch** reservation"; "the **Tuesday p.m.** pottery class" →
  binds "plans a Tuesday pottery **workshop weekend**"; "the **Monday a.m.** spin session" → binds "plans
  a Monday **swimming** session"; "the **Saturday a.m.** German lesson" → binds "plans a Saturday German
  **exam**"; "the **Rome vs.** Milan trip booking" → binds "plans a Rome **hiking** trip". Only the
  "…, etc. for June" phrasing refused. 3-variant A/B proves this is **PRE-EXISTING, not an R6 regression**
  (identical under pre-R6-Ma1) — what is new is the record asserting the class is safe. "cancelled the
  Friday 6 p.m. dinner reservation" is far more natural in an extracted fact than "Dr. Meyer" ever was.
  This is a **direct repeat of the lesson this team recorded one round ago** (lessons/process.md:702-704:
  *"never accept 'class X verified harmless' unless the record names the probes; re-run the class in the
  direction the probe did NOT cover"*) — R6 struck one unmeasured safety claim and replaced it with
  another unmeasured safety claim about the residual. Remedy (a) code: drop the `[A-Z]` restriction —
  refuse to split on a period after any 1-3-letter token (or require the period to be followed by
  whitespace + a capital) — and extend the R6 pin with the "Friday 6 p.m. dinner reservation" text so M1
  keeps it red; or (b) record: replace the addition with the measured truth, verbatim: *"the lookbehind
  covers CAPITALIZED 1-3-letter abbreviations only; a period after a lowercase abbreviation ('p.m.',
  'a.m.', 'vs.') still truncates the clause and still binds a different plan — 5 of 6 measured cue-first
  phrasings falsely bind."* (a) is preferred for the same reason R6 gave: this gate exists to be right
  BEFORE the parked plans-as-events decision flips.

**MAJOR**
- **(Ma1) R6-m1 made the negation window strictly NARROWER — a change in the BINDING direction — and it
  is neither pinned nor disclosed.** llm/supersession.py:177 `window = new_text[max(c_start, m.start()-40)
  :m.start()]`. Clause-bounding can only shrink the window, so fewer negations are seen and MORE cues
  bind. Measured **5/5 false binds** that the pre-R6-m1 code refused (A/B verified on 3, mechanism
  identical on all 5): "The user did **not** cancel the pottery class **and** cancel the pottery workshop
  weekend." → BINDS "plans a pottery workshop weekend"; same for "will not … and cancel the Tokyo museum
  tour", "has no plans to … and cancel the Rome training camp", "does not want to … and cancel the piano
  recital", "never asked to … and cancel the Denver hotel". A distributed negator elided across `and` is
  now invisible. **Unpinned:** M4 (revert to the raw 40-char lookback) leaves **54/54 green** — nothing in
  the suite detects the reversion, the R5-Ma2 class again (a fix's key ingredient with no negative pin).
  **Undisclosed:** disclosure 5 still says only *"a negation more than 40 characters before the cue is
  invisible"*; a negation in a PRIOR CLAUSE is now also invisible, and that list claims to enumerate the
  gate's residuals "all measured". Honest sizing: the trigger phrasing is strained, blast radius today
  zero (cancellation prompt-unreachable). Remedy: one disclosure clause + one negative pin (the "did not
  cancel X and cancel Y" text must refuse, or must be recorded as an accepted bind with reasoning).

**MINORS.** (m1) **The period branch of `_CLAUSE_SPLIT_RE` is entirely unpinned** — M3 (delete `\.` from
the splitter) leaves **54/54 green**. Errors here are refuse-direction only (longer clause → more named
words → less likely subset), so it is a recall risk, not a wrong-write risk; but the round asked me to
verify sentence-end splitting is pinned and it is not. One pin: a cancellation followed by an unrelated
second sentence must still bind. (m2) **The claims block's own headline is stale**: BUILD_LOG:1444-1445
still reads "236 tests green across the 5 scoped files (53/54/64/60/5)" while the tree is 237
(54/54/64/60/5) — under-claim direction, harmless, but a block nominated as *definitive* must not
disagree with the post-fix line 85 lines below it. (m3) The disclosure-5 addition lives in the **R6
round-notes section (1530-1533), not inside the MANDATORY DISCLOSURES list (1467-1479)** — anyone quoting
the mandatory disclosures verbatim omits it. Fold it in. (m4) Wording nit: "demonstrated end to end on a
real corpus **in both session orders**" (1430-1432) reads as if supersession fired in both; my rerun
measured 0 in-order / 1 out-of-order. Disclosure 2 corrects it, so this is a nit, not a defect.

**Refs:** llm/supersession.py:100-109(107-109),119-127,159-193(163-177,181-190),332-345;
tests/test_supersession.py:1213-1223,1226-1234,1269-1286(1281),1289-1298,1301-1310,1313-1336(1322);
CONSOLIDATION_V2_BUILD_LOG.md:1366-1380(1369-1374),1422-1497(1444-1445,1467-1479),1499-1534(1520-1525,
1527-1534); lessons/process.md:687-704(702-704),706-715. Harnesses: scratchpad/s4r7/ (pkg/ = shadow copy
+ sitecustomize; runmut.py + mut_results.json = 5 mutations; causation.py = the 3-variant A/B;
probe.py / probe2.py = 35 binding probes; astscan.py; smoke_r7.txt = my smoke run).

**Who needs to know:** Dev-Head — **two items**, both small. B1: delete `[A-Z]` from the three lookbehinds
(or require whitespace+capital after the period) and add the "Friday 6 p.m. dinner reservation" case to
the existing pin; if you choose the record route instead, use my verbatim replacement wording. Ma1: one
disclosure clause + one negative pin for the distributed-negation bind. Then m1-m3 are three one-line
edits. Bosses: **the R6 code fix is genuinely good and genuinely pinned** — M1 kills exactly the named
test alone, the honorific family is closed 7/7 with the true-positive control intact, R6-m2 is closed by
mutation, the smoke reproduces under my own run, and every headline number is exact for the fourth round
running. Stage 4 does not close because the record wrote a second unmeasured "this class is safe" line
immediately after being blocked for the first one — the same regex, the same direction, the same 5/6
rate. Founder: the fix stopped "cancelled my appointment with **Dr.** Meyer" from marking the wrong
appointment cancelled; it did not stop "cancelled the **Friday 6 p.m.** dinner" from doing exactly the
same thing, and the write-up says that case is safe. Still impossible in the product today. Still a small
fix.

## 2026-08-07 — Critics — Consolidation v2 STAGE 4 (per-fact supersession), G3 ROUND 8 (minimal) — **PASS-WITH-NOTES** → **STAGE VERDICT: PASS-WITH-NOTES**

**Claim reviewed:** "R7-B1 fixed (case-blind, TOKEN-ANCHORED lookbehinds) + pinned both ways; R7-Ma1
reverted to the wide raw 40-char negation window + pinned; the four record edits made (falsified R6
addition struck inline, splitter/negation behavior folded INSIDE mandatory disclosure 5, stale count
reconciled, both-orders phrasing corrected to BACKFILL); 239 tests (56/54/64/60/5); conflict 28/28;
coverage 98% uncovered exactly 341-354; smoke reproduces — render the STAGE VERDICT."
**Method:** reran the 5 scoped files (**239 passed, 125.0s**) + per-file collect counts;
`tests/test_conflict_detection.py` DIRECTLY as a script; branch coverage; **4 mutations on a scratch
shadow copy** (containment asserted INSIDE pytest: `agentmem_os.__path__[0]`,
`llm.supersession.__file__` and `tests.test_supersession.__file__` all resolve to the copy — the
containment test passed as a real test, not a print); 41 deterministic binding probes in 6 families
(conjunction truncation / growth-flip / string-start abbreviation / the full R7 honorific+lowercase set /
true-positive controls / negation scope); AST scan of the test file; **smoke RERUN myself** (local
llama3.1, free); line-by-line audit of all four record edits. Every shell exported
`AGENTMEM_OS_DB_PATH` to a scratch file BEFORE any import (lessons:547). **Read-only:** all five source
hashes byte-identical before and after (`llm/supersession.py` a71a87dc…, `tests/test_supersession.py`
6a607a32…, `CONSOLIDATION_V2_BUILD_LOG.md` 51561c69…, the smoke 4d2fd309…, `memory/conflict_detector.py`
caedfa66…); `git status` tracked-file set identical to session start — **one exception disclosed in n4.**

**Everything R8 was dispatched to verify is CLOSED, and every number is exact for the fifth round
running.** 239/239 with the split EXACT (56/54/64/60/5). Conflict **28/28, 100%**. Coverage
llm/supersession.py = **287 stmts, 6 missed, 102 branches, 0 partial = 98%, missing exactly 341-354**,
read line-by-line: precisely the `_llm` HTTP body (`req = urllib.request.Request(` → `return
json.loads(body["response"])`) and nothing else.

**The three demanded mutations each turn the NAMED test red, ALONE (1 failed / 55 passed in every case):**
- **M1** revert to capitalized-only lookbehinds (`[A-Za-z]`→`[A-Z]` in each lookbehind head) →
  `test_lowercase_abbreviations_do_not_truncate_and_negation_errs_wide` RED, and the Dr. Meyer pin stays
  GREEN — the discriminating result: the pin is specific to the lowercase half R7-B1 raised.
- **M2** delete the period branch → `test_sentence_period_split_is_load_bearing` RED. R7-m1 CLOSED; the
  branch that was unpinned in R7 is now load-bearing.
- **M3** re-apply R6-m1's clause-bounding (`max(0,…)`→`max(c_start,…)`, llm/supersession.py:186) →
  `test_lowercase_abbreviations_do_not_truncate_and_negation_errs_wide` RED. R7-Ma1 CLOSED; the wide
  window is now pinned by a NEGATIVE, which is exactly what R7-Ma1 said was missing.
- **M4 (mine, unasked — a record-narrative check)** un-anchored lookbehinds (`(?<![A-Za-z])…`, i.e. the
  "first fix attempt" the R7 record describes) → `test_sentence_period_split_is_load_bearing` RED. The
  record's story about WHY the token anchor exists ("'marathon.' matched through its last two letters,
  and the sentence-split pin caught it") is **verified by mutation, not taken on trust.**

**Behavioral verification, 41 probes.** The R7-B1 family is **closed 12/12**: all seven honorifics
(Dr./St./Mr./Mrs./Ave./Rev./Sgt.) and all five lowercase forms (p.m. ×2, a.m. ×2, vs.) REFUSE, each with
the naming word surviving into the reason (`…names things outside the plan (meyer)`, `(dinner)`,
`(spin)`, `(lesson)`, `(booking, milan)`). True-positive controls **4/4 BIND** (verbatim; true
cancellation WITH an honorific; cancellation followed by a second sentence; a sentence ending in a
3-letter word before the cue). Negation scope **0/5 binds** — all five R7-Ma1 "did not cancel X and
cancel Y" shapes refuse. The string-start abbreviation edge (`"Dr. Meyer's appointment was cancelled."`)
is **behaviorally inert** — the split fires there but only strips a ≤3-char token the 4-char floor
discards anyway; 0/2 binds, names survive. The wide window's disclosed recall cost reproduces exactly as
disclosure 5 describes (prior-clause negation suppresses a true cancellation → refuse; >40-char negation
invisible → bind).

**Record audit, all four edits, line by line — all four are real and all four are accurate:**
1. **Strike** — `BUILD_LOG:1540-1546` carries `**[FALSIFIED by R7-B1 — this addition was a SECOND
   unmeasured 'class is safe' sentence, broken at the same 5/6 rate …]**` immediately after the falsified
   R6 sentence (1536-1539), with the measured truth and the fix. Correct.
2. **Fold-in** — disclosure 5 runs 1468-1484 (item 6 begins 1485); the splitter and negation clauses are
   at **1480-1484, INSIDE it**. R7-m3 closed: a verbatim quoter of the mandatory disclosures now gets
   them. Both clauses are TRUE as written ("a period after ANY 1-3-letter token" — verified case-blind;
   "the negation lookback deliberately crosses clause boundaries" — verified at :186) — with one measured
   exception to a parenthetical, n2 below.
3. **Count reconciliation** — `BUILD_LOG:1448` `**[Counts updated through R7: 239 tests,
   56/54/64/60/5.]**`, three lines under the stale 1445 sentence, matching my own rerun exactly. R7-m2
   closed by the inline-correction pattern the log already uses. See n3.
4. **Both-orders** — 1431-1432 now reads "(run in both session orders; the supersession fired in the
   BACKFILL order — disclosure 2 carries the 0-vs-1 asymmetry)". "BACKFILL" is the smoke's **own** label
   (`benchmarks/consolidation_v2_stage4_smoke.py:11,130` — "B. OUT OF ORDER (backfill)"), and my rerun
   measured in-order=0 / out-of-order=1. R7-m4 closed and terminologically consistent.

**Smoke rerun by me, reproduces exactly:** in-order=**0**, out-of-order=**1** — Rachel [2023/05/23] "The
user plans to catch up with Rachel soon." superseded by [2023/05/26], **t_invalid=2023/05/26**,
**superseded_at set=True**, transition text present, and `as-of 2023/05/23` STILL returns the old fact at
its own date; boundary-case=**0**; **Part D superseded=1** under the honest "STORE-INJECTED,
HAND-AUTHORED facts, real judge LLM: the metric signal has never fired on an EXTRACTED fact (disclosed)"
label. AST scan of tests/test_supersession.py: **zero tautological asserts, zero unused why/reason
bindings** — R6-m2 and R7-m5 stay closed under two new tests.

**VERDICT: PASS-WITH-NOTES — 0 blockers, 0 unresolved majors, 4 notes.** Stage 4 closes. The **STAGE 4
HONEST CLAIMS OF RECORD block (BUILD_LOG:1422-1497, as amended through R7) is the stage's definitive
record**, subject to note n1, which is a one-clause addition to disclosure 5 and does not reopen the
stage. I could not find a single false number, a single false statement, or a single unpinned fix in it.

**NOTES (must ride along).**

- **(n1) MANDATORY record addition — the CONJUNCTION half of `_CLAUSE_SPLIT_RE` truncates exactly like
  the period half did, at the same 5/6 rate and in the same BINDING direction, and disclosure 5 does not
  name the mechanism.** llm/supersession.py:112-115 — R6-Ma1 and R7-B1 both fixed the `\.` alternative;
  the `\band\b|\bbut\b` alternatives of the SAME regex were never probed. A conjunction inside a
  coordinated noun phrase splits BEFORE the shared head noun and deletes it. Measured, cue-first, **5 of
  6**, and I can hit it with the R7 pin's own sentinel pairs: "cancelled the Friday **and Saturday dinner
  reservations**" → clause `"The user cancelled the Friday and"` → named={friday} ⊆ plan → **BINDS**
  "plans a Friday **lunch** reservation" (R7-B1's exact pair, re-reachable via `and`); "cancelled the
  German **and Spanish lessons**" → BINDS "plans a German **exam**" (again R7-B1's pair); "cancelled the
  pottery **and painting classes**" → BINDS "plans a pottery **workshop weekend**"; "cancelled the
  pottery **but kept the painting class**" → BINDS the same; "cancelled the Rome and Milan trips" → binds
  a Rome hiking trip (arguably correct, not counted as clearly false). Only the morning/evening shape
  refused. Write chain is unbroken: `binds=True` → :532 `cancel_plan.append(cid)` → :554
  `store.mark_event_cancelled(cid, db=txn)`, no gate between (R6 proved this end-to-end for the identical
  mechanism; **I did not re-run the write repro this round**).
  **Why this is a NOTE and not a blocker — stated plainly so the standard stays consistent:** R6-Ma1 and
  R7-B1 were blockers because the record made an affirmative safety claim that measurement falsified.
  Nothing in the record claims conjunction splits are safe, and disclosure 5 already discloses this
  class's SYMPTOM verbatim — *"a clause naming only GENERIC words that appear in the plan binds without
  naming anything distinctive"* — which is literally what named={friday} is. Blast radius is doubly zero
  (disclosure 4: cancellation is prompt-unreachable, 0/23 planned markers; disclosure 8: nothing wired
  into product retrieval). And a real code fix needs shared-head-noun detection over a coordinated NP,
  which is out of proportion to a gate that cannot fire; the `and|but` branch is also load-bearing (R7's
  M2 → 2 red), so it cannot simply be deleted. What IS required: disclosure 5 opens "The containment
  gate's own misses and residuals, **all measured**", and that is a completeness claim. Append, verbatim:
  *"the same truncation applies to the `and`/`but` branch of the splitter: a conjunction inside a
  coordinated noun phrase splits before the SHARED HEAD NOUN and deletes it, so 'cancelled the Friday and
  Saturday dinner reservations' names only {friday} and binds a Friday LUNCH plan — 5 of 6 measured
  cue-first phrasings; every splitter false split, period or conjunction, moves toward BINDING."*
  Recommend a pin alongside it so a future splitter edit cannot silently widen the class.
- **(n2) One measured exception to a parenthetical now inside disclosure 5.** 1482-1484 says missed
  sentence splits "GROW the clause, which is the refusing direction". Growth is refusing-direction for
  the `outside` test but BINDING-direction when it rescues an EMPTY named set: `named` empty → "cue
  clause names nothing" → refuse; grow the clause and it can become non-empty AND a subset → BIND.
  Measured 1/4, and only with unnatural text ("The user cancelled it. The pottery workshop weekend was
  fun." → BINDS, because "it." is a 2-letter token so the sentence period no longer splits). Note-level
  precisely because I could not reach it with natural extracted-fact phrasing — I tried four shapes and
  three refused. Suggested amendment: "…which is the refusing direction **except when the cue clause
  named nothing, where growth can flip a refusal into a bind (measured once, on contrived text)**".
- **(n3) The stale count text still stands in the CAN-claim sentence.** `BUILD_LOG:1445-1447` still reads
  "236 tests green across the 5 scoped files (53/54/64/60/5)", corrected only by the bracket at 1448.
  This is the same inline-correction pattern the log uses elsewhere and it preserves history honestly, so
  it is not a defect — but anyone lifting that SENTENCE alone quotes a stale number. Cleanest fix:
  put the bracket immediately after "(53/54/64/60/5)" rather than at the end of the paragraph.
- **(n4) Method disclosure, mine — I was not perfectly read-only.** Running the dispatched `--cov` checks
  inside the repo caused pytest-cov to write and combine `.coverage*` data files in the repo root: the
  untracked `.coverage.Sahiths-MacBook-Pro.local.34248.XYLfFMlx` present at session start is gone and a
  new parallel file is present. No tracked file, no source file, and no doc changed (hashes above), and
  these are regenerated by any coverage run — but it is a slip against my own standing rule and I am
  recording it rather than claiming a clean session. Lesson filed: export `COVERAGE_FILE` to scratch, or
  run coverage from the shadow copy.

**Refs:** llm/supersession.py:100-115(112-115),136-202(159-190,181,186),526-532,554;
tests/test_supersession.py:1313-1336,1339-1359,1361-1370; CONSOLIDATION_V2_BUILD_LOG.md:1422-1497
(1431-1432,1445-1448,1468-1484,1480-1484),1499-1546(1540-1546),1548-1585;
benchmarks/consolidation_v2_stage4_smoke.py:8-13,130,168-176. Harnesses: scratchpad/s4r8/ (pkg/ = shadow
copy; sitecustomize.py = editable-finder remap; runmut.py + runmut2.py = the 4 mutations; probe.py /
probe2.py = 41 binding probes; astscan.py; smoke_r8.txt = my smoke run).

**Who needs to know:** **Bosses — Stage 4 is CLOSED, PASS-WITH-NOTES, after 8 rounds.** Every number in
the record is exact for the fifth round running; all three demanded mutations kill exactly the named test
alone; the R7-B1 family is closed 12/12 with true-positive controls intact; the period branch and the
wide negation window are both now pinned by negatives, which is what R7 said was missing; the smoke
reproduces under my own run; the four record edits are real and accurate; and the record's own narrative
about why the token anchor exists is verified by mutation. Ship it with n1 attached. **Dev-Head —** one
required edit before the claims block is quoted anywhere: append n1's verbatim clause to disclosure 5
(and, if cheap, a pin). n2/n3 are one-line wording improvements. **Founder —** the cancellation gate is
now genuinely good: "cancelled my appointment with Dr. Meyer" and "cancelled the Friday 6 p.m. dinner"
both correctly refuse to touch a different plan, and the write-up no longer claims anything I could
measure as false. The remaining known hole is the mirror image via "and": "cancelled the Friday **and
Saturday** dinner reservations" can still mark a Friday lunch cancelled. It cannot happen in the product
today — the extractor never produces the marker this gate needs, and none of this is wired into
retrieval — and it is now written down as a known limit instead of being claimed safe. That last part is
the whole difference between rounds 6-7 and this one.

## 2026-08-07 — Critics — Consolidation v2 Stage 5 (facts-first retrieval wiring), ROUND 1 — **BLOCK**

**Claim reviewed:** Stage 5 "G1 24 tests + 255 regression, G2 $0 diagnostics with two byte-identical
runs, two real pre-existing bugs found and fixed" (CONSOLIDATION_V2_BUILD_LOG.md:1640-1837).
**Method:** read llm/fact_retrieval.py, llm/context_assembler.py, db/fact_entities.py,
db/semantic_facts.py, cache/redis_client.py, storage/store.py, tests/conftest.py,
tests/test_fact_retrieval.py, benchmarks/facts_first_diagnostics.py, mcp_server/server.py in full;
7 executed mutations via an importlib source-rewrite pytest plugin (scratch, repo untouched);
5 live probes; one full rerun of the $0 diagnostics (local llama3.1 only, $0); full 255-test
regression rerun. All runs env-pinned AGENTMEM_OS_DB_PATH=/tmp/critic-s5-r1/*, DISABLE_REDIS=1,
COVERAGE_FILE scratch. Repo hashes + `git status` identical at start and end.

**Verdict: BLOCK — 3 blockers, 7 majors, 5 minors/notes.**
- S5-R1-B1 BLOCKER: budget truncation destroys the top-ranked fact. build_block picks by rank, sorts
  chronological ASC, assembler head-truncates (context_assembler.py:125-126) -> the OLDEST facts
  survive and the rank-0 CURRENT fact is cut mid-word. Measured: rank-0 "Rachel is currently working
  at TechCorp as VP of Platform Engineering" rendered as "Rachel is cu" while three 2020 workshop
  facts survived whole. Contradicts fact_retrieval.py:166 ("the best evidence must survive the
  budget"). This is the SAME failure the repo already measured and fixed for chunks
  (qa_accuracy_eval.py:277-284: "chronological presentation made it catastrophic ... collapsed to
  0.13"). Transition lines are outside the budget accounting entirely (measured 1072 chars rendered
  for char_budget=200, 5.4x), so overflow is the normal case, not an edge.
- S5-R1-B2 BLOCKER: the get_history "fix" poisons the hot cache. len(cached)>=last_n means the
  assembler's last_n=20 can NEVER hit a 10-turn cache, so every assemble re-runs the repopulate loop
  (store.py:211-217) and duplicates turns. Modeled faithfully: a 5-turn session serves
  mcp_server/server.py:377 (last_n=10) ['t1'..'t5','t1'..'t5'] — duplicated turns on a shipped tool.
  Untestable by construction: conftest forces DISABLE_REDIS=1 for the whole suite.
- S5-R1-B3 BLOCKER: byte-identity pin is mutation-green. MUT4a (sem_budget -= 5000 whenever the facts
  tier runs) and MUT4b (empty block still enters the emit branch) both leave 24/24 GREEN, because
  _FakeChroma ignores top_k and returns 2 tiny chunks. The exact drift class that moves the banked
  66.0%/0.952 is invisible. Repeat of the logged lesson "An assertion that matches the mutant is not
  a tripwire" (2026-08-06, Stage 2 R3).
- Majors: fact_text newline + "[change history:" injection render as forged evidence lines (measured);
  _query_surfaces cap drops 9/10 sub-word surfaces — the arm the G1 catch was added for; the Redis
  root cause (session-id-only keys) is NOT fixed, only kill-switched, while the record says "FIXED";
  Part B's "raw-turn path finds it NOWHERE" is structurally guaranteed (TechCorp count in
  answer_b0f3dfff_1 is 0) and duplicates Part C; Part D's "27:12 stays visible / no false
  supersession" has ZERO output backing it (the script never probes 27:12, never prints Part D's
  supersession result); benchmarks/qa_accuracy_eval.py:349 is an affected live caller missing from
  the impact analysis; "no model load can EVER sit under a database lock" is false (the e5 alias
  model loads inside facts_for_entity's open session, reproduced on a Devanagari query).
- What HELD: mutations 1/2/3/5 each kill exactly their named test and nothing else; every G2 number
  reproduced EXACTLY on my rerun (6264==6264, 945, 900, 3463, 2631, facts-only=True,
  history_visible=False, 25:50 present, 188 live Redis keys); 255 regression reproduced; the
  cancelled-filter sibling sweep is closed on every live-fact reader except facts_as_of (disclosed);
  the honest notes about the non-firing Rachel supersession and the killed 4/6 mutual proposal are
  accurate to the audit rows.

**Refs:** llm/fact_retrieval.py:39-43,102-108,182-210,212-225; llm/context_assembler.py:117-138,
141-176,295-330; storage/store.py:173-218; cache/redis_client.py:18-28; tests/conftest.py:22;
tests/test_fact_retrieval.py:234-256,259-269,281-291,348-385; benchmarks/facts_first_diagnostics.py:
68-88,192-209; mcp_server/server.py:377,388-419; qa_accuracy_eval.py:277-285,349;
CONSOLIDATION_V2_BUILD_LOG.md:1640-1837. Harnesses: /tmp/critic-s5-r1/ (mut/ = 7 mutation plugins +
mutlib.py; probe1-5.py; diag_run1.log = my full $0 rerun).

**Who needs to know:** **Dev-Head — do not ship.** B1 is a wrong-answer generator on the product's
own headline question type and repeats a failure this repo already measured and fixed once. B2 is a
regression this stage introduced into a shipped MCP tool. B3 means the load-bearing no-regress pin
does not bear load. **Bosses —** the measurement half of Stage 5 is genuinely strong (every number
reproduced exactly, determinism held across windows); the failure is in unbudgeted/untruncated
rendering, cache side effects, and three claims written past their evidence. **Founder —** nothing
here spends money and nothing is wired into a paid path yet; the honest notes in the record are
honest. The problem is that under budget pressure the new facts block currently shows the STALE
answer and cuts the current one, and the record states three Part-B/D results the diagnostic never
measured.

## 2026-08-07 — Critics — Consolidation v2 Stage 5, ROUND 2 (fix-pass re-review) — **BLOCK**

**Claim reviewed:** "full fix pass for the R1 verdict has landed; 30 passed + 1 opt-in skip, 255
regression green, diagnostics clean, five inline record corrections"
(CONSOLIDATION_V2_BUILD_LOG.md:1839-1941).
**Method:** re-read every changed file; 12 executed mutations (11 new against the FIXES + 2 R1 replays)
via scratch source-rewrite plugins; 5 new probes incl. a 115-point budget sweep; full $0 diagnostics
rerun diffed against my R1 run; 255-regression rerun; 30-test run. All env-pinned to
/tmp/critic-s5-r2/*, DISABLE_REDIS=1. Repo hashes + `git status` unchanged.

**Verdict: BLOCK — 1 blocker, 2 majors, 5 minors/notes.** (R1's 3 blockers: B2 and B3 CLEARED, B1 NOT.)
- S5-R2-B1 BLOCKER: **B1 is not fixed — it recurs one layer down.** build_block now honours its CHAR
  budget, but the assembler's constraint is TOKENS. Measured density of a rendered fact block:
  3.68-3.84 chars/token, i.e. always < the 4.0 the `char_budget = sem_budget * 4` proxy assumes. So a
  full block overshoots the TOKEN budget, `_fit_to_budget`'s binary search fires, head-keeping cuts the
  newest = rank-0 fact. Swept semantic budgets 60..1200 step 10: **95 of 115 (83%) lose the rank-0
  current fact.** Ordinary English facts, ordinary dates, no adversarial input, no chains.
- S5-R2-B1a (same blocker, worse): **the new pin is a tautology that is green while the bug fires
  inside it.** `test_truncation_cannot_delete_top_ranked_fact` asserts `"TechCorp" in out` — and all 30
  filler facts contain "TechCorp". Ran its exact fixture at its exact budget (semantic=100):
  `"TechCorp" in out` -> True (green) while `current.fact_text in out` -> False, section ending
  `"[2024/12/31] (state) Rac"`. Second occurrence in this stage of the logged lesson "An assertion that
  matches the mutant is not a tripwire" (2026-08-06 Stage 2 R3), which I cited in R1.
- S5-R2-M1 MAJOR: the record states the pin "asserts surviving text ... through the real assembler at
  semantic=100" (log:1856-1858). It does not, and the property is violated at that exact budget.
- S5-R2-M2 MAJOR: G2 cannot see B1 — the diagnostics runs at default allocations (15360 tokens) where
  the fact block never approaches the budget, so a clean diagnostics rerun is not evidence for B1.
  Undisclosed. qa_accuracy_eval (~4740 tokens, named in the M6 correction as "MOST exposed to B1") sits
  in the failing regime.
- Minors: `break`-not-`continue` (no short-fact leapfrogging) is claimed in code + record and survives
  mutation N2 unpinned; `_predecessor_targets` scope filter survives N8 unpinned (not claimed pinned —
  note only); `_sanitize` residuals — zero-width space inside the marker (`[change​history:`)
  bypasses `\s+` and renders visually identical to the real marker, bracket homoglyphs pass, and inline
  (same-line) date/type forgery survives whitespace collapse; stale "10 turns" in
  cache/redis_client.py:10 and storage/store.py:181 after max_turns became 20; the faithful fake's
  pipeline executes immediately so an execute-ordering bug in replace_history would not be caught
  (opt-in live test partially covers).
- **CLEARED:** B2 (duplication gone at every session length; ≥20-turn sessions now hit 3/3 with 0
  writes; residual = short sessions still never hit, correct results, undisclosed perf note only).
  B3 (MUT4a now dies via chroma-calls inequality; MUT4b equivalence claim VERIFIED by code — 
  `_fit_to_budget("")` returns "" and the section is never appended). M1 line-forgery, M2 interleave,
  M7 docstring, m1/m2/m4/m5/n1/n2/n3 all verified. 10 of 12 mutations died under exactly their named
  test. Diagnostics: exit 0, Part A 6264==6264, Part D now genuinely measures both 27:12 and 25:50 with
  superseded=[] and the rendered lines; my R1-vs-R2 log diff shows every LLM output and every section
  size byte-identical — determinism now holds across three independent windows. 255 regression + 30
  tests reproduced.

**Refs:** llm/fact_retrieval.py:44-68,195-263; llm/context_assembler.py:117-138,295-330;
cache/redis_client.py:10,31-61; storage/store.py:173-220(181);
tests/test_fact_retrieval.py:276-301(301),304-322,352-375,443-466,469-514;
benchmarks/facts_first_diagnostics.py:76-100,212-245; CONSOLIDATION_V2_BUILD_LOG.md:1839-1941
(1845-1859,1856-1858,1930-1933,1938-1941). Harnesses: /tmp/critic-s5-r2/ (mut/n1-n10; probe_tok.py,
probe_sweep.py, probe_taut.py, probe_san.py, probe_cache.py; diag_run2.log + .norm diff vs R1).

**Who needs to know:** **Dev-Head —** one blocker left, and it is R1-B1 unchanged in substance. The
char-budget fix was the right idea applied in the wrong unit: build_block must budget in TOKENS
(counter.count on the accumulating block) so `_fit_to_budget` never truncates the facts section at all,
and the pin must assert a needle that exists ONLY in the rank-0 fact. Everything else you did this
round is real and verified. **Bosses —** 2 of 3 blockers genuinely closed with mutation-proven pins;
the remaining one is a unit mismatch, not a design flaw. **Founder —** the fix pass was honest work and
the record corrections are accurate; the one thing still wrong is the same thing as last round — at a
tight budget the assembled prompt can still show the stale answer and cut the current one, 83% of the
budgets I swept — and the test written to catch that passes while it happens.

## 2026-08-07 — Critics — Consolidation v2 Stage 5, ROUND 3 (fix-pass re-review) — **BLOCK (narrow)**

**Claim reviewed:** "B1 fixed in the right unit; B1a pin rewritten; all eight R2 minors landed; 35
passed + 1 opt-in skip; 255 green; record carries the FALSIFIED/CORRECTED strikes"
(CONSOLIDATION_V2_BUILD_LOG.md:1943-2018).
**Method:** re-read the rewritten build_block/_sanitize/store paths; 11 executed mutations (8 new
against the R3 fix + N2/N8 replays + 1 invalidated and redone); 5 probes incl. my R2 115-budget sweep
re-run against the real assembler, a 400-trial search for the re-count loop's precondition, and a
5-point latency curve; third independent $0 diagnostics rerun diffed against my R2 run; 255 regression;
36-test run. Env-pinned to /tmp/critic-s5-r3/*. Repo hashes + `git status` unchanged.

**Verdict: BLOCK — 0 blockers, 2 majors, 4 minors/notes. R2-B1 and R2-B1a are CLOSED.**
- **B1 CLOSED, verified independently.** My R2 sweep re-run against the real assembler: **0 of 115
  budgets lose the rank-0 fact** (was 95/115). build_block never exceeded its token budget at any of
  the 115; the `_fit_to_budget` char fast path (`len(text) > token_budget*4`) never tripped either —
  the mirror hole I looked for does not occur, because chars/token stays at 3.7-3.8.
- **B1a CLOSED.** Needle "Zephyrine Analytics" exists only in the rank-0 fact; assembler half asserts
  `current.fact_text in out`; the new 58-budget sweep pin is real. Mutations MA (token count → len//4)
  and MD (token_budget → sem_budget*4) both die on it.
- S5-R3-M1 MAJOR: **the record's "m1 break-not-continue now PINNED" is false.** N2 (`break`→`continue`)
  still leaves 35 passed. Measured cause: `test_fill_stops_at_first_nonfit_no_leapfrogging` uses
  token_budget=30, but the alpha+gamma block costs 38 tokens — gamma is excluded by the BUDGET under
  both branches. Discriminating budgets are 38..168; the test picked 30. **Third tautological pin in
  this stage** and a repeat of a lesson I filed twice (2026-08-06 Stage 2 R3; 2026-08-07 S5 R2).
  Fix: raise the budget to ~45 and assert gamma present under `continue`.
- S5-R3-M2 MAJOR: **20x latency regression on the product read path, undisclosed.** The accumulating
  `counter.count(candidate)` re-tokenizes the whole growing block once per candidate — quadratic.
  Measured at the product default budget: 100 facts 53 ms, 200 facts 198 ms, 350 facts 586 ms,
  **500 facts (= _LEXICAL_SCAN_CAP, the DESIGNED ceiling) 1150 ms/call**, vs 4.5 ms for a single count
  of the final block (257x). Sits inside synchronous recall_memory. Trivially fixable by incremental
  counting (count the new line + 1) with a single exact count at the end.
- Minors: the post-sort re-count loop is unreachable in practice (fires at 0 of 115 budgets; 0 of 400
  random orderings raise the sorted count above the fill-order count) so MB (remove it) and MC (drop
  the NEWEST instead of lowest-ranked) are both mutation-green — the docstring and record describe it
  as a working mechanism rather than an unreachable guard, and its central "never the newest" property
  has no test; disclosed rank-0 exception measured at 318 tokens against a 20-token budget (15.9x,
  contract-consistent, tail cut mid-word by the assembler); `_INLINE_STAMP_RE` rewrites legitimate
  bracketed dates in fact text to parentheses (benign, information-preserving, undocumented in the
  test); TokenCounter divergence checked — both sides construct `TokenCounter()` with the same default
  (gpt-4o → cl100k_base), no divergence.
- **Everything else verified fixed:** N8 now dies (scope-leak pin real), ME/MG/MF each die on
  test_render_forgery_residuals_neutralized (zero-width strip, homoglyph bracket class, inline stamp
  demotion), MH dies on test_repopulate_skipped_when_cache_already_correct, m4 stale comments gone, m5
  fake pipeline buffers, n2 staleness disclosed, n3 ledgered. 9 of 11 mutations died under exactly
  their named test. `cached = None` in the except block closes the NameError edge I looked for.
  255 regression reproduced (348s). Diagnostics: exit 0, 6264==6264, and byte-identical to my R2 run
  across every LLM output, section size and rendered metric line — determinism now holds across four
  independent windows. The FALSIFIED R2 and CORRECTED R2 strikes are accurate as written.

**Refs:** llm/fact_retrieval.py:58-83,209-277(252-262,268-276); llm/context_assembler.py:115-131;
storage/store.py:173-227(190,222); cache/redis_client.py:10,39,47-67;
tests/test_fact_retrieval.py:324-374,400-417(414); CONSOLIDATION_V2_BUILD_LOG.md:1943-2018(1993-1995).
Harnesses: /tmp/critic-s5-r3/ (mut/ma,mb,mc,md,me,mf,mg,mh,n2,n8; probe_pins.py, probe_sweep3.py,
probe_new.py, probe_perf.py; diag_run3.log + norm diff vs R2).

**Who needs to know:** **Dev-Head —** the blocker is gone and I could not re-open it from any angle I
tried. Two things stand between here and PASS, both small: change `token_budget=30` to ~45 in the
leapfrogging pin (and assert gamma IS admitted under `continue`), and make the fill count incrementally
so the designed 500-fact cap doesn't cost 1.15 s of tokenization per recall. Then correct the "now
PINNED" sentence. **Bosses —** this is a narrow gate, not a rejection: the round did the hard thing
right. I am holding only because a false "now PINNED" claim is the same class I blocked R2 for, and
because a 20x read-path regression should not enter the record silently. **Founder —** the prompt no
longer drops your current answer at any budget I could find, and I checked 115 of them plus every
mutation I could think of against the fix itself.

## 2026-08-07 — Critics — Consolidation v2 Stage 5, ROUND 4 (fix-pass re-review) — **BLOCK**

**Claim reviewed:** "M1 pin now discriminates; M2 O(n²) fixed (24.3ms at cap); all four R3 minors
landed; 38 passed + 1 opt-in skip; 255 green" (CONSOLIDATION_V2_BUILD_LOG.md:2030-2060).
**Method:** 8 mutations against the new incremental fill and _trim_to_budget + N2/MB/MC replays; an
output-equivalence probe over 58 budgets × 201 facts; perf re-measure at the 500-fact cap; two
correctness sweeps (115 budgets × 501 facts, and 53 budgets to 15360) plus a worst-case content probe;
255 regression; 39-test run. Env-pinned to /tmp/critic-s5-r4/*. Repo hashes + `git status` unchanged.

**Verdict: BLOCK — 1 blocker. Everything else this round PASSED.**
- S5-R4-B1 BLOCKER: **B1 has a third incarnation — `_fit_to_budget`'s CHAR fast path.**
  `context_assembler.py:313-316` does `char_budget = token_budget*4; if len(text) > char_budget:
  _cut(text, char_budget)` with keep="head", BEFORE the token check. build_block now guarantees TOKENS
  but nothing constrains CHARS, so any block whose content exceeds 4 chars/token is head-cut and the
  chronologically-newest = rank-0 fact is deleted. Measured on ordinary long-common-word English prose
  (5.9 chars/token — no adversarial input, no rare tokens): **rank-0 lost at 9 of 9 budgets, including
  qa_accuracy_eval's 4740 and the product default 15360**, with build_block's own contract intact at
  every one (4738 tok ≤ 4740; 28047 chars > 18960 → 32% of the block discarded from the tail). Also
  found organically at 6 of 115 budgets on a second fixture (4.13-4.20 chars/token).
  **Why nothing caught it:** every fixture in the suite and in both of my prior sweeps measures
  3.7-3.88 chars/token — just under the threshold. The sweep pin's filler is 3.88; ordinary prose is
  5.87. The fixture's chars/token ratio was a hidden parameter of every sweep anyone ran, mine included
  — my R3 "char fast path never trips, 0/115" was true of that fixture and false of the class.
  This is the S4-R8 alternation lesson: `_fit_to_budget` has TWO cuts and four rounds have hardened one.
- **M1 CLOSED:** N2 (`break`→`continue`) now DIES on test_fill_stops_at_first_nonfit_no_leapfrogging;
  budget 45 sits in my measured discriminating band (38-168) and the positive control is real.
- **M2 CLOSED:** re-measured at the 500-fact cap — 20.3 ms @15360, 16.2 ms @4740, 10.6 ms @1000 (record
  says 24.3/17.6/9.6 — same class, machine noise), block ≤ budget at all three; ~56× better than R3's
  1150 ms; scaling now roughly linear (6.3/9.7/27.4/21.7 ms at 100/200/350/500 facts).
- **R3 minors CLOSED:** MB (trim removed) and MC (trim drops newest) both now DIE on
  test_post_sort_trim_drops_lowest_ranked_never_newest, as does P4 (trim's guard inverted). m2/m3/n1
  pins verified.
- **P1/P2/P3/P5 survive but are MEASURED equivalent, not uncovered:** across 58 budgets × 201 facts,
  dropping the +1 join term, never rejecting at the boundary, deliberately under-counting, and skipping
  the drift reset each produce **0 output differences, 0 budget violations, 0 rank-0 losses**, and the
  incremental fill admits exactly as many facts as a true exact fill at every budget (no recall traded
  for speed). The estimate is genuinely a non-load-bearing fast path, as the docstring claims. P2 is
  output-equivalent but destroys the perf fix, and nothing guards that — the record is the only guard.
- 255 regression reproduced (346.8s). The [FALSIFIED R3] strike is accurate as written.

**Refs:** llm/context_assembler.py:295-330(313-316) ← the blocker; llm/fact_retrieval.py:250-300;
tests/test_fact_retrieval.py:324-374,400-420; CONSOLIDATION_V2_BUILD_LOG.md:2030-2060(2037-2043,
2045-2054). Harnesses: /tmp/critic-s5-r4/ (mut/n2,mb,mc,p1-p5; probe_equiv.py, probe_perf4.py,
probe_check.py, probe_char.py, probe_hi.py, probe_worst.py).

**Who needs to know:** **Dev-Head —** one blocker, and it is B1's third gate. Close the ALTERNATION this
time: make build_block satisfy BOTH constraints (tokens ≤ budget AND chars ≤ 4×budget), or give the
facts section a `_fit_to_budget` path with the char fast path disabled. And add a fixture whose content
measures >5 chars/token to the sweep pin — the suite currently cannot express this regime. **Bosses —**
M1 and M2 are properly closed and the equivalent-mutant analysis came out clean under measurement; this
round's work was good. The blocker is a sibling gate nobody had probed, including me. **Founder —** at
the product's own default budget, with plain long-worded English facts, the assembled prompt still drops
your current answer and keeps the stale ones. It is the same symptom as round 1, through a third
mechanism.

## 2026-08-07 — Critics — Consolidation v2 Stage 5, ROUND 5 (fix-pass re-review) — **BLOCK (one item)**

**Claim reviewed:** "R4-B1 alternation closed; the R4 note's perf guard added; 40 passed + 1 opt-in
skip; 255 green" (CONSOLIDATION_V2_BUILD_LOG.md:2077-2119).
**Method:** 6 mutations against the new char accounting + P2 replay; both 115-budget sweeps (sub-4 and
>5 ratio fixtures) and my exact R4 nine budgets through the real assembler; a q1/q2 differential; a
low-ratio differential for P2; fresh-eyes hunt for a fourth incarnation across every cut between
build_block and the final assembled string; 255 regression; 41-test run. Env-pinned to
/tmp/critic-s5-r5/*. Repo hashes + `git status` unchanged.

**Verdict: BLOCK — 0 blockers, 1 major, 3 notes. R4-B1 is CLOSED.**
- **R4-B1 CLOSED, verified in BOTH regimes.** HIGH-ratio prose (block ratio 5.91 — the exact content
  that failed 9/9 in R4): rank-0 lost **0 of 115** swept budgets and **0 of 9** including qa-eval's
  4740 and the default 15360; the char gate never fires; tokens never over budget. Sub-4 entity-dense
  fixture (3.99): same, 0/115 and 0/9. q3 (_CALLER_CHAR_FACTOR→8) and q4 (full revert to R4 token-only)
  both DIE on test_rank0_survives_high_ratio_content. The ratio self-assertion in that pin is the right
  answer to the hidden-parameter problem.
- S5-R5-M1 MAJOR: **test_count_calls_stay_linear does not guard anything — P2 survives it**, contrary
  to the record (:2114-2117). Measured cause: its fixture is 3.83 chars/token, where the NEW char break
  fires at the same point as the token break, so correct code and P2 make **identical** call counts
  (5 vs 5 at token_budget=60; 23 vs 23 at 600) against an assertion of `<= 30`. P2 is NOT an equivalent
  mutant — on low-ratio content (1.76 chars/token) it is plainly discriminable: 7 vs 16, 11 vs 28,
  19 vs 56, 35 vs 89 count calls, and 2× the facts admitted at fill. Fifth instance in this stage of a
  fixture placing both branches on the same side of the deciding threshold, and the SECOND caused by
  the chars/token hidden parameter I filed as a lesson in R4. Fix: low-ratio fixture (or a char cap put
  out of reach) and a threshold between the two measured counts. Output is provably unaffected, so this
  is a coverage + record-claim defect, not a user-visible one.
- q1 (char break removed from fill) and q2 (char term removed from trim) survive individually but are
  MEASURED EQUIVALENT — identical output at 5 budgets on 5.9-ratio content, neither exceeding the char
  cap; q4 (both) dies. Redundant defense in depth, verified not argued.
- **Fresh-eyes hunt for a fourth incarnation: none found.** `_fit_to_budget` is the only truncation
  between build_block and the assembled string; the section join and the total_tokens log are
  non-destructive; label wrapping adds ~14 tokens after the cut and never pushed the section over its
  allocation in measurement; tiny model_window (100 → sem_alloc 12) hits only the DISCLOSED
  single-oversized-rank-0 exception and stays rank-safe; qa_accuracy_eval's `context[:24000]`
  (:218,221) is a HEAD-keep on a section order that puts SYSTEM+FACTS first, so it cuts RECENT TURNS,
  not facts — undisclosed but not a facts hazard.
- **SELF-CORRECTION:** my R3 and R4 entries state `TokenCounter("gpt-4o")` resolves to cl100k_base. It
  resolves to **o200k_base** (`tiktoken.encoding_for_model('gpt-4o').name`). Counter parity between
  retriever and assembler is unaffected (both construct `TokenCounter()` identically), but every
  chars/token figure I have quoted this stage is an o200k measurement and my encoding name was wrong.
- 255 regression reproduced (348.0s); 40 passed + 1 opt-in skip. Record sentences at :2079-2113 all
  verify against my measurements.

**Refs:** llm/fact_retrieval.py:57-63,268-330; llm/context_assembler.py:313-316;
tests/test_fact_retrieval.py:446-469; benchmarks/qa_accuracy_eval.py:218,221;
CONSOLIDATION_V2_BUILD_LOG.md:2077-2119(2114-2117). Harnesses: /tmp/critic-s5-r5/ (mut/q1-q4,p2;
probe_main.py, probe_p2.py, probe_ratio.py, probe_final.py).

**Who needs to know:** **Dev-Head —** the blocker is closed and I could not find a fourth incarnation
from any cut on the path. One item stands: give test_count_calls_stay_linear a low-ratio fixture so the
char break cannot mask the token break, tighten the bound, and correct the record sentence. That is the
whole gate. **Bosses —** Stage 5's correctness work is done; this is a coverage-claim hold, the same
class I blocked in R2 and R3, applied consistently. **Founder —** at every budget and both content
regimes I could construct, the current answer now survives into the prompt. The thing still wrong is a
test that guards a speed property and doesn't, plus one sentence in the record that says it does.

## 2026-08-07 — Critics — Consolidation v2 Stage 5, ROUND 6 (fix-pass re-review) — **PASS-WITH-NOTES**

**Claim reviewed:** "R5-M1 fixed with a self-asserting low-ratio fixture; n1/n2/n3 landed; 40 passed +
1 opt-in skip; regression unchanged at 255" (CONSOLIDATION_V2_BUILD_LOG.md:2121-2167).
**Method:** P2 replay against the new fixture; margin/stability probe across 6 budgets; reconciliation
of the record's P2 figure; my own 255-regression receipt; 41-test run; full top-to-bottom read of the
Stage 5 record (CONSOLIDATION_V2_BUILD_LOG.md:1640-2167) as the stage-closing sanity pass. Env-pinned
to /tmp/critic-s5-r6/*. Repo hashes + `git status` unchanged.

**Verdict: PASS-WITH-NOTES — 0 blockers, 0 majors, 6 notes (all record/comment hygiene; 3 are MY
errors propagated into the record and code).**
- **R5-M1 CLOSED and verified.** P2 now dies: `assert 119 <= 45`. Measured on the exact fixture —
  sample line 125 chars / 101 tokens = **ratio 1.24** (self-assert < 2.5 holds with room), fixture char
  total 6300 vs cap 6800 (8% headroom, and a breach fails LOUD via the self-assert, which is the point).
  Separation is stable across budgets 1400-2500: correct branch **16/18/19/20/22/27** calls, P2
  **108/121/119/117/113/103** — the bound 45 sits between with 1.7-2.8x margin on the correct side and
  2.3-2.7x on the P2 side at every point. The self-asserting-fixture pattern is now the right default
  for this class.
- n1 (encoding dependence named at the constant + parity pin), n2 (qa_accuracy_eval's 24k HEAD-keep
  recorded), n3 (my o200k self-correction recorded) all verified in place.
- 255 regression reproduced on my own run (347.5s); 40 passed + 1 opt-in skip. The [FALSIFIED R5]
  strike and the R5 fix-record numbers verify as written.
- **STAGE-CLOSING RECORD PASS — 6 items to fix before the closing entry:**
  1. :1962 "_fit_to_budget is now a guard rail, not the working cut" — FALSIFIED by R4-B1 (its CHAR
     half was the working cut, 9/9 budgets) and never struck, unlike every other falsified sentence in
     this document. It is TRUE today (verified: char gate fires 0/115 in both regimes), which is why
     this is a note and not a major — but the strike discipline should be uniform.
  2. llm/context_assembler.py:127 carries the same "guard rail, not the working cut" claim as a bare
     fact. The coupling is documented at length on the retriever side (_CALLER_CHAR_FACTOR) and not at
     all here — a future editor of _fit_to_budget's factor or keep direction would never see the
     tripwire. This asymmetry is the shape that produced R4-B1.
  3-5. MY cl100k error propagated into three live places the n3 note does not reach:
     CONSOLIDATION_V2_BUILD_LOG.md:2059, llm/fact_retrieval.py:322 (_trim_to_budget docstring),
     tests/test_fact_retrieval.py:539 (test docstring). All should read o200k_base.
  6. No CURRENT-STATE line for Stage 5. Stages 1/3/4 each close with one ("CURRENT artifacts: 51/51
     tests…", "Stage table: Stage 3 ✅ DONE…"); Stage 5's first-encountered count is still "G1: 24
     tests" at :1767 with six later per-round Post-fix lines. A reader quoting the stage will quote 24.
  Also: the record says "P2-emulated 129 calls"; I measure **119** under both plausible emulations
  (token-boundary-only and boundary+no-char-break). Conclusion unaffected (bound 45, correct 19).
  And benchmarks/facts_first_diagnostics.py:245-248 "caps in play (disclosed)" omits the char cap
  (token_budget × _CALLER_CHAR_FACTOR), now a load-bearing cap on the block.

**Refs:** tests/test_fact_retrieval.py:446-484,539; llm/fact_retrieval.py:57-67,322;
llm/context_assembler.py:127; benchmarks/facts_first_diagnostics.py:245-248;
CONSOLIDATION_V2_BUILD_LOG.md:1640-2167(1767,1962,2059,2148-2151). Harnesses: /tmp/critic-s5-r6/
(mut/p2; probe_margin.py, probe_129.py).

**Who needs to know:** **Dev-Head — Stage 5 passes.** Six rounds, and every blocker is closed with a
mutation-proven pin: B1 in three incarnations (char fill, wrong unit, char fast path), B2 (cache
poisoning), B3 (mutation-green no-regress pin), plus the forgery, scope-leak, perf and coverage
findings. Nothing on my list is a code defect; the six items are record and comment hygiene, and three
of them are mine to own. **Bosses —** I am passing this because the substance is verified, not because
the loop is long: the last correctness sweep I could construct came back clean in both content regimes,
the fresh-eyes hunt for a fourth truncation found none, and the pin I blocked on last round now kills
its mutant with a 2.4x margin. **Founder —** the facts-first read path does what the record says it
does: at every budget and both content regimes I could build, your current answer survives into the
prompt, superseded and cancelled facts never appear, scope holds, and the renderer's evidence markers
can no longer be forged from user text. The honest caveats that remain are the ones already written
down — the entity-floor-only path for undated facts at scale, the single-oversized-fact exception, and
the Redis ghost-key root cause still open as a ledger item.

## 2026-08-08 — Critics — Consolidation v2 WHOLE-ARC (Stage 6 G3, final pass) — **BUILD READY WITH CONDITIONS**

**Claim reviewed:** Stage 6 G1+G2 green, four pre-existing defects fixed, 315+1 across eleven files,
BUILD READY per D6 (CONSOLIDATION_V2_BUILD_LOG.md:2268-2426).
**Method:** 6 mutations against the Stage 6 fixes (retry-once, both atomic increments, offline-first,
scratch-config); E2E file 3× plus ~10 mutation runs; race pin in isolation and in-file, 6+5 runs;
G2 script once end-to-end (local llama3.1, $0, exit 0); the eleven-file sweep in TWO orders; a 3-file
bisect; static tracing of ChromaManager/StorageManager resolution; whole-arc record + ledger read.
Env-pinned to /tmp/critic-s6-final/*. Repo unmodified.

**Verdict: BUILD READY WITH CONDITIONS — 2 conditions (must clear before cluster extraction),
2 majors, 8 minors/notes. No correctness defect found in the shipped facts path.**

CONDITIONS (D6's own criteria are not yet met):
- C1: **the eleven-file suite is NOT green — 3 failed / 312 passed / 1 skipped**, in my order AND in
  plain alphabetical order (both reproduced). Root cause bisected to
  `benchmarks/adapters/agentmem_adapter.py:54` — `install_best_chroma(ContextAssembler)` patches
  `_get_chroma` on the CLASS, so after test_agentmem_adapter/test_eval_harness run, every later
  ContextAssembler ignores its instance `_chroma`. The three casualties are Stage 5's assembler-wiring
  tests INCLUDING `test_empty_fact_store_is_byte_identical`, the load-bearing no-regress pin. Alone:
  40 passed. mcp_server+fact_retrieval: 48 passed. eval_harness+agentmem_adapter+fact_retrieval: 3
  failed. D6(2) unmet and the record's "315 passed" does not reproduce.
- C2: **tests/test_e2e_v2.py writes to the founder's DEV vector store on every recall.**
  `ChromaManager.search` → `get_or_create_collection(session_id)`; tests/conftest.py pins the DB path
  and Redis but NOT the StorageManager tree, which resolves `config.yaml` relative to pytest's cwd.
  Evidence: /Volumes/Sahith_SSD/AgentMem-OS/vectors/chroma.sqlite3 (mtime today) holds `e2e-rt-1`,
  `e2e-far-1`, `e2e-sup-2`, `hdr-test2`, `honest-test` — E2E session ids. Stage 6 fixed this channel in
  the G2 SCRIPT only; ledger #28 discloses the benchmark spill and says nothing about the suite, which
  runs constantly. Same class as the conftest's own founding incident (dev DB) and the Redis channel.
  **My own disclosure: I ran that file ~13 times this session and contributed to the spill I am
  reporting.**

MAJORS:
- W1: **retry-once is unpinned as the suite actually runs.** M2 (single attempt) in full-file context:
  5/5 PASS; the same mutant with the pin in ISOLATION: 6/6 FAIL. Earlier tests warm state that closes
  the collision window, so a revert of fix #2 ships unnoticed under `pytest tests/test_e2e_v2.py`.
- W2: **the edge-weight half of the atomic-increment fix is unpinned in every context.** M3b passes
  in-file and in isolation (3/3). The node half is genuinely pinned — M3 dies with `assert 6 == 16`,
  reproducing the record's "6 of 16 lost increments" exactly. The record says "atomic increments for
  both" and names one pin.

MINORS/NOTES: `_ingest_turn_once`'s generic `except Exception` does NOT invalidate the in-memory graph
though the IntegrityError branch does and the method mutates it pre-commit (same class, unguarded
sibling branch — the S4-R8 alternation shape) · offline-first is unpinned BY NATURE (no timing
assertion; `_await_background`'s 180s bound is the only backstop) and the record should say so ·
ledger #6 "NOTHING WIRED INTO PRODUCT RETRIEVAL YET" is stale — Stage 5 did it; D6(4) requires a
current ledger · "~87s" survives as a MODEL-LOAD cost in db/fact_entities.py:49,149,225 and the Stage 3
B1 record (:571) with no cross-reference to Stage 6's finding that it was mostly network (now ~6s) ·
D6 says "all eight test files", the record says ELEVEN · the HF constant mutation is process-global and
serialized only against this loader's lock (a concurrent unrelated HF user could see OFFLINE=True) ·
the G2 script's isolation is half-self-checking (the assert catches a failed config rewrite; nothing
catches a missing `os.chdir`) · Gate C interpretation: ledger #29 means the paid eval measures
facts + a TF-IDF chunk stand-in while the product ships facts-ONLY.

REPRODUCED EXACTLY: G1 9 tests × 3 runs at 6.93/6.12/5.67s · offline load 5.92s vs 82.59s online
(14×; independently confirms the network attribution) · M3 → 6==16 · G2 exit 0 with every number
matching (9+7 facts, 945-char facts section, 8+6, 3192, 5+7=12 sneaker facts, cross-session
facts-only=True, re-consolidate created=0, KG 235 chars) · zero KG drop warnings · SEMANTIC MEMORY
absent from 100% of G2 recalls (ledger #29 verified) · posthog telemetry errors (ledger #30) · dev-store
benchmark collections (ledger #28). NOT independently reproducible: "12 drops in one E2E run" (requires
reverting the fix and re-running G2) — stated as historical.

**Refs:** benchmarks/adapters/agentmem_adapter.py:54; benchmarks/real_code_utils.py:95-100;
db/chroma_client.py:38-58; storage/manager.py:9-20; tests/conftest.py; db/knowledge_graph.py:296-308,
340-353,455-460; db/entity_aliases.py:82-141; db/fact_entities.py:49,149,225;
tests/test_e2e_v2.py:266-311; benchmarks/consolidation_v2_e2e.py:29-53;
CONSOLIDATION_V2_BUILD_LOG.md:571,2268-2426; RUNNING_NOTES.md #6,#23-30.
Harnesses: /tmp/critic-s6-final/ (mut/m2,m3,m3b; probe_offline.py; g2_run.log; sweep.log).

**Who needs to know:** **Dev-Head —** two conditions, both isolation/coverage, neither in the facts
path: make the suite green in any order (the adapter's class patch must be scoped or reverted in
teardown) and extend the Stage 6 isolation pattern from the G2 script into tests/conftest.py. Then W1
(run the race pin in a fresh process, or force the collision) and W2 (pin the edge weight).
**Bosses —** the arc's product claims held up: I could not break the facts path, and every G2 number
reproduced on my own run. The defects are in the test environment and its coverage. **Founder —** two
things before extraction: the full test suite is not actually green in a normal invocation, and running
the tests writes empty collections into your dev vector store (I did it ~13 times today myself). Both
are contained and neither touches stored facts. The parked plans-as-events decision is due to you now
per D7.

## 2026-08-08 — Critics — Consolidation v2 WHOLE-ARC, CONFIRMATION ROUND — **BUILD READY (C1+C2 CLOSED); 1 major + 2 minors are RECORD-scoped, not extraction-scoped**

**Claim reviewed:** both conditions closed, W1/W2 pinned, m1-m8 landed, 317+1 in both orders
(CONSOLIDATION_V2_BUILD_LOG.md:2431-2500).
**Method:** bisect combo + the eleven files in BOTH my orders; 6 mutations against the NEW mechanisms
(class-patch revert, chdir neuter, retry revert, both bump reverts, m1 graph-clear removal); the new
pins run in isolation as well as in-file; empirical dev-vector-store check across the whole session.
Env-pinned to /tmp/critic-s6-confirm/*. Repo unmodified.

**Verdict: BUILD READY. Cluster extraction and Gate C may proceed.**

CONDITIONS — BOTH CLOSED, verified by mutation and by measurement:
- **C1 CLOSED.** Bisect combo (eval_harness + agentmem_adapter + fact_retrieval): **52 passed** (was 3
  failed). Eleven files: **317 passed + 1 skipped in stage order AND in alphabetical order**. X1
  (installer reverted to the class-level `_get_chroma` patch) reproduces the EXACT three failures
  including the byte-identity pin — so the `_chroma_override` data attribute is the load-bearing fix and
  the diagnosis was right.
- **C2 CLOSED, and self-verifying.** X2 (conftest `os.chdir` neutered) makes
  `assert StorageManager().base_path.startswith(str(scratch))` fail LOUDLY at session-fixture setup —
  the m7 lesson applied where it runs a thousand times more often than any script. Empirical proof: the
  dev store's chroma.sqlite3 mtime is UNCHANGED (Aug 8 00:48) after ~1 hour of my runs — the E2E file
  many times, two full eleven-file suites, the bisect combo and six mutation runs — still 24
  collections, no new e2e-* entries. The channel is closed in practice, not just in principle.
- **W1 CLOSED.** X3 (retry-once reverted) kills `test_upsert_retry_is_deterministically_pinned` in the
  FULL-FILE context. The `_find_node` seam makes the collision deterministic instead of thread-timed.

REMAINING (record-scoped — do not publish the closing record with these sentences as written):
- **W2 STILL OPEN, and now mis-recorded.** `test_lost_update_impossible_for_node_and_edge` pins
  NEITHER half. Measured in isolation: X4 (`_bump_node` → RMW) **1 passed**; X5 (`_bump_edge` → RMW)
  **1 passed**; X5 across the whole file **11 passed**. Mechanism: the pin's `dbA.rollback()`/
  `dbB.rollback()` — the step whose comment says "end reads; keep stale objects" — is exactly what
  EXPIRES the ORM instances, so `node.mention_count` / `edge.weight` refresh from the DB at bump time
  and the RMW revert computes the correct value. The record's "an ORM read-modify-write revert writes
  the stale snapshot and dies" is false. The node half is still covered only by the 16-thread
  integration test (X4 kills it there); the EDGE half remains unpinned for the second round running.
  Fix: `expunge()` the objects after loading (detach, keep loaded values) instead of rollback.
- **m1 unpinned and not disclosed as such.** X6 (graph-clear removed from the generic except): 11
  passed. m2 got an explicit "UNPINNED BY NATURE" disclosure; m1 deserves the same, or a pin — it IS
  pinnable (force a non-Integrity error at commit, assert the graph is empty).
- **m4 half-false + splice damage.** "all four '~87s' code comments **+ the Stage 3 B1 context** now
  carry [Stage 6: mostly NETWORK, ~6s offline-first]" — the four code comments do;
  CONSOLIDATION_V2_BUILD_LOG.md:571 does NOT. And the annotation was string-spliced mid-phrase:
  llm/consolidation_v2.py:407 now reads "~87s [Stage 6: ...]-cold alias model" (~110-char line, broken
  grammar); db/fact_entities.py:149,225 have the same pattern.

VERIFIED LANDINGS: m3 (ledger #6 "[RESOLVED Stage 5, kept for history]"), m5 (D6 corrected inline with
the strike), m6, m7, m8 (ledger #29 Gate-C interpretation rule), m2's honest unpinned-by-nature
disclosure, ledger #28 amended to cover the suite including my own runs.

**Refs:** llm/context_assembler.py:367-378; benchmarks/real_code_utils.py:104-107,147;
tests/conftest.py:35-70; db/knowledge_graph.py:446-487; tests/test_e2e_v2.py:322-357,359-397;
llm/consolidation_v2.py:407; db/fact_entities.py:49,149,225; CONSOLIDATION_V2_BUILD_LOG.md:571,
2319-2323,2431-2500. Harnesses: /tmp/critic-s6-confirm/ (mut/x1-x6; order_stage.log, order_alpha.log).

**Who needs to know:** **Dev-Head —** both conditions are genuinely closed and I verified each by
reverting the fix, not by reading it. Three items left, none blocking extraction: swap the pin's
`rollback()` for `expunge()` so W2 actually pins (it currently pins neither half), disclose m1 as
unpinned or pin it, and finish m4 (annotate :571, repair the three spliced comments). **Bosses — the
arc is BUILD READY.** Six adversarial rounds on Stage 5, a whole-arc pass and this confirmation; every
blocker closed with a mutation-proven pin; the product's facts path survived every attack I could
construct. **Founder —** you can start cluster extraction. The test suite no longer touches your dev
vector store (verified: zero new writes in an hour of my running it), the suite is green in both
orders, and the remaining three items are about the accuracy of the write-up, not the behaviour of the
system. Per D7 the parked plans-as-events decision is now yours to make.

## 2026-08-09 — Critics — PROFILE TIER (db/profile.py, profile_extractor, assembler section, G1/G2), ROUND 1 — **BLOCKED (7 blockers, 7 majors, 8 minors, 4 notes)**

**Claim reviewed:** the profile tier is built, triple-gated, and ready for Gate D (~$3.50, the next
spend). **Verdict: BLOCKED.** The gates that exist are mostly real and mostly pinned; the three
properties the build is SOLD on — supersession is read not invented, injection never starves other
tiers, the collapse fix is an improvement — are the three that do not hold. Isolation: forced
`AGENTMEM_OS_DB_PATH` + `DISABLE_REDIS` in-process before every import, config.yaml/base_path rewrite
+ chdir + `StorageManager().base_path` self-check for every assembler run, named test files only,
$0 (local llama3.1 + tiktoken).

### BLOCKERS
1. **The superseded filter is UNPINNED and load-bearing.** Mutation M5 (`db/profile.py:148`
   `SemanticFact.superseded_by.is_(None)` → `1==1`): **19/19 pass.** It matters — measured: old fact
   `t_occurred=2023/09/01` superseded by new `t_occurred=2023/02/01` gives `Bangalore-FEB` with the
   filter and the SUPERSEDED `Bangalore-SEP` without it. `test_superseded_fact_can_never_be_current`
   (tests/test_profile_tier.py:120-133) is tautological: its fixture makes the superseding fact also
   the latest by domain time, so the domain-time rule alone produces the asserted answer. Repeat of
   the logged S5 R3 lesson (prove the exclusion comes from the mechanism, not from a co-incident path).
2. **The budget reservation — the Gate-C-derived headline — is completely unpinned.** M9
   (`PROFILE_BUDGET_SHARE` 0.15→1.0): 19/19 pass. M20 (delete `sem_budget -= profile tokens`,
   context_assembler.py:154-155): 19/19 pass. Cause measured:
   `test_profile_section_injected_and_budget_reserved` (:271-286) asserts 267 ≤ 751 — **484 tokens of
   slack** — and its `prof_tokens` counts `out.split("</[USER PROFILE]>")[0]`, which begins at the
   `[SYSTEM]` section, so it does not measure the profile at all.
3. **`last_tier_budget` reports the profile's tokens as facts.** context_assembler.py:208-211 computes
   `facts_used = semantic - sem_budget` AFTER the profile already reduced `sem_budget`. Measured: a
   store with ZERO facts reports `facts_used: 249`; a realistic run reports 3184 vs 2801 actual. No
   profile key in the report at all. The Gate C lesson at :164-173 is about this exact alarm.
4. **The 4-chars/token proxy truncates the injected profile mid-key on non-ASCII — the Stage 6
   blocker, repeated.** `render(char_budget=profile_budget*4)` (context_assembler.py:147-148,
   db/profile.py:201) with no token cut. Measured at semantic=4740: Telugu/Hindi values render at
   **2.46 chars/token**, 2844 chars ≈ 1156 tokens, `_fit_to_budget` head-cuts and the last injected
   line is literally `pr`; rare-token ASCII at 1.75 chars/token injects **8 of 40** selected
   attributes. Nothing reports the loss. The facts tier already fixed this class
   (`fact_retrieval._CALLER_CHAR_FACTOR`, coupling note at context_assembler.py:180-188). This is the
   D6/Indic path.
5. **Claim "the fact tier owns supersession; the profile reads it" is materially false on the real
   corpus.** `db/profile.py:156-161` keeps exactly ONE row per attribute_key by domain time even when
   NO supersession link exists — a second direction rule, applied silently. Measured on the G2 corpus:
   18 of 41 keys carry 2-13 un-superseded facts; only 29 of 7,164 preference/identity facts in the
   whole corpus are superseded at all. **90 projected facts → 41 injected lines → 49 (54%) never reach
   the prompt.** `hobbies` = 13 facts → one line ("yoga classes"). "Recall becomes 1.0 by
   construction" is 1/N for every collapsed key. No test covers two un-superseded facts on one key.
6. **G2's before/after table flatters the fix by omitting the half that moved the wrong way.** I
   reproduced BOTH columns exactly (before 69/58/8/13 with the vocabulary+reuse+entity rules stripped;
   after 90/41/18/0 — credit, the measurement is real). But injected LINES went 58 → 41 (-29%) and
   hidden facts went 11 (16%) → **49 (54%)**. "keys carrying history" is also a misnomer — projection
   excludes superseded facts (profile_extractor.py:194), so these are coexisting values the reader
   hides, not change histories.
7. **Gate D wiring does not exist and the profile's session scoping FAILS OPEN.**
   context_assembler.py:146 reads `getattr(self, "profile_session_ids", None)` — nothing in the repo
   ever sets it, and `None` means no filter. The facts tier's Gate C equivalent REFUSES
   (`benchmarks/gate_c_facts_source.py:97-102` raises KeyError; its docstring: "that is not a
   measurement, it is leakage"). At Gate D the profile is either silently EMPTY (default
   `ProfileStore(get_session)` reads the eval's live DB, which has no rows and no projection step) or
   leaks all 2,965 sessions into every question. No preflight, no wiring, no test.

### MAJORS
1. **`project()` is not concurrency-safe and breaks the repo's own stated contract.** Read-then-insert
   TOCTOU, no handler (db/profile.py:94-110). Measured: 8 threads × 20 facts → **7 threads died** with
   uncaught `IntegrityError: UNIQUE constraint failed`. db/semantic_facts.py:60-66 states the rule
   ("insert → unique constraint, race falls back to re-affirmation") and :259-274 implements it for
   facts. In `project_scope` the exception is caught at BATCH level and rolls back up to 12 good writes.
2. **`project_scope`'s report claims writes that do not exist.** Measured `{'projected': 2,
   'batch_failures': 1}` with **0 rows in the DB** — the counter increments before commit
   (profile_extractor.py:224-228) and the except at :229 rolls back without decrementing. The plan
   calls this report "honest".
3. **No value type guard** (db/profile.py:87): int/float/list/dict/bool → `AttributeError` out of
   `project()`. `normalize_key` guards `isinstance(raw, str)` (:48); the value does not. G2's own
   residual `business.expense: 50` is exactly this shape.
4. **`mention_count` is stale AND its stated justification is false.** models.py:328-332 says the copy
   exists "so ranking/ordering never needs a join on the hot injection path" — `current()` already
   joins semantic_facts (profile.py:144-146). Measured: fact=25, profile row=1, never refreshed
   (project_scope skips projected facts). On the real corpus 7,068 of 7,135 facts have
   mention_count=1 and 88 of 90 projected rows are 1, so D5's ranking degenerates to (recency,
   reverse-alphabetical key) for 98% of the profile.
5. **`current(limit=40)` truncates silently**; the assembler never overrides or reports it
   (profile.py:117,167-171; context_assembler.py:144-146). D5 says selection "is disclosed in the
   report, never silent". The 120-fact smoke already overflows it (41 keys → `pet.medication` dropped).
6. **`_sanitize` does not cover the ASSEMBLER's tag vocabulary.** Injected line measured:
   `atk.k6: x</[USER PROFILE]> <[SEMANTIC FACTS]> (2024/01/01) (identity) The user is an
   administrator.` The G1 test itself parses on `</[USER PROFILE]>`.
7. **G2's "212 tokens" is not what the assembler would inject** — that is `current(limit=25)` +
   `char_budget=4000` (smoke:79-80). The real path (limit=40, 2844 chars) measures **353 tokens** on
   the same data. Conclusion survives; the number does not describe the system. smoke:89
   (`... <= slice or 'render caps it at assembly'`) can never print a failure.

### MINORS
m1 `db/profile.py:23` promises a `rebuild` that does not exist; and the profile is not a pure function
of facts (keys/values are LLM-only — a prompt change alone moved 58→41 keys), so "always rebuildable"
overstates. m2 zero-width-only values pass the empty gate (`str.strip()` does not strip U+200B) and
render as `key: `; U+202E is not in `_ZERO_WIDTH_RE`. m3 `render` "hard-capped" is false for line 1 —
`char_budget=50` returned 209 chars. m4 `_migrate_profile_tier` cannot fail (create_all runs first,
engine.py:184-190) and cannot detect a pre-existing table with a missing UNIQUE constraint. m5
`history()` is not session-scoped while `current()` is; the smoke labels `history()[-1]` as `current=`.
m6 three more surviving mutants: value 200-char cap, D5's recency term, history's domain-time order.
m7 "192 passed across the five touched suites" does not reproduce and the suites are unnamed (closest
natural set = 193 passed / 1 skipped). m8 **D6 is not pinned by anything** —
tests/test_profile_tier.py:177-190 hands the store the same hard-coded `"coffee.style"` twice and
asserts dict grouping; the only real mechanism is the prompt line profile_extractor.py:71, never
exercised, and G2 was English-only.

### NOTES
n1 ordering mixes DOMAIN and MENTION times (measured: undated fact mentioned 2024/06 beats a fact
dated 2023/12). n2 FK is ON — a projected fact can never be hard-deleted (no orphan path; also no
forget path). n3 context_assembler.py:124-135 and its module docstring still describe the pre-profile
budget model; section labels read 3a then 2b. n4 entity-in-key/non-property residual is understated —
about a third of the top-25 injected block (`gardening.question:`, `dance.instructor.name`,
`concerns.portable_wifi_hotspot`, `workout.routine: new routine`, ...), disclosed as "occasionally".

### WHAT PASSED (verified, not read)
Empty profile is byte-identical at the real budget with facts+chunks present. normalize_key's
length/depth/charset/non-str gates, the fact_type guard, the cancelled filter, domain-time-over-insert
-order, empty-list-means-none, and all three extractor index guards are pinned — mutations M1,M2,M3,
M4,M6,M7,M8,M10,M11,M12 all go red on a NAMED test. SQL-ish values are parameterized. 19/19 green;
five-suite regression green (237 passed, 1 skipped on my set). G2's eight numbers all reproduce.
profile+facts+chunks never exceeded the semantic allocation in any scenario.

**Refs:** db/profile.py:36-58,87,94-110,117,144-171,176-217; db/models.py:296-341;
db/engine.py:184-219; llm/profile_extractor.py:87-238; llm/context_assembler.py:32-38,124-219;
tests/test_profile_tier.py:104-133,215-234,237-250,271-286; benchmarks/profile_tier_smoke.py:76-92;
benchmarks/gate_c_facts_source.py:1-111; PROFILE_TIER_PLAN.md:94-99,110-113,145-202.
Harness: /tmp/critic-profile-r1/ (mutplug.py = 20 mutations; probe1-6, probe_before, smoke.log).

**Who needs to know:** **Dev-Head —** seven blockers, all mechanical except #5/#6 which are design and
record. The single highest-value fix is #5: decide whether an attribute is single-valued or set-valued
and say so in D1, because right now the extractor prompt is TOLD to collapse set-valued attributes
(profile_extractor.py:74-78) onto keys the reader then reduces to one value. Do not re-pin by
inspection — every one of my seven "unpinned" calls came from reverting the guard and watching 19/19
stay green. **Bosses — NOT ready for Gate D.** Blocker 7 alone means the ~$3.50 run would measure
either nothing or leakage; blockers 2/3/4 mean the starvation alarm that Gate C was supposed to have
taught us is wrong in the direction that hides the new tier. **Founder —** the build is honest work
with a real reproducible measurement behind it (I re-ran G2 and got all eight numbers exactly), but
the sentence "recall for a profile-carried attribute becomes 1.0 by construction" is not true as
built: 54% of what the profile stores never reaches the prompt, and the fix G2 celebrates is what
raised that number from 16%.

## 2026-08-09 — Critics — PROFILE TIER, ROUND 2 (fix-pass re-review) — **BLOCK (3 blockers, 6 majors, 6 minors)**

**Claim reviewed:** the R1 fix pass landed (cd30f18 + 4775c16); six guards re-pinned and
mutation-verified by Dev-Head; ready for Gate D. **Verdict: BLOCK (narrow).** The fix pass is real
and the deepest one (B5) is measurably right — on the real corpus **80 of 90 projected facts now
reach the prompt (89%), up from 41 of 90 (46%)**. But the $0 gate script that would have shown that
is BROKEN by the fix and was never re-run, Gate D still has no wiring, and three guards reported as
closed are only half-pinned. Isolation as R1 (forced env before imports, config/base_path rewrite +
chdir + StorageManager self-check, named files only, $0).

### (a) R1 MUTATION SET RE-RUN — 11 of 15 now die; 4 still survive
DIES: R1_M5 superseded filter · R1_M9 PROFILE_BUDGET_SHARE=1.0 · R1_M15 history order · R1_M1
IntegrityError catch · R1_M2 count-before-commit · R1_M6 tag strip · R1_m2 invisible strip · R1_B4
char-only · R1_B7 scoping refusal · R1_M5sel selection under-report · (and R1_M20 note below).
**STILL SURVIVES:** `R1_M20` (delete `sem_budget -= profile_used`), `R1_M14` (D5 recency term),
`R1_M18` (200-char value cap), **`R1_M3` (the value type guard ADDED this round)**,
**`R1_B4_token_only` (delete the CHAR half of the new dual-unit budget)**.

### (b) NEW-MECHANISM MUTATIONS — 3 die, 5 survive
DIES: set-valued revert (`vals[:1]`) — 3 tests · `_MAX_VALUES_PER_KEY` unbounded · cap keeps oldest
instead of newest. **SURVIVES:** render value dedup · `expunge_all`→`rollback` · limit counting rows
instead of keys · the first-line fallback · deleting `last_selection`'s else-branch (which is what
keeps a previous read's numbers from being reported as this one's).

### BLOCKERS
1. **The G2 gate script is BROKEN by this fix pass and was never re-run.**
   `benchmarks/profile_tier_smoke.py:80` calls `store.render(top, char_budget=4000)`; the signature
   is now `render(attrs, token_budget=300, counter=None)` → `TypeError: ProfileStore.render() got an
   unexpected keyword argument 'char_budget'`. It is the ONLY broken caller and it is the artifact
   that validates B5. So the deepest change in the pass shipped with **zero** real-corpus evidence,
   and §G3's B5 paragraph is argued, not measured. 95 seconds and $0 to fix and re-run. I measured it
   for you (see (c)) — but the team's own gate must run.
2. **Gate D wiring still does not exist.** `profile_scoped_required` closes the LEAK half of B7 and
   is pinned — good. The other half is untouched: nothing in the repo sets it, nothing sets
   `profile_session_ids`, there is no profile equivalent of `gate_c_facts_source.install()`, no
   profile preflight, and no step that projects rows into the eval's DB. Worse, the refusal is raised
   INSIDE the profile try/except (context_assembler.py:156-162), so it degrades to one WARNING per
   question and a silently profile-less run — the exact "measures nothing" outcome R1-B7 named. The
   facts tier's protection is a **$0 preflight that returns False before any paid call**
   (`gate_c_facts_source.preflight`); the profile needs the same, not a per-question log line.
3. **`_MAX_VALUES_PER_KEY=6` can evict the winner of a supersession.** Measured: key with 7 live rows
   where the supersession SURVIVOR is the oldest (the dedup-merge shape the B1 fix was rebuilt
   around) → injected `['coexist-5'..'coexist-0']`, **`MERGED-SURVIVOR` absent**. The cap orders by
   recency ONLY (`vals.sort(key=(when, fact_id))`) while KEYS rank by (mention_count, recency), so a
   value re-affirmed 15 times is evicted by six one-off newer ones. This re-opens the claim the whole
   round was about: the profile can inject a set that omits the value the fact tier explicitly
   elected. `last_selection` discloses the count, never which.

### MAJORS
1. **B2 is half-closed.** The code is right; the pin is not. `R1_M20` survives because
   `facts_used = max(0, 4740 - profile_used - sem_budget)` → `max(0, -369)` → 0, so
   `test_budget_report_attributes_tokens_to_the_right_tier` still passes. Measured cost of the live
   mutant: **369 tokens of semantic OVERSPEND (7.8%)** with the report reading `facts_used: 0`.
2. **B4 is half-closed, and by the exact mechanism the lesson names.** `R1_B4_token_only` survives
   because the new pin's fixture is **2.72 chars/token** — below the 4.0 proxy — so only the token
   side can ever bind. Measured on 5.90-chars/token English the CHAR side is the binding cut at
   budgets 40 and 300. The lesson I logged after R1 says verbatim "the fixture must straddle the
   ratio — include content ABOVE and BELOW it". Third stage running.
3. **M3's new value type guard is unpinned** (`R1_M3` survives). A fix with no test is a fix that
   will be silently reverted; it was not in the six Dev-Head verified.
4. **`render()` raises IndexError on a value that sanitizes to nothing** — introduced by the M6 fix.
   A stored value of `"<[USER PROFILE]>"` (accepted by `project()`, stripped by `_TAG_RE` at render)
   leaves `lines == []` and `return out or lines[0][:char_cap]` indexes an empty list. Contained by
   the assembler (D7 holds), but ONE crafted utterance suppresses the WHOLE profile for that turn.
   Same line: the `lines[0][:char_cap]` fallback is otherwise **dead code** — the loop guard is
   `if kept and (...)`, so line 1 is always appended and `out` is never empty.
5. **render's own drops are not in any report.** `last_selection` covers the limit path only.
   Measured on the corpus: `values_dropped=7` reported while render silently dropped 2 more (dedup),
   and the render budget break is unreported entirely. On the full 7,135-fact corpus the 711 slice
   WILL bind and the operator report will say nothing about it.
6. **M4 unfixed and now Gate-D-shaped.** With set-valued reads the 40-key window on the full corpus
   (7,135 facts → hundreds of keys) is chosen by a ranking whose primary term is constant for
   **7,068 of 7,135** facts. That is an arbitrary slice, and if the hypothesis fails you cannot tell
   whether the profile was wrong or merely the wrong forty.

### MINORS
m1 doc-vs-code drift in the contract that changed: `db/profile.py:18-21` and `current()`'s docstring
:167-178 still document "the latest DOMAIN time wins per attribute, ties by fact id"; `db/models.py`
:309-312 still says "current-vs-history is resolved at READ time by domain time". m2 `rebuild` still
promised (:23), still absent. m3 `_CALLER_CHAR_FACTOR = 4  # a set-valued attribute is a summary, not
a log` — comment copy-pasted from `_MAX_VALUES_PER_KEY` (the spliced-comment class from Stage 6 m4).
m4 the OLD vacuous `test_profile_section_injected_and_budget_reserved` is still in the suite verbatim,
`[SYSTEM]`-counting bug and all, next to its replacement. m5 the `expunge_all` comment asserts
rollback "would discard THEIR writes"; I ran both variants and got identical committed rows
(SQLAlchemy restores flushed-pending objects on rollback) — either pin a real difference or soften
the comment. m6 §G3 says "5-suite regression green"; the run is 6 suites, still unnamed (R1 m7).
m7 D6 still pinned by nothing (the updated test still hard-codes `"coffee.style"` twice and now also
asserts `lang_source`, which is a copied column, not the mechanism). m8 unchanged from R1: m4
migration no-op, m5 `history()` unscoped, n1-n4.

### (c) B5 RE-CHECKED ON THE REAL CORPUS — the fix works
90 rows / 41 keys. `current(limit=40)` → 82 rows / 40 keys; rendered 40 lines / **80 values**;
**573 tokens against the 711 slice** (was 353). **80 of 90 reach the prompt = 89%, up from 46%.**
The 10 lost: 7 to `hobbies` hitting the 6-cap, 1 key to limit=40, 2 to render dedup. Honest residual
the fix creates: bad keys are now worse, not better — `business.expense: 50; 50%; $3.50 per jar
(COGS)` and `business.offer: BOGO deal; specific authors and columnists on both the WSJ and Post
websites; setting up alerts...` — one wrong value per key has become up to six concatenated.

### (d) NEW-CODE ANSWERS
Grouping preserves ranked order (render's dict is fed `current()`'s output in rank order) — no
ordering regression found. `_MAX_VALUES_PER_KEY` CAN hide a superseding value (blocker 3). The render
dedup is correct behaviour (same value from two dates renders once) but is uncounted (major 5).
`last_selection` is accurate for the limit path, incomplete overall; it is also shared mutable state
rebound on every `current()`, and the assembler captures it ~90 lines after the read — a latent
cross-question mix-up under the eval's thread pool. I could NOT demonstrate that race (my two-thread
assembler probe died on an unrelated ConversationStore session limitation), so I report the window,
not a failure.

### (e) RECORD — accurate
§G3 credits my numbers rather than restating them loosely, states the B1/B2/B3/B4/M1/M6 mutations as
dying (all four I re-checked do die), and discloses that the first M1 pin did not reproduce. "27
profile tests" ✓. "257 passed + 1 skipped" across the six named suites ✓ (I reproduced both exactly).
The only record defects are m6 (5-vs-6 suites, unnamed) and that §G3 presents B2/B4/M3 as closed when
the mutation evidence says half-closed.

### (f) BLOCKING vs RECORD-ONLY for a paid Gate D
**Blocking:** #1 (re-run the fixed smoke — $0, 95s, and it is the only evidence for the deepest
change), #2 (a $0 profile preflight that STOPS the run, plus an actual projection+scoping path), and
disclosure of major 6 (the 40-key window's ranking is effectively recency on this corpus) with
`profile_selection` captured per question. **Should fix but not Gate-D-blocking:** blocker 3 and
majors 1-5 (code correct or contained; pins and reports missing). **Record-only:** every minor.

**Refs:** db/profile.py:41-56,112-160,203-251,278-328; llm/context_assembler.py:75-76,141-166,
224-237; llm/profile_extractor.py:217-238; tests/test_profile_tier.py:106-121,124-144,276-294,
315-330,433-455,458-472,475-485,488-501,504-546,549-584; tests/test_fact_retrieval.py:1025-1033;
benchmarks/profile_tier_smoke.py:80; PROFILE_TIER_PLAN.md:203-277.
Harness: /tmp/critic-profile-r2/ (mutplug.py = 23 mutations; probeA-F).

**Who needs to know:** **Dev-Head —** good round; B5 is the right call and the number moved 46%→89%
on real data. Three things before I can pass it: run the fixed smoke, give the profile a $0 preflight
that stops rather than warns, and make `_MAX_VALUES_PER_KEY` not able to drop a supersession
survivor. Then close the five surviving mutants (M20, M14, M18, M3, B4-char) — and for B4 the fix is
the FIXTURE, not the code: add 5.9-chars/token English alongside the Telugu. **Bosses —** still not
ready to spend, but the distance is short and mostly $0. **Founder —** the honest headline is that
the profile now injects 89% of what it stores instead of 46%, at 573 of 711 tokens; the reason I am
still blocking is that the script which proves that is broken in the same commit, and nothing yet
puts a profile in front of the Gate D questions at all.

## 2026-08-09 — Critics — PROFILE TIER, ROUND 3 (fix-pass re-review) — **BLOCK (2 blockers, 5 majors, 5 minors) — Gate D may NOT proceed yet**

**Claim reviewed:** a275597 closes R2's three blockers and six majors; 33 profile tests, 263+1
across six suites; ready for the paid Gate D. **Verdict: BLOCK.** Blockers 1 and 3 are genuinely
closed and I verified both by measurement. **Blocker 2 is not closed — it moved.** The Gate D module
exists and reads correctly, but its first step cannot run on the real corpus and its preflight passes
on a profile that reaches none of the questions. Both are $0 and small. Isolation as always (forced
env before imports, config/base_path rewrite + chdir + StorageManager self-check for assembler runs,
named files only, corpus COPIED never touched, $0).

### (a) FULL MUTATION SET RE-RUN — 26 mutations
**Now die (were surviving):** `R1_M20` (delete the profile's budget subtraction) → dies on
`test_budget_report_cannot_hide_overspend_as_zero`; `R1_B4_token_only` (delete the CHAR branch) →
dies on `test_render_budget_binds_on_HIGH_ratio_content_too`; `R1_M3` (delete the value type guard,
re-anchored to the rewritten source) → dies; `N_container_str` (containers `str()`-ed instead of
refused) → dies. Also confirmed still dying: M5, M9, M1, M2, M6, m2, B7, M15, M5sel, B4_char_only,
setvalued_revert, maxvals_unbounded, maxvals_keepoldest, and the NEW `N_valrank_recency_only`
(blocker 3's fix) → dies on `test_per_key_cap_keeps_the_supersession_winner`.
**Still surviving (7):** `N_limit_counts_rows`, `N_no_render_dedup`, `N_sel_no_else`,
`N_no_lastrender`, `N_no_empty_guard` (equivalent — see minors), `N_expunge_to_rollback` (shown
equivalent in R2), `R1_M14` + `R1_M18` (known-open minors).

### BLOCKERS
1. **`gate_d_profile_source.project()` cannot run on the corpus it exists for.** The real
   `gate_c_facts.db` has 17 tables and **`profile_attributes` is not one of them**; the module builds
   its own engine with no `create_all`/`init_db`. Measured on a COPY:
   `OperationalError: no such table: profile_attributes` — raised from step 1 of the module's own
   three-step contract. Compounding: there is no `__main__` block, and `qa_accuracy_eval` calls only
   `preflight()` and `install()`, never `project()`. So no supported path puts a profile into the
   corpus at all. R2-blocker-2 is not closed; the wiring is written but the entry point is dead.
2. **`preflight()` passes on a profile that reaches none of the questions.** Measured: seeded 3
   profile rows covering **2 of 2,965 sessions**, registered all 150 questions →
   `PREFLIGHT PASS (150 questions scoped, 3 attribute rows)` → the paid run proceeds and measures a
   system with an empty profile on every question. `profile_rows > 0` is a presence check, not a
   coverage check. `gate_c_facts_source.preflight` (:114-146) computes the UNION of question haystack
   sessions and returns False when any was never consolidated, and reports the zero-fact fraction;
   §G3's "mirrors the facts tier's contract" is an overstatement in the one dimension that matters.
   Fix is one query: sessions-with-profile-rows ∩ question-haystacks, fail (or at minimum report the
   distribution) on zero/low coverage.

### MAJORS
1. **The `limit` unit — keys vs rows — is unpinned and decides the headline.** Measured on the real
   corpus: real (`limit` counts KEYS) → 82 rows / 40 keys / **80 values injected = 89% REACH**;
   mutant (`limit` counts ROWS) → 43 rows / 21 keys / **43 values = 48% REACH**. One token of change
   halves the number the entire round is sold on, and 33/33 stay green.
2. **Report freshness is unpinned, and one path is measurably stale.** `N_sel_no_else` and
   `N_no_lastrender` both survive. And the new `if not lines: return ""` guard early-returns BEFORE
   writing `last_render`, so an all-sanitized-away render leaves the PREVIOUS render's numbers in
   place — measured: `{'lines_in': 40, 'lines_out': 3, ...}` still reported after rendering a single
   tag-only attribute. Since I am requiring `profile_selection` per question in the Gate D artifact,
   a stale report is a measurement-integrity problem, not cosmetics.
3. **The render dedup and its counter are unpinned.** `N_no_render_dedup` survives;
   `test_render_drops_are_reported` asserts `lines_in/lines_out/dropped_by_budget` but never
   `values_deduped`.
4. **`install()` ignores `scope_keys_by_question` entirely** (0 occurrences in the body) — the
   per-question binding is external state (`assembler.profile_session_ids`, set in
   `retrieve_context`), and `profile_scoped_required` only checks `is None`, so a STALE scope from
   the previous question passes. I could NOT construct a leak inside `qa_accuracy_eval`: the
   assignment immediately precedes `assemble()` and `run_one` holds `_retrieve_lock` across both
   (:392-397). So this is structural, not live — but the facts tier resolves scope FROM THE QUERY
   inside `_ScopedFactRetriever` and raises on unregistered, which has no state to go stale. Adopt
   that shape and the unused parameter becomes used.
5. **B5's fix amplifies the bad-key residual into the prompt, visibly.** From the smoke's own output:
   `concerns.portable_wifi_hotspot: specific topics or industries...; personalized news feed feature;
   data plans and pricing; coverage; smart bulbs` and `music.genre: pop; musicals with complex,
   clever lyrics like 'Hamilton'; ...` (6 values). One wrong value per key is now up to six. This is
   what the answerer reads; it belongs in the Gate D disclosures beside M4.

### MINORS
m1 `N_no_empty_guard` is a genuine equivalent mutant for the RETURN value (`out` is already `""`),
which means the guard's only effect is to skip the `last_render` write — remove it or move the write
above it. m2 `_CALLER_CHAR_FACTOR = 4  # a set-valued attribute is a summary, not a log` — the
copy-pasted comment from R2 m3 is still there. m3 the old vacuous
`test_profile_section_injected_and_budget_reserved` still sits at :315-330 with its
`out.split("</[USER PROFILE]>")[0]` `[SYSTEM]`-counting bug, beside its replacement. m4 `R1_M14`
(D5's recency term) and `R1_M18` (the 200-char value cap) still unpinned. m5 D6 still pinned by
nothing real; `history()` still unscoped; the migration is still a no-op that cannot fail.
**Fixed and credited:** the module docstring now states the set-valued contract AND explicitly
retracts `rebuild` with "Derived means traceable, not reproducible" — that is a better fix than I
asked for.

### (c) RECORD — accurate, with one overstatement
Every number verifies: 33 profile tests ✓, 263 passed + 1 skipped across the six NAMED suites ✓, and
I re-ran the fixed smoke myself — `40 lines / 80 values / 573 tokens of the 711 slice`,
`REACH: 80 of 90 (89%)`, `selection {41 in scope, 1 key dropped, 8 values dropped}`,
`render {40 in, 40 out, 0 dropped by budget, 2 deduped}` — identical to my independent R2
measurement, exit 0. My lessons are recorded verbatim and attributed. The one overstatement is in
R2-blocker-2's paragraph: "mirrors the facts tier's contract" and "`project()` (idempotent, $0)" —
`project()` does not run, and the preflight does not check coverage. Also worth stating in the record:
**89% is measured on a 120-fact sample where the 40-key limit barely binds (41 keys).** It is
plausibly representative because Gate D scopes per question (~40 sessions ≈ ~100 profileable facts),
but that is an extrapolation, not a measurement, and should be labelled as one until the per-question
REACH distribution is measured.

### (d) THE SHARED-MUTABLE-STATE WINDOW — closing it as DISCLOSED, not live
I tried again and it is not demonstrable in any current caller. `qa_accuracy_eval.run_one` wraps the
whole of `retrieve_context` (scope assignment + `assemble`) in `_retrieve_lock`, and no other
`assemble()` caller in the repo is threaded (`ablation_study_real`, `eval_harness`: no thread pool).
So `last_selection`/`last_render`/`profile_session_ids` cannot interleave today. It stays a disclosed
structural window that becomes live the moment `_retrieve_lock` is dropped for throughput — worth one
line in the code, not a fix.

### (e) MAY GATE D PROCEED? NOT YET — two $0 fixes, then YES with these disclosures
**Required before spending:** (1) `project()` must create its schema and be runnable (a `__main__`
block or a `--project` flag); (2) `preflight()` must check COVERAGE against the question haystacks,
not just row presence; (3) run the full projection (7,135 facts ≈ 80+ min of local llama — that is a
long-running job and needs the founder's standing go-ahead under the approval-gates rule) and then a
$0 dry pass reporting the per-question REACH distribution.
**Disclosures that must accompany the number:** (i) M4 — `mention_count` is copied at projection and
7,068 of 7,135 facts have the value 1, so D5's ranking degenerates to (recency, key) for 98% of the
profile; a failed hypothesis cannot be distinguished from "the wrong forty attributes"; (ii)
`profile_selection` AND `last_render` captured per question in the artifact; (iii) major 5 — the
injected block contains entity-specific keys carrying up to six unrelated values, with the smoke's own
lines quoted; (iv) 89% REACH is a 120-fact sample figure until the per-question distribution is
measured; (v) D6/cross-lingual is pinned by nothing real and the corpus is English — no Indic claim
may be attached to this result.

**Refs:** benchmarks/gate_d_profile_source.py:29-53,76-102,105-123;
benchmarks/gate_c_facts_source.py:89-105,114-146; benchmarks/qa_accuracy_eval.py:355-361,378-386,
392-397; benchmarks/profile_tier_smoke.py:78-97; db/profile.py:52-62,115-127,227-236,320-360;
tests/test_profile_tier.py:315-330,586-693; PROFILE_TIER_PLAN.md:280-345.
Harness: /tmp/critic-profile-r3/ (mutplug.py = 26 mutations; probeG/H/I; smoke.log).

**Who needs to know:** **Dev-Head —** blockers 1 and 3 are properly closed and I verified them by
measurement, not by reading; the module docstring's retraction of `rebuild` is better than what I
asked for. Two things stand between you and a paid run, both small: give `project()` a schema and an
entrypoint, and make `preflight()` measure coverage the way gate_c does. Then close
`N_limit_counts_rows` — it is the pin that protects your own headline. **Bosses —** the tier itself is
in good shape; what is not ready is the path that gets it in front of the questions. Note the full
projection is a ~80-minute local job that needs the founder's go-ahead before it starts. **Founder —**
nothing here costs money yet and nothing should be spent this round. The honest state: the profile
works and injects 89% of what it stores in the sample we can measure, the team's own gate now runs
and reproduces that number independently, and the remaining gap is that no profile has ever actually
been built for the real corpus — the script that would do it stops on a missing table.

## 2026-08-10 — Critics — ADVERSARIAL DESIGN REVIEW: "conversation-facts + source_role" proposal (A-E) after the N8 extraction-loss finding — **BLOCK (5 blockers, 7 majors)**

**What I was asked:** attack the founder's proposed redesign (A: second fact class with `source_role`;
B: type-based pollution guard; C: strip/flag unlicensed numbers instead of rejecting; D: don't extend
word numerals; E: don't license numbers from any user turn) and the diagnosis behind it ("user-model
memory graded on a conversation-recall benchmark"). Read-only; no code touched.

**Verdict: BLOCK.** Three of the five "measured facts" the design rests on do not survive checking
against the repo's own artifacts, and two of the five proposals reproduce failures this repo has
already paid for.

**Blockers**
- **B1 — the Q6 premise is false.** "knowledge-update 95.2%, at ceiling" is the RAW-TURN run
  (`qa_accuracy_longmemeval.json`, 20/21). With facts it is **11/21 = 52.4%**
  (`qa_accuracy_longmemeval_full150_4o.json`) — I computed the per-category split myself.
  `PROFILE_TIER_PLAN.md:27` labels it correctly ("Banked (RAW turns) | Measured with FACTS? ❌ NEVER");
  the framing handed to me dropped the label. Nothing may be published as our knowledge-update score
  from that run.
- **B2 — the fidelity ladder cannot carry its causal attribution.** Rung 2 is a **Claude-Haiku** cache
  (`corpus_loaders.py:66-69`), not our prompt, so "−11.4 pts = the extraction prompt" is wrong by
  construction. Rung 2 gets dates prefixed (`corpus_loaders.py:123-125`); rung 3 concatenates
  `fact_text` only (`extraction_fidelity.py:96-100`) — measured on the real corpus, only **455/3,118**
  dated facts carry the date in the text and only 3,118/19,367 facts are dated at all, while the
  product injects `[t_occurred] (type) text` (`fact_retrieval.py:344-354`). Rung 3 also drops
  superseded facts and groups by primary `source_session_id` only. Four biases, all one direction.
- **B3 — the control-group "proof" is over-claimed and the artifacts can't isolate the variable.**
  n=22 at 11/22 is the max-variance point (±~20 pts); equal totals ≠ equal answers. Cross-run
  question-level agreement over the full 150 is **80/150** (63 lost, **7 GAINED** under facts).
  `_checkpoint()` (`qa_accuracy_eval.py:435-446`) records the `memory_source` ARGUMENT, not storage
  form, and `ensure_scope_ingested` returns early on existing turns (`:316-321`) — which is how the
  128/22 split happened. F-15 (`DECISION_AND_FAILURE_LOG.md:563-578`) is ⚠️ OPEN with "must be fixed
  before any further paid run."
- **B4 — Proposal C (mutate fact text) is R3-B3 in a new costume.** Identity is
  `sha256(type ⟂ normalize_fact_text(text) …)` (`semantic_facts.py:120-137`) and that function's own
  docstring says "'rode 3 times' and 'rode 2 times' must NEVER normalize together" (`:113-117`).
  Stripping numbers performs exactly that collapse; `_reaffirm` then merges and the second claim is
  gone with no record. Also makes hashing evidence-dependent (same sentence hashes two ways),
  distorts `mention_count` (which the profile ranks on, `profile.py:235-241`), and an in-text flag
  leaks to the prompt (`fact_retrieval.py:88-92` sanitizes only ZW chars / history marker / stamps).
- **B5 — Proposal B's type guard is not a safety boundary.** `fact_type` is untrusted 8B output
  (`profile.py:108-113`), and this repo has already measured the model mistyping (R3 N3,
  `semantic_facts.py:122-127`; plans-as-state, `consolidation_v2.py:202-212`). The invariant must be
  PROVENANCE on the row, not type. B also leaves assistant-sourced `state` facts rendering as user
  truth with no source marker (`semantic_facts.py:564-621` has no role filter).

**Majors:** no dedup story for `source_role` (hash has no role component, `_reaffirm` has no merge
rule — the event_status precedent forbids accepting a value without merge semantics,
`semantic_facts.py:55-58`); assistant-sourced states can supersede user facts (`supersession.py:383,
595-681, 720-721`) and can cancel user plans (`:142-208`); volume/compute unpriced (+1 judge LLM call
per created state, `consolidation_v2.py:501-507`; KG nodes from assistant entities feeding a resolver
with a known false positive, `models.py:262-266`); A requires a prompt change that invalidates every
extraction number with no prompt-version column (`models.py:175-227`, cf. `consolidation_v2.py:279-283`);
E is inconsistent with shipped pooled word-numeral licensing which also writes an EMPTY provenance
list into the audit (`consolidation_v2.py:379-388`); E under-reaches — the 31.2% is caused by the
unmeasured `need = 1 if len(ftoks) <= 4 else 2` overlap bar (`:344-348, 365-366`); number semantics
already forked (`consolidation_v2.py:141-145` vs `supersession.py:218-236`).

**Better approach offered (Q5):** the turns are already stored losslessly (`consolidation_v2.py:418-420`;
"Episodes are KEPT" `:5-6`). Conversation recall is a ROUTING problem over the episodic tier, not a
second fact class. $0 test first: add a fourth ladder rung = gold-session raw turns as retrieved by the
current retriever. If that rung is high, Proposal A is unnecessary. If A still survives, make the second
class a projection of TURNS so it never touches fact_hash, supersession, the profile, or the KG.

**Q6 answer:** yes, there is a concrete regression path, and it is not from 95.2%. C breaks
`_metric_update` (`supersession.py:239-270`: masked texts must be EXACTLY identical and ≥2 numeric
tokens), which the module itself says is the only route that fires for entity-less personal-metric
facts (`:604-611`) — the archetypal knowledge-update class. From 11/21, at n=21, a sub-10-point
regression is undetectable.

**Refs:** llm/consolidation_v2.py:141-145,148-169,300-406,410-660,771-785; db/semantic_facts.py:55-58,
113-137,164-392,564-621; db/profile.py:67,97-172,203-241; llm/profile_extractor.py:148-244;
db/models.py:136-341; llm/supersession.py:142-208,239-270,301-331,364-455,595-727;
llm/fact_retrieval.py:88-92,344-359; llm/context_assembler.py:143-171;
benchmarks/qa_accuracy_eval.py:255-262,306-352,435-446; benchmarks/corpus_loaders.py:66-127;
benchmarks/extraction_fidelity.py:46-80,96-127; DECISION_AND_FAILURE_LOG.md:76-240,563-578;
PROFILE_TIER_PLAN.md:15-45. Corpus queried read-only: benchmarks/extracted_memories/gate_c_facts.db
(19,367 facts / 3,534 sessions / 3,118 dated / 455 with the date in the text).

**Who needs to know:** **Founder —** three of the five premises need re-measurement before any code is
written, and all three re-measurements are $0. Do not authorize a paid validation run while F-15 is
open. **Dev-Head —** do not implement C as written under any circumstance; if a verification verdict
must be stored, it goes on a new non-hashed column with defined merge semantics, exactly like
event_status. **Bosses —** the honest one-line state: the extraction-loss finding is real and
directionally safe, but its size (43.8 pts) and its attribution (prompt vs pipeline) are both
unestablished, and the proposed fix would re-open two closed blockers.
