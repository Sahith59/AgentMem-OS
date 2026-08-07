
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
