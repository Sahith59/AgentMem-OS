
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
