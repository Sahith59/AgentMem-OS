# Consolidation v2 — Build Log (running doc, updated every stage)

The long-running record of the semantic-tier build: what was built, what evidence
proves it works, what the critic pass found, and every decision with its reason.
Companion to `CONSOLIDATION_V2_DESIGN.md` (the design + plan of record),
`CONSOLIDATION_V2_RESEARCH.md` (biology + prior-art audit), `RUNNING_NOTES.md`
(day-to-day). Rule: nothing in the plan-of-record table flips to DONE unless its
three gates are recorded HERE with evidence.

## Operating contract (founder-set, 2026-08-06)

- Long-term reliability over fast results. No happy paths, no shortcuts, no jargon.
- **Triple gate for every stage:** (G1) tests green incl. failure paths; (G2)
  end-to-end smoke on real data with artifacts pasted here; (G3) adversarial critic
  pass, findings resolved or founder-accepted.
- Novelty claims: only what the research audit supports, re-verified at publish time.
  Target distinctives: atomic dated cited facts; count-preserving aggregation-aware
  consolidation; running aggregate facts (phase 2); cross-lingual canonical facts.
- Deliberate exclusions (decided, not forgotten): write-time fact INTEGRATION
  (CLS-theory + Mem0's broken shipped implementation argue against; cheap write-time
  entity indexing stays); procedural-tier expansion (no benchmark category tests it —
  over-engineering); KG rebuild-from-scratch (existing tested KG gets fed by facts
  instead).
- Founder approves stage transitions. Cluster access arrives when build is READY.

## Scope (founder-confirmed 2026-08-06)

Semantic fact tier + bi-temporal upgrade of the existing KG (fed by facts through
ALIAS_OF) + consolidation agent (session-end batch distillation) + phased sleep-time
compute (v1: session-end; phase 2: idle-time aggregate/tally passes) + profile tier
(next build, after the eval). NOT write-time fact integration.

## Stages

| Stage | Content | G1 tests | G2 smoke | G3 critic | Status |
|---|---|---|---|---|---|
| 0 | Implementation research + design freeze | — | — | founder review | research agent RUNNING (launched 2026-08-06) |
| 1 | SemanticFact schema + storage + CRUD | ✅ 52/52, 100% cov | ✅ real data | **R6 PASS** (R1-R5 ✗, each fixed) | ✅ DONE 08-06 |
| 2 | Consolidation engine rewrite (distillation) | ☐ | ☐ | ☐ | — |
| 3 | KG integration (facts→entities/edges, provenance) | ☐ | ☐ | ☐ | — |
| 4 | Per-fact supersession | ☐ | ☐ | ☐ | — |
| 5 | Facts-first retrieval wiring + $0 diagnostics | ☐ | ☐ | ☐ | — |
| 6 | Full E2E smoke + final critic pass → BUILD READY | ☐ | ☐ | ☐ | — |

After Stage 6: founder provides university-cluster access → slice-haystack extraction
through the REAL pipeline → $1.50 Gate C eval (stop rule: beat 0.519) → Gate D 150 →
Gate F 500 → cross-lingual/Sarvam only after we compete with current competitors.

## Decision log

- 2026-08-06: Founder reordered build-before-extract; proxy extraction retired at
  48/3,631 sessions (JSONL kept as reference only).
- 2026-08-06: Founder set the operating contract above; triple-gate protocol adopted.
- 2026-08-06: Scope confirmed (this doc); write-time integration excluded with reasons.

## Stage records

### Stage 1 — SemanticFact schema + storage (G1 ✅, G2 ✅, G3 in progress, 2026-08-06)

**Built:** `SemanticFact` model in `db/models.py` (tri-temporal, cited, scoped,
hash-deduped, superseded_by/superseded_at, mention_count/last_confirmed_at per
Stage 0 amendments; UNIQUE(scope_key, normalized_hash) as final dedup authority;
partial index `idx_facts_current WHERE superseded_by IS NULL` + range + provenance
indexes); `_migrate_semantic_tier()` in `db/engine.py` (retrofits the kg_edges
active-edge partial index); `db/semantic_facts.py` `SemanticFactStore` (add_fact
with validation-before-write, re-affirmation path with citation union, race →
re-affirm fallback; supersede with distinct/exists/same-scope/no-double/no-cycle
guards; current_facts with LITERAL `superseded_by IS NULL` predicate; facts_as_of
point-in-time reconstruction; chain walk; read-time transition_text synthesis;
provenance with citation-integrity flag).

**G1 evidence:** `tests/test_semantic_facts.py` — **18/18 passed** (0.52s).
Failure paths seeded from real competitor bugs: contradiction rows must not
coexist (Mem0 #4896 class), re-affirmation not duplication (bloat class #4573),
DB-constraint-as-final-authority under race (#6531 class), supersession × limit
windows never double-count (Letta #3088 class), numeric difference never
collapses ("three times" vs "two times" stay distinct — the aggregation-thesis
guard), cross-scope supersession refused, cycles refused, unparseable dates loud.

**G2 evidence:** `benchmarks/consolidation_v2_stage1_smoke.py` on REAL Gate-B
llama3.1 facts from real LongMemEval sessions, fresh scratch DB:
```
PASS 1: 93 created, 2 re-affirmed, 8 with in-text event dates
PASS 2 (idempotency): 0 created, 95 re-affirmed
current facts: 93 (== 93 expected: True)
  [2023/10/15 | said 2023/11/04] The user rode the Revenge of the Mummy rollercoaster three times...
  [2023/10/08 | said 2023/11/04] The user rode the Xcelerator rollercoaster at Knott's Berry Farm...
query plan: SEARCH semantic_facts USING INDEX idx_facts_current (scope_key=? AND fact_type=?)
PARTIAL INDEX CONFIRMED IN USE
```
The tri-temporal separation (occurred October, said November) is stored reality.

**G3 round 1: BLOCKED — 5 blockers, 11 majors, 5 minors.** The critic measured
real breaks: threaded writers on the shipped engine config lost 126 of 300 facts
with false success; racing supersedes recreated Mem0's contradiction bug inside
our own store and formed a chain cycle that hung the audit path; same-text
events on different dates collapsed into one row (under-counting the exact
aggregation questions v2 exists for); user facts merged across users sharing an
agent; FK violations reported as successful re-affirmations. Full verdict
preserved in the critic's report; resolutions:

| # | Finding | Resolution |
|---|---|---|
| F1 BLOCKER | Shared-connection concurrency destroyed writes | In-process `_WRITE_LOCK` write serialization (Stage-0 item 7's missing half) + threaded test: 4 threads × 100 shared facts on the PRODUCTION engine config → 100 rows, mention_count sums exact, zero errors |
| F2 BLOCKER | supersede TOCTOU; chain() infinite loop | Atomic conditional UPDATE (`WHERE superseded_by IS NULL`, rowcount-checked) + seen-guards/depth caps on every walk + racing-supersede test: exactly one winner |
| F3 BLOCKER | Event occurrences collapsed by text-only dedup | `t_occurred` joins the hash key for events; test: same text 2 dates → 2 rows, same date → re-affirm |
| F4 BLOCKER | Scope collapse (user ignored under agent; alice/alice; literal "global") | Composite `make_scope_key()` (sole derivation, both axes); read APIs accept agent_id/user_id; 3 collision tests |
| F5 BLOCKER | `except Exception` → false success on FK violations | Narrowed to the dedup constraint only — and writing its test exposed a REAL second bug: SQLite names COLUMNS not constraints in unique errors, so the original match could never fire. Both spellings matched now; ghost-session FK test raises loudly |
| F6 MAJOR | No calendar validation | strptime round-trip; month 13 / Feb 31 / 00/00 / 9999/99/99 / trailing-garbage "1x" all raise (tested) |
| F7 MAJOR | Future-dated events stored silently; smoke fabricated years | Smoke now skips derived dates later than the mention date (8→6 dated rows — the 2 impossible ones gone). Store DELIBERATELY permits future t_occurred (planned events are real facts); Stage 2 extractor gets explicit planned-event handling. **Founder may veto.** |
| F8 MAJOR | Partial dates fall out of range filters | Month-only dates stored as explicit [first_day, last_day] interval (`t_occurred_end`); `facts_overlapping()` overlap predicate; October-range test includes the month-only fact |
| F9 MAJOR | Full-scan + temp-sort on default read; unbounded facts_as_of | Leading NULL-check expression dropped (SQLite DESC natively sorts NULLs last); second partial index `idx_facts_current_all` for the untyped path; `idx_facts_mentioned`; limit on facts_as_of; tests assert NO TEMP B-TREE on the store's own compiled SQL |
| F10 MAJOR | Smoke EXPLAINed a lookalike string | Smoke + tests now EXPLAIN the ORM-compiled query itself, both hot paths |
| F11 MAJOR | Zero concurrency coverage; constraint test tested SQLite | 36 tests, 97% line coverage; threaded add/supersede; cross-process race simulated deterministically (pre-check blinded once). Remaining 8 uncovered lines are cross-process defense-in-depth branches unreachable in-process by design — documented, not faked |
| F12 MAJOR | Migration never ran, silent except:pass | Loud report dict + logger; RuntimeError on unrepairable state; **evidence: ran against a COPY of the real dev DB — kg_edges (34,905 rows) got idx_kg_edges_active, semantic_facts verified with full index set** |
| F13 MAJOR | No migration path for constraints | `_migrate_semantic_tier` verifies constraint+indexes on existing tables; missing constraint → RuntimeError with rebuild instructions (SQLite can't add constraints in place); both paths tested |
| F14 MAJOR | Per-fact autocommit forecloses batch consolidation | Every method accepts caller `db=`; store flushes, never commits/closes caller sessions; abort-mid-batch test → zero rows persisted |
| F15 MAJOR | Re-affirmation dropped sessions/langs/dates | `source_session_ids` + `langs` JSON columns, t_occurred backfill for undated states; cross-lingual re-affirmation test (en+hi → one fact, two sessions, two langs) |
| F16 MAJOR | chain() truncated merge branches | Closure algorithm over both directions to fixpoint; many-to-one test: identical complete view from any member |
| F17 MINOR | Dangling superseded_by hides facts | Latent (no delete path exists); documented as accepted risk |
| F18 MINOR | bools passed as turn ids | `type(i) is int` check, tested |
| F19 MINOR | LIKE wildcard injection in contains | Escaped, tested with "100%" |
| F20 MINOR | Vacuous smoke limit assert | Real count assertion against table cardinality |
| F21 MINOR | utcnow deprecation | Codebase-wide convention (models.py throughout) — carried as repo-wide debt, not fixed piecemeal |

**G3 round 2: BLOCKED AGAIN — and rightly.** The critic verified 8 findings
genuinely fixed (writer-vs-writer at 400 facts on the real config; a REAL
two-process race exercising the narrowed-IntegrityError fallback; the full
6-cell scope matrix; 12 adversarial dates; 4 range shapes; 100k-row index
performance at 1.3-1.8ms) but found 5 new blockers — the largest being that
round 1's concurrency fix covered only writer-vs-writer: **an ordinary
concurrent READ still destroyed writes (202/300 lost), and the critic proved
the defect is PRODUCT-WIDE and pre-existing — two plain Turn writers with no
semantic tier lose 45/300 silently on the shipped StaticPool engine.**

Round-2 resolutions (all fixed same day):

| # | Finding | Resolution |
|---|---|---|
| B1 | Reader-vs-writer write destruction; product-wide StaticPool defect | **Fixed at the ENGINE level**: NullPool per-session connections for file DBs + WAL + busy_timeout (db/engine.py, commented with the measured numbers). New regression test runs 4 writers + a hot reader loop on the real construction: zero errors, zero losses. This also fixes the pre-existing Turn-writer data loss for the whole product |
| B2 | Migration accepted ANY autoindex as "dedup enforced" (UNIQUE(fact_text) passed) | Constraint verified BY COLUMNS via PRAGMA index_list/index_info; UNIQUE(fact_text) attack table now refused (tested) |
| B3 | Migration repair wrote to the module engine's DB, not DB_PATH; failures downgraded to unread warnings | Repair binds to the DB being migrated (dedicated engine on DB_PATH); non-RuntimeError failures now RAISE (unknown schema state = do not run); divergent-path repair test added |
| B4 | Month-interval vs day-point same-text events merged; stored precision insert-order-dependent | Date PRECISION (start+end) joins the event hash — interval and point are different claims; both-orders test. Whether they describe one real occurrence is Stage 2 LLM adjudication's call, documented |
| B5 | Smoke/tests still EXPLAINed re-declared lookalikes with a comment claiming otherwise (round-1 repeat) | Both now capture the statements the store ACTUALLY EMITS (before_cursor_execute listener) and EXPLAIN those; exact index-name regex (idx_facts_current cannot be satisfied by _all); both hot paths |
| M1 | Coverage characterization false for 3/8 lines | In-batch re-affirmation + facts_overlapping(fact_type) now tested; remaining 3 uncovered lines (241/250/337) are genuinely cross-process re-raise branches |
| M2 | Racing-supersede test credited the atomic UPDATE while exercising only the pre-check | Rowcount guard now directly tested (pre-check blinded with a stale object → "superseded concurrently" refusal) |
| M3 | chain() silently returned two different incomplete views past the depth cap | Cap exhaustion now RAISES ("refusing to return a silently incomplete lineage"); tested via lowered cap |
| M4 | Running tests mutated the real dev DB (import-time init_db) | tests/conftest.py forces AGENTMEM_OS_DB_PATH to a scratch file before any import |
| M5 | Stale pre-fix artifacts presented as current | This entry; fresh artifacts below supersede the earlier G1/G2 blocks |
| M6 | Test-engine docstring claimed production parity it lacked | Split honestly: _make_engine (in-memory, functional) vs _make_production_engine (file, NullPool, WAL — all concurrency tests use it) |
| Minors | Undated facts evicted by dated-first ordering (Stage-5 landmine); trailing-garbage-after-whitespace accepted by design; undated-vs-dated same-text events = 2 rows pending Stage-2 adjudication; loose asserts | First three documented here as known behavior; asserts tightened (exact index regex, chain length == 2) |

**CURRENT artifacts (supersede all earlier pastes):**
- G1: `python3 -m pytest tests/test_semantic_facts.py` → **42 passed**, coverage
  **99%** (257 stmts, 3 missed: cross-process re-raise branches).
- G2 smoke (fresh run): `93 created, 2 re-affirmed, 6 with in-text event dates`;
  `PASS 2 (idempotency): 0 created, 95 re-affirmed`; `current facts: 93 of 93`;
  typed plan `SEARCH ... USING INDEX idx_facts_current`, untyped plan
  `SEARCH ... USING INDEX idx_facts_current_all`, both captured from the store's
  own emitted SQL, no temp sort.
- Engine change ships product-wide: `db/engine.py` NullPool + busy_timeout with
  the measured justification inline.

**G3 round 3: BLOCKED — all round-2 items verified fixed by execution (12
confirmations incl. a real 2-process race and 100k-row plans at 0.6-1.8ms), but
5 new majors.** Resolutions (same day):

| # | Finding | Resolution |
|---|---|---|
| N1 | Cross-process re-affirmation silently lost mention counts/sessions/langs (up to 2.7%, zero errors raised) — the module lock only guarded one process | Re-affirmation rewritten as an optimistic version-guarded UPDATE (mention_count is the version; lost race → fresh-snapshot re-read + retry, ≤8; caller-batch conflicts raise LOUDLY). ~~Proof: 3-OS-process test — mention_count exactly 60~~ **[SUPERSEDED by R4: this CAS design was defective; see R4/R5 records]** |
| N2 | Lock-ordering deadlock (_WRITE_LOCK vs SQLite write lock) = 31s freeze | **_WRITE_LOCK DELETED.** Every mutation is now DB-guarded (constraint / rowcount / version) — identical semantics in- and cross-process, nothing to deadlock |
| N3 | state/preference/identity same-text facts silently absorbed cross-type | fact_type joins the hash for ALL types; cross-type merging is Stage-2 adjudication's explicit call; tested |
| N4 | Concurrency regression test ran on a COPY of the engine construction — a pooling revert would not fail it | New test binds to db/engine.py ITSELF: pool-type tripwire + 3 writers + hot reader on the real module engine |
| N5 | Migration verifier accepted a PARTIAL unique index as dedup | `partial == 0` required (PRAGMA index_list col 4); attack table tested |
| N6-N8 | Coverage mischaracterized again; 3 false comments; conftest setdefault | Lines 337/241/250 now tested (supersede depth guard; fallback-refetch-None re-raise; caller-batch dedup race raises); comments corrected in both files; conftest FORCES the scratch path |
| N11 | NullPool's 4.6x throughput cost undisclosed; QueuePool equally correct and faster | **Engine switched to default QueuePool** (exclusive per-checkout connection = same isolation; 9.2k vs 1.9k ops/s measured by the critic; decision + numbers in the engine comment) |
| N9/N12 | _IS_MEMORY_DB dead code; smoke hardcoded a session temp path | Dead code removed (in-memory DBs documented unsupported); smoke uses tempfile |
| N10/N13 | Import-time-fatal migration; conftest breadth | Accepted as noted (N10 endorsed by critic; N13 checked — conftest only sets the env var) |

**G3 round 4: BLOCKED.** R3's optimistic-CAS re-affirmation read and wrote in
separate transactions — under contention its retry budget exhausted LOUDLY
(~2%/call; flagship test red 40-50% of runs). The R3 resolution table's N1 row
and its "exactly 60" proof are hereby superseded: that design was defective.
Resolution: re-affirmation REWRITTEN deterministic — relative increment
(coalesce(mention_count,1)+1) acquires the SQLite write lock, re-read + merges
happen in the SAME transaction under that lock; retry loop, version CAS, broad
OperationalError catch and both error raises REMOVED structurally. Pool ceiling
made explicit (pool_size=5/max_overflow=10, documented as a Stage-2 constraint).

**G3 round 5: BLOCKED on one new hole, R4 fix verified decisively** (critic
stalled the process between increment and re-read — two OS processes blocked
and merged exactly; mixed hammer 341/341). R5-1: expire_all() silently wiped
CALLER-staged unflushed edits in batch mode — fixed to targeted expire(fact) +
regression test staging a caller edit. R5-2/3/4 stale docs corrected (this
entry), R5-5 vanished-row guard covered, R5-6 mention_count NULL-proofed
(server_default + coalesce), R5-7 worker session config production parity.

**CURRENT artifacts: 51/51 tests, 100% line coverage on db/semantic_facts.py,
flagship cross-process test 10/10 consecutive, smoke green. **G3 round 6:
PASS-WITH-NOTES — STAGE 1 DONE.** Critic mutation-tested the fixes. Legacy-NULL
tripwire added (N2); same-fact caller-edit residual documented (N1); stage
table refreshed (N3). Stage-2 constraints on record: 15-live-session pool
ceiling; caller batch holds the SQLite write lock first-write→commit against
busy_timeout=30s.**

### Stage 0 — Implementation research (research COMPLETE 2026-08-06; founder APPROVED design freeze 2026-08-06)

Full report: `CONSOLIDATION_V2_STAGE0_RESEARCH.md`. Evidence quality: the agent read
Mem0's and Graphiti's ACTUAL source (vendored in `benchmarks/adapters/.venv-mem0/`
and `.venv-graphiti/` from Phase 2) — extraction prompts, dedup code, contradiction
handling — plus primary web sources. Headlines:

1. **Mem0 code-level confirmations:** v3 is ADD-only at the code level (the old
   ADD/UPDATE/DELETE prompt exists but is never called in `add()`); its only
   automatic dedup is exact MD5 hashing; a real production audit (issue #4573) found
   97.8% of 10,134 entries were junk, dominated by extraction over-triggering on
   system/tool noise. Their v3 prompt's temporal-grounding rule (anchor relative
   dates to conversation date, never system date) is worth copying verbatim.
2. **One genuine design fork found:** Mem0's v3 prompt explicitly rejects atomic
   facts ("Contextually Rich, Not Atomic") to preserve transition context ("switched
   from X to Y"). Resolution proposed: KEEP atomicity — Mem0 needs rich facts because
   it has no supersession chain or KG; we have both, so transitions are reconstructed
   by walking the supersession chain, with a synthesized transition sentence at READ
   time only. Flagged for founder decision at design freeze.
3. **Graphiti's contradiction pattern (read from code): LLM proposes, deterministic
   temporal check decides.** The LLM flags duplicates/contradictions; Python code
   only acts when the timestamps genuinely overlap. This is the exact shape for our
   per-fact supersession.
4. **Dedup cascade to build (no production system merges on a raw cosine threshold):**
   exact-hash → **numeric/date hard-guard** (never merge facts whose numbers/dates
   differ — the single highest-leverage rule: a false merge would silently corrupt
   the exact counts our whole thesis depends on) → embedding shortlist (generous
   0.6) → ONE batched LLM adjudication per session → deterministic temporal gate.
5. **Schema amendments:** add `mention_count` + `last_confirmed_at` (re-affirmation
   without duplicate rows — prevents the bloat Mem0/Letta users now retrofit cleanup
   for). Partial indexes `WHERE superseded_by IS NULL` (SQLite planner gotcha: query
   text must literally match the index predicate); retrofit the same onto `kg_edges`
   (has the columns, no index today).
6. **Precision fix on our own claims: the design is TRI-temporal** (occurred/
   mentioned/ingested), not "bi-temporal" — Zep's paper owns that term for a
   two-time model; we say "tri-temporal" and avoid a loose claim.
7. **Retrieval addition (near-mandatory):** LongMemEval's own time-aware query
   expansion — extract a date range from the query, filter facts on `t_occurred`
   before ranking; the paper measured it against exactly our weak categories
   (+11.3% recall on temporal).
8. **Extraction reliability:** use Ollama's schema-constrained output mode (stricter
   than Mem0's own production pipeline); dense few-shot (Mem0's own v1→v3 evolution
   proves it); semantic post-hoc validator (dates parse, digits present when source
   had them); self-consistency voting ONLY for numeric/date-bearing facts.
9. **Concurrency guard:** per-(agent_id,user_id) write serialization + DB-level
   unique constraint (Mem0's TOCTOU duplicate race, issue #6531).
10. **9 adversarial test cases** from real production failures (contradiction
    persistence, junk extraction from tool noise, race duplicates, pagination
    double-count, half-committed consolidation transactions...) — these seed the G1
    failure-path test suites for every stage.
