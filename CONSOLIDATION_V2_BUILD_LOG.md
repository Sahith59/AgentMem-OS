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
| 2 | Consolidation engine rewrite (distillation) | ✅ 97/97, 98% cov | ✅ real-model 20/4 | **R6 PASS-WITH-NOTES** (R1-R5 ✗, each fixed) | ✅ DONE 08-06 |
| 3 | KG integration (facts→entities/edges, provenance) | ✅ 180 tests, 100% line cov | ✅ 6× identical real-model | **R5 PASS-WITH-NOTES** (R1-R4 ✗, each fixed) | ✅ DONE 08-06 |
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

### Stage 2 — Consolidation engine v2 — **[SUPERSEDED: R1-era record kept for history; current state lives in the appended Stage-2 'G3 rounds record' LATER IN this file (no longer the end — Stage 3's records follow it). The G2 numbers below (42 facts/6 events/21% junk) are PRE-FIX and no longer true.]**

Built: `llm/consolidation_v2.py` — session-end distillation: schema-constrained
llama3.1 extraction (grammar-forced JSON), semantic validator (calendar dates,
session-date anchoring, vague-quantifier-where-source-had-numbers), citations,
ONE atomic batch (extraction runs BEFORE the txn — never holds the write lock).
G1: `tests/test_consolidation_v2.py` 8/8 — dead-LLM loud+zero-writes, junk
rejected (Mem0 #4573 class), batch abort atomic (#1826 class), idempotent
re-consolidation, session-date fallback, prompt anchors session date not today.
G2 (real llama3.1, ORACLE-SELECTED rollercoaster sessions — no noise, no
retrieval): 42 facts created, 0 rejected. **[CORRECTED after G3 R1 falsified
this record's first draft:]** counting rows gives 6 events, not the gold 10 —
reaching 10 still needs "three times" parsed from text (per-event count fields
are an open fix); ~21% of the 42 are assistant knowledge the validator never
catches; 2 facts cite zero turns and 19.8% of citation edges are unsupported
(structural: token-overlap over user turns only); the earlier "July point not
interval" wart was a MISDIAGNOSIS of the smoke's own printout — the store
holds the correct month interval. **Tier 2-4 semantic dedup is NOT BUILT**
(exact-hash + numeric/date/type guards only); Stage-1 F7's planned-event
handling is NOT BUILT (both now explicit, were undisclosed).

**G3 round 1: BLOCKED — 5 blockers, 8 majors** (junk/assistant-knowledge gate
absent; citation truth; 5 false docstring claims; record falsehoods above;
oracle-framing). Verified good under attack: SIGKILL-atomicity, concurrent
same-session consolidation (loud loser, zero dupes), 16-way parallel writes
480/480 at 60x margin under the Stage-1 constraints, schema-constrained output
across 13 real sessions, prompt-injection resisted. Fix pass next; stage stays
OPEN and UNCOMMITTED.

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


### Stage 2 — G3 rounds record (APPENDED 08-06 after two prior in-place
writes SILENTLY NO-OP'D on string mismatches — the R2/R3 critics were right
that this record was stale; log writes are now grep-verified)

**R1 BLOCKED (5B/8M):** support gate absent in practice, citation truth,
5 false docstring claims, record falsehoods, oracle framing. Fixed same day.
**R2 BLOCKED (4B/8M):** gate fixture-deep (tool numbers rode incidental
words), citations kept lowest ids, 4 fixes untripwired, stamp hijack, retype
deleted true facts, silent token clamp, dishonest audit row. Fixed: numbers-
in-user-evidence rule, strength-ranked disclosed citations, 10 tripwires,
header-only stamps, retype-to-state, prompt_eval_count readback, persisted
truncated_chars/rejected_count.
**R3 BLOCKED (4B/7M):** the R2 numbers rule DESTROYED true user facts (12.5%
random / 83% plan-heavy — it stripped user-stated dates from evidence before
checking; surface-form lottery: "3 times" rejected, "three times" accepted);
stamp scan role-blind (hijack from user turn 2; real fix measured FREE:
0/19,195 corpus stamps outside system-role line 0); retype-to-state collided
with the state hash and silently merged different plan dates (Stage-1 F3
revert); this record itself missing (write no-op).

**R3 resolutions (fix pass below):** numbers gate rebuilt on VALUES (raw user
evidence, comma-normalization, word-numeral mapping three→3, fact's own
t_occurred digits excluded, session-year exemption REMOVED, truthful
rejection message); stamp scan system-role-header-only with hijack test at
turn 2; planned events KEPT as events with "planned?" warning — NO retype
(hash corruption) — **FOUNDER DECISIONS OPEN: F7 planned-event storage policy
+ DESIGN §5.1 undated-fact default (built: NULL; design: session date)**;
cap-disclosure only for accepted facts; rejection REASONS persisted
(rejections_json, additive migration, narrow except, reported); provenance
adds user_turns_resolved; no-question-echo prompt rule (model-side,
disclosed); tripwires for the 3 green mutations.


**R4 BLOCKED (4B/6M) — R3 damage verified repaired (plans 0/19, randsess 4.7%
all-true, lottery gone in named forms, retype fix decisive). New: word-numeral
SUBSTRING matching manufactured numbers in 51.5% of sessions; own-t_occurred
exclusion let tool numbers ride the model's stamp; stamp gate skipped only
user-role (assistant could hijack); stale R1-era record unsuperseded.
R4 resolutions: whole-word numeral regexes BOTH sides; fact-side word numerals
checked (twelve→12 rejects); date exemption SHAPE-scoped to date literals in
fact text; system-role-only stamps (tripwired vs assistant); honest no-
evidence rejection reason; migration table-absent honesty + duplicate-column-
only swallow; unicode tokens (same-script Hindi sessions accept — CANONICAL-
ENGLISH-OVER-HINDI REMAINS OPEN, Gate E); provenance user_turns_resolved
asserted; cap-disclosure only for accepted; comma + strip-half tripwires;
notes combine.
**Fresh artifacts: 88/88 tests; smoke 20 facts / 4 coaster events (3+1+3+3
with counts+citations intact); rejections persisted.** R5 dispatched.

**R5 BLOCKED (3B/5M) — R4's four blockers verified dead; the numbers gate
blocked a FIFTH time on surface-form spellings: the date regex orphaned
digits ("February 2023" → phantom "23", 2 true facts destroyed on the same
sample where R4 had zero), glued units were invisible ("16GB" tool fact
STORED by the real model), and inline user-line timestamps licensed ~5
values per session for tool facts to ride.
R5 resolutions — the CLASS exit the critic prescribed, built once:
`_quantity_values()` — ONE numeric-mention parser used on BOTH sides:
digit runs incl. glued units/decimals (no trailing-boundary lottery),
whole-word numerals, comma + zero-pad normalization; complete-date-
expression spans (month-year form matched longest-first — no orphans);
inline stamps stripped from user evidence before licensing; per-turn
licensing + exempted-digit audit recorded in the report (numbers_audit);
true-cause reasons ordered before the numbers message. Tokenizer rebuilt
split-based so Devanagari words stay whole — the Hindi gate now functions
BOTH directions (user-grounded fact accepted, assistant knowledge rejected,
tested). Migration raise-path, user_turns-excludes-assistant, and stamp-
window tripwires added (R5-M3 all four).
**REQUIRED RECALL ARTIFACT (critic's rule: no gate change ships without
it): the critic's own 10-random-real-session probe, re-run post-fix —
43 candidates, 2 rejected (4.7%), BOTH true rejections; the two facts R5
measured as false rejections are now accepted. Smoke: 20 facts / 4 coaster
events, citations intact. 96/96 tests.**
Disclosed residuals (not hidden in claims): word-numeral map ends at
twelve (above-twelve counts unchecked, ~0.2% corpus prevalence); "one"
idiom licensing (~1.5%, accept-direction only); canonical-English-over-
Hindi support (Gate E); relative-date resolver; month-day-form counts
after month words. R6 dispatched.

**R6: PASS-WITH-NOTES — STAGE 2 G3 CLOSED (0 blockers).** The critic
independently reproduced the recall artifact (43/2, both true) AND ran a
fresh 30-session generalization probe at a different seed: **156 candidates,
4 rejected (2.6%), all four TRUE** — including the textbook vendor-number
case ("5 GB of free iCloud storage"). Licensing audit truthful at scale
(37/37 values resolve to real same-session user turns). Notes landed before
commit: tokenizer tripwire asserts on _tokens() output; false retype comment
fixed; dead regex removed; tuple docstring arity corrected.
**Disclosed limitations of record (Stage 3+ / Gate E backlog):** concurrent
same-text consolidation aborts the losing batch LOUDLY (Stage-1 design,
non-corrupting — callers retry); the gate grounds number VALUES not
PREDICATES ("38 questions" passed via user-typed "19/38"); support gate
inert for non-space-delimited scripts (CJK — matters for cross-lingual
claims, Gate E scope); word-numeral map ends at twelve; "one"-idiom
licensing affects both directions; model-side relative-date resolution
wrong ~50% (deterministic resolver queued); Tier 2-3 semantic dedup and
per-event count fields are SEPARATE build items, not part of this stage's
claims. **FOUNDER DECISIONS still open: F7 planned events; §5.1 undated
default.**
---

## 2026-08-06 — FOUNDER DECISIONS RESOLVED (both) + Stage 3 GO

Founder accepted both recommendations verbatim:

**F7 — planned/future events: DEDICATED MARKER (schema field), built in
Stage 3.** A future-dated event (t_occurred > session date at mention
time) is stored as a dated event with `planned=1`. Detection is
deterministic and single-sourced (one helper used by both the validator
warning and the storage flag — no duplicated comparison logic). Merge
rule on re-affirmation: `planned = planned AND incoming_planned` — any
affirmation made at a time when the event date is no longer in the
future turns the row into an occurrence claim (deterministic; occurrence
JUDGMENT beyond this stays Stage 4 supersession territory). Boundary
disclosed: only DATED facts can be flagged — "user plans to attend X"
with no date extracts as an undated state and carries no marker.

**§5.1 — undated facts: KEEP NULL (design doc's session-date default is
REVOKED).** t_occurred stays NULL when no event date was stated —
honest "we don't know when this happened"; retrieval anchors on
t_mentioned. What "when" means in public claims: t_occurred is only ever
a USER-STATED time, never an inferred one.

Stage 3 (KG integration) GO given same message, same gates (G1/G2/G3),
independent commit.

---

> RESTORED 2026-08-06 (content unchanged): the two blocks below-marked
> sections — this design section and the later 'Post-R1 smoke rerun'
> note — were originally appended to a STRAY file outside the repo
> (the shell's working directory silently reset mid-session and the
> relative-path append CREATED /Volumes/.../AgentMem-OS/CONSOLIDATION_
> V2_BUILD_LOG.md one level up; the grep-verify passed because it
> checked the same wrong file). Caught by G3 R2's 'points at nothing'
> finding. New standing rule: every log write and its grep-verify use
> the ABSOLUTE repo path.

## STAGE 3 — KG integration (fact→entity linking + event_status) — started 2026-08-06

### Research inputs (tech-researcher pass, sources verified in report)
- **Join table over JSON is settled**: SQLite's author, on the official
  forum, states array-membership indexing is structurally impossible
  ("indexes are one-to-one... you are asking for many-to-one") — a
  json_each() lookup is a full-scan at any scale. `SemanticFact.entities`
  (JSON) demotes to a display/inspection cache; the query path is a new
  `semantic_fact_entities` join table indexed both directions.
- **Graphiti/Zep = fact-as-EDGE** (fact text lives on the RELATES_TO edge,
  provenance = episode UUIDs only, NO mention spans — verified absence,
  read in full from vendored source). We are fact-as-ROW + join links —
  different representation, keep claims distinct. Graphiti/Zep count as
  ONE data point (same engine) in any write-up.
- **Mem0's OSS graph module NO LONGER EXISTS** (deleted in PR #4805; open
  issue #6591 documents the drift, 11 days old). Any "Mem0 graph variant"
  comparison in our docs is stale — re-verify at publish time. Noted for
  COMPETITIVE_ANALYSIS.md.
- **Graphiti's real bug classes to test against ours**: orphaned entities
  invisible to cleanup (#1083), concurrent ingest racing on shared state
  (#1331), dedup silently not firing (#875), batch dedup schema-fragile
  (#879).
- **Planned-event prior art**: NONE in Graphiti/Zep (verified absence —
  their valid_at/invalid_at is truth-validity, not prospectiveness).
  RFC 5545 (TENTATIVE|CONFIRMED|CANCELLED) and schema.org eventStatus
  both independently chose a small ENUM over a boolean. Generative Agents
  (Park et al. 2023) stores Plans as a distinct memory kind. Claim
  wording when this ships: "no comparable mechanism found in the
  strongest competitor examined" — NOT "following an established pattern".

### Design decisions of record
1. **`event_status` enum, not `planned` boolean** (upgrade of the
   founder-approved "dedicated marker" — same semantics, RFC-5545/
   schema.org precedent, spares a migration when Stage 4 needs
   'cancelled'). Values: 'occurred' (default for every event — extractor
   contract says events already happened), 'planned' (deterministic:
   occurrence start > session date at mention time, same single-sourced
   helper feeding the validator warning), NULL for non-events. NOT in
   the dedup hash. Re-affirmation merge: planned→occurred when any
   affirmation arrives post-date; never occurred→planned. Migration
   backfills existing dated events deterministically from stored columns
   (t_occurred > t_mentioned → planned). Boundary disclosed: undated
   plans extract as states and carry no status.
2. **Facts link to the SURFACE-form node, NOT the alias-canonical node**
   — deliberate divergence from Graphiti's uuid_map pattern (research
   rec 2). Graphiti canonical-links because its dedup MERGES nodes; our
   ALIAS_OF design is non-destructive precisely because τ=0.90 has a
   MEASURED false positive (Chennai/China 0.9010). Canonical-linking
   would hard-wire that error class into fact provenance; surface-linking
   keeps a false alias as inspectable retrieval noise (confidence on the
   edge), never silent misattribution. Cross-lingual unity comes from
   ALIAS_OF traversal in the read API.
3. **Provenance ambition capped at turn granularity** (source_turn_ids;
   even Graphiti stores episode-level only). Join rows carry
   surface_text + linked_via ('ner'|'alias') + confidence — no mention
   offsets, no role/relation column (deliberate: role extraction needs
   parsing we don't do; an always-NULL column is dead schema; research
   rec 10 declined, disclosed here).
4. **Linker must CREATE nodes, not just link** — verified: the KG is fed
   only via ConversationStore.save_turn background ingestion; benchmark
   pipelines write Turn rows directly and bypass it. In the real Gate-C
   pipeline the KG may be EMPTY; facts are the feeder. Node creation from
   facts does NOT bump mention_count on existing nodes (turn-mention
   counts must not be inflated by distilled restatements).
5. **kg_nodes gets UNIQUE(coalesce(agent_id,''), entity_text)** — its
   read-then-write upsert is the same race class Stage 1 fixed (and
   Graphiti #1331's). Dev DB verified: 10,911 nodes, 0 dup groups, index
   creates clean here; migration still carries a dedup-merge path
   (re-point edges, sum mention_counts, drop self-loops, delete dups,
   retry) for arbitrary DBs, exercised by a synthetic-dup test.
6. **Failure policy**: fact storage NEVER fails because linking failed.
   Links are recoverable metadata — per-fact savepoint isolation inside
   the caller batch, failures counted in the report + ConsolidationLog
   (new entities_linked column), and a `link_missing()` sweep makes the
   recoverability claim TRUE (crash between facts-commit and link,
   pre-Stage-3 backfill, Graphiti-#1083 orphan class).
   **[CORRECTED R3-B1/R4: the DEFAULT sweep recovers UNLINKED FACTS
   only; skipped surfaces on partially-linked facts need
   link_missing(rescan_all=True). And per R4-M5 the sweep now has a
   real product caller: consolidate_session auto-runs a bounded
   default-depth drain after a link_failure commit.]**

### Build surface
models.py (SemanticFact.event_status, SemanticFactEntity, kg_nodes unique
index, ConsolidationLog.entities_linked) · engine.py (_migrate_stage3:
columns + backfill + table verify + index-with-merge-path) ·
db/fact_entities.py (FactEntityLinker: link_fact, facts_for_entity,
link_missing) · knowledge_graph.py (NER extraction refactored to a shared
module-level function; class delegates — test_temporal_kg.py must stay
green untouched) · db/semantic_facts.py (event_status param, validation,
merge rule) · llm/consolidation_v2.py (single-sourced planned helper,
pre-txn NER, in-txn linking, report/log fields).
Out of scope (disclosed): retrieval assembly wiring (Stage 5), occurrence
JUDGMENT / cancellation (Stage 4), role columns, mention spans, CJK NER.

### Stage 3 — G1 + G2 record (2026-08-06)

**G1: 146/146 tests across 4 files** (test_fact_entities.py NEW: 33;
consolidation_v2: 48; semantic_facts: 60; temporal_kg: 5 untouched by the
NER refactor). **Coverage 100% on db/fact_entities.py AND
db/semantic_facts.py.** Failure paths pinned: real 2-OS-process node-
creation race (unique index authority: 1 node, both links), store-owned
retry-ONCE-then-loud, non-race IntegrityError never retried, synthetic-
duplicate migration merge (edges re-pointed, self-loops dropped,
CO_OCCURS weights SUMMED — the in-memory loader overwrites duplicate
pairs, so leaving two rows would silently drop weight), unrepairable
link table refused, event_status merge matrix (planned→occurred upgrade,
never downgrade, NULL backfill-on-touch), Devanagari surface
augmentation (caught mid-build: en_core_web_sm is BLIND to Devanagari —
without script-token augmentation a Hindi fact got ZERO surfaces and the
alias gate was never consulted; the fake resolver was made
DISCRIMINATING after it initially matched every Hindi word and hid
exactly this class).

**G2 (real llama3.1 + real spaCy + real multilingual-e5-small,
benchmarks/consolidation_v2_stage3_smoke.py):**
- A. Rollercoaster gold sessions, full pipeline: 20 facts → 25 entity
  links, 18 nodes, ConsolidationLog rows agree (25=25), link_failure
  None. Disneyland timeline artifact: dated occurred events with intact
  citations incl. "rode Space Mountain: Ghost Galaxy three times"
  [2023/09/24], and "The user is planning a trip to Disneyland" typed
  STATE undated → NO event marker (the disclosed F7 boundary, observed
  live).
- B. **MEASURED CONFIRMATION of the disclosed Stage-2 Gate-E residual:**
  Hindi session → llama3.1 emitted canonical-ENGLISH facts (per design)
  → support gate found zero stemmed overlap with Devanagari user turns →
  both TRUE facts rejected as unsupported. Cross-lingual consolidation
  needs translation-aware support evidence (Gate E work item, NOT Stage
  3 scope — recorded, not papered over).
- C. Gate-E write shape with the REAL resolver: English anchor session
  consolidated first (Google node via real pipeline), then Hindi fact →
  **गूगल admitted at cosine 0.9506 (τ=0.90), FOREIGN node + ALIAS_OF
  edge + 'alias' link; उपयोगकर्ता/में/काम/करता/है। all correctly kept OUT
  at τ. facts_for_entity('गूगल') == facts_for_entity('Google') == all 3
  facts across both languages — the cross-lingual unity claim,
  demonstrated.** (First run also proved the no-anchor NEGATIVE path:
  with no English node present, everything Indic was refused — junk
  cannot bootstrap junk.)

G3 critic round 1 dispatched.

### Stage 3 — G3 ROUND 1: BLOCK (3 blockers, 5 majors, 13 minors) → all fixed

The critic independently reproduced every G1/G2 claim that was true
(146/146, both 100% line coverages, the smoke to the digit, the
caller-batch lock precondition, backfill↔helper agreement 10/10,
mutation sweep 14/16) and then broke the rest. Resolutions:

**B1 — ~87s of DB-WIDE WRITE LOCK during in-batch alias planning
(measured: competing writer DIED on busy_timeout).** The first Indic
surface loaded sentence-transformers inside the consolidation batch.
Fixed as a CONTRACT, not a patch: planning (model load + embeddings) is
now a separate read-only step — engine calls plan_surfaces() BEFORE the
transaction and passes plan= into link_fact; caller-owned sessions
REFUSE to plan in-session (loud ValueError), and a tripwire test
asserts apply-with-plan never consults the resolver. Known disclosed
consequence: same-batch anchors are invisible to pre-txn plans; such
surfaces land in skipped and the sweep recovers them. **[CORRECTED
R3-B1: recovery of SKIPPED surfaces requires the deep sweep,
link_missing(rescan_all=True) — the default zero-link sweep cannot see
a partially-linked fact.]**

**B2 — the CO_OCCURS dedup-merge was INERT on production rows** (it
matched relation_type IS NULL; the ORM default writes the STRING
'CO_OCCURS' — 34,905/34,905 real rows; the test fixture omitted the
default and rubber-stamped it). Both shapes now merge as one family;
the fixture is production-shaped with one NULL row mixed in. **Digging
here exposed a REAL pre-existing product bug: the turn path's CO_OCCURS
lookup had the same NULL-vs-string mismatch, so it CREATED A NEW EDGE
PER CO-OCCURRENCE instead of incrementing weight — 1,979 duplicate
pairs accumulated in the dev DB, and the loader's add_edge() keeps only
the last row's weight.** Lookup fixed (both shapes), tripwire test
added (two ingests → one edge, weight 2), and the migration now runs a
global, idempotent duplicate-pair repair (weights SUMMED into the
lowest-id row — no information lost). NOTE FOR FOUNDER: next product
start against the dev DB will repair those 1,979 pairs.

**B3 — link_missing starved permanently** (zero-surface facts — 22% of
the real G2 sample — are forever-candidates; LIMIT without a cursor
re-swept them every call and never reached facts beyond the limit; the
documented drain loop never terminated). Now cursor-paged (after_id /
next_after_id); the drain visits every unlinked fact exactly once
**[CORRECTED R3-M4: once PER DRAIN — rowids may recycle without
AUTOINCREMENT; and per R3-B1 this default drain covers UNLINKED facts
only, skipped surfaces need rescan_all=True]**;
starvation repro pinned as a test with a termination guard.

**M1** index verified by NAME only → now verified by its sqlite_master
SQL (a scope-blind impostor with the right name gets dropped and
rebuilt; test pins it). **M2** "linking suspended" was indistinguishable
from "nothing to link" in the persisted audit → ConsolidationLog gains
link_failure TEXT (persisted, migrated, tested). **M3** one-hop
ALIAS_OF traversal returned SUBSETS through variant chains
(चेन्नई—चेन्नई।—Chennai measured) → full closure, depth-capped; test
walks the chain from all three surfaces. **M4** case-sensitive reads
went blind over the KG's 164 real case-variant groups → read-side seeds
are case-insensitive (node identity stays case-sensitive, turn-path
parity, disclosed). **M5 — REACHABILITY OF 'planned', disclosed
plainly: the extraction prompt types ALL plans as states (its own
example carries a date), so the marker fires only when the model
disobeys — 0/23 facts in the G2 sample carried 'planned'. The marker is
a correctness safety net, not the primary plan representation. Whether
dated plans should extract as planned EVENTS is a prompt-policy change
that would alter measured Gate-B behavior — QUEUED AS A FOUNDER
DECISION, not slipped in mid-arc.**

Minors landed same round: danda/double-danda excluded from Indic script
tokens (was costing 22-43% of τ margin and manufacturing M3's variant
nodes — entity_aliases regex, shared with the turn path, tested); false
parity comment corrected (linker anchor rule is deliberately STRICTER
than the turn path's); nullslast tripwire (deleting it now fails a
test); linking-suspension tripwire (flipping the guard now fails a
test); dead self._nlp removed; smoke's log-vs-join check is now an
ASSERT across three independent sources; coverage claim restated
honestly: 100% LINE / 99% branch on both modules.

Post-fix: 156 tests green across the 4 scoped files. Smoke re-run
against the fixed engine below. R2 dispatched.

**Post-R1 smoke rerun (fixed engine, plans pre-txn): identical artifacts
to the digit** — 20 facts / 25 links / 18 nodes, three-way ASSERT
(report == persisted log == join rows) passing, गूगल↔Google cosine
0.9506, both-surface reads identical, danda no longer part of any
script token (skip list shows है, not है।), Part-B Gate-E residual
unchanged (disclosed).

### Stage 3 — G3 ROUND 2: BLOCK (1 blocker, 5 majors, 14 minors) → all fixed

R2 verified R1's three blockers CLOSED with measurement (B1: resolver
instrumented, all 6 engine-path touches leave a competing writer FREE,
control probe in-batch correctly BLOCKED; B2: repair run against a COPY
of the dev DB — 34,905→32,194 rows, total weight exactly preserved,
**2,711 units of loader-visible weight recovered**, idempotent; B3
cursor terminates; fixes are revert-detected, not rubber stamps). Then
it broke the new code. Resolutions:

**R2-B1 — MY M4 FIX WAS THE REGRESSION: SQLite lower() folds ASCII
ONLY** — the lower()==lower() seed turned byte-identical non-ASCII
queries ('Übermensch', 11 real KG entity_texts) into silent empties.
Fixed: exact-match arm restored alongside the ASCII case-fold arm
(or_), ASCII-only folding DISCLOSED in the docstring, regression pinned
with a non-ASCII read test. Lesson logged: a fix is a change like any
other — R2's "mutate the fixes" pass exists because R1 fixes ship R2
bugs.

**R2-M1** plan→apply validated nothing: a stale plan whose alias anchor
vanished ADMITTED AN UNGATED INDIC NODE (falsifying the gate
invariant), and a plan for agent A applied under agent Z wrote into the
wrong scope. Fixed: plan_surfaces returns a scope-stamped dict;
link_fact validates shape+scope loudly (malformed via values can no
longer reach linked_via); apply resolves the ANCHOR FIRST and skips the
surface if it is gone ("alias anchor not found at apply") — junk
resolver output now lands in skipped instead of creating a node.
**R2-M2** index verification was a substring test (an extra trailing
column or a WHERE clause still read "verified") → EXACT normalized-DDL
equality; both lookalikes pinned as tests. **R2-M3** node-merge
re-pointing INVERTS src<tgt ordering (keeper=min(id)) so the per-group
ordered pair-dedup missed exactly the pairs the merge created (measured
5.0-of-7.0 loss) → global pass canonicalizes undirected-pair ordering
FIRST (SQLite UPDATE reads pre-update values), unordered grouping,
reversed historical rows merge too; inversion fixture pinned — the
THIRD "fixture cannot produce the failure" in this arc is recorded as a
standing test-design lesson. **R2-M4** POST /demo/reset 500'd on the
new FK once a global-scope fact was linked → link rows deleted before
nodes (facts themselves are not demo state and stay). **R2-M5** the B1
guard was one-sided (plan_surfaces accepted db=) → parameter REMOVED;
signature pinned by test.

Minors landed: closure truncation now WARNS (partial reads never
silent); Indic DIGIT runs dropped from script tokens; migration
connection timeout matched to the app's 30s; pair merge keeps
max(last_updated) ("no information lost" now includes recency);
node-merge re-points semantic_fact_entities too (raw conn runs FKs
OFF — orphaned links pinned by test, collision rows dropped);
"exactly once" softened to per-drain (rowid reuse); loader collapsing
CO_OCCURS+ALIAS_OF on the same node pair (measured 12.0→1.0) recorded
as a KNOWN ISSUE for the loader backlog (nx.Graph holds one edge per
pair — MultiGraph surgery, not this stage).

**PROCESS INCIDENT (logged as its own lesson): two build-log/notes
appends LANDED IN A STRAY FILE one directory above the repo** — the
shell cwd silently reset mid-session and relative-path appends CREATED
/Volumes/.../AgentMem-OS/CONSOLIDATION_V2_BUILD_LOG.md; the grep-verify
passed because it checked the same wrong file. Caught by R2's
"points at nothing" finding. Content restored in place with a marked
note; strays deleted. STANDING RULE: every log write AND its
grep-verify use the ABSOLUTE repo path (this entry does).

Post-fix: 167 tests green across the 4 scoped files; db/fact_entities.py
100% line / 99% branch (one partial branch — the duplicate-token
skip arm in surface dedup, not a loop exit [R4-m2 correction]). Smoke
rerun below. R3
dispatched.

**Post-R2 smoke rerun (plan-dict engine path): artifacts identical to
the digit again** — 20 facts / 25 links / 18 nodes, three-way assert
passing, गूगल↔Google 0.9506, both-surface unity, Part-B Gate-E residual
unchanged.

### Stage 3 — G3 ROUND 3: BLOCK (1 blocker, 5 majors, 16 minors) → all fixed/waived-with-record

R3 confirmed every R2 fix held under mutation (11/13 caught; the two
"survivors" were the untripwired minors it then flagged), reproduced the
smoke a third time to the digit, and verified the Mem0 claims against
the GitHub API. **R3 also disclosed its own incident: importing the
engine module in-process ran init_db and the duplicate-edge repair
EXECUTED AGAINST THE LIVE DEV DB** — byte-compared to the R2-vetted
copy: identical outcome, sum(weight) exactly conserved. The predicted
repair, confirmed on real data, by accident. Two enforced checks logged
in the lessons file (reviewers open production DBs read-only via URI;
destructive import-time migrations must write a pre-state first).

**R3-B1 — the recovery sweep could NOT recover skipped surfaces, and
four record sites said it could.** The zero-link sweep never revisits a
fact that linked ANY surface — a two-entity Indic fact whose anchor
arrived late was orphaned from that entity FOREVER (reproduced).
Fixed: link_missing gains rescan_all=True (re-plans every fact in
scope against current graph state; idempotent by construction) and ALL
FOUR sites now state the two depths precisely: default sweep = unlinked
facts only; skipped-surface recovery = deep sweep. Repro pinned: the
default sweep's blindness is ASSERTED as the documented bound, the deep
sweep recovers the गूगल-and-Microsoft fact.

**R3-M1** "logged as its own lesson" pointed at nothing → the
stray-file lesson now actually written to docs/memory/lessons/
process.md (the claim's own class — a reference is not a record).
**R3-M2** RUNNING_NOTES still told the founder the repair was pending
(it had already run, via R3's import) → corrected; AND the repair now
snapshots every row it deletes into kg_edges_dedup_backup BEFORE
deleting — a destructive import-time migration without pre-state is
unrecoverable if a future bug miscounts. **R3-M3** node-merge inverted
ALIAS_OF pairs and the ordered exists-check then DUPLICATED alias edges
(measured 1→2); the "typed edges left as-is" rationale was false for
ALIAS_OF → canonicalization now covers both undirected families,
exact-duplicate ALIAS_OF pairs collapse keeping best confidence,
rationale corrected (directional typed relations only), inversion test
pinned with backup-row assertion. **R3-M4** module docstring still said
"exactly once" → per-drain everywhere. **R3-M5** the demo-reset fix
was the only R2 fix without a regression test → REAL-endpoint test
added (TestClient POST /demo/reset with a linked global fact, FKs ON:
200, nodes+links purged, fact survives).

Minors landed: retry no longer double-appends apply-side skips (copy;
test); _validate_plan rejects empty surfaces, anchor-less alias
entries, malformed skipped entries (tests); Indic-digit filter
tripwired; dead `or True` assertion replaced with the real ASCII-only
disclosure pin; stale plan=(plan,skipped) docstring corrected;
facts_for_entity IN() bounded at 30k with loud truncation (patchable,
tested); smoke's legacy Query.get modernized; entities JSON display
cache semantics documented (NER mentions independent of link success);
demo-reset regrowth-via-sweep documented as deliberate; Stage-2
"END of file" pointer corrected. Waived WITH RECORD: engine's
poisoned-batch-raises-at-commit path untested (policy documented;
deterministic construction not worth the harness); kg_nodes' unique
index is a dedup index not a read index (reads measured 0.21-0.25ms at
10.9k nodes — a read index is future work at scale); loader collapsing
CO_OCCURS+ALIAS_OF on the same node pair (measured 12.0→1.0) is a
PRE-EXISTING loader limitation (nx.Graph = one edge per pair) recorded
in the known-issues memory — MultiGraph surgery is not this stage.

Post-fix: 174 tests green across the 4 scoped files; db/fact_entities.py
100% line / 99% branch (one partial branch — the duplicate-token skip
arm [R4-m2 correction]). Smoke rerun below. R4 dispatched.

**Post-R3 smoke rerun: fourth identical reproduction** — 20 facts / 25
links / 18 nodes, three-way assert, गूगल↔Google 0.9506, both-surface
unity intact.

### Stage 3 — G3 ROUND 4: BLOCK (0 blockers, 6 majors, 9 minors) → all fixed

R4 confirmed the SYSTEM sound: every R3 fix held under mutation (13/16
— the 3 survivors were themselves R4 findings), fifth identical smoke
reproduction (critic's own, env-pinned), suite/coverage claims exact,
RUNNING_NOTES repair claim verified against the live DB READ-ONLY
(sum(weight) conserved), Mem0 claims re-verified at the GitHub API. The
six majors were record-accuracy, untripwired assertions, one latent
fragility, and one missing caller:

**R4-M1** the historical R1-B1/R1-B3 sections still stated the false
sweep-recovery contract INSIDE the document that recorded its fix →
bracketed CORRECTED annotations applied at all three sites (the
SUPERSEDED convention this file already uses — history kept, never
silently rewritten). **R4-M2** the retry-copy test could not detect
reverting the copy (its mock raised BEFORE the real body — mutation-
proven) → rebuilt: the first attempt now runs the REAL apply, mutates
the shared list if uncopied, THEN loses the race. **R4-M3** the
CO_OCCURS backup INSERT — the branch that deleted 2,711 rows on the
real dev DB — had no tripwire → backup-content assertion added to the
reversed-rows test. **R4-M4 (latent import-killer, reproduced)** the
backup table's columns froze at creation; a future ALTER TABLE
kg_edges ADD COLUMN would make INSERT..SELECT * raise inside the
import-time migration — package unimportable on exactly the DBs with
duplicates → schema-drift guard: stale backups are preserved under
versioned names and recreated fresh; drift repro pinned. **R4-M5 (the
sharpest finding: the compensating control had NO CALLER)** —
link_missing existed only for tests; "recovery" was a human writing
the drain loop in a REPL → ConsolidationV2.recover_links() is now the
product-side drain (bounded, loud on runaway), AUTO-INVOKED after any
link_failure commit; report carries link_recovery; engine tests assert
the auto-drain actually relinks. **R4-M6** the founder-memory record
claimed the dev-DB pre-repair rows were in kg_edges_dedup_backup —
FALSE (the repair predates the backup mechanism; the only pre-state is
the R3 critic's purgeable scratchpad copy) → memory corrected with the
caveat stated plainly.

Minors landed: smoke pins AGENTMEM_OS_DB_PATH before any import (a
rerun can never migrate the live DB as a side effect); the bound test
now proves TRUNCATION (three facts vs bound two — dropping .limit()
turns it red); demo-reset test assertions scoped to its own artifacts
(global counts on a shared scratch DB fail spuriously); _validate_plan
surface rule matches _dedup_surfaces (stripped, len>=2);
facts_for_entity(None/empty) raises loudly; closure node-set bounded
(patchable, loud, tested); "commit below raises" corrected to
next-flush-or-commit; coverage-partial wording corrected (duplicate-
token skip arm, not loop exit); RUNNING_NOTES status corrected via
append. Waiver 3 (loader same-pair collapse) upgraded with the honesty
line R4 demanded: Stage 3 makes that collision the COMMON case for
code-switched pairs — recorded in the known-issues memory as a
must-fix-before-cross-lingual-serialization-claims.

Post-fix: **179 tests green** across the 4 scoped files;
db/fact_entities.py 100% line / 99% branch (duplicate-token skip arm).
R5 dispatched as the final verify round.

### Stage 3 — G3 ROUND 5: PASS-WITH-NOTES → STAGE 3 CLOSED (2026-08-06)

R5 (final verify round): 179/179 reproduced, coverage exact, **sixth
identical G2 smoke** (critic's own, env-pinned), live DB verified
untouched, 19/23 mutations caught INCLUDING both R4 mandatories
(retry-copy revert and backup-INSERT removal now turn tests red), all
three CORRECTED annotations verified in place with a repo-wide grep
finding no remaining false sweep claim, and memory #11's "2,711 units
lost" independently re-derived from the pre-repair copy (35,051.0 −
32,340.0). The one mandatory record correction (the "commit below
raises" comment fix that MISSED its site — a false "landed" line, the
R4-M1 class in its 5th appearance) is now actually landed at both
sites. R5's recover_links findings landed with it: the auto-recovery
is BEST-EFFORT (a recovery-side fault is recorded in the report, never
destroys it — facts and log are already committed), recovered links
are WRITTEN BACK into the persisted log row (audit coherence: 12 DB
links against a persisted 0 was incoherent), the scope-wide drain cost
is disclosed in the docstring, the runaway guard and complete-flag are
tripwired, the smoke's env pin is FORCED not setdefault, and the
_validate_plan surface rule has revert tests. Final: **180 tests
green.** (R5's sixth smoke ran pre-final-notes; the final changes are
recovery-path and comments only — the smoke path, where link_failure
is None, is byte-identical.)

**STAGE 3 HONEST CLAIMS OF RECORD (critic-approved wording):**
CAN claim: facts link to KG nodes through an indexed join table with
provenance (surface-form linking, never canonical); cross-lingual
entity unity demonstrated on real models SIX times identically
(गूगल↔Google at cosine 0.9506, both-surface reads equal, 5 common
Hindi words kept out at τ in the same call); planning separated from
applying so no model load ever runs under a write lock (measured:
competing writer FREE on all engine-path resolver touches); a real
production KG bug (duplicate CO_OCCURS edges) found, fixed, and
repaired with total weight conserved exactly; event_status (F7) with
deterministic detection and planned→occurred-only merges; recovery
auto-invoked after a suspended-linking commit, best-effort, audited.
MANDATORY DISCLOSURES alongside any of the above: Gate-E
canonical-English residual (a Hindi session still consolidates 0 facts
— the cross-lingual write proof is the store-level shape); the
'planned' marker is prompt-unreachable in practice (0/23 — prompt
types plans as states; extraction-policy change is a QUEUED FOUNDER
DECISION); read-side case folding is ASCII-only; two recovery depths
(the default sweep cannot see partially-linked facts; skipped surfaces
need rescan_all=True); the in-memory loader collapses CO_OCCURS +
ALIAS_OF on the same node pair and Stage 3 makes that the COMMON case
for code-switched pairs (known issue, must fix before cross-lingual
subgraph-serialization claims); NOTHING is wired into product
retrieval yet — ConsolidationV2 has no caller outside benchmarks and
tests (Stage 5's job).

Stage table: Stage 3 ✅ DONE (G1 ✅ 180 tests / 100% line coverage on
both core modules · G2 ✅ six identical real-model reproductions · G3 ✅
R5 PASS-WITH-NOTES after R1-R4 each BLOCKED and fixed).
