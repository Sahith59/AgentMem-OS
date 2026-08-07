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
| 4 | Per-fact supersession | ✅ 240 tests, 98% judge cov | ✅ real-corpus backfill supersession, critic-reproduced 4× | **R8 PASS-WITH-NOTES** (R1-R7 ✗, each fixed) | ✅ DONE 08-07 |
| 5 | Facts-first retrieval wiring + $0 diagnostics | ✅ 41 tests (40+1 opt-in skip), 255 regression | ✅ real-corpus facts-first + 2 real pre-existing bugs found (Redis) | **R6 PASS-WITH-NOTES** (R1-R5 ✗, each fixed; 3 incarnations of the truncation bug killed) | ✅ DONE 08-07 |
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

---

## STAGE 4 — Per-fact supersession JUDGMENT — started 2026-08-06

### Research inputs (tech-researcher pass; sources verified from vendored
### source + live GitHub; full report in the session record)
- **Graphiti (v0.29.3, read from vendored source): the reference shape.**
  One LLM call per new edge against a ~10-20 shortlist (hybrid search),
  small/cheap model by design (gpt-4.1-nano default), LLM only PROPOSES
  — a deterministic layer decides DIRECTION by domain time (valid_at),
  both directions checked, ingestion order explicitly not trusted
  (their worked case: a 2005 marriage ingested after the 2024 divorce
  must not supersede it).
- **The schema-ordering finding (graphiti#1666): 46.7% → 93.3% on the
  SAME small model, same eval, by putting a free-text reasoning field
  BEFORE the decision arrays.** Premature field-commitment is a general
  structured-output failure mode — directly transferable to our Ollama
  format-grammar calls. Reasoning-first is NON-NEGOTIABLE here.
- **Mem0: the cautionary tale on both sides.** v2's LLM
  ADD/UPDATE/DELETE decision once spuriously DELETED a memory on a
  0.99999-similar restatement (#1674 — the dangerous class, real); v3
  removed the mechanism entirely and maintainers CLOSED the
  restore request as not-planned (#4896) — stale duplicates accepted
  as shippable (the mild class). Their update prompt still sits in the
  repo, disconnected from the write path (verified from main).
- **Small-model defensibility: llama3.1-8B alone is NOT defensible**
  (7-8B zero-shot contradiction anchors 33.6-63.8%; no published eval
  of this exact model/task shape; our own measured ~50% relative-date
  error on the same model). Defensible only with: tight deterministic
  shortlist + reasoning-first schema + a deterministic co-signal gate
  + domain-time direction check.
- **Temporal gap found in OUR schema:** we store superseded_at (decision
  time) but NO domain-validity-end for states — Graphiti's invalid_at
  has no equivalent on SemanticFact (the KG's valid_until proves the
  pattern in-repo for 3 relations). Stage 4 adds it.
- **Planned-event cancellation: NO prior art anywhere** (schema.org/RFC
  5545 are publisher-DECLARED fields, never model-inferred). We would
  be first to mechanize — built only as one attribute the general
  judgment may update, never a bespoke detector; reachability disclosed
  as gated on the parked plans-as-events prompt decision.

### Design decisions of record
1. **Shape: batched-shortlist, single-new-fact judgment** (Graphiti's
   pattern). Shortlist = live facts, same scope_key, same fact_type,
   sharing ≥1 entity node via semantic_fact_entities (Stage 3's join
   table earning its keep), cap 12 ranked by most-recent t_mentioned,
   cap disclosed per judgment. No candidates → NO LLM call.
   **[CORRECTED (G2 + R2): THREE reserved pools — entity peers (cap
   12) + lexical peers (TF-IDF≥0.25 over the newest 300 same-type
   facts) + planned events (cap 4; entity-shared, or topically
   filtered for entity-less facts — R2-B2 showed the unrestricted
   fallback handed every entity-less fact 4 arbitrary plans).]**
2. **Type scope, conservative by construction: states and preferences
   ONLY.** identity excluded (highest cost-of-error — founder-visible
   scope call, revisit after measurement); events excluded EXCEPT
   planned→cancelled as an attribute update the judgment may propose.
3. **The LLM only PROPOSES; deterministic gates DECIDE:**
   (a) output ids must be in the shortlist (anything else dropped,
   logged); (b) a deterministic co-signal must agree — the
   conflict_detector contradiction-pair vocabulary OR its shipped
   lexical-similarity signal (same-attribute evidence) — an LLM-only
   verdict never acts; (c) DIRECTION is computed from DOMAIN time
   (t_occurred start, fallback t_mentioned), both directions
   considered: if the "old" fact is domain-later, the proposal applies
   REVERSED (the out-of-order-session case); exact ties SKIP,
   ambiguous is conservative; (d) supersede() remains the only writable
   action — no delete path exists for the model to reach.
   **[CORRECTED (R1+R2 — both (b) and (c) as written shipped broken
   and were rebuilt): the gates of record are now (1) shortlist
   membership, (2) TYPE guard (only same-type facts supersede; events
   are cancellation candidates only), (3) INDEPENDENT co-signal —
   node-guarded polarity flip, or the digit-aware metric-update signal
   (masked-identical texts, aligned numbers, EXACTLY ONE value
   differs), or bare cosine for entity-pool candidates only, plus a
   cue-vocabulary + shared-subject gate on cancellations, (4)
   interval-aware SAME-AXIS direction (overlap/touch = skip), (5) one
   transaction for actions + audit row.]**
4. **Reasoning-first output schema** (grammar-forced):
   {"reasoning": …, "superseded_ids": […], "cancelled_ids": […]} —
   the 46.7→93.3 finding applied verbatim.
5. **Failure bias is structural: dangerous class blocked, mild class
   degrades to today's shipped behavior.** A missed supersession leaves
   a stale duplicate (Mem0 v3's accepted status quo, and Stage-1
   history preserves everything anyway); a wrong supersession is
   blocked by gates (b)+(c) and is REVERSIBLE even if it lands
   (invalidate-don't-delete — superseded_by can be audited and undone
   by an operator; nothing is destroyed).
6. **Tri-temporal completed: SemanticFact gains t_invalid** (domain
   time a state STOPPED being true; nullable sortable string), set at
   supersession from the superseding fact's domain time — the KG's
   valid_until pattern extended to facts; additive migration;
   distinct from superseded_at (OUR decision time) and disclosed as
   such. facts_as_of stays mention-axis (a different question,
   documented); time-aware reads over t_invalid are Stage 5.
7. **Every judgment is PERSISTED** — new supersession_judgments table:
   fact judged, shortlist (with cap disclosure), co-signal results, raw
   model output, actions applied, proposals dropped and WHY, model
   name. Silent under-judgment ships undetected otherwise
   (graphiti#1666's lesson); the inspectable-history story requires
   showing why a supersession did or did not happen.
8. **Wiring mirrors Stage 3's hard-won shape:** judgment runs AFTER the
   consolidation batch commits (LLM compute NEVER under a write lock —
   R1-B1's lesson is a standing law), best-effort (a judge fault is
   recorded in the report, never destroys it), with a cursor-paged
   recovery sweep (facts with no judgment row → judge), auto-invoked
   on judge failure, product-callable (R4-M5's lesson: a compensating
   control without a caller is a docstring).
   **[CORRECTED (R1-B4): the sweep is product-callable
   (recover_judgments) and deliberately NOT auto-invoked — a
   scope-wide drain costs one LLM call per fact (the blast-radius
   lesson); judge_failure is persisted and operators invoke the
   drain. Bound: max_rounds × limit facts per call.]**
9. **cancelled semantics (F7 completion):** EVENT_STATUSES grows
   'cancelled'; only reachable transition is planned→cancelled via a
   validated judgment; occurred never cancels; cancelled is terminal
   for re-affirmation merges (disclosed). Store validates transitions;
   the judge writes status through a guarded targeted update.
   **[CORRECTED (R1/R2): EVENT_STATUSES (the add_fact INPUT set)
   deliberately does NOT grow — input validation still refuses
   'cancelled'; mark_event_cancelled() is the only writer, now behind
   a cue-vocabulary + shared-subject deterministic gate (R2-B2). Live
   reads exclude cancelled by default (include_cancelled opt-in);
   facts_as_of has no cancellation-time axis, disclosed in its
   docstring.]**
10. **Additive to the scored path:** knowledge-update sits at 0.952 on
    the old raw-turn path — untouched facts and all Stage 1-3 reads
    keep working unchanged; the judge only ever ADDS supersession
    links, t_invalid values, and audit rows.

### Build surface
llm/supersession.py (SupersessionJudge: shortlist, co-signal, LLM call,
validation, direction, apply, sweep) · db/models.py (t_invalid,
supersession_judgments table, EVENT_STATUSES+cancelled
**[CORRECTED R3-Ma4: EVENT_STATUSES lives in db/semantic_facts.py and
deliberately does NOT grow — mark_event_cancelled/
reinstate_cancelled_event are the only status writers]**) ·
db/semantic_facts.py (t_invalid param on supersede, transition
validation, provenance exposure) · db/engine.py (_migrate_stage4
additive) · llm/consolidation_v2.py (post-commit wiring + report/log
fields + recover_judgments).
Out of scope (disclosed): identity-fact judgment; time-aware retrieval
over t_invalid (Stage 5); Tier 2-3 dedup adjudication (separate item,
shares no code path yet); any prompt change to extraction (PARKED).

### Stage 4 — G1 + G2 record (2026-08-06/07)

**G1: 204/204 across 5 scoped files** (test_supersession.py NEW: 21;
consolidation_v2: 54; fact_entities: 64; semantic_facts: 60;
temporal_kg: 5) + the script-style conflict-detection suite at 100%
after a ROOT-CAUSE fix G1 forced: the shared contradiction vocabulary
could not see third-person "works at" (the KG typed-relation port had
noticed this and patched around it LOCALLY months ago instead of fixing
the source — the local patch is now redundant, noted in both places).
Coverage: llm/supersession.py 96% (the only uncovered lines are the
real-LLM HTTP body, G2's job — same disclosed pattern as the extraction
engine). Failure paths pinned: out-of-shortlist/garbage proposals
dropped loudly; co-signal disagreement blocks LLM-only verdicts;
domain-time REVERSAL (the Josh case) and tie-skip; identity/events/
superseded skip WITHOUT an LLM call; cancellation only reaches live
planned events, guarded by the same rowcount pattern supersede()
race-proves **[R4-Ma5 downgrade: the mark_event_cancelled rowcount
guard itself is NOT race-proven — waived with record]**; cancelled is
terminal
against re-affirmation; t_invalid validated; judgment rows on every
path incl. skips; poison-proof cursor sweep; engine wiring best-effort
with persisted judge_failure and recover_judgments drain.

**G1 caught one design flaw before any critic did:** a single capped
shortlist pool let recent same-type peers CROWD OUT planned events —
cancellation candidates could never be seen. Fixed as reserved pools
(peer cap 12 + planned cap 4), both caps disclosed in the audit row.

**G2 (real llama3.1 extraction + the judge's FIRST live runs,
benchmarks/consolidation_v2_stage4_smoke.py):**
- **The G2 gate did its job twice before passing.** Run 1: ZERO
  supersessions — every judgment skipped "no candidates" because the
  archetypal knowledge-update fact ("personal best 5K time") carries NO
  named entity and the shortlist was entity-only. Graphiti's candidate
  search is TEXT search; entity-only was our shortcut. Fixed: a third
  reserved pool — lexical candidates (TF-IDF >= 0.25 against the newest
  300 live same-type facts; window disclosed, older facts reachable via
  entities only), pinned by tests. Run 2 exposed the second truth: the
  5K "update" extracts as an EVENT, and events are excluded from
  judgment BY DESIGN (occurrences never supersede) — the metric-update
  class is a DISCLOSED boundary, not a silent miss, kept in the smoke
  as its own labeled pass (all skips, 0 supersessions, correct).
- **The demonstration case (employment question, real corpus, both
  session orders): the machine works end to end.** Backfill order
  produced a REAL supersession: "The user plans to catch up with Rachel
  soon." [2023/05/23] superseded BY the domain-later restatement
  [2023/05/26] — with the fact arriving LAST losing on DOMAIN time
  (direction REVERSED exactly as designed; processing order did not
  decide), t_invalid=2023/05/26, superseded_at stamped separately,
  transition_text renders the change, facts_as_of(2023/05/23) still
  shows the old fact at its own date.
- **The gates visibly prevented model confusion:** in-order pass, the
  model proposed a MUTUAL supersession (fact 4 -> 6 AND fact 6 -> 4,
  same-day cross-topic follow-ups with Jason/Alex) — both dropped by
  the domain-time tie gate; an incoherent bidirectional supersession
  never reached the store. One candidate double-listed in both arrays
  was dropped by the status guard. **[CORRECTED (R1/R2): the
  double-listing claim was FALSE as written — pre-fix the candidate
  was superseded AND cancelled with nothing dropped. Pinned behavior
  now: the TYPE guard drops the supersession; a cancellation with a
  cue and shared subject proceeds.]**
- **Honest asymmetry, disclosed:** in-order=0 vs out-of-order=1
  supersessions on the same sessions — the small model's judgment is
  not symmetric across presentation orders (it declined the Rachel
  pair in one order and proposed it in the other), and extraction
  itself varied slightly between passes (Ollama nondeterminism at
  temperature 0). Under-supersession in one order = the disclosed MILD
  class (stale duplicate visible, history intact). This is exactly the
  degradation the design chose over the dangerous class.

G3 round 1 dispatched.

### Stage 4 — G3 ROUND 1: BLOCK (4 blockers, 8 majors, 12 minors) → all fixed

**The round's headline: the critic PIERCED THE CENTRAL SAFETY CLAIM
twice** — over-supersession was possible through two independent paths,
both reproduced with in-scope model output and passing gates. Exactly
what adversarial rounds exist for. Resolutions:

**R1-B1** a planned EVENT could supersede a true current STATE (the
superseded loop had no type guard; the reserved planned pool — a G1
fix — guaranteed the event was present, and future-dated always won on
domain time) → TYPE GUARD: only same-fact_type facts supersede; events
are cancellation candidates only; pinned. **[CORRECTION to the G2
record above: the "double-listed candidate dropped by the status
guard" claim was FALSE as written — pre-fix it was superseded first
and cancelled second with nothing dropped. Post-fix behavior, pinned by
test: the type guard drops the supersession and the cancellation
legitimately proceeds.]** **R1-B2** the co-signal was VACUOUS for every
lexical-pool candidate (admission cosine == agreement cosine, same
pair, same constant — 'allergic to peanuts' invalidated by 'allergic to
shellfish', the Mem0 #1674 class; and the 0.25 provenance comment cited
a function that uses no cosine at all) → INDEPENDENCE RULE: similarity
may never approve what similarity admitted; lexical candidates need a
subject-guarded polarity flip OR the new METRIC-UPDATE signal (numbers
differ + numbers-stripped texts >=0.7 cosine — the 'PB is X → is Y'
replacement shape no vocabulary covers); bare cosine approves
entity-pool candidates only; peanuts/shellfish pinned. **R1-B3**
actions and audit row committed in SEPARATE transactions — a crash
window left a supersession applied with NO judgment row and the fact
unsweepable (superseded facts fail the sweep predicate) → ONE
transaction for all surviving actions + the audit row (store refusals
drop that action, roll back, retry the remainder — bounded, every
exclusion recorded); audit-write failure now rolls actions back;
pinned. **R1-B4** design bullet 8 claimed the judgment sweep was
"auto-invoked" — FALSE (zero callers; only the log-row patch ran) →
**[CORRECTED design decision 8: recover_judgments is PRODUCT-CALLABLE,
deliberately NOT auto-invoked — a scope-wide drain is one LLM call per
fact (the R5 blast-radius lesson); judge_failure is persisted and
operators/schedulers invoke the drain]**.

**Majors:** Ma1 bare polarity flip matched unrelated texts ('works at
Google' vs 'left the party early') → subject guard via CONTENT-word
overlap only (the capitalized-phrase entity proxy counts sentence-
initial 'The' — measured — and was excluded); disclosed cost: an
update naming only the new value loses its flip (mild class). Ma2
domain comparison mixed time axes and ignored intervals (a
month-interval fact could lose to a point INSIDE its own interval) →
interval-aware strict-precedence, same-axis only, overlap=skip;
pinned both directions. Ma3 'cancelled' had NO reader (current_facts/
facts_overlapping returned voided events as live; "EVENT_STATUSES
grows" was also false — input validation still refuses it, correctly)
→ live reads exclude cancelled by default (include_cancelled opt-in);
facts_as_of's no-cancellation-time-axis limit DISCLOSED in its
docstring. Ma4 audit rows now carry per-candidate POOL provenance,
domain intervals, mention dates, and the judgment's constants. Ma5
skip reason no longer blames entities when three pools ran. Ma6
fixture blindness: other-scope candidate made lexically identical so
dropping the scope filter turns the test red. Ma7 the module-level
conflict_detector import re-armed the import-time-migration trap (the
critic's run migrated the production DB AGAIN — second incident of the
class; additive-only, verified no data loss) → ALL conflict_detector
imports made lazy. Ma8 tests/test_conflict_detection.py (a script,
outside conftest protection) pins a scratch DB path before any import.

Post-fix: 211 tests green across the 5 scoped files + conflict script
100%. Smoke rerun next; R2 dispatched after.

**Post-R1 smoke rerun (rebuilt gates): artifacts reproduce** — same
Rachel supersession in backfill order (t_invalid=2023/05/26), same
in-order=0/out-of-order=1 disclosed asymmetry, boundary case 0 as
designed. The rebuilt co-signal did not cost the real demonstrated case.

### Stage 4 — G3 ROUND 2: BLOCK (4 blockers, 8 majors, 12 minors) → all fixed

Three of the four blockers lived INSIDE R1 fixes — the standing lesson
that a fix is a change like any other, at full force. Resolutions:

**R2-B1 — the metric-update signal was arithmetically vacuous**: the
TF-IDF tokenizer never sees digits, so "numbers-stripped" similarity
EQUALED plain similarity (20k randomized pairs, zero differences) —
similarity approving what similarity admitted, again; '5K run 25:31'
was superseded by '10K run 55:12' at cosine 1.000 → REBUILT
digit-aware: number-MASKED texts must be EXACTLY identical, numeric
positions align, and EXACTLY ONE normalized value may differ (two
differing numbers means one is an identity component — a different
race, not an update; '$2,000'/'$2000' and '70'/'70.0' normalize equal
= restatement). All five critic repro pairs pinned; disclosed cost:
simultaneous two-value updates missed (mild class). **R2-B2 —
cancellation was a PURE LLM verdict and Ma3 multiplied its blast
radius** ('prefers oat milk' cancelled a Tokyo flight the unrestricted
planned-pool fallback had shown it) → deterministic gate, both halves
required: cue vocabulary in the NEW text ('called off', 'cancelled',
'fell through'…) AND a shared subject (entity node or content word);
the entity-less planned pool is now TOPICALLY filtered; Tokyo pinned
three ways (pool, cue, positive control). **R2-B3 — the Ma8 script
guard had been injected INSIDE the module docstring and NEVER
executed** (proven: an inherited env var migrated 17 tables; third
incident of the class) → guard moved above the docstring and VERIFIED
BY EXECUTION against a decoy path this time, not by reading the diff.
**R2-B4 — content-word overlap was not a subject guard** ('studying at
Stanford' LOST to 'left Stanford Stadium'; 'lives in Berlin' LOST to
'moved from Berlin Hauptbahnhof') → the subject is now a SHARED ENTITY
NODE from Stage 3's join table (real NER identity, one query for the
whole shortlist); entity-less facts cannot flip at all — their only
route is the strict metric signal. Stanford/Stadium pinned with a
same-node positive control.

**Majors:** cancelled-reader tests written (dropping either filter or
flipping the default now fails); interval TOUCH pinned as ambiguous
(< vs <= mutation red) and the occ-axis reversal branch executes under
test; the five falsified design-decision lines carry INLINE CORRECTED
annotations at the lines themselves (file convention), including the
G2 record's false 'status guard' claim; cap tests no longer monkeypatch
constants back to their defaults (the no-op-patch blindness) and the
lexical window is pinned at defaults; the dead double-consumption guard
removed (unreachable once the type guard precedes); the new signal got
REAL-corpus-model exercise — smoke Part D: the digit-aware signal's
first live run, REAL judge LLM, 25:31 → 23:15 superseded with
t_invalid=2023/05/30 and the transition rendered (store-level, labeled
honestly). **Minors disposition (every one, per R2's demand):** m1
numeric normalization LANDED (in B1); m3 duplicate proposal ids
collapse (pinned); m8 threshold provenance comment rewritten honestly
(0.25 is OUR choice, 2.5x the shipped topic gate — the forget_about
citation was wrong); m9 stripped_cosine dup field REPLACED by
masked_equal/shared_nodes; m5 smoke result line no longer prints a
criterion the run fails; m4 recover_judgments bound stated
(max_rounds × limit); m2 multi-reversal audit wording, m6
judge_failure newest-row patch, m7 success counters: WAIVED WITH
RECORD (wording/plumbing, no correctness effect; newest-row patch
shares the link-recovery pattern and its disclosed same-session
caveat); m10 equivalent mutant acknowledged (not chased); m11
facts_as_of cancelled-inclusion already disclosed in its docstring;
m12 coverage now reported WITH its uncovered set.

Post-fix: **220 tests green** across the 5 scoped files + conflict
script 28/28 (guard execution-verified). Coverage llm/supersession.py:
**98% — uncovered set is exactly lines 222-235, the real-LLM HTTP
body** (G2's job, the standing disclosed pattern). Smoke: demonstration
+ boundary passes reproduce; NEW Part D exercises the rebuilt signal
live. R3 dispatched.

### Stage 4 — G3 ROUND 3: BLOCK (1 blocker, 8 majors, 12 minors) → all fixed

R3 verified every headline number exact, caught all seven demanded
mutations, and confirmed the round-2 rebuilds hold — then found the
last tautology. Resolutions:

**R3-B1 — the cancellation "shared subject" gate could not reject
anything reachable** (the planned pool's admission test WAS the gate:
same function, same pair — and a REAL llama3.1 run cancelled a
climbing-competition plan over a gym-membership cancellation, 3/3
deterministic, with no API back). Fixed two ways: (1) the BINDING gate
— independent of pool admission, three deterministic requirements: a
NON-NEGATED cue (the repo's own 40-char negation window, reused —
R3-Ma6's 'did not cancel' now refuses), >=2 shared content words (one
is topical coincidence), and the cue CLAUSE itself naming the plan;
gym/climbing, negation, and clause-binding all pinned end-to-end.
(2) reinstate_cancelled_event() — the store's only one-way destructive
transition gets its shipped, guarded escape hatch (operator-invoked by
design, never reachable from the judge).

**Majors:** Ma1 the "dead" length check was NOT dead (a literal '#' in
source text breaks equal-masks-imply-equal-counts — fuzz: 53/300k
misaligned comparisons returning True) → restored, counterexample
pinned; the lesson: my dead-code reasoning was itself an unverified
claim. Ma2 interval touch pinned in BOTH directions. Ma3 the
entity-node half now testable through the reachable path (binding gate
pins). Ma4 the last falsified line (Build surface EVENT_STATUSES claim)
inline-corrected. Ma5 the smoke RESULT line now states Part D is
store-injected/hand-authored, and the honest-claims list records that
THE METRIC SIGNAL HAS NEVER FIRED ON AN EXTRACTED FACT (the lexical
pool produced zero real-corpus candidates in three consecutive
rounds). Ma6 negation (in B1). Ma7 single-number pairs REFUSED —
value-update vs identity-difference ('child is 7'→'4' vs 'apartment
12B'→'14B') is deterministically undecidable and a real model proposed
the apartment class 3/3; the 'rent is 1800'→'1900' miss is the chosen
mild class, disclosed. Ma8 the stale _cosignal docstring (still
describing the killed cosine design) rewritten.

**Minors:** m1 European-decimal commas no longer collapse ('1,5' != 15
— grouping-shape-only stripping, pinned); m4 boolean-id rejection
pinned; m2/m3 pool and sweep live-filters pinned (superseded facts
never shortlist, never sweep); m9 cue vocabulary extended
(cancellation/cancels/calls off/backed out/pulled out/shelved); m7
recover bounds stated numerically in both docstrings; m8 test renamed
for what it pins; m5 rowcount-guard direct-race test WAIVED WITH
RECORD (the guard is the same shipped pattern supersede() proves under
its real 2-process test; constructing the interleave for
mark_event_cancelled needs an event-listener harness whose complexity
exceeds the claim — the claim is downgraded to "guarded by the same
rowcount pattern", not "race-proven"); m6 terminal-merge comment
reworded; m10 include_cancelled zero-callers = Stage-5 wiring reality,
recorded; m11 residual risk (entity-pool bare cosine) noted with the
critic's own 3/3-declined probes; notation restatements ('25:31' vs
'25.31') remain a disclosed residual. **[CORRECTED R4-m9/R5-m7: the
single-number refusal (R3-Ma7) CLOSED this class — such pairs carry
one numeric token and now refuse.]**

Post-fix: **228 tests green** (supersession 45 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5) + conflict
script 28/28. Coverage llm/supersession.py **98% — uncovered set
exactly the real-LLM HTTP body (287-300)**. Smoke reproduces:
demonstration + boundary + Part D (labeled store-level). R4 dispatched
as the intended closing round.

### Stage 4 — G3 ROUND 4: BLOCK (1 blocker, 5 majors, 12 minors) → all fixed

R4 verified every headline number exact, caught all 12 demanded
mutations by name, confirmed reinstate is judge-unreachable BY CODE
PATH, and then broke the v2 binding gate's calibration. Resolutions:

**R4-B1 — the ≥2-shared-content-words rule was calibrated on an axis
ORTHOGONAL to correctness**: two GENERIC shared words ('weekend
training') bound a false cancel 3/3 on the real model, while verbatim
TRUE cancellations of short-named plans ('cancelled the Rome
marathon') refused at one shared word; the critic also measured that
coverage RATIOS do not separate (0.50 both cases). REBUILT as
CONTAINMENT — you can only cancel a thing you NAME, and everything the
cancelling clause names must be part of the plan: per non-negated cue
occurrence (every occurrence checked — R4-m1), the clause's object
words (4-char floor, recovering the distinctive short words the 5-char
rule destroyed: 'yoga', 'rome'; cue SPANS removed before extraction so
'called' cannot leak from 'called off') must be non-empty and a SUBSET
of the plan's words. The physio case names session+physiotherapist —
not in the marathon-camp plan — and refuses; Rome binds. Pinned:
weekend-training repro end-to-end, Rome positive, multi-occurrence
negation, insurance-'cancellation'-noun refusal, pronoun-only
('cancelled it' names nothing) refusal. Reachability note stands: this
gate is prompt-unreachable TODAY (0/23 planned facts) — it exists to
be right BEFORE the parked plans-as-events decision flips, not after.

**Majors:** Ma1 the sweep live-filter 'pin' was a TAUTOLOGY (assert X
or True — an assertion that cannot fail, answering a 4th-round
survivor with a rubber stamp) → replaced with a real pin (a
superseded, never-judged fact must produce ZERO judgment rows through
the sweep; deleting the filter turns it red). Lesson recorded:
assertions that cannot fail are the lookalike-test class in its purest
form. Ma2 the R3 smoke-label claim had NOT landed (the exact wording
R3 rejected still printed) → landed and grep-verified: the RESULT line
now states Part D is STORE-INJECTED, HAND-AUTHORED, and that the
metric signal has never fired on an extracted fact. Ma3 the Stage-4
honest-claims block is required for close → written at the closing
verdict (below, R5). Ma4 the module design-of-record docstring still
described the killed cosine metric — same phrase, one level up from
where Ma8 fixed it → corrected to the current truth. Ma5 the m5
waiver's downgrade now sits AT the claim ('race-defended' →
'guarded by the same rowcount pattern supersede() race-proves', with
the not-race-proven waiver inline).

**Minors:** m1 multi-occurrence negation (in B1, pinned); m2 insurance
false-cue class (containment refuses, pinned); m10 non-dict model
output dropped loudly (pinned); m9 record corrected — the
single-number refusal CLOSED the notation-restatement residuals
('25:31'/'25.31' are single-number pairs and now refuse); m5
abbreviation clause-splitting verified harmless for cue-first
phrasings **[FALSIFIED by R6-Ma1: honorific periods (Dr./St./Mr.)
truncated the clause and 5/6 phrasings falsely BOUND — every splitter
false positive is in the binding direction; fixed by lookbehind and
pinned]**; m4 the 40-char negation window bound is DISCLOSED (beyond
it, a distant negation is invisible — vocabulary-window residual); m6
plans whose only distinctive words are under 4 chars are structurally
hard to cancel (disclosed miss); m7 'plans'/'attend' added to the
structural stop extension; m3/m11 hypothetical/question cue forms and
include_cancelled's zero callers recorded (extractor contract makes
the former unrealistic; the latter is Stage-5 wiring reality).

Post-fix: **233 tests green** (supersession 50 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5) + conflict
script 28/28. Coverage llm/supersession.py **98% — uncovered set
exactly the real-LLM HTTP body (312-325)**. Smoke reproduces with the
honest Part-D label. R5 dispatched as the closing round.

### Stage 4 — G3 ROUND 5: BLOCK (0 blockers, 3 majors, 10 minors) → all fixed

R5 verified every claim exact for the third round running, confirmed
R4-B1's core defect dead under 10 mutations (8 killed; the 2 survivors
became its majors), and blocked the CLOSE on three short items:

**R5-Ma1 — clause selection by STRING defeated per-occurrence
negation** (the same cue surface in two clauses always judged against
the FIRST clause containing it — a negated first mention plus a real
second one produced one measured wrong write; blast radius today zero,
but span-accurate negation demands span-accurate clauses) → clause
boundaries computed by POSITION from the splitter's spans; the
critic's exact wrong-write text pinned refusing, and the mirror
true-positive (a real cancellation AFTER an unrelated one) pinned
binding. **R5-Ma2 — the 4-char floor, the R4-B1 fix's key ingredient,
was unpinned** (the Rome POSITIVE could not fail at either floor —
containment is symmetric, a lesson now recorded: symmetric rules
cannot be pinned by positives) → pinned by its NEGATIVE: at floor 5,
'rome' and 'boston' vanish and the Rome cancellation falsely binds the
BOSTON marathon plan; floor-4 refuses. **R5-Ma3 — the killed v2
self-description survived at the call site and in a test comment**
(third round of this class in the same file) → both dead, verified by
grep. **Minors:** the structural stop extension pinned ('cancelled
the plans' must name nothing — an empty extension turns it red); the
positive-control reason asserted; the notation-residual record lines
reconciled (single-number refusal closed that class); the remaining
measured residuals fold into the disclosures below.

Post-fix: **236 tests green** (supersession 53 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5) + conflict
script 28/28. Coverage llm/supersession.py **98% — uncovered exactly
the real-LLM HTTP body (320-333)**. Smoke reproduces with the honest
Part-D label. R6 dispatched as the focused confirmation round.

### STAGE 4 HONEST CLAIMS OF RECORD (critic-approved wording, R5)

**CAN claim:** per-fact supersession judgment over the semantic tier —
batched shortlist from three reserved pools (entity / lexical /
planned, caps in the audit row), reasoning-first grammar-forced
schema, and the LLM only PROPOSES while five deterministic gates
decide (shortlist membership, same-fact_type, independent co-signal,
domain-time direction, guarded store writes); direction from DOMAIN
time, never processing order — demonstrated end to end on a real
corpus (run in both session orders; the supersession fired in the
BACKFILL order — disclosure 2 carries the 0-vs-1 asymmetry), the fact
arriving LAST losing because it was domain-earlier (Rachel [2023/05/23] superseded by [2023/05/26],
t_invalid=2023/05/26, superseded_at stamped separately, facts_as_of
still shows the old fact at its own date); every judgment persisted
(shortlist with per-candidate pool provenance, co-signals, raw output,
applied actions, dropped proposals with reasons, or the skip reason)
and committed in ONE transaction with the audit row, so
applied-but-unaudited is impossible; nothing destructive — supersede()
invalidates, never deletes, and mark_event_cancelled() is reversible
by an operator-invoked reinstate_cancelled_event() the judge cannot
reach by code path (verified: no dynamic dispatch, zero non-test
callers); a deterministic cancellation binding gate stands between the
model and any cancellation (cue vocabulary, position-accurate
per-occurrence negation, containment); 239 tests green across the 5
scoped files (56/54/64/60/5), conflict-detection 28/28,
llm/supersession.py at 98% coverage with only the real-LLM HTTP body
uncovered.

**MANDATORY DISCLOSURES alongside any of the above:**
1. Scope: judgment runs on state and preference facts ONLY; identity
   is excluded by design; events are excluded except the
   planned→cancelled attribute update.
2. Order asymmetry, measured: the same real sessions gave 0
   supersessions in order and 1 out of order — the small model's
   judgment is not order-symmetric, and extraction differs by order
   too.
3. The digit-aware metric-update signal has NEVER FIRED ON AN
   EXTRACTED FACT. Its only live demonstration (smoke Part D) uses
   STORE-INJECTED, HAND-AUTHORED facts with a real judge LLM; the
   real-corpus metric class extracts as an EVENT and is excluded by
   design.
4. Cancellation is PROMPT-UNREACHABLE today (0/23 planned markers; the
   extractor types cancellations as past events and the judge skips
   them). The gate exists to be right BEFORE the parked
   plans-as-events prompt decision flips — it is not evidence that
   cancellation works in the product.
5. The containment gate's own misses and residuals, all measured:
   pronoun-only cancellations ('cancelled it') name nothing and
   refuse; plans whose only distinctive words are under 4 characters
   are structurally hard to cancel; plurals do not stem ('marathons' ≠
   'marathon' → refuse); a negation more than 40 characters before the
   cue is invisible; natural true cancellations carrying any extra
   word refuse (5 of 8 measured phrasings — the chosen mild class); a
   clause naming only GENERIC words that appear in the plan binds
   without naming anything distinctive ('cancelled his weekend
   training' → the marathon-camp plan; real llama3.1 declined all
   such probes 3/3); verbose/composite plans are strictly easier to
   cancel because containment sees a word UNION with no adjacency;
   question and hypothetical cue forms bind; the clause splitter
   treats a period after ANY 1-3-letter token as an abbreviation
   (missed sentence splits GROW the clause, which is the refusing
   direction — pinned both ways; one contrived measured exception: a
   grown clause can rescue an EMPTY named set, R8-n2); the negation
   lookback deliberately crosses clause boundaries (erring wide is its
   safe direction); the same truncation applies to the and/but branch
   of the splitter: a conjunction inside a coordinated noun phrase
   splits before the SHARED HEAD NOUN and deletes it, so 'cancelled
   the Friday and Saturday dinner reservations' names only {friday}
   and binds a Friday LUNCH plan — 5 of 6 measured cue-first
   phrasings; every splitter false split, period or conjunction, moves
   toward BINDING (R8-n1; the branch is load-bearing and a real fix
   needs shared-head-noun detection — recorded, pinned, not claimed
   safe).
6. Sweeps are OPERATOR-INVOKED: judge_missing() is a hand-run
   cursor-paged recovery/backfill sweep; there is no scheduler.
7. mark_event_cancelled()'s rowcount guard is the same shipped pattern
   supersede() race-proves under a real 2-process test, but is NOT
   itself race-proven — waived with record.
8. NOTHING is wired into product retrieval. ConsolidationV2/
   SupersessionJudge have no caller outside benchmarks and tests;
   include_cancelled has zero non-test callers; t_invalid-aware
   retrieval is Stage 5.

**Stage-3 forward items answered:** 'cancelled' status reserved for
Stage 4 → DONE (terminal, operator-reversible); domain-validity-end
for states (Graphiti's invalid_at gap) → DONE (t_invalid, validated,
set from the deciding axis); occurrence JUDGMENT / cancellation → DONE
as one attribute the general judgment may propose, never a bespoke
detector, reachability disclosed; the 'planned' marker's
prompt-unreachability → STILL A QUEUED FOUNDER DECISION, unchanged
(0/23).

### Stage 4 — G3 ROUND 6: BLOCK (0 blockers, 1 major, 3 minors) → all fixed

R6 confirmed all three R5 majors closed (6/6 mutations killed, both R5
survivors dead, zero tautological asserts by AST scan, claims block
audited line-by-line) and found one PRE-EXISTING hole plus a falsified
record line:

**R6-Ma1 — abbreviation periods truncated the cue clause, and
truncation is always in the BINDING direction** (a smaller named set is
strictly more likely to be a subset): 'cancelled the appointment with
Dr. Meyer' lost 'Meyer' at the split and falsely bound a DIFFERENT
planned appointment — 5/6 honorific phrasings bound; the record's own
'verified harmless' claim about this class was FALSE (it had verified
only the safe cue-last direction) → the splitter now refuses to split
on a period after a 1-3-letter capitalized token (fixed-width
lookbehinds); pinned with the critic's honorific set INCLUDING the
true-cancellation-with-honorific positive; the false record line
struck inline. Blast radius today zero (real model declined 0/3;
cancellation prompt-unreachable) — but this gate exists to be right
before the parked decision flips, so it is fixed by code, not wording.

**Minors:** the negation window is now CLAUSE-BOUNDED (R6-m1: a raw
40-char lookback crossed the boundaries the clause selection computes
— half-applying the R5 fix's own principle); the two reason-string
assertions now prove the MECHANISM, not just the refusal (R6-m2); the
smoke was RERUN under the new splitter for this record (R6-m3 —
demonstration, boundary, and Part D all reproduce).

Post-fix: **237 tests green** (supersession 54 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5) + conflict
script 28/28. Coverage llm/supersession.py **98% — uncovered exactly
the real-LLM HTTP body (332-345)**. The honest-claims block above
stands with one addition to disclosure 5: abbreviation periods no
longer truncate clauses (pinned); single-letter abbreviations
('p.m.') still split, in the safe cue-first direction only. R7
dispatched as the final minimal verification.
**[FALSIFIED by R7-B1 — this addition was a SECOND unmeasured 'class
is safe' sentence, broken at the same 5/6 rate: lowercase
abbreviations ('p.m.', 'vs.') still truncated and BOUND ('Friday 6
p.m. dinner reservation' bound a Friday LUNCH plan). Fixed by code:
the guard is now case-blind over 1-3-letter tokens with
token-anchored lookbehinds, pinned in BOTH directions (truncation
refused; sentence-split still load-bearing); see the R7 record.]**

### Stage 4 — G3 ROUND 7: BLOCK (1 blocker, 1 major, 4 minors) → all fixed

R7 confirmed every R6 code fix (6/6 mutations, exact numbers for the
fourth round running, its own smoke rerun) and blocked on the RECORD:
the R6 disclosure addition was a SECOND unmeasured "class is safe"
sentence, written immediately after being blocked for the first one —
the exact lesson recorded one round earlier. Resolutions:

**R7-B1** lowercase abbreviations ('p.m.', 'vs.') still truncated
clauses and BOUND at the same 5/6 rate ('Friday 6 p.m. dinner
reservation' bound a Friday LUNCH plan) → the splitter guard is now
CASE-BLIND over 1-3-letter tokens with TOKEN-ANCHORED lookbehinds (the
first fix attempt over-blocked — 'marathon.' matched through its last
two letters — and the load-bearing sentence-split pin caught it within
minutes, which is what pins are for); pinned in BOTH directions
(p.m./vs. truncations refuse WITH the naming words surviving; a true
cancellation followed by another sentence still binds because the
sentence period still splits); the falsified R6 line struck inline
with the measured truth, and the splitter/negation behavior now lives
INSIDE mandatory disclosure 5 (R7-m3 — quoters must see it).

**R7-Ma1** R6-m1's clause-bounded negation window had NARROWED the
lookback — the BINDING direction — un-refusing 5/5 scope-ambiguous
negations, undetectably → REVERTED to the wide raw lookback with the
principle recorded where it lives: negation errs WIDE, clause naming
errs NARROW — the two features have OPPOSITE safe directions; the
'did not cancel X and cancel Y' case pinned refusing.

**Minors:** the sentence-period branch is now load-bearing by test
(deleting it fails the Rome-marathon-then-Paris positive); the stale
236-count line reconciled inline; the both-orders phrasing in the
claims block corrected (the supersession fired in the BACKFILL order;
disclosure 2 carries the asymmetry).

Post-fix: **239 tests green** (supersession 56 / consolidation 54 /
fact_entities 64 / semantic_facts 60 / temporal_kg 5) + conflict
script 28/28. Coverage llm/supersession.py **98% — uncovered exactly
the real-LLM HTTP body (341-354)**. Smoke reproduces. R8 dispatched:
verify R7's closures, render the stage verdict.

### Stage 4 — G3 ROUND 8: PASS-WITH-NOTES → STAGE 4 CLOSED (2026-08-07)

R8 (minimal verification): 239/239 reproduced, coverage exact
line-by-line, its own smoke rerun, all three demanded mutations red by
name PLUS one unasked mutation verifying the token-anchor story, 41
probes (honorific family 12/12 refusing with the naming word surviving,
true positives 4/4 binding, negation scope 0/5 binding), all four R7
record edits audited accurate, zero tautological asserts by AST. The
critic's own words: "I could not find a single false number, a single
false statement, or a single unpinned fix." Verdict: **the STAGE 4
HONEST CLAIMS OF RECORD block, as amended through R7 and R8-n1, is the
stage's definitive record.**

Notes landed with the close: **R8-n1** the and/but splitter branch
truncates coordinated noun phrases the same way the period branch did
(the critic's verbatim clause appended INSIDE disclosure 5; pinned AS
DISCLOSED — the pin asserts the measured residual so a future edit
cannot silently widen or silently "fix" the class; a real fix needs
shared-head-noun detection; blast radius doubly zero via disclosures
4/8). **R8-n2** the clause-growth safe-direction argument has one
contrived measured exception (recorded in disclosure 5). **R8-n3** the
stale count text corrected at source. **R8-n4** the critic disclosed
its own coverage-file write (untracked data only; lesson filed:
COVERAGE_FILE to scratch). Final count with the n1 pin: **240 tests**
(supersession 57).

**Eight rounds, blocker trajectory 4→4→1→1→0→0→1→0.** The first four
rounds broke the CODE (over-supersession twice, vacuous gates,
tautological binding); the last three litigated the RECORD (unmeasured
"safe" claims written twice, then struck and replaced with measured
truth). Both mattered. The standing lessons this stage minted: a fix
is a change like any other; similarity may never approve what
similarity admitted; symmetric rules cannot be pinned by positives;
fixing one alternative of an alternation leaves the siblings broken;
negation errs wide while naming errs narrow; and never write "class X
is safe" unless the record names the probes.

Stage table: Stage 4 ✅ DONE (G1 ✅ 240 tests / 98% judge coverage,
LLM-body-only uncovered · G2 ✅ real-corpus supersession in backfill
order + boundary + Part D, reproduced by the critic across four rounds
· G3 ✅ R8 PASS-WITH-NOTES after R1-R7 each BLOCKED and fixed).

## STAGE 5 — Facts-first retrieval wiring + $0 diagnostics — started 2026-08-07

Founder GO 2026-08-07 (same triple-gate discipline, "never a happy path,
brutally honest"). Scope, stated plainly (open-items ledger #6): NOTHING
was wired into product retrieval — ConsolidationV2 had no caller outside
benchmarks/tests and the live app answered from the old memory path.
Stage 5 wires BOTH directions: the product read path retrieves FACTS
first (design §5.3), and the product write path gains a real
ConsolidationV2 caller. Plus the $0 diagnostics that measure the new
path on real data without spending anything.

### Design (frozen before code, decisions numbered for the critic)

Product surfaces affected:
- `llm/context_assembler.py` — the ONE assembler. The MCP server's
  recall_memory calls it, AND `benchmarks/adapters/agentmem_adapter.py`
  calls it (`assemble(namespace, query, agent_id=namespace)`) — so this
  change flows into the Gate C path automatically. That makes D3's
  no-regress property load-bearing for the banked numbers (66.0% / 0.952
  knowledge-update were measured through this assembler with zero facts
  in store).
- `mcp_server/server.py` — new `consolidate_session` tool =
  ConsolidationV2's first product caller. v1 `summarize_session`
  (DBSCAN compression summaries) left untouched: different feature,
  different table; retiring it is not this stage's call.
- NOT touched: benchmark adapter, eval harness, api/app.py demo
  endpoints (demo fact-inspector queued in ledger).

D1. Fact ranking = lexical TF-IDF + entity-seeded KG reads, fused with
    RRF (k=60, repo precedent in multi_vector_retrieval.py). Grounds:
    the repo's own measured verdict — every dense/hybrid variant LOST
    to plain TF-IDF on entity-heavy English questions, and "the next
    lever is fact decomposition at ingestion, not a better ranker."
    The entity arm reuses Stage 3's `facts_for_entity` (case-blind
    seeds, ALIAS_OF closure) — the cross-lingual reach lexical cannot
    have. Dense fact embeddings EXCLUDED from v1 (measured loser on
    English; cross-lingual named-entity queries covered by the entity
    arm; token-free cross-lingual queries with NO named entity are a
    measured Gate-E gap, recorded, not silently claimed).
D2. Presentation (the CPP-informed part — their finding: holding
    retrieval fixed, presentation moved 0.36→0.61): facts render as a
    dated, chronologically ascending block — `[date] (type) text` —
    with `(planned)` markers and, when a fact superseded a predecessor,
    the transition line (store.transition_text). The fact line IS the
    highlighted evidence; raw-turn chunks below are the segments. The
    full CPP evidence-highlighting slice test stays a Gate C-era
    ledger item.
D3. Budget: facts share the EXISTING 20% semantic allocation — facts
    first, Chroma chunks fill the remainder. No other tier's budget
    changes. ZERO facts in store ⇒ assembled output BYTE-IDENTICAL to
    the pre-Stage-5 assembler (the no-regress guarantee; G1 pins it).
    Boundary disclosed: a scope with thousands of facts would starve
    chunk provenance within the section — revisit with Gate C data.
D4. Provenance NOT rendered into the prompt (density is the measured
    selling point — 8× fewer chars per unit of evidence; source ids
    live in the store, inspectable end-to-end). Disclosed trade-off.
D5. Scope: reads derive scope_key via the SAME make_scope_key used by
    writes (read and write can never disagree). MCP recall_memory
    passes agent_id=None today — single-tenant default, recorded.
D6. Failure containment: fact-tier failure logs a WARNING (not the
    debug-swallow the other tiers use) and falls through to the old
    path — availability over completeness on the READ path; the write
    path already persists its failures (link_failure/judge_failure).
D7. Caps, all disclosed: lexical arm ranks the newest 500 current
    facts (one fitted vectorizer per query, not per-pair); entity arm
    bounded by Stage 3's closure caps; per-query fetch depth is
    budget-aware like the chunk path. Diagnostics print every cap.
D8. Cancelled facts stay excluded (current_facts default). Rendering
    "X was cancelled" as knowledge is queued WITH the parked
    plans-as-events decision (they only make sense together — today
    cancellation is prompt-unreachable, 0/23 markers). facts_as_of's
    missing cancellation-time axis stays deferred: the reader shipping
    in Stage 5 is a current-facts view, not an as-of view.
D9. B1 lesson honored: query NER/surface extraction runs BEFORE any DB
    session opens; the retriever is strictly read-only; no model load
    can ever sit under a write lock on this path.

Deliverables: `llm/fact_retrieval.py` (FactRetriever), assembler
wiring, MCP `consolidate_session` tool, `tests/test_fact_retrieval.py`
(G1), `benchmarks/facts_first_diagnostics.py` ($0, env-pinned scratch
DB, real local models only — no API spend anywhere in this stage).

**[CORRECTED before build — D1 fusion dropped]** D1 as first written
said RRF(k=60) fusion. The repo's own measurement argues against it:
install_best_chroma's 6-variant table shows rank-fusing a weak signal
into the strong lexical one DILUTED it (hybrid RRF bare turns 7/30 vs
pure TF-IDF 11/30). Built instead: **lexical-primary, entity-floor** —
facts ranked by TF-IDF cosine in the champion's exact configuration
(max_features=512, sublinear_tf, floor 0.01); entity-linked facts the
lexical arm missed are APPENDED after the lexical ranking (ordered by
query-surface multiplicity, then recency) — they fill budget lexical
left unclaimed and never displace or dilute a lexical hit. Cross-lingual
reach is preserved: when lexical has zero hits (no shared tokens), the
entity floor IS the ranking. No RRF anywhere in Stage 5.

**Pre-build finding (latent bug, Stage 3 code):** `facts_for_entity`
filters `superseded_by IS NULL` but not `event_status = 'cancelled'` —
a judged-cancelled planned event has NO successor, so the entity read
path would surface a voided claim as live knowledge. This is exactly
the S4-R1-Ma3 class the Stage-4 `current_facts` filter exists to stop;
the guard was added to one reader and not the other. No production data
can trigger it today (cancellation is prompt-unreachable, operator-only
writes), which is why Stage 4's gates never saw it: its G1 tests
exercise `current_facts`, not the entity path. Fix at root this stage:
`include_cancelled=False` parameter on `facts_for_entity`, same
contract as `current_facts`, pinned in G1.

### Stage 5 — G1 + G2 record (2026-08-07)

**Built:** `llm/fact_retrieval.py` (FactRetriever: lexical-primary in
the champion TF-IDF configuration, entity recall floor through Stage
3's join table + ALIAS_OF closure, budget-fill by rank / render by
chronology, transition lines via `transition_text`); assembler wiring
(`[SEMANTIC FACTS]` claims the 20% semantic allocation first, chunks
take the remainder, empty store ⇒ byte-identical output); MCP
`consolidate_session` (ConsolidationV2's first product caller) +
`recall_memory` scope passthrough; `facts_for_entity` root fix
(include_cancelled=False, the S4-R1-Ma3 class on the entity path).
G1 catch during build: bare-name queries ('Rachel') and merged
name spans ('Rachel Priya') never reached the entity floor — NER
needs sentence context and merges adjacent names. Fix:
`_query_surfaces` unions NER surfaces, the turn path's regex
fallback, and capitalized sub-words of multi-word spans; generosity
is safe on the READ path because node existence gates admission (the
write path stays strict — it CREATES nodes; this path only looks
them up).

**G1: 24 tests** (tests/test_fact_retrieval.py) incl. failure paths:
facts-tier death → WARNING + raw fallback; disable={"facts"} skips
the retriever entirely; budget starvation is loud (section absent);
byte-identity pin; cancelled-via-entity pin; scope isolation; MCP
unknown-session refusal + report passthrough. Regression: **255
passed** across test_fact_entities, test_supersession,
test_semantic_facts, test_consolidation_v2, test_mcp_server,
test_eval_harness, test_agentmem_adapter (scoped files, never broad
pytest — the standing e2e-cost rule).

**G2 ($0 diagnostics, benchmarks/facts_first_diagnostics.py — real
LongMemEval `_s` sessions, real llama3.1 extraction + judge, engine-
backed store, champion TF-IDF chunk backend): TWO consecutive full
runs BYTE-IDENTICAL (modulo scratch path).** Artifacts:

- A. NO-REGRESS: empty fact store ⇒ assembled context byte-identical
  with facts tier enabled vs disabled (6264 == 6264 chars).
- B. Rachel knowledge-update (gold 'TechCorp'): answer present as a
  dated fact in [SEMANTIC FACTS] (945 chars, first hit at char 900);
  **the raw-turn path finds it NOWHERE** (its own 3463-char semantic
  section misses it — the fact tier surfaces what turn retrieval
  structurally cannot from a different session's context).
- C. CROSS-SESSION REACH: recall from an unrelated session
  (b10f3828_1) still surfaces TechCorp, facts-only=True — per-session
  chunk search cannot do this at any budget; the scope-wide fact tier
  is the only path.
- D. Boundary (disclosed): the 5K metric — '25:50' surfaces as a
  dated fact; the older 27:12 event stays visible too (events and
  identity facts are excluded from judgment by design — no false
  supersession, mild visible duplicate, newest-last chronology).

**TWO REAL PRE-EXISTING BUGS FOUND AND FIXED (G2 doing its job —
Part A FAILED on the first full run, 6264 vs 4960 chars, and the
diff instrumentation traced it):**
1. **Redis ghost keys**: the hot cache keys on session_id ONLY — no
   DB identity — so any test/benchmark against a live localhost
   Redis reads stale turns from EARLIER runs and writes its own (188
   stale keys observed live, including the exact benchmark session
   id in play). The scratch-DB pin cannot cover this channel. Fix:
   `AGENTMEM_OS_DISABLE_REDIS=1` kill-switch in RedisCache (before
   any connect), FORCED in tests/conftest.py and the diagnostics.
2. **Warm-cache depth halving**: RedisCache trims to 10 turns but
   `get_history(last_n=20)` served a cache hit anyway — cold reads
   returned 20 turns, warm reads 10 (measured: identical assemble()
   calls, 6264 chars cold vs 4960 warm). Every repeat-assembly on a
   warm cache was silently shortchanged (RECENT TURNS grew 1327 →
   2631 chars in Parts B/C after the fix). Fix in store.get_history:
   a cache hit may answer only when it can satisfy last_n. Both
   fixes pinned in G1 (kill-switch short-circuit; cache-contract).

**Honest notes of record:**
- The Rachel supersession did NOT fire in this window: the audit rows
  show the gate OPEN (entity pool, shared_nodes=true, cosine 0.3688,
  agrees=true) and the temp-0 LLM PROPOSING [] — while in Stage 4's
  window the same pair superseded, critic-reproduced 4×. Ollama
  temp-0 determinism holds within a machine state (two consecutive
  Stage-5 runs byte-identical), not across windows/versions. The
  retrieval layer under test surfaces the CURRENT answer either way;
  supersession only adds the change-history annotation — hence
  history_visible=False here, reported as measured.
- Same artifact, other direction: judgments on facts 4/6 show the
  LLM proposing a WRONG mutual supersession (Jason-collab ↔ Alex-
  partnership) — its own reasoning admits they differ — and the
  co-signal gate killing both ("similarity may never approve what
  similarity admitted"). LLM proposes, gates decide, on real data.
- The 10-turn/20-turn cache bug predates Stage 5 and may have
  touched any prior measurement that assembled twice against a warm
  cache on this machine; banked benchmark numbers should be
  re-verified with AGENTMEM_OS_DISABLE_REDIS=1 at the next paid
  re-run (Gate C) — queued in the open-items ledger, not silently
  absorbed.

### Stage 5 — G3 round 1: BLOCKED (3 blockers, 7 majors, 5 minors/notes) — fix pass record

The critic executed 7 mutations, 5 probes, a full independent
diagnostics rerun (every recorded number reproduced EXACTLY), and the
255-test regression. What broke and what was done:

**B1 (blocker) — truncation deleted the top-ranked fact.** build_block
filled on bare fact_text lengths, re-sorted chronologically ascending,
and the assembler's head-keeping cut the NEWEST — i.e. the rank-0
current answer — while stale 2020 facts survived (critic-measured).
Transition lines weren't budgeted at all (5.4× overshoot measured);
qa_accuracy_eval had already measured this exact class for chunks
(temporal collapsed to 0.13). FIX: rank fills against the FULL
rendered line (transition included) and never exceeds char_budget;
stop at first non-fit (no short-fact leapfrogging); one disclosed
exception — the rank-0 fact is always included and if oversized the
caller truncates that single line's tail, which is rank-safe. Pinned:
test_truncation_cannot_delete_top_ranked_fact (rank-0 chronologically
newest under 30 old facts, asserts surviving text at budget 300 AND
through the real assembler at semantic=100),
test_transition_lines_are_counted_against_the_budget.

**B2 (blocker) — my own G2 cache fix was wrong twice.** With
max_turns=10 and the new depth contract, the assembler's last_n=20
could NEVER hit (L1 became a pure write amplifier), and the unchanged
repopulate loop lpush'd onto warm lists — measured duplication
['t1'..'t5','t1'..'t5'] served to shallower readers as real history.
FIX: max_turns 10→20 (a cache must be at least as deep as the deepest
read it claims to serve); repopulation REPLACES atomically
(RedisCache.replace_history: delete+push+trim pipeline) instead of
appending. Pinned with a FAITHFUL fake modeling lpush/ltrim/lrange/
delete/pipeline (the first fake's push_turn was `pass` — poisoning
was unmodelable by construction, the critic's exact charge):
test_repopulate_replaces_never_duplicates,
test_cache_depth_covers_the_assembler_read, plus an OPT-IN live-Redis
test (db 9, unique key, cleanup; default-skipped per the no-live-
infra rule) — test_live_redis_replace_semantics_opt_in.

**B3 (blocker) — the byte-identity pin was mutation-green against
sem_budget drift.** The fake chroma's chunks were tiny (nothing
approached any budget) and its recorded calls were never asserted.
FIX: chunks are budget-sized (truncation live) and the chroma CALLS —
top_k derives from sem_budget — must be identical between facts-
enabled and facts-disabled assemblies. The critic's MUT4a
(sem_budget -= 5000) is now caught by calls inequality. MUT4b
(`if block:`→`if True:`) is an EQUIVALENT mutant: _fit_to_budget("")
returns "" and empty sections are never appended — no behavior
change exists to pin; claimed as reasoned equivalence, not coverage.

**M1** fact-line forgery (embedded newline forged a ranked line;
embedded '[change history:' forged a supersession story): _sanitize
collapses whitespace and neuters the marker case-insensitively at
render. Pinned: test_render_forgery_neutralized. **M2** the
8-surface cap spent itself on merged spans and dropped 9/10
sub-words: sub-words now INTERLEAVE with their span and the cap is 24
(surface probes are one indexed seed lookup each; Latin never
embeds). Pinned: test_query_surfaces_interleave_spans_with_subwords.
**M7** the "no model load can EVER sit under a database lock" claim
was false (Indic alias fallback can cold-load the encoder inside an
open READ session): claim rewritten to its true scope in the module
docstring — WAL read lock, writers unblocked, Indic-only, disclosed.
**m2** chunk starvation now logs (record wording matched to code).
**m4** _predecessor_targets now scope-filters (defense in depth).
**m5** MCP consolidate_session runs via asyncio.to_thread (was
freezing the whole MCP event loop for tens of seconds). **n5** the
kill-switch test sets its env explicitly and proves the switch (not a
dead constructor) prevents the connect — both mutants now die. **n2**
diagnostics docstring names the function it calls. **n3** diagnostics
exits 1 on a Part A regression. **m1** offsets reported as
within-section and labelled. **m3** undated-facts-rank-last cap
consequence disclosed in the module docstring + ledger.

**RECORD CORRECTIONS (inline, history kept):**
- [CORRECTED R1 — M3] The G2 record's header "TWO REAL PRE-EXISTING
  BUGS FOUND AND FIXED" over-claimed: bug 2 (depth) is fixed at root;
  bug 1 (ghost keys) is CONTAINED by the kill-switch — the root cause
  (session-id-only keys, no DB identity) remains open, now a named
  ledger item. One root-cause fix + one containment, not two fixes.
- [CORRECTED R1 — M4] Part B's "the raw-turn path finds it NOWHERE"
  presented a STRUCTURAL impossibility as a measurement: the gold
  string exists only in the OTHER gold session, so session-scoped
  chunk retrieval could never find it from sids[0] at any budget.
  It is the same axis Part C measures. The diagnostics now say so in
  the output itself; the real claim is only that the fact tier DOES
  surface it.
- [CORRECTED R1 — M5] Part D's "27:12 stays visible / no false
  supersession / newest-last" were asserted, not measured. Now
  measured and printed: both metric lines render ([noted 2023/05/23]
  (event) ... 27:12 / [noted 2023/05/30] (identity) ... 25:50),
  superseded=[] printed per session, old value also present in
  SEMANTIC MEMORY.
- [CORRECTED R1 — M6] The affected-surfaces list omitted
  benchmarks/qa_accuracy_eval.py — a live assemble() caller with the
  TIGHTEST semantic budget (≈4740 tokens), i.e. the caller MOST
  exposed to B1. Now listed; its runs get facts-first automatically.
- [CORRECTED R1 — n1] D5's "MCP recall_memory passes agent_id=None
  today" went stale within the same document — recall_memory now
  accepts and passes agent_id/user_id (G1-pinned).

Post-fix: **30 passed + 1 opt-in skip** in tests/test_fact_retrieval,
**255 regression green**, diagnostics rerun clean (exit 0, Part A
byte-identical 6264==6264, Part D lines printed, LLM outputs
byte-identical to the pre-fix runs).

### Stage 5 — G3 round 2: BLOCKED (1 blocker in 2 facets, 2 majors, 8 minors/notes) — fix pass record

R1's B2 and B3 were CLEARED by the critic with mutation-proven pins
(10 of its 12 mutations died under exactly their named test; the
MUT4b equivalence claim was VERIFIED from code). What remained, and
what was done:

**R2-B1 (blocker) — the R1 fix was applied in the WRONG UNIT.**
build_block honoured a CHAR budget via the chars≈tokens×4 proxy; the
assembler truncates in TOKENS, and rendered fact blocks measure
3.68–3.84 chars/token — always under 4 — so near-full blocks
exceeded the token budget and head-keeping truncation deleted the
chronologically-newest = rank-0 fact at **95 of 115 swept budgets
(83%)**, ordinary English, no adversarial input. FIX: build_block now
fills against TokenCounter.count of the accumulating block versus a
TOKEN budget (the same counter the assembler cuts with), and because
chronological re-ordering can move BPE boundaries, the sorted block
is RE-COUNTED and the lowest-RANKED survivors dropped until it fits —
never the newest. The assembler passes token_budget=sem_budget;
_fit_to_budget is now a guard rail, not the working cut.
**[FALSIFIED R4, struck retroactively at R6 for strike-discipline
uniformity: R4-B1 proved _fit_to_budget's CHAR half WAS still the
working cut at that point — rank-0 lost at 9/9 budgets. The sentence
became true only after the R4 fix closed both units; today the char
gate fires 0/115 in both regimes, critic-verified.]** Lesson
minted (critic's words): **a proxy is a fast path, never a contract —
fixing a budget in the wrong unit moves the failure, it doesn't
remove it.**

**R2-B1a (blocker facet) — the R1 pin was a tautology.**
test_truncation_cannot_delete_top_ranked_fact asserted
`"TechCorp" in out` while every filler contained "TechCorp" — the pin
passed BECAUSE OF the failure state (current fact truncated to
"Rac"), the second time this stage a replacement pin matched its
mutant. **[FALSIFIED R2 — the R1 record sentence "Pinned: …asserts
surviving text at budget 300 AND through the real assembler at
semantic=100" was false on both halves: the assembler half asserted
a shared brand token, and at semantic=100 the current fact did NOT
survive.]** FIX: the fixture needle ("Zephyrine Analytics") now
exists ONLY in the rank-0 fact; the assembler half asserts the FULL
fact text; and a NEW sweep pin
(test_rank0_survives_across_swept_budgets) asserts survival at all
58 swept semantic budgets 60→1200 — one budget value proves nothing.
Lesson minted: **write the pin's needle so it exists only in the
thing you are protecting.**

**[CORRECTED R2 — M2] The R1 fix-pass record cited the clean
diagnostics rerun in its closing evidence.** G2 runs at the DEFAULT
allocation (15,360 tokens ≈ 61,440 chars) against a store whose
largest section is 3,192 chars — it structurally cannot reach B1's
regime, so a clean G2 was never evidence about B1. Struck from the
fix-evidence chain; G2's actual claims (no-regress byte-identity,
cross-session reach, Part D measurement) stand on their own and were
independently reproduced by the critic, exactly, in a third window.

Minors landed: **m1** break-not-continue now PINNED
(test_fill_stops_at_first_nonfit_no_leapfrogging — pins the fill
mechanism under a controlled rank order, not TF-IDF coincidences).
**m2/N8** _predecessor_targets scope filter now PINNED
(test_predecessor_scope_filter_blocks_cross_scope_leak — raw
cross-scope superseded_by write; ShadowCorp must not leak). **m3**
_sanitize hardened + honestly re-scoped: zero-width chars stripped
(ZWSP inside the marker bypassed \s+ while rendering visually
identical), bracket homoglyphs added to the marker class, inline
[YYYY/MM/DD] stamps demoted to parentheses so a mid-line stamp
cannot impersonate the authoritative leading stamp; docstring now
says LINE forgery is what died and names the disclosed residual —
content-level lies are extraction-validation's problem, not the
renderer's. Pinned: test_render_forgery_residuals_neutralized.
**m4** both stale "10 turns" comments fixed. **m5** the fake redis
pipeline now BUFFERS until execute() like redis-py. **n1** fixed AND
pinned: repopulation is skipped when the cache already holds exactly
this answer (short sessions were paying delete+N×lpush+ltrim per
assemble to rewrite identical data —
test_repopulate_skipped_when_cache_already_correct); the
shorter-than-last_n never-hits residual stays disclosed in the code.
**n2** the replace_history one-call staleness window disclosed in its
docstring (SQLite authoritative; cache self-heals). **n3** the
pre-existing pool-exhaustion find (assembler-per-recall ×
ConversationStore-per-assembler holds a pooled session for life;
critic exhausted QueuePool at 116 assemblers) is OUT of Stage 5's
lane — ledgered as open item #27, founder-visible.

Post-fix: **35 passed + 1 opt-in skip** in tests/test_fact_retrieval.

### Stage 5 — G3 round 3: BLOCKED (0 blockers, 2 majors, 5 minors/notes) — fix pass record

**R2-B1 and R2-B1a were CLOSED by the critic**: its own 115-budget
sweep re-run against the real assembler found 0 rank-0 losses (was
95/115), 0 blocks over token budget, and 0 blocks tripping
_fit_to_budget's char fast path (the mirror hole it went hunting —
rendered blocks at 3.7–3.8 chars/token never reach the ×4 gate).
Mutations MA (count→len//4) and MD (budget×4 revert) both died on the
sweep pin. Determinism now spans FOUR independent windows.

**R3-M1 (major) — my break-not-continue pin was tautological. The
THIRD tautological pin this stage.** Gamma was excluded by the BUDGET
(a+c = 38 tokens > budget 30), not by the mechanism — the critic
measured the discriminating band ([38..43+]) and the test sat outside
it. **[FALSIFIED R3 — the R2 fix-record sentence "m1
break-not-continue now PINNED" was false: mutation N2 stayed green.]**
FIX: budget moved into the band (45) AND a POSITIVE CONTROL added —
the test asserts a+c genuinely fits the budget (TokenCounter on the
rendered lines) so gamma's absence can only be the break. Lesson
minted (critic's words): **a pin whose fixture is excluded by the
BUDGET, not by the mechanism, pins nothing.**

**R3-M2 (major) — the R2 unit fix cost 20× on the read path,
undisclosed.** Re-tokenizing the accumulating block per candidate was
O(n²): measured 1150ms of pure tokenization at the module's OWN
500-fact scan cap, inside synchronous recall. FIX: incremental fill —
per-line token counts + 1 per join as the running estimate, ONE exact
count only at the boundary, and the post-sort trim as the exact
guarantee regardless of estimate error. RE-MEASURED at the cap:
**24.3ms at budget 15360 (47× faster), 17.6ms at qa-eval's 4740,
9.6ms at 1000; block tokens ≤ budget at every point.** Lesson
(critic's words): **enforcing a budget in the right unit can cost you
an order of magnitude — measure the fix's cost, not just its
correctness.**

Minors landed: **m1** the post-sort trim is now honestly labelled —
with real o200k_base content [CORRECTED R6: originally written
"cl100k" from the critic's R3 log; TokenCounter's gpt-4o default
resolves to o200k_base — its own n3 correction] it fired at 0/115
budgets and 0/400 random
orderings (fill estimates are per-line sums real joins never
exceeded), so it is documented as the exact-guarantee mechanism
reachable when estimates UNDER-count, and its drop order is PINNED by
a constructed inflating-counter test
(test_post_sort_trim_drops_lowest_ranked_never_newest — the critic's
MB/MC mutants now have a killing test instead of an unreachability
excuse). **m2** the oversized-rank-0 exception now states its
measured magnitude (15.9× at token_budget=20) in the docstring.
**m3** _INLINE_STAMP_RE's rewrite of LEGITIMATE bracketed dates is
documented by test (honest text: [2024/03/15]→(2024/03/15),
unbracketed dates untouched). **n1** counter-unit parity pinned
(test_token_counters_share_the_unit — the whole B1 fix rests on
retriever and assembler counting in the same unit). **n2** the
critic confirmed the cached-NameError edge closed.

Post-fix: **38 passed + 1 opt-in skip**; **255 regression green**.

### Stage 5 — G3 round 4: BLOCKED (1 blocker, 0 majors, 1 note) — fix pass record

R3's M1 and M2 were CLOSED by the critic: N2 now dies (budget in the
discriminating band + real positive control); the perf fix re-measured
at 20.3/16.2/10.6ms (same class as our 24.3/17.6/9.6 — machine noise),
~56× better than R3's 1150ms, curve no longer quadratic. MB/MC/P4 all
die on the constructed-counter trim pin. The four incremental-estimate
mutants (P1/P2/P3/P5) were proven EQUIVALENT BY MEASUREMENT — the
critic ran both variants differentially over 58 budgets × 201 facts:
0 output differences, 0 over-budget, 0 rank-0 losses, 0 recall cost.
"Prove equivalence by measurement, not by argument" — its lesson, now
ours too.

**R4-B1 (blocker) — B1's THIRD incarnation, open since the stage's
first commit: _fit_to_budget cuts TWICE, and the first cut is in
CHARS.** The char fast path (len > tokens×4, keep="head") runs BEFORE
the token check; ordinary long-common-word English prose measures
~5.9 chars/token, so a block PERFECT in tokens was still head-cut —
rank-0 lost at 9/9 budgets including qa-eval's 4740 and the default
15360. Neither the suite nor the critic's own R3 sweep could see it:
every fixture in play sat at ≤3.88 chars/token — the fixture's
chars/token ratio was a HIDDEN PARAMETER of every sweep ever run.
(The critic corrected its own R3 record on this: "the mirror hole
does not occur" was true of a fixture and stated as true of the
class.) This is the S4-R8 alternation lesson in a new shape: four
rounds hardened one branch of a two-branch cut and never touched the
other.

FIX — close the alternation: build_block now satisfies BOTH
constraints — tokens ≤ token_budget AND chars ≤ token_budget ×
_CALLER_CHAR_FACTOR(=4) — chars tracked exactly in the fill (len is
free), _trim_to_budget enforcing both units post-sort. The coupling
to the assembler's factor is named at the constant with its tripwire.
PINNED: test_rank0_survives_high_ratio_content — the fixture ASSERTS
its own ratio > 4.5 before asserting survival (the hidden parameter
made explicit, so it can never silently drift back under the
threshold), then sweeps the critic's exact 9 failing budgets through
the real assembler. Also pinned per the R4 note:
test_count_calls_stay_linear guards the O(n) property itself (P2 was
output-equivalent but silently reverted the whole perf fix — guard
the call count, not the wall clock).

Post-fix: **40 passed + 1 opt-in skip**; **255 regression green**.

### Stage 5 — G3 round 5: BLOCKED (0 blockers, 1 major, 3 notes) — fix pass record

**R4-B1 CLOSED by the critic**: both content regimes swept at 115
budgets each (high-ratio 5.91 and sub-4 at 3.99) — 0 rank-0 losses, 0
char-gate firings, 0 token overruns; the exact R4 9/9-failing content
now survives at all nine budgets. q3 (factor 4→8) and q4 (full
revert) both die on the new pin; q1/q2 (either single char mechanism
removed) proven redundant-but-covered BY MEASUREMENT (identical
output, q4 dies). The critic called the self-asserting-ratio fixture
"a better fix than I asked for". Its fresh-eyes hunt for a FOURTH
incarnation across every cut on the path found none (label wrapping,
section join, model_window scaling, unit assumptions — all cleared;
one downstream note below).

**R5-M1 (major) — the O(n) guard guarded nothing: my own char break
masked it.** The fixture sat at 3.83 chars/token, where the NEW char
cap stops the fill before the token boundary either branch — P2 made
IDENTICAL calls (5 vs 5). The fifth same-side-of-threshold fixture
this stage, and the second caused by the chars/token hidden
parameter. **[FALSIFIED R5 — the R4 fix-record sentence
"test_count_calls_stay_linear guards the O(n) property itself" was
false: P2 stayed green.]** P2 is NOT equivalent (critic-measured
2-3× calls and 2× fill admissions on low-ratio content; outputs
equal at every budget — coverage defect, not user-visible). FIX: the
fixture is now digit-dense low-ratio content that SELF-ASSERTS both
hidden parameters — ratio < 2.5 (token boundary decides, not chars)
AND all 50 lines fit under the char cap (under P2 nothing stops the
fill before the facts exhaust). Measured on the final fixture:
correct branch 19 calls, P2-emulated 119-129 calls [CORRECTED R6:
my emulation measured 129, the critic's two plausible emulations
both measured 119 — the only number this stage that didn't reproduce
exactly; conclusion unaffected, bound 45], bound 45 sits
between with ≥2× margin on both sides. First sizing (60 lines) was
caught by the fixture's OWN self-assert (7560 > 6800) — the
self-asserting pattern paying for itself within the hour. Lesson
minted (critic's words): **a new guard can MASK the mechanism an
older test was written to observe.**

Notes landed: **n1** the factor's safe-ratio dependence on
TokenCounter's ENCODING (gpt-4o → o200k_base) is now named at the
constant. **n2** qa_accuracy_eval's own 24,000-char post-assembly cut
is a HEAD-keep with facts first — facts safe, RECENT TURNS can be
crowded out at that cap; recorded here as the disclosure the
affected-surfaces list owed. **n3** the critic corrected its own
record: every chars/token figure this stage is an o200k_base
measurement, not cl100k as its R3/R4 logs said — counter parity
unaffected (both sides construct TokenCounter() identically, and
that parity is pinned).

Post-fix: **40 passed + 1 opt-in skip**; regression unchanged (255).

### Stage 5 — G3 round 6: **PASS-WITH-NOTES** (0 blockers, 0 majors, 6 notes — all record/comment hygiene, three the critic's own)

P2 died on the rebuilt guard (119 > 45); both hidden parameters
measured (ratio 1.24, char headroom 8% failing LOUDLY on the
self-assert); separation stable across six budgets (correct 16-27 vs
mutant 103-121 — tokenizer drift would need to move one branch ~65%
to break the bound). The critic's verdict, verbatim: **"I am passing
on substance, not fatigue: the last correctness sweep I could
construct came back clean in both regimes, the fresh-eyes hunt for a
fourth truncation on the path found none, and the pin I blocked on
last round now kills its mutant with a 2.4x margin across a budget
band."** All six hygiene notes were landed before this closing entry
(retroactive [FALSIFIED R4] strike; assembler-side coupling comment;
three o200k corrections; the 119/129 reconciliation; the diagnostics
caps line now names the char cap; this CURRENT-STATE block is note
#6).

---

## STAGE 5 — HONEST CLAIMS OF RECORD (definitive; quote nothing about this stage without these)

**What ships:** facts-first retrieval wired into the product —
`llm/fact_retrieval.py` (lexical-primary TF-IDF in the measured
champion configuration + entity-linked recall floor through Stage 3's
join table with full ALIAS_OF closure; rank fills a DUAL-unit budget
— tokens AND the caller's ×4 char gate — chronology orders only the
survivors; transition lines render the change story; render
vocabulary sanitized), the assembler's `[SEMANTIC FACTS]` section
(facts claim the 20% semantic allocation first, chunks take the
remainder), MCP `consolidate_session` (ConsolidationV2's first
product caller, event-loop-safe) + scope passthrough on
recall_memory, and `benchmarks/facts_first_diagnostics.py` ($0,
env-isolated, exit-coded).

**Mandatory disclosures:**
1. ZERO facts in store ⇒ assembled output BYTE-IDENTICAL to the
   pre-Stage-5 assembler (pinned incl. chroma-call equality; the
   banked 66.0%/0.952 flow through this exact code path).
2. Rank-0 survival under budget pressure is proven at 115 budgets ×
   TWO content regimes (3.99 and 5.91 chars/token) + the nine
   R4-failing budgets — after THREE incarnations of the same
   truncation bug (chars-as-proxy, wrong unit, char fast path) each
   found only by adversarial rounds. Disclosed exception: a single
   oversized rank-0 line at tiny budgets (15.9× at 20 tokens) is
   tail-truncated by the caller — rank-safe.
3. Undated facts (structurally most preferences/identity) rank
   through the entity floor ONLY once the 500-fact lexical scan cap
   binds — ledger item #25.
4. Real-corpus G2: the current answer ('TechCorp') surfaces as a
   dated fact where session-scoped raw-turn retrieval STRUCTURALLY
   cannot reach it (the answer lives in another session) — including
   from an entirely unrelated session (facts-only=True). That is the
   fact tier's claim; it is NOT a retrieval-quality comparison.
5. The Rachel supersession did NOT fire in this stage's windows
   (gates open, temp-0 LLM declined; fired in Stage 4's window,
   critic-reproduced 4×) — Ollama temp-0 determinism holds WITHIN a
   machine state (six consecutive byte-identical diagnostic runs
   across critic and builder), not across windows. Retrieval
   surfaces the current answer either way; supersession only adds
   the change-history annotation.
6. TWO real pre-existing bugs found by G2's byte-identity check:
   Redis ghost keys (CONTAINED by AGENTMEM_OS_DISABLE_REDIS
   kill-switch, forced in tests/diagnostics; root cause — no DB
   identity in keys — OPEN, ledger #23) and warm-cache depth halving
   (FIXED at root: max_turns 20, replace-not-append repopulation,
   depth contract, skip-if-identical). Banked benchmark numbers owe
   a Redis-disabled re-verification at Gate C — ledger #24.
7. Render sanitization neuters this renderer's MACHINE VOCABULARY
   (line structure incl. zero-width chars, history marker incl.
   bracket homoglyphs, inline date stamps); fact CONTENT can still
   say anything — that is extraction-validation's lane.
8. Perf at the module's own 500-fact cap: ~20-24ms per build_block
   (O(n) pinned by call-count guard). Pre-existing pool exhaustion
   on long-lived MCP servers (~15 recalls) is NOT fixed — ledger
   #27. The Indic alias fallback can still cold-load the encoder
   inside an open READ session — disclosed, WAL read lock only.

**CURRENT artifacts: 41 tests in tests/test_fact_retrieval.py (40
passed + 1 opt-in live-Redis skip) · 255-test regression green across
all seven touched surfaces · diagnostics deterministic and
byte-identical across SIX consecutive runs (builder ×4, critic ×2+)
· G3: six rounds, blocker trajectory 3→1→0→1→0→0, PASS-WITH-NOTES.**

Stage table: Stage 5 ✅ DONE (G1 ✅ 41 tests · G2 ✅ real-corpus
facts-first retrieval + 2 real pre-existing bugs found · G3 ✅ R6
PASS-WITH-NOTES after R1-R5 each BLOCKED and fixed). Stage 6 (full
E2E + final critic pass → BUILD READY) remains.
