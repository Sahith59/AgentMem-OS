# Semantic Fact Tier — Stage 0 Implementation Research (verbatim, 2026-08-06)

Full unedited deliverable of the implementation-mechanics research pass (Stage 0 of
the build plan in `CONSOLIDATION_V2_BUILD_LOG.md`). Distillate lives in the build
log's Stage 0 record; this file preserves the complete report and sources.

Evidence-quality note: sections on Mem0 and Graphiti are based on DIRECT READS of
their source code vendored in this repo (`benchmarks/adapters/.venv-mem0/`,
`.venv-graphiti/`) — the strongest evidence tier available.

---

## 1. Fact extraction pipeline mechanics

### 1.1 Mem0's actual extraction prompt/pipeline — verified from source

Files read directly: `mem0/configs/prompts.py`, `mem0/memory/main.py` (vendored).

Mem0 shipped three generations of extraction prompt in the same file; the newest
directly contradicts the "atomic fact" philosophy:

- `FACT_RETRIEVAL_PROMPT` (v1, legacy) — 5 sparse few-shot examples, terse facts.
- `USER_MEMORY_EXTRACTION_PROMPT` — hardens attribution ("GENERATE FACTS SOLELY
  BASED ON THE USER'S MESSAGES... YOU WILL BE PENALIZED IF...").
- `ADDITIVE_EXTRACTION_PROMPT` (v3, current, ADD-only) — ~950 lines, 12 dense worked
  examples, and an explicit **"Contextually Rich, Not Atomic"** rule:
  > "Bad: 'User has a dog' | Good: 'User has a dog named Poppy and their morning
  > walks together are the highlight of their day'... This applies especially to
  > transitions and changes... Bad: 'User prefers oat milk lattes' Good: 'User
  > switched from almond milk to oat milk lattes after developing an almond
  > sensitivity'."

**Tension with our locked atomic-fact design — and why it doesn't force us off
atomicity:** Mem0 has no KG and no supersession chain reaching back to the prior
fact — transition context has nowhere else to live, so it gets crammed into the
fact text. Our design has both (`superseded_by` chain + KG entity linking), so a
transition ("switched from X to Y") is reconstructed by walking the supersession
chain (old fact → new fact, both atomic and dated). Recommendation: keep atomicity;
when a new fact supersedes an old one, synthesize a transition sentence ON READ for
retrieval-surface convenience only ("{old.text} → superseded {t_occurred} by:
{new.text}"), never as the row of record. Founder decision at design freeze.

**Temporal grounding to copy verbatim from Mem0 v3:** it distinguishes Observation
Date (when the conversation happened — the ONLY valid anchor for resolving
"yesterday"/"last week") from Current Date (system date, explicitly NOT for
resolving relative references). Rule to steal: "Always ground relative references
to specific dates... 'User went to Paris last week' is useless 6 months later."

**Dedup/merge in the real pipeline (`_add_to_vector_store`, main.py ~849-1130):**
1. Vector search top_k=10, NO similarity threshold — candidates shown to the LLM as
   context for LINKING (`linked_memory_ids`), not merging.
2. Single LLM call, ADD-only. No ADD/UPDATE/DELETE decision in the live path.
3. Exact-only dedup: MD5 of normalized fact text vs existing + in-batch hashes.
   Anything not byte-identical inserts as a new row.
4. Entity-level linking uses a hard 0.95 cosine cutoff (entities only, not facts).

**Confirmed dead code:** `DEFAULT_UPDATE_MEMORY_PROMPT` / `get_update_memory_messages()`
(the old ADD/UPDATE/DELETE/NONE adjudication) are NOT called anywhere in the current
`add()` path — only in the explicit user-invoked `update()` API. Code-level
confirmation of the ADD-only regression (beyond issues #4896/#4956/#5867).

Net: Mem0's only automatic fact-merge today is exact-hash matching.

### 1.2 Mem0 GitHub issues — real-world failure modes (fetched directly)

| Issue | Finding |
|---|---|
| #4896 | Docs claim "latest truth wins"; code only MD5-dedups. Contradictory name facts both persist. |
| #4956 | "Works at Company A" (stale) and "Works at Company B" coexist; retrieval can rank stale first. |
| #5867 | Preference flip → two ADD rows, no UPDATE/DELETE path. |
| #4573 | Production audit of 10,134 entries: **97.8% junk** — 52.7% system-prompt restating, 11.5% heartbeat noise, 8.2% architecture dumps, 5.2% hallucinated profiles, 2.1% privacy leaks. Dominant failure: extraction over-triggering on non-user content. |
| #6531 | TOCTOU race: concurrent add() calls both pass the pre-insert dedup check → silent duplicates. Fix direction: per-scope mutex + re-fetch hashes at insert. |
| #5330 (UNVERIFIED proposal, unmerged) | No expiration → retrieval pollution; proposed decay `min(access,255)*0.5^(days/7)`, threshold 0.05. |

### 1.3 Graphiti's contradiction detection — exact mechanics from source

Files: `edge_operations.py`, `prompts/dedupe_edges.py`, `prompts/extract_edges.py`.

One LLM call + deterministic temporal post-processing:
1. Candidate generation: `related_edges` (same node pair ∩ hybrid BM25+embedding
   search) = duplicate candidates; broader hybrid search minus those = invalidation
   candidates.
2. Single LLM call returns `EdgeDuplicate{duplicate_facts, contradicted_facts}`.
   Hard prompt rule verbatim: "NEVER mark facts as duplicates if they have key
   differences, particularly around numeric values, dates, or key qualifiers."
   ("Bob ran 5 miles Tuesday" vs "3 miles Wednesday" → neither.)
3. **Deterministic temporal gating** (`resolve_edge_contradictions`): the LLM's
   contradiction judgment is NOT directly acted on — Python checks valid_at/invalid_at
   overlap; temporally disjoint facts are left alone even if the LLM flagged them.
   Only a chronologically older live candidate gets `invalid_at = new.valid_at`.
4. Out-of-order ingestion handled symmetrically (a newer candidate expires the NEW
   edge instead).
5. `expired_at` (transaction time) tracked separately from `invalid_at` (valid
   time); nothing hard-deleted. Literal Snodgrass valid/transaction split in code.

**This is the exact pattern for our `superseded_by`:** LLM proposes on semantic
grounds; a zero-LLM temporal check gates the action. Our `kg_edges` deterministic
supersession is already the deterministic half; `semantic_facts` needs the
LLM-judgment half for non-structural attribute matches.

### 1.4 Atomicity criteria — OpenIE / FActScore

- SAOKE/OpenIE: atomic fact = minimal self-contained proposition, one predicate,
  all entities and temporal references explicitly instantiated, coordinated
  predicates split. (Survey-level sourcing.)
- FActScore (Min et al. 2023): atomic facts = "short statements containing a single
  piece of information"; LLM decomposition enforcing independence + minimality.
  Its assumption that the reference KB has no conflicting facts is exactly what our
  supersession chain enforces — extraction-quality evals must treat superseding
  pairs as valid, not as bugs.
- **Actionable:** use FActScore-style independent decomposition as the extraction
  QA method — judge model decomposes source turns independently; measure precision
  (our facts are real claims) AND recall (we captured the judge's facts). Stronger
  than number-preservation alone; catches over-extraction/hallucination.

### 1.5 LongMemEval's own techniques (arXiv:2410.10813, ICLR 2025)

1. Session decomposition into rounds — helped GPT-4o reader, neutral for 8B reader.
2. **Fact-augmented key expansion** — extracted facts appended to the retrieval KEY
   (not content) → multi-pathway matching. Paper attributes ~9.4% recall@k and
   ~5.4% accuracy improvement substantially to this (per-technique split
   approximate — re-check the paper's tables before quoting externally).
3. **Time-aware query expansion** — index values by contained event dates; extract
   a time range from time-sensitive queries; filter before ranking. **+11.3% recall
   (round values) / +6.8% (session values) on temporal subset.**

(2) is essentially free once `semantic_facts` exists. (3) is a concrete cheap
addition — near-mandatory given our diagnosed weak categories: extract
`[t_occurred_min, t_occurred_max]` from the query (LLM or regex for common
patterns), apply as SQL predicate before ranking.

## 2. Deduplication and merge

### 2.1 Core finding
No production system merges facts on an embedding-cosine threshold alone. Embedding
similarity builds a candidate shortlist (generous threshold); the merge decision is
gated by a deterministic rule or an LLM call — never the raw score.

| System | Candidates (wide) | Merge gate (narrow) |
|---|---|---|
| Graphiti nodes | cosine ≥0.6 (loose) | exact string → MinHash/Jaccard ≥0.9 → LLM residual |
| Graphiti edges | hybrid RRF search | LLM call → deterministic temporal gate |
| Mem0 facts | top_k=10, no threshold | MD5 exact-hash only |
| Mem0 entities | top_k=1 | hard 0.95 cosine, no LLM (short names only) |

### 2.2 Graphiti's entity dedup cascade (from `dedup_helpers.py`)
1. Exact normalized-string match.
2. Entropy-gated fuzzy: names <6 chars / <2 tokens / Shannon entropy <1.5 skip to
   LLM; else 3-gram shingles → 32-perm MinHash → LSH banding (band 4) → Jaccard,
   merge ≥0.9.
3. LLM escalation only for the residual.

**Translation to semantic_facts dedup (the cascade to build):**
1. Tier 0 — exact/near-exact normalized-hash (free; keep).
2. Tier 1 — **numeric/date hard-guard**: regex-extract numbers/dates from both
   facts; if the sets materially differ, HARD-REJECT the merge regardless of
   similarity. Encodes Graphiti's prompt rule as a zero-cost deterministic rule.
   **Highest-leverage addition of this research pass** — a false merge of
   "three times Sept 24" with "twice Sept 24" silently corrupts the exact counts
   the entire v2 thesis depends on.
3. Tier 2 — embedding candidate shortlist scoped to same alias-resolved entity ±
   fact_type, generous threshold (~0.6), shortlist not decision.
4. Tier 3 — ONE batched LLM adjudication per session (not per-fact; cheaper than
   Graphiti's per-edge calls; session-end batching makes this natural), mirroring
   `dedupe_edges.resolve_edge`'s output contract + few-shot examples.
5. Tier 4 — deterministic temporal gate before writing `superseded_by`, mirroring
   `resolve_edge_contradictions` (only act if candidate genuinely precedes the new
   fact for the same attribute bucket).

### 2.3 Re-affirmation
No audited system has a clean shipped answer. Recommendation: on Tier 0/1 match, do
NOT insert a row — increment `mention_count`, update `last_confirmed_at`, append
new source turn ids to citations. **Requires adding `mention_count` and
`last_confirmed_at` to the §5.1 schema (currently absent).** Distinguishes "said
once" from "confirmed 15×" (retrieval confidence) and prevents the bloat Mem0 users
now propose decay-cleanup for (#5330) and Letta needs a defrag subagent for.

## 3. Bi-temporal schema patterns

### 3.1 Terminology precision
Valid time (true in reality) vs transaction time (recorded in DB) — Snodgrass,
SQL:2011. Our mapping: `t_occurred` = valid time; `t_mentioned` = a second
conversational valid-time axis (when talked about); `t_ingested` = transaction
time. **This is TRI-temporal, not classically bi-temporal — say "tri-temporal" in
external docs; Zep's paper owns "bi-temporal" for a two-time model and loose usage
next to them is a claim risk.**

### 3.2 SQLite indexing (verified at sqlite.org/partialindex.html)
Partial indexes since 3.8.0. **Planner gotcha: term matching, not theorem-proving —
the query's WHERE must contain the index's predicate LITERALLY** (`WHERE
superseded_by IS NULL AND entity_id = ?`) or the index is silently unused (slow, not
wrong — easy to miss until the table is big).

```sql
CREATE INDEX idx_facts_current
  ON semantic_facts(entity_id, fact_type, t_occurred)
  WHERE superseded_by IS NULL;
CREATE INDEX idx_facts_valid_range
  ON semantic_facts(entity_id, t_occurred, t_ingested);
CREATE INDEX idx_facts_source_session ON semantic_facts(source_session_id);
```

`kg_edges` has the temporal columns but NO index today — retrofit the equivalent
`WHERE valid_until IS NULL` partial index while building.

## 4. Local-LLM extraction reliability

### 4.1 Ollama mechanism (verified at ollama.com/blog/structured-outputs)
`format` parameter accepts full JSON Schema → grammar-based constrained decoding
(FSM masks invalid tokens): STRUCTURAL guarantee (shape), not semantic (values).
Best practice: temperature=0 + still say "return as JSON" in the prompt.
**Mem0's production call does NOT use this** — loose json_object mode + regex
fallback parser. Our local extractor can be strictly better than Mem0's shipped
pipeline here, for free.

### 4.2 Reliability findings (directional; some UNVERIFIED at primary level)
- 8B-class beats 3-4B-class on structured reliability (consistent with our Gate B:
  qwen14b 90.1%, llama3.1-8b 91.2%).
- arXiv:2605.02363 ("When Correct Isn't Usable", small-model structured output):
  small models often produce semantically-correct content in structurally-invalid
  JSON. Mitigations by effectiveness: grammar-constrained decoding; few-shot >
  zero-shot; self-consistency; retry-with-refined-prompt. (Exact numbers not
  extracted — re-read before quoting.)

### 4.3 Recommended extraction pipeline
1. Ollama `format` = SemanticFact Pydantic `model_json_schema()` — kills the
   malformed-JSON class.
2. Dense few-shot mirroring Mem0's own v1→v3 evolution (~12 worked examples:
   multi-topic, negation, transitions, numeric content).
3. **Semantic post-hoc validator** (not just parse-retry): every fact has
   t_mentioned; t_occurred parses if present; fact text contains a digit/date token
   when the source did (catches dropped counts). One correction retry on failure,
   then the existing deterministic fallback path.
4. **Targeted self-consistency:** only for facts containing numbers/dates — extract
   twice, flag if numeric tokens disagree. Concentrates compute exactly on the
   hard-core taxonomy (21 aggregation + 6 date + 7 relative-date).

## 5. Production failure modes → adversarial test cases

| # | System | Failure | Source | Our test |
|---|---|---|---|---|
| 1 | Mem0 | Contradictions persist as separate rows | #4896/#4956/#5867 | "works at A" then "works at B" → exactly one live fact + chain |
| 2 | Mem0 | 97.8% junk; over-trigger on system/tool noise | #4573 | sessions with tool output/boilerplate → zero facts extracted |
| 3 | Mem0 | TOCTOU race → silent duplicates | #6531 | concurrent consolidation on same scope → mutex + DB unique constraint as final authority |
| 4 | Mem0 | No expiration → retrieval pollution | #5330 (proposal) | mention_count/last_confirmed built in from day one |
| 5 | FalkorDB | Ingest step timeout kills pipeline mid-write | FalkorDB#1826 | full per-session consolidation in ONE transaction, no per-fact autocommit |
| 6 | Graphiti | Per-activity LLM-call cost blowup at scale | community (UNVERIFIED specific issue) | batched per-session calls (extraction once, dedup once) |
| 7 | Letta | Memory bloat needs defrag subagent + /doctor | letta docs | supersession + mention_count correct from start |
| 8 | Letta | Pagination duplicate rows | #3088 | supersession × pagination window test (no double-count) |
| 9 | third-party Zep usage | Same entity as two nodes on resolution miss | MicroFish-En#143 (weak tier) | same-language near-duplicate entity mentions in adversarial set (ALIAS_OF handles cross-lingual; "Chennai/China 0.9010" known hard negative) |

## Summary — action items against CONSOLIDATION_V2_DESIGN.md

1. Schema: add `mention_count`, `last_confirmed_at`.
2. Build the 5-tier dedup cascade (never a lone cosine threshold).
3. Numeric/date hard-guard FIRST — protects the aggregation thesis.
4. Add time-aware query expansion to retrieval (paper-measured on our weak spots).
5. Partial indexes (+ retrofit kg_edges); queries must literally match predicates.
6. Ollama schema-constrained extraction + dense few-shot + semantic validator +
   targeted self-consistency.
7. Concurrency: per-scope write serialization + DB unique constraint.
8. Design fork flagged: atomic vs contextually-rich → keep atomic + read-time
   transition synthesis (founder decision at freeze).
9. Terminology: "tri-temporal", not "bi-temporal", externally.

## Sources

Primary (read directly, vendored in this repo): mem0 `configs/prompts.py`,
`memory/main.py`; graphiti_core `edge_operations.py`, `dedup_helpers.py`,
`node_operations.py`, `prompts/dedupe_edges.py`, `prompts/extract_edges.py`; this
repo's `db/models.py`, `db/entity_aliases.py`, `llm/consolidation_engine.py`.

Web: arXiv 2410.10813 (LongMemEval); FActScore (Min et al. 2023); OpenIE (Wikipedia
survey); sqlite.org/partialindex.html; Martin Fowler, Bitemporal History;
ollama.com/blog/structured-outputs; arXiv 2605.02363; Mem0 issues #4896 #4956 #5867
#4573 #6531 #5330; FalkorDB #1826; Letta memory docs + #3088; MicroFish-En #143
(weak tier, flagged).
