# Consolidation v2 — Episodic→Semantic Distillation
### The representation fix for the measured bottleneck (design doc, 2026-08-05)

## 0. Plan of record (live status — updated every step; the long-running tracker)

**End goal:** a strong, honest, publicly inspectable baseline (beat 66.0%, contextualize
vs TiMem 76.88 and Zep 71.2) → THEN the Sarvam pivot (cross-lingual proof → sarvam-mcp
fixes → issue #83 notebook → IndicMem). Nothing Sarvam-facing ships before the baseline
is solid. Companion docs: `CONSOLIDATION_V2_RESEARCH.md` (full research report),
`COMPETITIVE_ANALYSIS.md` §4b, `RUNNING_NOTES.md` (day-to-day).

| # | Step | What it proves / builds | Cost | Status |
|---|---|---|---|---|
| 1 | Diagnosis | Representation is the bottleneck (10 refutations, hard-core taxonomy) | ~$5 (slices) | ✅ DONE 08-05 |
| 2 | Design + research | This doc; biology grounding; prior-art audit (TiMem found); claim boundaries | $0 | ✅ DONE 08-05 |
| 3 | Gate A — facts contain the answers | Paid haiku extraction assembles hard-core answers raw turns can't; 8× denser | $0 | ✅ PASSED |
| 4 | Gate B — local extractor quality | qwen2.5:14b 90.1% / llama3.1 91.2% number-preservation vs paid extraction | $0 | ✅ PASSED ×2 |
| 5 | **BUILD the true semantic tier** (reordered 2026-08-06 per founder: build first, extract through the real pipeline after) | SemanticFact table + consolidation engine rewrite (distillation, not compression) + KG fed by facts through ALIAS_OF + per-fact supersession + facts-first retrieval + tests/smoke | $0, ~1 day dev | ⏳ FOUNDER GO |
| 6 | Cluster extraction — slice haystacks (3,631 sessions) through the REAL pipeline | The dataset for the proof, in the product's real schema; university GPU cluster (~<1h) — founder provides access; portable bundle prepped | $0 | after 5 |
| 7 | Gate C-eval — 79q slice, real semantic tier as memory source | THE decision number: must beat raw turns' 0.519, else design is wrong → stop | ~$1.50 | ⏳ FOUNDER GATE |
| 8 | Gate D — full 150 re-measure | New banked number vs 66.0% (and vs TiMem 76.88, honestly) | ~$3.50 | ⏳ FOUNDER GATE |
| 9 | Gate E — cross-lingual proof | Hindi-translated sessions → same canonical facts, score holds | $0–1 | after 8 |
| 10 | Gate F — 500-question publishable run | The README number (500-haystack extraction also on cluster: ~37 Mac-hrs → ~2-4 cluster-hrs) | ~$9–12 | ⏳ FOUNDER GATE |
| 11 | Sarvam Stage-1 | sarvam-mcp fixes, #83 notebook, IndicMem scoping — only after we compete with current competitors | $0 | after 10 |

Note (08-06): proxy-extraction shortcut (old step 5, benchmark-side JSONL) retired after
48/3,631 sessions — founder reordered to build-first so extraction runs through real
product code into the real schema. The 48-session JSONL kept as reference only.
Procedural tier deliberately NOT expanded in this arc (no LongMemEval category tests
it — expanding it now would be over-engineering); profile tier is the next build after
the semantic tier proves itself at step 7.

**Model roles (fixed, to avoid confusion):** GPT-4o = benchmark ANSWERER + JUDGE,
unchanged everywhere. Local Ollama models (llama3.1/qwen2.5) = the PRODUCT's
ingestion-time fact extractor only (the consolidation engine component) — chosen local
because (a) $0, (b) AgentMem OS is local-first: if the memory pipeline required a paid
API, the product's core claim dies. Extraction model is disclosed in every artifact.

## 1. The measured problem

Ten refutations across the 2026-08-04/05 arc narrowed the LongMemEval `_s` gap to one
cause. The evidence chain, every step measured:

1. **Retrieval is saturated.** Gold-turn recall 0.967 at top_k=40 ($0 ablation harness).
2. **Coverage is not the constraint.** Raising the context budget 24k→40k chars lifts
   gold-session coverage 0.77→0.90 but accuracy is FLAT (ms 0.436, temporal 0.525).
3. **The answerer is not the constraint.** gpt-5.4-mini (newer reasoning model) scored
   0.392 vs GPT-4o's 0.519 on the identical slice — worse, net −10 questions.
4. **The hard core:** 34/79 slice questions fail under BOTH answerers. Taxonomy:
   21 aggregation, 6 date-arithmetic, 7 relative-date recall. On 22/34 even the
   reasoning model answers "not mentioned" — because **the answer exists in no single
   turn.** "You rode rollercoasters 10 times" is written nowhere; it is distributed
   across 4 sessions as 3+1+3+3.

**We store episodes and ask questions that only semantic memory can answer.**

## 2. Why 0.967 recall coexists with 0.36–0.57 accuracy

"Recall 0.967" means: for 96.7% of questions, at least one gold TURN reaches the
context. But:
- Gold **density is 0.196** — ~80% of every assembled context is noise.
- Aggregation questions need ALL evidence pieces simultaneously, then arithmetic.
- Median evidence footprint per hard-core question (measured, n=34):
  - as extracted facts: **2,392 chars** — fits the 24k budget in 34/34 cases
  - as raw gold turns: **18,152 chars** — fits in only 26/34 even with perfect retrieval
  - as the full haystack: 255,210 chars — 10× over budget
- Sessions are dated by when the user TALKED; questions ask about when events
  HAPPENED ("July to October" events discussed in a November session). Raw turns
  carry the wrong timestamp for the question being asked.

Finding one needle is solved. Assembling ten needles plus arithmetic inside a noisy
24k window is not — and no retrieval or answerer tuning fixes it (measured, ten ways).

## 3. Gate A (2026-08-05, $0): distilled facts contain what raw turns cannot surface

Using the already-paid X-MemoryArch haiku extraction (940 sessions, 4,964 memories,
`benchmarks/extracted_memories/`), for all 34 hard-core questions:

- Evidence-session coverage in the cache: **34/34 full**.
- Canonical case — "How many times did I ride rollercoasters (July–October)?",
  gold "10 times": the facts read "rode Revenge of the Mummy **three times** on
  **October 15th**", "rode Xcelerator on **October 8th**", "rode Space Mountain
  **three times** on **September 24th**", "rode Mako, Kraken, Manta in one night in
  **July**". 3+1+3+3 = 10, with event dates, in six lines. The same evidence in raw
  turns is buried mid-conversation across four ~47-session haystacks.
- Second case — "How many days a week do I attend fitness classes?", gold "4 days":
  facts enumerate Zumba (Tue+Thu), yoga (Wed), weightlifting (Sat). Assemblable.
- **Measured weakness that v2 must fix:** `session_date` is null on 100% of cached
  memories and only 38/515 fact texts carry an explicit date. Extraction must stamp
  every fact with the session date (known from source data — $0 to recover) and
  preserve in-text event dates. Without this, temporal questions stay broken.

## 4. What exists today (audit) vs what v2 must be

`llm/consolidation_engine.py` ("Sleep Consolidation Engine") already claims the
biology: hippocampus→neocortex, sleep replay, DBSCAN clustering, abstraction levels.
What it actually does:

| Today | Biology / what v2 needs |
|---|---|
| Selects the LEAST important 30% of turns | Consolidates the MOST important content |
| Goal: free tokens (compression) | Goal: build queryable semantic knowledge |
| DELETES source turns after summarizing | Episodes retained; semantic layer added alongside (complementary learning systems) |
| Output: undated prose cluster summaries | Output: dated atomic facts with typed structure |
| Triggers at 70% of 128k context — never fires in benchmarks | Runs at session end / idle, always |
| Output goes to `summaries` table that benchmark retrieval NEVER queries | Semantic tier is the PRIMARY retrieval target |

The name was right; the implementation optimizes a different objective. v2 rebuilds
the cycle around distillation, not compression.

## 5. The v2 design

### 5.1 Semantic fact tier (new table: `semantic_facts`)
```
fact_id, agent_id, user_id
text            — one atomic fact, self-contained, canonical English
fact_type       — event | state | preference | identity
t_occurred      — when the thing happened (event date if stated, else session date)
t_mentioned     — session date (always known)
t_ingested      — consolidation timestamp          [bi-temporal + audit]
source_session_id, source_turn_ids                 [citations — inspectability]
entities        — linked through the existing KG (ALIAS_OF-aware)
lang_source     — original language of the source turn
superseded_by   — nullable; per-fact supersession (generalizes SUPERSEDABLE_RELATIONS)
```
Every fact is inspectable end-to-end: fact → source turns → session. Nothing stored
that cannot be traced. This is the transparency differentiator: publish the store.

### 5.2 Consolidation cycle (rewritten engine)
At session end (or scheduled idle, "sleep"):
1. Extract atomic facts from the session's turns (LLM; local Ollama default, API
   opt-in). Aggregation-aware prompt: **preserve counts, quantities, dates, and
   schedules verbatim** — the measured failure mode of generic summarization.
2. Stamp `t_mentioned` = session date; parse in-text event dates → `t_occurred`.
3. Canonicalize: facts stored in canonical English regardless of source language;
   entity mentions resolved through the KG's ALIAS_OF edges (τ=0.90, non-destructive
   — already built and tested). A Hindi session and an English session about the
   same fact produce ONE fact with two cited sources.
4. Supersession: new state/preference facts about the same (entity, attribute) mark
   the old fact `superseded_by` — never delete (knowledge-update is our BEST category,
   0.952, precisely because supersession exists for 3 relations; v2 generalizes it).
5. Episodes (raw turns) are KEPT. Semantic tier is additive.

### 5.3 Retrieval change
Context assembler's semantic section retrieves FACTS first (dense: ~8× fewer chars
per unit of evidence, measured), raw turns as provenance/fallback. Facts carry date
stamps → chronological ordering (already built, currently neutral) becomes load-bearing.

### 5.4 Profile tier (unchanged from agreed plan)
Preference/identity facts additionally project into a per-user profile: O(1) session-
start injection, per-attribute supersession, ALIAS_OF-keyed cross-lingual unity.
Targets the preference category (competitor pattern: user-model systems 86–90 vs
retrieval-only 53–57).

### 5.5 Why this is the cross-lingual answer (the Sarvam-relevant claim)
Raw-turn lexical retrieval is structurally zero across languages (a Hindi query
shares no tokens with English turns; measured dense-retriever fallback ranks topical
chatter above needles). Canonical facts move language handling to WRITE time:
retrieval operates on one canonical store no matter the input language. This must be
proven, not asserted — Gate E below.

## 6. Validation gates (worst-case-first; every paid step pre-gated)

| Gate | What it proves | Cost | Stop rule |
|---|---|---|---|
| A ✅ done | Facts contain what raw turns can't surface (34/34 coverage; density 8×) | $0 | — |
| B | Extraction quality floor: local Ollama vs paid haiku on 20 sessions, fact-recall vs gold checklist | $0 (local) | If local extraction misses >30% of countable events → extraction model must be API (~$2-4/slice) |
| C | Representation lift on the REAL `_s` slice: extract facts for all ~3,750 slice-scope sessions (noise included — no oracle shortcut), re-run 79q slice | $0 extraction (local, overnight, parallel+checkpointed) + ~$1.50 eval | If slice doesn't beat 0.519 → design wrong, stop and rethink |
| D | Full 150 with winning config vs banked 66.0% | ~$3.50 | Founder word required |
| E | Cross-lingual: translate N slice sessions to Hindi (local model), consolidate, verify same canonical facts + slice score holds | $0-1 | If score collapses → fix before any Sarvam claim |
| F | 500-question publishable run | ~$9-12 | Founder word required |

Extraction runs are parallelized + checkpointed from day one (Graphiti lesson:
never serial multi-hour ingest again).

## 7. Biology grounding (research audit 2026-08-05 — every claim sourced)

The design decisions above are not aesthetics; each maps to a specific, sourced
mechanism — and in two places the right design is to BEAT the biology, not copy it:

- **Two stores, populated differently** (Tulving 1972; McClelland/McNaughton/O'Reilly
  1995 complementary learning systems): a fast append-only episodic store + a slow
  semantic store built FROM it by an offline distillation step. CLS theory's
  catastrophic-interference argument is also why consolidation runs as a BATCH at
  session end ("sleep replay", Wilson & McNaughton 1994) rather than per-turn at
  write time — per-turn integration into the knowledge base is the single-system
  failure mode the theory warns against (and is Mem0's architecture).
- **Gist and verbatim are co-equal parallel traces** (fuzzy-trace theory, Brainerd &
  Reyna): facts and raw turns are independent representations of the same event,
  queried by task — not a summary plus an afterthought backlink. Hence episodes are
  KEPT and both tiers stay queryable.
- **Invalidate, don't delete** (reconsolidation, Nader/Schafe/LeDoux 2000): updating
  is supersession with the old trace recoverable — `superseded_by`, never UPDATE-in-place.
- **The self is a privileged schema** (Markus 1977; Rogers/Kuiper/Kirker 1977):
  self-knowledge has a structurally distinct access path in the brain — the profile
  tier is a separate tier with its own lifecycle, not a fact_type flag.
- **Beat the biology #1 — counting** (Menon 1993): humans don't enumerate episodes to
  answer "how many times" — past a handful they switch to rate estimation and get it
  wrong (systematic underestimation). The benchmark's aggregation questions are a
  task humans FAIL. Atomic dated facts turn it into exact enumeration — the one
  advantage silicon has. We copy the architecture, not the failure mode.
- **Beat the biology #2 — dating** (Friedman 1993): humans reconstruct dates from
  landmarks, badly. Our turns carry ground-truth timestamps; consolidation resolves
  relative references ("four weeks ago") to absolute dates AT WRITE TIME.
- **Associative retrieval** (Collins & Loftus 1975; Teyler & DiScenna hippocampal
  indexing): cue-driven traversal over entity links, which the existing KG enables
  over facts. Note: "hippocampal indexing" as a framing is already claimed by
  HippoRAG (NeurIPS 2024) — cite it, don't echo it.

## 8. Prior art + what we may honestly claim (audit verdicts, binding)

- **TiMem (ACL Findings 2026, arXiv:2601.02845) MUST be reckoned with before any
  publication: published 76.88% on LongMemEval `_s` — ABOVE our 66.0% on the same
  split.** Hierarchical temporal memory tree (raw turns → summaries → persona), open
  source. Representation-level (paper read directly): free-text summaries, NOT atomic
  facts; no source citations; no explicit supersession; no cross-lingual anything.
  Action: read/run TiMem; our Gate C/D numbers get compared against it honestly.
- Nemori (arXiv:2508.03341): session segmentation + free-text "insights",
  whole-insight replace, no citations, no cross-lingual. SeCom (ICLR 2025): evidence
  that turn-, session-, and summary-granularity all underperform segment-level —
  supports finer-grained units, doesn't implement atomic facts.
- Zep ALREADY SHIPS bi-temporal edges (valid_at/invalid_at, invalidate-don't-delete)
  AND fact→episode provenance (their engineering blog). Mem0 designed ADD/UPDATE/
  DELETE supersession but OSS v3 REGRESSED to ADD-only (open issues #4896/#4956/#5867)
  — contradictions currently accumulate. Letta sleep-time agents: separate background
  process, but free-text memory blocks, not structured facts.
- **Claims we may publish:** (1) no surveyed consolidation system stores distilled
  output as structured, individually-dated, individually-cited ATOMIC facts (all use
  narrative text or graph edges); (2) no single system combines a separate,
  correctly-functioning bi-temporal per-attribute profile tier (Zep: no separate
  tier; Letta: free-text; Mem0: broken in shipped OSS); (3) "no publicly documented
  competitor claims write-time cross-lingual canonicalization of consolidated facts"
  — worded exactly so, re-verified against non-English competitor docs before
  publishing.
- **Claims we must NEVER make:** "we invented consolidation" (Nemori/SeCom/TiMem),
  "first bi-temporal memory" (Zep's paper title), "first with provenance" (Zep ships
  it), "first per-attribute supersession" (Mem0's design intent).
- Design axis we currently lack (noted, not scoped): Hindsight's separation of
  subjective opinion (with confidence) from objective fact.

## 9. Honesty rules (unchanged, binding)
- Extraction model disclosed in every artifact (`extraction_model` field).
- No label-conditioning: consolidation sees turns, never benchmark question types.
- Two-table publishing: ours-measured vs vendor-published, never merged.
- The haiku cache is used ONLY for oracle-split diagnostics (it covers only evidence
  sessions); every `_s` number comes from extraction over the FULL haystack.
