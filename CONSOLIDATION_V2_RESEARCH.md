# Consolidation v2 — Research Report (verbatim, research agent, 2026-08-05)

Full unedited deliverable of the biology + prior-art research pass. The
decision-relevant distillate lives in `CONSOLIDATION_V2_DESIGN.md` §7–8 and
`COMPETITIVE_ANALYSIS.md` §4b; this file preserves the complete report with all
sources so nothing exists only in a chat transcript.

---

Grounded in: `AgentMem-OS/RUNNING_NOTES.md` and `AgentMem-OS/COMPETITIVE_ANALYSIS.md` (both read before researching). Current measured state: LongMemEval `_s` 66.0% overall, knowledge-update at ceiling (0.952), multi-session 0.462 and temporal 0.575 well below ceiling, gold-turn retrieval recall 0.967 — confirming the diagnosis in the prompt: raw-turn TF-IDF retrieval is not the bottleneck, representation is. All claims below are sourced; anything I could not verify is marked **UNVERIFIED**.

## 1. Biology of human memory — mechanisms that matter here

### 1.1 Episodic vs. semantic memory, and how semantic forms FROM episodes

**Neuroscience.** Tulving's original distinction: episodic memory is memory for personally experienced events located in time and space; semantic memory is context-free general knowledge, facts, and concepts (Tulving, 1972, "Episodic and Semantic Memory," in *Organization of Memory*, Academic Press). Semantic memory is not populated by a separate route — it is widely held to be abstracted out of accumulated episodes as their specific contexts fade and their shared regularities are retained (the "semanticization" view; the complementary-learning-systems account in 1.2 formalizes it computationally).

**Architectural implication.** AgentMem OS currently has one tier (raw turns, retrieved by TF-IDF). Biology says you need two functionally distinct stores that are POPULATED differently: an append-only episodic store (what happened, when, verbatim) and a semantic store that is *built from* the episodic store by an explicit distillation step, not queried the same way. This names the proposed consolidation engine as biologically correct in structure, not just a QA-accuracy hack.

Source: Tulving 1972, cited via Springer (10.3758/s13421-022-01299-x).

### 1.2 Systems consolidation: hippocampus → neocortex, sleep replay, complementary learning systems

**Neuroscience.** McClelland, McNaughton & O'Reilly (1995, *Psychological Review*, "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex") formalized *why* the brain needs two systems, not one: a single fast-learning system that integrates every new experience directly into a structured knowledge base destroys existing structure — **catastrophic interference**. The hippocampus is a sparse, pattern-separated, fast-learning system for episodes; the neocortex is a slow, distributed system that gradually extracts statistical regularities across many episodes. Memories are hippocampally stored first, then reinstated and slowly "taught" into neocortex over repeated replay — empirical support: Wilson & McNaughton (1994, *Science*): place-cell ensembles active during a spatial task were replayed, in correct temporal order, during subsequent slow-wave sleep.

**Architectural implication.** The strongest single justification for the design: consolidation should be a **separate, offline, batch process** — triggered at session end, not streaming per-turn. Per-turn extraction at write time (Mem0's model) is closer to a single fast-learning system integrating directly — the failure mode catastrophic-interference theory warns against. A batch job at session boundary is the closer biological analog to sleep-triggered replay-based consolidation.

Sources: PubMed 7624455; Science 10.1126/science.8036517.

### 1.3 Gist vs. verbatim traces (fuzzy-trace theory)

**Cognitive science.** Brainerd & Reyna: verbatim traces (exact surface form) and gist traces (semantic essence) are encoded **in parallel, independently**, from the same experience. Different forgetting rates, different retrieval signatures; both persist as separate stores.

**Architectural implication.** Validates "citations to source turns" but reframes it: the raw turn (verbatim) and the atomic fact (gist) are **co-equal, independently stored representations of the same event**, queried differently by task. The current single-tier system only has verbatim; it lacks gist entirely — exactly the measured failure mode (aggregation/relative-date questions need gist).

Sources: fuzzy-trace theory literature; Brainerd & Reyna 2002 (10.1111/1467-8721.00192).

### 1.4 Reconsolidation and memory updating

**Neuroscience.** Nader, Schafe & LeDoux (2000, *Nature* 406:722-726): a consolidated memory, once retrieved, re-enters a labile state and must be re-stabilized. Biological memory updating happens through retrieval-triggered labilization and re-storage, not silent in-place overwrite.

**Architectural implication.** Neuroscience backing for **invalidate-don't-delete** supersession (mark old fact invalid, add new fact, keep both queryable) — the bi-temporal pattern Zep/Graphiti implements and our knowledge-update category (at ceiling) already partially does.

Source: Nature 35021052.

### 1.5 Schemas and the self-schema

**Cognitive science.** Bartlett (1932, *Remembering*): recall is reconstructive, driven by schemas. Markus (1977, *JPSP*): self-schemata are trait-specific cognitive generalizations about oneself that speed encoding/retrieval of self-relevant information. Rogers, Kuiper & Kirker (1977, *JPSP* 35:677-688): self-referential encoding produces the best incidental recall of any encoding condition tested — a "superordinate schema," structurally privileged.

**Architectural implication.** "Knowledge about me" sits in a **structurally distinct, privileged access path** in the brain. Supports a **separate profile tier** — not just another fact type in the semantic store.

Sources: PMC3815569; Markus 1977 (Stanford PDF); PubMed 909043.

### 1.6 How humans answer "how many times have I X" — and how accurate they are

**Cognitive science.** Hasher & Zacks (1979/1984): frequency-of-occurrence is registered largely automatically — but coarse/comparative, not counts. Menon (1993, *J. Consumer Research* 20:431-440): people switch strategy by frequency regime — for infrequent behaviors, **episodic enumeration** (recall + count); for frequent behaviors, **rate-based estimation** ("about N per period × periods"). Humans **do not replay 47 sessions to count concerts** — past a low threshold they switch to arithmetic on a remembered rate. Accuracy: systematic regression pattern — small frequencies overestimated, large frequencies **underestimated**.

**Architectural implication.** The most counter-intuitive finding: **humans are bad at exactly the task the benchmark tests** — past a handful of instances they stop counting and start estimating. A memory system should NOT imitate this failure mode. Consolidation into countable, atomic, dated facts turns "how many X total" into exact enumeration — a case where the machine categorically outperforms the biology it's modeled on. For judgment/preference questions, human-like gist still applies.

Sources: Hasher & Zacks (Rotman-Baycrest); Menon 1993.

### 1.7 Temporal memory: dating by reconstruction, not lookup

**Cognitive science.** Friedman (1993, *Psychological Bulletin* 113:44-66, "Memory for the Time of Past Events"): humans do not store a queryable absolute-time index. Dating is **reconstructive** — inferred from temporal landmarks and general time patterns. No evidence for a chronologically organized memory store.

**Architectural implication.** Beat the biology: our turns carry ground-truth timestamps. Consolidation resolves **relative** references ("last Tuesday," "four weeks ago") into **absolute** dates *at write time*, using the turn's real timestamp as anchor. RUNNING_NOTES already identified dropped question/haystack dates as a root-cause bug class — the biology confirms this is the central mechanism to design around, not an edge case.

Source: Friedman 1993.

### 1.8 Spreading activation / associative retrieval

**Cognitive science.** Collins & Loftus (1975, *Psychological Review* 82:407-428): semantic memory as a network; activation spreads along weighted links. Teyler & DiScenna (1986, *Behavioral Neuroscience*; updated Teyler & Rudy 2007, *Hippocampus*): hippocampal **indexing** theory — hippocampus stores an index pointing to distributed neocortical regions active during the episode; partial cue → index → cortical pattern → retrieval.

**Architectural implication.** Retrieval should be cue-driven graph traversal over entities/relations, not flat bag-of-words over independent documents — the mechanism a consolidated fact graph with entity nodes enables. Note: "hippocampal indexing" as framing is already claimed by HippoRAG — cite correctly if invoked.

Sources: Collins & Loftus 1975; PubMed 3008780; PubMed 17696170.

## 2. Prior art audit — representation-level, ruthlessly honest

### 2.1 HippoRAG (Gutiérrez et al., NeurIPS 2024)

Offline: schemaless OpenIE over passages → triples → KG (relation edges, synonym edges, context edges). Online: Personalized PageRank seeded at query-entity nodes ranks passages. Explicitly modeled as neocortex + hippocampal index + parahippocampal retrieval encoder.

**Distance from our proposal:** close on the *retrieval* metaphor, **not close on the memory lifecycle**: no session boundaries, no temporal validity/supersession, no bi-temporal invalidation, no distillation to atomic dated facts. It's a retrieval index, not a consolidation-and-profile system.

Sources: arXiv 2405.14831; NeurIPS 2024; github.com/osu-nlp-group/hipporag.

### 2.2 Letta / MemGPT "sleep-time compute"

Two things under one name: (1) product feature — a background agent owns core memory blocks (free-text, labeled) and calls `rethink_memory()` asynchronously; (2) research paper (arXiv 2504.13171) — precompute inferences before queries arrive (test-time-compute amortization), not episodic→semantic distillation of personal history. Neither documents atomic dated facts, per-turn citations, or per-attribute bi-temporal supersession. **UNVERIFIED**: internal provenance in Letta memory blocks.

Sources: Letta docs/blog; arXiv 2504.13171.

### 2.3 Stanford Generative Agents (Park et al., UIST 2023)

Memory stream of timestamped observations; retrieval = recency + importance + relevance; **reflection trees** — periodic LLM-generated higher-level statements over clusters of memories, recursively. Closest historical antecedent to batch distillation — but the unit is a narrative sentence with an importance score, no citations field, no supersession, evaluated on believability not QA.

Source: ACM 10.1145/3586183.3606763.

### 2.4 Mem0 — write-time fact extraction

Per exchange: LLM extracts candidate facts; vector-compare against existing; ops ADD/UPDATE/DELETE/NONE — the closest published analog to per-attribute supersession. **Verified regression: OSS v3 `add()` is single-pass ADD-only** — UPDATE/DELETE events no longer emitted; contradictions accumulate (open issues #4896, #4956, #5867). No atomicity guarantee documented; provenance backlink **UNVERIFIED**.

Sources: Mem0 docs; arXiv 2504.19413; GitHub issues above.

### 2.5 Zep / Graphiti — bi-temporal KG WITH provenance

Closest shipped system to two of our four proposed capabilities. Edges carry valid_at/invalid_at (bi-temporal); **superseding facts invalidate rather than delete**. Provenance: every node/edge carries `episodes` lists (raw source messages kept byte-for-byte), traversable — confirmed on Zep's own engineering blog. What Zep does NOT have: a **separate profile tier** — user attributes live in the same general-purpose graph substrate as event facts; custom Pydantic types approximate but don't separate lifecycle.

Sources: arXiv 2501.13956; blog.getzep.com provenance post; Graphiti docs.

### 2.6 Hindsight — opinion networks

Four networks: World (objective), Experience (first-person), Opinion (subjective judgments with confidence, strengthened/weakened by evidence), Observation (profiles). Not a close mechanism match, but a design axis we lack: **explicit separation of subjective judgment (with confidence) from objective fact**.

Source: arXiv 2512.12818.

### 2.7 Honcho / Plastic Labs

Two-stage pipeline: ingest-time "Deriver" (small fine-tuned model, batches, structured conclusions about a peer) + "dream-time" background revisiting for new deductions — structurally close to "write-time light extraction + periodic batch consolidation." **UNVERIFIED**: conclusion representation, supersession, citations — docs don't specify output schema; needs a code read before comparative claims.

Sources: github.com/plastic-labs/honcho.

### 2.8 Academic consolidation work — most important section

- **SeCom** (Microsoft, ICLR 2025, arXiv 2502.05589): turn-level, session-level, AND summary-based granularities all underperform; segment-level + denoising wins. Evidence for finer, purpose-cut units.
- **Nemori** (arXiv 2508.03341): episode segmentation + semantic "insights" via predict-calibrate; insights are **free-text narrative, not structured atomic facts**; whole-insight replace, **no citations, no cross-lingual**. Claims SOTA-at-publication on LoCoMo/LongMemEval.
- **TiMem** (arXiv 2601.02845, ACL Findings 2026) — **the single most important prior-art finding**: Temporal Memory Tree (leaf raw turns → summary nodes → persona root). Published: **75.30% LoCoMo, 76.88% LongMemEval-`_s`** — **above our 66.0% on the identical split**. Read directly: **free-text hierarchical summaries, not structured facts; no back-citations; no explicit per-attribute supersession; no cross-lingual handling anywhere.** Open source (github.com/TiMEM-AI/timem). Must be read/run and reckoned with before finalizing scope — "we built consolidation" alone is not defensible once TiMem exists.
- **SCM** (arXiv 2604.20943): NREM/REM-staged consolidation framing, but evaluated on an 8-test suite over 10-turn conversations — research preview, not a benchmarked competitor.

### 2.9 Cross-lingual canonicalization — weakest-evidenced

**No named, benchmarked memory system documents cross-lingual canonicalization of consolidated facts** (Hindi turn + English turn → one canonical entry). Adjacent literature exists (cross-lingual entity linking/coreference), not applied in any audited product. Our repo already ships retrieval-time ALIAS_OF aliasing; consolidation-time canonicalization goes further than anything found. **Caveat: absence of evidence ≠ proof of absence** — re-verify against non-English competitor docs before publishing.

## 3. Novelty verdict

**(a) Batch consolidation producing dated atomic facts with citations — NOT novel as a category** (Nemori, SeCom, TiMem, Honcho dream-time). Never publish "we invented consolidation." **Defensible narrow claim:** no surveyed consolidation system stores the distilled unit as structured, field-level, dated, individually-cited atomic facts (all use narrative text or graph edges).

**(b) Bi-temporal per-attribute profile tier — mechanisms not novel individually; combination plausibly is.** Zep ships bi-temporal edges (no separate tier); Letta has a separate tier (free-text, not bi-temporal); Mem0 designed multi-op supersession (currently broken in shipped OSS). Never claim "first bi-temporal memory" (Zep's paper title) or "first per-attribute conflict resolution" (Mem0's design intent).

**(c) Cross-lingual canonical facts — weakest evidence base, hedge accordingly.** Safe wording: "no publicly documented competitor claims write-time cross-lingual canonicalization of consolidated facts." Never absolute "no one has built this."

**(d) Public inspectability — not novel as "having provenance"** (Zep ships it). Defensible as product/UX differentiation: end-user-inspectable "why does the agent believe this" down to the raw turn, without graph queries — a product claim, not a research-novelty claim.

**Net:** the defensible position is the *combination + representation granularity*: structured/dated/cited atomic facts + separate functioning bi-temporal profile tier + write-time cross-lingual canonicalization + end-user inspectability. TiMem must be read directly and beaten (or honestly contextualized) on the number — representation elegance alone doesn't close a 10.9-point gap; it must be measured.
