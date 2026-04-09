# AgentMem OS (MemNAI) — Complete Project Guide

> **"Memory is the foundation of intelligence. LLMs have none."**
> AgentMem OS is a research-grade, local-first AI memory operating system that gives LLM agents persistent, structured, biologically inspired memory — enabling them to remember, reason, and improve across thousands of conversations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem We're Solving](#2-the-problem-were-solving)
3. [Our Solution — The 4-Tier Memory Architecture](#3-our-solution--the-4-tier-memory-architecture)
4. [Novel Algorithms (Research Contributions)](#4-novel-algorithms-research-contributions)
5. [System Architecture & File Map](#5-system-architecture--file-map)
6. [Technology Stack](#6-technology-stack)
7. [Database Schema](#7-database-schema)
8. [How the System Works End-to-End](#8-how-the-system-works-end-to-end)
9. [Benchmark Metrics](#9-benchmark-metrics)
10. [How to Run the Project Locally](#10-how-to-run-the-project-locally)
11. [Phase-by-Phase Build History](#11-phase-by-phase-build-history)
12. [Research Positioning](#12-research-positioning)
13. [Interview Q&A — Deep Dive](#13-interview-qa--deep-dive)

---

## 1. Project Overview

**Name:** AgentMem OS (codebase: `memnai`)
**Type:** Research system / open-source framework
**Target:** Anthropic engineering teams, NeurIPS 2026 ML for Systems Workshop
**Cost to run:** $0.00 — fully local, zero cloud dependency
**Stack:** Python 3.11 · Ollama · SQLite · ChromaDB · spaCy · NetworkX · Redis

AgentMem OS solves one of the most critical open problems in LLM agent design: **stateless memory**. Today's LLMs forget everything between sessions, within sessions they silently drop old context when the window fills, and they have no concept of *why* certain information matters more than other information.

This system implements a four-tier memory hierarchy inspired by human cognitive memory science — the same model that underlies how humans encode short-term experience into long-term knowledge. It runs entirely on a developer's laptop, requires no API keys, and is designed so that every algorithmic decision is motivated by neuroscience and measurable via rigorous benchmarks.

---

## 2. The Problem We're Solving

### 2.1 The Stateless LLM Problem

When you talk to any LLM today:

- **Between sessions:** Zero memory. Every conversation starts fresh. The agent cannot remember your name, your project, or what it told you yesterday.
- **Within a session:** When the context window fills, the model silently truncates the oldest messages. All early context — often the most important setup information — is lost forever.
- **No prioritization:** Current systems discard context chronologically ("oldest first"). They don't ask: *which messages matter most?*
- **No abstraction:** Even if memory is kept, it stays as raw text. There's no mechanism to learn patterns like "this user always prefers code examples" or "when they mention 'the pipeline', they mean the Kafka cluster."

### 2.2 The Business Impact

- Customer support agents lose context between sessions, forcing users to repeat themselves.
- Coding assistants forget established project conventions mid-session.
- Long-horizon research assistants can't maintain coherent understanding across weeks of work.
- Multi-agent systems have no shared knowledge layer; every agent starts blind.

### 2.3 What's Missing From Existing Work

| Approach | Gap |
|---|---|
| Simple buffer (keep last N turns) | Drops important early context; no prioritization |
| Summarization-only (e.g., LangChain ConversationSummaryMemory) | Loses specific facts; no entity tracking; no learning |
| RAG (vector search over past turns) | No structure; doesn't distinguish episodic from semantic; no pattern learning |
| MemGPT | Good start, but uses expensive LLM calls for all memory operations; no procedural tier |

AgentMem OS addresses all four gaps with four novel algorithms.

---

## 3. Our Solution — The 4-Tier Memory Architecture

Inspired by the Atkinson-Shiffrin model of human memory:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENTMEM OS MEMORY HIERARCHY                     │
├──────────────────┬──────────────────────────────────────────────────┤
│  TIER 1          │  Working Memory          (Redis L1 cache)        │
│  Episodic        │  Raw conversation turns  (SQLite)                │
│  Semantic        │  Compressed summaries    (ChromaDB)              │
│  TIER 4          │  Behavioral Patterns     (SQLite ProceduralPat.) │
├──────────────────┴──────────────────────────────────────────────────┤
│                    CONTEXT ASSEMBLER                                │
│  Combines all tiers into a structured prompt with token budgets     │
└─────────────────────────────────────────────────────────────────────┘
```

### Tier 1 — Working Memory (Redis)

- **What:** The last 5–10 turns held in-memory as a circular buffer.
- **Why Redis:** Sub-millisecond read latency; data evicts automatically when TTL expires.
- **Analogy:** Human working memory — what you're actively thinking about right now.
- **Token budget in context:** 40% of context window.

### Tier 2 — Episodic Memory (SQLite)

- **What:** Every conversation turn ever stored, with importance metadata.
- **Why SQLite:** Zero-config, file-based, WAL mode for concurrent background writes. Perfect for local-first deployment.
- **What's stored per turn:** `content`, `role`, `turn_index`, `importance_score`, `entity_count`, `semantic_novelty`, `is_compressed`.
- **Analogy:** Human episodic memory — specific events you experienced ("I remember when we debugged that auth issue").
- **Token budget in context:** 25% (highest-importance uncompressed turns).

### Tier 3 — Semantic Memory (ChromaDB)

- **What:** Abstractive summaries produced by the Sleep Consolidation Engine. Clustered, compressed representations of groups of turns.
- **Why ChromaDB:** Local vector database supporting cosine similarity search and MMR (Maximal Marginal Relevance) for diverse retrieval.
- **What's stored:** `content` (LLM-generated summary), `cluster_id`, `abstraction_level` (1=episode, 2=pattern, 3=principle), `is_shared` (cross-agent pool).
- **Analogy:** Human semantic memory — generalized knowledge ("I know how to debug Python, I don't remember learning it").
- **Token budget in context:** 20% (most relevant summaries via MMR search).

### Tier 4 — Procedural Memory (SQLite)

- **What:** Mined behavioral patterns extracted from conversation sequences. Things like "When user reports a bug → agent always asks for stack trace first."
- **Novel:** This tier did not exist in any prior LLM memory framework.
- **What's stored:** `trigger_type`, `action_type`, `confidence`, `support_count`, `agent_id`, `is_shared`.
- **Analogy:** Human procedural memory — learned skills you execute automatically without thinking.
- **Token budget in context:** 3% (lightweight; patterns are short strings).

### Global Map — Entity Knowledge Graph (NetworkX + SQLite)

- **What:** A co-occurrence graph where nodes are named entities (people, orgs, tools, places) and edges represent how often they appear together.
- **Why it matters:** Fills the 7% "Global Map" slot in context — gives the agent a structured world model.
- **Example:** After 100 turns about a software project, the graph knows: `Redis ↔ SessionStorage (12×)`, `Sahith ↔ AgentMem (7×)`.

### Context Assembler — Putting It All Together

At inference time, the `ContextAssembler` constructs a structured prompt:

```xml
<SYSTEM>You are an AI assistant with persistent memory...</SYSTEM>
<INHERITED CONTEXT>Agent: default | Session: abc123 | Turns: 142</INHERITED CONTEXT>
<SEMANTIC MEMORY>... top-k ChromaDB summaries ...</SEMANTIC MEMORY>
<WORLD MODEL>... entity subgraph ...</WORLD MODEL>
<BEHAVIORAL PATTERNS>... procedural patterns ...</BEHAVIORAL PATTERNS>
<RECENT TURNS>... last 5-10 turns from Redis ...</RECENT TURNS>
```

The assembler enforces token budgets per section and counts tokens using `tiktoken`.

---

## 4. Novel Algorithms (Research Contributions)

These are the four algorithmic contributions that differentiate this work from existing systems. Each is implemented from scratch without depending on external APIs.

---

### Algorithm #1: Memory Importance Scorer

**File:** `memnai/llm/importance_scorer.py`
**Class:** `MemoryImportanceScorer`

**The Core Idea:** Instead of compressing the *oldest* turns when memory fills (the naive approach), we compress the *least important* turns. This preserves rare, entity-rich, semantically novel information even if it's old.

**The Formula:**

```
importance(t) = 0.30 × TF-IDF_rarity(t)
              + 0.35 × semantic_novelty(t)
              + 0.20 × entity_density(t)
              + 0.15 × recency_decay(t)
```

**Signal 1 — TF-IDF Rarity (weight 0.30)**
Uses scikit-learn's `TfidfVectorizer` on the current batch of turns. Turns with rare, specific vocabulary (e.g., "DBSCAN clustering with cosine metric") score higher than turns with common words ("ok", "yes", "sounds good"). The IDF component naturally rewards content that stands out from the rest of the conversation.

**Signal 2 — Semantic Novelty (weight 0.35, highest)**
Computes cosine distance from each turn's embedding to the nearest existing summary embedding in ChromaDB. A turn that's *very different* from anything previously summarized is highly novel and should be preserved. Formula: `novelty = 1 - max_cosine_similarity(turn, summaries)`. Falls back to `0.5` (neutral) if no embedder is available.

**Signal 3 — Entity Density (weight 0.20)**
Uses spaCy NER to count named entities per turn (PERSON, ORG, PRODUCT, GPE, etc.). High-value entity types get a 2× bonus weight. A turn mentioning "Google DeepMind partnered with NASA on Gemini Pro" scores higher than a turn saying "the thing the guy mentioned". Falls back to a regex heuristic (capitalized word sequences) when spaCy is unavailable.

**Signal 4 — Recency Decay (weight 0.15)**
Exponential decay: `score = 2^(-(age/HALF_LIFE))` where `HALF_LIFE = 20` turns. This means a turn 20 positions ago has half the recency score of the most recent turn. Recency is lowest-weighted because new information isn't automatically more important — it just hasn't been reviewed yet.

**Normalization:** Each signal is min-max normalized within the current batch to `[0, 1]`. When all values are identical (uniform), the normalizer returns `[0.5]` to avoid division-by-zero.

**Key methods:**
- `score_turns(turns, existing_embeddings, get_embedding_fn)` → `List[(turn_dict, float)]` sorted ascending (compress first)
- `get_compression_candidates(turns, compress_fraction=0.30)` → `(to_compress, to_keep)`

---

### Algorithm #2: Sleep Consolidation Engine

**File:** `memnai/llm/consolidation_engine.py`
**Class:** `SleepConsolidationEngine`

**The Core Idea:** Inspired by human sleep consolidation — the biological process where the brain replays and reorganizes episodic memories into semantic knowledge during sleep. Our engine runs as a **background daemon thread** (not triggered on every turn), clusters similar turns together, and generates abstractive summaries.

**The Pipeline:**

```
1. Score turns → identify least-important 30%
2. Cluster those turns → DBSCAN on turn embeddings
3. For each cluster → generate one LLM summary (or deterministic fallback)
4. Write summary to ChromaDB → mark turns as compressed in SQLite
5. Log consolidation event → ConsolidationLog table
6. Sleep for interval_seconds → repeat
```

**Clustering with DBSCAN:**
We use `DBSCAN(eps=0.25, min_samples=2, metric='cosine')` from scikit-learn. DBSCAN is the right algorithm here because:
- It doesn't require specifying the number of clusters in advance (we don't know how many topics are in a conversation).
- It handles noise points well — turns that don't fit any cluster get their own single-turn cluster.
- Cosine metric is appropriate for high-dimensional embedding spaces.

**Noise handling:** DBSCAN labels some points as noise (`cluster_id = -1`). We assign each noise point its own singleton cluster so it still gets a summary.

**Abstraction levels:**
Summaries are tagged with `abstraction_level`:
- L1 (1): Episode-level — "User asked about Redis configuration, I provided connection settings"
- L2 (2): Pattern-level — "User frequently asks about infrastructure configuration"
- L3 (3): Principle-level — "User prefers concise answers with code examples over explanations"

**Fallback when LLM unavailable:**
`_fallback_summary()` generates a deterministic string: "Session X cluster Y: [Role] Turn A — content preview..."

**Background scheduler:**
```python
engine.start_background_scheduler(interval_seconds=300)  # runs every 5 min
engine.stop_background_scheduler()  # graceful shutdown
```

---

### Algorithm #3: Entity Knowledge Graph

**File:** `memnai/db/knowledge_graph.py`
**Class:** `EntityKnowledgeGraph`

**The Core Idea:** Extract named entities from every turn and build a co-occurrence graph. Two entities that appear in the same turn share an edge; the edge weight increases each time they co-occur. This gives the agent a structured "world model" of who/what/where exists in its context.

**Storage:** Dual persistence — SQLite tables (`knowledge_graph_nodes`, `knowledge_graph_edges`) for persistence across restarts + NetworkX in-memory `DiGraph` for fast BFS traversal.

**Entity Extraction:**
Primary: spaCy NER targeting these types: `PERSON, ORG, PRODUCT, GPE, WORK_OF_ART, EVENT, LANGUAGE, FAC, LOC`
Fallback: Regex pattern `\b[A-Z][a-zA-Z]{2,}\b` (capitalized word sequences), deduped, filtered for stopwords and len > 2, capped at 15 entities per turn.

**Graph Construction:**
- Each entity → node with `entity_type`, `mention_count`, `first_seen_session`
- Each co-occurring entity pair → edge with `weight` (incremented per co-occurrence)

**Retrieval — Subgraph BFS:**
```python
kg.get_relevant_subgraph(query, agent_id, top_k=10, max_hops=2)
```
1. Extract entities from the query string.
2. BFS from those entities up to `max_hops` in the NetworkX graph.
3. Filter to neighbors with `weight >= 2` (confirmed co-occurrences, not one-time noise).
4. Serialize to text: `"[WORLD MODEL]\nEntities: Google, NASA, Redis\nRelationships: Google ↔ NASA (5×)\n..."`

**Ingest flow (background thread in `store.py`):**
Every turn is passed to `_ingest_kg()` in a daemon thread, so entity extraction doesn't block the main response loop.

---

### Algorithm #4: Procedural Memory

**File:** `memnai/llm/procedural_memory.py`
**Class:** `ProceduralMemory`

**The Core Idea:** This is the first formulation of procedural memory for LLM agents. Using sequence mining over `(user_turn, assistant_turn)` pairs, we learn recurring behavioral patterns: "When user X happens → agent consistently does Y." Equivalent to imitation learning from self-play.

**Step 1 — Trigger Classification:**
`classify_trigger(text)` maps user turns to one of 8 categories using regex pattern matching:
- `bug_report` — crash\w*, fail\w*, error, exception, broken
- `feature_request` — add, implement\w*, build\w*, create\w*, want, need
- `question` — how, what, why, explain\w*, tell me
- `code_review` — review\w*, check\w*, feedback, refactor\w*
- `debugging` — debug\w*, trace\w*, stack trace, why is
- `planning` — plan\w*, design\w*, architect\w*, roadmap
- `clarification` — clarif\w*, unclear, confus\w*
- `general` — fallback

**Step 2 — Action Extraction:**
`extract_action(assistant_response)` maps assistant responses to action labels using regex:
- `provided code block` — contains ```
- `suggested solution` — suggest\w*
- `explained concept` — explain\w*
- `wrote code` — wrote...code
- And 8 more patterns with default fallback `"responded"`

**Step 3 — Pattern Mining:**
`mine_patterns(session_id, agent_id)` slides over all `(user, assistant)` pairs, extracts `(trigger, action)` tuples, and counts frequency. Patterns with `support_count >= MIN_SUPPORT_COUNT (2)` are persisted.

**Confidence scoring:**
```
confidence = count(trigger=X, action=Y) / count(trigger=X)
```
A pattern with confidence 0.85 means "85% of the time this trigger happened, the agent took this action."

**Global promotion:**
`promote_to_global(pattern_id)` sets `is_shared=True` for patterns with `confidence >= 0.8`. These are injected into all agents' contexts, not just the originating agent.

**Retrieval:**
```python
pm.get_relevant_patterns(query, agent_id, top_k=3)
# → "[BEHAVIORAL PATTERNS]\n• bug_report → provided code block (confidence=87%, seen 14×)\n..."
```

---

## 5. System Architecture & File Map

```
memnai/
├── Guide.md                        ← This file
├── requirements.txt                ← Pinned dependencies (reproducible)
├── pyproject.toml                  ← pip installable, CLI entry point
├── setup.sh                        ← One-command setup script
├── config.yaml                     ← DB path, Redis settings, Ollama model
│
├── db/
│   ├── models.py                   ← SQLAlchemy ORM — 9 tables
│   ├── engine.py                   ← DB initialization, session factory, WAL mode
│   ├── knowledge_graph.py          ← Algorithm #3: Entity Knowledge Graph
│   └── [database.py]               ← Legacy (iCloud-managed, used on Mac)
│
├── llm/
│   ├── importance_scorer.py        ← Algorithm #1: Memory Importance Scorer
│   ├── consolidation_engine.py     ← Algorithm #2: Sleep Consolidation Engine
│   ├── procedural_memory.py        ← Algorithm #4: Procedural Memory
│   ├── context_assembler.py        ← Context budget assembly (all 4 tiers)
│   └── [summarizer.py]             ← Legacy LiteLLM summarizer
│
├── storage/
│   └── store.py                    ← Main entry point: turn ingestion, retrieval
│
├── benchmarks/
│   └── eval_harness.py             ← CRS, TES, LCS benchmark metrics
│
├── alembic/
│   ├── env.py                      ← Alembic migration config (render_as_batch=True)
│   └── versions/                   ← Migration scripts
│
├── api/
│   └── routes.py                   ← FastAPI REST endpoints
│
└── tests/
    ├── test_phase2_db.py            ← 11 DB schema tests
    └── test_phase3_algorithms.py   ← 37 algorithm tests (all passing)
```

---

## 6. Technology Stack

| Component | Technology | Why |
|---|---|---|
| LLM inference | Ollama (llama3.2 / mistral) | 100% local, free, GPU-accelerated |
| LLM routing | LiteLLM | Model-agnostic; swap providers without code changes |
| Embeddings | langchain-ollama / nomic-embed | Local embeddings via Ollama |
| Episodic store | SQLite + SQLAlchemy | Zero-config, WAL mode, excellent Python support |
| Semantic store | ChromaDB | Local vector DB with MMR search |
| Working memory | Redis | Sub-ms latency, TTL-based eviction |
| NLP / NER | spaCy (en_core_web_sm) | Fast, accurate entity extraction |
| Graph | NetworkX | In-memory graph for BFS traversal |
| Clustering | scikit-learn DBSCAN | No fixed K, handles noise, cosine metric |
| TF-IDF | scikit-learn TfidfVectorizer | Batch rarity scoring |
| Token counting | tiktoken | Precise context budget enforcement |
| DB migrations | Alembic | Reproducible schema evolution |
| API layer | FastAPI | Async, modern, auto-docs |
| Logging | Loguru | Structured, leveled, beautiful |
| Testing | pytest | Standard Python testing |

**Cost: $0.00** — all components run locally. No API keys required. No cloud services.

---

## 7. Database Schema

### 9 Tables Total

**`agent_namespaces`** — Multi-agent isolation
`id`, `agent_id` (unique slug), `name`, `system_prompt`, `created_at`

**`sessions`** — Conversation sessions
`id`, `session_id` (UUID), `agent_id` (FK → agent_namespaces), `created_at`, `updated_at`, `is_archived`

**`turns`** — Every message ever sent
`id`, `session_id` (FK), `role` (user/assistant/system), `content`, `turn_index`, `tokens`, `importance_score`, `entity_count`, `semantic_novelty`, `is_compressed`, `created_at`

**`summaries`** — ChromaDB-backed semantic memories
`id`, `session_id` (FK), `content`, `tokens`, `cluster_id`, `abstraction_level` (1/2/3), `is_shared`, `created_at`

**`cost_log`** — LLM usage tracking
`id`, `session_id`, `model`, `input_tokens`, `output_tokens`, `cost_usd`, `created_at`

**`procedural_patterns`** — Algorithm #4 output
`id`, `agent_id` (FK), `trigger_type`, `action_type`, `confidence`, `support_count`, `example_turn_id`, `is_shared`, `created_at`, `updated_at`

**`knowledge_graph_nodes`** — Algorithm #3 entities
`id`, `agent_id` (FK), `entity_key` (normalized text), `entity_type`, `mention_count`, `first_seen_session`, `created_at`

**`knowledge_graph_edges`** — Algorithm #3 co-occurrences
`id`, `agent_id` (FK), `source_key`, `target_key`, `weight`, `created_at`, `updated_at`

**`consolidation_log`** — Algorithm #2 audit trail
`id`, `session_id`, `turns_compressed`, `summaries_created`, `algorithm_version`, `duration_seconds`, `created_at`

### Key Design Decisions

**WAL Mode:** `PRAGMA journal_mode=WAL` enables concurrent readers while the background consolidation thread writes.

**`expire_on_commit=False`:** Prevents `DetachedInstanceError` when background threads access ORM objects after their session closes.

**`StaticPool + check_same_thread=False`:** Required for background daemon threads sharing the same SQLite connection.

**`render_as_batch=True` in Alembic:** SQLite doesn't support `ALTER TABLE ADD COLUMN` directly; Alembic's batch mode rewrites the table to apply schema changes.

---

## 8. How the System Works End-to-End

### A Complete Turn Lifecycle

```
User sends message: "The app crashes when I upload a file"
                              │
                              ▼
                    ┌──────────────────┐
                    │   store.py       │
                    │   add_turn()     │
                    └────────┬─────────┘
                             │
               ┌─────────────┼──────────────────┐
               ▼             ▼                   ▼
        Save to Redis   Save to SQLite    Background threads:
        (Tier 1)        (Tier 2)         ├─ ingest_kg()
                                         └─ check_and_compress()
                              │
                              ▼
                    ┌──────────────────┐
                    │ context_assembler│
                    │ get_context()    │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                   ▼                  ▼
   Recent turns         ChromaDB           Entity KG         Procedural
   from Redis        MMR search           BFS from          patterns
   (40% budget)      (20% budget)         "crash","upload"  (3% budget)
                                          (7% budget)
                             │
                             ▼
               Structured XML prompt assembled
               with enforced token budgets
                             │
                             ▼
                    Ollama LLM → response
                             │
                             ▼
                    Response saved as assistant turn
                    ProceduralMemory.mine_patterns() updates patterns
```

### Background Consolidation (Every 5 Minutes)

```
SleepConsolidationEngine wakes up
    │
    ▼
Score all uncompressed turns → importance_scorer.score_turns()
    │
    ▼
Take bottom 30% → cluster with DBSCAN (cosine metric)
    │
    ▼
For each cluster → LLM summarize → write to ChromaDB
    │
    ▼
Mark turns is_compressed=True in SQLite
    │
    ▼
Log to consolidation_log → sleep again
```

---

## 9. Benchmark Metrics

Three custom metrics measure system quality. All are implemented in `benchmarks/eval_harness.py`.

### CRS — Context Relevance Score

**What it measures:** How relevant the assembled context is to the current query compared to random context.

**Formula:**
```
CRS = cosine_similarity(query_embedding, assembled_context_embedding)
    - cosine_similarity(query_embedding, random_context_embedding)
```

**Range:** [-1, 1]. Higher = better. A system that always returns perfectly relevant context would score near 1.0.

**Why it matters:** Validates that our importance scoring and retrieval algorithms are actually selecting memory that helps answer the current question.

### TES — Token Efficiency Score

**What it measures:** How much useful information fits in the context window (compression quality).

**Formula:**
```
TES = geometric_mean(compression_ratio, entity_preservation_rate)

compression_ratio = 1 - (output_tokens / input_tokens)
entity_preservation_rate = entities_in_summary / entities_in_original_turns
```

**Range:** [0, 1]. Higher = better compression while preserving key facts.

**Why it matters:** A system that compresses aggressively but loses all entity references is useless. TES rewards the balance.

### LCS — Long-Horizon Continuity Score

**What it measures:** Can the agent recall specific facts from early in a long conversation?

**Formula:**
```
LCS = recall_rate(facts_from_turn_T, context_at_turn_T+K)
```
where K is a "horizon" offset (e.g., 50 turns later).

**Why it matters:** This is the core capability that distinguishes AgentMem OS from naive context truncation. A system with LCS = 0.8 correctly preserves 80% of facts from 50+ turns ago.

---

## 10. How to Run the Project Locally

### Prerequisites

- macOS or Linux (tested on both)
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Redis (optional but recommended for working memory)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/memnai.git
cd memnai

# 2. One-command setup
chmod +x setup.sh
./setup.sh

# This will:
# - Create a Python virtual environment
# - Install all dependencies from requirements.txt
# - Download the spaCy English model
# - Check Ollama and Redis availability
# - Create a .env template file
```

### Manual Setup (if setup.sh fails)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Start Required Services

```bash
# Start Ollama (download model if needed)
ollama pull llama3.2
ollama pull nomic-embed-text  # for embeddings
ollama serve

# Start Redis
redis-server
```

### Run the API Server

```bash
source .venv/bin/activate
uvicorn memnai.api.routes:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### Run Tests

```bash
# Install pytest in venv
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific phase
pytest tests/test_phase2_db.py -v
pytest tests/test_phase3_algorithms.py -v
```

### Initialize the Database

```bash
# Apply all migrations
alembic upgrade head

# Or let engine.py auto-initialize
python -c "from memnai.db.engine import init_db; init_db()"
```

### Environment Variables

```bash
# .env (auto-created by setup.sh)
MEMNAI_DB_PATH=~/.memnai/memnai.db  # SQLite location
OLLAMA_BASE_URL=http://localhost:11434
REDIS_URL=redis://localhost:6379
MEMNAI_DEFAULT_MODEL=ollama/llama3.2
```

---

## 11. Phase-by-Phase Build History

### Phase 1 — Project Scaffolding

**What was built:**
- `requirements.txt` — pinned dependency versions for full reproducibility
- `pyproject.toml` — makes project `pip install -e .` installable, registers `memnai` CLI
- `setup.sh` — one-command setup script

**Why it mattered:** The project existed as scattered files without proper packaging. Without `requirements.txt`, collaborators and paper reviewers couldn't reproduce the environment.

### Phase 2 — Database Schema & Engine

**What was built:**
- `db/engine.py` — new database engine replacing the iCloud-locked `database.py`
- `db/models.py` — expanded from 4 tables to 9 tables
- `alembic/env.py` — migration config with `render_as_batch=True`

**Schema changes:**
- Added `AgentNamespace`, `ProceduralPattern`, `KnowledgeGraphNode`, `KnowledgeGraphEdge`, `ConsolidationLog`
- Added columns to `turns` (importance_score, entity_count, semantic_novelty, is_compressed)
- Added columns to `summaries` (cluster_id, abstraction_level, is_shared)
- Added columns to `sessions` (agent_id FK, is_archived)

### Phase 3 — Four Novel Algorithms

**Phase 3A:** Memory Importance Scorer (`llm/importance_scorer.py`)
**Phase 3B:** Sleep Consolidation Engine (`llm/consolidation_engine.py`)
**Phase 3C:** Entity Knowledge Graph (`db/knowledge_graph.py`)
**Phase 3D:** Procedural Memory (`llm/procedural_memory.py`)

**Context Assembler updated** to stitch all four tiers into a structured XML prompt with token budgets.

**Test suite:** `tests/test_phase3_algorithms.py` — 37 tests, all passing.

### Phase 4 — Memory Federation Protocol (Complete)

**What was built:**
- `agents/namespace_manager.py` — Full agent lifecycle: create, fork, merge, lineage
- `agents/trust_network.py` — Directed inter-agent trust graph with EMA updates and transitive propagation
- `agents/memory_federation.py` — The Memory Federation Protocol (MFP): promote, retrieve, feedback, decay
- `db/models.py` updated with 4 new tables: `AgentTrustScore`, `FederatedMemoryEntry`, `AgentForkRecord`, `MemoryAccessLog`
- `tests/test_phase4_multiagent.py` — 56 tests across 5 sections, all passing

**The Memory Federation Protocol (MFP) — Phase 4's core novel algorithm:**

MFP is a decentralized protocol where agents asynchronously share high-quality abstract memories through a trust-weighted federated pool. Five sub-innovations make this unique:

1. **Memory Forking** — Like git branching: new agents inherit an existing agent's L2/L3 semantic knowledge (patterns and principles) without touching its episodic layer. The fork is atomic: all inherited memories are copied or none. Inherited patterns decay by 15% confidence (slight uncertainty about applicability in new context). This is the first formalization of git-style memory branching for LLM agents.

2. **Privacy-by-Design** — Raw episodic turns (Tier 2) NEVER enter the shared pool. Only L2 (patterns) and L3 (principles) can be promoted. This is a hard architectural guarantee, not a configuration option.

3. **Trust-Weighted Retrieval** — A memory's effective score at retrieval time is `relevance × trust × age_weight` — not just relevance. A memory from a highly trusted agent at 60% relevance outranks a memory from an unknown agent at 80% relevance. Trust is directional, persisted, and updated via EMA.

4. **Relevance Aging** — Shared memories accumulate a staleness score. Memories not accessed by other agents in `DECAY_DAYS` (30) and with fewer than `MIN_USEFUL_ACCESSES` (2) cross-agent retrievals are retired (`is_active=False`). This keeps the pool fresh.

5. **Agent Affinity** — `get_agent_affinity()` tracks which source agents a given agent retrieves from most often, building a routing preference graph over time.

**Promotion scoring formula:**
```
promotion_score = abstraction_level × (1 + confidence_bonus)
L1 → score=1.0 < threshold=2.0  → never promotes
L2 → score=2.0+                 → promotes when above threshold
L3 → score=3.0+                 → always promotes
```

**Trust update rule (EMA):**
```
trust_new = 0.80 × trust_old + 0.20 × feedback_signal
New pairs default to 0.50 (neutral)
Fork child→parent defaults to 0.90 (child trusts parent from birth)
```

**Transitive trust:**
```
If A trusts B = 0.9, and B trusts C = 0.8:
Transitive A→C via B = 0.9 × 0.8 = 0.72
Blended A→C = 0.70 × direct + 0.30 × transitive
```

**Test suite:** 56/56 passing across AgentTrustNetwork, MFP helpers, MemoryFederationProtocol, AgentNamespaceManager, and integration tests.

---

## 12. Research Positioning

### Target Venues

- **Primary:** NeurIPS 2026 — ML for Systems Workshop
- **Secondary:** Anthropic Research (direct exposure via open-source + blog post)
- **Tertiary:** ICML 2026, ICLR 2027 (if extended)

### What Makes This Novel

1. **Memory Importance Scoring** — No prior work uses a 4-signal composite score (TF-IDF + semantic novelty + entity density + recency decay) for LLM context compression. Existing work uses either chronological cutoff or simple TF-IDF.

2. **Sleep Consolidation** — Prior work (MemGPT, LangChain) triggers compression only when the context window fills. Our background daemon runs continuously on a schedule, mirroring biological sleep consolidation. DBSCAN clustering for episodic memory has not been published before.

3. **Entity Knowledge Graph** — Most memory systems store raw text. Ours builds a structured knowledge graph from every turn, enabling BFS-based relational retrieval. This fills the "global map" slot missing from all prior work.

4. **Procedural Memory** — No prior LLM memory paper formalizes procedural memory as a distinct tier. Our sequence mining formulation (trigger → action) is a direct translation of behavioral psychology into a machine learning primitive.

### Comparison Table

| System | Episodic | Semantic | Procedural | Entity KG | Importance Scoring | Local |
|---|---|---|---|---|---|---|
| ChatGPT Memory | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MemGPT | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LangChain Memory | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **AgentMem OS** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

---

## 13. Interview Q&A — Deep Dive

This section prepares you to answer any question about the project with complete confidence.

---

### Foundational Questions

**Q: What problem are you solving and why does it matter?**

A: Large language models suffer from three interconnected memory failures. First, they're completely stateless between sessions — every conversation starts fresh with no recollection of prior interactions. Second, within a session, they truncate context chronologically when the window fills, discarding important early setup information. Third, they have no mechanism to distinguish important information from trivial filler — they treat all messages as equally worth keeping or discarding.

This matters enormously in production. Customer support agents force users to repeat their account information every session. Coding assistants forget your project's coding standards mid-session. Research assistants can't maintain context across weeks of work. The cost is real: users disengage, tasks fail, and the promise of long-horizon AI assistance goes unfulfilled.

---

**Q: What's your solution at a high level?**

A: We implement a four-tier memory hierarchy inspired by human cognitive science — specifically the Atkinson-Shiffrin model of memory. Working memory in Redis holds the last few turns with millisecond latency. Episodic memory in SQLite stores every turn with importance metadata. Semantic memory in ChromaDB holds abstractive summaries of compressed episodes. Procedural memory in SQLite stores learned behavioral patterns mined from conversation sequences. An entity knowledge graph built on NetworkX connects named entities through co-occurrence relationships. At inference time, a Context Assembler draws from all four tiers under enforced token budgets to construct a rich, relevant prompt.

Everything runs locally. No cloud services. No API keys. $0 cost.

---

**Q: Why local-first? Why not just use a cloud provider?**

A: Three reasons. First, privacy — conversation history is sensitive data. Storing it in a cloud provider's infrastructure requires trust that users may not want to extend. Second, cost — cloud LLM APIs charge per token. A system that stores and retrieves thousands of turns would generate significant ongoing costs. Third, research integrity — a paper contribution should be reproducible by any reviewer with a modern laptop, without requiring cloud accounts or API keys.

The local stack (Ollama, SQLite, ChromaDB, Redis) is production-grade and capable of running the full system on a MacBook Pro with 16GB RAM.

---

### Architecture Questions

**Q: Walk me through exactly what happens when a user sends a message.**

A: The turn arrives at `store.add_turn()`. It's immediately written to both Redis (Tier 1 working memory, for fast next-turn retrieval) and SQLite (Tier 2 episodic memory, for permanent storage). Two background daemon threads are triggered: `_ingest_kg()` runs entity extraction via spaCy and updates the NetworkX knowledge graph, and `_check_and_compress()` counts uncompressed turns and optionally triggers the Sleep Consolidation Engine.

When the system needs to respond, `ContextAssembler.get_context()` is called. It fetches recent turns from Redis (40% of token budget), runs MMR search on ChromaDB for the most relevant and diverse summaries (20% budget), queries the knowledge graph for entities related to the current query via BFS (7% budget), retrieves top-3 procedural patterns from the procedural memory module (3% budget), and includes high-importance episodic turns (25% budget). The result is an XML-structured prompt that goes to Ollama.

After the LLM responds, the response is saved as an assistant turn, and `ProceduralMemory.mine_patterns()` updates the behavioral pattern database.

---

**Q: Why did you choose SQLite instead of PostgreSQL?**

A: Three reasons. First, local-first deployment — SQLite is a file. No server process, no configuration, no connection pooling. This dramatically reduces setup friction for researchers and developers who want to reproduce results. Second, WAL mode — SQLite's Write-Ahead Logging mode allows concurrent readers while the background consolidation thread writes, which is the only concurrency pattern our system needs. Third, Alembic compatibility — with `render_as_batch=True`, Alembic manages schema migrations reliably. The tradeoff is no horizontal scaling, but for a local-first research system, that's not a requirement.

---

**Q: Why ChromaDB for semantic memory instead of just SQLite with embeddings stored as BLOBs?**

A: ChromaDB provides MMR (Maximal Marginal Relevance) search out of the box. MMR is critical — it returns results that are both relevant to the query *and* diverse relative to each other. If you stored embeddings in SQLite and ran cosine similarity, you'd get the top-k most similar results, but they'd likely all be about the same topic (duplicating information). MMR's `lambda` parameter (we use 0.5) balances relevance vs. diversity, giving the context assembler a wider spread of relevant memories in limited tokens.

---

**Q: Why DBSCAN for clustering instead of K-means or hierarchical clustering?**

A: DBSCAN doesn't require specifying the number of clusters in advance. For conversation turns, we genuinely don't know how many distinct topics a session covers — it could be 2, it could be 15. K-means requires K. Hierarchical clustering requires a cut threshold. DBSCAN with cosine metric finds natural density-based clusters in embedding space and labels low-density points as noise (which we handle by giving each noise point its own singleton cluster). It's also more robust to outliers, which matter in conversation where a single off-topic turn shouldn't pollute a topic cluster.

---

### Algorithm Questions

**Q: Explain the Memory Importance Scorer in detail. Why these four signals?**

A: The scorer computes a weighted composite of four signals.

TF-IDF rarity (30%) rewards turns with rare, specific vocabulary. The intuition is that generic turns like "ok, sounds good" contribute little unique information and can be safely compressed. Turns discussing specific technical details ("configure DBSCAN with eps=0.25 and cosine metric") have high IDF and should be preserved.

Semantic novelty (35%, highest weight) rewards turns that are most different from what's already been summarized. This prevents the system from compressing turns that introduce genuinely new information. It's computed as `1 - max_cosine_similarity(turn_embedding, summary_embeddings)`. We give this the highest weight because preserving diverse information is the core goal.

Entity density (20%) rewards turns that mention many named entities. Entities are information anchors — "Google acquired DeepMind in 2014" contains three facts (actor, action, time) in the entities alone. Turns with high entity density have high information density and should be preserved.

Recency decay (15%, lowest weight) gives a mild preference to recent turns. We use `2^(-(age/20))` with a half-life of 20 turns. It's the lowest-weighted signal because new information isn't automatically more important — the other three signals can override recency if old information is rare, novel, and entity-rich.

---

**Q: What happens when spaCy isn't available? How robust is your system to missing dependencies?**

A: Every algorithm has a graceful degradation path. If spaCy isn't available, `_compute_entity_scores()` falls back to a regex heuristic that counts capitalized word sequences (`\b[A-Z][a-zA-Z]{2,}\b`) as entity proxies. It's less accurate but directionally correct — proper nouns are more likely entities.

If the LLM is unavailable for summarization, `_fallback_summary()` produces a deterministic string listing turn roles, indices, and content previews. No LLM call is ever required for the system to run — it degrades gracefully.

If Redis is unavailable, the Redis client fails silently and working memory falls back to direct SQLite queries. If ChromaDB has an error, semantic retrieval returns an empty string and the context assembler fills the budget with more episodic turns.

---

**Q: How does procedural memory actually "learn"? Isn't it just counting?**

A: It is counting, deliberately. The value of procedural memory isn't complex ML — it's the formulation of the right abstraction. By mining `(trigger_type, action_type)` pairs with `support_count >= 2`, we're doing statistical implication learning: "given trigger X, action Y follows with confidence C." This is equivalent to frequent pattern mining (Apriori/FP-growth) applied to conversation sequences.

The novelty isn't the algorithm, it's the application. No prior LLM memory paper defines a procedural tier. The engineering contribution is the full pipeline: trigger classification → action extraction → frequency counting → confidence scoring → retrieval at inference time → context injection. Each step has research decisions: which trigger categories to support, which action labels to define, what the minimum support threshold should be, how to format patterns for injection.

At higher abstraction levels (future work), an LLM could be used to generalize specific patterns into more abstract principles, but the foundation is deliberately rule-based and deterministic for reproducibility.

---

**Q: What is the confidence formula for procedural patterns?**

A: `confidence = count(trigger=X, action=Y) / count(trigger=X)`. This is the same formula as association rule confidence in market basket analysis. If the agent has seen 10 `bug_report` triggers and produced a `provided code block` response 8 times, the confidence is 0.8 (80%). Patterns with confidence >= 0.8 are promoted to the shared global pool via `promote_to_global()`, meaning they're injected into all agents' contexts.

---

**Q: How does the Entity Knowledge Graph get queried at runtime?**

A: `get_relevant_subgraph(query, agent_id, top_k=10, max_hops=2)`. First, entities are extracted from the query string using the same extraction pipeline as ingestion. Then we run BFS from those entities in the NetworkX in-memory graph up to 2 hops away. We filter to neighbors with `weight >= 2` (entities that have co-occurred at least twice — reduces noise from one-off mentions). The resulting subgraph is serialized to a human-readable text format: `"[WORLD MODEL]\nEntities: ...\nRelationships: A ↔ B (N×)"` and injected into the context assembler's global map slot (7% budget).

---

### Engineering Questions

**Q: How do you handle database concurrency? The background thread writes while the main thread reads.**

A: Three mechanisms work together. WAL mode (`PRAGMA journal_mode=WAL`) allows readers to proceed without waiting for writers — readers see the last committed snapshot while the writer works in the WAL file. `StaticPool` in SQLAlchemy reuses a single connection rather than opening new ones per thread, which is required for SQLite thread safety. `check_same_thread=False` explicitly allows the same connection object to be used from different threads (safe with WAL). And `expire_on_commit=False` prevents `DetachedInstanceError` — without it, accessing ORM object attributes after their session closes would raise an exception, which is exactly what happens in background threads.

---

**Q: Why Alembic for migrations? SQLAlchemy can create tables with `Base.metadata.create_all()` — why not just use that?**

A: `create_all()` only creates tables that don't exist. If you add a column to an existing model, `create_all()` does nothing for existing tables — it doesn't detect schema drift. Alembic tracks schema versions and generates migration scripts that apply delta changes: add column, rename column, drop column, etc. With `render_as_batch=True`, Alembic handles SQLite's limitation of not supporting `ALTER TABLE ADD COLUMN` for non-nullable columns by recreating the table with the new schema. This is the production-correct approach — it's what every serious application uses.

---

**Q: What's the iCloud file lock problem you mentioned in your code comments?**

A: When files are stored in iCloud Drive and haven't been downloaded to the local machine (they're "evicted" to cloud storage), the macOS kernel returns `EDEADLK` (deadlock error) when any process tries to write to them. The sandbox environment running on Linux cannot trigger iCloud to download these files. This affected `database.py`, `manager.py`, `redis_client.py`, `summarizer.py`, and `sync.py`. The solution was to create `db/engine.py` as a new implementation of the database layer, never touching the legacy files. All new code imports from `engine.py`. The legacy files continue to work on the user's Mac where iCloud is available.

---

### Research Questions

**Q: What makes this "research-grade"? Isn't this just engineering?**

A: The system is an engineering artifact, but it embodies four research contributions: (1) the formulation of memory importance scoring as a multi-signal composite for LLM context compression, (2) the first application of sleep-stage-inspired consolidation with DBSCAN clustering to LLM memory, (3) the entity knowledge graph approach to filling the "world model" slot in agent context, (4) the first formalization of procedural memory as a distinct tier in LLM agent memory systems.

The benchmarks (CRS, TES, LCS) are quantitative metrics designed to measure these contributions. A paper would present ablation studies: what happens to CRS when you remove semantic novelty from the importance scorer? What happens to LCS when you disable the consolidation engine? These controlled experiments validate each contribution independently.

---

**Q: How does this compare to MemGPT?**

A: MemGPT (Park et al., 2023) is the closest prior work. Key differences: MemGPT uses LLM function calls to manage memory, meaning every compression or retrieval decision requires an LLM inference. This is expensive and slow. AgentMem OS uses lightweight algorithms (TF-IDF, DBSCAN, regex, NetworkX BFS) for all memory management decisions, invoking LLMs only for abstractive summarization. Additionally, MemGPT has no procedural memory tier, no entity knowledge graph, and no importance-based scoring. MemGPT requires OpenAI API access; AgentMem OS runs entirely locally.

---

**Q: What would you measure to prove this system works?**

A: Three metrics:

CRS (Context Relevance Score) proves the retrieval is correct. Run 100 queries, compare cosine similarity of assembled context vs. random context. If CRS > 0.3, the system is meaningfully selecting relevant memories.

TES (Token Efficiency Score) proves compression quality. Compare entity preservation rate between our importance-scored compression vs. naive chronological compression. If TES > 0.7, we're compressing without losing key facts.

LCS (Long-Horizon Continuity Score) is the critical one. Seed a session with 100 specific factual turns, then query at turn 200. Measure what fraction of facts from turn 1-50 are still accessible in context. A system without AgentMem OS should score ~0.0 (all truncated). A system with it should score > 0.6.

---

**Q: What's the path from here to a published paper?**

A: Phase 4 (multi-agent namespaces) completes the system implementation. Then we run controlled experiments: three LLM baselines (no memory, simple buffer, MemGPT) vs. AgentMem OS on standardized long-horizon conversation benchmarks. We report CRS, TES, LCS across all systems. The paper structure would be: Abstract → Introduction (the problem) → Related Work → Architecture → Four Algorithm Sections → Experiments → Results → Conclusion. Target venue is NeurIPS 2026 Workshop on ML for Systems, with a blog post and open-source release timed to coincide with submission.

---

**Q: If someone from Anthropic asked you to explain the key insight of this project in 30 seconds, what would you say?**

A: "LLMs forget. Not because they can't remember — because nothing in their architecture decides *what* to remember. We built a system that scores every conversation turn across four signals: how rare is the vocabulary, how different is it from what's already been summarized, how many named entities does it contain, and how recent is it. The lowest-scoring turns get compressed during background 'sleep' cycles. The result is that after 500 turns, the agent still knows the specific technical decisions made in turn 3, even though it's forgotten the small talk. We also built the first procedural memory tier and the first multi-agent memory federation protocol for LLM agents. The whole thing runs locally for free."

---

### Phase 4 Questions — Memory Federation Protocol

**Q: What is the Memory Federation Protocol and why is it novel?**

A: MFP is a decentralized protocol where autonomous agents asynchronously share high-quality abstract memories through a trust-weighted shared pool. Think of it as a peer-reviewed journal for AI memories: agents submit only their most abstract learnings (patterns and principles), other agents read them weighted by how much they trust the source, and memories that no one cites eventually decay and are removed.

What makes it novel is the combination of five elements working together: (1) Privacy-by-design — raw episodes never enter the shared pool, only L2/L3 abstractions. (2) Trust-weighted retrieval — effective score is `relevance × trust × age_weight`, not just relevance. (3) EMA trust updates — trust changes slowly in both directions based on actual feedback. (4) Relevance aging — the pool self-cleans by retiring uncited memories. (5) Memory forking — git-style inheritance of semantic knowledge without copying episodic history. No prior work combines all five.

---

**Q: Explain memory forking. What's the analogy?**

A: Memory forking is exactly like git branching, applied to agent memory. When you fork an agent, the child agent inherits the parent's L2 and L3 summaries (patterns and principles) — its accumulated semantic knowledge — but starts with a completely clean episodic layer. The child builds its own conversation history from scratch, diverging from the parent over time while preserving the knowledge baseline.

The analogy: imagine a senior doctor trains a junior doctor. The junior inherits all the medical knowledge (semantic memory — patterns and principles) accumulated over the senior's career. But the junior hasn't seen the senior's actual patient records (episodic memory) — those are private. The junior starts seeing their own patients from day one, building their own episodic history while benefiting from the inherited foundation.

Why this matters technically: inherited patterns decay by 15% confidence on fork. The confidence decay reflects that patterns learned in one context may not apply perfectly in a new context — the child must re-validate them through its own experience.

---

**Q: How does the trust EMA work? Why EMA specifically?**

A: The trust update rule is: `trust_new = 0.80 × trust_old + 0.20 × feedback_signal`. This is an Exponential Moving Average. The α=0.80 weight on the old trust value means that any single feedback signal only moves the trust by 20% of the gap between old trust and the new signal. After 5 interactions, a signal has approximately `(1 - 0.8^5) ≈ 67%` cumulative influence. After 20 interactions, it's essentially `(1 - 0.8^20) ≈ 99%`.

Why EMA specifically? Three reasons: (1) It gives recent feedback more weight than old feedback without creating sudden trust swings. (2) It has a single hyperparameter (α) that's interpretable: higher α = more inertia = trust changes slowly. (3) It's computationally trivial — one multiply and one add per update, no history storage needed. The alternative, keeping a rolling window of last N feedbacks, requires O(N) storage and has a cliff effect at the window boundary. EMA avoids both.

---

**Q: What's the difference between `promote()` and `is_shared=True` on a summary?**

A: `is_shared` on a Summary record is a flag that says "this summary has been promoted to the federated pool." The `FederatedMemoryEntry` is the actual shared record in the pool — it's a denormalized copy of the summary content, enriched with federation-specific metadata: `source_agent_id`, `promotion_score`, `access_count`, `last_accessed_at`, and `is_active`. This separation exists for three reasons: (1) The federated entry can have a different `is_active` state than the original summary — the summary remains a permanent record of what was produced, even if the federated entry is retired. (2) Different agents can query the federated pool without needing JOIN queries across sessions and agent namespaces. (3) The access_count and last_accessed_at on the federated entry enable the decay mechanism without mutating the original summary.

---

**Q: How does transitive trust work? Give an example.**

A: Transitive trust allows trust to propagate through intermediary agents. The formula: if Agent A trusts Agent B at 0.9, and Agent B trusts Agent C at 0.8, then A has a transitive path to C through B with score `0.9 × 0.8 = 0.72`. This is blended with the direct trust score: `blended = 0.70 × direct + 0.30 × transitive`. If A has no direct experience with C (direct = 0.5 neutral), the blended score is `0.70 × 0.5 + 0.30 × 0.72 = 0.35 + 0.216 = 0.566` — above neutral.

The 70/30 blend weights direct experience more heavily than inferred trust. This prevents trust from propagating too aggressively through long chains — a chain of 3 hops at 0.9 each would give `0.9^3 = 0.729` transitive, but with 30% weight this only moves the blended score moderately above neutral.

Real-world analogy: if your best friend (0.9 trust) vouches for someone you've never met, you'd extend that person a bit more trust than a random stranger — but you wouldn't trust them as much as your direct friend. That's exactly what the blending computes.

---

**Q: How do you prevent the federated pool from becoming a garbage dump of low-quality memories?**

A: Three gates prevent this. First, the promotion threshold: only memories with `abstraction_level × (1 + confidence_bonus) >= 2.0` get promoted. L1 episodes (score=1.0) never pass. L2 patterns need at least moderate word count (confidence_bonus > 0) to qualify. L3 principles always pass (score=3.0+). Second, the minimum trust threshold in `retrieve()`: by default, memories from agents with trust < 0.3 are filtered out entirely, regardless of their content relevance. Third, the decay mechanism: memories that no other agent has retrieved within 30 days and with fewer than 2 cross-agent accesses are marked `is_active=False` and excluded from all future retrievals. Together these create a natural quality filter: low-quality memories don't attract retrievals, don't get positive feedback, and eventually get retired.

---

**Q: What new benchmark metrics would Phase 4 introduce for a paper?**

A: Two new metrics:

**FFS — Federation Fidelity Score:** Measures how much of a parent agent's semantic knowledge successfully transfers to a forked child and remains useful. Computed as `recall_rate(parent_facts_in_child_context_after_K_turns)`. A high FFS means forking successfully bootstraps the child with relevant prior knowledge.

**TAS — Trust Accuracy Score:** Measures whether trust scores reflect actual memory quality. For each source agent, compute `correlation(trust_score, actual_feedback_signal)` across all interactions. A trust network with high TAS has accurately learned which source agents produce useful memories. This validates that the EMA update rule is working — agents that produce good memories earn higher trust, and that trust correctly up-ranks their future contributions.

---

**Q: What does the complete system look like now with all 4 phases done?**

A: The full system has five layers working together. Tier 1 (Redis working memory) holds the last 5-10 turns for sub-millisecond retrieval. Tier 2 (SQLite episodic memory) stores every turn with importance scores. The Sleep Consolidation Engine runs in the background, scoring turns with the 4-signal composite, clustering with DBSCAN, and generating abstractive summaries that flow into Tier 3 (ChromaDB semantic memory). The Entity Knowledge Graph runs concurrently, extracting entities from every turn and building a NetworkX co-occurrence graph. Tier 4 (procedural memory) mines `(trigger, action)` behavioral patterns from conversation sequences. The Memory Federation Protocol sits above all this — it takes the highest-abstraction memories from the semantic tier and routes them into a trust-weighted shared pool accessible to all agents in the federation. The Context Assembler then pulls from all five sources under strict token budgets, producing a structured XML prompt with `<SEMANTIC MEMORY>`, `<WORLD MODEL>`, `<BEHAVIORAL PATTERNS>`, `<FEDERATED MEMORY>`, and `<RECENT TURNS>` sections. Every component runs locally, for free, with full graceful degradation when optional dependencies are unavailable.

---

*Guide.md — AgentMem OS v0.4.0 | Last updated: April 2026 | All 4 phases complete*
