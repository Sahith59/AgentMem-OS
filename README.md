# AgentMem OS (MemNAI)

**A local-first, persistent memory operating system for long-horizon LLM agents.**

## Overview

Modern LLM agents forget everything when a conversation ends. AgentMem OS solves this with a four-tier memory hierarchy that persists knowledge across sessions, compresses it intelligently, and retrieves the most relevant context at inference time — all running locally with no cloud dependencies.

The system ships four novel ML algorithms as its core contributions, each targeting a different failure mode of naive long-context approaches (truncation, full-history replay, and flat retrieval).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Agent                            │
│               (Claude / Ollama / any LiteLLM model)         │
└──────────────────────┬──────────────────────────────────────┘
                       │  query
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Context Assembler                          │
│          (MMR retrieval across all 4 tiers)                 │
└────┬────────────┬───────────────┬──────────────┬────────────┘
     │            │               │              │
     ▼            ▼               ▼              ▼
┌─────────┐  ┌─────────┐  ┌───────────┐  ┌──────────────┐
│  Redis  │  │ SQLite  │  │ ChromaDB  │  │  Procedural  │
│ Tier 1  │  │ Tier 2  │  │  Tier 3   │  │   Tier 4     │
│Hot Cache│  │Episodic │  │ Semantic  │  │   Patterns   │
│ <100ms  │  │ History │  │ Retrieval │  │   Library    │
└─────────┘  └─────────┘  └───────────┘  └──────────────┘
```

### Memory Tiers

| Tier | Backend | Role | Latency |
|------|---------|------|---------|
| 1 — Working Memory | Redis | Recent turns, immediate context | < 5 ms |
| 2 — Episodic Memory | SQLite | Full session history, structured recall | < 20 ms |
| 3 — Semantic Memory | ChromaDB | Vector similarity search across all sessions | < 50 ms |
| 4 — Procedural Memory | SQLite + NetworkX | Recurring interaction patterns | < 30 ms |

---

## Novel ML Algorithms

### 1. `MemoryImportanceScorer`
Scores each conversation turn with an EMA-weighted importance signal combining entity density, semantic novelty, and recency decay. Drives selective retention — only high-signal turns survive consolidation.

```python
from agentmem_os.llm.importance_scorer import MemoryImportanceScorer
scorer = MemoryImportanceScorer(get_db)
score = scorer.score_turn(session_id, turn_content, role="user")
```

### 2. `SleepConsolidationEngine`
Runs offline compression using DBSCAN clustering over turn embeddings. Groups semantically similar turns into clusters, extracts representative summaries, and writes them to the Summary table — analogous to hippocampal replay during sleep.

```python
from agentmem_os.llm.consolidation_engine import SleepConsolidationEngine
engine = SleepConsolidationEngine(get_db, summarizer, chroma, scorer, get_embedding)
engine.consolidate(session_id)
```

### 3. `EntityKnowledgeGraph`
Builds a persistent co-occurrence graph (NetworkX) of named entities extracted from conversation turns. Supports subgraph retrieval for world-model queries, updated incrementally as new turns arrive.

```python
from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
kg = EntityKnowledgeGraph(get_db)
subgraph = kg.get_relevant_subgraph("Tell me about Sahith AgentMem", agent_id=None)
```

### 4. `ProceduralMemory`
Mines recurring interaction patterns from session history (e.g., user always asks for code before explanation). Patterns are scored by frequency and recency, retrieved at inference time to pre-shape responses.

```python
from agentmem_os.llm.procedural_memory import ProceduralMemory
pm = ProceduralMemory(get_db)
patterns = pm.get_relevant_patterns("explain research methodology", agent_id=None)
```

---

## Benchmark Metrics

Three novel metrics designed for memory-augmented LLM evaluation:

### CRS — Context Relevance Score
Measures how relevant the assembled memory context is to the current query vs. a random baseline context.

```
CRS = cosine_sim(embed(query), embed(assembled_context))
      vs.
      cosine_sim(embed(query), embed(random_context))
```

### TES — Token Efficiency Score
Measures compression quality: how much token reduction is achieved while preserving key entities.

```
TES = √(compression_ratio × entity_preservation_rate)

compression_ratio      = 1 - (tokens_after / tokens_before)
entity_preservation    = |entities_in_summary ∩ entities_in_original| / |entities_in_original|
```

### LCS — Long-Horizon Continuity Score
Measures whether the agent can answer factual questions about things said K turns ago — the core capability that distinguishes a memory system from naive context truncation.

```
LCS = (facts correctly recalled with AgentMem OS) / (total facts seeded)
baseline = (facts recalled with recent-only context)
```

---

## Project Structure

```
agentmem_os/
├── agents/                  # Multi-agent memory federation
│   ├── memory_federation.py
│   ├── namespace_manager.py
│   └── trust_network.py
├── api/                     # FastAPI REST interface
│   └── app.py
├── benchmarks/
│   └── eval_harness.py      # CRS / TES / LCS evaluators
├── cache/
│   └── redis_client.py      # Tier 1: Redis working memory
├── cli/
│   └── main.py              # Typer CLI
├── db/
│   ├── chroma_client.py     # Tier 3: ChromaDB semantic store
│   ├── engine.py            # SQLAlchemy engine + session factory
│   ├── knowledge_graph.py   # EntityKnowledgeGraph (NetworkX)
│   └── models.py            # Turn, Session, Summary, CostLog, Pattern
├── llm/
│   ├── adapters.py          # LiteLLM universal adapter + prompt caching
│   ├── consolidation_engine.py   # SleepConsolidationEngine (DBSCAN)
│   ├── context_assembler.py      # MMR retrieval across all tiers
│   ├── importance_scorer.py      # MemoryImportanceScorer (EMA)
│   ├── procedural_memory.py      # ProceduralMemory (pattern mining)
│   ├── summarizer.py             # Extractive + LLM-based summarizer
│   └── token_counter.py          # tiktoken wrapper
├── storage/
│   └── store.py             # ConversationStore (coordinates all tiers)
├── tests/
│   └── test_e2e_claude.py   # Full end-to-end benchmark test
├── config.yaml
├── requirements.txt
└── .env.example
```

---

## Installation

**Requirements:** Python 3.11+, Redis running locally, optional Ollama for local embeddings.

```bash
# Clone
git clone https://github.com/yourusername/agentmem-os.git
cd agentmem-os

# Virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (or GROQ_API_KEY for free tier)

# Database
python -c "from agentmem_os.db.engine import init_db; init_db()"
```

---

## Running the End-to-End Benchmark

The E2E test runs a full 25-turn conversation with Claude, then evaluates all three benchmark metrics:

```bash
# Start Redis (required for Tier 1)
redis-server &

# Optional: start Ollama for best CRS scores (768-dim embeddings)
ollama pull nomic-embed-text

# Run
python tests/test_e2e_claude.py
```

**What the test does:**
- Turns 1–5: Seeds 5 grounding facts (name, project, tiers, algorithms, deadline)
- Turns 6–15: Work turns that push the grounding turns beyond the context window
- Turns 16–25: Probe turns with no hints — agent must retrieve from memory
- Step 6: Forces sleep consolidation to generate summaries for TES
- Step 7: Verifies Entity Knowledge Graph population
- Step 8: Mines procedural patterns
- Step 9: Measures token cost and prompt caching savings
- Step 10: Evaluates CRS / TES / LCS against baselines

**Example output:**
```
╔══════════════════════════════════════════════════╗
║   AgentMem OS — Benchmark Report                 ║
╠══════════════════════════════════════════════════╣
║ Metric      :  Ours   Baseline      Δ            ║
║ CRS         : 0.6821    0.4103  +0.2718  ↑       ║
║ TES         : 0.5940    0.3211  +0.2729  ↑       ║
║ LCS         : 1.0000    0.0000  +1.0000  ↑       ║
╚══════════════════════════════════════════════════╝
```

---

## Configuration

Edit `config.yaml` to change models and storage paths:

```yaml
models:
  default_model: "anthropic/claude-haiku-4-5-20251001"   # cheapest Claude
  fallback_model: "ollama/llama3.1"                       # local fallback
  compression_threshold: 0.70                             # trigger consolidation at 70% context

storage:
  base_path: "~/.agentmem_os/"
```

Supported model strings (LiteLLM format):

| Model | String | Use case |
|-------|--------|----------|
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5-20251001` | E2E testing (cheap) |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` | Best benchmark quality |
| Llama 3.1 (local) | `ollama/llama3.1` | Free, no API key needed |
| Groq Llama | `groq/llama-3.1-8b-instant` | Free tier fallback |

---

## Cost Efficiency

A key claim of the paper is that AgentMem OS reduces API token costs through prompt caching and aggressive context compression. The E2E test measures this automatically:

```
Input tokens    : 45,230
Cached tokens   : 38,190   (84.4% of input — charged at 10% rate)
Est. session cost: $0.0089
Cache savings   : 75.8% reduction in effective input token cost
```

Prompt caching works because AgentMem OS always places the system context (assembled memory) at the beginning of the message, making it eligible for Anthropic's cache prefix matching.

---

## Research Context

This project is being developed as part of PhD research on persistent memory architectures for LLM agents. The NeurIPS 2026 workshop paper will include:

- Formal definitions of CRS, TES, and LCS metrics
- Ablation study: individual contribution of each memory tier
- Comparison against baselines: full history replay, sliding window, RAG-only
- Cost-efficiency analysis across session lengths

---

## Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...      # Required for Claude models
GROQ_API_KEY=gsk_...              # Optional: free fallback
OLLAMA_BASE_URL=http://localhost:11434   # Optional: local embeddings
REDIS_URL=redis://localhost:6379  # Default Redis location
```

---

## License

MIT License — see `LICENSE` file.

---
