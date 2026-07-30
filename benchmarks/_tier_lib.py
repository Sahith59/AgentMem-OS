#!/usr/bin/env python3
"""
Shared primitives for AgentMem OS's standalone benchmark scripts.

These scripts (ablation_study.py, phase1_multi_run.py, phase2_long_horizon.py,
phase3_baselines.py, head_to_head.py) are self-contained proxy simulations of
AgentMem OS's tier architecture — zero dependency on the real `agentmem_os`
package — used for fast, cheap, offline-friendly architecture comparisons.
See ablation_study_real.py for an ablation that exercises the real
ContextAssembler instead, which is the number that should be cited in the
paper; these scripts are explicitly labeled as simulations wherever their
results are reported.

Consolidated here because five near-identical copies of the same ~150 lines
of helpers (only cosmetically renamed per file) were the direct mechanism
that let ablation_study_v2.py exist as an undetected duplicate of
ablation_study.py, and were the reason bugs had to be fixed in five places
instead of one.
"""

import math
import re

# ── Terminal colours / print helpers ────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m";  E = "\033[0m"


def ok(m):   print(f"  {G}✓{E}  {m}")
def warn(m): print(f"  {Y}!{E}  {m}")
def info(m): print(f"  {C}→{E}  {m}")
def hdr(t):  print(f"\n{B}{C}{'═'*60}{E}\n{B}{C}  {t}{E}\n{B}{C}{'═'*60}{E}")
def sub(t):  print(f"\n{B}  ── {t}{E}")

# ── Shared constants ─────────────────────────────────────────────────────────
MODEL          = "claude-haiku-4-5-20251001"
COST_PER_MTOK  = 0.80      # USD / 1M input tokens
SLEEP_THRESH   = 15        # compress after this many turns
RECENT_WIN     = 8         # turns kept as raw recent context
KG_TOP_N       = 5         # top KG entities to inject
SEMANTIC_TOP_N = 3         # top semantic turns to inject

# ── 25-turn benchmark conversation ───────────────────────────────────────────
# Shared by ablation_study.py, phase1_multi_run.py, phase3_baselines.py,
# and head_to_head.py. phase2_long_horizon.py uses its own 50-turn variant
# (CONVERSATION_50) since it's deliberately testing a different horizon.
CONVERSATION = [
    # Grounding turns (T1-5) — seeded facts for long-horizon recall
    "Hi, my name is Sahith and I'm building AgentMem OS for my PhD research.",
    "My primary research goal is to add persistent memory to LLM agents across sessions.",
    "The project has four memory tiers: Redis working memory, SQLite episodic, TF-IDF semantic, and procedural pattern memory.",
    "I'm targeting a NeurIPS 2026 workshop submission. The paper deadline is June 2026.",
    "The four novel algorithms are: MemoryImportanceScorer, SleepConsolidationEngine, EntityKnowledgeGraph, and ProceduralMemory.",
    # Work turns (T6-15)
    "Can you explain how DBSCAN clustering helps group semantically similar memories?",
    "What's the difference between episodic and semantic memory in cognitive science?",
    "How does prompt caching in Claude reduce API costs for long conversations?",
    "Please write a short Python function to compute cosine similarity between two vectors.",
    "What are the trade-offs between TF-IDF and dense vector retrieval for local memory systems?",
    "Explain the concept of memory importance scoring — how would you rank conversation turns?",
    "What evaluation metrics are typically used in memory-augmented language model papers?",
    "How does retrieval-augmented generation differ from a full persistent memory system?",
    "Can you describe the PageRank algorithm and how it applies to knowledge graphs?",
    "What is the typical architecture of a sleep consolidation system in AI memory research?",
    # Long-horizon probes (T16-25) — no in-turn hints, rely on memory
    "Without me reminding you, what is my name?",
    "What specific research deadline am I working towards, including the month and year?",
    "Can you list all four memory tiers in the system I described earlier?",
    "Name all four novel ML algorithms I mentioned at the start of our conversation.",
    "What is the conference name and year I plan to submit this work to?",
    "What makes a good NeurIPS workshop paper — what do reviewers typically look for?",
    "How would you structure an ablation study for a memory system like AgentMem OS?",
    "What baseline systems should I compare against in my evaluation section?",
    "Can you help me draft a one-sentence summary of the AgentMem OS contribution?",
    "Final question: what are the three benchmark metrics — CRS, TES, and LCS?",
]

# Probe indices (0-based) -> keyword that must appear in the reply to count
# as a recall hit. Fixed and identical across every run and every script —
# this is what Group B's eval_harness.py fix (a fixed probe-query set)
# mirrors for the real-embedding CRS evaluator.
PROBE_RECALLS = {
    15: "sahith",
    16: "neurips 2026",
    17: "sqlite",
    18: "procedural",
    19: "neurips",
}

# ── Token / entity / similarity primitives ───────────────────────────────────


def tok(text: str) -> int:
    """Rough token count (chars/4)."""
    return max(1, len(text) // 4)


def entities(text: str) -> set:
    """Extract capitalized words as a proxy for named entities."""
    return set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text))


def tfidf_cosine(query: str, doc: str) -> float:
    """Pure-Python TF-IDF cosine similarity (no sklearn needed)."""
    def _tokenize(t):
        return re.findall(r'[a-z]+', t.lower())
    qt, dt = _tokenize(query), _tokenize(doc)
    all_w = set(qt) | set(dt)
    df = {w: (1 if w in qt else 0) + (1 if w in dt else 0) for w in all_w}

    def _vec(tokens):
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        n = len(tokens) or 1
        return {w: (c / n) * math.log(1 + 1 / df[w]) for w, c in tf.items()}

    qv, dv = _vec(qt), _vec(dt)
    dot = sum(qv.get(k, 0) * dv.get(k, 0) for k in all_w)
    nq = math.sqrt(sum(v ** 2 for v in qv.values())) or 1e-9
    nd = math.sqrt(sum(v ** 2 for v in dv.values())) or 1e-9
    return dot / (nq * nd)


def bm25_score(query: str, doc: str, corpus_size: int,
               k1: float = 1.5, b: float = 0.75, avg_len: float = 50.0) -> float:
    """BM25 scoring — used to simulate Zep's retrieval in head_to_head.py."""
    def _tokenize(t):
        return re.findall(r'[a-z]+', t.lower())
    qt, dt = _tokenize(query), _tokenize(doc)
    if not qt or not dt:
        return 0.0
    dl = len(dt)
    tf_d = {}
    for w in dt:
        tf_d[w] = tf_d.get(w, 0) + 1
    score = 0.0
    for w in set(qt):
        if w not in tf_d:
            continue
        tf = tf_d[w]
        idf = math.log(1 + (corpus_size - 1 + 0.5) / (1 + 1))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_len))
    return score


def extract_summary(turns: list) -> str:
    """
    Simulate Sleep Consolidation:
    - Cluster turns in groups of 5 (proxy for DBSCAN clusters)
    - Keep top-2 entity-rich turns per cluster (extractive)
    - Append entity inventory (proxy for KG snapshot)
    """
    def _entity_count(t):
        return len(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', t["content"]))

    chunks = [turns[i:i + 5] for i in range(0, len(turns), 5)]
    excerpts = []
    for chunk in chunks:
        for t in sorted(chunk, key=_entity_count, reverse=True)[:2]:
            excerpts.append(f"[{t['role'].upper()}]: {t['content'][:280]}")
    all_ents = entities(" ".join(t["content"] for t in turns))
    if all_ents:
        excerpts.append("Entities: " + ", ".join(sorted(all_ents)[:30]))
    return "\n\n".join(excerpts)


def tes(user_turns: list, compressed_text: str) -> float:
    """TES = sqrt(compression_ratio x entity_preservation_rate), clamped to [0,1]."""
    orig = " ".join(t["content"] for t in user_turns)
    rc = max(0.0, min(1.0, 1 - tok(compressed_text) / max(1, tok(orig))))
    e_orig = entities(orig)
    e_comp = entities(compressed_text)
    preservation = len(e_orig & e_comp) / max(1, len(e_orig))
    return round(math.sqrt(rc * preservation), 4)


def tes_naive(user_turns: list) -> float:
    """Naive TES baseline: keep newest 70% unchanged."""
    keep = user_turns[-max(1, int(len(user_turns) * 0.7)):]
    return tes(user_turns, " ".join(t["content"] for t in keep))


def semantic_retrieve(query: str, turns: list, top_n: int) -> list:
    """TF-IDF top-N retrieval from turn history."""
    scored = [(t, tfidf_cosine(query, t["content"])) for t in turns]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def kg_entities(turns: list, top_n: int) -> str:
    """Simulate KG: return the most-mentioned entities across history."""
    freq = {}
    for t in turns:
        for e in entities(t["content"]):
            freq[e] = freq.get(e, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:top_n]
    return "Known entities: " + ", ".join(top) if top else ""


def procedural_patterns(turns: list) -> str:
    """Simulate Procedural Memory: detect repeated user intent patterns."""
    patterns = []
    user_msgs = [t["content"] for t in turns if t["role"] == "user"]
    if any("explain" in m.lower() for m in user_msgs):
        patterns.append("User frequently asks for explanations → provide structured answers.")
    if any("python" in m.lower() or "code" in m.lower() for m in user_msgs):
        patterns.append("User requests code examples → include runnable snippets.")
    if any("research" in m.lower() or "paper" in m.lower() for m in user_msgs):
        patterns.append("User is in research mode → cite definitions and relate to literature.")
    return "\n".join(patterns)


# ── Context assembler (simulates AgentMem OS's 4-tier assembly) ─────────────

def assemble_context(turns: list, query: str, flags: dict,
                      sleep_summary: str | None) -> str:
    """
    Build the assembled context string for a given ablation variant.

    Tiers included based on flags:
      Tier 1 (Working)  — always: last RECENT_WIN turns
      Tier 2 (Episodic) — always: sleep summary (if available)
      Tier 3 (Semantic) — disabled if no_semantic
      Tier 4a (KG)      — disabled if no_kg
      Tier 4b (Proc)    — disabled if no_proc
      recent_only       — Tier 1 only (last 10 turns)
    """
    parts = []
    sys_prompt = (
        "You are AgentMem OS, an AI assistant with hierarchical persistent memory. "
        "Use the memory context below to answer accurately, especially for long-horizon questions."
    )
    parts.append(f"[SYSTEM]\n{sys_prompt}")

    if flags.get("recent_only"):
        recent = turns[-10:]
        turns_txt = "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent)
        parts.append(f"[RECENT TURNS (last 10)]\n{turns_txt}")
        return "\n\n".join(parts)

    # Tier 2 — Episodic / sleep summary
    if sleep_summary and not flags.get("no_sleep"):
        parts.append(f"[EPISODIC MEMORY — Consolidated Summary]\n{sleep_summary}")

    # Tier 3 — Semantic retrieval
    if not flags.get("no_semantic") and len(turns) > RECENT_WIN:
        older = turns[:-RECENT_WIN]
        semantic_hits = semantic_retrieve(query, older, SEMANTIC_TOP_N)
        if semantic_hits:
            sem_txt = "\n".join(
                f"[{t['role'].upper()}]: {t['content'][:200]}" for t in semantic_hits
            )
            parts.append(f"[SEMANTIC MEMORY — Top-{SEMANTIC_TOP_N} Relevant Turns]\n{sem_txt}")

    # Tier 4a — KG entities
    if not flags.get("no_kg") and turns:
        kg_txt = kg_entities(turns, KG_TOP_N)
        if kg_txt:
            parts.append(f"[ENTITY KNOWLEDGE GRAPH]\n{kg_txt}")

    # Tier 4b — Procedural patterns
    if not flags.get("no_proc") and turns:
        proc = procedural_patterns(turns)
        if proc:
            parts.append(f"[PROCEDURAL MEMORY — Behavioural Patterns]\n{proc}")

    # Tier 1 — Recent working memory (always last)
    recent = turns[-RECENT_WIN:]
    turns_txt = "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent)
    parts.append(f"[WORKING MEMORY — Last {RECENT_WIN} Turns]\n{turns_txt}")

    return "\n\n".join(parts)


# ── Claude API call ───────────────────────────────────────────────────────────

def call_claude(client, context: str, user_msg: str, max_tokens: int = 512) -> tuple[str, int, int]:
    """Returns (reply_text, input_tokens, output_tokens)."""
    messages = [
        {"role": "user", "content": f"{context}\n\n[CURRENT QUERY]\n{user_msg}"}
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    reply = resp.content[0].text if resp.content else ""
    return reply, resp.usage.input_tokens, resp.usage.output_tokens
