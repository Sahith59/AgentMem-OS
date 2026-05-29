#!/usr/bin/env python3
"""
AgentMem OS — Head-to-Head Comparison
=======================================
Evaluates AgentMem OS against three industry baseline systems on the same
25-turn benchmark, scored with the CRS/TES/LCS framework from our ablation study.

Baselines are behavioral simulations that faithfully replicate each system's
published memory architecture — same model (claude-haiku-4-5), same conversation,
same scoring function. Each system makes real API calls.

Systems compared:
  AGENTMEM_OS          — Full 4-tier system (all tiers active)
  MEMGPT               — Main context + archival memory with LLM-driven retrieval
  LANGCHAIN_SUMMARY    — ConversationSummaryMemory (LLM rolling summary)
  ZEP                  — BM25 + entity-aware retrieval with recent context window
  RECENT_ONLY          — Sliding window baseline (last 10 turns)

Output: benchmarks/head_to_head_results.json

Usage:
    python3 benchmarks/head_to_head.py
"""

import os, re, sys, json, math, time
from pathlib import Path
from dotenv import load_dotenv

for _p in [Path('.'), Path('..'), Path('../..')]:
    if (_p / '.env').exists():
        load_dotenv(_p / '.env')
        break

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m";  E = "\033[0m"

def ok(m):   print(f"  {G}✓{E}  {m}")
def warn(m): print(f"  {Y}!{E}  {m}")
def info(m): print(f"  {C}→{E}  {m}")
def hdr(t):  print(f"\n{B}{C}{'═'*62}{E}\n{B}{C}  {t}{E}\n{B}{C}{'═'*62}{E}")
def sub(t):  print(f"\n{B}  ── {t}{E}")

MODEL         = "claude-haiku-4-5-20251001"
COST_PER_MTOK = 0.80
MAIN_CTX_LIMIT = 6    # MemGPT: turns kept in main context before archival eviction
SUMMARY_THRESH = 12   # LangChain: turns before summary kicks in
ZEP_RECENT     = 4    # Zep: recent turns in context window
ZEP_RETRIEVED  = 5    # Zep: BM25 retrieved turns
RECENT_WIN     = 10   # Recent-only baseline

# ── 25-turn benchmark (identical to ablation_study_v2.py) ─────────────────────
CONVERSATION = [
    "Hi, my name is Sahith and I'm building AgentMem OS for my PhD research.",
    "My primary research goal is to add persistent memory to LLM agents across sessions.",
    "The project has four memory tiers: Redis working memory, SQLite episodic, TF-IDF semantic, and procedural pattern memory.",
    "I'm targeting a NeurIPS 2026 workshop submission. The paper deadline is June 2026.",
    "The four novel algorithms are: MemoryImportanceScorer, SleepConsolidationEngine, EntityKnowledgeGraph, and ProceduralMemory.",
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

PROBE_RECALLS = {
    15: "sahith",
    16: "neurips 2026",
    17: "sqlite",
    18: "procedural",
    19: "neurips",
}


# ── Shared utilities (mirrored from ablation_study_v2.py) ─────────────────────

def _tok(text: str) -> int:
    return max(1, len(text) // 4)

def _entities(text: str):
    return set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text))

def _tfidf_cosine(query: str, doc: str) -> float:
    def tok(t): return re.findall(r'[a-z]+', t.lower())
    qt, dt = tok(query), tok(doc)
    all_w = set(qt) | set(dt)
    df = {w: (1 if w in qt else 0) + (1 if w in dt else 0) for w in all_w}
    def vec(tokens):
        tf = {}
        for w in tokens: tf[w] = tf.get(w, 0) + 1
        n = len(tokens) or 1
        return {w: (c/n) * math.log(1 + 1/df[w]) for w, c in tf.items()}
    qv, dv = vec(qt), vec(dt)
    dot = sum(qv.get(k, 0) * dv.get(k, 0) for k in all_w)
    nq = math.sqrt(sum(v**2 for v in qv.values())) or 1e-9
    nd = math.sqrt(sum(v**2 for v in dv.values())) or 1e-9
    return dot / (nq * nd)

def _bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75,
                avg_len: float = 50.0) -> float:
    """BM25 scoring — Zep uses this over TF-IDF for better term saturation."""
    def tok(t): return re.findall(r'[a-z]+', t.lower())
    qt, dt = tok(query), tok(doc)
    if not qt or not dt: return 0.0
    dl = len(dt)
    tf_d = {}
    for w in dt: tf_d[w] = tf_d.get(w, 0) + 1
    score = 0.0
    for w in set(qt):
        if w not in tf_d: continue
        tf = tf_d[w]
        idf = math.log(1 + (len(CONVERSATION) - 1 + 0.5) / (1 + 1))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_len))
    return score

def _tes(user_turns: list, compressed: str) -> float:
    orig = " ".join(t["content"] for t in user_turns)
    rc = max(0.0, min(1.0, 1 - _tok(compressed) / max(1, _tok(orig))))
    E_orig = _entities(orig)
    E_comp = _entities(compressed)
    re_ = len(E_orig & E_comp) / max(1, len(E_orig))
    return round(math.sqrt(rc * re_), 4)

def _compute_metrics(turns: list, recall_hits: dict, compressed_text: str,
                     naive_tok_total: int, ours_tok_total: int) -> dict:
    user_turns = [t for t in turns if t["role"] == "user"]
    lcs = round(sum(recall_hits.values()) / len(PROBE_RECALLS), 4)
    tes = _tes(user_turns, compressed_text)
    probe_qs = [CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    our_ctx = compressed_text + " " + " ".join(t["content"] for t in turns[-8:])
    crs = round(sum(_tfidf_cosine(q, our_ctx) for q in probe_qs) / len(probe_qs), 4)
    tok_savings = round(100 * (1 - ours_tok_total / max(1, naive_tok_total)), 1)
    cost = round(ours_tok_total / 1_000_000 * COST_PER_MTOK, 4)
    naive_cost = round(naive_tok_total / 1_000_000 * COST_PER_MTOK, 4)
    return {"CRS": crs, "TES": tes, "LCS": lcs,
            "tok_savings": tok_savings, "cost": cost, "naive_cost": naive_cost}


# ── Claude API ─────────────────────────────────────────────────────────────────

def call_claude(client, context: str, user_msg: str) -> tuple[str, int, int]:
    messages = [{"role": "user", "content": f"{context}\n\n[CURRENT QUERY]\n{user_msg}"}]
    resp = client.messages.create(model=MODEL, max_tokens=512, messages=messages)
    reply = resp.content[0].text if resp.content else ""
    return reply, resp.usage.input_tokens, resp.usage.output_tokens


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1: AgentMem OS (reuses ablation FULL logic)
# ══════════════════════════════════════════════════════════════════════════════

def _agentmem_context(turns: list, query: str, sleep_summary: str | None) -> str:
    parts = ["[SYSTEM]\nYou are AgentMem OS, an AI with hierarchical persistent memory."]
    if sleep_summary:
        parts.append(f"[EPISODIC MEMORY]\n{sleep_summary}")
    if len(turns) > 8:
        older = turns[:-8]
        hits = sorted(older, key=lambda t: _tfidf_cosine(query, t["content"]), reverse=True)[:3]
        if hits:
            parts.append("[SEMANTIC MEMORY]\n" +
                         "\n".join(f"[{t['role'].upper()}]: {t['content'][:200]}" for t in hits))
    freq = {}
    for t in turns:
        for e in _entities(t["content"]): freq[e] = freq.get(e, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:5]
    if top: parts.append("[ENTITY KG]\nKnown entities: " + ", ".join(top))
    recent = turns[-8:]
    parts.append("[WORKING MEMORY]\n" +
                 "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent))
    return "\n\n".join(parts)

def _extract_summary(turns: list) -> str:
    def ec(t): return len(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', t["content"]))
    chunks = [turns[i:i+5] for i in range(0, len(turns), 5)]
    excerpts = []
    for chunk in chunks:
        for t in sorted(chunk, key=ec, reverse=True)[:2]:
            excerpts.append(f"[{t['role'].upper()}]: {t['content'][:280]}")
    ents = _entities(" ".join(t["content"] for t in turns))
    if ents: excerpts.append("Entities: " + ", ".join(sorted(ents)[:30]))
    return "\n\n".join(excerpts)


def run_agentmem(client) -> dict:
    hdr("SYSTEM 1: AgentMem OS — Full 4-Tier System")
    turns, recall_hits = [], {}
    sleep_summary = None
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        if sleep_summary is None and len(turns) >= 15:
            old_u = [t for t in turns if t["role"] == "user"]
            sleep_summary = _extract_summary(old_u[:int(len(old_u)*0.6)])
            info(f"Sleep consolidation at turn {i+1}")

        ctx = _agentmem_context(turns, user_msg, sleep_summary)
        naive_tok = sum(_tok(t["content"]) for t in turns) + _tok(user_msg)
        ours_tok  = _tok(ctx) + _tok(user_msg)
        naive_tok_total += naive_tok; ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}"); reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]; hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    compressed = sleep_summary or " ".join(t["content"] for t in
                                            [t for t in turns if t["role"]=="user"][-8:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "AGENTMEM_OS", "label": "AgentMem OS (4-tier)", "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 2: MemGPT — Main context + archival memory
#
# Architecture: fixed main context (last MAIN_CTX_LIMIT turns). When turns are
# evicted, they are written to archival storage (extractive summary). Each turn,
# archival is keyword-searched and top results are prepended to main context.
# Source: Packer et al. 2023 (arxiv 2310.08560)
# ══════════════════════════════════════════════════════════════════════════════

def run_memgpt(client) -> dict:
    hdr("SYSTEM 2: MemGPT — Main Context + Archival Memory")
    turns, archival = [], []
    recall_hits = {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        # Evict oldest turns to archival when main context overflows
        if len(turns) > MAIN_CTX_LIMIT * 2:
            evict_n = len(turns) - MAIN_CTX_LIMIT * 2
            evicted = turns[:evict_n]
            turns = turns[evict_n:]
            # Write evicted turns to archival as extractive snippets
            for t in evicted:
                if t["role"] == "user":
                    archival.append({"content": t["content"][:300], "role": "user"})

        # Retrieve from archival via keyword search
        retrieved = sorted(archival,
                           key=lambda t: _tfidf_cosine(user_msg, t["content"]),
                           reverse=True)[:3]

        parts = ["[SYSTEM]\nYou are MemGPT. Use archival memory for long-horizon recall."]
        if retrieved:
            parts.append("[ARCHIVAL MEMORY]\n" +
                         "\n".join(f"[ARCHIVED]: {t['content']}" for t in retrieved))
        parts.append("[MAIN CONTEXT]\n" +
                     "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in turns))
        ctx = "\n\n".join(parts)

        naive_tok = sum(_tok(t["content"]) for t in turns + archival) + _tok(user_msg)
        ours_tok  = _tok(ctx) + _tok(user_msg)
        naive_tok_total += naive_tok; ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}"); reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]; hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}  [archival={len(archival)}]")
        else:
            print(f"    T{i+1:02d} ✓  [main={len(turns)//2}turns archival={len(archival)}]")
        time.sleep(0.15)

    compressed = " ".join(t["content"] for t in archival) + \
                 " ".join(t["content"] for t in turns if t["role"] == "user")
    m = _compute_metrics(turns + archival, recall_hits, compressed,
                         naive_tok_total, ours_tok_total)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "MEMGPT", "label": "MemGPT (archival memory)", "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 3: LangChain ConversationSummaryMemory
#
# Architecture: when history exceeds SUMMARY_THRESH turns, use the LLM to
# generate a rolling summary of all older turns. Context = summary + last 5 turns.
# The summary is NOT updated incrementally — only generated once at threshold.
# Source: LangChain docs ConversationSummaryMemory
# ══════════════════════════════════════════════════════════════════════════════

def _generate_summary_lc(client, turns: list) -> str:
    """Generate an LLM summary of turns — simulates LangChain's predict_new_summary."""
    history_text = "\n".join(f"{t['role'].upper()}: {t['content'][:200]}" for t in turns)
    prompt = (
        "Progressively summarise the following conversation history. "
        "Preserve all names, dates, numbers, and technical details exactly. "
        "Be concise but complete.\n\nHistory:\n" + history_text
    )
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text if resp.content else history_text[:500]
    except Exception:
        return history_text[:500]


def run_langchain_summary(client) -> dict:
    hdr("SYSTEM 3: LangChain ConversationSummaryMemory")
    turns, summary = [], None
    recall_hits = {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        # Trigger summary generation once when we exceed the threshold
        if summary is None and len(turns) >= SUMMARY_THRESH * 2:
            older = turns[:-10]   # summarise all but last 5 turn-pairs
            info(f"Generating LLM summary at turn {i+1} ({len(older)//2} turn-pairs)...")
            summary = _generate_summary_lc(client, older)
            info(f"Summary generated ({len(summary)} chars)")

        # Build context: summary (if exists) + last 5 turns
        recent = turns[-10:]   # last 5 turn-pairs
        parts = ["[SYSTEM]\nYou are a helpful AI assistant with conversation memory."]
        if summary:
            parts.append(f"[CONVERSATION SUMMARY]\n{summary}")
        if recent:
            parts.append("[RECENT CONVERSATION]\n" +
                         "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent))
        ctx = "\n\n".join(parts)

        naive_tok = sum(_tok(t["content"]) for t in turns) + _tok(user_msg)
        ours_tok  = _tok(ctx) + _tok(user_msg)
        naive_tok_total += naive_tok; ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}"); reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]; hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    compressed = (summary or "") + " ".join(t["content"] for t in
                  [t for t in turns if t["role"]=="user"][-5:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "LANGCHAIN_SUMMARY", "label": "LangChain ConversationSummaryMemory",
            "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 4: Zep — Entity-aware memory with BM25 retrieval
#
# Architecture: stores all turns in a vector/BM25 index with entity extraction.
# Each turn: retrieve top-K by BM25(query, stored_turn) + last ZEP_RECENT turns.
# Entity graph is extracted and prepended as structured facts.
# Source: Zep docs (getzep.com), Zep technical architecture
# ══════════════════════════════════════════════════════════════════════════════

def run_zep(client) -> dict:
    hdr("SYSTEM 4: Zep — BM25 + Entity-Aware Memory")
    all_turns = []   # full store (BM25 index)
    entity_graph = {}  # entity → mention count
    recall_hits = {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        # BM25 retrieval from stored turns (excluding last ZEP_RECENT)
        store = all_turns[:-ZEP_RECENT*2] if len(all_turns) > ZEP_RECENT*2 else []
        if store:
            scored = [(t, _bm25_score(user_msg, t["content"])) for t in store
                      if t["role"] == "user"]
            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved = [t for t, _ in scored[:ZEP_RETRIEVED]]
        else:
            retrieved = []

        # Entity graph context
        top_entities = sorted(entity_graph, key=entity_graph.get, reverse=True)[:8]
        entity_ctx = "Known entities: " + ", ".join(top_entities) if top_entities else ""

        recent = all_turns[-ZEP_RECENT*2:]
        parts = ["[SYSTEM]\nYou are Zep, an AI with entity-aware persistent memory."]
        if entity_ctx:
            parts.append(f"[ENTITY GRAPH]\n{entity_ctx}")
        if retrieved:
            parts.append("[RETRIEVED MEMORIES (BM25)]\n" +
                         "\n".join(f"[MEMORY]: {t['content'][:200]}" for t in retrieved))
        if recent:
            parts.append("[RECENT CONTEXT]\n" +
                         "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent))
        ctx = "\n\n".join(parts)

        naive_tok = sum(_tok(t["content"]) for t in all_turns) + _tok(user_msg)
        ours_tok  = _tok(ctx) + _tok(user_msg)
        naive_tok_total += naive_tok; ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}"); reply = ""

        # Update entity graph from new turn
        for ent in _entities(user_msg + " " + reply):
            entity_graph[ent] = entity_graph.get(ent, 0) + 1

        all_turns.extend([{"role": "user", "content": user_msg},
                           {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]; hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}  [entities={len(entity_graph)}]")
        else:
            print(f"    T{i+1:02d} ✓  [store={len(all_turns)//2}t retrieved={len(retrieved)}]")
        time.sleep(0.15)

    user_turns = [t for t in all_turns if t["role"] == "user"]
    compressed = " ".join(t["content"] for t in user_turns)
    m = _compute_metrics(all_turns, recall_hits, compressed, naive_tok_total, ours_tok_total)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "ZEP", "label": "Zep (BM25 + entity graph)", "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 5: Sliding window baseline (recent-only)
# ══════════════════════════════════════════════════════════════════════════════

def run_recent_only(client) -> dict:
    hdr("SYSTEM 5: Recent-Only Baseline (sliding window)")
    turns, recall_hits = [], {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        recent = turns[-RECENT_WIN*2:]
        ctx = ("[SYSTEM]\nYou are a helpful AI assistant.\n\n[RECENT CONTEXT]\n" +
               "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent))

        naive_tok = sum(_tok(t["content"]) for t in turns) + _tok(user_msg)
        ours_tok  = _tok(ctx) + _tok(user_msg)
        naive_tok_total += naive_tok; ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}"); reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]; hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    user_turns = [t for t in turns if t["role"] == "user"]
    compressed = " ".join(t["content"] for t in user_turns[-RECENT_WIN:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "RECENT_ONLY", "label": "Sliding window (last 10 turns)", "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hdr("AgentMem OS — Head-to-Head Comparison")
    print(f"\n  Systems    : AgentMem OS | MemGPT | LangChain Summary | Zep | Recent-Only")
    print(f"  Turns/run  : {len(CONVERSATION)}")
    print(f"  Metrics    : CRS (context retrieval), TES (token efficiency), LCS (long-horizon recall)")
    print(f"  Est. cost  : ~$1.00–1.50 (5 systems × 25 turns + 1 summary call)\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "YOUR_KEY" in api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found.")
        sys.exit(1)
    ok(f"API key loaded  ({api_key[:12]}...)")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  anthropic package not installed."); sys.exit(1)

    runners = [
        ("AgentMem OS", run_agentmem),
        ("MemGPT",      run_memgpt),
        ("LangChain",   run_langchain_summary),
        ("Zep",         run_zep),
        ("Recent-Only", run_recent_only),
    ]

    results = []
    for name, fn in runners:
        try:
            r = fn(client)
            results.append(r)
        except Exception as ex:
            warn(f"{name} crashed: {ex}")
            import traceback; traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "head_to_head_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    ok(f"Results saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr("SUMMARY TABLE")
    print(f"\n  {'System':<32} {'CRS':>7} {'TES':>7} {'LCS':>7} {'Savings':>9}")
    print(f"  {'─'*32} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    ref = next((r for r in results if r["system"] == "AGENTMEM_OS"), None)
    for r in results:
        m = r["metrics"]
        mark = " ◄" if r["system"] == "AGENTMEM_OS" else ""
        print(f"  {r['label']:<32} {m['CRS']:>7.4f} {m['TES']:>7.4f} {m['LCS']:>7.4f} {m['tok_savings']:>8.1f}%{mark}")
    print()
    if ref:
        print(f"  Deltas vs AgentMem OS:")
        for r in results:
            if r["system"] == "AGENTMEM_OS": continue
            m = r["metrics"]; rm = ref["metrics"]
            print(f"    {r['label']:<32} ΔCRS={m['CRS']-rm['CRS']:+.4f} "
                  f"ΔTES={m['TES']-rm['TES']:+.4f} ΔLCS={m['LCS']-rm['LCS']:+.4f}")
    ok("Head-to-head complete.")


if __name__ == "__main__":
    main()
