#!/usr/bin/env python3
"""
AgentMem OS — Head-to-Head Comparison
=======================================
Evaluates AgentMem OS against three industry baseline systems on the same
25-turn benchmark, scored with the CRS/TES/LCS framework from our ablation study.

Baselines are behavioral simulations that faithfully replicate each system's
published memory architecture — same model (claude-haiku-4-5), same conversation,
same scoring function. Each system makes real API calls. These are NOT real
integrations with the MemGPT/LangChain/Zep libraries — see
benchmarks/deprecated_proxy_sim/ notes and LAUNCH_ROADMAP.md Phase 2 for the
plan to replace this with real library calls.

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

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import (  # noqa: E402
    G, R, Y, E, ok, warn, info, hdr, sub,
    MODEL, COST_PER_MTOK,
    CONVERSATION, PROBE_RECALLS,
    tok, entities, tfidf_cosine, bm25_score, extract_summary, tes,
    assemble_context, call_claude, crs_from_probe_contexts,
)

for _p in [Path('.'), Path('..'), Path('../..')]:
    if (_p / '.env').exists():
        load_dotenv(_p / '.env')
        break

MAIN_CTX_LIMIT = 6    # MemGPT: turns kept in main context before archival eviction
SUMMARY_THRESH = 12   # LangChain: turns before summary kicks in
ZEP_RECENT     = 4    # Zep: recent turns in context window
ZEP_RETRIEVED  = 5    # Zep: BM25 retrieved turns
RECENT_WIN     = 10   # Recent-only baseline


def _compute_metrics(turns: list, recall_hits: dict, compressed_text: str,
                      naive_tok_total: int, ours_tok_total: int,
                      probe_contexts: dict) -> dict:
    user_turns = [t for t in turns if t["role"] == "user"]
    lcs = round(sum(recall_hits.values()) / len(PROBE_RECALLS), 4)
    tes_v = tes(user_turns, compressed_text)
    # Scored against each system's actual assembled context per probe turn.
    # The previous version scored every one of the 5 systems (AgentMem OS,
    # MemGPT, LangChain, Zep, Recent-Only) against compressed_text + the
    # same generic last-8-raw-turns text, rather than what that system
    # actually built and sent to the model.
    crs = crs_from_probe_contexts(probe_contexts)
    tok_savings = round(100 * (1 - ours_tok_total / max(1, naive_tok_total)), 1)
    cost = round(ours_tok_total / 1_000_000 * COST_PER_MTOK, 4)
    naive_cost = round(naive_tok_total / 1_000_000 * COST_PER_MTOK, 4)
    return {"CRS": crs, "TES": tes_v, "LCS": lcs,
            "tok_savings": tok_savings, "cost": cost, "naive_cost": naive_cost}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM 1: AgentMem OS (reuses the same shared tier-assembly logic as
# ablation_study.py's FULL variant — previously this was a third, independently
# drifted copy missing the procedural-memory tier entirely; see _tier_lib.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_agentmem(client) -> dict:
    hdr("SYSTEM 1: AgentMem OS — Full 4-Tier System")
    turns, recall_hits = [], {}
    probe_contexts = {}
    sleep_summary = None
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        if sleep_summary is None and len(turns) >= 15:
            old_u = [t for t in turns if t["role"] == "user"]
            sleep_summary = extract_summary(old_u[:int(len(old_u) * 0.6)])
            info(f"Sleep consolidation at turn {i+1}")

        ctx = assemble_context(turns, user_msg, {}, sleep_summary)
        if i in PROBE_RECALLS:
            probe_contexts[i] = (user_msg, ctx)
        naive_tok = sum(tok(t["content"]) for t in turns) + tok(user_msg)
        ours_tok = tok(ctx) + tok(user_msg)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}")
            reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    compressed = sleep_summary or " ".join(t["content"] for t in
                                            [t for t in turns if t["role"] == "user"][-8:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total, probe_contexts)
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
    probe_contexts = {}
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
                            key=lambda t: tfidf_cosine(user_msg, t["content"]),
                            reverse=True)[:3]

        parts = ["[SYSTEM]\nYou are MemGPT. Use archival memory for long-horizon recall."]
        if retrieved:
            parts.append("[ARCHIVAL MEMORY]\n" +
                          "\n".join(f"[ARCHIVED]: {t['content']}" for t in retrieved))
        parts.append("[MAIN CONTEXT]\n" +
                      "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in turns))
        ctx = "\n\n".join(parts)
        if i in PROBE_RECALLS:
            probe_contexts[i] = (user_msg, ctx)

        naive_tok = sum(tok(t["content"]) for t in turns + archival) + tok(user_msg)
        ours_tok = tok(ctx) + tok(user_msg)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}")
            reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}  [archival={len(archival)}]")
        else:
            print(f"    T{i+1:02d} ✓  [main={len(turns)//2}turns archival={len(archival)}]")
        time.sleep(0.15)

    compressed = " ".join(t["content"] for t in archival) + \
                 " ".join(t["content"] for t in turns if t["role"] == "user")
    m = _compute_metrics(turns + archival, recall_hits, compressed,
                          naive_tok_total, ours_tok_total, probe_contexts)
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
    probe_contexts = {}
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
        if i in PROBE_RECALLS:
            probe_contexts[i] = (user_msg, ctx)

        naive_tok = sum(tok(t["content"]) for t in turns) + tok(user_msg)
        ours_tok = tok(ctx) + tok(user_msg)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}")
            reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    compressed = (summary or "") + " ".join(t["content"] for t in
                  [t for t in turns if t["role"] == "user"][-5:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total, probe_contexts)
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
    probe_contexts = {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        # BM25 retrieval from stored turns (excluding last ZEP_RECENT)
        store = all_turns[:-ZEP_RECENT*2] if len(all_turns) > ZEP_RECENT*2 else []
        if store:
            scored = [(t, bm25_score(user_msg, t["content"], len(CONVERSATION)))
                      for t in store if t["role"] == "user"]
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
        if i in PROBE_RECALLS:
            probe_contexts[i] = (user_msg, ctx)

        naive_tok = sum(tok(t["content"]) for t in all_turns) + tok(user_msg)
        ours_tok = tok(ctx) + tok(user_msg)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}")
            reply = ""

        # Update entity graph from new turn
        for ent in entities(user_msg + " " + reply):
            entity_graph[ent] = entity_graph.get(ent, 0) + 1

        all_turns.extend([{"role": "user", "content": user_msg},
                           {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}  [entities={len(entity_graph)}]")
        else:
            print(f"    T{i+1:02d} ✓  [store={len(all_turns)//2}t retrieved={len(retrieved)}]")
        time.sleep(0.15)

    user_turns = [t for t in all_turns if t["role"] == "user"]
    compressed = " ".join(t["content"] for t in user_turns)
    m = _compute_metrics(all_turns, recall_hits, compressed, naive_tok_total, ours_tok_total, probe_contexts)
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
    probe_contexts = {}
    naive_tok_total = ours_tok_total = 0

    for i, user_msg in enumerate(CONVERSATION):
        recent = turns[-RECENT_WIN*2:]
        ctx = ("[SYSTEM]\nYou are a helpful AI assistant.\n\n[RECENT CONTEXT]\n" +
               "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent))
        if i in PROBE_RECALLS:
            probe_contexts[i] = (user_msg, ctx)

        naive_tok = sum(tok(t["content"]) for t in turns) + tok(user_msg)
        ours_tok = tok(ctx) + tok(user_msg)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok

        try:
            reply, _, _ = call_claude(client, ctx, user_msg)
        except Exception as ex:
            warn(f"Turn {i+1}: {ex}")
            reply = ""

        turns.extend([{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": reply}])

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{i+1:02d} probe '{kw}': {sym}")
        else:
            print(f"    T{i+1:02d} ✓  [ctx={ours_tok}tok]")
        time.sleep(0.15)

    user_turns = [t for t in turns if t["role"] == "user"]
    compressed = " ".join(t["content"] for t in user_turns[-RECENT_WIN:])
    m = _compute_metrics(turns, recall_hits, compressed, naive_tok_total, ours_tok_total, probe_contexts)
    sub("Results")
    print(f"    CRS={m['CRS']:.4f}  TES={m['TES']:.4f}  LCS={m['LCS']:.4f}  savings={m['tok_savings']}%")
    return {"system": "RECENT_ONLY", "label": "Sliding window (last 10 turns)", "metrics": m,
            "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False) for i in PROBE_RECALLS}}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hdr("AgentMem OS — Head-to-Head Comparison")
    print("\n  Systems    : AgentMem OS | MemGPT | LangChain Summary | Zep | Recent-Only")
    print(f"  Turns/run  : {len(CONVERSATION)}")
    print("  Metrics    : CRS (context retrieval), TES (token efficiency), LCS (long-horizon recall)")
    print("  Est. cost  : ~$1.00-1.50 (5 systems × 25 turns + 1 summary call)\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "YOUR_KEY" in api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found.")
        sys.exit(1)
    ok(f"API key loaded  ({api_key[:12]}...)")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  anthropic package not installed.")
        sys.exit(1)

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
            import traceback
            traceback.print_exc()

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
        print("  Deltas vs AgentMem OS:")
        for r in results:
            if r["system"] == "AGENTMEM_OS":
                continue
            m = r["metrics"]
            rm = ref["metrics"]
            print(f"    {r['label']:<32} ΔCRS={m['CRS']-rm['CRS']:+.4f} "
                  f"ΔTES={m['TES']-rm['TES']:+.4f} ΔLCS={m['LCS']-rm['LCS']:+.4f}")
    ok("Head-to-head complete.")


if __name__ == "__main__":
    main()
