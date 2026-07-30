#!/usr/bin/env python3
"""
Phase 3 — Baseline Comparison
===============================
Compares AgentMem OS (FULL) against three external baselines over 25 turns:

  FULL          — AgentMem OS with all 4 tiers active
  FULL_HISTORY  — Naive: send ALL conversation turns to model (no compression)
                  Simulates a system with no memory management — token cost grows
                  unboundedly, shows why compression is necessary.
  NAIVE_RAG     — TF-IDF top-5 retrieval only (no sleep, no KG, no proc, no recent)
                  Simulates a plain retrieval-augmented system without our tiers.
  RECENT_ONLY   — Sliding window last 10 turns (most common simple baseline)

Output: benchmarks/baseline_comparison.json

Usage:
    python3 benchmarks/phase3_baselines.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import (  # noqa: E402
    G, R, Y, E, ok, warn, hdr, sub,
    COST_PER_MTOK, SLEEP_THRESH,
    CONVERSATION, PROBE_RECALLS,
    tok, tfidf_cosine, extract_summary, tes,
    assemble_context, call_claude,
)

for _p in [Path('.'), Path('..'), Path('../..')]:
    if (_p / '.env').exists():
        load_dotenv(_p / '.env')
        break

BASELINES = [
    ("AGENTMEM_OS",  "AgentMem OS — Full System (all 4 tiers)", "ours"),
    ("FULL_HISTORY", "Full History — All turns, no compression", "full_history"),
    ("NAIVE_RAG",    "Naive RAG — TF-IDF retrieval only",        "naive_rag"),
    ("RECENT_ONLY",  "Recent-Only — Sliding window, last 10",    "recent_only"),
]


def build_context(turns, query, mode, sleep_sum):
    """Build context string for each baseline mode."""
    sys_p = "You are an AI assistant. Use the conversation context to answer accurately."

    if mode == "ours":
        # AgentMem OS: reuse the shared, flag-aware tier assembler (no flags
        # disabled here — this is the FULL variant of the same simulation
        # ablation_study.py runs, so both scripts now score AgentMem OS
        # identically instead of via two independently-drifting copies).
        return assemble_context(turns, query, {}, sleep_sum)

    elif mode == "full_history":
        # Send EVERYTHING — no compression at all
        all_txt = "".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in turns)
        return f"{sys_p}\n\n[FULL CONVERSATION HISTORY]\n{all_txt}"

    elif mode == "naive_rag":
        # TF-IDF top-5 only — no recent, no sleep, no KG, no proc
        if len(turns) < 2:
            return f"{sys_p}\n\n[No history yet]"
        hits = sorted(turns, key=lambda t: tfidf_cosine(query, t["content"]), reverse=True)[:5]
        hits_txt = "".join(f"[{t['role'].upper()}]: {t['content'][:300]}\n" for t in hits)
        return f"{sys_p}\n\n[TOP-5 RETRIEVED TURNS]\n{hits_txt}"

    elif mode == "recent_only":
        recent = turns[-10:]
        recent_txt = "".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in recent)
        return f"{sys_p}\n\n[RECENT TURNS (last 10)]\n{recent_txt}"

    return sys_p


def run_baseline(client, name, label, mode):
    hdr(f"{name}  —  {label}")
    sub("Running 25-turn conversation...")
    turns = []
    recall = {}
    n_tok = 0
    b_tok = 0
    sleep_sum = None

    for i, msg in enumerate(CONVERSATION):
        turn_num = i + 1

        # Sleep consolidation for OURS only
        if (mode == "ours" and sleep_sum is None and len(turns) >= SLEEP_THRESH):
            old = [t for t in turns if t["role"] == "user"]
            sleep_sum = extract_summary(old[:int(len(old) * .6)])
            print(f"    → Sleep consolidation at T{turn_num}")

        ctx = build_context(turns, msg, mode, sleep_sum)
        naive = sum(tok(t["content"]) for t in turns) + tok(msg)
        ours = tok(ctx) + tok(msg)
        n_tok += ours
        b_tok += naive

        try:
            reply, _, _ = call_claude(client, ctx, msg, max_tokens=400)
        except Exception as ex:
            warn(f"T{turn_num}: {ex}")
            reply = ""

        turns.append({"role": "user", "content": msg})
        turns.append({"role": "assistant", "content": reply})

        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{turn_num:02d} {sym} probe '{kw}' | ctx={ours}tok naive={naive}tok")
        else:
            print(f"    T{turn_num:02d} ✓ | ctx={ours}tok")
        time.sleep(0.12)

    lcs = round(sum(recall.values()) / len(PROBE_RECALLS), 4)
    sav = round(100 * (1 - n_tok / max(1, b_tok)), 1)
    user_turns = [t for t in turns if t["role"] == "user"]
    if mode == "full_history":
        compressed = " ".join(t["content"] for t in user_turns)
    elif mode == "naive_rag":
        compressed = " ".join(t["content"] for t in user_turns[-max(1, int(len(user_turns) * .5)):])
    elif mode == "recent_only":
        compressed = " ".join(t["content"] for t in user_turns[-max(1, int(len(user_turns) * .3)):])
    else:  # ours
        compressed = sleep_sum if sleep_sum else " ".join(t["content"] for t in user_turns)
    tes_v = tes(user_turns, compressed)
    probe_qs = [CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    our_ctx = (sleep_sum or "") + " " + " ".join(t["content"] for t in turns[-10:])
    crs = round(sum(tfidf_cosine(q, our_ctx) for q in probe_qs) / len(probe_qs), 4)

    sub("Results")
    cost = round(n_tok / 1_000_000 * COST_PER_MTOK, 4)
    naive_cost = round(b_tok / 1_000_000 * COST_PER_MTOK, 4)
    print(f"    LCS : {lcs:.4f}  ({sum(recall.values())}/{len(PROBE_RECALLS)} probes recalled)")
    print(f"    TES : {tes_v:.4f}")
    print(f"    CRS : {crs:.4f}")
    print(f"    Tok savings: {sav:.1f}%  (${cost:.4f} vs ${naive_cost:.4f} naive)")

    return {"baseline": name, "label": label, "mode": mode,
            "metrics": {"CRS": crs, "TES": tes_v, "LCS": lcs},
            "tokens": {"savings_pct": sav, "cost": cost, "naive_cost": naive_cost},
            "recall": {PROBE_RECALLS[i]: recall.get(i, False) for i in PROBE_RECALLS}}


def main():
    hdr("Phase 3 — Baseline Comparison")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"  {R}✗{E}  Set ANTHROPIC_API_KEY")
        sys.exit(1)
    ok(f"API key ({api_key[:12]}...)")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  pip install anthropic")
        sys.exit(1)

    print(f"\n  4 baselines × {len(CONVERSATION)} turns  |  Est. cost ~${4*0.31:.2f}\n")
    results = []
    for name, label, mode in BASELINES:
        r = run_baseline(client, name, label, mode)
        results.append(r)

    out = Path(__file__).parent / "baseline_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    ok(f"Results → {out}")

    hdr("BASELINE COMPARISON TABLE")
    print(f"  {'System':<18} {'LCS':>7} {'TES':>7} {'CRS':>7} {'Savings':>9} {'Cost':>8}")
    print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*8}")
    for r in results:
        m = r["metrics"]
        t = r["tokens"]
        print(f"  {r['baseline']:<18} {m['LCS']:>7.4f} {m['TES']:>7.4f} {m['CRS']:>7.4f} "
              f"{t['savings_pct']:>8.1f}% ${t['cost']:.4f}")
    print()

    # Show deltas vs AGENTMEM_OS
    ours = next(r for r in results if r["baseline"] == "AGENTMEM_OS")
    print("  Deltas vs AgentMem OS (negative = AgentMem OS wins):")
    for r in results:
        if r["baseline"] == "AGENTMEM_OS":
            continue
        m = r["metrics"]
        om = ours["metrics"]
        dl = round(m["LCS"] - om["LCS"], 3)
        dt = round(m["TES"] - om["TES"], 3)
        dc = round(m["CRS"] - om["CRS"], 4)
        print(f"  {r['baseline']:<18} ΔLCS={dl:+.3f}  ΔTES={dt:+.3f}  ΔCRS={dc:+.4f}")
    ok("Phase 3 complete.")


if __name__ == "__main__":
    main()
