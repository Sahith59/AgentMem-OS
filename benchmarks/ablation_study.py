#!/usr/bin/env python3
"""
AgentMem OS — Ablation Study
=============================
Self-contained simulation: zero dependency on agentmem_os.* internals.
Only requires: anthropic, python-dotenv  (both in your venv)

Six variants isolate each component's contribution:
  FULL        — All four tiers active (baseline)
  NO_SEMANTIC — Disable TF-IDF semantic retrieval (Tier 3)
  NO_KG       — Disable Entity Knowledge Graph (Tier 4a)
  NO_SLEEP    — Disable Sleep Consolidation (no compression)
  NO_PROC     — Disable Procedural Memory (Tier 4b)
  RECENT_ONLY — Last 10 turns only (Tier 1, pure recency)

This script is a behavioral simulation of AgentMem OS's tier architecture,
not the real ContextAssembler — see ablation_study_real.py for an ablation
that exercises the real package. Shared helpers live in _tier_lib.py.

Usage (from ANY directory — .env auto-located):
    python3 benchmarks/ablation_study.py

Results: benchmarks/ablation_results.json
Cost:    ~6 x $0.31 ~= $1.86
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
    COST_PER_MTOK, SLEEP_THRESH,
    CONVERSATION, PROBE_RECALLS,
    tok, tfidf_cosine, extract_summary, tes, tes_naive,
    assemble_context, call_claude,
)

# ── Locate .env (walk up from this file's location) ───────────────────────────
_here = Path(__file__).resolve()
for _p in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    _env = _p / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
        break

VARIANTS = [
    ("FULL",        "Full AgentMem OS (all tiers)",             {}),
    ("NO_SEMANTIC", "w/o Semantic Retrieval (Tier 3 disabled)", {"no_semantic": True}),
    ("NO_KG",       "w/o Entity Knowledge Graph",               {"no_kg": True}),
    ("NO_SLEEP",    "w/o Sleep Consolidation",                  {"no_sleep": True}),
    ("NO_PROC",     "w/o Procedural Memory",                    {"no_proc": True}),
    ("RECENT_ONLY", "Recent-only (Tier 1, last 10 turns)",      {"recent_only": True}),
]

# ══════════════════════════════════════════════════════════════════════════════
# Core: run one ablation variant
# ══════════════════════════════════════════════════════════════════════════════


def run_variant(client, variant_name: str, label: str, flags: dict) -> dict:
    hdr(f"VARIANT: {variant_name}  —  {label}")
    sub("Running 25-turn conversation...")

    turns = []          # full history: {role, content}
    recall_hits = {}
    naive_tok_total = 0
    ours_tok_total = 0
    token_log = []
    sleep_summary = None    # set once sleep consolidation triggers

    for i, user_msg in enumerate(CONVERSATION):
        turn_num = i + 1

        # ── Sleep Consolidation trigger (>= SLEEP_THRESH turns, once) ─────────
        if (not flags.get("no_sleep") and not flags.get("recent_only")
                and sleep_summary is None and len(turns) >= SLEEP_THRESH):
            old_turns = [t for t in turns if t["role"] == "user"]
            compress_n = int(len(old_turns) * 0.6)
            sleep_summary = extract_summary(old_turns[:compress_n])
            info(f"Sleep consolidation fired at turn {turn_num} "
                 f"({compress_n} turns compressed)")

        # ── Assemble context for this variant ─────────────────────────────────
        context = assemble_context(turns, user_msg, flags, sleep_summary)

        # ── Naive token count (full raw history) ──────────────────────────────
        naive_tok = sum(tok(t["content"]) for t in turns) + tok(user_msg)
        ours_tok = tok(context) + tok(user_msg)

        savings = round(100 * (1 - ours_tok / max(1, naive_tok)), 1)
        naive_tok_total += naive_tok
        ours_tok_total += ours_tok
        token_log.append({"turn": turn_num, "naive": naive_tok,
                           "ours": ours_tok, "savings_pct": savings})

        # ── Call Claude ────────────────────────────────────────────────────────
        try:
            reply, in_toks, out_toks = call_claude(client, context, user_msg)
        except Exception as ex:
            warn(f"Turn {turn_num} API error: {ex}")
            reply = ""

        turns.append({"role": "user",      "content": user_msg})
        turns.append({"role": "assistant", "content": reply})

        # ── Long-horizon recall check ──────────────────────────────────────────
        if i in PROBE_RECALLS:
            kw = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{turn_num:02d} probe '{kw}': {sym}  "
                  f"[ctx={ours_tok}tok, saved {savings:+.0f}%]")
        else:
            print(f"    T{turn_num:02d} ✓  [ctx={ours_tok}tok, naive={naive_tok}tok]")

        time.sleep(0.15)   # gentle rate-limiting

    # ── LCS ───────────────────────────────────────────────────────────────────
    lcs_ours = round(sum(recall_hits.values()) / len(PROBE_RECALLS), 4)
    lcs_base = 0.70    # deterministic recent-8 baseline

    # ── TES ───────────────────────────────────────────────────────────────────
    user_turns = [t for t in turns if t["role"] == "user"]
    if flags.get("no_sleep") or flags.get("recent_only") or sleep_summary is None:
        compressed_text = " ".join(t["content"] for t in
                                    user_turns[-max(1, int(len(user_turns) * 0.7)):])
        tes_ours = tes(user_turns, compressed_text)
    else:
        tes_ours = tes(user_turns, sleep_summary)
    tes_base = tes_naive(user_turns)

    # ── CRS ───────────────────────────────────────────────────────────────────
    probe_qs = [CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    # Our context: sleep summary + last 10 turns text
    our_ctx = (sleep_summary or "") + " " + " ".join(
        t["content"] for t in turns[-10:])
    crs_ours = round(sum(tfidf_cosine(q, our_ctx) for q in probe_qs)
                      / len(probe_qs), 4)
    # Baseline: middle-5-turns only
    mid = len(turns) // 2
    base_ctx = " ".join(t["content"] for t in turns[max(0, mid - 2):mid + 3])
    crs_base = round(sum(tfidf_cosine(q, base_ctx) for q in probe_qs)
                      / len(probe_qs), 4)

    # ── Token savings ─────────────────────────────────────────────────────────
    tok_savings = round(100 * (1 - ours_tok_total / max(1, naive_tok_total)), 1)
    cost_ours = round(ours_tok_total / 1_000_000 * COST_PER_MTOK, 4)
    cost_naive = round(naive_tok_total / 1_000_000 * COST_PER_MTOK, 4)

    sub("Results")
    print(f"    CRS : {crs_ours:.4f}  (base {crs_base:.4f},  Δ {crs_ours-crs_base:+.4f})")
    print(f"    TES : {tes_ours:.4f}  (base {tes_base:.4f},  Δ {tes_ours-tes_base:+.4f})")
    print(f"    LCS : {lcs_ours:.4f}  (base {lcs_base:.4f},  Δ {lcs_ours-lcs_base:+.4f})")
    print(f"    Tok : {tok_savings:+.1f}% savings  "
          f"(${cost_ours:.4f} vs ${cost_naive:.4f} naive)")

    return {
        "variant": variant_name,
        "label": label,
        "flags": flags,
        "metrics": {
            "CRS": {"ours": crs_ours, "baseline": crs_base,
                    "delta": round(crs_ours - crs_base, 4)},
            "TES": {"ours": tes_ours, "baseline": tes_base,
                    "delta": round(tes_ours - tes_base, 4)},
            "LCS": {"ours": lcs_ours, "baseline": lcs_base,
                    "delta": round(lcs_ours - lcs_base, 4)},
        },
        "tokens": {
            "ours_total": ours_tok_total,
            "naive_total": naive_tok_total,
            "savings_pct": tok_savings,
            "cost_ours": cost_ours,
            "cost_naive": cost_naive,
        },
        "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False)
                   for i in PROBE_RECALLS},
        "token_log": token_log,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    hdr("AgentMem OS — Ablation Study")
    print(f"\n  Variants   : {len(VARIANTS)}")
    print(f"  Turns/var  : {len(CONVERSATION)}")
    print(f"  Est. cost  : ~${len(VARIANTS)*0.31:.2f} total\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "YOUR_KEY" in api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found.\n"
              f"     Run:  set -a && source .env && set +a   then re-run.")
        sys.exit(1)
    ok(f"API key loaded  ({api_key[:12]}...)")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  'anthropic' package not installed.\n"
              f"     Run:  pip install anthropic")
        sys.exit(1)

    results = []
    for vname, label, flags in VARIANTS:
        try:
            r = run_variant(client, vname, label, flags)
            results.append(r)
        except Exception as ex:
            warn(f"Variant {vname} crashed: {ex}")
            import traceback
            traceback.print_exc()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    ok(f"Results → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr("ABLATION SUMMARY TABLE")
    print(f"  {'Variant':<18} {'CRS':>7} {'TES':>7} {'LCS':>7} {'Savings':>9}")
    print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    full_r = next((r for r in results if r["variant"] == "FULL"), None)
    for r in results:
        m = r["metrics"]
        sav = r["tokens"]["savings_pct"]
        crs = m["CRS"]["ours"]
        tes_v = m["TES"]["ours"]
        lcs = m["LCS"]["ours"]
        print(f"  {r['variant']:<18} {crs:>7.4f} {tes_v:>7.4f} {lcs:>7.4f} {sav:>8.1f}%")
        if full_r and r["variant"] != "FULL":
            dc = round(crs - full_r["metrics"]["CRS"]["ours"], 3)
            dt = round(tes_v - full_r["metrics"]["TES"]["ours"], 3)
            dl = round(lcs - full_r["metrics"]["LCS"]["ours"], 3)
            print(f"  {'':18} ΔCRS={dc:+.3f}  ΔTES={dt:+.3f}  ΔLCS={dl:+.3f}")
        else:
            print(f"  {'':18} ← full system (reference)")
    print()
    ok("Ablation study complete.")


if __name__ == "__main__":
    main()
