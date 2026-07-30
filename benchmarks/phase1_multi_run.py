#!/usr/bin/env python3
"""
Phase 1 — Multi-Run Ablation (Runs 2-N)
========================================
Runs the 6-variant ablation N-1 more times and combines with run 1
(ablation_results.json) to produce mean +/- std across N runs.

Output: benchmarks/ablation_multi_run.json   (all N runs + statistics)
        benchmarks/ablation_summary_stats.json (mean +/- std per variant)

Usage:
    python3 benchmarks/phase1_multi_run.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tier_lib import (  # noqa: E402
    R, E, ok, warn, hdr, sub,
    SLEEP_THRESH,
    CONVERSATION, PROBE_RECALLS,
    tok, tfidf_cosine, extract_summary, tes, tes_naive,
    assemble_context, call_claude,
)

for _p in [Path('.'), Path('..'), Path('../..')]:
    if (_p / '.env').exists():
        load_dotenv(_p / '.env')
        break

# Total number of runs to accumulate (matches Mem0's own published
# multi-run rigor — see LAUNCH_ROADMAP.md Phase 1 Group F task 19).
TARGET_RUNS = 10

VARIANTS = [
    ("FULL",        "Full AgentMem OS (all tiers)",             {}),
    ("NO_SEMANTIC", "w/o Semantic Retrieval (Tier 3 disabled)", {"no_semantic": True}),
    ("NO_KG",       "w/o Entity Knowledge Graph",               {"no_kg": True}),
    ("NO_SLEEP",    "w/o Sleep Consolidation",                  {"no_sleep": True}),
    ("NO_PROC",     "w/o Procedural Memory",                    {"no_proc": True}),
    ("RECENT_ONLY", "Recent-only (Tier 1, last 10 turns)",      {"recent_only": True}),
]


def run_variant(client, vname, label, flags, run_id):
    print(f"  [{vname}] run {run_id}", end="", flush=True)
    turns = []
    recall = {}
    n_tok = 0
    b_tok = 0
    sleep_sum = None

    for i, msg in enumerate(CONVERSATION):
        if (not flags.get("no_sleep") and not flags.get("recent_only")
                and sleep_sum is None and len(turns) >= SLEEP_THRESH):
            old = [t for t in turns if t["role"] == "user"]
            sleep_sum = extract_summary(old[:int(len(old) * .6)])

        ctx = assemble_context(turns, msg, flags, sleep_sum)
        naive = sum(tok(t["content"]) for t in turns) + tok(msg)
        ours = tok(ctx) + tok(msg)
        n_tok += ours
        b_tok += naive

        try:
            reply, _, _ = call_claude(client, ctx, msg, max_tokens=400)
        except Exception:
            reply = ""
        turns.append({"role": "user", "content": msg})
        turns.append({"role": "assistant", "content": reply})
        if i in PROBE_RECALLS:
            recall[i] = PROBE_RECALLS[i].lower() in reply.lower()
        time.sleep(0.12)

    print(" ✓")
    lcs = round(sum(recall.values()) / len(PROBE_RECALLS), 4)
    sav = round(100 * (1 - n_tok / max(1, b_tok)), 1)
    user_turns = [t for t in turns if t["role"] == "user"]
    compressed = sleep_sum if sleep_sum and not flags.get("no_sleep") and not flags.get("recent_only") else \
        " ".join(t["content"] for t in user_turns[-max(1, int(len(user_turns) * .7)):])
    tes_ours = tes(user_turns, compressed)
    tes_b = tes_naive(user_turns)
    probe_qs = [CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    our_ctx = (sleep_sum or "") + " " + " ".join(t["content"] for t in turns[-10:])
    crs = round(sum(tfidf_cosine(q, our_ctx) for q in probe_qs) / len(probe_qs), 4)
    mid = len(turns) // 2
    base_ctx = " ".join(t["content"] for t in turns[max(0, mid - 2):mid + 3])
    crs_b = round(sum(tfidf_cosine(q, base_ctx) for q in probe_qs) / len(probe_qs), 4)

    return {"variant": vname, "run": run_id,
            "metrics": {"CRS": {"ours": crs, "baseline": crs_b},
                        "TES": {"ours": tes_ours, "baseline": tes_b},
                        "LCS": {"ours": lcs, "baseline": 0.70}},
            "tokens": {"savings_pct": sav}}


def main():
    hdr("Phase 1 — Multi-Run Ablation")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found")
        sys.exit(1)
    ok(f"API key loaded ({api_key[:12]}...)")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  pip install anthropic")
        sys.exit(1)

    bench_dir = Path(__file__).parent
    # Load existing run 1
    run1_path = bench_dir / "ablation_results.json"
    if run1_path.exists():
        run1 = json.loads(run1_path.read_text())
        for r in run1:
            r["run"] = 1
        all_runs = run1
        ok("Loaded run 1 from ablation_results.json")
    else:
        warn(f"ablation_results.json not found — will run from scratch ({TARGET_RUNS} runs)")
        all_runs = []

    first_new_run = 2 if all_runs else 1
    n_new = (TARGET_RUNS - 1) if all_runs else TARGET_RUNS

    print(f"\n  Running {n_new} new run(s) × {len(VARIANTS)} variants × {len(CONVERSATION)} turns")
    print(f"  Est. cost: ~${n_new * len(VARIANTS) * 0.31:.2f}\n")

    for run_id in range(first_new_run, first_new_run + n_new):
        sub(f"Run {run_id}")
        for vname, label, flags in VARIANTS:
            r = run_variant(client, vname, label, flags, run_id)
            all_runs.append(r)

    # Save all runs
    multi_path = bench_dir / "ablation_multi_run.json"
    multi_path.write_text(json.dumps(all_runs, indent=2))
    ok(f"All runs → {multi_path}")

    # Compute mean +/- std per variant
    total_runs = first_new_run + n_new - 1
    stats = {}
    for vname, _, _ in VARIANTS:
        runs = [r for r in all_runs if r["variant"] == vname]
        for metric in ["CRS", "TES", "LCS"]:
            vals = [r["metrics"][metric]["ours"] for r in runs]
            stats.setdefault(vname, {})
            stats[vname][metric] = {"mean": round(statistics.mean(vals), 4),
                                     "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                                     "n": len(vals)}
        sav_vals = [r["tokens"]["savings_pct"] for r in runs]
        stats[vname]["savings"] = {"mean": round(statistics.mean(sav_vals), 1),
                                    "std": round(statistics.stdev(sav_vals) if len(sav_vals) > 1 else 0.0, 1)}

    stats_path = bench_dir / "ablation_summary_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    ok(f"Stats → {stats_path}")

    # Print summary
    hdr(f"ABLATION SUMMARY  (mean ± std, n={total_runs} runs)")
    print(f"  {'Variant':<18} {'CRS':>12} {'TES':>12} {'LCS':>12} {'Savings':>10}")
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for vname, _, _ in VARIANTS:
        s = stats[vname]

        def fmt(m):
            return f"{s[m]['mean']:.3f}±{s[m]['std']:.3f}"
        print(f"  {vname:<18} {fmt('CRS'):>12} {fmt('TES'):>12} {fmt('LCS'):>12} "
              f"{s['savings']['mean']:>8.1f}%±{s['savings']['std']:.1f}")
    print()
    ok("Phase 1 complete.")


if __name__ == "__main__":
    main()
