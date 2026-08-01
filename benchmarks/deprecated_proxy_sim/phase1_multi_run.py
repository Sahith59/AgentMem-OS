#!/usr/bin/env python3
"""
DEPRECATED — behavioral simulation, not a real ablation of AgentMem OS.
Superseded by benchmarks/ablation_study_real.py. Kept only for historical
reference — do not cite these numbers. See LAUNCH_ROADMAP.md Phase 2.

Phase 1 — Multi-Run Ablation (Runs 1-N, fresh each invocation)
================================================================
Runs the 6-variant ablation N times to produce mean +/- std across N runs.

Output: benchmarks/deprecated_proxy_sim/ablation_multi_run.json   (all N runs + statistics)
        benchmarks/deprecated_proxy_sim/ablation_summary_stats.json (mean +/- std per variant)

Usage:
    python3 benchmarks/deprecated_proxy_sim/phase1_multi_run.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # _tier_lib.py lives one level up
from _tier_lib import (  # noqa: E402
    R, E, ok, hdr, sub,
    SLEEP_THRESH,
    CONVERSATION, PROBE_RECALLS,
    tok, extract_summary, tes, tes_naive,
    assemble_context, call_claude,
    crs_from_probe_contexts, patch_baselines_from_recent_only,
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
    probe_contexts = {}  # {turn_index: (query, assembled_context)} — for CRS
    n_tok = 0
    b_tok = 0
    sleep_sum = None

    for i, msg in enumerate(CONVERSATION):
        if (not flags.get("no_sleep") and not flags.get("recent_only")
                and sleep_sum is None and len(turns) >= SLEEP_THRESH):
            old = [t for t in turns if t["role"] == "user"]
            sleep_sum = extract_summary(old[:int(len(old) * .6)])

        ctx = assemble_context(turns, msg, flags, sleep_sum)
        if i in PROBE_RECALLS:
            probe_contexts[i] = (msg, ctx)
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
    # CRS scored from the actual assembled context per probe turn (Group B
    # fix); LCS/CRS baselines are patched from RECENT_ONLY's own measured
    # score for this run in main(), not a hardcoded 0.70.
    crs = crs_from_probe_contexts(probe_contexts)

    return {"variant": vname, "run": run_id,
            "metrics": {"CRS": {"ours": crs, "baseline": 0.0},
                        "TES": {"ours": tes_ours, "baseline": tes_b},
                        "LCS": {"ours": lcs, "baseline": 0.0}},
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

    # NOTE: this intentionally does NOT load/append to the old
    # ablation_results.json — that file was produced by the pre-fix
    # ablation_study.py (hardcoded LCS baseline, tier-blind CRS) and its
    # numbers aren't compatible with runs produced after the Group B fixes.
    # Every run here is fresh, all TARGET_RUNS of them.
    all_runs = []
    print(f"\n  Running {TARGET_RUNS} run(s) × {len(VARIANTS)} variants × {len(CONVERSATION)} turns")
    print(f"  Est. cost: ~${TARGET_RUNS * len(VARIANTS) * 0.31:.2f}\n")

    for run_id in range(1, TARGET_RUNS + 1):
        sub(f"Run {run_id}")
        run_results = []
        for vname, label, flags in VARIANTS:
            r = run_variant(client, vname, label, flags, run_id)
            run_results.append(r)
        # Patch this run's LCS/CRS baselines from ITS OWN RECENT_ONLY
        # result — baselines must not leak across runs, since each run's
        # RECENT_ONLY score is itself noisy (real API calls).
        patch_baselines_from_recent_only(run_results)
        all_runs.extend(run_results)

    # Save all runs
    multi_path = bench_dir / "ablation_multi_run.json"
    multi_path.write_text(json.dumps(all_runs, indent=2))
    ok(f"All runs → {multi_path}")

    # Compute mean +/- std per variant
    total_runs = TARGET_RUNS
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
