#!/usr/bin/env python3
"""
AgentMem OS — Multi-Run Benchmark Aggregator
=============================================
Runs the E2E benchmark N times and computes mean ± std for each metric.
Outputs a paper-ready results table and saves all individual reports.

Usage:
    python benchmarks/run_multi_bench.py              # default 5 runs
    python benchmarks/run_multi_bench.py --runs 3     # custom count
"""

import subprocess
import json
import sys
import os
import statistics
from pathlib import Path
from datetime import datetime

N_RUNS = 5

# Parse --runs arg
for i, arg in enumerate(sys.argv):
    if arg == "--runs" and i + 1 < len(sys.argv):
        N_RUNS = int(sys.argv[i + 1])

ROOT = Path(__file__).parent.parent
TEST_SCRIPT = ROOT / "tests" / "test_e2e_claude.py"
REPORTS_DIR = ROOT / "benchmarks" / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def run_single(run_idx: int) -> dict:
    """Run one benchmark and return the report dict."""
    print(f"\n{BOLD}{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}{CYAN}  RUN {run_idx + 1} / {N_RUNS}{RESET}")
    print(f"{BOLD}{CYAN}{'='*55}{RESET}\n")

    result = subprocess.run(
        [sys.executable, str(TEST_SCRIPT)],
        cwd=str(ROOT),
        capture_output=False,
    )

    report_path = ROOT / "benchmarks" / "latest_report.json"
    if not report_path.exists():
        print(f"  {YELLOW}! Run {run_idx + 1} produced no report{RESET}")
        return None

    with open(report_path) as f:
        report = json.load(f)

    # Save individual report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = REPORTS_DIR / f"run_{run_idx + 1}_{ts}.json"
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def aggregate(reports: list) -> dict:
    """Compute mean ± std across runs for each metric."""
    metrics = {}
    for report in reports:
        for r in report.get("results", []):
            name = r["metric"]
            if name not in metrics:
                metrics[name] = {"ours": [], "baseline": [], "delta": []}
            metrics[name]["ours"].append(r["score"])
            metrics[name]["baseline"].append(r["baseline_score"])
            metrics[name]["delta"].append(r["improvement"])

    agg = {}
    for name, vals in metrics.items():
        n = len(vals["ours"])
        agg[name] = {
            "n": n,
            "ours_mean": statistics.mean(vals["ours"]),
            "ours_std": statistics.stdev(vals["ours"]) if n > 1 else 0.0,
            "baseline_mean": statistics.mean(vals["baseline"]),
            "baseline_std": statistics.stdev(vals["baseline"]) if n > 1 else 0.0,
            "delta_mean": statistics.mean(vals["delta"]),
            "delta_std": statistics.stdev(vals["delta"]) if n > 1 else 0.0,
        }
    return agg


def print_table(agg: dict, n_runs: int):
    """Print a paper-ready results table."""
    print(f"\n{BOLD}{CYAN}{'='*62}{RESET}")
    print(f"{BOLD}{CYAN}  AgentMem OS — Aggregated Benchmark Results ({n_runs} runs){RESET}")
    print(f"{BOLD}{CYAN}{'='*62}{RESET}")
    print()
    print(f"  {'Metric':<8}  {'Ours':>14}  {'Baseline':>14}  {'Δ Improvement':>16}")
    print(f"  {'─'*56}")

    for metric in ("CRS", "TES", "LCS"):
        if metric in agg:
            a = agg[metric]
            ours_str = f"{a['ours_mean']:.4f}±{a['ours_std']:.4f}"
            base_str = f"{a['baseline_mean']:.4f}±{a['baseline_std']:.4f}"
            delta_str = f"{a['delta_mean']:+.4f}±{a['delta_std']:.4f}"
            status = f"{GREEN}↑{RESET}" if a["delta_mean"] > 0 else f"{YELLOW}↓{RESET}"
            print(f"  {metric:<8}  {ours_str:>14}  {base_str:>14}  {delta_str:>16}  {status}")

    print()

    # Overall
    overall_scores = [r.get("overall_score", 0) for r in reports if r]
    if overall_scores:
        mean_overall = statistics.mean(overall_scores)
        std_overall = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        print(f"  Overall Score: {BOLD}{mean_overall:.4f} ± {std_overall:.4f}{RESET}")
    print()


def save_aggregated(agg: dict, n_runs: int):
    """Save aggregated results to JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "n_runs": n_runs,
        "timestamp": ts,
        "metrics": agg,
    }
    path = REPORTS_DIR / f"aggregated_{n_runs}runs_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {path.relative_to(ROOT)}")

    # Also save as latest
    latest = ROOT / "benchmarks" / "aggregated_latest.json"
    with open(latest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → benchmarks/aggregated_latest.json")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}AgentMem OS — Multi-Run Benchmark{RESET}")
    print(f"Running {N_RUNS} iterations...\n")

    reports = []
    for i in range(N_RUNS):
        report = run_single(i)
        if report:
            reports.append(report)

    if not reports:
        print(f"\n{YELLOW}No successful runs.{RESET}")
        sys.exit(1)

    agg = aggregate(reports)
    print_table(agg, len(reports))
    save_aggregated(agg, len(reports))

    # Final verdict
    all_positive = all(
        agg.get(m, {}).get("delta_mean", 0) > 0
        for m in ("CRS", "TES", "LCS")
    )
    if all_positive:
        print(f"  {GREEN}{BOLD}✓ All metrics show positive improvement. Paper-ready!{RESET}")
    else:
        neg = [m for m in ("CRS", "TES", "LCS") if agg.get(m, {}).get("delta_mean", 0) <= 0]
        print(f"  {YELLOW}{BOLD}! Negative improvement on: {', '.join(neg)} — needs tuning.{RESET}")

    print()
