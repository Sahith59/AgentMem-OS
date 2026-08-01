#!/bin/bash
# DEPRECATED — orchestrates three behavioral-simulation scripts, not real
# integrations. Superseded by benchmarks/ablation_study_real.py and
# benchmarks/real_baseline_eval.py. Kept only for historical reference —
# do not cite these numbers. See LAUNCH_ROADMAP.md Phase 2.
#
# AgentMem OS — Run All 3 Evaluation Phases
# ==========================================
# Phase 1: Multi-run ablation (fresh N runs → mean ± std)
# Phase 2: Long-horizon test (50 turns, probes at T35–T50)
# Phase 3: Baseline comparison (AgentMem OS vs Full-History vs Naive-RAG vs Recent-Only)
#
# Usage: bash benchmarks/deprecated_proxy_sim/run_all_phases.sh
# Est. total cost: ~$7.50  |  Est. time: 25–35 min

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     AgentMem OS — Full Evaluation Suite                 ║"
echo "║     Phase 1: Multi-run Ablation  (~$3.72, ~12 min)      ║"
echo "║     Phase 2: Long-Horizon 50T    (~$1.24, ~8 min)       ║"
echo "║     Phase 3: Baseline Compare   (~$1.24, ~8 min)        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
    echo "  ✓  Loaded .env"
else
    echo "  ✗  No .env found at $PROJECT_DIR/.env"
    exit 1
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "  ✗  ANTHROPIC_API_KEY not set in .env"
    exit 1
fi
echo "  ✓  API key loaded (${ANTHROPIC_API_KEY:0:12}...)"
echo ""

cd "$PROJECT_DIR"

# ── Phase 1 ──────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: Multi-Run Ablation (runs 2–4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 benchmarks/deprecated_proxy_sim/phase1_multi_run.py
echo ""

# ── Phase 2 ──────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: Long-Horizon Test (50 turns)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 benchmarks/deprecated_proxy_sim/phase2_long_horizon.py
echo ""

# ── Phase 3 ──────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3: Baseline Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 benchmarks/deprecated_proxy_sim/phase3_baselines.py
echo ""

echo "All 3 phases complete. Results saved to benchmarks/deprecated_proxy_sim/:"
echo "  ablation_multi_run.json"
echo "  ablation_summary_stats.json"
echo "  long_horizon_results.json"
echo "  baseline_comparison.json"
