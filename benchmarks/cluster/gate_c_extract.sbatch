#!/bin/bash -l
#SBATCH --account=trends517s113
#SBATCH --job-name=agentmem-gatec
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=11:30:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
#
# Gate C extraction shard — one GPU, one private Ollama, one disjoint
# slice of the 3,631-session worklist, one private SQLite DB.
#
# Submit:  sbatch --array=0-N benchmarks/cluster/gate_c_extract.sbatch
# The array index IS the shard index; GATE_C_NSHARDS must equal the
# array size (passed via --export or edited below).
set -euo pipefail

WORK=$HOME/agent-work
REPO=$WORK/agentmem_os
NSHARDS=${GATE_C_NSHARDS:-4}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
# Distinct port per task: array tasks CAN land on the same node.
PORT=$((11434 + SHARD))

source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/4.3.1

export OLLAMA_MODELS=$WORK/ollama-models        # persists across jobs
export OLLAMA_HOST=127.0.0.1:$PORT
export AGENTMEM_OS_OLLAMA_URL=http://127.0.0.1:$PORT
export AGENTMEM_OS_DISABLE_REDIS=1
# One server, several concurrent generations per GPU.
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}
export OLLAMA_KEEP_ALIVE=24h
export GATE_C_SHARD=$SHARD
export GATE_C_NSHARDS=$NSHARDS
# Prefix is parameterised so an EXTENSION run writes new shard DBs and
# never appends to the shards of a completed run — the old artifact must
# stay byte-reproducible. Default preserves the original run's paths.
PREFIX=${GATE_C_DB_PREFIX:-gate_c_shard}
export GATE_C_DB=$WORK/shards/${PREFIX}${SHARD}.db
export GATE_C_FAILLOG=$WORK/shards/${PREFIX}failures_${SHARD}.jsonl
# Passed through to the runner: which questions to cover, and which
# sessions are already done in a reference corpus (skip, don't redo).
export GATE_C_QSOURCE=${GATE_C_QSOURCE:-qa_accuracy_longmemeval_answerer54mini.json}
export GATE_C_EXCLUDE_DB=${GATE_C_EXCLUDE_DB:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no phone-home (Stage 6)
mkdir -p "$WORK/shards" "$OLLAMA_MODELS"

echo "[$(date)] shard=$SHARD/$NSHARDS node=$(hostname) port=$PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── start a private Ollama server on this node's GPU ──────────────────
singularity exec --nv \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  --env OLLAMA_HOST="$OLLAMA_HOST" \
  "$WORK/ollama.sif" ollama serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Wait for readiness — bounded and LOUD, never a blind sleep.
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; then
    echo "[$(date)] ollama up after ${i}s"; break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then echo "FATAL: ollama did not start"; exit 1; fi
done

# Model must already be present (pulled once by the setup job) — a
# shard must never race others downloading 4.7GB.
if ! curl -sf "http://127.0.0.1:$PORT/api/tags" | grep -q "llama3.1"; then
  echo "FATAL: llama3.1 missing from $OLLAMA_MODELS — run the setup job first"
  exit 1
fi

# ── run the shard ─────────────────────────────────────────────────────
cd "$WORK"
./venv/bin/python "$REPO/benchmarks/consolidation_v2_extract_live.py" \
  "${GATE_C_WORKERS:-2}" ${GATE_C_LIMIT:+$GATE_C_LIMIT}

echo "[$(date)] shard=$SHARD DONE"
