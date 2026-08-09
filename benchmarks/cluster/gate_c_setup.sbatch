#!/bin/bash -l
#SBATCH --account=trends517s113
#SBATCH --job-name=agentmem-setup
#SBATCH --partition=qDEV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# One-time setup: pull llama3.1 into the persistent model store and
# prove the GPU path works. Run this ONCE before the extraction array
# so no shard ever races another downloading 4.7GB.
set -euo pipefail

WORK=$HOME/agent-work
source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/4.3.1

export OLLAMA_MODELS=$WORK/ollama-models
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p "$OLLAMA_MODELS"

echo "[$(date)] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

singularity exec --nv \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  --env OLLAMA_HOST="$OLLAMA_HOST" \
  "$WORK/ollama.sif" ollama serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
  sleep 1
  [ "$i" -eq 60 ] && { echo "FATAL: ollama did not start"; exit 1; }
done
echo "[$(date)] ollama up"

singularity exec --nv \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  --env OLLAMA_HOST="$OLLAMA_HOST" \
  "$WORK/ollama.sif" ollama pull llama3.1

echo "[$(date)] model store:"
du -sh "$OLLAMA_MODELS"

# Timed generation probe on THIS GPU — the honest per-GPU rate input.
START=$(date +%s)
curl -s http://127.0.0.1:11434/api/generate -d \
  '{"model":"llama3.1","prompt":"Reply with exactly: READY","stream":false,"options":{"temperature":0,"num_predict":16}}' \
  | head -c 200
END=$(date +%s)
echo ""
echo "[$(date)] first-token+gen probe: $((END-START))s (includes model load)"
echo "SETUP DONE"
