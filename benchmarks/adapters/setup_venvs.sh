#!/usr/bin/env bash
# Bootstraps one isolated venv per benchmarks/adapters/requirements-*.txt
# (mem0, graphiti, langmem — SDKs whose dependency trees conflict with the
# main venv's pinned chromadb==0.5.20 / langchain==0.3.9). Letta uses the
# main venv (thin httpx-based client, low collision risk) and needs no
# entry here — see LAUNCH_ROADMAP.md Phase 2 Task 19.
#
# Usage: bash benchmarks/adapters/setup_venvs.sh [python-binary]
# Default python binary: python3.13 (falls back to python3 if not found).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${1:-$(command -v python3.13 || command -v python3)}"

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

for req_file in "$SCRIPT_DIR"/requirements-*.txt; do
    [ -e "$req_file" ] || continue
    name="$(basename "$req_file" .txt | sed 's/^requirements-//')"
    venv_dir="$SCRIPT_DIR/.venv-$name"

    echo ""
    echo "── $name ──────────────────────────────────────────────"
    if [ -d "$venv_dir" ]; then
        echo "  $venv_dir already exists, skipping creation (delete it to rebuild)."
    else
        "$PYTHON_BIN" -m venv "$venv_dir"
        echo "  Created $venv_dir"
    fi

    "$venv_dir/bin/pip" install --quiet --upgrade pip
    "$venv_dir/bin/pip" install --quiet -r "$req_file"
    echo "  Installed $req_file into $venv_dir"
done

echo ""
echo "Done. Adapter worker processes are launched by SubprocessAdapter using"
echo "each .venv-<name>/bin/python3 — see benchmarks/adapters/base.py."
