#!/bin/bash
# ============================================================
# AgentMem OS — One-click local setup
# Supports: Python 3.11 / 3.12 / 3.13 · macOS (Intel + Apple Silicon) · Linux
# Cost: $0.00 — all local, no cloud required
# ============================================================

# NOTE: No 'set -e' — we handle errors explicitly so the script
# never silently dies on an optional dependency.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; }
step() { echo -e "\n→ $1"; }

echo ""
echo "=========================================="
echo "  AgentMem OS — Environment Setup"
echo "=========================================="

# ─── Python version check ────────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: $PYTHON_VERSION"

if ! python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    fail "Python 3.11+ required. Current: $PYTHON_VERSION"
    echo "  Install via: brew install python@3.12"
    exit 1
fi
ok "Python version OK"

# ─── Virtual environment ─────────────────────────────────────────────────────
step "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Created venv/"
else
    ok "venv/ already exists — reusing"
fi

source venv/bin/activate
ok "Activated venv"

# ─── Upgrade pip + build tools first (critical for Python 3.13) ──────────────
step "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
ok "pip / setuptools / wheel up to date"

# ─── Core packages (binary-only where possible to avoid source compilation) ──
step "Installing core packages (this may take 2-4 minutes)..."

install_pkg() {
    local pkg="$1"
    local flags="${2:---quiet}"
    if pip install $flags "$pkg" --quiet 2>/dev/null; then
        ok "$pkg"
    else
        warn "$pkg failed — will retry or skip"
        return 1
    fi
}

# Install numpy FIRST with binary-only flag — avoids source compilation hell
step "Installing numpy (binary wheel only)..."
pip install "numpy>=2.0.0,<2.2.0" --only-binary :all: --quiet && ok "numpy" || {
    warn "numpy binary wheel not found for your Python version — trying latest"
    pip install numpy --only-binary :all: --quiet && ok "numpy (latest)" || fail "numpy install failed"
}

# Install scipy binary
step "Installing scipy (binary wheel only)..."
pip install scipy --only-binary :all: --quiet && ok "scipy" || warn "scipy install failed — cosine similarity will use fallback"

# Install scikit-learn binary
pip install scikit-learn --only-binary :all: --quiet && ok "scikit-learn" || warn "scikit-learn not installed"

# Core framework — all pure Python or have universal wheels
step "Installing framework packages..."
pip install \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    typer \
    rich \
    sqlalchemy \
    alembic \
    loguru \
    pyyaml \
    python-dotenv \
    httpx \
    tenacity \
    tiktoken \
    networkx \
    --quiet && ok "Framework packages installed" || warn "Some framework packages failed"

# Redis client
pip install redis --quiet && ok "redis" || warn "redis not installed — working memory will skip Redis tier"

# LLM routing
pip install litellm --quiet && ok "litellm" || warn "litellm not installed"

# Embeddings / LangChain
pip install langchain langchain-core langchain-community langchain-ollama --quiet \
    && ok "langchain + ollama" || warn "langchain not installed"

# Vector DB
pip install chromadb --quiet && ok "chromadb" || warn "chromadb not installed — semantic memory will be limited"

# ─── spaCy: install binary wheel only (no source compilation) ────────────────
step "Installing spaCy NLP (binary wheel only — no compilation)..."
pip install spacy --only-binary :all: --quiet && ok "spaCy" || {
    warn "spaCy binary wheel not available for this Python/platform combination."
    warn "Entity Knowledge Graph will use regex fallback (still fully functional)."
    warn "To install manually later: pip install spacy --only-binary :all:"
}

# ─── spaCy English model ─────────────────────────────────────────────────────
step "Downloading spaCy English model (en_core_web_sm)..."
if python -c "import spacy" 2>/dev/null; then
    python -m spacy download en_core_web_sm --quiet 2>/dev/null \
        && ok "en_core_web_sm downloaded" \
        || warn "spaCy model download failed — will retry on first run"
else
    warn "Skipping spaCy model — spaCy not installed"
fi

# ─── Dev + test tools ────────────────────────────────────────────────────────
step "Installing dev tools..."
pip install pytest pytest-asyncio pytest-cov black ruff --quiet \
    && ok "pytest + dev tools" || warn "Some dev tools not installed"

# ─── Install the agentmem_os package itself ───────────────────────────────────────
step "Installing agentmem_os package (editable mode)..."
pip install -e . --no-build-isolation --quiet \
    && ok "agentmem_os package installed (editable)" \
    || warn "Editable install skipped — package importable from this directory"

# ─── Ollama check ────────────────────────────────────────────────────────────
step "Checking Ollama (local LLM)..."
if command -v ollama &>/dev/null; then
    ok "Ollama found"
    echo "     Tip: ollama pull llama3.2:3b    ← lightweight, fast"
    echo "     Tip: ollama pull nomic-embed-text ← local embeddings"
else
    warn "Ollama not found"
    echo "     Install from: https://ollama.ai  (free, runs locally)"
fi

# ─── Redis check ─────────────────────────────────────────────────────────────
step "Checking Redis (working memory tier)..."
if command -v redis-cli &>/dev/null; then
    ok "Redis found — start with: redis-server"
else
    warn "Redis not found"
    echo "     Install: brew install redis   (Mac)"
    echo "     Install: sudo apt install redis-server   (Linux)"
    echo "     Note: System runs fine without Redis (graceful degradation)"
fi

# ─── Create .env if missing ──────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    step "Creating .env template..."
    cat > .env << 'EOF'
# AgentMem OS — Environment Variables
# All entries are OPTIONAL — system works fully with just Ollama (free, local)

# Local Config
AGENTMEM_OS_DB_PATH=~/.agentmem_os/agentmem_os.db
AGENTMEM_OS_VECTOR_PATH=~/.agentmem_os/vectors
AGENTMEM_OS_LOG_LEVEL=INFO

# Ollama (default — free, runs locally)
OLLAMA_BASE_URL=http://localhost:11434
AGENTMEM_OS_DEFAULT_LLM=ollama/llama3.2:3b
AGENTMEM_OS_EMBEDDING_MODEL=ollama/nomic-embed-text

# Optional: Groq (free tier, much faster than Ollama)
# GROQ_API_KEY=gsk_...

# Optional: Anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# Optional: OpenAI
# OPENAI_API_KEY=sk-...
EOF
    ok ".env template created"
fi

# ─── Quick sanity check ───────────────────────────────────────────────────────
step "Running sanity check..."
python3 - << 'PYEOF'
import sys
results = []

def chk(name, mod):
    try:
        __import__(mod)
        results.append(("✅", name))
    except ImportError:
        results.append(("⚠️ ", name + " (not installed — fallback active)"))

chk("sqlalchemy",   "sqlalchemy")
chk("fastapi",      "fastapi")
chk("loguru",       "loguru")
chk("networkx",     "networkx")
chk("numpy",        "numpy")
chk("scikit-learn", "sklearn")
chk("chromadb",     "chromadb")
chk("redis",        "redis")
chk("spaCy",        "spacy")
chk("tiktoken",     "tiktoken")
chk("litellm",      "litellm")

print("\n  Dependency check:")
for icon, name in results:
    print(f"    {icon} {name}")
PYEOF

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Activate:  source venv/bin/activate"
echo "  Run tests: pytest tests/test_phase3_algorithms.py"
echo "  Start API: uvicorn agentmem_os.api.app:app --reload"
echo ""
echo "  ⚠️  warnings above = optional deps (system still fully works)"
echo "  ❌  errors above   = check output and re-run that pip install"
echo "=========================================="
