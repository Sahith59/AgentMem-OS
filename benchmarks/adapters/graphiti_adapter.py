"""
Graphiti (Zep's OSS temporal-KG core) baseline adapter — thin
SubprocessAdapter subclass. All real work (import graphiti_core, run
episode ingestion/search against Neo4j) happens in
graphiti_subprocess_worker.py inside .venv-graphiti. See base.py for the
stdin/stdout JSON protocol, setup_venvs.sh to bootstrap the venv, and
docker-compose.baselines.yml to start Neo4j first.

Why Graphiti specifically, not paid Zep: it's the honest, actually
installable core of what Zep runs — integrating it rather than
hand-simulating "Zep-like" behavior is the credibility fix this repo
needed most, given the public Zep-vs-Mem0 LoCoMo dispute (see
LAUNCH_ROADMAP.md Phase 2 §2.4).
"""
from __future__ import annotations

from benchmarks.adapters.base import SubprocessAdapter


class GraphitiAdapter(SubprocessAdapter):
    name = "graphiti"
    worker_module = "benchmarks.adapters.graphiti_subprocess_worker"
    venv_path = "benchmarks/adapters/.venv-graphiti"
