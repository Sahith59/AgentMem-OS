"""
Mem0 baseline adapter — thin SubprocessAdapter subclass. All real work
(import mem0, run extraction/retrieval) happens in
mem0_subprocess_worker.py inside .venv-mem0, isolated from the main venv's
pinned chromadb==0.5.20. See base.py for the stdin/stdout JSON protocol
and setup_venvs.sh to bootstrap the venv.
"""
from __future__ import annotations

from benchmarks.adapters.base import SubprocessAdapter


class Mem0Adapter(SubprocessAdapter):
    name = "mem0"
    worker_module = "benchmarks.adapters.mem0_subprocess_worker"
    venv_path = "benchmarks/adapters/.venv-mem0"
