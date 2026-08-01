"""
LangMem baseline adapter — thin SubprocessAdapter subclass. All real work
(import langmem/langgraph, run create_memory_store_manager() extraction
and store search) happens in langmem_subprocess_worker.py inside
.venv-langmem, isolated from the main venv's pinned langchain==0.3.9. See
base.py for the stdin/stdout JSON protocol and setup_venvs.sh to
bootstrap the venv.
"""
from __future__ import annotations

from benchmarks.adapters.base import SubprocessAdapter


class LangMemAdapter(SubprocessAdapter):
    name = "langmem"
    worker_module = "benchmarks.adapters.langmem_subprocess_worker"
    venv_path = "benchmarks/adapters/.venv-langmem"
