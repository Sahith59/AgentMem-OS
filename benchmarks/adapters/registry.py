"""
System-name-string -> adapter-class lookup for real_baseline_eval.py, so
new adapters can be added without touching the orchestrator.

Competitor adapters (mem0/graphiti/letta/langmem) register here as they're
built (Phase 2.3-2.6) — only agentmem_os and recent_only exist so far.
"""
from __future__ import annotations

from benchmarks.adapters.base import MemoryAdapter
from benchmarks.adapters.agentmem_adapter import AgentMemAdapter
from benchmarks.adapters.recent_only_adapter import RecentOnlyAdapter
from benchmarks.adapters.mem0_adapter import Mem0Adapter
from benchmarks.adapters.graphiti_adapter import GraphitiAdapter
from benchmarks.adapters.letta_adapter import LettaAdapter
from benchmarks.adapters.langmem_adapter import LangMemAdapter

ADAPTERS: dict[str, type[MemoryAdapter]] = {
    "agentmem_os": AgentMemAdapter,
    "recent_only": RecentOnlyAdapter,
    "mem0": Mem0Adapter,
    "graphiti": GraphitiAdapter,
    "letta": LettaAdapter,
    "langmem": LangMemAdapter,
}


def get_adapter(name: str) -> MemoryAdapter:
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown adapter '{name}'. Registered: {sorted(ADAPTERS)}"
        ) from None
    return cls()
