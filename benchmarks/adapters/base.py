"""
Shared adapter interface for benchmarks/real_baseline_eval.py.

Every memory system under comparison (AgentMem OS itself, Mem0, Graphiti,
Letta, LangMem, and a trivial recent-only floor) implements this same
four-method contract, so the orchestrator can ingest a dataset and
retrieve context identically regardless of which system it's talking to.

`namespace` is the isolation boundary for one LoCoMo conversation or one
LongMemEval haystack scope — different questions' haystacks must never
bleed into each other's retrieval within a single system.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class MemoryAdapter(ABC):
    """Common contract every system under test implements."""

    name: str = "unnamed"

    @abstractmethod
    def setup(self) -> None:
        """Verify the backend is reachable and create any client/connection
        needed. Called once before any ingest/retrieve calls."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Wipe (or newly create) an isolated namespace for one run. Must
        be safe to call before first use of a namespace."""
        raise NotImplementedError

    @abstractmethod
    def ingest_session(self, namespace: str, session_id: str, turns: list) -> None:
        """
        Ingest one session's dialogue turns into this namespace.

        turns: [{"role": str, "content": str}, ...] — matches
        corpus_loaders.MemEntry.turns exactly, so a BenchDataset's memories
        can be passed straight through.

        Should be idempotent-ish: calling this twice with the same
        (namespace, session_id) should not duplicate content in a way
        that materially changes retrieval quality.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, namespace: str, query: str, top_k: int = 10) -> list:
        """Return up to top_k ranked memory strings, most relevant first."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional cleanup hook — most adapters don't need this."""
        pass


class SubprocessAdapter(MemoryAdapter):
    """
    Base class for adapters whose SDK has dependencies too heavy or
    conflicting to install into the main agentmem_os venv (Mem0, Graphiti,
    LangMem all pull dependency trees that may not coexist with this
    project's pinned versions).

    Spawns `python -m benchmarks.adapters.<worker_module>` once, inside a
    dedicated venv, and keeps it alive for the whole run — speaking
    newline-delimited JSON over stdin/stdout. Respawning per call would
    make library import time dominate wall-clock time across a full
    dataset run.

    Subclasses set `worker_module` (dotted path, e.g.
    "benchmarks.adapters.mem0_subprocess_worker") and `venv_path` (relative
    to the repo root, e.g. "benchmarks/adapters/.venv-mem0").
    """

    worker_module: str = ""
    venv_path: str = ""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._repo_root = Path(__file__).resolve().parent.parent.parent

    def _venv_python(self) -> Path:
        venv_dir = self._repo_root / self.venv_path
        py = venv_dir / "bin" / "python3"
        if not py.exists():
            raise RuntimeError(
                f"{self.name} adapter's venv not found at {venv_dir}. "
                f"Run benchmarks/adapters/setup_venvs.sh first."
            )
        return py

    def setup(self) -> None:
        if self._proc is not None:
            return
        python = self._venv_python()
        self._proc = subprocess.Popen(
            [str(python), "-m", self.worker_module],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            cwd=str(self._repo_root), text=True, bufsize=1,
        )
        # Handshake — worker prints one ready line once its imports/clients
        # are warm, so setup() blocks until the worker can actually serve
        # requests instead of racing the first real call against cold-start.
        ready = self._call_raw({"op": "ping"})
        if not ready.get("ok"):
            raise RuntimeError(f"{self.name} worker failed to start: {ready}")

    def _call_raw(self, request: dict) -> dict:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError(f"{self.name} worker process not started — call setup() first")
        with self._lock:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} worker process died (empty response). "
                                f"Check stderr output above for the real error.")
        return json.loads(line)

    def reset(self, namespace: str) -> None:
        self._call_raw({"op": "reset", "namespace": namespace})

    def ingest_session(self, namespace: str, session_id: str, turns: list) -> None:
        resp = self._call_raw({"op": "ingest_session", "namespace": namespace,
                                "session_id": session_id, "turns": turns})
        if not resp.get("ok"):
            raise RuntimeError(f"{self.name} ingest_session failed: {resp.get('error')}")

    def retrieve(self, namespace: str, query: str, top_k: int = 10) -> list:
        resp = self._call_raw({"op": "retrieve", "namespace": namespace,
                                "query": query, "top_k": top_k})
        if not resp.get("ok"):
            raise RuntimeError(f"{self.name} retrieve failed: {resp.get('error')}")
        return resp.get("result", [])

    def teardown(self) -> None:
        if self._proc is not None:
            try:
                self._call_raw({"op": "shutdown"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None
