"""
Test-suite guard: importing agentmem_os.db.engine runs init_db() at import
time against whatever DB path resolves — without this, MERELY RUNNING TESTS
mutates the founder's real dev DB (G3 round 2, M4: an empty semantic_facts
table and idx_kg_edges_active appeared in the production DB as a side
effect of a pytest run). Point the engine at a per-session scratch file
BEFORE anything imports it.
"""
import os
import tempfile

# FORCED, not setdefault (R3 N8): the founder's own .env/setup.sh export
# this variable pointing at a real DB — a test run must never write there
# regardless of the inherited environment.
os.environ["AGENTMEM_OS_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="agentmem-test-"), "test.db")

# Same guard, second channel (Stage 5 finding): Redis hot-cache keys are
# session-id-only — no DB identity — so a test run against the live
# localhost Redis reads GHOST turns from earlier runs and leaves its own
# behind (188 stale keys observed). The DB pin above cannot cover this.
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

# Third channel (Stage 6 final pass, C2): StorageManager reads
# config.yaml RELATIVE TO CWD, and pytest runs from the repo root —
# every recall's get_or_create_collection was writing collections into
# the DEV vector store (/Volumes/Sahith_SSD/AgentMem-OS/vectors;
# e2e-*/hdr-test2/honest-test session ids found there, so the suite
# had been spilling since long before Stage 6 named the channel). A
# scratch config.yaml + chdir points the whole storage tree at
# scratch; the fixtures below run AFTER collection so pytest's own
# relative test ids are unaffected.
import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _isolate_storage_tree():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    scratch = Path(tempfile.mkdtemp(prefix="agentmem-test-storage-"))
    cfg = (repo_root / "config.yaml").read_text()
    rewritten = cfg.replace(
        'base_path: "/Volumes/Sahith_SSD/AgentMem-OS/"',
        f'base_path: "{scratch}/"')
    assert str(scratch) in rewritten, "config base_path rewrite failed"
    (scratch / "config.yaml").write_text(rewritten)
    old_cwd = os.getcwd()
    os.chdir(scratch)
    # Self-check (the m7 lesson): prove the tree actually resolved to
    # scratch before any test writes through it.
    from agentmem_os.storage.manager import StorageManager
    assert StorageManager().base_path.startswith(str(scratch))
    yield
    os.chdir(old_cwd)


@pytest.fixture(autouse=True)
def _reset_chroma_backend_override():
    """The benchmark install_* helpers set a class-level chroma
    backend override (C1). Tests that want it set it themselves
    (adapter/eval-harness tests call install during the test); nothing
    may LEAK it into the next test."""
    yield
    try:
        from agentmem_os.llm.context_assembler import ContextAssembler
        ContextAssembler._chroma_override = None
    except Exception:
        pass
