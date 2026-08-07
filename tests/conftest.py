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
