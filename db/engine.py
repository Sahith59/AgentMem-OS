"""
AgentMem OS — Database Engine & Session Factory
=================================================
Central SQLAlchemy setup. All modules import get_session() from here.

Why engine.py and not database.py?
  The original database.py is an older implementation. This file is the
  upgraded Phase 2 version. All new code imports from agentmem_os.db.engine.
  The old database.py is preserved for backward compatibility with any
  legacy code that still references it.

Design decisions:
  - SQLite for local-first, zero-config operation (free, no server)
  - Path resolved: AGENTMEM_OS_DB_PATH env → config.yaml → ~/.agentmem_os/agentmem_os.db
  - check_same_thread=False — background threads (consolidation engine,
    KG ingestion) access DB from threads other than the main thread
  - QueuePool (default) — exclusive connection per checked-out Session;
    the earlier StaticPool single-shared-connection design destroyed
    concurrent writes (see the pooling comment at the engine definition)
  - WAL journal mode — allows concurrent background reads + main thread writes
  - foreign_keys=ON — enforces FK constraints (SQLite disables by default)
  - expire_on_commit=False — prevents DetachedInstanceError in background threads
  - create_all on import — Alembic handles schema upgrades after first run
"""

import os
import yaml
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
# (poolclass deliberately not imported/overridden — see the pooling
# comment above the engine; default QueuePool is a measured decision)

from agentmem_os.db.models import Base


# ─────────────────────────────────────────────────────────────────────────────
# DB Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_db_path() -> str:
    """
    Resolve the SQLite database file path.

    Priority:
      1. AGENTMEM_OS_DB_PATH environment variable  (testing / CI override)
      2. storage.base_path in config.yaml     (primary SSD)
      3. storage.fallback_path in config.yaml (SSD disconnected)
      4. ~/.agentmem_os/agentmem_os.db                  (universal fallback)
    """
    # 1. Env var override
    env_path = os.environ.get("AGENTMEM_OS_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # 2–3. config.yaml
    config_file = Path(__file__).parent.parent / "config.yaml"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f) or {}

            base = config.get("storage", {}).get("base_path", "")
            if base:
                base_path = Path(base).expanduser()
                if base_path.exists():
                    db_dir = base_path / "db"
                    db_dir.mkdir(parents=True, exist_ok=True)
                    return str(db_dir / "agentmem_os.db")

            fallback = config.get("storage", {}).get("fallback_path", "")
            if fallback:
                fb_path = Path(fallback).expanduser()
                db_dir = fb_path / "db"
                db_dir.mkdir(parents=True, exist_ok=True)
                return str(db_dir / "agentmem_os.db")

        except Exception:
            pass

    # 4. Universal fallback
    default = Path.home() / ".agentmem_os" / "agentmem_os.db"
    default.parent.mkdir(parents=True, exist_ok=True)
    return str(default)


# ─────────────────────────────────────────────────────────────────────────────
# Engine Setup
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = _resolve_db_path()
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Pooling (changed 2026-08-06, Consolidation v2 Stage 1 G3 rounds 2-3):
# StaticPool shared ONE DBAPI connection across every Session in the
# process. Measured consequence: a concurrent reader closing its session
# issued a ROLLBACK on the shared connection and destroyed a writer's
# uncommitted work — 202 of 300 semantic-fact writes lost, and even two
# plain Turn writers lost 45/300 with commit() reporting success. The
# default QueuePool checks each Session out an EXCLUSIVE connection
# (same isolation as one-connection-per-session), SQLite WAL provides
# reader/writer isolation, and busy_timeout serializes concurrent writers
# at the database. QueuePool over NullPool is a measured decision (R3
# N11): identical zero-loss result on the concurrency attack, ~4.6x the
# throughput of NullPool's per-session connection churn (9.2k vs 1.9k
# ops/s), which matters in per-turn ingestion loops. In-memory DBs are
# NOT supported by this module (the raw-sqlite3 migrations below connect
# by path and would see a different database).
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    # Explicit ceiling (R4-4): 5 pooled + 10 overflow = 15 concurrent live
    # sessions; a 16th blocks up to 30s then TimeoutErrors. Store sessions
    # are short-lived; long-held caller-batch sessions (Stage 2) must stay
    # well under this or raise it deliberately.
    pool_size=5,
    max_overflow=10,
    echo=False,   # flip to True to log every SQL statement during debugging
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """
    Configure SQLite pragmas on every new connection.

    WAL mode:        background thread can read while main thread writes
    foreign_keys:    enforce FK constraints (OFF by default in SQLite)
    synchronous:     NORMAL = good balance of safety vs speed
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Concurrent writers meet at the DB lock — wait instead of failing
    # fast with "database is locked".
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session Factory
# ─────────────────────────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,   # safe for background thread usage
)


def get_session() -> Session:
    """
    Return a new SQLAlchemy Session. Caller must close it.

    Usage:
        db = get_session()
        try:
            turns = db.query(Turn).filter(...).all()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    """
    return SessionLocal()


# ─────────────────────────────────────────────────────────────────────────────
# Table Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables defined in models.py if they don't already exist.
    Safe to call multiple times — SQLAlchemy's create_all is idempotent.
    After the first run, use Alembic to apply schema changes instead.
    """
    Base.metadata.create_all(bind=engine)
    _migrate_phase1()
    _migrate_temporal_kg()
    _migrate_semantic_tier()


def _migrate_phase1() -> None:
    """
    Safe additive migration for Phase 1 (Conflict Detection) columns.
    ALTER TABLE ADD COLUMN is idempotent in SQLite — silently skipped if
    the column already exists (OperationalError caught and ignored).
    """
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(DB_PATH)
        for col, definition in [
            ("is_active",       "INTEGER NOT NULL DEFAULT 1"),
            ("contradicted_by", "INTEGER REFERENCES turns(id)"),
        ]:
            try:
                conn.execute(f"ALTER TABLE turns ADD COLUMN {col} {definition}")
                conn.commit()
            except _sqlite3.OperationalError:
                pass  # column already exists
        conn.close()
    except Exception:
        pass


def _migrate_temporal_kg() -> None:
    """
    Safe additive migration for the Temporal Knowledge Graph columns
    (LAUNCH_ROADMAP.md Phase 6 Priority 2, ported from X-MemoryArch's
    graph_builder.py). Same idempotent ALTER-TABLE idiom as _migrate_phase1.
    """
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(DB_PATH)
        for col, definition in [
            ("last_confirmed_at", "DATETIME"),
        ]:
            try:
                conn.execute(f"ALTER TABLE kg_nodes ADD COLUMN {col} {definition}")
                conn.commit()
            except _sqlite3.OperationalError:
                pass  # column already exists
        for col, definition in [
            ("relation_type", "VARCHAR DEFAULT 'CO_OCCURS'"),
            ("confidence",    "FLOAT DEFAULT 0.5"),
            ("valid_from",    "DATETIME"),
            ("valid_until",   "DATETIME"),
            ("superseded_by", "INTEGER REFERENCES kg_edges(id)"),
        ]:
            try:
                conn.execute(f"ALTER TABLE kg_edges ADD COLUMN {col} {definition}")
                conn.commit()
            except _sqlite3.OperationalError:
                pass  # column already exists
        conn.close()
    except Exception:
        pass


def _migrate_semantic_tier() -> dict:
    """
    Consolidation v2 companion migration. Three jobs, all LOUD (G3 findings
    12-13 — the silent except-pass idiom hid a migration that had never run):

      1. Retrofit the kg_edges active-edge partial index (columns existed
         since the Temporal KG port; no index ever served the lookup).
      2. VERIFY semantic_facts carries its dedup constraint + partial
         indexes. SQLite cannot ALTER TABLE ADD CONSTRAINT, so a
         semantic_facts table that exists WITHOUT uq_facts_scope_hash can
         never be repaired in place — that is a RuntimeError, not a shrug:
         the operator must migrate the rows to a rebuilt table.
      3. Report what happened.
    """
    import sqlite3 as _sqlite3
    from loguru import logger as _logger

    report = {"kg_edges_index": "unknown", "semantic_facts": "unknown"}
    conn = None
    try:
        conn = _sqlite3.connect(DB_PATH)
        import sqlite3 as _sq
        added = []
        clog = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='consolidation_log'").fetchone() is not None
        if not clog:
            report["consolidation_log_columns"] = "table absent (create_all creates it)"
        else:
            for col in ("truncated_chars INTEGER DEFAULT 0",
                        "rejected_count INTEGER DEFAULT 0",
                        "rejections_json TEXT"):
                try:
                    conn.execute(f"ALTER TABLE consolidation_log ADD COLUMN {col}")
                    added.append(col.split()[0])
                except _sq.OperationalError as col_err:
                    # R4-M4: duplicate-column is the ONLY acceptable case
                    if "duplicate column" not in str(col_err):
                        raise
            conn.commit()
            report["consolidation_log_columns"] = (
                f"added {added}" if added else "verified")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_edges_active "
            "ON kg_edges(source_id, relation_type) "
            "WHERE valid_until IS NULL"
        )
        conn.commit()
        report["kg_edges_index"] = "present"

        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_facts'"
        ).fetchone() is not None
        if not table_exists:
            report["semantic_facts"] = "absent (create_all will create it)"
        else:
            # Verify the dedup constraint BY ITS COLUMNS — SQLite names
            # constraint indexes 'sqlite_autoindex_*', never by constraint
            # name, and "some autoindex exists" proves nothing (G3 round 2,
            # B2: a UNIQUE(fact_text) table sailed through the old check
            # with dedup completely unenforced).
            has_dedup_unique = False
            # r = (seq, name, unique, origin, partial): require unique AND
            # NOT partial — a partial unique index enforces nothing for
            # rows outside its predicate (R3 N5: a UNIQUE ... WHERE
            # superseded_by IS NULL index accepted duplicate live rows'
            # history and still read as "verified").
            for name in [r[1] for r in conn.execute(
                    "PRAGMA index_list('semantic_facts')")
                    if r[2] == 1 and r[4] == 0]:
                cols = [c[2] for c in conn.execute(f"PRAGMA index_info('{name}')")]
                if sorted(cols) == ["normalized_hash", "scope_key"]:
                    has_dedup_unique = True
                    break
            if not has_dedup_unique:
                raise RuntimeError(
                    "semantic_facts exists WITHOUT a UNIQUE index on "
                    "(scope_key, normalized_hash). SQLite cannot add "
                    "constraints in place — rebuild the table: create "
                    "semantic_facts_new via create_all, copy rows, drop old, "
                    "rename. Refusing to run with dedup unenforced."
                )
            index_names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='semantic_facts'")}
            required = {"idx_facts_current", "idx_facts_current_all",
                        "idx_facts_mentioned"}
            missing = required - index_names
            if missing:
                _logger.warning(
                    f"[migrate] semantic_facts missing indexes {missing} — "
                    "recreating on the DB being migrated")
                # Repair the DB this function is inspecting — NOT the
                # module-level engine, which may point elsewhere (G3 round
                # 2, B3: the old code mutated the production DB while
                # leaving the target unrepaired).
                from sqlalchemy import create_engine as _ce
                from agentmem_os.db.models import SemanticFact
                repair_engine = _ce(f"sqlite:///{DB_PATH}")
                try:
                    for idx in SemanticFact.__table__.indexes:
                        if idx.name in missing:
                            idx.create(bind=repair_engine)
                finally:
                    repair_engine.dispose()
            report["semantic_facts"] = "verified"
        _logger.info(f"[migrate] semantic tier: {report}")
        return report
    except RuntimeError:
        raise
    except Exception as e:
        # Loud and fatal: a migration that cannot run means the schema
        # state is UNKNOWN — running anyway risks silent duplicate facts
        # (the old warn-and-continue here was except:pass wearing a hat).
        raise RuntimeError(
            f"semantic tier migration failed against {DB_PATH}: {e}"
        ) from e
    finally:
        if conn is not None:
            conn.close()


def get_db_info() -> dict:
    """
    Return metadata about the current DB for the CLI status command.
    """
    db = get_session()
    try:
        result = db.execute(text("SELECT COUNT(*) FROM sessions")).scalar()
        turns  = db.execute(text("SELECT COUNT(*) FROM turns")).scalar()
        return {
            "path": DB_PATH,
            "url": DATABASE_URL,
            "sessions": result,
            "turns": turns,
        }
    except Exception:
        return {"path": DB_PATH, "url": DATABASE_URL}
    finally:
        db.close()


# Auto-initialise on import
init_db()
