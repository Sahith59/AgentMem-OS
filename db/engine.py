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
    _migrate_stage3()
    _migrate_stage4()
    _migrate_profile_tier()


def _migrate_profile_tier() -> dict:
    """
    Profile tier migration. The table itself is created by create_all
    (same absent-table honesty as every stage before it); this verifies
    presence and reports it rather than assuming. Additive only — the
    profile is DERIVED state, so a rebuild is always available and no
    destructive path is ever needed.
    """
    import sqlite3 as _sqlite3
    from loguru import logger as _logger

    conn = None
    try:
        conn = _sqlite3.connect(DB_PATH, timeout=30)
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='profile_attributes'").fetchone() is not None
        report = {"profile_attributes":
                  "present" if present else "absent (create_all creates it)"}
        _logger.info(f"[migrate] profile tier: {report}")
        return report
    except Exception as e:
        raise RuntimeError(
            f"profile tier migration failed against {DB_PATH}: {e}") from e
    finally:
        if conn is not None:
            conn.close()


def _migrate_stage4() -> dict:
    """
    Stage 4 additive migration (supersession judgment): t_invalid and
    judge_failure columns; supersession_judgments is created by
    create_all (absent-table honesty like every stage before it). Same
    discipline: duplicate-column-only swallow, loud otherwise.
    """
    import sqlite3 as _sqlite3
    from loguru import logger as _logger

    report = {}
    conn = None
    try:
        conn = _sqlite3.connect(DB_PATH, timeout=30)
        for table, col in (("semantic_facts", "t_invalid TEXT"),
                           ("consolidation_log", "judge_failure TEXT")):
            if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name=?", (table,)).fetchone() is None:
                report[table] = "table absent (create_all creates it)"
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
                conn.commit()
                report[table] = f"{col.split()[0]} added"
            except _sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):
                    raise
                report[table] = "verified"
        report["supersession_judgments"] = (
            "present" if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='supersession_judgments'").fetchone()
            else "absent (create_all creates it)")
        _logger.info(f"[migrate] stage 4: {report}")
        return report
    except Exception as e:
        raise RuntimeError(
            f"stage 4 migration failed against {DB_PATH}: {e}") from e
    finally:
        if conn is not None:
            conn.close()


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


def _merge_duplicate_kg_nodes(conn) -> dict:
    """
    Collapse kg_nodes rows that share (coalesce(agent_id,''), entity_text)
    into the lowest-id row so uq_kg_nodes_scope_text can be created.
    Split out of _migrate_stage3 so tests can drive it against a
    synthetic-duplicate DB directly (our dev DB has zero dup groups —
    verified 2026-08-06 over 10,911 nodes — this path exists for
    arbitrary user DBs and must be exercised by tests, not by luck).

    Per group: edges (kg_edges AND semantic_fact_entities links — the
    raw connection runs with FKs off, so forgetting the link table
    orphans fact links silently, G3 R2 minor) re-pointed to the keeper;
    self-loops the re-pointing produced are dropped; keeper absorbs
    mention_count sums and the latest last_seen/last_confirmed_at.
    CO_OCCURS pair dedup is then delegated to the GLOBAL
    _merge_duplicate_co_occurs_edges pass (G3 R2 M3: re-pointing does
    not preserve src<tgt ordering — keeper=min(id) makes INVERSION the
    common case, and a per-group ordered GROUP BY missed exactly the
    pairs this merge creates; the global pass canonicalizes ordering
    first, so it sees them all — for BOTH undirected families, ALIAS_OF
    included: R3 M3 measured an inverted ALIAS_OF pair defeating the
    linker's ordered exists-check and duplicating the edge). DIRECTIONAL
    typed edges (WORKS_AT/LIVES_AT/STUDIES_AT) are left as-is —
    direction is their meaning, and supersession chains reference their
    ids with valid_from/valid_until disambiguating.
    """
    groups = conn.execute(
        "SELECT coalesce(agent_id,''), entity_text, min(id), count(*), "
        "sum(coalesce(mention_count, 1)) FROM kg_nodes "
        "GROUP BY 1, 2 HAVING count(*) > 1"
    ).fetchall()
    merged_nodes = 0
    has_link_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='semantic_fact_entities'").fetchone() is not None
    for scope, text_, keeper, n, mention_sum in groups:
        losers = [r[0] for r in conn.execute(
            "SELECT id FROM kg_nodes WHERE coalesce(agent_id,'') = ? "
            "AND entity_text = ? AND id != ?", (scope, text_, keeper))]
        ph = ",".join("?" * len(losers))
        conn.execute(f"UPDATE kg_edges SET source_id = ? WHERE source_id IN ({ph})",
                     [keeper] + losers)
        conn.execute(f"UPDATE kg_edges SET target_id = ? WHERE target_id IN ({ph})",
                     [keeper] + losers)
        conn.execute("DELETE FROM kg_edges WHERE source_id = ? AND target_id = ?",
                     (keeper, keeper))
        if has_link_table:
            # A fact linked to BOTH a loser and the keeper would
            # collide with uq_fact_entity on re-point — drop the loser
            # row (same link, same fact, same entity identity).
            conn.execute(
                f"DELETE FROM semantic_fact_entities WHERE node_id IN ({ph}) "
                f"AND fact_id IN (SELECT fact_id FROM semantic_fact_entities "
                f"WHERE node_id = ?)", losers + [keeper])
            conn.execute(
                f"UPDATE semantic_fact_entities SET node_id = ? "
                f"WHERE node_id IN ({ph})", [keeper] + losers)
        conn.execute(
            "UPDATE kg_nodes SET mention_count = ?, "
            "last_seen = (SELECT max(last_seen) FROM kg_nodes "
            "  WHERE coalesce(agent_id,'') = ? AND entity_text = ?), "
            "last_confirmed_at = (SELECT max(last_confirmed_at) FROM kg_nodes "
            "  WHERE coalesce(agent_id,'') = ? AND entity_text = ?) "
            "WHERE id = ?",
            (mention_sum, scope, text_, scope, text_, keeper))
        conn.execute(f"DELETE FROM kg_nodes WHERE id IN ({ph})", losers)
        merged_nodes += len(losers)
    conn.commit()
    merged_edges = _merge_duplicate_co_occurs_edges(conn) if groups else 0
    return {"groups": len(groups), "nodes_removed": merged_nodes,
            "co_occurs_edges_merged": merged_edges}


def _merge_duplicate_co_occurs_edges(conn) -> int:
    """
    Repair for a REAL measured corruption (Stage-3 G3 R1): the turn
    path's CO_OCCURS lookup filtered relation_type IS NULL while the ORM
    default wrote the string 'CO_OCCURS' — so instead of incrementing
    weight it created a NEW edge per co-occurrence (1,979 duplicate
    pairs in the dev DB), and the in-memory loader's add_edge() then
    kept only the LAST row's weight. The lookup is fixed; this merges
    the damage: duplicate (source,target) CO_OCCURS pairs (both storage
    shapes) collapse into the lowest-id row with weights SUMMED and the
    newest last_updated kept; duplicate ALIAS_OF pairs keep the highest
    confidence. Idempotent (second run finds nothing).

    Every row this function deletes is FIRST copied into
    kg_edges_dedup_backup (G3 R3 M2: this runs as an import-time
    migration against whatever DB resolves — a destructive repair with
    no pre-state is unrecoverable if a future bug miscounts; the backup
    is the pre-state, kept until an operator drops it deliberately).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kg_edges_dedup_backup "
        "AS SELECT * FROM kg_edges WHERE 0")
    # Schema-drift guard (R4-M4, reproduced): CREATE..AS SELECT froze
    # the backup's columns at creation time; after a later ALTER TABLE
    # kg_edges ADD COLUMN, the INSERT..SELECT * below would raise and —
    # because init_db runs at import — make the whole package
    # unimportable on exactly the DBs that have duplicates. On drift:
    # preserve the old backup under a versioned name, recreate fresh.
    live_cols = [c[1] for c in conn.execute("PRAGMA table_info(kg_edges)")]
    bak_cols = [c[1] for c in conn.execute(
        "PRAGMA table_info(kg_edges_dedup_backup)")]
    if bak_cols != live_cols:
        n = 1
        while conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (f"kg_edges_dedup_backup_old{n}",)).fetchone():
            n += 1
        conn.execute(f"ALTER TABLE kg_edges_dedup_backup "
                     f"RENAME TO kg_edges_dedup_backup_old{n}")
        conn.execute("CREATE TABLE kg_edges_dedup_backup "
                     "AS SELECT * FROM kg_edges WHERE 0")
    _CO = "(relation_type IS NULL OR relation_type = 'CO_OCCURS')"
    # Canonicalize ordering FIRST (G3 R2 M3 + R3 M3): CO_OCCURS and
    # ALIAS_OF are UNDIRECTED — ingest writes src<tgt, but node-merge
    # re-pointing (and any historical writer) can leave inverted rows;
    # (a,b) and (b,a) are the SAME pair, an ordered GROUP BY treats
    # them as two, and the linker's ordered ALIAS_OF exists-check
    # duplicates against an inverted row (measured). SQLite UPDATE
    # reads the pre-update row values, so this swap is safe.
    conn.execute(
        f"UPDATE kg_edges SET source_id = target_id, target_id = source_id "
        f"WHERE source_id > target_id "
        f"AND ({_CO} OR relation_type = 'ALIAS_OF')")
    # Exact-duplicate ALIAS_OF pairs collapse to the lowest id, keeping
    # the highest confidence (an alias link's strength is its best
    # measured similarity; there is no additive weight semantic here).
    for src, tgt in conn.execute(
            "SELECT source_id, target_id FROM kg_edges "
            "WHERE relation_type = 'ALIAS_OF' "
            "GROUP BY source_id, target_id HAVING count(*) > 1").fetchall():
        keep_edge, best = conn.execute(
            "SELECT min(id), max(confidence) FROM kg_edges WHERE "
            "source_id = ? AND target_id = ? AND relation_type = 'ALIAS_OF'",
            (src, tgt)).fetchone()
        conn.execute(
            "INSERT INTO kg_edges_dedup_backup SELECT * FROM kg_edges "
            "WHERE source_id = ? AND target_id = ? "
            "AND relation_type = 'ALIAS_OF' AND id != ?",
            (src, tgt, keep_edge))
        conn.execute(
            "DELETE FROM kg_edges WHERE source_id = ? AND target_id = ? "
            "AND relation_type = 'ALIAS_OF' AND id != ?",
            (src, tgt, keep_edge))
        conn.execute("UPDATE kg_edges SET confidence = ? WHERE id = ?",
                     (best, keep_edge))
    merged = 0
    for src, tgt in conn.execute(
            f"SELECT source_id, target_id FROM kg_edges WHERE {_CO} "
            f"GROUP BY source_id, target_id HAVING count(*) > 1").fetchall():
        keep_edge, total, latest = conn.execute(
            f"SELECT min(id), sum(weight), max(last_updated) FROM kg_edges "
            f"WHERE source_id = ? AND target_id = ? AND {_CO}",
            (src, tgt)).fetchone()
        conn.execute(
            f"INSERT INTO kg_edges_dedup_backup SELECT * FROM kg_edges "
            f"WHERE source_id = ? AND target_id = ? AND {_CO} AND id != ?",
            (src, tgt, keep_edge))
        conn.execute(
            f"DELETE FROM kg_edges WHERE source_id = ? AND target_id = ? "
            f"AND {_CO} AND id != ?", (src, tgt, keep_edge))
        # weight summed AND the newest last_updated kept — "no
        # information lost" must include the recency column (G3 R2 minor).
        conn.execute("UPDATE kg_edges SET weight = ?, last_updated = ? "
                     "WHERE id = ?", (total, latest, keep_edge))
        merged += 1
    conn.commit()
    return merged


def _migrate_stage3() -> dict:
    """
    Consolidation v2 Stage 3 migration (fact→entity linking +
    event_status). Same discipline as _migrate_semantic_tier: verify by
    INSPECTION (PRAGMA index columns, never index names alone), repair
    the DB at DB_PATH, absent-table honesty, loud on anything
    unexpected.

      1. consolidation_log.entities_linked column.
      2. semantic_facts.event_status column + deterministic backfill for
         pre-Stage-3 events (t_occurred > t_mentioned → 'planned', else
         'occurred' — both columns already stored, no guessing).
      3. Verify semantic_fact_entities carries UNIQUE(fact_id, node_id) —
         without it concurrent linkers dup-link and the constraint
         fallback in the linker dead-codes. Unrepairable in place →
         RuntimeError, same as the facts dedup constraint.
      4. kg_nodes uq_kg_nodes_scope_text unique index; existing
         duplicate rows go through _merge_duplicate_kg_nodes first.
    """
    import re as _re
    import sqlite3 as _sqlite3
    from loguru import logger as _logger

    report = {}
    conn = None
    try:
        # timeout matches the app engine's busy_timeout (G3 R2 minor:
        # the default 5s raw connection raced a 30s app writer during
        # the multi-second repair pass).
        conn = _sqlite3.connect(DB_PATH, timeout=30)

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='consolidation_log'").fetchone() is None:
            report["consolidation_log"] = "table absent (create_all creates it)"
        else:
            added_cols = []
            for col in ("entities_linked INTEGER DEFAULT 0",
                        "link_failure TEXT"):
                try:
                    conn.execute(f"ALTER TABLE consolidation_log "
                                 f"ADD COLUMN {col}")
                    added_cols.append(col.split()[0])
                except _sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e):
                        raise
            conn.commit()
            report["consolidation_log"] = (
                f"added {added_cols}" if added_cols else "verified")

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='semantic_facts'").fetchone() is None:
            report["event_status"] = "table absent (create_all creates it)"
        else:
            try:
                conn.execute("ALTER TABLE semantic_facts "
                             "ADD COLUMN event_status TEXT")
                conn.commit()
                added = True
            except _sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):
                    raise
                added = False
            backfilled = conn.execute(
                "UPDATE semantic_facts SET event_status = CASE "
                "WHEN t_occurred IS NOT NULL AND t_occurred > t_mentioned "
                "THEN 'planned' ELSE 'occurred' END "
                "WHERE fact_type = 'event' AND event_status IS NULL"
            ).rowcount
            conn.commit()
            report["event_status"] = (
                f"{'added' if added else 'verified'}, "
                f"backfilled {backfilled} event rows")

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='semantic_fact_entities'").fetchone() is None:
            report["semantic_fact_entities"] = "absent (create_all creates it)"
        else:
            has_link_unique = False
            for name in [r[1] for r in conn.execute(
                    "PRAGMA index_list('semantic_fact_entities')")
                    if r[2] == 1 and r[4] == 0]:
                cols = [c[2] for c in conn.execute(f"PRAGMA index_info('{name}')")]
                if sorted(cols) == ["fact_id", "node_id"]:
                    has_link_unique = True
                    break
            if not has_link_unique:
                raise RuntimeError(
                    "semantic_fact_entities exists WITHOUT a UNIQUE index "
                    "on (fact_id, node_id). SQLite cannot add constraints "
                    "in place — rebuild the table (create new via "
                    "create_all, copy rows, drop, rename). Refusing to run "
                    "with link dedup unenforced.")
            report["semantic_fact_entities"] = "verified"

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='kg_edges'").fetchone() is None:
            report["co_occurs_dedup"] = "table absent (create_all creates it)"
        else:
            merged_pairs = _merge_duplicate_co_occurs_edges(conn)
            report["co_occurs_dedup"] = (
                f"merged {merged_pairs} duplicate pairs" if merged_pairs
                else "clean")

        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='kg_nodes'").fetchone() is None:
            report["kg_nodes_unique"] = "table absent (create_all creates it)"
        else:
            # Verify by the index's OWN SQL, never its name (G3 R1 M1 —
            # the name-only check accepted a scope-blind
            # UNIQUE(entity_text) impostor that permanently rejected a
            # second agent's nodes while reporting "verified"; same
            # lesson as the semantic_facts autoindex check above).
            # Expression indexes don't expose columns through PRAGMA
            # index_info, so the sqlite_master.sql text is the artifact.
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name='uq_kg_nodes_scope_text'").fetchone()
            sql_norm = ""
            if row and row[0]:
                sql_norm = _re.sub(r"\s+", "", row[0].lower().replace('"', "'"))
            # EXACT normalized-DDL equality, not substring presence (G3
            # R2 M2: a substring test "verified" an index with an extra
            # trailing id column — unique over nothing — and one with a
            # WHERE agent_id IS NOT NULL clause that left NULL-scope
            # rows unguarded; any extra text must fail equality).
            good = sql_norm == (
                "createuniqueindexuq_kg_nodes_scope_text"
                "onkg_nodes(coalesce(agent_id,''),entity_text)")
            if row is not None and good:
                report["kg_nodes_unique"] = "verified"
            else:
                if row is not None:
                    conn.execute("DROP INDEX uq_kg_nodes_scope_text")
                    conn.commit()
                ddl = ("CREATE UNIQUE INDEX uq_kg_nodes_scope_text "
                       "ON kg_nodes(coalesce(agent_id, ''), entity_text)")
                rebuilt = "rebuilt (wrong shape)" if row is not None else "created"
                try:
                    conn.execute(ddl)
                    conn.commit()
                    report["kg_nodes_unique"] = rebuilt
                except _sqlite3.IntegrityError:
                    merge = _merge_duplicate_kg_nodes(conn)
                    conn.execute(ddl)   # loud if it STILL fails
                    conn.commit()
                    report["kg_nodes_unique"] = f"{rebuilt} after merge {merge}"

        _logger.info(f"[migrate] stage 3: {report}")
        return report
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"stage 3 migration failed against {DB_PATH}: {e}"
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
