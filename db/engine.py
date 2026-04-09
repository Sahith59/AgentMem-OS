"""
AgentMem OS — Database Engine & Session Factory
=================================================
Central SQLAlchemy setup. All modules import get_session() from here.

Why engine.py and not database.py?
  The original database.py is an older implementation. This file is the
  upgraded Phase 2 version. All new code imports from memnai.db.engine.
  The old database.py is preserved for backward compatibility with any
  legacy code that still references it.

Design decisions:
  - SQLite for local-first, zero-config operation (free, no server)
  - Path resolved: MEMNAI_DB_PATH env → config.yaml → ~/.memnai/memnai.db
  - check_same_thread=False — background threads (consolidation engine,
    KG ingestion) access DB from threads other than the main thread
  - StaticPool — single shared connection, correct for SQLite
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
from sqlalchemy.pool import StaticPool

from memnai.db.models import Base


# ─────────────────────────────────────────────────────────────────────────────
# DB Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_db_path() -> str:
    """
    Resolve the SQLite database file path.

    Priority:
      1. MEMNAI_DB_PATH environment variable  (testing / CI override)
      2. storage.base_path in config.yaml     (primary SSD)
      3. storage.fallback_path in config.yaml (SSD disconnected)
      4. ~/.memnai/memnai.db                  (universal fallback)
    """
    # 1. Env var override
    env_path = os.environ.get("MEMNAI_DB_PATH")
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
                    return str(db_dir / "memnai.db")

            fallback = config.get("storage", {}).get("fallback_path", "")
            if fallback:
                fb_path = Path(fallback).expanduser()
                db_dir = fb_path / "db"
                db_dir.mkdir(parents=True, exist_ok=True)
                return str(db_dir / "memnai.db")

        except Exception:
            pass

    # 4. Universal fallback
    default = Path.home() / ".memnai" / "memnai.db"
    default.parent.mkdir(parents=True, exist_ok=True)
    return str(default)


# ─────────────────────────────────────────────────────────────────────────────
# Engine Setup
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = _resolve_db_path()
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
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
