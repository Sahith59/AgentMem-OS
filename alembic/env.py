"""
Alembic Environment — AgentMem OS
===================================
Wires Alembic to the same engine and models as the rest of the application.
Uses autogenerate so schema changes in models.py automatically produce migrations.

Usage:
  # Create a new migration after editing models.py:
  alembic revision --autogenerate -m "add importance_score to turns"

  # Apply all pending migrations:
  alembic upgrade head

  # Rollback one migration:
  alembic downgrade -1
"""

import sys
import os
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make sure the project root is on sys.path so models can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the app's engine and models
from memnai.db.engine import engine, DB_PATH
from memnai.db.models import Base

# Alembic config object
config = context.config

# Setup logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at the actual DB used by the app
config.set_main_option("sqlalchemy.url", f"sqlite:///{DB_PATH}")

# MetaData for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection.
    Useful for generating SQL scripts to review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,   # required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations with a live DB connection.
    Standard mode — used by `alembic upgrade head`.
    """
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,   # required for SQLite ALTER TABLE support
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
