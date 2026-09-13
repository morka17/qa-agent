"""
Alembic environment script. Wires Alembic's migration machinery to
`storage.models.Base.metadata` (autogenerate diffs against the live ORM
models) and to `config.settings.Settings.database_url` (so migrations
run against whatever database the deployment is actually configured for,
without a separate `alembic.ini`-hardcoded URL to keep in sync).

Run migrations from the repo root with:
    alembic upgrade head
    alembic revision --autogenerate -m "add index to bugs.severity"

This file follows Alembic's standard generated template shape
(`run_migrations_offline` / `run_migrations_online`) with the two
project-specific changes: the target metadata import, and pulling the
connection URL from Settings instead of `alembic.ini`.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from qa_agent.config.settings import get_settings
from qa_agent.storage.models import Base

# Alembic Config object, providing access to values within alembic.ini.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate support: Alembic diffs the live DB schema against this
# metadata to produce migration scripts.
target_metadata = Base.metadata

# Override whatever's in alembic.ini with the actual runtime setting, so
# a single Settings-driven DATABASE_URL is the one source of truth across
# the whole app rather than two places that can drift out of sync.
config.set_main_option("sqlalchemy.url", get_settings().database_url)


def run_migrations_offline() -> None:
    """
    Generate SQL without a live DB connection (`alembic upgrade head --sql`).
    Useful for review-before-apply workflows in a change-managed production
    environment.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations against a live async database connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())