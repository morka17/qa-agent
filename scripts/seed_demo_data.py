#!/usr/bin/env python3
"""
Seeds the local database with a handful of realistic runs, steps, bugs,
and artifacts, so `make dev` / a fresh clone has something to look at in
the operator dashboard (`web/`) and API (`api/`) without needing to run
a real agent loop first.

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --reset   # drop and recreate tables first
    python scripts/seed_demo_data.py --count 20  # seed more/fewer runs

Uses `settings.database_url` — point it at a local/dev database, never
production. Refuses to run if `settings.is_production` is true.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone

from qa_agent.config.logging_config import configure_logging, get_logger
from qa_agent.config.settings import get_settings
from qa_agent.storage.models import (
    ArtifactORM,
    Base,
    BugORM,
    RunORM,
    StepORM,
    get_engine,
    get_sessionmaker,
)

logger = get_logger(__name__)

_DEMO_TARGET_URLS = [
    "https://staging.shop.example.com/checkout",
    "https://staging.shop.example.com/cart",
    "https://staging.dashboard.example.com/settings",
]

_DEMO_BUG_TITLES = [
    ("Cart total does not update after adding an item", "app_bug", "high"),
    ("Checkout fails with 500 error on large orders", "app_bug", "critical"),
    ("Add to Wishlist button occasionally not found", "selector_drift", "info"),
    ("Report generation button does nothing", "app_bug", "medium"),
]

_STEP_ACTIONS = ["navigate", "click", "fill", "wait_for"]


def _random_past_timestamp(max_days_ago: int = 14) -> datetime:
    delta = timedelta(days=random.uniform(0, max_days_ago), hours=random.uniform(0, 24))
    return datetime.now(timezone.utc) - delta


async def seed(count: int, reset: bool) -> None:
    settings = get_settings()
    if settings.is_production:
        print("Refusing to seed demo data against a production environment.", file=sys.stderr)
        sys.exit(1)

    engine = get_engine(settings)
    sessionmaker = get_sessionmaker(engine)

    if reset:
        logger.info("Dropping and recreating all tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    else:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async with sessionmaker() as session:
        for i in range(count):
            created_at = _random_past_timestamp()
            state = random.choices(["passed", "failed", "error"], weights=[0.55, 0.4, 0.05])[0]
            run = RunORM(
                story_id=f"demo-story-{i}",
                target_url=random.choice(_DEMO_TARGET_URLS),
                state=state,
                created_at=created_at,
                finished_at=created_at + timedelta(seconds=random.uniform(20, 180)),
                error="Playwright navigation timeout" if state == "error" else None,
            )
            session.add(run)
            await session.flush()  # assign run.id before dependent rows

            for step_order in range(random.randint(2, 5)):
                success = state == "passed" or step_order < 2  # later steps fail on failed runs
                session.add(
                    StepORM(
                        run_id=run.id,
                        step_order=step_order,
                        action=random.choice(_STEP_ACTIONS),
                        target_description=f"the demo element #{step_order}",
                        description=f"Demo step {step_order}",
                        success=success,
                        error=None if success else "Element resolution timed out",
                        duration_ms=random.uniform(80, 1200),
                        resolution_strategy=random.choice(["heuristic", "cache_hit", "visual_grounding"]),
                        resolution_confidence=random.uniform(0.6, 1.0),
                    )
                )

            if state == "failed":
                title, category, severity = random.choice(_DEMO_BUG_TITLES)
                session.add(
                    BugORM(
                        run_id=run.id,
                        title=title,
                        severity=severity,
                        category=category,
                        summary=f"{title}.",
                        description=f"Demo-seeded bug report for run {run.id}.",
                        expected_behavior="The feature should work as described in the story.",
                        actual_behavior="The feature did not behave as expected.",
                        labels=["sentinel-qa", f"category:{category}", f"severity:{severity}"],
                        occurrence_count=random.randint(1, 4),
                        created_at=created_at,
                        filed_at=created_at + timedelta(minutes=1),
                    )
                )
                session.add(
                    ArtifactORM(
                        run_id=run.id,
                        type="screenshot",
                        description="Failure screenshot (demo data - not a real file)",
                        path_or_url=f"/demo-artifacts/{run.id}/failure.png",
                        is_uploaded=False,
                        created_at=created_at,
                    )
                )

        await session.commit()

    await engine.dispose()
    logger.info("Seeded %d demo run(s).", count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Number of demo runs to seed.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables before seeding.")
    args = parser.parse_args()

    configure_logging()
    asyncio.run(seed(count=args.count, reset=args.reset))


if __name__ == "__main__":
    main()
