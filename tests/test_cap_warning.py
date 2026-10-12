import logging
import tempfile

import pytest

from infinitum.database import Database
from infinitum.models import Memory


async def _seed_three(db: Database) -> None:
    for i in range(3):
        await db.create_memory(Memory(memory_type="fact", topic="cap", content=f"memory {i}"))


@pytest.mark.asyncio
async def test_list_active_memories_warns_when_scan_hits_cap(caplog):
    # Tiny-limit trick: 3 rows fetched with limit=3 fills the cap exactly,
    # so the scan must warn without seeding 5001 rows.
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed_three(db)
        with caplog.at_level(logging.WARNING, logger="infinitum.database"):
            result = await db.list_active_memories(limit=3)
        assert len(result) == 3
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("active memory scan hit limit=3" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_list_active_memories_silent_below_cap(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed_three(db)
        with caplog.at_level(logging.WARNING, logger="infinitum.database"):
            result = await db.list_active_memories(limit=4)
        assert len(result) == 3
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
