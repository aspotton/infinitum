"""Temporal metadata on memories (todo 5 of phase1-evaluation-quality-loop).

Covers the acceptance list for the additive temporal columns
(`valid_from`, `valid_until`, `observed_at`) and the extraction-eligible
temporal fields:

1. A fresh DB carries the three new columns.
2. A hand-created old-schema DB migrates in place; existing rows get NULL.
3. A candidate whose ``valid_from`` is not ISO still CREATES the memory with
   the field NULL (the validator coerces, it never drops the candidate).
4. Superseding closes the old row's ``valid_until`` (only when NULL).
5. A candidate round-trips valid dates through the create path.

Plus the two derived behaviours the plan names: ``observed_at`` = earliest
source event timestamp, and re-initialising a migrated DB is a no-op.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory, MemoryCandidate


TEMPORAL_COLUMNS = {"valid_from", "valid_until", "observed_at"}


# Pre-change `memories` CREATE, copied verbatim from database.py's schema block
# (before the temporal columns were added). Hand-copied on purpose: this is the
# migration test precedent from tests/test_request_context.py, and the drift
# risk is accepted deliberately -- if the real CREATE changes this is the one
# place that still models the old table.
LEGACY_MEMORIES_CREATE = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT 'general',
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.7,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    superseded_by TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(superseded_by) REFERENCES memories(id)
);
"""


def _memory_columns(db: Database) -> set[str]:
    return {row["name"] for row in db._conn.execute("PRAGMA table_info(memories)")}  # type: ignore[union-attr]


# 1 -- fresh DB carries the three temporal columns --------------------------------
@pytest.mark.asyncio
async def test_fresh_database_has_temporal_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/fresh.db")
        await db.connect()
        try:
            assert TEMPORAL_COLUMNS <= _memory_columns(db)
        finally:
            await db.close()


# 2 -- old-schema DB migrates in place, existing rows NULL ------------------------
@pytest.mark.asyncio
async def test_old_database_migrates_in_place_with_null_new_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(LEGACY_MEMORIES_CREATE)
        conn.execute(
            "INSERT INTO memories(id, memory_type, topic, content, created_at, updated_at) "
            "VALUES ('mem_old', 'fact', 'general', 'a legacy fact', "
            "'2025-01-01T00:00:00+00:00', '2025-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        db = Database(path)
        await db.connect()
        try:
            assert TEMPORAL_COLUMNS <= _memory_columns(db)
            old = await db.get_memory("mem_old")
            assert old is not None
            assert old.valid_from is None
            assert old.valid_until is None
            assert old.observed_at is None
        finally:
            await db.close()


# stale-state adversarial class: migrating a migrated DB again is a no-op ---------
@pytest.mark.asyncio
async def test_reinitializing_migrated_database_is_a_noop():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(LEGACY_MEMORIES_CREATE)
        conn.commit()
        conn.close()

        db = Database(path)
        await db.connect()
        try:
            await db.initialize()  # second migrate pass must not error / duplicate
            assert TEMPORAL_COLUMNS <= _memory_columns(db)
        finally:
            await db.close()


# 3 -- malformed input: invalid valid_from keeps the candidate, field NULL ---------
def test_candidate_with_non_iso_valid_from_keeps_candidate_with_null_field():
    candidate = MemoryCandidate.model_validate(
        {
            "memory_type": "fact",
            "topic": "general",
            "content": "We use PostgreSQL for persistence.",
            "valid_from": "last tuesday",
            "valid_until": "when the mood strikes",
        }
    )
    # The validator coerces rather than raising, so model_validate did NOT throw.
    assert candidate.content == "We use PostgreSQL for persistence."
    assert candidate.valid_from is None
    assert candidate.valid_until is None


@pytest.mark.asyncio
async def test_invalid_valid_from_still_creates_memory_with_null_field():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        db = Database(cfg.memory.database_path)
        await db.connect()
        try:
            learner = MemoryLearner(db, MagicMock(), MagicMock(), MagicMock(), cfg)
            learner.retriever.search = AsyncMock(return_value=[])  # type: ignore[attr-defined]
            learner.embeddings.embed = AsyncMock(return_value=None)  # type: ignore[attr-defined]
            candidate = MemoryCandidate.model_validate(
                {
                    "memory_type": "fact",
                    "topic": "general",
                    "content": "We use PostgreSQL for persistence.",
                    "valid_from": "last tuesday",
                }
            )
            affected = await learner._apply(candidate, [], set())
            created = await db.get_memory(next(iter(affected)))
            assert created is not None
            assert created.valid_from is None
        finally:
            await db.close()


# 4 -- supersession closes the old row's valid_until ------------------------------
@pytest.mark.asyncio
async def test_supersede_closes_old_valid_until():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/sup.db")
        await db.connect()
        try:
            old = await db.create_memory(Memory(content="Old current state."))
            new = await db.create_memory(Memory(content="New current state."))
            assert old.valid_until is None
            await db.supersede_memory(old.id, new.id)
            closed = await db.get_memory(old.id)
            assert closed is not None
            assert closed.status == "superseded"
            assert closed.valid_until is not None
        finally:
            await db.close()


# an explicit past valid_until must survive supersession (COALESCE, not overwrite)
@pytest.mark.asyncio
async def test_supersede_preserves_existing_valid_until():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/sup2.db")
        await db.connect()
        try:
            old = await db.create_memory(
                Memory(content="Fact valid until a date.", valid_until="2020-06-30")
            )
            new = await db.create_memory(Memory(content="Replacement."))
            await db.supersede_memory(old.id, new.id)
            closed = await db.get_memory(old.id)
            assert closed is not None
            assert closed.valid_until == "2020-06-30"
        finally:
            await db.close()


# 5 -- candidate round-trip preserves valid dates on create -----------------------
@pytest.mark.asyncio
async def test_candidate_valid_dates_persist_through_create():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        db = Database(cfg.memory.database_path)
        await db.connect()
        try:
            learner = MemoryLearner(db, MagicMock(), MagicMock(), MagicMock(), cfg)
            learner.retriever.search = AsyncMock(return_value=[])  # type: ignore[attr-defined]
            learner.embeddings.embed = AsyncMock(return_value=None)  # type: ignore[attr-defined]
            candidate = MemoryCandidate(
                memory_type="decision",
                topic="database",
                content="We adopted PostgreSQL 17.",
                valid_from="2024-01-01",
                valid_until="2026-12-31",
            )
            affected = await learner._apply(candidate, [], set())
            created = await db.get_memory(next(iter(affected)))
            assert created is not None
            assert created.valid_from == "2024-01-01"
            assert created.valid_until == "2026-12-31"
        finally:
            await db.close()


# observed_at = earliest source event created_at; NULL when no source ids ---------
@pytest.mark.asyncio
async def test_observed_at_is_earliest_source_event_or_null():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/obs.db")
        await db.connect()
        try:
            now = datetime.now(timezone.utc)
            older = await db.add_event(
                Event(
                    session_id="s",
                    event_type="message.user",
                    content="early",
                    created_at=now - timedelta(days=10),
                )
            )
            newer = await db.add_event(
                Event(
                    session_id="s",
                    event_type="message.user",
                    content="late",
                    created_at=now - timedelta(days=1),
                )
            )
            mem = await db.create_memory(
                Memory(content="derived", source_event_ids=[newer.id, older.id])
            )
            fetched = await db.get_memory(mem.id)
            assert fetched is not None
            assert fetched.observed_at == older.created_at.isoformat()

            bare = await db.create_memory(Memory(content="no sources"))
            fetched_bare = await db.get_memory(bare.id)
            assert fetched_bare is not None
            assert fetched_bare.observed_at is None
        finally:
            await db.close()
