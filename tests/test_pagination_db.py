"""Database-level keyset (row-value) paging + total-order tiebreaker tests.

These pin the ``before=(sort_value, tiebreak)`` contract on the three listers
(list_memories / list_events / list_topics) that the cursor routes build on:
pages must be disjoint, complete, and stably ordered even across identical
timestamps.
"""
from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta

import pytest

from infinitum.database import Database
from infinitum.models import Event, Memory, TopicSummary

# Shared fixed instant for the identical-timestamp tests: all rows carry the
# exact same stored string so only the id/topic tiebreaker can order them.
_SAME = datetime(2026, 1, 1, tzinfo=UTC)


def _stamp(i: int) -> str:
    return (_SAME + timedelta(minutes=i)).isoformat()


@pytest.mark.asyncio
async def test_stored_timestamp_strings_are_isoformat_stable():
    """A created memory stores updated_at as a round-trip-stable isoformat string.

    The cursor codec compares stored timestamp strings directly, so this pins the
    string-equality assumption against any future format drift.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        mem = await db.create_memory(Memory(memory_type="fact", content="pin"))
        row = await db.fetchone("SELECT updated_at FROM memories WHERE id=?", (mem.id,))
        stored = row["updated_at"]
        assert stored == datetime.fromisoformat(stored).isoformat()
        await db.close()


@pytest.mark.asyncio
async def test_keyset_pages_are_disjoint_and_complete():
    """Walking memories with before=... yields three disjoint, complete pages.

    The union must equal the seeded set with zero dupes, and the concatenation
    must equal a single full ``ORDER BY updated_at DESC, id DESC`` read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        seeded = set()
        for i in range(5):
            mem = await db.create_memory(Memory(memory_type="fact", content=f"m{i}"))
            seeded.add(mem.id)
            await db.execute(
                "UPDATE memories SET updated_at=? WHERE id=?", (_stamp(i), mem.id)
            )

        page1 = await db.list_memories(limit=2)
        page2 = await db.list_memories(
            limit=2, before=(page1[-1].updated_at.isoformat(), page1[-1].id)
        )
        page3 = await db.list_memories(
            limit=2, before=(page2[-1].updated_at.isoformat(), page2[-1].id)
        )

        walked = [m.id for m in (*page1, *page2, *page3)]
        assert len(set(walked)) == 5
        assert set(walked) == seeded
        full = await db.list_memories(limit=10)
        assert walked == [m.id for m in full]
        await db.close()


@pytest.mark.asyncio
async def test_tiebreaker_gives_total_order():
    """With one shared updated_at, pages are repeatable and walk in id DESC order.

    A missing id tiebreaker would let SQLite return ties in arbitrary order, so
    two identical calls could disagree and a walk could overlap or skip a row.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        seeded = set()
        for i in range(3):
            mem = await db.create_memory(Memory(memory_type="fact", content=f"t{i}"))
            seeded.add(mem.id)
            await db.execute(
                "UPDATE memories SET updated_at=? WHERE id=?",
                (_SAME.isoformat(), mem.id),
            )

        a = await db.list_memories(limit=2)
        b = await db.list_memories(limit=2)
        assert [m.id for m in a] == [m.id for m in b]

        walked = [m.id for m in a]
        page = a
        while page:
            page = await db.list_memories(
                limit=2, before=(page[-1].updated_at.isoformat(), page[-1].id)
            )
            walked.extend(m.id for m in page)

        assert set(walked) == seeded
        assert len(set(walked)) == 3
        assert walked == sorted(seeded, reverse=True)
        await db.close()


@pytest.mark.asyncio
async def test_keyset_with_status_filter():
    """A status filter and a keyset cursor compose: only active rows, no dupes.

    Two archived rows sharing the same timestamp space must never leak into the
    active walk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        active = set()
        for i in range(4):
            mem = await db.create_memory(Memory(memory_type="fact", content=f"a{i}"))
            active.add(mem.id)
            await db.execute(
                "UPDATE memories SET updated_at=? WHERE id=?", (_stamp(i), mem.id)
            )
        archived = set()
        for i in range(2):
            mem = await db.create_memory(
                Memory(memory_type="fact", content=f"x{i}", status="archived")
            )
            archived.add(mem.id)
            await db.execute(
                "UPDATE memories SET updated_at=? WHERE id=?",
                (_stamp(4 + i), mem.id),
            )

        walked = []
        before = None
        for _ in range(10):
            page = await db.list_memories(limit=2, status="active", before=before)
            if not page:
                break
            walked.extend(m.id for m in page)
            before = (page[-1].updated_at.isoformat(), page[-1].id)

        assert set(walked) == active
        assert len(set(walked)) == 4
        assert not (set(walked) & archived)
        await db.close()


@pytest.mark.asyncio
async def test_events_and_topics_keyset():
    """Events keyset on (created_at, id) and topics on (updated_at, topic), disjoint.

    All rows in each group share one timestamp, so only the tiebreaker column can
    produce a stable disjoint walk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()

        evt_ids = set()
        for _i in range(3):
            evt = await db.add_event(
                Event(
                    session_id="s",
                    event_type="request.received",
                    content="x",
                    created_at=_SAME,
                )
            )
            evt_ids.add(evt.id)

        walked = []
        before = None
        for _ in range(10):
            page = await db.list_events(limit=2, before=before)
            if not page:
                break
            walked.extend(e.id for e in page)
            before = (page[-1].created_at.isoformat(), page[-1].id)

        assert set(walked) == evt_ids
        assert len(set(walked)) == 3

        topics = set()
        for i in range(3):
            topic = f"topic_{i}"
            await db.upsert_topic(
                TopicSummary(
                    topic=topic, summary="s", memory_count=1, updated_at=_SAME
                )
            )
            topics.add(topic)

        walked_topics = []
        before = None
        for _ in range(10):
            page = await db.list_topics(limit=2, before=before)
            if not page:
                break
            walked_topics.extend(t.topic for t in page)
            before = (page[-1].updated_at.isoformat(), page[-1].topic)

        assert set(walked_topics) == topics
        assert len(set(walked_topics)) == 3
        await db.close()
