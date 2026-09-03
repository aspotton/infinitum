import tempfile

import pytest

from infinitum.database import Database
from infinitum.models import Event, Memory


@pytest.mark.asyncio
async def test_memory_provenance_reinforcement_and_supersession():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        e1 = await db.add_event(Event(session_id="s", event_type="message.user", role="user", content="Use PostgreSQL"))
        old = await db.create_memory(Memory(memory_type="decision", topic="database", content="We use PostgreSQL.", source_event_ids=[e1.id]))
        e2 = await db.add_event(
            Event(
                session_id="s",
                event_type="message.user",
                role="user",
                content="Still use PostgreSQL",
            )
        )
        await db.reinforce_memory(
            old.id, confidence=0.95, importance=0.8, source_event_ids=[e2.id]
        )
        reinforced = await db.get_memory(old.id)
        assert reinforced is not None
        assert reinforced.observation_count == 2
        assert e1.id in reinforced.source_event_ids
        assert e2.id in reinforced.source_event_ids

        # Retrying the same observation is idempotent and must not inflate the
        # evidence count.
        await db.reinforce_memory(
            old.id, confidence=0.95, importance=0.8, source_event_ids=[e2.id]
        )
        retried = await db.get_memory(old.id)
        assert retried is not None
        assert retried.observation_count == 2

        e3 = await db.add_event(
            Event(
                session_id="s",
                event_type="message.user",
                role="user",
                content="Use PostgreSQL 17",
            )
        )
        new = await db.create_memory(
            Memory(
                memory_type="decision",
                topic="database",
                content="PostgreSQL 17 is the current standard.",
                source_event_ids=[e3.id],
            )
        )
        await db.supersede_memory(old.id, new.id)
        old_after = await db.get_memory(old.id)
        assert old_after is not None
        assert old_after.status == "superseded"
        assert old_after.superseded_by == new.id
        await db.close()
