"""Refine-not-replace (issue #34): a reinforcement may carry refined wording.

The DB primitive is ``Database.reinforce_memory(..., content=...)``: the same
id, observation chain, and validity window are kept while the stored content
converges to the newer wording. A replayed refine (all source events already
attached) converges content ONLY — no count bump, no observation row.
"""

import tempfile

from infinitum.database import Database
from infinitum.models import Event, Memory


async def _db(tmp: str) -> Database:
    db = Database(f"{tmp}/runtime.db")
    await db.connect()
    return db


async def _add_event(db: Database, session_id: str, content: str) -> Event:
    event = Event(session_id=session_id, event_type="message.user", role="user", content=content)
    await db.add_event(event)
    return event


# --- DB primitive: reinforce_memory(content=...) -----------------------------

_OLD = "we use postgres for the primary db"
_REFINED = "postgresql 17 is the canonical datastore"
_REFINED_AGAIN = "postgresql 18 is the canonical datastore"


async def test_refine_updates_content_without_replacing_the_memory_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            e1 = await _add_event(db, "s1", _OLD)
            e2 = await _add_event(db, "s2", _REFINED)
            memory = await db.create_memory(
                Memory(topic="database", content=_OLD, source_event_ids=[e1.id])
            )

            refined = await db.reinforce_memory(
                memory.id,
                content=_REFINED,
                confidence=0.9,
                importance=0.8,
                source_event_ids=[e2.id],
            )

            assert refined is not None
            assert refined.id == memory.id
            assert refined.content == _REFINED
            assert refined.observation_count == 2
            assert refined.status == "active"
            # FTS sees the new wording (terms unique to it), not the old one.
            assert memory.id in await db.fts_memory_ids("canonical datastore")
            assert memory.id not in await db.fts_memory_ids("postgres")
        finally:
            await db.close()


async def test_replayed_refine_converges_content_only_without_new_observations():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            e1 = await _add_event(db, "s1", _OLD)
            e2 = await _add_event(db, "s2", _REFINED)
            memory = await db.create_memory(
                Memory(topic="database", content=_OLD, source_event_ids=[e1.id])
            )
            await db.reinforce_memory(
                memory.id,
                content=_REFINED,
                confidence=0.9,
                importance=0.8,
                source_event_ids=[e2.id],
            )
            before = await db.fetchone("SELECT COUNT(*) AS n FROM memory_observations")

            # Replay: identical source ids (all already attached), newer wording.
            replayed = await db.reinforce_memory(
                memory.id,
                content=_REFINED_AGAIN,
                confidence=0.95,
                importance=0.9,
                source_event_ids=[e2.id],
            )

            assert replayed is not None
            assert replayed.content == _REFINED_AGAIN
            assert replayed.observation_count == 2
            after = await db.fetchone("SELECT COUNT(*) AS n FROM memory_observations")
            assert int(after["n"]) == int(before["n"])
            assert memory.id in await db.fts_memory_ids("canonical datastore")
        finally:
            await db.close()
