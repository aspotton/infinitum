"""Refine-not-replace (issue #34): a reinforcement may carry refined wording.

The DB primitive is ``Database.reinforce_memory(..., content=...)``: the same
id, observation chain, and validity window are kept while the stored content
converges to the newer wording. A replayed refine (all source events already
attached) converges content ONLY — no count bump, no observation row.
"""

import json
import tempfile
from unittest.mock import AsyncMock

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient


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


# --- Learner route: unmarked supersede proposals refine, not replace ---------

_SEED = "The PostgreSQL database uses a nightly backup strategy"
_REFINED_SCENARIO = (
    "The PostgreSQL database uses a nightly backup strategy: a nightly pg_dump "
    "job writing compressed archives to object storage"
)


async def _learner(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    upstream = UpstreamClient(cfg)
    retriever = MemoryRetriever(db, embeddings, cfg)
    learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
    return cfg, db, embeddings, upstream, learner


def _extraction_envelope(candidates: list[dict]) -> dict:
    """Extraction reply: candidate JSON as assistant content (no tool_calls)."""

    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"memories": candidates}),
                    "tool_calls": None,
                },
            }
        ]
    }


def _supersede_candidate(target_id: str, content: str = _REFINED_SCENARIO, **extra) -> dict:
    candidate = {
        "memory_type": "fact",
        "topic": "database",
        "content": content,
        "importance": 0.8,
        "confidence": 0.9,
        "operation_hint": "supersede",
        "supersedes_memory_ids": [target_id],
        "explicit_correction": False,
    }
    candidate.update(extra)
    return candidate


async def _seed_fact(db: Database, content: str = _SEED) -> Memory:
    event = Event(session_id="s1", event_type="message.user", role="user", content=content)
    await db.add_event(event)
    return await db.create_memory(
        Memory(
            memory_type="fact", topic="database", content=content, source_event_ids=[event.id]
        )
    )


async def _learn_reply(db: Database, learner: MemoryLearner, envelope: dict) -> None:
    """Drive one learn() turn with a fresh source event and the mock envelope."""

    new_event = Event(
        session_id="s2", event_type="message.user", role="user", content="We refined the details."
    )
    await db.add_event(new_event)
    learner.upstream.learning_chat_completion = AsyncMock(return_value=envelope)
    await learner.learn(
        {
            "request_id": "req_1",
            "session_id": "ses_2",
            "model": "memory-model",
            "user_text": "Tell me about our database backup strategy.",
            "assistant_text": "We back up PostgreSQL nightly.",
            "source_event_ids": [new_event.id],
            "request_context": {},
        }
    )


async def test_unmarked_supersede_refines_named_target_instead_of_replacing():
    """Issue #34: an unmarked single-target supersede of a lexically-related,
    same type/topic memory must refine it in place, never supersede + create."""

    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            seed = await _seed_fact(db)
            await _learn_reply(db, learner, _extraction_envelope([_supersede_candidate(seed.id)]))

            active = await db.fetchall(
                "SELECT id, content, observation_count FROM memories WHERE status='active'"
            )
            assert len(active) == 1
            assert active[0]["id"] == seed.id
            assert "pg_dump" in active[0]["content"]
            assert int(active[0]["observation_count"]) == 2
            superseded = await db.fetchall("SELECT id FROM memories WHERE status='superseded'")
            assert superseded == []
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_explicit_correction_supersede_still_replaces_the_target():
    """Current behavior preserved: an explicit correction supersedes + creates."""

    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            seed = await _seed_fact(db)
            candidate = _supersede_candidate(seed.id, explicit_correction=True)
            await _learn_reply(db, learner, _extraction_envelope([candidate]))

            old = await db.get_memory(seed.id)
            assert old is not None and old.status == "superseded"
            active = await db.fetchall("SELECT id FROM memories WHERE status='active'")
            assert len(active) == 1
            assert active[0]["id"] != seed.id
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_low_similarity_supersede_declines_refine_and_creates_side_by_side():
    """A lexically unrelated named target declines the refine route; the
    unchanged create + supersede path then skips the supersede (below floor)."""

    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            seed = await _seed_fact(db)
            candidate = _supersede_candidate(
                seed.id, content="The team reviews every pull request on Fridays"
            )
            await _learn_reply(db, learner, _extraction_envelope([candidate]))

            active = await db.fetchall("SELECT id FROM memories WHERE status='active'")
            assert len(active) == 2
            superseded = await db.fetchall("SELECT id FROM memories WHERE status='superseded'")
            assert superseded == []
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_multi_id_supersede_uses_the_current_create_and_supersede_path():
    """Two named targets decline the single-target refine gate; today's path
    creates the new memory and supersedes both related actives."""

    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            first = await _seed_fact(db)
            second = await _seed_fact(db, content="The PostgreSQL database runs on Amazon RDS")
            candidate = _supersede_candidate(
                first.id,
                content="The PostgreSQL database uses a nightly backup strategy on Amazon RDS",
                supersedes_memory_ids=[first.id, second.id],
            )
            await _learn_reply(db, learner, _extraction_envelope([candidate]))

            active = await db.fetchall("SELECT id FROM memories WHERE status='active'")
            assert len(active) == 1
            assert active[0]["id"] not in (first.id, second.id)
            for old_id in (first.id, second.id):
                old = await db.get_memory(old_id)
                assert old is not None and old.status == "superseded"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_refined_memory_marks_its_topic_dirty_for_summary_refresh():
    """The refine route returns the refined id, so learn() marks the topic
    dirty with the target id exactly like any other applied change."""

    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            cfg.learning.topic_summaries = True
            seed = await _seed_fact(db)
            await _learn_reply(db, learner, _extraction_envelope([_supersede_candidate(seed.id)]))

            updates = await db.get_topic_updates("database")
            assert updates
            assert any(memory_id == seed.id for memory_id, _ in updates)
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
