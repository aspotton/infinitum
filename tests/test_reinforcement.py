import tempfile
from unittest.mock import AsyncMock

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory, MemoryCandidate, ScoredMemory
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient


async def _learner(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    upstream = UpstreamClient(cfg)
    retriever = MemoryRetriever(db, embeddings, cfg)
    learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
    return cfg, db, embeddings, upstream, retriever, learner


@pytest.mark.asyncio
async def test_semantic_equivalence_reinforces_different_wording():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            e1 = await db.add_event(
                Event(
                    session_id="s1",
                    event_type="message.user",
                    role="user",
                    content="We standardized on PG17 for relational persistence.",
                )
            )
            existing = await db.create_memory(
                Memory(
                    memory_type="decision",
                    topic="database",
                    content="We standardized on PG17 for all relational persistence.",
                    source_event_ids=[e1.id],
                )
            )
            e2 = await db.add_event(
                Event(
                    session_id="s2",
                    event_type="message.user",
                    role="user",
                    content="PostgreSQL version 17 is our primary relational datastore.",
                )
            )
            retriever.search = AsyncMock(
                return_value=[
                    ScoredMemory(
                        memory=existing,
                        score=0.78,
                        semantic_score=0.95,
                        lexical_score=0.20,
                        topic_score=0.8,
                        freshness_score=1.0,
                    )
                ]
            )
            candidate = MemoryCandidate(
                memory_type="decision",
                topic="database",
                content="PostgreSQL version 17 is our primary relational datastore.",
                confidence=0.95,
                importance=0.8,
            )

            affected = await learner._apply(candidate, [e2.id], {existing.id})
            assert affected == {existing.id}
            memories = await db.list_memories(limit=10)
            assert len(memories) == 1
            reinforced = await db.get_memory(existing.id)
            assert reinforced is not None
            assert reinforced.observation_count == 2
            assert reinforced.metadata["last_reinforcement"]["method"] == "semantic_equivalence"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_reinforcement_requires_matching_type_and_topic():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            existing = await db.create_memory(
                Memory(
                    memory_type="decision",
                    topic="database",
                    content="PostgreSQL 17 is the database standard.",
                )
            )
            retriever.search = AsyncMock(
                return_value=[
                    ScoredMemory(
                        memory=existing,
                        score=0.95,
                        semantic_score=0.99,
                        lexical_score=0.99,
                        topic_score=1.0,
                        freshness_score=1.0,
                    )
                ]
            )
            candidate = MemoryCandidate(
                memory_type="fact",
                topic="database",
                content="PostgreSQL 17 is the database standard.",
                operation_hint="reinforce",
                reinforces_memory_id=existing.id,
            )

            await learner._apply(candidate, [], {existing.id})
            memories = await db.list_memories(limit=10)
            assert len(memories) == 2
            old = await db.get_memory(existing.id)
            assert old is not None
            assert old.observation_count == 1
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_model_targeted_reinforce_uses_guarded_lower_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            e1 = await db.add_event(
                Event(session_id="s1", event_type="message.user", role="user", content="Use uv.")
            )
            existing = await db.create_memory(
                Memory(
                    memory_type="preference",
                    topic="python-packaging",
                    content="Prefer uv for Python dependency management.",
                    source_event_ids=[e1.id],
                )
            )
            e2 = await db.add_event(
                Event(
                    session_id="s2",
                    event_type="message.user",
                    role="user",
                    content="Use uv rather than pip for packages.",
                )
            )
            retriever.search = AsyncMock(
                return_value=[
                    ScoredMemory(
                        memory=existing,
                        score=0.61,
                        semantic_score=0.76,
                        lexical_score=0.35,
                        topic_score=0.8,
                        freshness_score=1.0,
                    )
                ]
            )
            candidate = MemoryCandidate(
                memory_type="preference",
                topic="python-packaging",
                content="Use uv rather than pip for Python packages.",
                operation_hint="reinforce",
                reinforces_memory_id=existing.id,
                confidence=0.9,
            )

            await learner._apply(candidate, [e2.id], {existing.id})
            reinforced = await db.get_memory(existing.id)
            assert reinforced is not None
            assert reinforced.observation_count == 2
            assert (
                reinforced.metadata["last_reinforcement"]["method"]
                == "model_targeted_reinforce"
            )
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_explicit_supersede_is_not_accidentally_reinforced():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            existing = await db.create_memory(
                Memory(
                    memory_type="decision",
                    topic="database",
                    content="PostgreSQL 16 is the production database standard.",
                )
            )
            retriever.search = AsyncMock(
                return_value=[
                    ScoredMemory(
                        memory=existing,
                        score=0.98,
                        semantic_score=0.99,
                        lexical_score=0.95,
                        topic_score=1.0,
                        freshness_score=1.0,
                    )
                ]
            )
            candidate = MemoryCandidate(
                memory_type="decision",
                topic="database",
                content="PostgreSQL 17 is now the production database standard.",
                operation_hint="supersede",
                supersedes_memory_ids=[existing.id],
                explicit_correction=True,
            )

            affected = await learner._apply(candidate, [], {existing.id})
            assert existing.id in affected
            old = await db.get_memory(existing.id)
            assert old is not None
            assert old.status == "superseded"
            assert old.observation_count == 1
            active = await db.list_memories(limit=10, status="active")
            assert len(active) == 1
            assert active[0].content.startswith("PostgreSQL 17")
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
