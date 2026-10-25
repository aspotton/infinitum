"""Deterministic episodic duplicate-skip + the broadened boilerplate prompt ban.

An episodic candidate that is near-verbatim (``memory.dedup_similarity``) to a
still-live active episodic memory in the nearby set adds no new information, so
``MemoryLearner._apply`` skips the create entirely: zero mutations, and the
diagnostic ``episodic_skips_total`` meta counter (read by GET /status as
``skips.episodic_total``) goes up. Explicit corrections must never be silently
dropped, so the correction/supersede path stays reachable. Learner construction
follows ``tests/test_reinforcement.py``.
"""

import json
import tempfile
from unittest.mock import AsyncMock

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Memory, MemoryCandidate, ScoredMemory
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient

_SEED_CONTENT = "Shipped the v0.6.2 release on 2026-10-12 after the QA pass."
_NEAR_VERBATIM = "Shipped the v0.6.2 release on 2026-10-12 after the quick QA pass."
_REWORDED = "Rolled out the v0.6.2 release after QA signed off on 2026-10-12."


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


async def _seed_episodic(db: Database) -> Memory:
    return await db.create_memory(
        Memory(memory_type="episodic", topic="releases", content=_SEED_CONTENT)
    )


def _hit(memory: Memory) -> ScoredMemory:
    return ScoredMemory(
        memory=memory,
        score=0.7,
        semantic_score=0.0,
        lexical_score=0.9,
        topic_score=0.0,
        freshness_score=1.0,
    )


async def test_near_verbatim_episodic_skips_and_counts():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            seeded = await _seed_episodic(db)
            retriever.search = AsyncMock(return_value=[_hit(seeded)])
            candidate = MemoryCandidate(
                memory_type="episodic",
                topic="deploys",
                content=_NEAR_VERBATIM,
            )
            affected = await learner._apply(candidate, [], {seeded.id})
            assert affected == set()
            active = await db.list_memories(status="active", limit=10)
            assert [m.id for m in active] == [seeded.id]
            assert await db.get_meta_value("episodic_skips_total") == "1"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_below_floor_episodic_creates_normally():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            seeded = await _seed_episodic(db)
            retriever.search = AsyncMock(return_value=[_hit(seeded)])
            candidate = MemoryCandidate(
                memory_type="episodic",
                topic="deploys",
                content=_REWORDED,
            )
            affected = await learner._apply(candidate, [], {seeded.id})
            assert affected and seeded.id not in affected
            assert len(await db.list_memories(status="active", limit=10)) == 2
            assert await db.get_meta_value("episodic_skips_total") is None
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_guard_never_fires_for_non_episodic_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            seeded = await _seed_episodic(db)
            retriever.search = AsyncMock(return_value=[_hit(seeded)])
            candidate = MemoryCandidate(
                memory_type="fact",
                topic="release-notes",
                content=_NEAR_VERBATIM,
            )
            affected = await learner._apply(candidate, [], {seeded.id})
            assert affected and seeded.id not in affected
            assert len(await db.list_memories(status="active", limit=10)) == 2
            assert await db.get_meta_value("episodic_skips_total") is None
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_explicit_correction_still_supersedes_despite_near_verbatim_twin():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, retriever, learner = await _learner(tmp)
        try:
            seeded = await _seed_episodic(db)
            retriever.search = AsyncMock(return_value=[_hit(seeded)])
            candidate = MemoryCandidate(
                memory_type="episodic",
                topic="incidents",
                content=_NEAR_VERBATIM,
                operation_hint="supersede",
                explicit_correction=True,
                supersedes_memory_ids=[seeded.id],
            )
            affected = await learner._apply(candidate, [], {seeded.id})
            assert seeded.id in affected
            assert (await db.get_memory(seeded.id)).status == "superseded"
            assert len(affected) == 2
            new_id = next(iter(affected - {seeded.id}))
            assert (await db.get_memory(new_id)).status == "active"
            assert await db.get_meta_value("episodic_skips_total") is None
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


async def test_extraction_prompt_broadens_boilerplate_ban():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        db = Database(cfg.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(cfg.embeddings)
        upstream = UpstreamClient(cfg)
        retriever = MemoryRetriever(db, embeddings, cfg)
        learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
        try:
            envelope = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"memories": []}),
                            "tool_calls": None,
                        },
                    }
                ]
            }
            mock = AsyncMock(return_value=envelope)
            learner.upstream.learning_chat_completion = mock
            await learner.learn(
                {
                    "request_id": "req_1",
                    "session_id": "ses_1",
                    "model": "memory-model",
                    "user_text": "continue the plan",
                    "assistant_text": "Plan execution complete in 3m12s.",
                    "source_event_ids": [],
                    "request_context": {},
                }
            )
            assert mock.await_count == 1
            prompt = mock.call_args.kwargs["messages"][1]["content"]
            assert "plan-execution-complete or elapsed-time lines, branch and PR housekeeping" in prompt
            assert "memory only when it carries a durable decision" in prompt
            assert prompt.count("session-start boilerplate") == 1
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
