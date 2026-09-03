import tempfile
from unittest.mock import AsyncMock

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Memory, TopicSummary
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient


@pytest.mark.asyncio
async def test_topic_updates_are_coalesced_into_one_debounced_job():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        first = await db.create_memory(
            Memory(memory_type="fact", topic="database", content="First database fact")
        )
        second = await db.create_memory(
            Memory(memory_type="fact", topic="database", content="Second database fact")
        )

        await db.mark_topic_dirty(
            "database",
            [first.id],
            model="memory-model",
            debounce_seconds=30,
            update_threshold=5,
        )
        await db.mark_topic_dirty(
            "database",
            [second.id],
            model="memory-model",
            debounce_seconds=30,
            update_threshold=5,
        )

        jobs = await db.fetchall(
            "SELECT * FROM jobs WHERE job_type='refresh_topic_summary' AND status='pending'"
        )
        assert len(jobs) == 1
        assert await db.count_topic_updates("database") == 2
        await db.close()


@pytest.mark.asyncio
async def test_topic_summary_updates_incrementally_from_dirty_memories():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.topic_summary_min_memories = 1
        cfg.learning.topic_summary_context_memories = 1
        cfg.learning.topic_summary_max_changed_memories = 4
        cfg.learning.topic_summary_max_tokens = 333

        db = Database(cfg.memory.database_path)
        await db.connect()
        old_memories = []
        for i in range(6):
            old_memories.append(
                await db.create_memory(
                    Memory(
                        memory_type="fact",
                        topic="database",
                        content=f"Unrelated old detail {i}",
                    )
                )
            )
        changed = await db.create_memory(
            Memory(
                memory_type="decision",
                topic="database",
                content="PostgreSQL 17 is now the database standard.",
            )
        )
        await db.upsert_topic(
            TopicSummary(
                topic="database",
                summary="Old summary baseline that should be updated.",
                memory_count=6,
            )
        )
        await db.mark_topic_dirty(
            "database",
            [changed.id],
            model="memory-model",
            debounce_seconds=30,
            update_threshold=5,
        )

        embeddings = EmbeddingClient(cfg.embeddings)
        upstream = UpstreamClient(cfg)
        retriever = MemoryRetriever(db, embeddings, cfg)
        learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"role": "assistant", "content": "Updated database summary."}}
                ]
            }
        )

        try:
            followup = await learner.refresh_topic_summary("database", "memory-model")
            assert followup is False
            call = upstream.learning_chat_completion.await_args
            prompt = call.kwargs["messages"][1]["content"]
            assert "Old summary baseline" in prompt
            assert "PostgreSQL 17 is now the database standard" in prompt
            # The incremental refresh includes only a small context sample rather
            # than resending all six older memories.
            assert prompt.count("Unrelated old detail") <= 1
            assert call.kwargs["max_tokens"] == 333
            topic = await db.get_topic("database")
            assert topic is not None
            assert topic.summary == "Updated database summary."
            assert await db.count_topic_updates("database") == 0
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_empty_topic_summary_uses_deterministic_fallback_and_clears_dirty_state():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.topic_summary_min_memories = 1
        cfg.learning.topic_summary_fallback_memories = 2

        db = Database(cfg.memory.database_path)
        await db.connect()
        first = await db.create_memory(
            Memory(memory_type="fact", topic="database", content="PostgreSQL 17 is the standard.")
        )
        await db.create_memory(
            Memory(memory_type="decision", topic="database", content="RDS is used for managed production.")
        )
        await db.mark_topic_dirty(
            "database",
            [first.id],
            model="reasoning-model",
            debounce_seconds=0,
            update_threshold=1,
        )

        embeddings = EmbeddingClient(cfg.embeddings)
        upstream = UpstreamClient(cfg)
        retriever = MemoryRetriever(db, embeddings, cfg)
        learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "reasoning consumed the budget",
                        },
                    }
                ],
                "usage": {"completion_tokens": 1024},
            }
        )

        try:
            followup = await learner.refresh_topic_summary("database", "reasoning-model")
            assert followup is False
            topic = await db.get_topic("database")
            assert topic is not None
            assert "PostgreSQL 17 is the standard" in topic.summary
            assert "RDS is used for managed production" in topic.summary
            assert await db.count_topic_updates("database") == 0
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_dirty_topic_is_requeued_after_previous_summary_job_failed():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        memory = await db.create_memory(
            Memory(memory_type="fact", topic="database", content="Persistent database fact")
        )
        await db.mark_topic_dirty(
            "database",
            [memory.id],
            model="memory-model",
            debounce_seconds=0,
            update_threshold=1,
        )
        job = await db.claim_job()
        assert job is not None
        assert job["job_type"] == "refresh_topic_summary"
        await db.fail_job(job["id"], "empty output", retry=False)

        recovered = await db.recover_dirty_topic_summary_jobs()
        assert recovered == 1
        rows = await db.fetchall(
            "SELECT * FROM jobs WHERE job_type='refresh_topic_summary' AND status='pending'"
        )
        assert len(rows) == 1
        assert await db.count_topic_updates("database") == 1
        await db.close()
