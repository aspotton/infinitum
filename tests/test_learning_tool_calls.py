import json
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


@pytest.mark.asyncio
async def test_memory_extraction_accepts_schema_shaped_tool_call_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "emit_memories",
                                        "arguments": json.dumps(
                                            {
                                                "memories": [
                                                    {
                                                        "memory_type": "decision",
                                                        "topic": "database",
                                                        "content": "PostgreSQL 17 is the database standard.",
                                                        "importance": 0.8,
                                                        "confidence": 0.95,
                                                        "operation_hint": "new",
                                                        "reinforces_memory_id": None,
                                                        "supersedes_memory_ids": [],
                                                        "explicit_correction": False,
                                                        "reason": "Explicit decision",
                                                    }
                                                ]
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"completion_tokens": 39},
            }
        )
        try:
            await learner.learn(
                {
                    "model": "memory-model",
                    "user_text": "We use PostgreSQL 17.",
                    "assistant_text": "Understood.",
                    "source_event_ids": [],
                }
            )
            memories = await db.list_memories(limit=10)
            assert len(memories) == 1
            assert memories[0].memory_type == "decision"
            assert memories[0].topic == "database"
            assert memories[0].content == "PostgreSQL 17 is the database standard."
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_memory_extraction_ignores_unrelated_tool_call_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": "/tmp/test"}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        try:
            await learner.learn(
                {
                    "model": "memory-model",
                    "user_text": "Remember something.",
                    "assistant_text": "Okay.",
                    "source_event_ids": [],
                }
            )
            assert await db.list_memories(limit=10) == []
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_topic_summary_accepts_summary_from_tool_call_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        cfg.learning.topic_summary_min_memories = 1
        memory = await db.create_memory(
            Memory(memory_type="fact", topic="database", content="PostgreSQL 17 is standard.")
        )
        await db.upsert_topic(
            TopicSummary(topic="database", summary="Old summary.", memory_count=1)
        )
        await db.mark_topic_dirty(
            "database",
            [memory.id],
            model="memory-model",
            debounce_seconds=0,
            update_threshold=1,
        )
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "emit_summary",
                                        "arguments": json.dumps(
                                            {"summary": "PostgreSQL 17 remains the database standard."}
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        try:
            followup = await learner.refresh_topic_summary("database", "memory-model")
            assert followup is False
            topic = await db.get_topic("database")
            assert topic is not None
            assert topic.summary == "PostgreSQL 17 remains the database standard."
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
