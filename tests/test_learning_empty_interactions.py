import json
import tempfile
from unittest.mock import AsyncMock

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
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


def _no_memories_response() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"memories": []}),
                },
                "finish_reason": "stop",
            }
        ]
    }


def _payload(user_text: str, assistant_text: str) -> dict:
    return {
        "request_id": "req_1",
        "session_id": "ses_1",
        "model": "m",
        "user_text": user_text,
        "assistant_text": assistant_text,
        "source_event_ids": ["evt_1", "evt_2"],
        "request_context": {},
    }


@pytest.mark.asyncio
async def test_learn_skips_upstream_when_assistant_text_empty():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_no_memories_response()
        )
        retriever = learner.retriever
        retriever.search = AsyncMock(return_value=[])
        try:
            await learner.learn(
                _payload(
                    "[internal] Continue from the previous assistant state.", ""
                )
            )
            assert not upstream.learning_chat_completion.called
            assert not retriever.search.called
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_learn_skips_whitespace_only_assistant_text():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_no_memories_response()
        )
        retriever = learner.retriever
        retriever.search = AsyncMock(return_value=[])
        try:
            await learner.learn(_payload("What did we decide?", "   \n "))
            assert not upstream.learning_chat_completion.called
            assert not retriever.search.called
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_learn_extracts_when_assistant_text_present():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_no_memories_response()
        )
        retriever = learner.retriever
        retriever.search = AsyncMock(return_value=[])
        try:
            await learner.learn(_payload("We use PostgreSQL 17.", "Understood."))
            assert upstream.learning_chat_completion.await_count == 1
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
