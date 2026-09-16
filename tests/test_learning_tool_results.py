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

_EMPTY_JSON = json.dumps({"memories": []})


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


def _payload(**extra):
    payload = {
        "request_id": "req_1",
        "session_id": "ses_1",
        "model": "memory-model",
        "user_text": "Delegate the release notes task.",
        "assistant_text": "Done, the sub-agent published it.",
        "source_event_ids": [],
        "request_context": {},
    }
    payload.update(extra)
    return payload


async def _prompt_of_single_call(upstream) -> str:
    upstream.learning_chat_completion.assert_awaited_once()
    call = upstream.learning_chat_completion.await_args
    return call.kwargs["messages"][1]["content"]


@pytest.mark.asyncio
async def test_learn_prompt_includes_tool_results_section_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": _EMPTY_JSON,
                            "tool_calls": None,
                        },
                    }
                ]
            }
        )
        try:
            await learner.learn(
                _payload(tool_results="[task] Release v0.6.0 published, PR merged")
            )
            prompt = await _prompt_of_single_call(upstream)
            assert "TOOL RESULTS" in prompt
            assert "Release v0.6.0 published, PR merged" in prompt
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_learn_prompt_omits_tool_results_section_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": _EMPTY_JSON,
                            "tool_calls": None,
                        },
                    }
                ]
            }
        )
        try:
            await learner.learn(_payload())
            prompt = await _prompt_of_single_call(upstream)
            assert "TOOL RESULTS" not in prompt
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_learn_prompt_without_tool_results_keeps_stable_sections():
    """Byte-compat proxy: the no-tool_results prompt still carries the
    pre-existing Interaction:/Schema: anchors in their original places."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        upstream.learning_chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": _EMPTY_JSON,
                            "tool_calls": None,
                        },
                    }
                ]
            }
        )
        try:
            await learner.learn(_payload())
            prompt = await _prompt_of_single_call(upstream)
            assert "Interaction:" in prompt
            assert "Schema:" in prompt
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
