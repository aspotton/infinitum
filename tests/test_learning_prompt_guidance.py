"""Pin the extraction prompt: the generic how-to ban must reach the learning upstream.

Live eval showed a generic "how do I wipe docker volumes" Q&A turn was learned as
a ``procedure`` memory and leaked into retrieval probes. The extraction prompt now
bans generic how-to answers; this test drives one ``learn()`` turn with a mocked
extraction call and asserts the rendered user prompt (messages[1]) carries the
ban sentence. Learner construction follows ``tests/test_learning_refine.py``.
"""

import json
import tempfile
from unittest.mock import AsyncMock

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient

_BAN_SUBSTRING = "a procedure is memory only when it describes the user's own project"


async def test_extraction_prompt_bans_generic_howto_answers():
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
                    "user_text": "how do I wipe docker volumes?",
                    "assistant_text": "You can remove unused volumes with docker volume rm.",
                    "source_event_ids": [],
                    "request_context": {},
                }
            )
            assert mock.await_count == 1
            prompt = mock.call_args.kwargs["messages"][1]["content"]
            assert _BAN_SUBSTRING in prompt
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
