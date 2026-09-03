import json
import tempfile

import httpx
from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig


def test_chat_completions_forwards_and_injects_memory():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["custom_future_field"] == {"kept": True}
        has_memory = any("<infinitum_memory>" in str(m.get("content", "")) for m in body["messages"])
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"memory={has_memory}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        cfg.upstream.passthrough_authorization = False
        app = create_app(cfg)
        with TestClient(app) as client:
            runtime = app.state.runtime
            runtime.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            created = client.post(
                "/memory",
                json={
                    "memory_type": "decision",
                    "topic": "database",
                    "content": "PostgreSQL 17 is the current database standard.",
                    "importance": 1.0,
                    "confidence": 1.0,
                },
            )
            assert created.status_code == 200
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "What database standard do we use?"}],
                    "custom_future_field": {"kept": True},
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "memory=True"
