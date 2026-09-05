import contextlib
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


@contextlib.contextmanager
def _capture_app(tmp: str, captured: list[dict]):
    """App with a MockTransport upstream that records every forwarded body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = False
    cfg.upstream.passthrough_authorization = False
    cfg.memory.tools_enabled = True
    app = create_app(cfg)
    with TestClient(app) as client:
        app.state.runtime.upstream.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        yield client


def _memory_block(body: dict) -> str | None:
    for msg in body["messages"]:
        content = str(msg.get("content", ""))
        if "<infinitum_memory>" in content:
            return content
    return None


def _chat(client, session_id: str) -> None:
    client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What database standard do we use?"}
            ],
        },
        headers={"X-Infinitum-Session-ID": session_id},
    )


def test_pinned_memory_block_is_byte_identical_across_turns():
    captured: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, _capture_app(tmp, captured) as client:
        client.post(
            "/memory",
            json={
                "memory_type": "decision",
                "topic": "database",
                "content": "PostgreSQL 17 is the current database standard.",
                "importance": 1.0,
                "confidence": 1.0,
            },
        )
        _chat(client, "sess-stable")
        _chat(client, "sess-stable")
    first = _memory_block(captured[0])
    assert first is not None and _memory_block(captured[1]) == first


def test_memory_write_between_turns_invalidates_pinned_block():
    captured: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, _capture_app(tmp, captured) as client:
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
        _chat(client, "sess-invalidate")
        assert client.delete(f"/memory/{created.json()['id']}").status_code == 200
        _chat(client, "sess-invalidate")
    assert _memory_block(captured[0]) is not None
    assert _memory_block(captured[1]) != _memory_block(captured[0])


def test_debug_header_reports_zero_tool_rounds_with_empty_db():
    captured: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, _capture_app(tmp, captured) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Anything relevant?"}],
            },
            headers={"X-Infinitum-Session-ID": "sess-empty", "X-Infinitum-Debug": "true"},
        )
    # Documented behavior change: static tool exposure means the header is
    # present with "0" even when no memories (and so no tool rounds) exist.
    assert response.headers["x-infinitum-memory-tool-calls"] == "0"


def test_tools_array_is_byte_identical_across_turns():
    captured: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, _capture_app(tmp, captured) as client:
        client.post(
            "/memory",
            json={
                "memory_type": "decision",
                "topic": "database",
                "content": "PostgreSQL 17 is the current database standard.",
                "importance": 1.0,
                "confidence": 1.0,
            },
        )
        _chat(client, "sess-tools")
        _chat(client, "sess-tools")
    names = [t["function"]["name"] for t in captured[0]["tools"]]
    assert "infinitum_memory_search" in names and "infinitum_memory_get" in names
    assert json.dumps(captured[1]["tools"], sort_keys=True) == json.dumps(
        captured[0]["tools"], sort_keys=True
    )
