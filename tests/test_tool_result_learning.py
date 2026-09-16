"""Tests for capped current-turn tool results in learn payloads (learn-tool-results, todo 2).

A parent-session request whose current turn carries a delegation report
(role:"tool" message after the last role:"user" message) enqueues a
learn_interaction whose payload gains a "tool_results" key; tool-less
requests keep exactly today's payload key set. Helper semantics
(window/caps/name resolution/empty) are pinned by direct import below.

Idioms mirror tests/test_learning_subsessions.py: create_app + build_runtime
wired manually (ASGITransport does not run the lifespan, so the learning
worker stays stopped and job rows are never claimed), httpx.MockTransport
faking the upstream, and direct sqlite asserts on jobs.payload_json.
"""

import json
import sqlite3
import tempfile

import httpx

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.routes.openai import (
    TOOL_RESULT_PER_MESSAGE_MAX_CHARS,
    TOOL_RESULTS_TOTAL_MAX_CHARS,
    _tool_results,
)
from infinitum.runtime import build_runtime

PLAIN_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "ship the release"}],
}

# Parent-style tool round: user -> assistant(tool_calls) -> tool(report).
TOOL_ROUND_BODY = {
    "model": "test-model",
    "messages": [
        {"role": "user", "content": "delegate the release"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "task", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "Release v0.6.0 published, PR merged",
        },
    ],
}


def _nonstream_handler(request: httpx.Request) -> httpx.Response:
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


def _sse_handler(request: httpx.Request) -> httpx.Response:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
    }
    final = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    body = (
        f"data: {json.dumps(chunk)}\n\n"
        f"data: {json.dumps(final)}\n\n"
        "data: [DONE]\n\n"
    ).encode()
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def _proxy_app(tmp: str, handler=_nonstream_handler):
    """App + runtime wired for ASGITransport (worker left stopped on purpose)."""
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.topic_summaries = False
    cfg.upstream.passthrough_authorization = False
    app = create_app(cfg)
    rt = await build_runtime(cfg)
    app.state.runtime = rt
    rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app, rt


async def _post_body(app, body: dict) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://infinitum.test"
    ) as client:
        if body.get("stream"):
            async with client.stream(
                "POST", "/v1/chat/completions", json=body
            ) as resp:
                assert resp.status_code == 200
                # Drain to [DONE]; completed() (the enqueue site) runs after the last byte.
                async for _ in resp.aiter_bytes():
                    pass
        else:
            resp = await client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200


def _job_payload(db_path: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM jobs WHERE job_type='learn_interaction'"
        ).fetchone()
    assert row is not None, "no learn_interaction job was enqueued"
    return json.loads(row[0])


async def test_nonstream_tool_round_payload_carries_report():
    # (b) parent tool round: the report text rides the learn payload.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post_body(app, TOOL_ROUND_BODY)
            payload = _job_payload(rt.config.memory.database_path)
            assert "Release v0.6.0 published" in payload.get("tool_results", "")
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_plain_request_payload_keyset_unchanged():
    # (c) baseline characterization: a tool-less request keeps exactly today's
    # payload keys (no "tool_results") before AND after the change.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post_body(app, PLAIN_BODY)
            payload = _job_payload(rt.config.memory.database_path)
            assert set(payload) == {
                "request_id",
                "session_id",
                "model",
                "user_text",
                "assistant_text",
                "source_event_ids",
                "request_context",
            }
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_streaming_tool_round_payload_carries_report():
    # (d) same contract on the streaming completed() enqueue site.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp, handler=_sse_handler)
        try:
            await _post_body(app, {**TOOL_ROUND_BODY, "stream": True})
            payload = _job_payload(rt.config.memory.database_path)
            assert "Release v0.6.0 published" in payload.get("tool_results", "")
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


# --- (a) Direct helper asserts: window / caps / name resolution / empty -------


def _tool_round(name: str = "task", content: str = "Release v0.6.0 published"):
    return [
        {"role": "user", "content": "delegate the release"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]


def test_helper_captures_current_turn_report():
    result = _tool_results(_tool_round())
    assert result == "[task] Release v0.6.0 published"


def test_helper_ignores_tool_message_before_last_user():
    # Old-turn tool results (before the last user message) are out of the window.
    messages = _tool_round(content="old report") + [
        {"role": "user", "content": "next question"}
    ]
    assert _tool_results(messages) == ""


def test_helper_window_is_whole_list_without_user_message():
    messages = [m for m in _tool_round() if m["role"] != "user"]
    result = _tool_results(messages)
    assert "Release v0.6.0 published" in result


def test_helper_caps_entry_at_per_message_limit():
    result = _tool_results(_tool_round(content="y" * 5000))
    assert len(result) <= TOOL_RESULT_PER_MESSAGE_MAX_CHARS
    assert result.endswith("[truncated]")


def test_helper_keeps_whole_entries_under_total_cap():
    big = "x" * 800
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "task", "arguments": "{}"},
                }
                for i in range(10)
            ],
        },
        *[
            {"role": "tool", "tool_call_id": f"c{i}", "content": big}
            for i in range(10)
        ],
    ]
    result = _tool_results(messages)
    assert len(result) <= TOOL_RESULTS_TOTAL_MAX_CHARS
    full = [f"[task] {big}" for _ in range(10)]
    lines = result.split("\n")
    # Whole-entry harvest: the result is an exact whole-entry prefix of the
    # full list, and the next entry genuinely would not fit.
    assert lines == full[: len(lines)]
    assert len(lines) < 10
    assert len("\n".join(full[: len(lines) + 1])) > TOOL_RESULTS_TOTAL_MAX_CHARS


def test_helper_resolves_name_via_tool_call_id():
    result = _tool_results(_tool_round(name="delegate"))
    assert result.startswith("[delegate] ")


def test_helper_falls_back_to_literal_tool_name():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "unknown", "content": "orphan output"},
    ]
    assert _tool_results(messages) == "[tool] orphan output"


def test_helper_empty_without_tool_messages():
    assert _tool_results([]) == ""
    assert _tool_results([{"role": "user", "content": "hi"}]) == ""
