"""Tests for learning.skip_subsessions (subsession-learning-skip plan, todos 1-2).

Requests carrying a parent-session marker (x-infinitum-parent-session-id /
x-parent-session-id, e.g. OpenCode task-tool child sessions) do not enqueue
learn_interaction jobs by default; events and retrieval are unaffected. The
explicit X-Infinitum-Learning header outranks the config default.

Idioms mirror tests/test_learning_defer.py: create_app + build_runtime wired
manually (ASGITransport does not run the lifespan, so the learning worker
stays stopped and job rows are never claimed), httpx.MockTransport faking a
fixed non-stream upstream completion, and direct sqlite asserts on the jobs/
events tables. Streaming coverage is a later todo, not here.
"""

import sqlite3
import tempfile

import httpx

from infinitum.app import create_app
from infinitum.config import AppConfig, LearningConfig
from infinitum.runtime import build_runtime

CHAT_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
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


async def _proxy_app(tmp: str, **learn_overrides):
    """App + runtime wired for ASGITransport (worker left stopped on purpose)."""
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.topic_summaries = False
    cfg.upstream.passthrough_authorization = False
    for key, value in learn_overrides.items():
        setattr(cfg.learning, key, value)
    app = create_app(cfg)
    rt = await build_runtime(cfg)
    app.state.runtime = rt
    rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(_nonstream_handler))
    return app, rt


async def _post(app, headers: dict[str, str]) -> int:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://infinitum.test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers=headers
        )
    assert resp.status_code == 200
    return resp.status_code


def _job_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT count(*) FROM jobs WHERE job_type='learn_interaction'"
        ).fetchone()
    return int(row[0])


def _event_types(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT event_type FROM events").fetchall()
    return {str(r[0]) for r in rows}


def test_skip_subsessions_default_is_true():
    assert LearningConfig().skip_subsessions is True
    assert AppConfig().learning.skip_subsessions is True


async def test_subsession_skips_learning_but_records_events():
    # (a) parent marker + default config: no learn job, events still recorded.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post(app, {"x-parent-session-id": "ses-parent-1"})
            assert _job_count(rt.config.memory.database_path) == 0
            types = _event_types(rt.config.memory.database_path)
            assert {"message.user", "message.assistant"} <= types
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_flag_off_subsession_enqueues_job():
    # (b) skip_subsessions=False restores learning for subsessions.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp, skip_subsessions=False)
        try:
            await _post(app, {"x-parent-session-id": "ses-parent-1"})
            assert _job_count(rt.config.memory.database_path) == 1
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_explicit_on_outranks_subsession_skip():
    # (c) explicit X-Infinitum-Learning: on wins over the config skip.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post(
                app,
                {
                    "x-parent-session-id": "ses-parent-1",
                    "X-Infinitum-Learning": "on",
                },
            )
            assert _job_count(rt.config.memory.database_path) == 1
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_plain_request_still_enqueues_job():
    # (d) no control headers at all: unchanged default behavior.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post(app, {})
            assert _job_count(rt.config.memory.database_path) == 1
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_explicit_off_still_enqueues_nothing():
    # (e) pins existing behavior: off wins everywhere.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post(app, {"X-Infinitum-Learning": "off"})
            assert _job_count(rt.config.memory.database_path) == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_unparseable_learning_header_treated_as_absent():
    # (f) an unrecognised header value must not flip either direction.
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            await _post(app, {"X-Infinitum-Learning": "maybe"})
            assert _job_count(rt.config.memory.database_path) == 1
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


async def test_streaming_subsession_skips_learning():
    # (g) streaming + parent marker + default config: no learn job. Drain to
    # [DONE] first; completed() (the enqueue site) runs after the last byte.
    def sse_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        rt.upstream.client = httpx.AsyncClient(
            transport=httpx.MockTransport(sse_handler)
        )
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={**CHAT_BODY, "stream": True},
                    headers={"x-parent-session-id": "ses-parent-1"},
                ) as resp:
                    assert resp.status_code == 200
                    body = b"".join([chunk async for chunk in resp.aiter_bytes()])
                    assert b"[DONE]" in body
            assert _job_count(rt.config.memory.database_path) == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()
