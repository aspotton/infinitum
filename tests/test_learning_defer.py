"""Tests for learning.skip_when_upstream_busy (scenarios A-G of
.omo/plans/defer-learning-when-upstream-busy.md).

Idioms mirror tests/test_api.py (httpx.MockTransport swapped onto
runtime.upstream.client) and tests/test_incremental_topics.py (direct
Database/build_runtime + AsyncMock on learner).
"""

import asyncio
import tempfile
from unittest.mock import AsyncMock

import httpx
import pytest

from infinitum.app import create_app
from infinitum.config import AppConfig, LearningConfig
from infinitum.database import Database
from infinitum.routes.openai import _counted
from infinitum.runtime import ActiveRequestCounter, build_runtime

# Minimal learn_interaction payload; learner.learn is always mocked in
# these tests so the worker only needs a job row to claim.
JOB_PAYLOAD = {
    "request_id": "req_test",
    "session_id": "ses_test",
    "model": "test-model",
    "user_text": "hi",
    "assistant_text": "hello",
    "source_event_ids": [],
    "request_context": {},
}


def _defer_config(
    tmp: str, *, skip: bool, grace: float = 0.0, poll: float = 0.05
) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = True
    cfg.learning.topic_summaries = False  # keep the queue free of summary jobs
    cfg.learning.poll_interval_seconds = poll
    cfg.learning.skip_when_upstream_busy = skip
    cfg.learning.upstream_idle_grace_seconds = grace
    cfg.upstream.passthrough_authorization = False
    return cfg


async def _job_row(db: Database, job_id: str) -> dict:
    row = await db.fetchone(
        "SELECT status, attempts FROM jobs WHERE id=?", (job_id,)
    )
    assert row is not None
    return dict(row)


async def _wait_for_job_done(db: Database, job_id: str, timeout: float = 3.0) -> None:
    """Poll the durable jobs row until finish_job marks it 'done'."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        row = await _job_row(db, job_id)
        if row["status"] == "done":
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} never reached done: {row}")


@pytest.mark.asyncio
async def test_worker_defers_while_upstream_busy_and_runs_after_drain():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _defer_config(tmp, skip=True)
        rt = await build_runtime(cfg)
        try:
            learn = AsyncMock(return_value=None)
            rt.learner.learn = learn
            job_id = await rt.db.enqueue_job("learn_interaction", JOB_PAYLOAD)

            rt.active_requests.increment()
            rt.worker.start()
            # Several poll intervals' worth of "busy" time.
            await asyncio.sleep(0.3)
            row = await _job_row(rt.db, job_id)
            assert row["status"] == "pending"
            assert row["attempts"] == 0
            assert learn.await_count == 0

            rt.active_requests.decrement()
            await _wait_for_job_done(rt.db, job_id)
            assert learn.await_count == 1
        finally:
            await rt.worker.stop()
            await rt.db.close()


@pytest.mark.asyncio
async def test_flag_off_claims_while_counter_busy():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _defer_config(tmp, skip=False)
        rt = await build_runtime(cfg)
        try:
            learn = AsyncMock(return_value=None)
            rt.learner.learn = learn
            job_id = await rt.db.enqueue_job("learn_interaction", JOB_PAYLOAD)

            # Counter held >0 the whole time; with the flag off the worker
            # must behave exactly like before the feature existed.
            rt.active_requests.increment()
            rt.worker.start()
            await _wait_for_job_done(rt.db, job_id)
            assert learn.await_count == 1
            assert rt.active_requests.value == 1
        finally:
            await rt.worker.stop()
            await rt.db.close()


def _nonstream_handler() -> httpx.Response:
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


async def _proxy_app(tmp: str):
    """App + runtime wired for ASGITransport (lifespan is not run by that
    transport, so build_runtime is called manually and the worker left stopped)."""
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = False
    cfg.upstream.passthrough_authorization = False
    app = create_app(cfg)
    rt = await build_runtime(cfg)
    app.state.runtime = rt
    return app, rt


CHAT_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ok", "connect-error"])
async def test_non_stream_drains_counter(mode):
    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "connect-error":
            raise httpx.ConnectError("boom", request=request)
        return _nonstream_handler()

    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                resp = await client.post("/v1/chat/completions", json=CHAT_BODY)
            if mode == "ok":
                assert resp.status_code == 200
            else:
                assert resp.status_code == 502
            assert rt.active_requests.value == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


def _sse_bytes(*chunks: str) -> bytes:
    return "".join(chunk for chunk in chunks).encode()


SSE_OK = _sse_bytes(
    'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n',
    "data: [DONE]\n\n",
)


@pytest.mark.asyncio
async def test_stream_success_drains_counter():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=SSE_OK,
        )

    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                # Two full streams: a double-decrement bug would leave the
                # saturating counter at 0 anyway, but a leaked increment or a
                # non-saturating negative count fails here.
                for _ in range(2):
                    async with client.stream(
                        "POST", "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
                    ) as resp:
                        assert resp.status_code == 200
                        body = b"".join([chunk async for chunk in resp.aiter_bytes()])
                        assert b"[DONE]" in body
                    assert rt.active_requests.value == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


@pytest.mark.asyncio
async def test_stream_upstream_4xx_drains_counter():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions", json={**CHAT_BODY, "stream": True}
                )
                assert resp.status_code == 400
            assert rt.active_requests.value == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_drains_counter():
    # httpx's ASGITransport (0.28) fully buffers the app call before the
    # client ever sees bytes (see httpx/_transports/asgi.py: it awaits
    # `self.app(...)` to completion, then returns a buffered stream), so an
    # in-process client cannot disconnect mid-stream and a real uvicorn
    # cannot be driven deterministically in a unit test. This is the same
    # GeneratorExit path a real ASGI server triggers: on disconnect it calls
    # aclose() on the response body iterator, which is routes/openai._counted
    # itself. Closing that generator after one chunk must run its finally.
    counter = ActiveRequestCounter()
    counter.increment()

    async def fake_stream():
        yield b"data: partial\n\n"
        yield b"data: never-reached\n\n"  # client is gone before this

    gen = _counted(fake_stream(), counter)
    first = await gen.__anext__()
    assert first == b"data: partial\n\n"
    assert counter.value == 1  # still held mid-stream
    await gen.aclose()  # GeneratorExit, exactly what ASGI disconnect does
    assert counter.value == 0


@pytest.mark.asyncio
async def test_health_exposes_active_requests():
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _proxy_app(tmp)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                rt.active_requests.increment()
                resp = await client.get("/health")
                assert resp.status_code == 200
                payload = resp.json()
                assert isinstance(payload["active_requests"], int)
                assert payload["active_requests"] == rt.active_requests.value == 1
                rt.active_requests.decrement()
                resp = await client.get("/health")
                assert resp.json()["active_requests"] == 0
        finally:
            await rt.upstream.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_grace_defers_claim_until_window_passes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _defer_config(tmp, skip=True, grace=0.3, poll=0.02)
        rt = await build_runtime(cfg)
        try:
            learn = AsyncMock(return_value=None)
            rt.learner.learn = learn
            job_id = await rt.db.enqueue_job("learn_interaction", JOB_PAYLOAD)

            rt.active_requests.increment()
            rt.worker.start()
            await asyncio.sleep(0.05)
            rt.active_requests.decrement()  # counter drains; grace window starts
            # Still inside the window: the drained counter alone must not
            # re-enable claiming.
            await asyncio.sleep(0.15)
            row = await _job_row(rt.db, job_id)
            assert row["status"] == "pending"
            assert learn.await_count == 0
            # The window expires ~0.1s from now; the poll loop claims then.
            await _wait_for_job_done(rt.db, job_id)
            assert learn.await_count == 1
        finally:
            await rt.worker.stop()
            await rt.db.close()


@pytest.mark.asyncio
async def test_traffic_during_grace_restarts_window():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _defer_config(tmp, skip=True, grace=0.3, poll=0.02)
        rt = await build_runtime(cfg)
        try:
            learn = AsyncMock(return_value=None)
            rt.learner.learn = learn
            job_id = await rt.db.enqueue_job("learn_interaction", JOB_PAYLOAD)

            rt.active_requests.increment()
            rt.worker.start()
            await asyncio.sleep(0.05)
            rt.active_requests.decrement()  # idle; window A starts
            await asyncio.sleep(0.15)
            # A request comes and goes inside window A, restarting the
            # window from THIS activity, not window A's start.
            rt.active_requests.increment()
            rt.active_requests.decrement()
            await asyncio.sleep(0.15)  # > 0.3s after window A, ~0.15s after B
            row = await _job_row(rt.db, job_id)
            assert row["status"] == "pending"
            assert learn.await_count == 0
            await _wait_for_job_done(rt.db, job_id)
            assert learn.await_count == 1
        finally:
            await rt.worker.stop()
            await rt.db.close()


@pytest.mark.asyncio
async def test_grace_zero_claims_once_counter_drains():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _defer_config(tmp, skip=True, grace=0.0)
        rt = await build_runtime(cfg)
        try:
            learn = AsyncMock(return_value=None)
            rt.learner.learn = learn
            job_id = await rt.db.enqueue_job("learn_interaction", JOB_PAYLOAD)

            rt.active_requests.increment()
            rt.worker.start()
            await asyncio.sleep(0.1)
            assert (await _job_row(rt.db, job_id))["status"] == "pending"
            rt.active_requests.decrement()
            # Grace 0 must claim promptly on the next poll, with no window.
            await _wait_for_job_done(rt.db, job_id, timeout=1.0)
            assert learn.await_count == 1
        finally:
            await rt.worker.stop()
            await rt.db.close()


def test_upstream_idle_grace_default_is_zero():
    assert LearningConfig().upstream_idle_grace_seconds == 0.0
    assert AppConfig().learning.upstream_idle_grace_seconds == 0.0
