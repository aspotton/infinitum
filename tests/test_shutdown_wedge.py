import asyncio
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from infinitum import learning
from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import LearningWorker, MemoryLearner


@pytest.mark.asyncio
async def test_stop_cancels_wedged_job_within_grace():
    """Given a worker wedged in a long learn call, When stop() is called, Then
    it returns within the grace window and the job stays recoverable as pending
    after recover_interrupted_jobs()."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/t.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        cfg = AppConfig()
        cfg.learning.poll_interval_seconds = 0.05
        cfg.embeddings.enabled = False
        learner = MagicMock()

        # Bounded-but-longer-than-grace hang: a never-resolving Event().wait() would make a
        # regressed stop() HANG the suite forever instead of failing it (pytest-timeout is
        # not a dependency). asyncio.sleep(30) is cancel-safe (same CancelledError path)
        # and bounds the broken-fix case to ~30s of clean failure. Must be an async def:
        # AsyncMock does not await coroutines returned by a plain-callable side_effect.
        async def _hang(payload):
            await asyncio.sleep(30)

        learner.learn = AsyncMock(side_effect=_hang)
        worker = LearningWorker(db, learner, cfg)
        with patch.object(learning, "STOP_GRACE_SECONDS", 0.1):
            worker.start()
            for _ in range(100):  # wait deterministically for the claim
                row = await db.fetchone("SELECT status FROM jobs WHERE id=?", (job_id,))
                if row["status"] == "running":
                    break
                await asyncio.sleep(0.02)
            assert row["status"] == "running"
            t0 = time.monotonic()
            await worker.stop()
            elapsed = time.monotonic() - t0
        assert elapsed < 2.0
        row = await db.fetchone("SELECT status, last_error FROM jobs WHERE id=?", (job_id,))
        assert row["status"] == "running" and row["last_error"] is None
        assert await db.recover_interrupted_jobs() == 1
        row = await db.fetchone("SELECT status FROM jobs WHERE id=?", (job_id,))
        assert row["status"] == "pending"
        await db.close()


@pytest.mark.asyncio
async def test_stop_of_idle_worker_is_unchanged():
    """Given a worker with no jobs, When stop() is called, Then it returns
    promptly (<0.5s), the task is cleared to None, and no exception is raised."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/t.db")
        await db.connect()
        cfg = AppConfig()
        cfg.learning.poll_interval_seconds = 0.05
        cfg.embeddings.enabled = False
        learner = MagicMock()
        worker = LearningWorker(db, learner, cfg)
        worker.start()
        t0 = time.monotonic()
        await worker.stop()
        elapsed = time.monotonic() - t0
    assert elapsed < 0.5
    assert worker._task is None
    await db.close()


def _seed_job(path: str) -> None:
    async def go() -> None:
        db = Database(path)
        await db.connect()
        await db.enqueue_job("learn_interaction", {"event_id": "evt_seed"})
        await db.close()
    asyncio.run(go())


def _job_row(path: str) -> tuple[str, str | None]:
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT status, last_error FROM jobs LIMIT 1").fetchone()
    finally:
        conn.close()
    return row[0], row[1]


def _recover(path: str) -> int:
    async def go() -> int:
        db = Database(path)
        await db.connect()
        try:
            return await db.recover_interrupted_jobs()
        finally:
            await db.close()
    return asyncio.run(go())


async def _hang(self, payload):
    # Bounded-on-purpose: a never-resolving wait would make a regressed stop() HANG the
    # suite forever (no pytest-timeout dependency); 30 s gives the broken-fix control a
    # clean, bounded failure while the fix cuts it to ~0.2 s via cancel-after-grace.
    await asyncio.sleep(30)


def test_lifespan_shutdown_bounded_by_grace_not_upstream_timeout():
    """Given a seeded job claimed by a worker whose learn hangs, When the
    TestClient exits its lifespan, Then shutdown completes in under 3 s, the job
    row is left (running, None), and recovery requeues it exactly once."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/life.db"
        _seed_job(path)
        cfg = AppConfig()
        cfg.memory.database_path = path
        cfg.learning.enabled = True
        cfg.learning.poll_interval_seconds = 0.05
        cfg.embeddings.enabled = False
        # No learning.timeout_seconds tweak: the patched learn bypasses the HTTP layer
        # where that config applies; the broken-fix ceiling is _hang's 30 s sleep.
        with (
            patch.object(learning, "STOP_GRACE_SECONDS", 0.2),
            patch.object(MemoryLearner, "learn", new=_hang),
        ):
            client = TestClient(create_app(cfg))
            client.__enter__()
            try:
                deadline = time.monotonic() + 5.0
                while _job_row(path)[0] != "running":
                    assert time.monotonic() < deadline, "worker never claimed the job"
                    time.sleep(0.02)
                t0 = time.monotonic()
            finally:
                client.__exit__(None, None, None)
            elapsed = time.monotonic() - t0
        assert elapsed < 3.0
        assert _job_row(path) == ("running", None)
        assert _recover(path) == 1
