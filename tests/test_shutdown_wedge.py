import asyncio
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infinitum import learning
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import LearningWorker


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
