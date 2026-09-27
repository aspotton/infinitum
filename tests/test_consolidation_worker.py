import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import MemoryLearner, LearningWorker
from infinitum.models import Memory


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _worker_config(tmp: str) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = True
    cfg.learning.topic_summaries = False  # keep the queue free of summary jobs
    cfg.learning.poll_interval_seconds = 0.05
    return cfg


def _mocked_learner(db: Database, cfg: AppConfig) -> MemoryLearner:
    learner = MemoryLearner(db, MagicMock(), MagicMock(), MagicMock(), cfg)
    learner.learn = AsyncMock(return_value=None)
    learner.refresh_topic_summary = AsyncMock(return_value=False)
    return learner


async def _wait_for_row(db: Database, job_id: str, statuses: set[str], timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    row = None
    while asyncio.get_running_loop().time() < deadline:
        row = await db.fetchone("SELECT status, last_error FROM jobs WHERE id=?", (job_id,))
        if row["status"] in statuses:
            return row
        await asyncio.sleep(0.05)
    return row


def test_last_sweep_initializes_to_neg_inf_so_the_first_poll_is_due():
    # F3's premise, stated directly: the timestamp starts at -inf, not at
    # monotonic()-at-startup, so the first poll is due unconditionally even on
    # a freshly-booted host where time.monotonic() can still be under the 60s
    # tick. No loop needed; construction-time invariant.
    worker = LearningWorker(MagicMock(), MagicMock(), AppConfig())
    assert worker._last_consolidation_sweep == float("-inf")


@pytest.mark.asyncio
async def test_sweep_enqueues_on_first_poll_and_dispatch_threads_topic_model_continuation():
    """Given consolidation on with a model and one active topic, When the worker
    polls (last-sweep -inf makes the very first poll due unconditionally),
    Then the sweep enqueues exactly one consolidate_topic job, the dispatch
    branch awaits consolidator.consolidate_topic with (topic, model,
    continuation=False), and the job reaches done."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        cfg.learning.consolidation = True
        cfg.learning.model = "test-model"
        await db.create_memory(
            Memory(memory_type="fact", topic="dupes", content="We use SQLite.")
        )

        consolidator = MagicMock()
        consolidator.consolidate_topic = AsyncMock(return_value={})
        worker = LearningWorker(db, _mocked_learner(db, cfg), cfg, consolidator=consolidator)
        try:
            worker.start()
            row = None
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                rows = await db.fetchall(
                    "SELECT status FROM jobs WHERE job_type='consolidate_topic'"
                )
                if rows and rows[0]["status"] == "done":
                    row = rows[0]
                    break
                await asyncio.sleep(0.05)
            assert row is not None, await db.fetchall("SELECT * FROM jobs")
        finally:
            await worker.stop()
        consolidator.consolidate_topic.assert_called_once_with(
            "dupes", "test-model", continuation=False
        )
        rows = await db.fetchall("SELECT * FROM jobs WHERE job_type='consolidate_topic'")
        assert len(rows) == 1 and rows[0]["status"] == "done", rows
        assert worker._last_consolidation_sweep > float("-inf")  # stamped when it ran
        await db.close()


@pytest.mark.asyncio
async def test_sweep_tick_does_not_advance_while_upstream_busy_defers():
    """Given the upstream-busy deferral active, When the worker spins deferred
    polls, Then ensure_consolidation_jobs is never called and the tick
    timestamp stays at the -inf init value (deferred iterations never advance
    it); once idle,
    the still-due tick fires exactly once with (interval, batch 1, model)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        cfg.learning.consolidation = True
        cfg.learning.model = "test-model"
        cfg.learning.skip_when_upstream_busy = True

        counter = MagicMock()
        counter.value = 1  # busy -> every iteration defers before the tick
        worker = LearningWorker(
            db, _mocked_learner(db, cfg), cfg,
            active_requests=counter, consolidator=MagicMock(),
        )
        ensure = AsyncMock(return_value=0)
        db.ensure_consolidation_jobs = ensure
        try:
            worker.start()
            await asyncio.sleep(0.3)  # several deferred poll cycles
            assert ensure.await_count == 0
            assert worker._last_consolidation_sweep == float("-inf")  # never stamped while deferred
            counter.value = 0  # go idle: the still-due tick must now fire
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline and ensure.await_count == 0:
                await asyncio.sleep(0.05)
            assert ensure.await_count == 1
            await asyncio.sleep(0.2)  # stamped ~now: the 60s tick is not due again
            assert ensure.await_count == 1
            ensure.assert_awaited_once_with(
                cfg.learning.consolidation_interval_seconds, 1, model="test-model"
            )
        finally:
            await worker.stop()
        await db.close()


@pytest.mark.asyncio
async def test_unknown_job_kind_fails_loudly_never_done():
    """Given a job whose kind the worker does not know, When it is claimed,
    Then the final else raises so the job lands failed with last_error naming
    'unknown job_type' — never the old silent finish_job success."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        cfg.learning.max_attempts = 1  # claim -> attempts 1; retry = 1 < 1 is False
        job_id = await db.enqueue_job("bogus_kind", {})

        worker = LearningWorker(db, _mocked_learner(db, cfg), cfg)
        try:
            worker.start()
            row = await _wait_for_row(db, job_id, {"failed"})
        finally:
            await worker.stop()
        assert row["status"] == "failed", row
        assert "unknown job_type" in row["last_error"]
        await db.close()


@pytest.mark.asyncio
async def test_consolidation_job_missing_topic_or_model_fails_terminal_not_done():
    """Given a consolidate_topic job with neither a payload topic nor a
    configured learning.model, When dispatched, Then it is warned about and
    permanently failed, not silently finished."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        job_id = await db.enqueue_job("consolidate_topic", {"model": ""})

        consolidator = MagicMock()
        consolidator.consolidate_topic = AsyncMock(return_value={})
        worker = LearningWorker(db, _mocked_learner(db, cfg), cfg, consolidator=consolidator)
        try:
            worker.start()
            row = await _wait_for_row(db, job_id, {"failed"})
        finally:
            await worker.stop()
        assert row["status"] == "failed", row
        assert "missing topic or learning.model" in row["last_error"]
        consolidator.consolidate_topic.assert_not_awaited()
        await db.close()


@pytest.mark.asyncio
async def test_poisoned_adopted_job_past_ceiling_fails_without_execution():
    """Given a crashed job left running with a stale lock and attempts at the
    ceiling, When the worker's stale adoption bumps attempts past max, Then the
    right-after-claim guard permanently fails it and the learner never runs."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(days=1))
        await db.execute(
            "UPDATE jobs SET locked_at=?, attempts=? WHERE id=?",
            (stale_lock, cfg.learning.max_attempts, job_id),
        )

        learner = _mocked_learner(db, cfg)
        worker = LearningWorker(db, learner, cfg)
        try:
            worker.start()
            row = await _wait_for_row(db, job_id, {"failed"})
        finally:
            await worker.stop()
        assert row["status"] == "failed", row
        assert row["last_error"] == "max attempts exceeded"
        learner.learn.assert_not_awaited()
        await db.close()


@pytest.mark.asyncio
async def test_legitimate_final_retry_at_ceiling_still_executes_once():
    """Given a pending job one attempt below the ceiling (claim increments
    BEFORE the guard, so post-claim attempts == max_attempts), When the worker
    claims it, Then it still RUNS once and reaches done — the strict '>' does
    not shorten legitimate retries."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        await db.execute(
            "UPDATE jobs SET attempts=? WHERE id=?",
            (cfg.learning.max_attempts - 1, job_id),
        )

        learner = _mocked_learner(db, cfg)
        worker = LearningWorker(db, learner, cfg)
        try:
            worker.start()
            row = await _wait_for_row(db, job_id, {"done", "failed"})
        finally:
            await worker.stop()
        assert row["status"] == "done", row
        assert learner.learn.await_count == 1
        await db.close()


@pytest.mark.asyncio
async def test_gate_off_default_config_never_calls_ensure_consolidation_jobs():
    """Given the default config (consolidation off), When the worker runs a few
    poll cycles with a consolidator present, Then ensure_consolidation_jobs is
    never called and no consolidate_topic row appears (IS-6: inert when off)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = _worker_config(tmp)  # consolidation left at its default False

        consolidator = MagicMock()  # present on purpose: the gate, not None, must stop it
        worker = LearningWorker(db, _mocked_learner(db, cfg), cfg, consolidator=consolidator)
        ensure = AsyncMock(return_value=0)
        db.ensure_consolidation_jobs = ensure
        try:
            worker.start()
            await asyncio.sleep(0.4)  # several poll cycles
        finally:
            await worker.stop()
        ensure.assert_not_awaited()
        rows = await db.fetchall("SELECT id FROM jobs WHERE job_type='consolidate_topic'")
        assert rows == []
        await db.close()
