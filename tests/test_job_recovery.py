import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import MemoryLearner, LearningWorker
from infinitum.models import Event, Memory
from infinitum.runtime import build_runtime
from infinitum.upstream import UpstreamClient


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.mark.asyncio
async def test_stale_running_job_is_reclaimed_after_its_lock_expires():
    """Given a crashed job left running with an old locked_at, When the worker
    claims with a stale_cutoff past that lock, Then the same job is re-claimed
    with attempts incremented and can finish cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})

        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        assert first["attempts"] == 1

        # Backdate the lock one day via SQL; no sleeps, deterministic state.
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(days=1))
        await db.execute("UPDATE jobs SET locked_at=? WHERE id=?", (stale_lock, job_id))

        cutoff = _iso(datetime.now(timezone.utc) - timedelta(seconds=660))
        second = await db.claim_job(stale_cutoff=cutoff)
        assert second is not None
        assert second["id"] == job_id
        assert second["attempts"] == 2

        await db.finish_job(job_id)
        row = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert row["status"] == "done"


@pytest.mark.asyncio
async def test_worker_steal_of_a_fresh_running_lock_is_refused():
    """Given a job claimed moments ago, When another claim runs with a recent
    stale_cutoff, Then it returns None and the row is untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})

        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        before = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))

        cutoff = _iso(datetime.now(timezone.utc) - timedelta(seconds=660))
        stolen = await db.claim_job(stale_cutoff=cutoff)
        assert stolen is None

        after = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert after["status"] == before["status"] == "running"
        assert after["attempts"] == before["attempts"] == 1
        assert after["locked_at"] == before["locked_at"]


@pytest.mark.asyncio
async def test_stale_cutoff_binds_locked_at_not_run_after():
    """Given the only job is pending with a future run_after, When a claim uses
    a past stale_cutoff, Then run_after still gates on now (job deferred);
    once run_after is past, the same call claims it. Proves the cutoff binds
    locked_at only, never run_after."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        future = _iso(datetime.now(timezone.utc) + timedelta(seconds=10))
        await db.execute("UPDATE jobs SET run_after=? WHERE id=?", (future, job_id))

        cutoff = _iso(datetime.now(timezone.utc) - timedelta(seconds=660))
        assert await db.claim_job(stale_cutoff=cutoff) is None

        past = _iso(datetime.now(timezone.utc) - timedelta(seconds=1))
        await db.execute("UPDATE jobs SET run_after=? WHERE id=?", (past, job_id))
        claimed = await db.claim_job(stale_cutoff=cutoff)
        assert claimed is not None
        assert claimed["id"] == job_id


@pytest.mark.asyncio
async def test_startup_requeue_preserves_attempts_so_a_poisoned_job_stays_poisoned():
    """Given a job left running by a crash, When startup recovery requeues it,
    Then attempts survive so a later failure still respects max_attempts instead
    of restarting an unbounded requeue loop."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})

        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(hours=3))
        await db.execute("UPDATE jobs SET locked_at=? WHERE id=?", (stale_lock, job_id))

        assert await db.recover_interrupted_jobs() == 1
        row = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert row["locked_at"] is None

        second = await db.claim_job()
        assert second is not None and second["attempts"] == 2

        now = datetime.now(timezone.utc)
        await db.fail_job(job_id, str(RuntimeError("boom")), retry=True, delay_seconds=60.0)
        row = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert row["status"] == "pending"
        assert row["attempts"] == 2
        assert row["run_after"] > _iso(now)


@pytest.mark.asyncio
async def test_startup_requeue_is_a_no_op_for_an_unexpired_pending_job():
    """Given only a pending job due in the future, When startup recovery runs,
    Then it requeues nothing and the row is byte-for-byte untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        future = _iso(datetime.now(timezone.utc) + timedelta(seconds=600))
        await db.execute("UPDATE jobs SET run_after=? WHERE id=?", (future, job_id))
        before = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))

        assert await db.recover_interrupted_jobs() == 0

        after = await db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        assert after["status"] == before["status"] == "pending"
        assert after["run_after"] == before["run_after"]
        assert after["locked_at"] == before["locked_at"]
        assert after["created_at"] == before["created_at"]


@pytest.mark.asyncio
async def test_stale_running_job_reclaimed_by_worker():
    """Given a job left running by a crash whose lock is a day old, When the
    worker polls with the lease cutoff derived from learning.timeout_seconds,
    Then it adopts the orphan and the row reaches done."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()

        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = True
        cfg.learning.topic_summaries = False  # keep the queue free of summary jobs
        cfg.learning.poll_interval_seconds = 0.05
        # learning.timeout_seconds intentionally left at its default.

        job_id = await db.enqueue_job("learn_interaction", {"event_id": "evt_1"})
        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(days=1))
        await db.execute("UPDATE jobs SET locked_at=? WHERE id=?", (stale_lock, job_id))

        # The worker only needs a claimable row; extraction itself is mocked so
        # the test exercises the lease wiring, not the learner.
        learner = MemoryLearner(db, MagicMock(), MagicMock(), MagicMock(), cfg)
        learner.learn = AsyncMock(return_value=None)

        worker = LearningWorker(db, learner, cfg)
        worker.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5.0
            row = None
            while asyncio.get_running_loop().time() < deadline:
                row = await db.fetchone("SELECT status FROM jobs WHERE id=?", (job_id,))
                if row["status"] == "done":
                    break
                await asyncio.sleep(0.1)
            assert row["status"] == "done", row
        finally:
            await worker.stop()
            await db.close()


@pytest.mark.asyncio
async def test_startup_integration_recovers_stale_job_and_worker_finishes_it():
    """Given a crashed job left running with a day-old lock, When build_runtime
    starts (topic summaries off) and the worker polls, Then startup recovery
    requeues it and the worker drives the row to done."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/runtime.db"
        db = Database(db_path)
        await db.connect()
        job_id = await db.enqueue_job("learn_interaction", {
            "event_id": "evt_1",
            "model": "test-model",
            "user_text": "What database do we use?",
            "assistant_text": "We use SQLite.",
        })
        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(days=1))
        await db.execute("UPDATE jobs SET locked_at=? WHERE id=?", (stale_lock, job_id))
        await db.close()

        cfg = AppConfig()
        cfg.memory.database_path = db_path
        cfg.learning.enabled = True
        cfg.learning.topic_summaries = False
        cfg.learning.poll_interval_seconds = 0.05

        empty = {"choices": [{"message": {"content": '{"memories": []}'}}]}
        with patch.object(
            UpstreamClient, "learning_chat_completion", AsyncMock(return_value=empty)
        ):
            rt = await build_runtime(cfg)
            try:
                rt.worker.start()
                deadline = asyncio.get_running_loop().time() + 5.0
                row = None
                while asyncio.get_running_loop().time() < deadline:
                    row = await rt.db.fetchone("SELECT status FROM jobs WHERE id=?", (job_id,))
                    if row["status"] == "done":
                        break
                    await asyncio.sleep(0.1)
                assert row["status"] == "done", row
            finally:
                await rt.worker.stop()
                await rt.upstream.close()
                await rt.embeddings.close()
                await rt.db.close()


@pytest.mark.asyncio
async def test_startup_recovery_after_orphaned_summary_job_causes_no_double_enqueue():
    """Given a crash-orphaned running summary job plus dirty topic state, When
    build_runtime runs its startup passes (worker never started), Then the
    orphan is normalized to pending and topic recovery adds no second row."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/runtime.db"
        db = Database(db_path)
        await db.connect()
        # SEED ORDER IS LOAD-BEARING: the memory must exist before
        # mark_topic_dirty (topic_updates FK), and the running owner must exist
        # before mark_topic_dirty so its ensure inserts nothing. Inverting this
        # yields two rows and false-fails correct code.
        mem = await db.create_memory(
            Memory(memory_type="fact", topic="storage", content="We use SQLite.")
        )
        job_id = await db.enqueue_job("refresh_topic_summary", {"topic": "storage", "model": ""})
        first = await db.claim_job()
        assert first is not None and first["id"] == job_id
        stale_lock = _iso(datetime.now(timezone.utc) - timedelta(days=1))
        await db.execute("UPDATE jobs SET locked_at=? WHERE id=?", (stale_lock, job_id))
        await db.mark_topic_dirty(
            "storage", [mem.id], model="", debounce_seconds=30.0, update_threshold=3
        )
        await db.close()

        cfg = AppConfig()
        cfg.memory.database_path = db_path
        cfg.learning.enabled = True
        cfg.learning.topic_summaries = True

        rt = await build_runtime(cfg)
        try:
            rows = await rt.db.fetchall(
                "SELECT status FROM jobs WHERE job_type='refresh_topic_summary'"
            )
            assert len(rows) == 1, rows
            assert rows[0]["status"] == "pending", rows
        finally:
            await rt.upstream.close()
            await rt.embeddings.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_reinforce_with_already_attached_source_event_is_replay_idempotency():
    """Given a memory whose provenance already contains an event, When the
    recovered-job path reinforces it with that same source event, Then
    observation_count stays 1 (the guard replayed jobs rely on)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        evt = await db.add_event(
            Event(session_id="s", event_type="message.user", role="user", content="Use SQLite")
        )
        mem = await db.create_memory(
            Memory(memory_type="decision", topic="database", content="We use SQLite.")
        )
        await db.add_memory_source(mem.id, evt.id)

        updated = await db.reinforce_memory(
            mem.id, confidence=0.5, importance=0.5, source_event_ids=[evt.id]
        )
        assert updated is not None
        assert updated.observation_count == 1
        await db.close()
