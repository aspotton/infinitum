import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from infinitum.database import Database


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
