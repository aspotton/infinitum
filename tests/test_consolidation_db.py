import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from infinitum.database import Database, _CONSOLIDATION_META_PREFIX
from infinitum.models import Memory


def _iso(dt: datetime) -> str:
    return dt.isoformat()


async def _seed(db: Database, topic: str, content: str = "fact") -> Memory:
    memory = Memory(topic=topic, content=content)
    return await db.create_memory(memory)


@pytest.mark.asyncio
async def test_sweep_enqueues_stale_topic_and_claim_returns_the_job():
    """Given a topic whose checkpoint is older than the interval, When the sweep
    runs, Then exactly one consolidate_topic job is queued and claim_job picks
    it up FIFO."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed(db, "old-topic")
        stale = _iso(datetime.now(timezone.utc) - timedelta(days=8))
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + "old-topic", stale)

        queued = await db.ensure_consolidation_jobs(604800.0, 20, "test-model")
        assert queued == 1

        row = await db.fetchone(
            "SELECT * FROM jobs WHERE job_type='consolidate_topic'"
        )
        assert json.loads(row["payload_json"]) == {
            "topic": "old-topic",
            "model": "test-model",
        }
        assert row["status"] == "pending"

        claimed = await db.claim_job()
        assert claimed is not None and claimed["id"] == row["id"]
        assert claimed["payload"]["topic"] == "old-topic"


@pytest.mark.asyncio
async def test_second_immediate_sweep_is_deduped_against_pending_job():
    """Given a sweep already queued a job for the topic, When the sweep runs
    again immediately, Then zero new jobs (pending row blocks re-enqueue)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed(db, "t")

        assert await db.ensure_consolidation_jobs(0.0, 20, "m") == 1
        assert await db.ensure_consolidation_jobs(0.0, 20, "m") == 0
        n = await db.fetchone("SELECT COUNT(*) AS n FROM jobs")
        assert int(n["n"]) == 1


@pytest.mark.asyncio
async def test_fresh_checkpoint_topic_is_skipped_by_the_sweep():
    """Given a topic checkpoint newer than now-interval, When the sweep runs,
    Then that topic is not due and zero jobs are enqueued."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed(db, "fresh")
        fresh = _iso(datetime.now(timezone.utc))
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + "fresh", fresh)

        assert await db.ensure_consolidation_jobs(604800.0, 20, "m") == 0


@pytest.mark.asyncio
async def test_batch_cap_one_enqueues_exactly_one_of_twenty_due_topics():
    """Given 20 topics all due (no checkpoint, no job), When the sweep runs with
    batch_cap=1, Then exactly one job exists — the anti-starvation guard."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        for i in range(20):
            await _seed(db, f"topic-{i}")

        assert await db.ensure_consolidation_jobs(604800.0, 1, "m") == 1
        n = await db.fetchone("SELECT COUNT(*) AS n FROM jobs")
        assert int(n["n"]) == 1


@pytest.mark.asyncio
async def test_oldest_first_listing_and_since_count():
    """Given three active rows with staggered updated_at, Then the oldest-first
    page returns them ASC by (updated_at, id) and the since-count sees only
    rows created or updated after the checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        base = datetime.now(timezone.utc) - timedelta(days=5)
        for i in range(3):
            memory = Memory(topic="t", content=f"c{i}")
            await db.create_memory(memory)
            stamp = _iso(base + timedelta(hours=i))
            await db.execute(
                "UPDATE memories SET updated_at=?, created_at=? WHERE id=?",
                (stamp, stamp, memory.id),
            )

        page = await db.list_active_topic_memories_oldest("t", 2)
        assert [m.content for m in page] == ["c0", "c1"]
        assert all(m.updated_at < base + timedelta(hours=2) for m in page)

        since = _iso(base + timedelta(hours=1, minutes=30))
        assert await db.count_active_topic_memories_updated_since("t", since) == 1


@pytest.mark.asyncio
async def test_sweep_prefers_higher_churn_topic_when_batch_cap_is_one():
    """Given two due topics, one with 3 active memories and one with 1, When the
    sweep runs with batch_cap=1, Then the single enqueued job is for the
    higher-churn topic, not the alphabetically-first one."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        for _ in range(3):
            await _seed(db, "busy")
        await _seed(db, "quiet")

        assert await db.ensure_consolidation_jobs(604800.0, 1, "m") == 1
        n = await db.fetchone("SELECT COUNT(*) AS n FROM jobs")
        assert int(n["n"]) == 1
        row = await db.fetchone(
            "SELECT * FROM jobs WHERE job_type='consolidate_topic'"
        )
        assert json.loads(row["payload_json"])["topic"] == "busy"


@pytest.mark.asyncio
async def test_sweep_skips_topics_below_churn_floor():
    """Given an aged topic with 1 changed memory and a checkpoint-less fresh
    topic with 1 memory, When the sweep runs with min_changed=8, Then zero jobs
    are enqueued and the jobs table holds no consolidate_topic row."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed(db, "low")
        stale = _iso(datetime.now(timezone.utc) - timedelta(days=8))
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + "low", stale)
        await _seed(db, "newborn")  # never consolidated: churn = lifetime count

        assert await db.ensure_consolidation_jobs(604800.0, 20, "m", min_changed=8) == 0
        rows = await db.fetchall(
            "SELECT id FROM jobs WHERE job_type='consolidate_topic'"
        )
        assert rows == []
        await db.close()


@pytest.mark.asyncio
async def test_sweep_enqueues_topic_at_churn_floor():
    """Given an aged topic with exactly 9 changed memories and floor 8, When
    the sweep runs, Then exactly one consolidate_topic job is enqueued."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        for i in range(9):
            await _seed(db, "busy", content=f"c{i}")
        stale = _iso(datetime.now(timezone.utc) - timedelta(days=8))
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + "busy", stale)

        assert await db.ensure_consolidation_jobs(604800.0, 20, "m", min_changed=8) == 1
        n = await db.fetchone(
            "SELECT COUNT(*) AS n FROM jobs WHERE job_type='consolidate_topic'"
        )
        assert int(n["n"]) == 1
        await db.close()


@pytest.mark.asyncio
async def test_enqueue_churn_gate_matches_runtime_count_formula():
    """Given a checkpoint-less topic with N active memories, Then the enqueue
    churn gate flips at exactly N — the same row-for-row formula as
    count_active_topic_memories_updated_since(topic, epoch) — and the epoch
    constants agree across database.py and consolidation.py."""
    import inspect

    from infinitum import consolidation
    from infinitum.database import _CONSOLIDATION_EPOCH

    assert _CONSOLIDATION_EPOCH == consolidation._EPOCH_ISO

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        for i in range(2):
            await _seed(db, "t", content=f"c{i}")

        # The run-time gate must see exactly 2 changed rows since the epoch.
        assert await db.count_active_topic_memories_updated_since(
            "t", _CONSOLIDATION_EPOCH
        ) == 2
        # Enqueue-time gate must agree: floor above the count enqueues nothing...
        assert await db.ensure_consolidation_jobs(604800.0, 20, "m", min_changed=3) == 0
        # ...and floor at the count enqueues the topic.
        assert await db.ensure_consolidation_jobs(604800.0, 20, "m", min_changed=2) == 1

        # The sweep SQL must compare created_at/updated_at per-row against the
        # checkpoint (COALESCE'd to the epoch), same as the count formula.
        sql = inspect.getsource(Database._ensure_consolidation_jobs_sync)
        assert "m.created_at > COALESCE(mt.value, ?)" in sql
        assert "m.updated_at > COALESCE(mt.value, ?)" in sql
        assert "HAVING churn >= ?" in sql
        await db.close()
