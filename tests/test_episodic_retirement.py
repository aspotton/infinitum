"""Deterministic episodic retirement (plan next-quality-loop-release, task 5).

Direct-DB idioms mirror tests/test_incremental_topics.py; the live-runtime /
ASGITransport idioms mirror tests/test_learning_defer.py.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from infinitum.app import create_app
from infinitum.compiler import ContextCompiler
from infinitum.config import AppConfig
from infinitum.database import Database, _EPISODIC_RETIRE_BATCH
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner, LearningWorker
from infinitum.models import Memory, TopicSummary
from infinitum.retrieval import MemoryRetriever
from infinitum.runtime import build_runtime
from infinitum.tokenizer import TokenCounter
from infinitum.upstream import UpstreamClient

RETENTION_DAYS = 1


def _cutoff_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()


def _old_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS + 10)).isoformat()


def _retirement_config(tmp: str) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.memory.episodic_retention_days = RETENTION_DAYS
    cfg.learning.upstream_idle_grace_seconds = 0.0
    cfg.learning.poll_interval_seconds = 0.05
    return cfg


async def _seed_six(db: Database) -> dict[str, Memory]:
    """Six rows exercising every arm of the retirement predicate.

    e_never is the load-bearing COALESCE case: never injected, so
    last_accessed_at stays NULL while updated_at/created_at are stale.
    """
    old = datetime.fromisoformat(_old_iso())
    rows = {}
    rows["e_never"] = await db.create_memory(
        Memory(
            memory_type="episodic",
            topic="qa-session",
            content="Ran the legacy QA pass on staging.",
            created_at=old,
            updated_at=old,
        )
    )
    rows["f_sibling"] = await db.create_memory(
        Memory(memory_type="fact", topic="qa-session", content="Staging uses Postgres 17.")
    )
    rows["e_reinforced"] = await db.create_memory(
        Memory(
            memory_type="episodic",
            topic="qa-session",
            content="Ran a QA pass twice this quarter.",
            observation_count=2,
            created_at=old,
            updated_at=old,
        )
    )
    rows["e_young"] = await db.create_memory(
        Memory(memory_type="episodic", topic="qa-session", content="Ran QA today.")
    )
    rows["d_old"] = await db.create_memory(
        Memory(
            memory_type="decision",
            topic="qa-session",
            content="QA stays manual for now.",
            created_at=old,
            updated_at=old,
        )
    )
    rows["e_touched"] = await db.create_memory(
        Memory(
            memory_type="episodic",
            topic="qa-session",
            content="Ran QA last month too.",
            created_at=old,
            updated_at=old,
            last_accessed_at=datetime.now(timezone.utc),
        )
    )
    return rows


async def _pending_refresh_jobs(db: Database) -> int:
    row = await db.fetchone(
        "SELECT COUNT(*) AS n FROM jobs"
        " WHERE job_type='refresh_topic_summary' AND status='pending'"
    )
    return int(row["n"]) if row else 0


def _learner_with_empty_llm(db: Database, cfg: AppConfig) -> tuple[MemoryLearner, AsyncMock]:
    embeddings = EmbeddingClient(cfg.embeddings)
    upstream = UpstreamClient(cfg)
    empty_llm = AsyncMock(return_value={"choices": [{"message": {"content": ""}}]})
    upstream.learning_chat_completion = empty_llm
    retriever = MemoryRetriever(db, embeddings, cfg)
    return MemoryLearner(db, retriever, embeddings, upstream, cfg), empty_llm


@pytest.mark.asyncio
async def test_direct_call_archives_exactly_the_intended_rows_and_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        db = Database(cfg.memory.database_path)
        await db.connect()
        try:
            rows = await _seed_six(db)
            retired = await db.retire_stale_episodic(_cutoff_iso(), _EPISODIC_RETIRE_BATCH)
            assert retired == [(rows["e_never"].id, "qa-session")]
            status = {name: (await db.get_memory(m.id)).status for name, m in rows.items()}
            assert status["e_never"] == "archived"
            for name in ("f_sibling", "e_reinforced", "e_young", "d_old", "e_touched"):
                assert status[name] == "active", name
            # Re-run is a zero-row no-op (idempotent).
            assert await db.retire_stale_episodic(_cutoff_iso(), _EPISODIC_RETIRE_BATCH) == []
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_archived_rows_visible_via_status_filter():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.learning.enabled = False  # keep the worker off; retire directly below
        app = create_app(cfg)
        rt = await build_runtime(cfg)
        app.state.runtime = rt  # ASGITransport skips the lifespan
        try:
            rows = await _seed_six(rt.db)
            archived = await rt.db.retire_stale_episodic(_cutoff_iso(), _EPISODIC_RETIRE_BATCH)
            assert archived == [(rows["e_never"].id, "qa-session")]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://infinitum.test"
            ) as client:
                listed = (await client.get("/memory", params={"status": "archived"})).json()
            assert {m["id"] for m in listed} == {rows["e_never"].id}
        finally:
            await rt.upstream.close()
            await rt.embeddings.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_retirement_coalesces_into_the_pending_refresh_job_for_its_topic():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        db = Database(cfg.memory.database_path)
        await db.connect()
        try:
            rows = await _seed_six(db)
            await db.mark_topic_dirty(
                "qa-session",
                [rows["f_sibling"].id],
                model="memory-model",
                debounce_seconds=30,
                update_threshold=5,
            )
            assert await _pending_refresh_jobs(db) == 1
            worker = LearningWorker(db, MagicMock(), cfg)
            assert await worker._retire_stale_episodic() == 1
            # Still exactly one pending refresh job: the retirement enqueue
            # coalesced against the still-pending one instead of stacking.
            assert await _pending_refresh_jobs(db) == 1
            # Dirty rows = the pre-marked sibling fact + the archived episodic.
            assert await db.count_topic_updates("qa-session") == 2
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_rebuilt_summary_excludes_archived_content_via_deterministic_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.learning.topic_summary_min_memories = 1
        cfg.learning.topic_summary_fallback_memories = 8
        db = Database(cfg.memory.database_path)
        await db.connect()
        learner, empty_llm = _learner_with_empty_llm(db, cfg)
        try:
            await _seed_six(db)
            worker = LearningWorker(db, learner, cfg)
            assert await worker._retire_stale_episodic() == 1
            followup = await learner.refresh_topic_summary("qa-session", "memory-model")
            assert followup is False
            topic = await db.get_topic("qa-session")
            assert topic is not None
            assert "legacy QA pass" not in topic.summary
            assert "Staging uses Postgres 17" in topic.summary
            assert empty_llm.await_count == 1  # bootstrap prompt, empty -> fallback
        finally:
            await learner.upstream.close()
            await learner.embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_zero_active_topic_summary_is_cleared_never_stale():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.learning.topic_summary_min_memories = 1
        db = Database(cfg.memory.database_path)
        await db.connect()
        learner, empty_llm = _learner_with_empty_llm(db, cfg)
        try:
            stale = await db.create_memory(
                Memory(
                    memory_type="episodic",
                    topic="solo-log",
                    content="Attended the all-hands retro on Tuesday.",
                    created_at=datetime.fromisoformat(_old_iso()),
                    updated_at=datetime.fromisoformat(_old_iso()),
                )
            )
            await db.upsert_topic(
                TopicSummary(
                    topic="solo-log",
                    summary="Logged: attended the all-hands retro on Tuesday.",
                    memory_count=1,
                )
            )
            worker = LearningWorker(db, learner, cfg)
            assert await worker._retire_stale_episodic() == 1
            followup = await learner.refresh_topic_summary("solo-log", "memory-model")
            assert followup is False
            topic = await db.get_topic("solo-log")
            assert topic is not None
            assert topic.summary == ""  # cleared, never stale text
            assert empty_llm.await_count == 0  # zero active set gets no LLM call
            assert await db.count_topic_updates("solo-log") == 0
            assert (await db.get_memory(stale.id)).status == "archived"
        finally:
            await learner.upstream.close()
            await learner.embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_worker_tick_retires_without_consolidation_or_learning_model():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.learning.consolidation = False
        cfg.learning.model = ""
        cfg.learning.topic_summaries = False
        rt = await build_runtime(cfg)
        try:
            rows = await _seed_six(rt.db)
            rt.worker.start()
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                row = await rt.db.fetchone(
                    "SELECT status FROM memories WHERE id=?", (rows["e_never"].id,)
                )
                if row["status"] == "archived":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("worker tick never retired the stale episodic row")
            kept = await rt.db.get_memory(rows["e_young"].id)
            assert kept.status == "active"
        finally:
            await rt.worker.stop()
            await rt.upstream.close()
            await rt.embeddings.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_retention_window_zero_keeps_the_worker_from_ever_calling_retirement():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.memory.episodic_retention_days = 0.0
        cfg.learning.topic_summaries = False
        rt = await build_runtime(cfg)
        try:
            rows = await _seed_six(rt.db)
            rt.worker.start()
            await asyncio.sleep(0.3)
            row = await rt.db.fetchone(
                "SELECT status FROM memories WHERE id=?", (rows["e_never"].id,)
            )
            assert row["status"] == "active"
        finally:
            await rt.worker.stop()
            await rt.upstream.close()
            await rt.embeddings.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_cleared_topic_never_injects_an_empty_summary_stub():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _retirement_config(tmp)
        cfg.learning.topic_summary_min_memories = 1
        cfg.embeddings.enabled = False
        db = Database(cfg.memory.database_path)
        await db.connect()
        learner, _ = _learner_with_empty_llm(db, cfg)
        try:
            await db.create_memory(
                Memory(
                    memory_type="episodic",
                    topic="retro",
                    content="Attended the all-hands retro on Tuesday.",
                    created_at=datetime.fromisoformat(_old_iso()),
                    updated_at=datetime.fromisoformat(_old_iso()),
                )
            )
            await db.upsert_topic(
                TopicSummary(
                    topic="retro",
                    summary="Logged: attended the all-hands retro on Tuesday.",
                    memory_count=1,
                )
            )
            worker = LearningWorker(db, learner, cfg)
            assert await worker._retire_stale_episodic() == 1
            assert await learner.refresh_topic_summary("retro", "memory-model") is False
            compiler = ContextCompiler(
                db, MemoryRetriever(db, learner.embeddings, cfg), TokenCounter(), cfg
            )
            compiled = await compiler.compile(
                [{"role": "user", "content": "what happened in the retro?"}]
            )
            assert "topic-summary" not in compiled.text
            assert "memories=0" not in compiled.text
        finally:
            await learner.upstream.close()
            await learner.embeddings.close()
            await db.close()
