"""Tests for the read-only GET /status consolidated diagnostics endpoint.

Pattern copied from tests/test_learning_defer.py (ASGITransport + manual
build_runtime; the worker never runs, so seeded state stays frozen). Every
number asserted here is ALSO independently recomputed from the raw SQLite
file in ``_raw_expectations`` — the endpoint must agree with plain SQL, not
with its own re-query.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.models import Event, Memory
from infinitum.runtime import build_runtime


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _app(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = False
    app = create_app(cfg)
    rt = await build_runtime(cfg)
    app.state.runtime = rt
    return app, rt


async def _seed(rt) -> None:
    db = rt.db
    # Memories: recent fact, recent episodic, episodic created 8 days ago,
    # plus a superseded decision that must not count anywhere.
    await db.create_memory(
        Memory(memory_type="fact", topic="db-choice", content="We use PostgreSQL.")
    )
    await db.create_memory(
        Memory(memory_type="episodic", topic="meetings", content="Discussed rollout Monday.")
    )
    old = _now() - timedelta(days=8)
    await db.create_memory(
        Memory(
            memory_type="episodic",
            topic="meetings-old",
            content="Kickoff two weeks of churn.",
            created_at=old,
            updated_at=old,
        )
    )
    await db.create_memory(
        Memory(
            memory_type="decision",
            topic="rollout",
            content="Ship Friday.",
            status="superseded",
        )
    )
    # Events: two consolidation.pass inside the 7-day window (one merge of 3,
    # one no-op), one consolidation.pass 8 days ago (must be excluded), plus a
    # memory.tool_call whose metadata must not leak into consolidation stats.
    await db.add_event(
        Event(
            session_id="s",
            event_type="consolidation.pass",
            metadata={"topic": "meetings", "no_op": False, "merged": 3},
        )
    )
    await db.add_event(
        Event(
            session_id="s",
            event_type="consolidation.pass",
            metadata={"topic": "db-choice", "no_op": True, "merged": 0},
        )
    )
    await db.add_event(
        Event(
            session_id="s",
            event_type="consolidation.pass",
            metadata={"topic": "meetings-old", "no_op": False, "merged": 9},
            created_at=old,
        )
    )
    await db.add_event(
        Event(
            session_id="s",
            event_type="memory.tool_call",
            metadata={"no_op": False, "merged": 100},
        )
    )
    # Jobs: 2 pending, 1 running (claimed), 1 failed, 1 done. After the claim
    # the remaining mix is pending=1, running=1, failed=1, done=1.
    await db.enqueue_job("learn_interaction", {})
    await db.enqueue_job("refresh_topic_summary", {})
    await db.claim_job()
    failed = await db.enqueue_job("consolidate_topic", {})
    await db.fail_job(failed, "boom", retry=False)
    done = await db.enqueue_job("learn_interaction", {})
    await db.finish_job(done)
    await db.set_meta_value("episodic_skips_total", "4")


def _raw_expectations(path: str) -> dict:
    """Recompute every /status number straight from the SQLite file."""
    cutoff = (_now() - timedelta(days=7)).isoformat()
    conn = sqlite3.connect(path)
    try:
        active_total, active_topics = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT topic) FROM memories WHERE status='active'"
        ).fetchone()
        by_type = {
            t: n for t, n in conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories WHERE status='active'"
                " GROUP BY memory_type"
            )
        }
        episodic = by_type.get("episodic", 0)
        (created_7d,) = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active' AND created_at>=?",
            (cutoff,),
        ).fetchone()
        jobs = {
            s: n
            for s, n in conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        }
        passes, no_ops, merges = conn.execute(
            "SELECT COUNT(*),"
            " COALESCE(SUM(json_extract(metadata_json,'$.no_op')), 0),"
            " COALESCE(SUM(json_extract(metadata_json,'$.merged')), 0)"
            " FROM events"
            " WHERE event_type='consolidation.pass' AND created_at>=?",
            (cutoff,),
        ).fetchone()
        (episodic_total,) = conn.execute(
            "SELECT value FROM meta WHERE key='episodic_skips_total'"
        ).fetchone() or (0,)
    finally:
        conn.close()
    return {
        "store": {
            "active_total": active_total,
            "active_by_type": by_type,
            "active_topics": active_topics,
            "episodic_share": round(episodic / active_total, 2) if active_total else 0.0,
            "created_7d": created_7d,
        },
        "jobs": {
            "pending": jobs.get("pending", 0),
            "running": jobs.get("running", 0),
            "failed": jobs.get("failed", 0),
        },
        "consolidation_7d": {
            "passes": passes,
            "no_ops": no_ops,
            "merges": merges,
        },
        "skips": {"episodic_total": int(episodic_total)},
    }


HEALTH_KEYS = {
    "status",
    "version",
    "memory_enabled",
    "learning_enabled",
    "embeddings_enabled",
    "fts_enabled",
    "active_requests",
}


@pytest.mark.asyncio
async def test_status_matches_raw_sqlite_after_seeding():
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _app(tmp)
        try:
            await _seed(rt)
            expected = _raw_expectations(f"{tmp}/runtime.db")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://infinitum.test"
            ) as client:
                resp = await client.get("/status")
                assert resp.status_code == 200
                payload = resp.json()
                # /health fields ride along unchanged.
                health = (await client.get("/health")).json()
            # Cross-check: endpoint output == independent raw-SQLITE numbers.
            assert payload["store"] == expected["store"]
            assert payload["jobs"] == expected["jobs"]
            assert payload["consolidation_7d"] == expected["consolidation_7d"]
            assert payload["skips"] == expected["skips"]
            # Spell out the concrete seed expectations too (misleading-output
            # guard: expected dict above could itself drift silently).
            assert payload["store"] == {
                "active_total": 3,
                "active_by_type": {"episodic": 2, "fact": 1},
                "active_topics": 3,
                "episodic_share": 0.67,
                "created_7d": 2,
            }
            assert payload["jobs"] == {"pending": 1, "running": 1, "failed": 1}
            # The 8-day-old pass (merged=9) and the memory.tool_call are excluded.
            assert payload["consolidation_7d"] == {"passes": 2, "no_ops": 1, "merges": 3}
            assert payload["skips"] == {"episodic_total": 4}
            assert HEALTH_KEYS <= set(payload)
            assert all(payload[k] == health[k] for k in HEALTH_KEYS)
        finally:
            await rt.upstream.close()
            await rt.db.close()


@pytest.mark.asyncio
async def test_status_empty_store_returns_all_zeros():
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _app(tmp)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://infinitum.test"
            ) as client:
                resp = await client.get("/status")
                assert resp.status_code == 200
                payload = resp.json()
            assert payload["store"] == {
                "active_total": 0,
                "active_by_type": {},
                "active_topics": 0,
                "episodic_share": 0.0,  # no ZeroDivision on empty store
                "created_7d": 0,
            }
            assert payload["jobs"] == {"pending": 0, "running": 0, "failed": 0}
            assert payload["consolidation_7d"] == {"passes": 0, "no_ops": 0, "merges": 0}
            assert payload["skips"] == {"episodic_total": 0}  # absent meta key reads as 0
        finally:
            await rt.upstream.close()
            await rt.db.close()
