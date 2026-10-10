"""Startup topic-canonicalization backfill: cluster, rename, stamp, zero re-writes."""

import json
import sqlite3
import tempfile

from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.models import Memory, TopicSummary, utc_now
from infinitum.runtime import backfill_topic_canonicalization, build_runtime

CLUSTER_A_CANON = "web-app-deploy"
CLUSTER_A_VARIANT = "web-app-deployment"
CLUSTER_B_CANON = "runbook-rollbacks"
CLUSTER_B_VARIANT = "runbook-rollback"
EARLY = "2020-01-01T00:00:00+00:00"
LATE = "2024-01-01T00:00:00+00:00"
DATE_A = "runbook-start-2026-09-24"
DATE_B = "runbook-start-2026-10-01"
SINGLETON = "lakeside-outpost-log"
MAPPING_KEYS = {CLUSTER_A_VARIANT, CLUSTER_B_VARIANT}
CHECKPOINT_OLD = "2021-01-01T00:00:00+00:00"
CHECKPOINT_NEW = "2023-01-01T00:00:00+00:00"


def _config(path: str) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = path
    return cfg


def _dump(path: str) -> dict[str, list[str]]:
    """Whole-database snapshot (FTS shadow tables excluded) as sorted reprs."""
    conn = sqlite3.connect(path)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'"
        ).fetchall()
    ]
    out = {t: sorted(repr(r) for r in conn.execute(f'SELECT * FROM "{t}"').fetchall()) for t in tables}
    conn.close()
    return out


async def _seed_soup(db: Database) -> dict[str, list[str]]:
    """Two variant clusters + one singleton + one date-distinct pair, plus
    topics rows and consolidation checkpoints on every slug."""
    by_topic: dict[str, list[str]] = {
        CLUSTER_A_CANON: [],
        CLUSTER_A_VARIANT: [],
        CLUSTER_B_CANON: [],
        CLUSTER_B_VARIANT: [],
        DATE_A: [],
        DATE_B: [],
        SINGLETON: [],
    }
    for topic, ids in by_topic.items():
        n = 3 if topic == CLUSTER_A_CANON else 2 if topic == CLUSTER_A_VARIANT else 1
        for i in range(n):
            mem = await db.create_memory(Memory(topic=topic, content=f"{topic} note {i}"))
            ids.append(mem.id)
    # Cluster B tie: equal counts, so the earliest-created slug must win the
    # representative rule (lexicographically the variant would otherwise).
    await db.execute("UPDATE memories SET created_at=? WHERE id=?", (EARLY, by_topic[CLUSTER_B_CANON][0]))
    await db.execute("UPDATE memories SET created_at=? WHERE id=?", (LATE, by_topic[CLUSTER_B_VARIANT][0]))
    counts = {
        CLUSTER_A_CANON: 3,
        CLUSTER_A_VARIANT: 2,
        CLUSTER_B_CANON: 1,
        CLUSTER_B_VARIANT: 1,
        DATE_A: 1,
        DATE_B: 1,
        SINGLETON: 1,
    }
    now = utc_now()
    for topic, count in counts.items():
        await db.upsert_topic(
            TopicSummary(topic=topic, summary=f"{topic} summary", memory_count=count, updated_at=now)
        )
    await db.set_meta_value("consolidation:last:" + CLUSTER_A_CANON, CHECKPOINT_NEW)
    await db.set_meta_value("consolidation:last:" + CLUSTER_A_VARIANT, CHECKPOINT_OLD)
    return by_topic


async def _topics_of(db: Database) -> dict[str, str]:
    rows = await db.fetchall("SELECT id, topic FROM memories ORDER BY id")
    return {r["id"]: r["topic"] for r in rows}


async def _refresh_jobs(db: Database) -> list[dict]:
    rows = await db.fetchall(
        "SELECT payload_json FROM jobs WHERE job_type='refresh_topic_summary' AND status='pending'"
    )
    return [json.loads(r["payload_json"]) for r in rows]


async def test_backfill_remaps_clusters_and_keeps_date_distinct_pair():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        by_topic = await _seed_soup(db)

        await backfill_topic_canonicalization(db, _config(f"{tmp}/runtime.db"))

        topics = await _topics_of(db)
        for mem_id in by_topic[CLUSTER_A_CANON] + by_topic[CLUSTER_A_VARIANT]:
            assert topics[mem_id] == CLUSTER_A_CANON
        for mem_id in by_topic[CLUSTER_B_CANON] + by_topic[CLUSTER_B_VARIANT]:
            assert topics[mem_id] == CLUSTER_B_CANON  # earliest-created tie-break, not lex order
        for mem_id in by_topic[DATE_A]:
            assert topics[mem_id] == DATE_A
        for mem_id in by_topic[DATE_B]:
            assert topics[mem_id] == DATE_B  # digit veto keeps the date-distinct pair apart
        for mem_id in by_topic[SINGLETON]:
            assert topics[mem_id] == SINGLETON
        assert await db.get_meta_value("topic_canonicalization_done") == "1"
        await db.close()


async def test_backfill_summarizes_only_threshold_clusters():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(f"{tmp}/runtime.db")
        cfg.learning.model = "test-model"
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        by_topic = await _seed_soup(db)

        await backfill_topic_canonicalization(db, cfg)

        # Threshold cluster (5 actives >= topic_summary_min_memories=3):
        # dirty via the gained variant rows + exactly one summary job each.
        dirty = await db.fetchall("SELECT memory_id FROM topic_updates WHERE topic=?", (CLUSTER_A_CANON,))
        assert {r["memory_id"] for r in dirty} == set(by_topic[CLUSTER_A_VARIANT])
        jobs = await _refresh_jobs(db)
        assert [p["topic"] for p in jobs] == [CLUSTER_A_CANON]
        assert jobs[0]["model"] == "test-model"
        # Sub-threshold cluster (2 actives < 3): never marked, never scheduled.
        assert not await db.fetchall("SELECT 1 FROM topic_updates WHERE topic=?", (CLUSTER_B_CANON,))
        assert not await db.fetchall(
            "SELECT 1 FROM jobs WHERE status='pending' AND payload_json LIKE '%runbook-roll%'"
        )
        await db.close()


async def test_backfill_merges_consolidation_checkpoints_via_min():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed_soup(db)

        await backfill_topic_canonicalization(db, _config(f"{tmp}/runtime.db"))

        assert await db.get_meta_value("consolidation:last:" + CLUSTER_A_CANON) == CHECKPOINT_OLD
        assert await db.get_meta_value("consolidation:last:" + CLUSTER_A_VARIANT) is None
        await db.close()


async def test_second_build_runtime_and_crash_resume_write_zero_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        cfg = _config(path)
        db = Database(path)
        await db.connect()
        await _seed_soup(db)
        await db.close()

        runtime = await build_runtime(cfg)
        await runtime.db.close()
        dump_after_first = _dump(path)
        assert repr(("topic_canonicalization_done", "1")) in dump_after_first["meta"]

        runtime2 = await build_runtime(cfg)  # done-flag short-circuit
        await runtime2.db.close()
        assert _dump(path) == dump_after_first

        # Crash-resume: stamp lost before the process died -> re-run is a
        # no-op-mapping pass that re-stamps and changes nothing.
        db = Database(path)
        await db.connect()
        await db.execute("DELETE FROM meta WHERE key='topic_canonicalization_done'")
        await backfill_topic_canonicalization(db, cfg)
        await db.close()
        assert _dump(path) == dump_after_first


async def test_topics_endpoint_shares_no_variant_rows_after_startup():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        db = Database(path)
        await db.connect()
        await _seed_soup(db)
        await db.close()

        cfg = _config(path)
        cfg.learning.model = "test-model"
        with TestClient(create_app(cfg)) as client:
            rows = client.get("/topics").json()

        topics = {r["topic"]: r for r in rows}
        assert not (MAPPING_KEYS & set(topics))
        assert topics[CLUSTER_A_CANON]["memory_count"] == 5  # survivor count refreshed at rename
        assert topics[CLUSTER_B_CANON]["memory_count"] == 2
        assert set(topics) == {CLUSTER_A_CANON, CLUSTER_B_CANON, DATE_A, DATE_B, SINGLETON}
