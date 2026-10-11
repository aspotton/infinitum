import sqlite3
import tempfile
from datetime import datetime, timezone

from infinitum.database import Database, _CONSOLIDATION_META_PREFIX
from infinitum.models import Memory, TopicSummary

VARIANT = "web-app-deploy-variant"
CANONICAL = "web-app-deploy"
OLD = "2020-01-01T00:00:00+00:00"


async def _memory(db: Database, topic: str, content: str = "shared decision") -> Memory:
    return await db.create_memory(Memory(topic=topic, content=content))


async def _seed_pair(db: Database) -> list[Memory]:
    """Two memories on the variant slug, one already on the canonical slug."""
    a = await _memory(db, VARIANT)
    b = await _memory(db, VARIANT)
    c = await _memory(db, CANONICAL)
    return [a, b, c]


def _dump(path: str) -> dict[str, list[object]]:
    """Row-level state snapshot used for zero-write idempotence comparisons."""
    conn = sqlite3.connect(path)
    out = {
        "memories": conn.execute(
            "SELECT id, topic, status, updated_at FROM memories ORDER BY id"
        ).fetchall(),
        "jobs": conn.execute(
            "SELECT id, job_type, status, payload_json, run_after, attempts"
            " FROM jobs ORDER BY id"
        ).fetchall(),
        "meta": conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall(),
        "topics": conn.execute(
            "SELECT topic, summary, memory_count, updated_at FROM topics ORDER BY topic"
        ).fetchall(),
        "topic_updates": conn.execute(
            "SELECT topic, memory_id, created_at FROM topic_updates ORDER BY topic, memory_id"
        ).fetchall(),
    }
    conn.close()
    return out


async def test_rename_remaps_memories_and_bumps_updated_at():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        a, b, c = await _seed_pair(db)
        await db.execute("UPDATE memories SET updated_at=? WHERE topic=?", (OLD, VARIANT))

        moved = await db.rename_topics({VARIANT: CANONICAL})

        assert moved == 2
        rows = await db.fetchall(
            "SELECT id, topic, updated_at FROM memories ORDER BY id"
        )
        assert all(r["topic"] == CANONICAL for r in rows)
        for r in rows:
            if r["id"] in (a.id, b.id):
                assert r["updated_at"] > OLD  # the intentional reconciliation bump
            else:
                assert r["updated_at"] != OLD  # canonical rows were not touched
        counts = await db.list_topic_counts()
        assert counts[0][0] == CANONICAL and counts[0][1] == 3
        assert all(topic != VARIANT for topic, _, _ in counts)
        await db.close()


async def test_list_topic_counts_counts_active_only_and_orders_deterministically():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _memory(db, "beta")
        await _memory(db, "alpha")
        await _memory(db, "alpha")
        archived = await _memory(db, "archived-topic")
        await db.execute(
            "UPDATE memories SET status='archived' WHERE id=?", (archived.id,)
        )

        counts = await db.list_topic_counts()

        assert [t for t, _, _ in counts] == ["alpha", "beta"]
        assert dict((t, n) for t, n, _ in counts) == {"alpha": 2, "beta": 1}
        assert all(isinstance(earliest, str) for _, _, earliest in counts)
        await db.close()


async def test_topic_updates_pk_collision_merges_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        m = await _memory(db, VARIANT)
        # One memory already referenced dirty under BOTH sides.
        await db.execute(
            "INSERT INTO topic_updates(topic, memory_id, created_at) VALUES(?, ?, ?)",
            (CANONICAL, m.id, OLD),
        )
        await db.execute(
            "INSERT INTO topic_updates(topic, memory_id, created_at) VALUES(?, ?, ?)",
            (VARIANT, m.id, OLD),
        )

        await db.rename_topics({VARIANT: CANONICAL})

        rows = await db.fetchall("SELECT topic, memory_id FROM topic_updates")
        assert [(r["topic"], r["memory_id"]) for r in rows] == [(CANONICAL, m.id)]
        await db.close()


async def test_job_payload_topic_rewritten_but_content_bytes_survive():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        note = f"notes mentioning {VARIANT} verbatim in prose"
        payloads = {
            "pending": ("refresh_topic_summary", "pending",
                        f'{{"topic":"{VARIANT}","model":"m","note":"{note}"}}',
                        f'{{"topic":"{CANONICAL}","model":"m","note":"{note}"}}'),
            "running": ("consolidate_topic", "running",
                        f'{{"topic":"{VARIANT}","model":"m","continuation":true}}',
                        f'{{"topic":"{CANONICAL}","model":"m","continuation":true}}'),
            "failed": ("consolidate_topic", "failed",
                       f'{{"topic":"{VARIANT}","model":"m"}}', None),
            "done": ("learn_interaction", "done",
                     f'{{"topic":"{VARIANT}"}}', None),
        }
        for tag, (job_type, status, payload, _) in payloads.items():
            await db.execute(
                "INSERT INTO jobs(id, job_type, status, payload_json, created_at,"
                " run_after, attempts) VALUES(?, ?, ?, ?, ?, ?, 0)",
                (f"job_{tag}", job_type, status, payload, OLD, OLD),
            )

        await db.rename_topics({VARIANT: CANONICAL})

        rows = {
            r["id"]: r
            for r in await db.fetchall("SELECT id, payload_json FROM jobs")
        }
        for tag, (_, _, payload, expected) in payloads.items():
            if expected is None:
                assert rows[f"job_{tag}"]["payload_json"] == payload  # byte-identical
            else:
                assert json_equivalent(rows[f"job_{tag}"]["payload_json"], expected)
        got = await db.fetchone(
            "SELECT json_extract(payload_json, '$.note') AS note FROM jobs WHERE id='job_pending'"
        )
        assert got is not None and got["note"] == note  # slug-mentioning content untouched
        await db.close()


def json_equivalent(a: str, b: str) -> bool:
    import json

    return json.loads(a) == json.loads(b)


async def test_consolidation_checkpoints_merge_with_min():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _memory(db, VARIANT)
        await _memory(db, "solo-variant")
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + VARIANT, OLD)
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + CANONICAL, "2025-01-01T00:00:00+00:00")
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + "solo-variant", "2024-06-01T00:00:00+00:00")

        await db.rename_topics({VARIANT: CANONICAL, "solo-variant": CANONICAL})

        assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + CANONICAL) == OLD
        assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + VARIANT) is None
        assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + "solo-variant") is None
        await db.close()


async def test_topics_table_scoped_to_mapping_survivors_keep_summaries():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed_pair(db)
        unrelated = await _memory(db, "unrelated-archived")
        await db.execute(
            "UPDATE memories SET status='archived' WHERE id=?", (unrelated.id,)
        )
        await db.upsert_topic(TopicSummary(topic=VARIANT, summary="V", memory_count=9,
                                           updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
        await db.upsert_topic(TopicSummary(topic=CANONICAL, summary="C", memory_count=0,
                                           updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
        await db.upsert_topic(TopicSummary(topic="unrelated-archived", summary="U", memory_count=0,
                                           updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))

        await db.rename_topics({VARIANT: CANONICAL})

        assert await db.get_topic(VARIANT) is None  # merged-side row deleted
        survivor = await db.get_topic(CANONICAL)
        assert survivor is not None
        assert survivor.summary == "C"  # survivor keeps its own summary
        assert survivor.memory_count == 3  # refreshed from the live counts
        untouched = await db.get_topic("unrelated-archived")
        assert untouched is not None
        assert (untouched.summary, untouched.memory_count) == ("U", 0)
        assert untouched.updated_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
        await db.close()


async def test_fts_rebuilt_once_after_rename():
    variant_slug = "variantopicqz"
    canonical_slug = "canonicaltopicqz"
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        if not db.fts_enabled:  # env without FTS5: the guarded skip is the contract
            await _memory(db, variant_slug)
            await db.rename_topics({variant_slug: canonical_slug})  # must not crash
            assert await db.fts_memory_ids(canonical_slug) == []
            await db.close()
            return
        m = await _memory(db, variant_slug)
        assert m.id in await db.fts_memory_ids(variant_slug)

        await db.rename_topics({variant_slug: canonical_slug})

        assert await db.fts_memory_ids(variant_slug) == []  # stale variant token gone
        assert m.id in await db.fts_memory_ids(canonical_slug)  # rebuilt under canonical
        await db.close()


async def test_rename_is_idempotent_zero_writes_on_reapply():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        db = Database(path)
        await db.connect()
        m = await _seed_pair(db)
        await db.execute(
            "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after,"
            " attempts) VALUES('job_p', 'refresh_topic_summary', 'pending', ?, ?, ?, 0)",
            (f'{{"topic":"{VARIANT}","model":"m"}}', OLD, OLD),
        )
        await db.set_meta_value(_CONSOLIDATION_META_PREFIX + VARIANT, OLD)
        await db.upsert_topic(TopicSummary(topic=VARIANT, summary="V", memory_count=2,
                                           updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))

        before = _dump(path)
        await db.rename_topics({VARIANT: CANONICAL})
        after = _dump(path)
        assert after["memories"] != before["memories"]  # first pass did work

        await db.rename_topics({})
        assert _dump(path) == after
        await db.rename_topics({VARIANT: CANONICAL})  # re-apply: variant has no rows left
        assert _dump(path) == after
        await db.rename_topics({CANONICAL: CANONICAL})  # self-pair is a no-op
        assert _dump(path) == after
        assert m[0].id  # memories still present
        await db.close()


async def test_dirty_marking_gated_by_min_memories_and_model():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        moved = [await _memory(db, VARIANT), await _memory(db, VARIANT)]
        await _memory(db, CANONICAL)  # already canonical: unchanged, never marked dirty
        small_variant = "solo-topic"
        small = await _memory(db, small_variant)

        await db.rename_topics(
            {VARIANT: CANONICAL, small_variant: "small-canonical"},
            model="test-model",
            debounce_seconds=0.0,
            min_memories=3,
        )

        rows = await db.fetchall(
            "SELECT topic, memory_id FROM topic_updates ORDER BY topic, memory_id"
        )
        assert sorted((r["topic"], r["memory_id"]) for r in rows) == sorted(
            (CANONICAL, m.id) for m in moved
        )
        assert all(r["topic"] != "small-canonical" for r in rows)  # sub-threshold: not marked
        jobs = await db.fetchall(
            "SELECT json_extract(payload_json, '$.topic') AS topic FROM jobs"
            " WHERE job_type='refresh_topic_summary'"
        )
        assert [r["topic"] for r in jobs] == [CANONICAL]  # one job, only for the survivor
        assert small.id  # small-canonical memory exists under its canonical topic
        topic_row = await db.fetchone("SELECT topic FROM memories WHERE id=?", (small.id,))
        assert topic_row is not None and topic_row["topic"] == "small-canonical"
        await db.close()
