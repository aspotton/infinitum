"""Route-level cursor pagination tests for GET /events and GET /topics (issue #11).

Events use the raw-sqlite3-after-schema bulk-seed recipe (Database has no
executemany); topics use the real `db.upsert_topic` API so the TopicSummary
model path is exercised. Seed BEFORE app construction; run requests inside
``with TestClient(...)`` so lifespan sets app.state.runtime.
"""
import base64
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.models import TopicSummary

_INSERT = "INSERT INTO events(id, session_id, event_type, created_at) VALUES (?,?,?,?)"


def _ts(i: int) -> str:
    """Unique minute-spaced ISO timestamp for seed row i."""
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i)).isoformat()


async def _seed_events(path: str, count: int) -> set[str]:
    """Create the schema, then bulk-insert `count` events; return the id set."""
    seed = Database(path)
    await seed.connect()
    await seed.close()
    rows = [
        (f"evt_{i}", "sess_a" if i % 2 == 0 else "sess_b", "request.received", _ts(i))
        for i in range(count)
    ]
    conn = sqlite3.connect(path)
    conn.executemany(_INSERT, rows)
    conn.commit()
    conn.close()
    return {f"evt_{i}" for i in range(count)}


async def _seed_topics(path: str, count: int, updated_at: datetime) -> list[str]:
    """Seed `count` topics sharing one updated_at via the real upsert_topic API."""
    db = Database(path)
    await db.connect()
    for i in range(count):
        await db.upsert_topic(
            TopicSummary(topic=f"t{i}", summary="s", memory_count=1, updated_at=updated_at)
        )
    await db.close()
    return [f"t{i}" for i in range(count)]


def _cfg(path: str) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = path
    cfg.embeddings.enabled = False
    return cfg


def _walk(
    client, route: str, limit: int, extra: dict | None = None
) -> tuple[list[int], list[dict]]:
    """Follow X-Next-Cursor until absent; return (page sizes, collected rows)."""
    sizes: list[int] = []
    rows_out: list[dict] = []
    params = {"limit": limit, **(extra or {})}
    for _ in range(10):  # hard stop so a broken cursor can't hang the walk
        resp = client.get(route, params=params)
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        sizes.append(len(rows))
        rows_out.extend(rows)
        cursor = resp.headers.get("X-Next-Cursor")
        if not cursor:
            return sizes, rows_out
        params = {**params, "cursor": cursor}
    raise AssertionError("cursor walk did not terminate within 10 pages")


async def test_events_walk_reaches_every_event_exactly_once():
    """700 events walked with limit=300: pages [300,300,100], 700 unique ids."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/events_walk.db"
        seeded = await _seed_events(path, 700)
        with TestClient(create_app(_cfg(path))) as client:
            sizes, rows = _walk(client, "/events", 300)
            ids = [row["id"] for row in rows]
            dup_count = len(ids) - len(set(ids))
            print(f"events walk page sizes {sizes}, dup-count {dup_count}")
            assert sizes == [300, 300, 100]
            assert len(set(ids)) == 700, "unique ids must equal the 700 seeded, not row-count"
            assert set(ids) == seeded
            assert dup_count == 0


async def test_events_filter_plus_cursor():
    """Walking ?session_id=sess_b returns only that session, disjoint pages."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/events_filter.db"
        await _seed_events(path, 700)
        with TestClient(create_app(_cfg(path))) as client:
            sizes, rows = _walk(client, "/events", 200, {"session_id": "sess_b"})
            ids = [row["id"] for row in rows]
            print(f"sess_b walk page sizes {sizes}")
            assert sizes == [200, 150]
            assert all(row["session_id"] == "sess_b" for row in rows)
            assert len(set(ids)) == len(ids) == 350


async def test_topics_walk():
    """250 identical-timestamp topics walk stably; tie order is topic DESC."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/topics_walk.db"
        shared = datetime(2026, 1, 1, tzinfo=UTC)
        seeded = await _seed_topics(path, 250, shared)
        with TestClient(create_app(_cfg(path))) as client:
            sizes, rows = _walk(client, "/topics", 100)
            topics = [row["topic"] for row in rows]
            print(f"topics walk page sizes {sizes}")
            assert sizes == [100, 100, 50]
            assert len(set(topics)) == 250
            assert set(topics) == set(seeded)
            # The ORDER BY tiebreaker is topic DESC — assert that direction, not ascending.
            assert topics == sorted(seeded, reverse=True)


async def test_offset_rejected_on_both():
    """?offset=0 (even 0) must 400 on /events and /topics, never serve page 1."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/offset.db"
        await _seed_events(path, 4)
        with TestClient(create_app(_cfg(path))) as client:
            for route in ("/events", "/topics"):
                resp = client.get(route, params={"offset": 0})
                assert resp.status_code == 400, f"{route}: {resp.text}"
                assert "cursor" in resp.json()["detail"]


async def test_bad_cursor_rejected_on_both():
    """Malformed cursors (bad base64, wrong JSON shape) must 400 on both routes."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/badcursor.db"
        await _seed_events(path, 4)
        bogus = base64.urlsafe_b64encode(b'{"a":1}').decode()
        with TestClient(create_app(_cfg(path))) as client:
            for route in ("/events", "/topics"):
                for cursor in ("!!!", bogus):
                    resp = client.get(route, params={"cursor": cursor})
                    assert resp.status_code == 400, f"{route} cursor={cursor}: {resp.text}"
