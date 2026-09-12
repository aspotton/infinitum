"""Route-level cursor pagination tests for GET /memory (issue #11 regression).

Seeding recipe (per the plan): create the schema via Database(path)+connect()+
close(), then bulk-insert raw rows with sqlite3 (Database has no executemany).
Seed BEFORE app construction; run requests inside ``with TestClient(...)`` so
lifespan sets app.state.runtime.
"""
import base64
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database

_INSERT = (
    "INSERT INTO memories(id, memory_type, topic, content, status, importance,"
    " confidence, observation_count, created_at, updated_at, metadata_json)"
    " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
)


def _ts(i: int) -> str:
    """Unique minute-spaced ISO timestamp for seed row i."""
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i)).isoformat()


async def _seed(path: str, count: int, status_at=lambda i: "active") -> set[str]:
    """Create the schema, then bulk-insert `count` memories; return the id set."""
    seed = Database(path)
    await seed.connect()
    await seed.close()
    rows = [
        (
            f"mem_{i}",
            "fact",
            "bulk",
            f"bulk memory {i}",
            status_at(i),
            0.5,
            0.7,
            1,
            _ts(i),
            _ts(i),
            "{}",
        )
        for i in range(count)
    ]
    conn = sqlite3.connect(path)
    conn.executemany(_INSERT, rows)
    conn.commit()
    conn.close()
    return {f"mem_{i}" for i in range(count)}


def _cfg(path: str) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.database_path = path
    cfg.embeddings.enabled = False
    return cfg


def _walk(client, limit: int, extra: dict | None = None) -> tuple[list[int], list[dict]]:
    """Follow X-Next-Cursor until absent; return (page sizes, collected rows)."""
    sizes: list[int] = []
    rows_out: list[dict] = []
    params = {"limit": limit, **(extra or {})}
    for _ in range(10):  # hard stop so a broken cursor can't hang the walk
        resp = client.get("/memory", params=params)
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        sizes.append(len(rows))
        rows_out.extend(rows)
        cursor = resp.headers.get("X-Next-Cursor")
        if not cursor:
            return sizes, rows_out
        params = {**params, "cursor": cursor}
    raise AssertionError("cursor walk did not terminate within 10 pages")


async def test_walk_reaches_every_row_exactly_once():
    """1,200-row walk with limit=500: 3 pages, every id exactly once (issue #11)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/walk.db"
        seeded = await _seed(path, 1200)
        with TestClient(create_app(_cfg(path))) as client:
            sizes, rows = _walk(client, 500)
            ids = [row["id"] for row in rows]
            dup_count = len(ids) - len(set(ids))
            print(f"page sizes {sizes}, dup-count {dup_count}")
            assert sizes == [500, 500, 200]
            assert len(set(ids)) == 1200, "unique ids must equal the 1200 seeded, not row-count"
            assert set(ids) == seeded
            assert dup_count == 0


async def test_offset_is_rejected_loudly():
    """?offset (even 0) must 400 pointing at cursor pagination, never serve page 1."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/offset.db"
        await _seed(path, 1200)
        with TestClient(create_app(_cfg(path))) as client:
            for offset in (1000, 0):
                resp = client.get("/memory", params={"offset": offset})
                assert resp.status_code == 400, resp.text
                assert "cursor" in resp.json()["detail"]


async def test_bad_cursor_is_rejected():
    """Malformed cursors (bad base64, wrong JSON shape) must 400."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/badcursor.db"
        await _seed(path, 1200)
        with TestClient(create_app(_cfg(path))) as client:
            resp = client.get("/memory", params={"cursor": "!!!"})
            assert resp.status_code == 400, resp.text
            bogus = base64.urlsafe_b64encode(b'{"a":1}').decode()
            resp = client.get("/memory", params={"cursor": bogus})
            assert resp.status_code == 400, resp.text


async def test_last_page_has_no_next_cursor():
    """Exactly 2xlimit rows: the second page carries no header and is full-size."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/lastpage.db"
        seeded = await _seed(path, 1000)
        with TestClient(create_app(_cfg(path))) as client:
            first = client.get("/memory", params={"limit": 500})
            assert len(first.json()) == 500
            cursor = first.headers["X-Next-Cursor"]
            second = client.get("/memory", params={"limit": 500, "cursor": cursor})
            assert second.status_code == 200
            rows = second.json()
            assert len(rows) == 500
            assert "X-Next-Cursor" not in second.headers
            first_ids = {row["id"] for row in first.json()}
            assert {row["id"] for row in rows} == seeded - first_ids


async def test_status_filter_with_cursor():
    """Walking ?status=archived returns only archived rows across disjoint pages."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/status.db"
        await _seed(path, 1200, status_at=lambda i: "archived" if i % 2 else "active")
        with TestClient(create_app(_cfg(path))) as client:
            sizes, rows = _walk(client, 250, {"status": "archived"})
            ids = [row["id"] for row in rows]
            print(f"archived walk page sizes {sizes}")
            assert sizes == [250, 250, 100]
            assert all(row["status"] == "archived" for row in rows)
            assert len(set(ids)) == len(ids) == 600
