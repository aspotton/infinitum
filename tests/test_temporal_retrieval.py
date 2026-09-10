import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory
from infinitum.retrieval import MemoryRetriever

QUERY = "PostgreSQL database"


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _yesterday() -> str:
    return (datetime.now(UTC).date() - timedelta(days=1)).isoformat()


async def _open(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    retriever = MemoryRetriever(db, embeddings, cfg)
    return db, embeddings, retriever


async def _seed(db: Database, content: str, **temporal: str | None) -> str:
    memory = await db.create_memory(
        Memory(
            memory_type="fact",
            topic="database",
            content=content,
            importance=0.8,
            confidence=0.9,
            valid_from=temporal.get("valid_from"),
            valid_until=temporal.get("valid_until"),
        )
    )
    return memory.id


@pytest.mark.asyncio
async def test_default_view_parity_ignores_temporal_fields():
    """Given two DBs seeded identically except one carries an extra active row
    whose valid_until is far in the past, When searching both with the default
    (no-argument) view, Then the second DB's results equal the first DB's plus
    that extra row, with identical ids and scores within pytest.approx
    (freshness reads wall-clock now, so floats are not bit-exact). The default
    view must be byte-identical behavior to the pre-temporal retriever."""
    with tempfile.TemporaryDirectory() as plain, tempfile.TemporaryDirectory() as extra:
        db_a, emb_a, ret_a = await _open(plain)
        db_b, emb_b, ret_b = await _open(extra)
        try:
            base_ids = [
                await _seed(db_a, "The primary database runs PostgreSQL 16."),
                await _seed(db_a, "Our database backups use PostgreSQL WAL archiving."),
            ]
            for memory_id in base_ids:
                row = await db_a.get_memory(memory_id)
                await db_b.create_memory(row)
            expired_id = await _seed(
                db_b, "The database cache uses PgBouncer pooling.",
                valid_until="2020-01-01",
            )

            res_a = await ret_a.search(QUERY)
            res_b = await ret_b.search(QUERY)

            ids_a = sorted(item.memory.id for item in res_a)
            ids_b = sorted(item.memory.id for item in res_b)
            assert ids_a == sorted(base_ids)
            assert ids_b == sorted([*base_ids, expired_id])

            scores_a = {item.memory.id: item.score for item in res_a}
            scores_b = {item.memory.id: item.score for item in res_b}
            for memory_id in base_ids:
                assert scores_b[memory_id] == pytest.approx(scores_a[memory_id])

            # The same DB under "current" DOES drop the expired row: the only
            # observable difference between default and current is temporal.
            current_ids = [
                item.memory.id for item in await ret_b.search(QUERY, temporal_view="current")
            ]
            assert expired_id not in current_ids
            assert sorted(current_ids) == sorted(base_ids)
        finally:
            await emb_a.close()
            await emb_b.close()
            await db_a.close()
            await db_b.close()


@pytest.mark.asyncio
async def test_current_view_keeps_fact_valid_until_today():
    """Date-only valid_until equal to TODAY (normalized to 23:59:59 UTC) stays
    in the current view through the whole day."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            memory_id = await _seed(
                db, "The primary database runs PostgreSQL 16.", valid_until=_today()
            )
            ids = [
                item.memory.id for item in await retriever.search(QUERY, temporal_view="current")
            ]
            assert memory_id in ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_current_view_excludes_past_valid_until():
    """A date-only valid_until strictly before today is dropped by the current
    view but still returned by the default view."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            memory_id = await _seed(
                db, "The primary database runs PostgreSQL 16.", valid_until=_yesterday()
            )
            current_ids = [
                item.memory.id for item in await retriever.search(QUERY, temporal_view="current")
            ]
            default_ids = [item.memory.id for item in await retriever.search(QUERY)]
            assert memory_id not in current_ids
            assert memory_id in default_ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_as_of_view_before_inside_after_window():
    """as_of keeps rows where valid_from <= as_of and valid_until > as_of, with
    date-only bounds normalized (start 00:00:00, end 23:59:59 UTC)."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            memory_id = await _seed(
                db,
                "The migration database used PostgreSQL 14.",
                valid_from="2020-03-01",
                valid_until="2020-03-31",
            )

            async def as_of(date: str) -> list[str]:
                results = await retriever.search(QUERY, temporal_view="as_of", as_of=date)
                return [item.memory.id for item in results]

            assert memory_id not in await as_of("2020-02-15")
            assert memory_id in await as_of("2020-03-15")
            assert memory_id in await as_of("2020-03-31")
            assert memory_id not in await as_of("2020-04-15")
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_as_of_view_without_as_of_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            with pytest.raises(ValueError):
                await retriever.search(QUERY, temporal_view="as_of")
        finally:
            await embeddings.close()
            await db.close()


def _seed_for_route(db_path: str) -> None:
    import asyncio

    async def run() -> None:
        db = Database(db_path)
        await db.connect()
        try:
            await _seed(
                db, "The legacy database ran PostgreSQL 12.",
                valid_until="2020-01-01",
            )
            await _seed(db, "The current database runs PostgreSQL 16.")
        finally:
            await db.close()

    asyncio.run(run())


def test_route_default_current_hides_expired_row():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/runtime.db"
        _seed_for_route(db_path)
        cfg = AppConfig()
        cfg.memory.database_path = db_path
        cfg.learning.enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            results = client.post("/memory/search", json={"query": QUERY})
            assert results.status_code == 200
            contents = [item["memory"]["content"] for item in results.json()]
            assert any("PostgreSQL 16" in c for c in contents)
            assert not any("PostgreSQL 12" in c for c in contents)


def test_route_invalid_as_of_returns_400():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            response = client.post(
                "/memory/search",
                json={"query": QUERY, "temporal_view": "as_of", "as_of": "last tuesday"},
            )
            assert response.status_code == 400


def test_route_as_of_view_without_as_of_returns_400():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            response = client.post(
                "/memory/search", json={"query": QUERY, "temporal_view": "as_of"}
            )
            assert response.status_code == 400
