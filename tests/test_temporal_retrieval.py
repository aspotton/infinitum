import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import infinitum.retrieval as retrieval_mod
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

            # The same DB under "current" KEEPS the expired row but demotes
            # it: presence matches the default view exactly; the only
            # observable difference between default and current is the score.
            scores_c = {
                item.memory.id: item.score
                for item in await ret_b.search(QUERY, temporal_view="current")
            }
            assert sorted(scores_c) == sorted([*base_ids, expired_id])
            for memory_id in base_ids:
                assert scores_c[memory_id] == pytest.approx(scores_a[memory_id])
            assert scores_c[expired_id] == pytest.approx(scores_b[expired_id] * 0.70)
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
async def test_current_view_demotes_past_valid_until():
    """A date-only valid_until strictly before today stays in the current view
    but is demoted to EXPIRED_FACTOR of its default-view score; the default
    view is unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            memory_id = await _seed(
                db, "The primary database runs PostgreSQL 16.", valid_until=_yesterday()
            )
            current = await retriever.search(QUERY, temporal_view="current")
            default = await retriever.search(QUERY)
            current_scores = {item.memory.id: item.score for item in current}
            default_scores = {item.memory.id: item.score for item in default}
            assert memory_id in current_scores
            assert memory_id in default_scores
            assert current_scores[memory_id] < default_scores[memory_id]
            assert current_scores[memory_id] == pytest.approx(
                default_scores[memory_id] * 0.7
            )
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_validity_factor_pure_helper():
    """Given/When/Then: validity_factor returns EXPIRED_FACTOR only for a
    strictly-past valid_until under end-of-day normalization; == now, today,
    NULL, and unparseable all return 1.0."""
    from infinitum.retrieval import validity_factor

    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    assert validity_factor(None, now) == 1.0
    assert validity_factor("2026-06-01T12:00:00+00:00", now) == 1.0
    assert validity_factor("2026-06-01T11:59:59+00:00", now) == pytest.approx(0.70)
    assert validity_factor("2026-06-01", now) == 1.0
    assert validity_factor("2026-05-31", now) == pytest.approx(0.70)
    assert validity_factor("last tuesday", now) == 1.0


@pytest.mark.asyncio
async def test_current_view_boundary_valid_until_equal_now_not_demoted(monkeypatch):
    """A valid_until exactly equal to `now` is not strictly past, so the
    current-view score must be byte-identical to the default view. A fixed
    clock removes any midnight flakiness."""
    fixed = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    class _FixedNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz is not None else fixed

    monkeypatch.setattr(retrieval_mod, "datetime", _FixedNow)
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            memory_id = await _seed(
                db,
                "The primary database runs PostgreSQL 16.",
                valid_until="2026-06-01T12:00:00+00:00",
            )
            current = await retriever.search(QUERY, temporal_view="current")
            default = await retriever.search(QUERY)
            current_scores = {item.memory.id: item.score for item in current}
            default_scores = {item.memory.id: item.score for item in default}
            assert memory_id in current_scores
            assert current_scores[memory_id] == pytest.approx(default_scores[memory_id])
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_current_view_demotion_applies_after_gates():
    """The demotion factor is applied AFTER the relevance gates: an expired
    row with a threshold between 0.7*base and base stays present (score below
    the threshold, factor applied post-gate), while an expired row below the
    ungated threshold or with no relevance signal is evicted, never resurrected
    by the factor."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            strong_id = await _seed(
                db, "The primary database runs PostgreSQL 16.", valid_until=_yesterday()
            )
            # Seeded outside _seed: topic "coffee" + low importance/confidence
            # give it zero relevance signal, so the relevance gate must evict
            # it in every view and the factor can never resurrect it.
            weak = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="coffee",
                    content="The team prefers dark roast coffee.",
                    importance=0.1,
                    confidence=0.1,
                    valid_until=_yesterday(),
                )
            )
            weak_id = weak.id
            default_scores = {
                item.memory.id: item.score for item in await retriever.search(QUERY)
            }
            assert weak_id not in default_scores
            base = default_scores[strong_id]

            # Threshold strictly between 0.7*base (post-factor) and base
            # (pre-factor): gating on the ungated base keeps the row, demoted
            # BELOW the threshold. Gating after the factor would evict it.
            retriever.config.memory.minimum_retrieval_score = (base + base * 0.70) / 2
            scores = {
                item.memory.id: item.score
                for item in await retriever.search(QUERY, temporal_view="current")
            }
            assert scores[strong_id] == pytest.approx(base * 0.70)
            assert scores[strong_id] < retriever.config.memory.minimum_retrieval_score

            # Above the ungated base the row is evicted; the factor cannot
            # rescue it either way.
            retriever.config.memory.minimum_retrieval_score = base + 0.01
            current_ids = [
                item.memory.id for item in await retriever.search(QUERY, temporal_view="current")
            ]
            assert strong_id not in current_ids
            assert weak_id not in current_ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_current_view_orders_active_before_equal_expired():
    """Two equal-content rows, one active and one expired, must share equal
    base scores under "all" (byte-parity) and under "current" the active row
    must rank first with the expired row demoted to 0.7x."""
    content = "The primary database runs PostgreSQL 16."
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            active_id = await _seed(db, content)
            expired_id = await _seed(db, content, valid_until=_yesterday())
            all_scores = {item.memory.id: item.score for item in await retriever.search(QUERY)}
            assert all_scores[active_id] == pytest.approx(all_scores[expired_id])
            current = await retriever.search(QUERY, temporal_view="current")
            current_ids = [item.memory.id for item in current]
            assert current_ids.index(active_id) < current_ids.index(expired_id)
            current_scores = {item.memory.id: item.score for item in current}
            assert current_scores[active_id] == pytest.approx(all_scores[active_id])
            assert current_scores[expired_id] == pytest.approx(
                all_scores[expired_id] * 0.70
            )
        finally:
            await embeddings.close()
            await db.close()


SUCCESSOR = (
    "Our PostgreSQL backups run through a nightly pg_dump job writing compressed "
    "archives to object storage."
)
EXPIRED_ORIGINAL = "The PostgreSQL database uses a nightly backup strategy."
BACKUP_QUERY = "PostgreSQL backup strategy"


@pytest.mark.asyncio
async def test_current_view_ranks_reworded_successor_above_expired_original():
    """Realistic successor-ordering pin (external review follow-up): the expired
    row carries the query's exact vocabulary (the lexical ceiling of a mis-dated
    supersession-survivor) while the active successor is reworded and more
    specific, at identical importance/confidence. The raw all-view gap measures
    ~1.14x, below the     ~1.43x inversion threshold, so the demotion factor must decide
    the head-to-head. Seeded active-first: the expired row cannot win any
    freshness tie-break. Non-vacuity: the expired row must be PRESENT in the
    current view at exactly all-view x EXPIRED_FACTOR - retrieved and demoted,
    not absent; the ORDER (not the ratio literal) is the pin, so a deeper
    factor like 0.15 still passes here while a factor of 1.0 (no demotion)
    flips it."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            successor_id = await _seed(db, SUCCESSOR)
            expired_id = await _seed(db, EXPIRED_ORIGINAL, valid_until="2020-01-01")
            current = await retriever.search(BACKUP_QUERY, temporal_view="current")
            current_ids = [item.memory.id for item in current]
            current_scores = {item.memory.id: item.score for item in current}
            all_scores = {
                item.memory.id: item.score
                for item in await retriever.search(BACKUP_QUERY)
            }
            assert expired_id in current_ids
            assert current_ids.index(successor_id) < current_ids.index(expired_id)
            assert current_scores[expired_id] == pytest.approx(
                all_scores[expired_id] * retrieval_mod.EXPIRED_FACTOR
            )
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_current_view_expired_outranks_low_stakes_reworded_successor():
    """Trade-off pin: the same pair, but the successor is materially lower-stakes
    (importance 0.4, confidence 0.5 vs 0.8/0.9). The raw gap widens past the
    inversion threshold - raw_expired > raw_successor / EXPIRED_FACTOR ~= 1.43x -
    so the demoted expired row outranks the successor in the current view. That
    inversion IS the accepted current behavior; if a future factor, shadowing,
    or rendering change flips this pin, update it deliberately. Spot-checked
    against EXPIRED_FACTOR=0.15, where this pin flips (expired demoted too hard
    for the low-stakes gap), proving the factor is load-bearing here."""
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, retriever = await _open(tmp)
        try:
            successor = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="database",
                    content=SUCCESSOR,
                    importance=0.4,
                    confidence=0.5,
                )
            )
            successor_id = successor.id
            expired_id = await _seed(db, EXPIRED_ORIGINAL, valid_until="2020-01-01")
            current = await retriever.search(BACKUP_QUERY, temporal_view="current")
            current_ids = [item.memory.id for item in current]
            assert expired_id in current_ids
            assert current_ids.index(expired_id) < current_ids.index(successor_id)
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


def test_route_default_current_shows_expired_row_demoted():
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
            assert any("PostgreSQL 12" in c for c in contents)


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
