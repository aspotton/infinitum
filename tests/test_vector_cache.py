"""Watermark-invalidated embedding-matrix cache (N2 plan todo 5).

The semantic arm and the full-scan scorer share one decoded-vector cache keyed
by the embeddings table's OWN watermark (MAX(updated_at), COUNT(*) FROM
memory_embeddings) plus the config model and query dim. These tests pin the
invalidation matrix (hit / insert / replace), the model-change poison guard, the
model+dim build-time filter, the embeddings-disabled short-circuit, and that
_full_scan is served by the cache with ZERO direct Database.get_embeddings
calls. Rebuild counts are asserted as exact integers so a vacuous
rebuild-every-call cache fails loudly.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from infinitum import retrieval as retrieval_mod
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory
from infinitum.retrieval import MemoryRetriever

_DIM = 8
# Zero-token gibberish: no lexical/topic signal for any fixture row, so only a
# stored vector can surface a memory. Mirrors the parity test's gibberish query.
_QUERY = "zzqx pluv"
_QUERY_VECTOR = np.eye(1, _DIM, 0, dtype=np.float32).ravel()
_MODEL = AppConfig().embeddings.model
_LONG_TOPIC = "-".join(
    ["postgres", "redis", "nginx", "kafka", "airflow", "terraform", "gateway", "session"]
)


class _StubEmbeddings(EmbeddingClient):
    """EmbeddingClient with the network replaced by a fixed vector (None = off)."""

    def __init__(self, vector: np.ndarray | None):
        super().__init__(AppConfig().embeddings)
        self.vector = vector

    async def embed(self, text: str) -> np.ndarray | None:
        return self.vector


def _vector_for(cos: float) -> np.ndarray:
    vector = np.zeros(_DIM, dtype=np.float32)
    vector[0] = cos
    if cos < 1.0:
        vector[1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return vector


def _cfg(threshold: int, embeddings_on: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.full_scan_threshold = threshold
    cfg.embeddings.enabled = embeddings_on
    return cfg


def _memory(mid: str, content: str, ts: datetime) -> Memory:
    return Memory(
        id=mid,
        topic=_LONG_TOPIC,
        content=content,
        importance=0.5,
        confidence=0.9,
        created_at=ts,
        updated_at=ts,
    )


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, ts: datetime) -> None:
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return ts if tz is None else ts.astimezone(tz)

    monkeypatch.setattr(retrieval_mod, "datetime", Frozen)


def _trace(result: list) -> list[tuple]:
    return [(s.memory.id, s.score, s.semantic_score) for s in result]


@pytest.mark.asyncio
async def test_cache_hit_and_invalidation_matrix_over_forced_fast_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Given a 2-row store forced onto the fast path (threshold=1, ceiling=150),
    When searching twice, then exactly one rebuild and identical results; when
    a vectorless memory is embedded (COUNT+MAX move), the next search surfaces
    it with rebuild 2; when that vector is replaced (same id, same COUNT, only
    updated_at moves), the next search forgets it with rebuild 3."""
    db = Database(str(tmp_path / "invalidate.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await db.create_memory(_memory("mem_a", "the storage layer compacts segments nightly", ts))
        await db.create_memory(_memory("mem_b", "nightly vault backups rotate through three keys", ts))
        await db.set_embedding("mem_a", _MODEL, _vector_for(0.5))
        retriever = MemoryRetriever(db, _StubEmbeddings(_QUERY_VECTOR), _cfg(1, True))
        _freeze_clock(monkeypatch, ts)

        first = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 1
        assert [s.memory.id for s in first] == ["mem_a"]

        again = await retriever.search(_QUERY)
        assert _trace(again) == _trace(first)
        assert retriever._vector_cache_rebuilds == 1, "cache hit must not rebuild"

        await db.set_embedding("mem_b", _MODEL, _vector_for(0.95))
        surfaced = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 2, "insert must move the watermark"
        by_id = {s.memory.id: s for s in surfaced}
        assert "mem_b" in by_id, "the new vector must surface semantically"
        assert by_id["mem_b"].semantic_score >= 0.9

        await db.set_embedding("mem_b", _MODEL, np.zeros(_DIM, dtype=np.float32))
        replaced = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 3, "replace moves only updated_at; the bump must invalidate"
        assert "mem_b" not in {s.memory.id for s in replaced}, "stale vector served: replace was ignored"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_dim_mismatched_row_scores_zero_without_exception(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Given a stored vector of the wrong length under the right model, When
    searching, Then that memory surfaces on its lexical signal with
    semantic_score 0.0 and no exception (excluded rows never enter the matrix)."""
    db = Database(str(tmp_path / "dim.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await db.create_memory(_memory("mem_a", "redis cache handles the session layer", ts))
        await db.set_embedding("mem_a", _MODEL, np.zeros(4, dtype=np.float32))
        retriever = MemoryRetriever(db, _StubEmbeddings(_QUERY_VECTOR), _cfg(1, True))
        _freeze_clock(monkeypatch, ts)
        results = await retriever.search("redis cache")
        assert [s.memory.id for s in results] == ["mem_a"]
        assert results[0].semantic_score == 0.0
        assert retriever._vector_cache_rebuilds == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_disabled_embeddings_never_read_the_watermark(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Given embeddings disabled, When searching, Then the matrix is never
    built (zero watermark reads, zero rebuilds) and results stay lexical."""
    db = Database(str(tmp_path / "disabled.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await db.create_memory(_memory("mem_a", "redis cache handles the session layer", ts))
        reads: list = []
        original = db.embeddings_watermark

        async def spy():
            reads.append(1)
            return await original()

        monkeypatch.setattr(db, "embeddings_watermark", spy)
        retriever = MemoryRetriever(db, _StubEmbeddings(None), _cfg(1, False))
        _freeze_clock(monkeypatch, ts)
        results = await retriever.search("redis cache")
        assert [s.memory.id for s in results] == ["mem_a"]
        assert all(s.semantic_score == 0.0 for s in results)
        assert reads == []
        assert retriever._vector_cache_rebuilds == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_full_scan_is_served_by_the_cache_with_zero_direct_reads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Given the full-scan dispatch (threshold=10**9) with embeddings on, When
    a second search runs at an unchanged watermark, Then Database.get_embeddings
    is called zero times across it, and the cached map equals a direct
    get_embeddings(ids) fetch key-for-key and bit-for-bit."""
    db = Database(str(tmp_path / "fullscan.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await db.create_memory(_memory("mem_a", "the storage layer compacts segments nightly", ts))
        await db.create_memory(_memory("mem_b", "nightly vault backups rotate through three keys", ts))
        await db.set_embedding("mem_a", _MODEL, _vector_for(0.5))
        calls: list = []
        original = db.get_embeddings

        async def spy(memory_ids=None):
            calls.append(memory_ids)
            return await original(memory_ids)

        monkeypatch.setattr(db, "get_embeddings", spy)
        retriever = MemoryRetriever(db, _StubEmbeddings(_QUERY_VECTOR), _cfg(10**9, True))
        _freeze_clock(monkeypatch, ts)

        warm = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 1
        assert calls == [None], "rebuild must be exactly one fetch-all read"
        calls.clear()

        second = await retriever.search(_QUERY)
        assert calls == [], "_full_scan must make zero direct get_embeddings calls"
        assert retriever._vector_cache_rebuilds == 1
        assert _trace(second) == _trace(warm)

        ids = ["mem_a", "mem_b"]
        calls.clear()
        direct = await db.get_embeddings(ids)
        served = {k: v for k, v in retriever._vec_rows.items() if k in set(ids)}
        assert set(served) == set(direct)
        for key, (model, vector) in served.items():
            assert direct[key][0] == model
            assert bool(np.array_equal(direct[key][1], vector))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_model_change_mid_life_forces_rebuild_not_a_poisoned_matrix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Given a warm cache under model A, When the config model switches to B at
    an unchanged watermark, Then the matrix rebuilds (count 2) and model A's
    vector no longer scores: the model component of the key cannot serve a
    matrix built for a different embedding space."""
    db = Database(str(tmp_path / "poison.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await db.create_memory(_memory("mem_a", "the storage layer compacts segments nightly", ts))
        await db.set_embedding("mem_a", _MODEL, _vector_for(0.5))
        cfg = _cfg(1, True)
        retriever = MemoryRetriever(db, _StubEmbeddings(_QUERY_VECTOR), cfg)
        _freeze_clock(monkeypatch, ts)

        warm = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 1
        assert [s.memory.id for s in warm] == ["mem_a"]

        cfg.embeddings.model = "some-other-embed-model"
        after = await retriever.search(_QUERY)
        assert retriever._vector_cache_rebuilds == 2
        assert after == []
    finally:
        await db.close()
