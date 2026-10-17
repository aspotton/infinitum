"""Load-bearing parity gate for N2 candidate-union retrieval (plan todo 4).

Seeded and crafted corpora are searched through MemoryRetriever.search on the
candidate fast path (full_scan_threshold=1, ceiling=max(1, 50*3)=150 so a
sub-150-row store exercises the fast path genuinely) versus the verbatim full
scan (full_scan_threshold=10**9) against the SAME database, and the
(id, score, semantic_score) triples must be EXACTLY equal. Anti-vacuity:
every fast run proves build_retrieval_candidates was actually called and the
fallback memo stayed empty - parity by vacuous fallback fails loudly.
"""

import logging
import random
from datetime import UTC, datetime
from difflib import SequenceMatcher

import numpy as np
import pytest

from infinitum import retrieval as retrieval_mod
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory, ScoredMemory
from infinitum.retrieval import MemoryRetriever
from infinitum.text import tokens

_DIM = 8
_WORDS = [
    "postgres", "redis", "nginx", "kafka", "airflow", "terraform", "gateway",
    "session", "vector", "index", "deploy", "rollback", "cluster", "shard",
    "token", "budget", "cache", "queue", "socket", "ledger", "canvas",
    "beacon", "harbor", "meadow", "quartz", "summit", "window", "cipher",
]
_QUERY_VECTOR = np.eye(1, _DIM, 0, dtype=np.float32).ravel()
# Kept >= 1e-3 away from the 0.08 gate so float32 noise can never straddle it.
_COS_CYCLE = [0.0, 0.05, 0.079, 0.081, 0.12, 0.3, 0.6, 0.95]


class _StubEmbeddings(EmbeddingClient):
    """EmbeddingClient with the network replaced by a fixed vector (None = off)."""

    def __init__(self, vector: np.ndarray | None):
        super().__init__(AppConfig().embeddings)
        self.vector = vector

    async def embed(self, text: str) -> np.ndarray | None:
        return self.vector


def _vector_for(cos: float, index: int) -> np.ndarray:
    vector = np.zeros(_DIM, dtype=np.float32)
    vector[0] = cos
    if cos < 1.0:
        vector[index % (_DIM - 1) + 1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return vector


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, ts: datetime) -> None:
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return ts if tz is None else ts.astimezone(tz)

    monkeypatch.setattr(retrieval_mod, "datetime", Frozen)


def _cfg(threshold: int, embeddings_on: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.memory.full_scan_threshold = threshold
    cfg.embeddings.enabled = embeddings_on
    # Parity fixtures are score-tuned against the pre-0.7 0.18 precision floor;
    # pin it so the crafted counterexample cosines keep landing where designed.
    cfg.memory.minimum_retrieval_score = 0.18
    return cfg


async def _seed_memories(
    db: Database,
    memories: list[Memory],
    embeddings_on: bool,
    cos_map: dict[str, float] | None = None,
) -> None:
    model = AppConfig().embeddings.model
    for index, memory in enumerate(memories):
        await db.create_memory(memory)
        if embeddings_on:
            cos = (cos_map or {}).get(memory.id, _COS_CYCLE[index % len(_COS_CYCLE)])
            await db.set_embedding(memory.id, model, _vector_for(cos, index))


async def _assert_parity(
    db: Database,
    memories: list[Memory],
    queries: list[str],
    embeddings_on: bool,
    monkeypatch: pytest.MonkeyPatch,
    ts: datetime,
    cos_map: dict[str, float] | None = None,
) -> list[str]:
    """Given the same db, When each query runs on both paths, Then the
    (id, score, semantic_score) triples are exactly equal and the fast path
    genuinely ran (spy count > 0, fallback memo empty)."""
    await _seed_memories(db, memories, embeddings_on, cos_map)
    stub = _StubEmbeddings(_QUERY_VECTOR if embeddings_on else None)
    retriever_fast = MemoryRetriever(db, stub, _cfg(1, embeddings_on))
    retriever_slow = MemoryRetriever(db, stub, _cfg(10**9, embeddings_on))
    calls: list[str] = []
    original = db.build_retrieval_candidates

    async def spy(**kwargs):
        calls.append(str(kwargs.get("qnorm", "")))
        return await original(**kwargs)

    monkeypatch.setattr(db, "build_retrieval_candidates", spy)
    _freeze_clock(monkeypatch, ts)
    for query in queries:
        fast = await retriever_fast.search(query)
        slow = await retriever_slow.search(query)
        assert [(s.memory.id, s.score, s.semantic_score) for s in fast] == [
            (s.memory.id, s.score, s.semantic_score) for s in slow
        ], f"parity broken for query {query!r}"
        assert calls, "fast path never ran: build_retrieval_candidates was not called"
        assert retriever_fast._warned_fallbacks == set(), "fast path fell back: parity is vacuous"
    return calls


def _seeded_memories(seed: int, ts: datetime) -> list[Memory]:
    rng = random.Random(seed)
    memories: list[Memory] = []
    for index in range(rng.randint(20, 40)):
        words = rng.sample(_WORDS, k=rng.randint(2, 6))
        topic = "-".join(words)
        body = " ".join(rng.choices(words, k=rng.randint(6, 30)))
        kind = index % 7
        if kind == 3:
            body = " ".join(rng.choices(_WORDS, k=140))
        elif kind == 6:
            body = "\t" * 200 + " ".join(words)
        if kind == 1:
            topic = "-".join(rng.sample(_WORDS, k=8))
        memories.append(
            Memory(
                id=f"mem_s{seed}_{index:03d}",
                memory_type=rng.choice(
                    ["fact", "decision", "preference", "goal", "procedure", "lesson"]
                ),
                topic=topic,
                content=body,
                importance=round(rng.uniform(0.1, 0.99), 2),
                confidence=round(rng.uniform(0.3, 0.99), 2),
                created_at=ts,
                updated_at=ts,
            )
        )
    return memories


_SEEDED_QUERIES = [
    "session gateway kafka index",
    "postgres deploy rollback vector",
    "cache queue cipher ledger",
    "zzqx pluv",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_candidate_fast_path_matches_full_scan_on_seeded_corpora(
    seed: int, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    for embeddings_on in (False, True):
        db = Database(str(tmp_path / f"seed{seed}-{embeddings_on}.db"))
        await db.connect()
        try:
            ts = datetime.now(UTC)
            memories = _seeded_memories(seed, ts)
            queries = _SEEDED_QUERIES + [
                " ".join(random.Random(seed * 31 + i).sample(_WORDS, k=3)) for i in range(2)
            ]
            await _assert_parity(db, memories, queries, embeddings_on, monkeypatch, ts)
        finally:
            await db.close()


def _crafted_memories(ts: datetime) -> list[Memory]:
    long_topic = "-".join(_WORDS[:10])
    return [
        Memory(id="mem_c_para", topic=long_topic,
               content="The team settled on single-node writes with multiversion concurrency control.",
               importance=0.5, created_at=ts, updated_at=ts),
        Memory(id="mem_c_glued", topic="redis-cache",
               content="Cached values are kept in the session layer.",
               importance=0.5, created_at=ts, updated_at=ts),
        Memory(id="mem_c_tabs", topic=long_topic, content="\t" * 200 + "redis cache",
               importance=0.4, created_at=ts, updated_at=ts),
        Memory(id="mem_c_neardup", topic=long_topic, content="drchestration pipeliner",
               importance=0.4, created_at=ts, updated_at=ts),
        Memory(id="mem_c_goal", memory_type="goal", topic=long_topic,
               content="Ship the analytics migration by end of quarter with zero downtime.",
               importance=0.9, created_at=ts, updated_at=ts),
        Memory(id="mem_c_decision", memory_type="decision", topic=long_topic,
               content=" ".join(_WORDS[:14]),
               importance=0.88, created_at=ts, updated_at=ts),
        Memory(id="mem_c_long", topic=long_topic,
               content=("padding word " * 700)[:8300] + " vault backup rotates nightly",
               importance=0.5, created_at=ts, updated_at=ts),
        Memory(id="mem_c_nearmiss", topic=long_topic,
               content="Beacon meadow quartz window cipher socket ledger shard token budget harbor canvas index summit.",
               importance=0.4, created_at=ts, updated_at=ts),
        Memory(id="mem_c_arch", topic="redis-cache",
               content="Archived redis cache default layer remains invisible to retrieval.",
               importance=0.6, status="archived", created_at=ts, updated_at=ts),
    ]


@pytest.mark.asyncio
async def test_zero_token_neardup_pair_is_token_disjoint_and_ratio_bound():
    assert not (tokens("orchestration pipeline") & tokens("drchestration pipeliner"))
    assert SequenceMatcher(
        None, "orchestration pipeline", "drchestration pipeliner"
    ).ratio() >= 0.5


@pytest.mark.asyncio
async def test_candidate_fast_path_matches_full_scan_on_crafted_counterexamples(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    queries = [
        "redis cache",
        "rediscache",
        "orchestration pipeline",
        "we chose row-based mvcc storage",
        "high level quarterly targets",
        "vault backup schedule",
        "zzqx pluv",
    ]
    # The paraphrase pair needs a clearly-above-floor cosine, the archived
    # decoy a perfect vector the loader must drop, the near-miss a below-floor
    # cosine both paths must drop.
    cos_map = {"mem_c_para": 0.5, "mem_c_arch": 0.95, "mem_c_nearmiss": 0.079}
    for embeddings_on in (False, True):
        db = Database(str(tmp_path / f"crafted-{embeddings_on}.db"))
        await db.connect()
        try:
            ts = datetime.now(UTC)
            result = await _assert_parity(
                db, _crafted_memories(ts), queries, embeddings_on, monkeypatch, ts, cos_map
            )
            stub = _StubEmbeddings(_QUERY_VECTOR if embeddings_on else None)
            retriever = MemoryRetriever(db, stub, _cfg(1, embeddings_on))
            by_query = {}
            for query in queries:
                by_query[query] = [s.memory.id for s in await retriever.search(query)]
            assert "mem_c_glued" in by_query["rediscache"]
            assert "mem_c_tabs" in by_query["rediscache"]
            assert "mem_c_neardup" in by_query["orchestration pipeline"]
            assert "mem_c_goal" in by_query["high level quarterly targets"]
            assert "mem_c_long" in by_query["vault backup schedule"]
            expected_no_hit = {"mem_c_goal", "mem_c_decision"}
            if not embeddings_on:
                assert set(by_query["we chose row-based mvcc storage"]) == expected_no_hit
                assert set(by_query["zzqx pluv"]) == expected_no_hit
            else:
                assert "mem_c_para" in by_query["we chose row-based mvcc storage"]
                assert set(by_query["zzqx pluv"]) == expected_no_hit | {
                    "mem_c_para", "mem_c_neardup", "mem_c_long"
                }
            assert "mem_c_arch" not in {i for ids in by_query.values() for i in ids}
            assert result
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_degenerate_union_falls_back_with_exactly_one_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
):
    db = Database(str(tmp_path / "degenerate.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        for index in range(200):
            await db.create_memory(
                Memory(id=f"mem_d_{index:03d}", topic=f"bulk-topic-{index}",
                       content="x" * 8200 + " filler session word", created_at=ts, updated_at=ts)
            )
        retriever_fast = MemoryRetriever(db, _StubEmbeddings(None), _cfg(5, False))
        retriever_slow = MemoryRetriever(db, _StubEmbeddings(None), _cfg(10**9, False))
        _freeze_clock(monkeypatch, ts)
        with caplog.at_level(logging.WARNING, logger="infinitum.retrieval"):
            first = await retriever_fast.search("pipeline gateway session")
            second = await retriever_fast.search("pipeline gateway session")
            third = await retriever_fast.search("rollback cipher ledger")
        warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "infinitum.retrieval"
        ]
        reference = await retriever_slow.search("pipeline gateway session")
        other = await retriever_slow.search("rollback cipher ledger")
        assert [(s.memory.id, s.score) for s in first] == [(s.memory.id, s.score) for s in reference]
        assert [(s.memory.id, s.score) for s in third] == [(s.memory.id, s.score) for s in other]
        assert first and second
        assert len(warnings) == 1
        assert len(retriever_fast._warned_fallbacks) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fts_disabled_falls_back_with_warning_and_identical_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
):
    db = Database(str(tmp_path / "nofts.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await _seed_memories(db, _seeded_memories(11, ts), False)
        db.fts_enabled = False
        retriever_fast = MemoryRetriever(db, _StubEmbeddings(None), _cfg(1, False))
        retriever_slow = MemoryRetriever(db, _StubEmbeddings(None), _cfg(10**9, False))
        _freeze_clock(monkeypatch, ts)
        with caplog.at_level(logging.WARNING, logger="infinitum.retrieval"):
            fast = await retriever_fast.search("postgres deploy rollback vector")
            fast_again = await retriever_fast.search("postgres deploy rollback vector")
        fast2 = await retriever_slow.search("postgres deploy rollback vector")
        warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "infinitum.retrieval"
        ]
        assert [(s.memory.id, s.score) for s in fast] == [(s.memory.id, s.score) for s in fast2]
        assert [(s.memory.id, s.score) for s in fast] == [(s.memory.id, s.score) for s in fast_again]
        assert len(warnings) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_arm_building_crash_falls_back_loudly_with_identical_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
):
    db = Database(str(tmp_path / "crash.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await _seed_memories(db, _crafted_memories(ts), False)

        async def boom(**kwargs):
            raise RuntimeError("injected arm failure")

        monkeypatch.setattr(db, "build_retrieval_candidates", boom)
        retriever_fast = MemoryRetriever(db, _StubEmbeddings(None), _cfg(1, False))
        retriever_slow = MemoryRetriever(db, _StubEmbeddings(None), _cfg(10**9, False))
        _freeze_clock(monkeypatch, ts)
        with caplog.at_level(logging.WARNING, logger="infinitum.retrieval"):
            fast = await retriever_fast.search("redis cache")
            fast_again = await retriever_fast.search("redis cache")
        slow = await retriever_slow.search("redis cache")
        warnings = [
            record for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "infinitum.retrieval"
        ]
        assert [(s.memory.id, s.score) for s in fast] == [(s.memory.id, s.score) for s in slow]
        assert fast == fast_again
        assert len(warnings) == 1
        assert retriever_fast._warned_fallbacks == {
            ("arm-error", await db.memory_state_watermark())
        }
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sub_008_floor_always_uses_full_scan(tmp_path, monkeypatch: pytest.MonkeyPatch):
    db = Database(str(tmp_path / "floor.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        await _seed_memories(db, _seeded_memories(12, ts), False)
        cfg = _cfg(1, False)
        cfg.memory.minimum_relevance_score = 0.05
        retriever = MemoryRetriever(db, _StubEmbeddings(None), cfg)
        calls: list[str] = []
        original = db.build_retrieval_candidates

        async def spy(**kwargs):
            calls.append("x")
            return await original(**kwargs)

        monkeypatch.setattr(db, "build_retrieval_candidates", spy)
        _freeze_clock(monkeypatch, ts)
        results = await retriever.search("postgres deploy rollback vector")
        assert results
        assert calls == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_empty_union_is_a_real_empty_result_not_a_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    db = Database(str(tmp_path / "empty.db"))
    await db.connect()
    try:
        ts = datetime.now(UTC)
        long_topic = "-".join(_WORDS[:10])
        long_content = " ".join(_WORDS[:20] * 12)
        for index in range(2):
            await db.create_memory(
                Memory(id=f"mem_e_{index}", topic=f"{long_topic}-{index}", content=long_content,
                       importance=0.3, created_at=ts, updated_at=ts)
            )
        retriever_fast = MemoryRetriever(db, _StubEmbeddings(None), _cfg(1, False))
        retriever_slow = MemoryRetriever(db, _StubEmbeddings(None), _cfg(10**9, False))
        calls: list[str] = []
        original = db.build_retrieval_candidates

        async def spy(**kwargs):
            calls.append(str(kwargs.get("qnorm", "")))
            return await original(**kwargs)

        monkeypatch.setattr(db, "build_retrieval_candidates", spy)
        _freeze_clock(monkeypatch, ts)
        fast = await retriever_fast.search("zqjx vkwt")
        slow = await retriever_slow.search("zqjx vkwt")
        assert fast == []
        assert [(s.memory.id, s.score) for s in fast] == [(s.memory.id, s.score) for s in slow]
        assert calls == ["zqjx vkwt"]
        assert retriever_fast._warned_fallbacks == set()
    finally:
        await db.close()
