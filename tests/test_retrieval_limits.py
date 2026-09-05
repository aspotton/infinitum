import tempfile
from datetime import UTC, datetime

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory
from infinitum.retrieval import MemoryRetriever


@pytest.mark.asyncio
async def test_explicit_retrieval_limit_is_honored():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.memory.retrieve_candidates = 50
        db = Database(cfg.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(cfg.embeddings)
        try:
            for i in range(20):
                await db.create_memory(
                    Memory(
                        memory_type="fact",
                        topic="database",
                        content=f"Database detail number {i} uses PostgreSQL.",
                        importance=0.8,
                        confidence=0.9,
                    )
                )
            retriever = MemoryRetriever(db, embeddings, cfg)
            results = await retriever.search("PostgreSQL database", limit=3)
            assert len(results) == 3
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_equal_score_results_sort_by_memory_id():
    """Given two active memories that score identically (same content, same
    timestamps), When searching, Then the returned order is deterministic:
    ascending memory.id, independent of insertion/DB row order.

    The context compiler caches compiled blocks byte-for-byte across turns, so
    an unstable tie-break in the candidate sort would reshuffle the injected
    block and defeat prompt caching (cache-stable-context-injection todo 2).
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        db = Database(cfg.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(cfg.embeddings)
        try:
            content = "The deployment runs on port 8788 behind nginx."
            now = datetime.now(UTC)
            first = Memory(
                id="mem_zzz",
                memory_type="fact",
                topic="deploy",
                content=content,
                created_at=now,
                updated_at=now,
            )
            second = Memory(
                id="mem_aaa",
                memory_type="fact",
                topic="deploy",
                content=content,
                created_at=now,
                updated_at=now,
            )
            await db.create_memory(second)
            await db.create_memory(first)

            retriever = MemoryRetriever(db, embeddings, cfg)
            results = await retriever.search("deployment port nginx")
            ids = [s.memory.id for s in results]
            scores = {s.memory.id: s.score for s in results}

            assert set(ids) == {"mem_zzz", "mem_aaa"}
            assert scores["mem_zzz"] == scores["mem_aaa"]
            assert ids == sorted(ids, key=lambda i: (-scores[i], i))
            assert ids == ["mem_aaa", "mem_zzz"]

            fresh = MemoryRetriever(db, embeddings, cfg)
            again = [s.memory.id for s in await fresh.search("deployment port nginx")]
            assert again == ids
        finally:
            await embeddings.close()
            await db.close()
