import tempfile

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
