import tempfile

import pytest

from infinitum.compiler import ContextCompiler
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory
from infinitum.retrieval import MemoryRetriever
from infinitum.tokenizer import TokenCounter


@pytest.mark.asyncio
async def test_compiler_injects_relevant_active_memory_and_skips_superseded():
    with tempfile.TemporaryDirectory() as tmp:
        config = AppConfig()
        config.memory.database_path = f"{tmp}/runtime.db"
        config.embeddings.enabled = False
        config.memory.minimum_retrieval_score = 0.10
        db = Database(config.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(config.embeddings)
        retriever = MemoryRetriever(db, embeddings, config)
        compiler = ContextCompiler(db, retriever, TokenCounter(), config)

        old = await db.create_memory(Memory(memory_type="decision", topic="database", content="MySQL is the database.", importance=0.9, confidence=1.0))
        current = await db.create_memory(Memory(memory_type="decision", topic="database", content="PostgreSQL 17 is the current database standard.", importance=0.95, confidence=1.0))
        await db.supersede_memory(old.id, current.id)

        messages = [{"role": "user", "content": "What database standard are we using?"}]
        compiled = await compiler.compile(messages)
        assert "PostgreSQL 17" in compiled.text
        assert "MySQL is the database" not in compiled.text
        injected = compiler.inject(messages, compiled)
        assert len(injected) == 2
        assert injected[0]["role"] == "system"

        await embeddings.close()
        await db.close()
