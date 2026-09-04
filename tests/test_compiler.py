import tempfile

import pytest

from infinitum.compiler import CompiledMemoryContext, ContextCompiler
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


def _inject_compiler(config: AppConfig) -> ContextCompiler:
    return ContextCompiler(None, None, TokenCounter(), config)


def _compiled(text: str = "<infinitum_memory>x</infinitum_memory>") -> CompiledMemoryContext:
    return CompiledMemoryContext(text, [], 0, 0)


def test_inject_suffix_places_block_before_last_user_message():
    # Given: default suffix config and a multi-turn conversation
    config = AppConfig()
    assert config.context.inject_position == "suffix"
    compiler = _inject_compiler(config)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest"},
    ]

    # When: injecting the memory block
    injected = compiler.inject(messages, _compiled())

    # Then: the block sits immediately before the last user message, after the assistant turn
    assert len(injected) == 5
    assert injected[3]["content"] == "<infinitum_memory>x</infinitum_memory>"
    assert injected[3]["role"] == config.context.memory_message_role
    assert injected[2]["role"] == "assistant"
    assert injected[4] == {"role": "user", "content": "latest"}
    assert injected[0] == {"role": "system", "content": "sys"}


def test_inject_prefix_preserves_legacy_position():
    config = AppConfig()
    config.context.inject_position = "prefix"
    compiler = _inject_compiler(config)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest"},
    ]

    injected = compiler.inject(messages, _compiled())

    assert injected[1]["content"] == "<infinitum_memory>x</infinitum_memory>"
    assert injected[0] == {"role": "system", "content": "sys"}


def test_inject_suffix_single_user_message_lands_at_index_zero():
    config = AppConfig()
    compiler = _inject_compiler(config)
    messages = [{"role": "user", "content": "hello"}]

    injected = compiler.inject(messages, _compiled())

    assert len(injected) == 2
    assert injected[0]["role"] == config.context.memory_message_role
    assert injected[1] == {"role": "user", "content": "hello"}


def test_inject_suffix_without_user_message_falls_back_to_legacy_rule():
    config = AppConfig()
    compiler = _inject_compiler(config)
    messages = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "reply"}]

    injected = compiler.inject(messages, _compiled())

    assert injected[1]["content"] == "<infinitum_memory>x</infinitum_memory>"


def test_inject_suffix_empty_messages_does_not_raise():
    config = AppConfig()
    compiler = _inject_compiler(config)

    injected = compiler.inject([], _compiled())

    assert injected == [
        {"role": config.context.memory_message_role, "content": "<infinitum_memory>x</infinitum_memory>"}
    ]


def test_inject_suffix_uses_configured_memory_message_role():
    config = AppConfig()
    config.context.memory_message_role = "developer"
    compiler = _inject_compiler(config)
    messages = [{"role": "user", "content": "hi"}]

    injected = compiler.inject(messages, _compiled())

    assert injected[0]["role"] == "developer"
