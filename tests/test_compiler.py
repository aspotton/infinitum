import tempfile

import pytest

from infinitum.compiler import CompiledMemoryContext, ContextCompiler
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory, RequestContext
from infinitum.retrieval import MemoryRetriever
from infinitum.tokenizer import TokenCounter


async def _session_compiler(tmp: str) -> tuple[Database, EmbeddingClient, ContextCompiler]:
    config = AppConfig()
    config.memory.database_path = f"{tmp}/runtime.db"
    config.embeddings.enabled = False
    config.memory.minimum_retrieval_score = 0.10
    db = Database(config.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(config.embeddings)
    retriever = MemoryRetriever(db, embeddings, config)
    return db, embeddings, ContextCompiler(db, retriever, TokenCounter(), config)


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


async def _seed_database_memory(db: Database) -> Memory:
    return await db.create_memory(
        Memory(
            memory_type="decision",
            topic="database",
            content="PostgreSQL 17 is the current database standard.",
            importance=0.95,
            confidence=1.0,
        )
    )


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


@pytest.mark.asyncio
async def test_session_pin_returns_byte_identical_block_until_memory_changes():
    # Given: one active memory and a session-pinned compile
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        # When: two turns in the same session with no DB mutation between them
        first = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )
        second = await compiler.compile(
            _user("Tell me more about the database choice."), session_id="s1"
        )

        # Then: the block is pinned byte-for-byte and non-empty
        assert first.text
        assert first.text == second.text

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_session_pin_invalidates_when_memory_created_between_turns():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        first = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )
        await db.create_memory(
            Memory(
                topic="database",
                content="SQLite is the embedded cache store.",
                importance=0.9,
                confidence=1.0,
            )
        )
        second = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )

        assert first.text != second.text
        assert "SQLite" in second.text

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_without_session_id_cache_is_bypassed():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        await compiler.compile(_user("What database standard are we using?"))
        await compiler.compile(_user("What database standard are we using?"))

        assert len(compiler._session_cache) == 0

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_mutating_returned_text_does_not_poison_cached_block():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        # Given: a primed cache entry for the session
        await compiler.compile(_user("What database standard are we using?"), session_id="s1")

        # When: successive hit returns are mutated by the caller
        first = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )
        pristine = first.text
        first.text += "|MUT"
        second = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )

        # Then: hits hand out replace() copies; the stored block never absorbs the append
        assert second.text == pristine
        assert "|MUT" not in second.text

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_same_session_different_user_context_never_shares_block():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        await compiler.compile(
            _user("What database standard are we using?"),
            request_context=RequestContext(user_id="a"),
            session_id="s1",
        )
        await compiler.compile(
            _user("What database standard are we using?"),
            request_context=RequestContext(user_id="b"),
            session_id="s1",
        )

        assert len(compiler._session_cache) == 2
        assert ("s1", "a", "") in compiler._session_cache
        assert ("s1", "b", "") in compiler._session_cache

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_session_pin_invalidates_when_memory_archived_between_turns():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        memory = await _seed_database_memory(db)

        first = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )
        assert await db.archive_memory(memory.id)
        second = await compiler.compile(
            _user("What database standard are we using?"), session_id="s1"
        )

        assert "PostgreSQL 17" in first.text
        assert "PostgreSQL 17" not in second.text

        await embeddings.close()
        await db.close()


@pytest.mark.asyncio
async def test_session_cache_evicts_oldest_beyond_capacity():
    with tempfile.TemporaryDirectory() as tmp:
        db, embeddings, compiler = await _session_compiler(tmp)
        await _seed_database_memory(db)

        for idx in range(65):
            await compiler.compile(
                _user("What database standard are we using?"), session_id=f"s{idx}"
            )

        assert len(compiler._session_cache) == 64
        assert ("s0", "", "") not in compiler._session_cache
        assert ("s64", "", "") in compiler._session_cache

        await embeddings.close()
        await db.close()
