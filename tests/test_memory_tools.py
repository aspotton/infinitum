"""Unit-tier tests for the read-only memory drill-down tools (Batch 1)."""

import json
import tempfile
from types import SimpleNamespace

from infinitum import memory_tools
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Event, Memory
from infinitum.retrieval import MemoryRetriever


async def make_runtime(tmp: str):
    """Build a minimal AppConfig + Database + Retriever as the tool runtime."""
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    retriever = MemoryRetriever(db, EmbeddingClient(cfg.embeddings), cfg)
    return SimpleNamespace(db=db, retriever=retriever)


async def add_event(db: Database) -> str:
    """Insert an event so memory_sources foreign keys resolve."""
    event = Event(session_id="s1", event_type="message.user", content="source")
    await db.add_event(event)
    return event.id


# --- execute: infinitum_memory_search ---------------------------------------


async def test_search_result_shape_and_score_rounding():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            memory = Memory(topic="database", content="We use PostgreSQL 17.", importance=1.0)
            await runtime.db.create_memory(memory)
            result = await memory_tools.execute(
                "infinitum_memory_search", '{"query": "PostgreSQL", "limit": 5}', runtime, None
            )
            payload = json.loads(result)
            assert payload["results"], "expected at least one hit"
            item = payload["results"][0]
            assert set(item) == {"id", "memory_type", "topic", "status", "score", "content"}
            assert item["id"] == memory.id
            assert item["status"] == "active"
            assert isinstance(item["score"], float)
            assert item["score"] == round(item["score"], 3)
        finally:
            await runtime.db.close()


async def test_search_limit_clamped_to_max():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            for i in range(3):
                await runtime.db.create_memory(
                    Memory(topic="misc", content=f"Fact number {i} about databases.")
                )
            result = await memory_tools.execute(
                "infinitum_memory_search", '{"query": "databases", "limit": 100000}', runtime, None
            )
            payload = json.loads(result)
            assert len(payload["results"]) <= memory_tools.MAX_SEARCH_LIMIT
            assert len(payload["results"]) == 3
        finally:
            await runtime.db.close()


async def test_search_caps_result_at_max_chars():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            for i in range(4):
                await runtime.db.create_memory(
                    Memory(
                        topic="bulk",
                        content=("database padding " + str(i) + " ") * 900,
                        importance=1.0,
                        confidence=1.0,
                    )
                )
            result = await memory_tools.execute(
                "infinitum_memory_search", '{"query": "database", "limit": 10}', runtime, None
            )
            assert len(result) <= memory_tools.MAX_RESULT_CHARS
            payload = json.loads(result)
            assert payload["truncated"] is True
        finally:
            await runtime.db.close()


# --- execute: infinitum_memory_get ------------------------------------------


async def test_get_unknown_id_returns_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            result = await memory_tools.execute(
                "infinitum_memory_get", '{"memory_id": "mem_does_not_exist"}', runtime, None
            )
            assert json.loads(result) == {"error": "not found"}
        finally:
            await runtime.db.close()


async def test_get_known_id_returns_full_payload():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            source_id = await add_event(runtime.db)
            memory = Memory(
                topic="database", content="PostgreSQL it is.", source_event_ids=[source_id]
            )
            await runtime.db.create_memory(memory)
            result = await memory_tools.execute(
                "infinitum_memory_get", json.dumps({"memory_id": memory.id}), runtime, None
            )
            payload = json.loads(result)
            assert payload["id"] == memory.id
            assert payload["content"] == "PostgreSQL it is."
            assert payload["topic"] == "database"
            assert payload["status"] == "active"
            assert payload["source_event_ids"] and payload["source_event_ids"][0].startswith("evt_")
        finally:
            await runtime.db.close()


async def test_search_non_numeric_limit_falls_back_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            await runtime.db.create_memory(Memory(topic="misc", content="A database fact."))
            result = await memory_tools.execute(
                "infinitum_memory_search", '{"query": "database", "limit": "abc"}', runtime, None
            )
            assert "results" in json.loads(result)
        finally:
            await runtime.db.close()


async def test_get_oversized_content_stays_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            await runtime.db.create_memory(Memory(topic="bulk", content="x" * 20000))
            memory_id = (await runtime.db.list_active_memories())[0].id
            result = await memory_tools.execute(
                "infinitum_memory_get", json.dumps({"memory_id": memory_id}), runtime, None
            )
            payload = json.loads(result)
            assert payload["truncated"] is True
            assert len(result) <= memory_tools.MAX_RESULT_CHARS
            assert payload["id"] == memory_id
        finally:
            await runtime.db.close()


# --- execute: model-caused errors never raise --------------------------------


async def test_malformed_arguments_returns_error_string():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            result = await memory_tools.execute(
                "infinitum_memory_search", "{not json", runtime, None
            )
            assert json.loads(result) == {"error": "invalid arguments"}
        finally:
            await runtime.db.close()


async def test_missing_keys_do_not_raise():
    with tempfile.TemporaryDirectory() as tmp:
        runtime = await make_runtime(tmp)
        try:
            got = await memory_tools.execute("infinitum_memory_get", "{}", runtime, None)
            assert json.loads(got) == {"error": "not found"}
            searched = await memory_tools.execute("infinitum_memory_search", "{}", runtime, None)
            assert "results" in json.loads(searched)
        finally:
            await runtime.db.close()


# --- classify_tool_calls ------------------------------------------------------


def call(name: str) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_classify_ours_only_is_true():
    ours = set(memory_tools.TOOL_NAMES)
    assert memory_tools.classify_tool_calls([call("infinitum_memory_search")], ours) is True


def test_classify_empty_is_false():
    assert memory_tools.classify_tool_calls([], set(memory_tools.TOOL_NAMES)) is False


def test_classify_mixed_is_false():
    ours = set(memory_tools.TOOL_NAMES)
    calls = [call("infinitum_memory_get"), call("web")]
    assert memory_tools.classify_tool_calls(calls, ours) is False


def test_classify_theirs_only_is_false():
    assert memory_tools.classify_tool_calls([call("web")], set(memory_tools.TOOL_NAMES)) is False


# --- injected_tool_names ------------------------------------------------------


def test_injected_names_no_client_tools():
    assert memory_tools.injected_tool_names(None) == list(memory_tools.TOOL_NAMES)


def test_injected_names_skips_client_name():
    client_tools = [{"type": "function", "function": {"name": "infinitum_memory_search"}}]
    assert memory_tools.injected_tool_names(client_tools) == ["infinitum_memory_get"]


def test_injected_names_ignores_malformed_entries():
    assert memory_tools.injected_tool_names(["junk", 42]) == list(memory_tools.TOOL_NAMES)


def test_build_tool_defs_order_and_schema():
    defs = memory_tools.build_tool_defs(list(memory_tools.TOOL_NAMES))
    assert [d["function"]["name"] for d in defs] == list(memory_tools.TOOL_NAMES)
    search = next(d for d in defs if d["function"]["name"] == "infinitum_memory_search")
    params = search["function"]["parameters"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["required"] == ["query"]
    assert params["properties"]["limit"]["type"] == "integer"
    get = next(d for d in defs if d["function"]["name"] == "infinitum_memory_get")
    assert get["function"]["parameters"]["required"] == ["memory_id"]


# --- reassemble_stream_tool_calls ---------------------------------------------


def test_reassemble_indexed_fragments():
    chunks = [
        {"index": 0, "id": "call_1", "name": "infinitum_memory_search", "arguments": '{\"query\"'},
        {"index": 0, "arguments": ': "db"}'},
        {"index": 1, "id": "call_2", "name": "infinitum_memory_get", "arguments": "{}"},
    ]
    calls = memory_tools.reassemble_stream_tool_calls(chunks)
    assert len(calls) == 2
    assert calls[0]["id"] == "call_1"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "infinitum_memory_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "db"}
    assert calls[1]["id"] == "call_2"


def test_reassemble_no_index_fallback():
    chunks = [
        {"id": "call_a", "name": "infinitum_memory_search", "arguments": '{"query":'},
        {"arguments": ' "x"}'},
        {"id": "call_b", "name": "infinitum_memory_get", "arguments": "{}"},
    ]
    calls = memory_tools.reassemble_stream_tool_calls(chunks)
    assert len(calls) == 2
    assert calls[0]["id"] == "call_a"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}
    assert calls[1]["function"]["name"] == "infinitum_memory_get"


# --- constants ----------------------------------------------------------------


def test_constants():
    assert memory_tools.MAX_RESULT_CHARS == 8000
    assert memory_tools.MAX_SEARCH_LIMIT == 50
    assert memory_tools.MAX_ITERATIONS == 4
