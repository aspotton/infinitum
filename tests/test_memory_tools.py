"""Unit-tier tests for the read-only memory drill-down tools (Batch 1)."""

import json
import tempfile
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from infinitum import memory_tools
from infinitum.app import create_app
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


# --- client_tool_names --------------------------------------------------------


def test_client_tool_names_none_and_empty():
    assert memory_tools.client_tool_names(None) == set()
    assert memory_tools.client_tool_names([]) == set()


def test_client_tool_names_collects_only_function_names():
    tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": 42}},
        {"function": {}},
        {"function": None},
        "junk",
        42,
        None,
        {"name": "flat_name"},
    ]
    assert memory_tools.client_tool_names(tools) == {"web_search"}


def test_injected_names_uses_client_tool_names():
    client_tools = [{"type": "function", "function": {"name": "infinitum_memory_get"}}]
    assert memory_tools.injected_tool_names(client_tools) == ["infinitum_memory_search"]


# --- is_rejectable_memory_name ------------------------------------------------


def test_reject_hallucinated_infinitum_name():
    ours = set(memory_tools.TOOL_NAMES)
    assert memory_tools.is_rejectable_memory_name("infinitum_retrieve", ours, set()) is True


def test_reject_exposed_names_are_false():
    ours = set(memory_tools.TOOL_NAMES)
    for name in memory_tools.TOOL_NAMES:
        assert memory_tools.is_rejectable_memory_name(name, ours, set()) is False


def test_reject_client_defined_infinitum_name_is_false():
    ours = set(memory_tools.TOOL_NAMES)
    client = {"infinitum_whatever"}
    assert memory_tools.is_rejectable_memory_name("infinitum_whatever", ours, client) is False


def test_reject_foreign_and_nameless_are_false():
    ours = set(memory_tools.TOOL_NAMES)
    for name in ("web_search", "", None, 123):
        assert memory_tools.is_rejectable_memory_name(name, ours, set()) is False


def test_reject_prefix_check_is_case_insensitive():
    ours = set(memory_tools.TOOL_NAMES)
    assert memory_tools.is_rejectable_memory_name("Infinitum_Retrieve", ours, set()) is True


# --- build_reject_result ------------------------------------------------------


def test_build_reject_result_round_trip():
    exposed = list(memory_tools.TOOL_NAMES)
    parsed = json.loads(memory_tools.build_reject_result("infinitum_retrieve", exposed))
    assert parsed == {
        "error": "unknown memory tool 'infinitum_retrieve'",
        "available_memory_tools": exposed,
        "hint": "answer from the results above, or call one of these tools",
    }


HINT_EXCLUSIVITY_CLAUSE = (
    " Those are the only memory tools that exist; use them for any memory"
    " lookup; never invent another memory tool name."
)
DEFS_COMPLETE_SET_CLAUSE = (
    " This is the complete set of memory tools: infinitum_memory_search"
    " and infinitum_memory_get; never call another memory tool name."
)


def test_tool_def_descriptions_carry_complete_set_clause():
    defs = memory_tools.build_tool_defs(list(memory_tools.TOOL_NAMES))
    assert len(defs) == 2
    for tool_def in defs:
        assert DEFS_COMPLETE_SET_CLAUSE in tool_def["function"]["description"]


def test_compiled_block_hint_carries_exclusivity_clause():
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "hint-excl")
            assert response.status_code == 200
            assert upstream.calls == 1
            contents = [str(m.get("content", "")) for m in upstream.bodies[0]["messages"]]
            assert any(HINT_EXCLUSIVITY_CLAUSE in c for c in contents)


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


# --- Batch 3: non-streaming server-side tool loop (QA A-R) ---------------------
#
# Idiom: swap runtime.upstream.client for a MockTransport that scripts canned
# upstream replies in order and records every forwarded request body.


def _completion(text: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _tool_reply(calls: list[tuple[str, str, str]], content: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for (name, arguments, call_id) in calls
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _chat_app(
    tmp: str,
    *,
    tools_enabled: bool = True,
    memory_enabled: bool = True,
    learning_enabled: bool = False,
) -> object:
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.learning.enabled = learning_enabled
    cfg.memory.enabled = memory_enabled
    cfg.memory.tools_enabled = tools_enabled
    cfg.upstream.passthrough_authorization = False
    return create_app(cfg)


class _ScriptedUpstream:
    """Canned-reply upstream; records each forwarded request body."""

    def __init__(self, runtime, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.bodies.append(json.loads(request.content))
            return httpx.Response(200, json=self.replies.pop(0))

        runtime.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @property
    def calls(self) -> int:
        return len(self.bodies)


def _seed_memory(client: TestClient) -> None:
    created = client.post(
        "/memory",
        json={
            "memory_type": "decision",
            "topic": "database",
            "content": "PostgreSQL 17 is the current database standard.",
            "importance": 1.0,
            "confidence": 1.0,
        },
    )
    assert created.status_code == 200


def _chat(
    client: TestClient, session: str, extra: dict | None = None, headers: dict | None = None
):
    body: dict = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What database standard do we use?"}],
    }
    if extra:
        body.update(extra)
    hdrs = {"X-Infinitum-Session-ID": session}
    if headers:
        hdrs.update(headers)
    return client.post("/v1/chat/completions", json=body, headers=hdrs)


def _events(client: TestClient, session: str) -> list[dict]:
    response = client.get(f"/events?session_id={session}&limit=1000")
    assert response.status_code == 200
    return response.json()


def test_qa_a_flag_off_request_untouched():
    reply = _completion("plain answer")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=False)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "qa-a")
            assert response.status_code == 200
            assert response.json() == reply
            assert upstream.calls == 1
            assert "tools" not in upstream.bodies[0]


def test_qa_b_search_roundtrip_records_final_once():
    search = ("infinitum_memory_search", '{"query": "database", "limit": 5}', "call_s")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime, [_tool_reply([search]), _completion("PostgreSQL 17.")]
            )
            response = _chat(client, "qa-b")
            assert response.json()["choices"][0]["message"]["content"] == "PostgreSQL 17."
            assert upstream.calls == 2
            second = upstream.bodies[1]
            assistant = [m for m in second["messages"] if m.get("tool_calls")]
            tools = [m for m in second["messages"] if m.get("role") == "tool"]
            assert len(assistant) == 1
            assert assistant[0]["role"] == "assistant"
            assert assistant[0]["tool_calls"][0]["id"] == "call_s"
            assert len(tools) == 1
            assert tools[0]["tool_call_id"] == "call_s"
            results = json.loads(tools[0]["content"])["results"]
            assert any("PostgreSQL" in r["content"] for r in results)
            for body in upstream.bodies:
                contents = [str(m.get("content", "")) for m in body["messages"]]
                assert any("<infinitum_memory>" in c for c in contents)
                names = [t["function"]["name"] for t in body["tools"]]
                assert names == list(memory_tools.TOOL_NAMES)
            hint = any("Deeper detail" in str(m.get("content", "")) for m in second["messages"])
            assert hint
            events = _events(client, "qa-b")
            assert sum(e["event_type"] == "message.assistant" for e in events) == 1
            tool_events = [e for e in events if e["event_type"] == "memory.tool_call"]
            assert len(tool_events) == 1
            meta = tool_events[0]["metadata"]
            assert meta["name"] == "infinitum_memory_search"
            assert meta["tool_call_id"] == "call_s"
            assert meta["result_chars"] == len(tools[0]["content"])
            assert meta["assistant_message"]["tool_calls"][0]["id"] == "call_s"


def test_qa_c_client_tool_call_forwarded_verbatim():
    client_tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    reply = _tool_reply([("get_weather", '{"city": "Oslo"}', "call_w")])
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "qa-c", extra={"tools": [client_tool]})
            assert response.status_code == 200
            assert response.json() == reply
            assert upstream.calls == 1
            names = [t["function"]["name"] for t in upstream.bodies[0]["tools"]]
            assert names == ["get_weather", *memory_tools.TOOL_NAMES]


def test_qa_d_mixed_tool_calls_terminal_not_looped():
    reply = _tool_reply(
        [
            ("infinitum_memory_search", '{"query": "db"}', "call_s"),
            ("get_weather", "{}", "call_w"),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "qa-d")
            assert response.json() == reply
            assert upstream.calls == 1


def test_qa_g_get_unknown_id_error_result_roundtrip():
    call = ("infinitum_memory_get", '{"memory_id": "mem_missing"}', "call_g")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime, [_tool_reply([call]), _completion("not in memory")]
            )
            response = _chat(client, "qa-g")
            assert response.json()["choices"][0]["message"]["content"] == "not in memory"
            assert upstream.calls == 2
            tools = [m for m in upstream.bodies[1]["messages"] if m.get("role") == "tool"]
            assert json.loads(tools[0]["content"]) == {"error": "not found"}
            events = _events(client, "qa-g")
            assert sum(e["event_type"] == "memory.tool_call" for e in events) == 1


def test_qa_h_four_round_cap_forces_terminal_answer():
    # Past the tool-round cap the proxy must answer, not forward the last
    # internal tool-call message (content=null == blank client response).
    call = ("infinitum_memory_search", '{"query": "db"}', "call_h")
    replies = [_tool_reply([call]) for _ in range(memory_tools.MAX_ITERATIONS)]
    replies.append(_completion("PostgreSQL 17 it is."))
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, replies)
            response = _chat(client, "qa-h")
            assert response.status_code == 200
            assert response.json() == replies[-1]
            assert upstream.calls == memory_tools.MAX_ITERATIONS + 1
            last_names = {t["function"]["name"] for t in upstream.bodies[-1].get("tools", [])}
            assert not last_names & set(memory_tools.TOOL_NAMES)
            events = _events(client, "qa-h")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert [e["content"] for e in assistants] == ["PostgreSQL 17 it is."]
            assert sum(e["event_type"] == "memory.tool_call" for e in events) == 4


def test_qa_i_client_owned_name_not_shadowed_or_looped():
    client_search = {
        "type": "function",
        "function": {"name": "infinitum_memory_search", "parameters": {}},
    }
    reply = _tool_reply([("infinitum_memory_search", '{"query": "x"}', "call_c")])
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "qa-i", extra={"tools": [client_search]})
            assert response.json() == reply
            assert upstream.calls == 1
            names = [t["function"]["name"] for t in upstream.bodies[0]["tools"]]
            assert names == ["infinitum_memory_search", "infinitum_memory_get"]


def test_qa_j_learning_enqueue_carries_final_text_only():
    call = ("infinitum_memory_search", '{"query": "db"}', "call_j")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, learning_enabled=True)
        with TestClient(app) as client:
            _seed_memory(client)
            enqueued: list[tuple[str, dict]] = []

            async def spy(job_type: str, payload: dict) -> str:
                enqueued.append((job_type, payload))
                return "job-spy"

            app.state.runtime.db.enqueue_job = spy
            _ScriptedUpstream(
                app.state.runtime,
                [_tool_reply([call]), _completion("FINAL ANSWER ONLY")],
            )
            response = _chat(client, "qa-j")
            assert response.json()["choices"][0]["message"]["content"] == "FINAL ANSWER ONLY"
            learn_jobs = [payload for kind, payload in enqueued if kind == "learn_interaction"]
            assert len(learn_jobs) == 1
            assert learn_jobs[0]["assistant_text"] == "FINAL ANSWER ONLY"


def test_qa_n_memory_off_header_injects_no_tools():
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "qa-n", headers={"X-Infinitum-Memory": "off"})
            assert response.status_code == 200
            assert upstream.calls == 1
            assert "tools" not in upstream.bodies[0]


def test_qa_p_malformed_arguments_error_result_no_500():
    call = ("infinitum_memory_search", '{"query": broken', "call_p")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime, [_tool_reply([call]), _completion("recovered")]
            )
            response = _chat(client, "qa-p")
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "recovered"
            assert upstream.calls == 2
            tools = [m for m in upstream.bodies[1]["messages"] if m.get("role") == "tool"]
            assert json.loads(tools[0]["content"]) == {"error": "invalid arguments"}


def test_qa_q_debug_header_counts_tool_rounds():
    call = ("infinitum_memory_search", '{"query": "db"}', "call_q")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime,
                [_tool_reply([call]), _tool_reply([("get_weather", "{}", "call_w")])],
            )
            response = _chat(client, "qa-q", headers={"X-Infinitum-Debug": "true"})
            assert response.status_code == 200
            assert response.headers["x-infinitum-memory-tool-calls"] == "1"
            assert upstream.calls == 2


def test_qa_r_memory_disabled_injects_no_tools():
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, memory_enabled=False)
        with TestClient(app) as client:
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "qa-r")
            assert response.status_code == 200
            assert upstream.calls == 1
            assert "tools" not in upstream.bodies[0]


# --- Cache-stable todo 1: static tool-def exposure (no compiled.text gate) ----


def test_static_tools_empty_memory_db_still_exposes_both_defs():
    # compiled.text is empty (no memories): tool defs must still be appended so
    # the tools region never flaps between turns.
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=True)
        with TestClient(app) as client:
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "static-empty")
            assert response.status_code == 200
            assert upstream.calls == 1
            names = [t["function"]["name"] for t in upstream.bodies[0]["tools"]]
            assert names == list(memory_tools.TOOL_NAMES)


def test_static_tools_flag_off_forwards_tools_byte_identical():
    client_tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    sent = [client_tool]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=False)
        with TestClient(app) as client:
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "static-off", extra={"tools": sent})
            assert response.status_code == 200
            assert upstream.calls == 1
            assert json.dumps(upstream.bodies[0]["tools"], sort_keys=True) == json.dumps(
                sent, sort_keys=True
            )


def test_static_tools_client_owned_name_not_duplicated():
    # Empty DB + static exposure: a client that shadows one of our names gets
    # only the non-shadowed def appended; no duplicate names in the tools array.
    client_search = {
        "type": "function",
        "function": {"name": "infinitum_memory_search", "parameters": {}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=True)
        with TestClient(app) as client:
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = _chat(client, "static-shadow", extra={"tools": [client_search]})
            assert response.status_code == 200
            names = [t["function"]["name"] for t in upstream.bodies[0]["tools"]]
            assert names == ["infinitum_memory_search", "infinitum_memory_get"]


def test_static_tools_byte_identical_across_memory_state_change():
    # Turn 1 with zero memories, turn 2 after one memory is created: the tools
    # region forwarded upstream is byte-identical across the two turns.
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=True)
        with TestClient(app) as client:
            upstream = _ScriptedUpstream(
                app.state.runtime, [_completion("plain"), _completion("plain")]
            )
            first = _chat(client, "static-stab-1")
            assert first.status_code == 200
            _seed_memory(client)
            second = _chat(client, "static-stab-2")
            assert second.status_code == 200
            assert upstream.calls == 2
            tools_a = json.dumps(upstream.bodies[0].get("tools"), sort_keys=True)
            tools_b = json.dumps(upstream.bodies[1].get("tools"), sort_keys=True)
            assert tools_a == tools_b
            assert tools_a != "null"


# --- Batch 4: streaming server-side tool loop (QA E, F, K, L, M, S) -----------
#
# SSE idiom (per tests/test_upstream_none_callback.py): MockTransport handler
# returning httpx.Response(200, content=async-iterator) so bytes arrive as
# network chunks and errors can be raised mid-consumption.


def _delta_event(delta: dict, finish_reason: str | None = None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
        ).encode()
        + b"\n\n"
    )


def _content_chunk(text: str) -> bytes:
    return _delta_event({"content": text})


def _tool_chunk(index: int, call_id: str | None, name: str | None, arguments: str) -> bytes:
    return _delta_event(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "function": {"name": name, "arguments": arguments},
                }
            ]
        }
    )


def _finish_chunk(reason: str) -> bytes:
    return _delta_event({}, finish_reason=reason)


_DONE = b"data: [DONE]\n\n"


class _SseUpstream:
    """Scripted SSE upstream; each reply is (network chunks, mid-stream error)."""

    def __init__(
        self, runtime, replies: list[tuple[list[bytes], Exception | None]]
    ) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.bodies.append(json.loads(request.content))
            chunks, error = self.replies.pop(0)

            async def body():
                for chunk in chunks:
                    yield chunk
                if error is not None:
                    raise error

            return httpx.Response(200, content=body())

        runtime.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @property
    def calls(self) -> int:
        return len(self.bodies)


def test_qa_e_stream_tool_round_suppressed_and_recorded_once():
    round1 = [
        _tool_chunk(0, "call_e", "infinitum_memory_search", '{"query": "database"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    round2 = [_content_chunk("PostgreSQL 17."), _finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(round1, None), (round2, None)])
            response = _chat(client, "qa-e", extra={"stream": True})
            assert response.status_code == 200
            body_text = response.content.decode()
            # Client sees ONLY round-2 bytes: no tool-call deltas, ends [DONE].
            assert body_text == b"".join(round2).decode()
            assert "tool_calls" not in body_text
            assert body_text.rstrip().endswith("data: [DONE]")
            assert upstream.calls == 2
            second = upstream.bodies[1]
            assistant = [m for m in second["messages"] if m.get("tool_calls")]
            assert len(assistant) == 1
            assert assistant[0]["role"] == "assistant"
            assert assistant[0]["tool_calls"][0]["id"] == "call_e"
            events = _events(client, "qa-e")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert len(assistants) == 1
            assert assistants[0]["content"] == "PostgreSQL 17."
            # stream_complete True proves the terminal recording saw the
            # complete bytes INCLUDING the [DONE] sentinel.
            assert assistants[0]["metadata"]["stream_complete"] is True
            tool_events = [e for e in events if e["event_type"] == "memory.tool_call"]
            assert len(tool_events) == 1
            meta = tool_events[0]["metadata"]
            assert meta["name"] == "infinitum_memory_search"
            assert meta["tool_call_id"] == "call_e"
            assert meta["assistant_message"]["role"] == "assistant"


def test_qa_f_content_first_stream_passthrough_single_call():
    stream = [_content_chunk("he"), _content_chunk("llo"), _finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(stream, None)])
            response = _chat(client, "qa-f", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(stream)
            assert upstream.calls == 1
            events = _events(client, "qa-f")
            assert sum(e["event_type"] == "message.assistant" for e in events) == 1
            assert not any(e["event_type"] == "memory.tool_call" for e in events)


def test_qa_k_midstream_failure_yields_502_unrecorded():
    partial = [_tool_chunk(0, "call_k", "infinitum_memory_search", '{"query": "db"}')]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(
                app.state.runtime, [(partial, httpx.ConnectError("boom"))]
            )
            response = _chat(client, "qa-k", extra={"stream": True})
            assert response.status_code == 502
            assert upstream.calls == 1
            events = _events(client, "qa-k")
            assert not any(e["event_type"] == "message.assistant" for e in events)


def test_qa_l_data_line_split_across_chunks_still_passthrough():
    # The data: line is split mid-JSON; the content delta straddles the split.
    chunk1 = b'data: {"choices":[{"delta":{"content":"hel'
    chunk2 = b'lo"}}],"finish_reason":null}\n\ndata: [DONE]\n\n'
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [([chunk1, chunk2], None)])
            response = _chat(client, "qa-l", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == chunk1 + chunk2
            assert upstream.calls == 1
            events = _events(client, "qa-l")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert len(assistants) == 1
            assert assistants[0]["content"] == "hello"
            assert assistants[0]["metadata"]["stream_complete"] is True


def test_qa_m_client_tool_stream_passes_through_intact():
    client_tool = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}},
    }
    stream = [
        _tool_chunk(0, "call_w", "get_weather", '{"city": "Oslo"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(stream, None)])
            response = _chat(client, "qa-m", extra={"stream": True, "tools": [client_tool]})
            assert response.status_code == 200
            assert response.content == b"".join(stream)
            assert '"get_weather"' in response.content.decode()
            assert upstream.calls == 1


def test_qa_t_stream_round_cap_forces_answer_not_empty_stream():
    # Static tool-def exposure made N consecutive suppress rounds reachable on
    # any request (an auto-parsing upstream may call our tools unprompted).
    # Past the cap the client must receive a real answer, never a blank reply:
    # the old code emitted a zero-byte SSE stream (final_iterator = iter(())).
    tool_rounds = [
        [
            _tool_chunk(0, f"call_t{i}", "infinitum_memory_search", '{"query": "db"}'),
            _finish_chunk("tool_calls"),
            _DONE,
        ]
        for i in range(memory_tools.MAX_ITERATIONS)
    ]
    final_round = [_content_chunk("PostgreSQL 17."), _finish_chunk("stop"), _DONE]
    replies = [(round_bytes, None) for round_bytes in tool_rounds]
    replies.append((final_round, None))
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, replies)
            response = _chat(client, "qa-t", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert "PostgreSQL 17." in text
            assert "tool_calls" not in text
            assert text.rstrip().endswith("data: [DONE]")
            assert upstream.calls == memory_tools.MAX_ITERATIONS + 1
            last_names = [t["function"]["name"] for t in upstream.bodies[-1].get("tools", [])]
            assert not set(last_names) & set(memory_tools.TOOL_NAMES)
            events = _events(client, "qa-t")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert len(assistants) == 1
            assert assistants[0]["content"] == "PostgreSQL 17."


# --- Forced terminal round: tool_choice:"none" + blank-answer synthesis --------
#
# Verified against a local OpenAI-compatible server: stripping our defs is not enough, the automatic
# tool-call parser re-emits our tool names and the old forced round forwarded a
# content:null tool_calls message (non-stream) or a blank SSE (stream).


def test_forced_round_sets_tool_choice_none():
    call = ("infinitum_memory_search", '{"query": "db"}', "call_fc1")
    replies = [_tool_reply([call]) for _ in range(memory_tools.MAX_ITERATIONS)]
    replies.append(_completion("PostgreSQL 17 it is."))
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, replies)
            response = _chat(client, "forced-tc")
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            assert response.json()["choices"][0]["finish_reason"] == "stop"
            assert message["content"] == "PostgreSQL 17 it is."
            assert "tool_calls" not in message
            forced_body = upstream.bodies[-1]
            assert forced_body["tool_choice"] == "none"
            forced_names = {t["function"]["name"] for t in forced_body.get("tools", [])}
            assert not forced_names & set(memory_tools.TOOL_NAMES)
            events = _events(client, "forced-tc")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert [e["content"] for e in assistants] == ["PostgreSQL 17 it is."]


def test_forced_round_synthesizes_when_tool_choice_ignored():
    # Server ignores tool_choice:"none": even the forced round comes back as
    # finish_reason=tool_calls with content=null. The proxy must synthesize an
    # answer from the already-gathered tool results, never forward the blank
    # message or a dangling tool_call.
    call = ("infinitum_memory_search", '{"query": "db"}', "call_fc2")
    replies = [_tool_reply([call]) for _ in range(memory_tools.MAX_ITERATIONS + 1)]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, replies)
            response = _chat(client, "forced-synth")
            assert response.status_code == 200
            payload = response.json()
            assert upstream.bodies[-1]["tool_choice"] == "none"
            choice = payload["choices"][0]
            assert choice["finish_reason"] == "stop"
            content = choice["message"]["content"]
            assert isinstance(content, str) and len(content) > 0
            assert "tool_calls" not in choice["message"]
            events = _events(client, "forced-synth")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert len(assistants) == 1
            assert len(assistants[0]["content"]) > 0


def test_forced_round_stream_not_blank():
    # Tool-choice-aware SSE fake (mirrors a server with an automatic tool parser): honors
    # tool_choice:"none" with a plain answer; otherwise its auto parser keeps
    # emitting our tool calls. The old forced round stripped defs but sent no
    # tool_choice, so the client received the stray tool-call stream.
    tool_stream = [
        _tool_chunk(0, "call_fc3", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    answer_stream = [_content_chunk("PostgreSQL 17."), _finish_chunk("stop"), _DONE]
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        chunks = answer_stream if request_body_forces_answer(bodies[-1]) else tool_stream

        async def stream_body():
            for chunk in chunks:
                yield chunk

        return httpx.Response(200, content=stream_body())

    def request_body_forces_answer(body: dict) -> bool:
        return body.get("tool_choice") == "none"

    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            app.state.runtime.upstream.client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            response = _chat(client, "forced-stream", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert "PostgreSQL 17." in text
            assert "tool_calls" not in text
            assert text.rstrip().endswith("data: [DONE]")
            assert bodies[-1]["tool_choice"] == "none"
            events = _events(client, "forced-stream")
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert [e["content"] for e in assistants] == ["PostgreSQL 17."]


def test_flag_off_body_unchanged():
    # tools_enabled=false: the forced-round code must never add a tool_choice
    # key (byte-identical flag-off invariant).
    client_tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    sent_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [client_tool],
    }
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, tools_enabled=False)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [_completion("plain")])
            response = client.post(
                "/v1/chat/completions",
                json=sent_body,
                headers={"X-Infinitum-Session-ID": "forced-off"},
            )
            assert response.status_code == 200
            assert upstream.calls == 1
            forwarded = upstream.bodies[0]
            assert "tool_choice" not in forwarded
            assert forwarded["tools"] == sent_body["tools"]


# --- Hallucination guard: reject-and-instruct in the non-stream tool loop ------

REJECT_D2 = {
    "error": "unknown memory tool 'infinitum_retrieve'",
    "available_memory_tools": ["infinitum_memory_get", "infinitum_memory_search"],
    "hint": "answer from the results above, or call one of these tools",
}
HALLUCINATED = ("infinitum_retrieve", '{"query": "db"}', "call_hal")


def _rejected_events(events: list[dict]) -> list[dict]:
    return [
        e
        for e in events
        if e["event_type"] == "memory.tool_call" and e["metadata"].get("rejected") is True
    ]


def _tool_messages(body: dict) -> list[dict]:
    return [m for m in body["messages"] if m.get("role") == "tool"]


def test_reject_hallucinated_retrieve_mid_loop():
    get_call = ("infinitum_memory_get", '{"memory_id": "mem_x"}', "call_r1")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp, learning_enabled=True)
        with TestClient(app) as client:
            _seed_memory(client)
            enqueued: list[tuple[str, dict]] = []

            async def spy(job_type: str, payload: dict) -> str:
                enqueued.append((job_type, payload))
                return "job-spy"

            app.state.runtime.db.enqueue_job = spy
            upstream = _ScriptedUpstream(
                app.state.runtime,
                [
                    _tool_reply([get_call]),
                    _tool_reply([HALLUCINATED]),
                    _completion("PostgreSQL 17."),
                ],
            )
            response = _chat(client, "reject-mid")
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "stop"
            assert choice["message"]["content"] == "PostgreSQL 17."
            assert "tool_calls" not in choice["message"]
            assert upstream.calls == 3
            reject_tools = [
                m
                for m in _tool_messages(upstream.bodies[2])
                if json.loads(m["content"]).get("error", "").startswith("unknown memory tool")
            ]
            assert len(reject_tools) == 1
            assert json.loads(reject_tools[0]["content"]) == REJECT_D2
            assert reject_tools[0]["tool_call_id"] == "call_hal"
            rejected = _rejected_events(_events(client, "reject-mid"))
            assert len(rejected) == 1
            assert rejected[0]["metadata"]["name"] == "infinitum_retrieve"
            assert rejected[0]["metadata"]["tool_call_id"] == "call_hal"
            assert json.loads(rejected[0]["metadata"]["result"]) == REJECT_D2
            learn_jobs = [payload for kind, payload in enqueued if kind == "learn_interaction"]
            assert len(learn_jobs) == 1


def test_reject_with_no_defs_this_round():
    search_call = ("infinitum_memory_search", '{"query": "db"}', "call_nd1")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime,
                [_tool_reply([search_call]), _tool_reply([HALLUCINATED]), _completion("answered")],
            )
            response = _chat(client, "reject-nodefs")
            assert response.json()["choices"][0]["message"]["content"] == "answered"
            assert upstream.calls == 3
            loop1_names = {t["function"]["name"] for t in upstream.bodies[0]["tools"]}
            assert loop1_names == set(memory_tools.TOOL_NAMES)
            assert json.loads(_tool_messages(upstream.bodies[2])[-1]["content"]) == REJECT_D2
            assert len(_rejected_events(_events(client, "reject-nodefs"))) == 1


def test_reject_sibling_executes():
    search_call = ("infinitum_memory_search", '{"query": "database"}', "call_sib1")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(
                app.state.runtime,
                [_tool_reply([search_call, HALLUCINATED]), _completion("sibling done")],
            )
            response = _chat(client, "reject-sibling")
            assert response.json()["choices"][0]["message"]["content"] == "sibling done"
            assert "tool_calls" not in response.json()["choices"][0]["message"]
            assert upstream.calls == 2
            tools = _tool_messages(upstream.bodies[1])
            assert len(tools) == 2
            assert json.loads(tools[0]["content"])["results"]
            assert json.loads(tools[1]["content"]) == REJECT_D2
            events = _events(client, "reject-sibling")
            tool_events = [e for e in events if e["event_type"] == "memory.tool_call"]
            assert len(tool_events) == 2
            assert len(_rejected_events(events)) == 1


def test_client_shadowed_memory_name_forwards():
    client_search = {
        "type": "function",
        "function": {"name": "infinitum_memory_search", "parameters": {}},
    }
    reply = _tool_reply([("infinitum_memory_search", '{"query": "x"}', "call_shadow")])
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "reject-shadow", extra={"tools": [client_search]})
            assert response.json() == reply
            assert upstream.calls == 1
            assert not any(
                e["event_type"] == "memory.tool_call" for e in _events(client, "reject-shadow")
            )


def test_reject_mixed_with_client_tool_forwards():
    client_tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    reply = _tool_reply(
        [
            ("infinitum_memory_search", '{"query": "db"}', "call_mx1"),
            ("get_weather", '{"city": "Oslo"}', "call_mx2"),
            HALLUCINATED,
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "reject-mixed", extra={"tools": [client_tool]})
            assert response.json() == reply
            assert upstream.calls == 1
            assert not any(
                e["event_type"] == "memory.tool_call" for e in _events(client, "reject-mixed")
            )


def test_reject_skipped_on_forced_round():
    replies = [
        _tool_reply([("infinitum_retrieve", '{"query": "db"}', f"call_f{i}")])
        for i in range(memory_tools.MAX_ITERATIONS + 1)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, replies)
            response = _chat(client, "reject-forced")
            assert upstream.calls == memory_tools.MAX_ITERATIONS + 1
            assert upstream.bodies[-1]["tool_choice"] == "none"
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "stop"
            assert choice["message"]["content"]
            assert "tool_calls" not in choice["message"]
            rejected = _rejected_events(_events(client, "reject-forced"))
            assert len(rejected) == memory_tools.MAX_ITERATIONS


def test_memory_off_header_forwards_hallucination_verbatim():
    reply = _tool_reply([HALLUCINATED])
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, [reply])
            response = _chat(client, "reject-off", headers={"X-Infinitum-Memory": "off"})
            assert response.json() == reply
            assert upstream.calls == 1
            assert "tools" not in upstream.bodies[0]
            assert not any(
                e["event_type"] == "memory.tool_call" for e in _events(client, "reject-off")
            )


def test_debug_header_reject_count_present():
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            _ScriptedUpstream(
                app.state.runtime, [_tool_reply([HALLUCINATED]), _completion("done")]
            )
            response = _chat(client, "reject-dbg", headers={"X-Infinitum-Debug": "true"})
            assert response.status_code == 200
            assert response.headers["x-infinitum-memory-tool-rejects"] == "1"
            assert response.headers["x-infinitum-memory-tool-calls"] == "1"


def test_debug_header_reject_count_absent_without_rejects():
    search_call = ("infinitum_memory_search", '{"query": "db"}', "call_nr")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            _ScriptedUpstream(
                app.state.runtime, [_tool_reply([search_call]), _completion("done")]
            )
            response = _chat(client, "reject-nodbg", headers={"X-Infinitum-Debug": "true"})
            assert response.status_code == 200
            assert response.headers["x-infinitum-memory-tool-calls"] == "1"
            assert "x-infinitum-memory-tool-rejects" not in response.headers


# --- Hallucination guard: streaming classifier reject path ----------------------


def test_stream_reject_no_leak():
    hallucinated = _tool_chunk(0, "call_sh1", "infinitum_retrieve", '{"query": "db"}')
    round1 = [hallucinated, _finish_chunk("tool_calls"), _DONE]
    round2 = [_content_chunk("PostgreSQL 17."), _finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(round1, None), (round2, None)])
            response = _chat(
                client, "stream-reject", extra={"stream": True},
                headers={"X-Infinitum-Debug": "true"},
            )
            assert response.status_code == 200
            text = response.content.decode()
            # Zero bytes of the hallucinated name reach the client.
            assert "infinitum_retrieve" not in text
            assert "tool_calls" not in text
            assert "PostgreSQL 17." in text
            # Streaming responses carry the debug counters as trailing SSE
            # comments (rounds 2+ run inside the response; headers impossible).
            assert ": x-infinitum-memory-tool-rejects 1" in text
            assert ": x-infinitum-memory-tool-calls 1" in text
            assert "x-infinitum-memory-tool-calls" not in response.headers
            assert upstream.calls == 2
            second = upstream.bodies[1]
            assistant = [m for m in second["messages"] if m.get("tool_calls")]
            assert len(assistant) == 1
            assert assistant[0]["role"] == "assistant"
            assert assistant[0]["tool_calls"][0]["id"] == "call_sh1"
            reject_tools = [
                m
                for m in _tool_messages(second)
                if json.loads(m["content"]).get("error", "").startswith("unknown memory tool")
            ]
            assert len(reject_tools) == 1
            assert json.loads(reject_tools[0]["content"]) == REJECT_D2
            assert reject_tools[0]["tool_call_id"] == "call_sh1"
            rejected = _rejected_events(_events(client, "stream-reject"))
            assert len(rejected) == 1
            assert rejected[0]["metadata"]["name"] == "infinitum_retrieve"
            assert rejected[0]["metadata"]["tool_call_id"] == "call_sh1"


def test_stream_reject_case_variant():
    # Case-variant hallucinations must not leak: Infinitum_Retrieve is still an
    # infinitum_-prefixed name the client never defined.
    hallucinated = _tool_chunk(0, "call_sh2", "Infinitum_Retrieve", '{"query": "db"}')
    round1 = [hallucinated, _finish_chunk("tool_calls"), _DONE]
    round2 = [_content_chunk("PostgreSQL 17."), _finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(round1, None), (round2, None)])
            response = _chat(client, "stream-reject-case", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert "retrieve" not in text.lower()
            assert "tool_calls" not in text
            assert "PostgreSQL 17." in text
            assert upstream.calls == 2
            reject_tools = [
                m
                for m in _tool_messages(upstream.bodies[1])
                if json.loads(m["content"]).get("error", "").startswith("unknown memory tool")
            ]
            assert len(reject_tools) == 1
            payload = json.loads(reject_tools[0]["content"])
            assert payload["error"] == "unknown memory tool 'Infinitum_Retrieve'"
            assert payload["available_memory_tools"] == sorted(memory_tools.TOOL_NAMES)
            rejected = _rejected_events(_events(client, "stream-reject-case"))
            assert len(rejected) == 1
            assert rejected[0]["metadata"]["name"] == "Infinitum_Retrieve"


def test_stream_reject_mixed_foreign_replays():
    # Rejectable + genuinely foreign name in one stream: the whole stream is
    # passthrough/replayed verbatim (current contract pinned, not improved).
    chunks = [
        _tool_chunk(0, "call_sh3", "infinitum_retrieve", '{"query": "db"}'),
        _tool_chunk(1, "call_sh4", "get_weather", '{"city": "Oslo"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(chunks, None)])
            response = _chat(client, "stream-reject-mixed", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(chunks)
            assert b"infinitum_retrieve" in response.content
            assert upstream.calls == 1
            assert not any(
                e["event_type"] == "memory.tool_call"
                for e in _events(client, "stream-reject-mixed")
            )


def test_stream_upstream_4xx_represented_verbatim():
    # Round 2 opens INSIDE the response generator; an HTTPStatusError with zero
    # bytes forwarded is re-presented as a plain Response (prefetched first
    # fragment), not raised through the stream and not recorded.
    round1 = [
        _tool_chunk(0, "call_4xx", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    bodies: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)

            async def handler(request: httpx.Request) -> httpx.Response:
                bodies.append(json.loads(request.content))
                if len(bodies) == 1:

                    async def round1_body():
                        for chunk in round1:
                            yield chunk

                    return httpx.Response(200, content=round1_body())
                return httpx.Response(429, content=b'{"error":{"message":"rate"}}')

            app.state.runtime.upstream.client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            response = _chat(client, "stream-4xx", extra={"stream": True})
            assert response.status_code == 429
            assert response.content == b'{"error":{"message":"rate"}}'
            assert len(bodies) == 2
            events = _events(client, "stream-4xx")
            assert not any(e["event_type"] == "message.assistant" for e in events)


def test_stream_error_after_forward_yields_sse_error():
    # Round 1 (in-body decision) suppresses silently; round 2 streams its
    # reasoning under live mode, then the iterator dies. Client bytes exist, so
    # the failure surfaces as one in-stream SSE error event and nothing records.
    round1 = [
        _tool_chunk(0, "call_err", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    reasoning = _delta_event({"reasoning": "pondering the database"})
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(
                app.state.runtime,
                [(round1, None), ([reasoning], httpx.ConnectError("boom"))],
            )
            response = _chat(client, "stream-err", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert "pondering the database" in text
            assert "tool_calls" not in text
            assert 'data: {"error"' in text
            assert text.rstrip().endswith("data: [DONE]")
            assert upstream.calls == 2
            events = _events(client, "stream-err")
            assert not any(e["event_type"] == "message.assistant" for e in events)


# --- StreamClassifier reasoning awareness + tee-mode consume (todo 2) ---------
#
# Direct-unit tier: no HTTP, no app. Chunk builders above (_delta_event et al.)
# produce `data: {json}\n\n` so every event is a data line + a blank line.


_OURS = set(memory_tools.TOOL_NAMES)
_REASONING_FIELDS = ("reasoning", "reasoning_content")


def _reasoning_chunk(text: str, field: str = "reasoning") -> bytes:
    return _delta_event({field: text})


def _role_chunk() -> bytes:
    return _delta_event({"role": "assistant"})


def _tee_classifier(**overrides) -> memory_tools.StreamClassifier:
    kwargs: dict = {
        "reasoning_fields": _REASONING_FIELDS,
        "tee_forward_enabled": True,
    }
    kwargs.update(overrides)
    return memory_tools.StreamClassifier(_OURS, **kwargs)


def test_consume_classifier_reasoning_only_forwards_with_split_chunks():
    clf = _tee_classifier()
    line = _reasoning_chunk("deep thought")
    # First half of the line: incomplete, nothing can be forwarded yet.
    assert clf.consume(line[:20]) == b""
    # Completing the line forwards it verbatim (data line + blank).
    assert clf.consume(line[20:]) == line
    assert clf._reasoning_seen
    assert clf.consume(_DONE) == b""  # [DONE] freezes, holds itself
    assert clf.forwarded
    assert clf.flush_held() == _DONE


def test_consume_classifier_reasoning_prefix_flushes_held_role_lines_once():
    # A role line held before reasoning was seen rides the prefix flush and is
    # never double-sent: forwarded bytes form a byte prefix of the raw stream.
    chunks = [_role_chunk(), _reasoning_chunk("think"), _reasoning_chunk("ing more")]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == b""  # role line held: reasoning not seen yet
    assert outs[1] == chunks[0] + chunks[1]  # role + first reasoning at once
    assert outs[2] == chunks[2]
    assert clf.forwarded


def test_consume_classifier_reasoning_then_tool_call_suppresses_without_leak():
    tool = _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}')
    chunks = [_role_chunk(), _reasoning_chunk("plan"), tool, _finish_chunk("tool_calls"), _DONE]
    clf = _tee_classifier()
    forwarded = b"".join(clf.consume(c) for c in chunks)
    # Leak-thinking-keep-loop: reasoning streamed, every tool byte held.
    assert b"plan" in forwarded
    assert b"infinitum_memory_search" not in forwarded  # zero-leak lock
    assert clf.finish() == "suppress"
    assert clf.flush_held() == tool + _finish_chunk("tool_calls") + _DONE
    # The loop still executes the call: reassembly sees the full call.
    calls = memory_tools.reassemble_stream_tool_calls(clf.calls)
    assert calls[0]["function"]["name"] == "infinitum_memory_search"
    assert b"infinitum_memory_search" in clf.collected_bytes()


def test_consume_classifier_reasoning_content_only_flushes_full_raw():
    chunks = [_content_chunk("hello"), _content_chunk(" world"), _finish_chunk("stop"), _DONE]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == chunks[0]  # first content line decides: flush everything
    assert outs[1:] == chunks[1:]  # raw thereafter, byte-identical
    assert clf.finish() == "passthrough"
    assert clf.flush_held() == b""


def test_consume_classifier_reasoning_freeze_then_content_is_byte_identical():
    tool = _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}')
    chunks = [
        _role_chunk(),
        _reasoning_chunk("plan"),
        tool,
        _finish_chunk("tool_calls"),
        _content_chunk("changed my mind"),
        _finish_chunk("stop"),
        _DONE,
    ]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == b"" and outs[1] == chunks[0] + chunks[1]  # hold, then tee
    assert outs[2] == b"" and outs[3] == b""  # frozen: tool + finish lines held
    assert b"".join(outs) == b"".join(chunks)  # flush + raw == exact full round
    assert clf.finish() == "passthrough"


def test_consume_classifier_reasoning_freeze_then_foreign_is_byte_identical():
    chunks = [
        _reasoning_chunk("plan"),
        _tool_chunk(0, "call_f", "get_weather", '{"city": "Oslo"}'),
        _content_chunk("sunny"),
    ]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == chunks[0]
    # A foreign name decides passthrough the moment it is scanned (existing
    # feed semantics): the whole held round flushes in order on that chunk.
    assert outs[1] == chunks[1]
    assert outs[2] == chunks[2]  # raw thereafter
    assert b"".join(outs) == b"".join(chunks)
    assert clf.finish() == "passthrough"


def test_consume_classifier_reasoning_unparseable_line_freezes_then_content():
    corrupt = b"data: {not json at all\n\n"
    chunks = [_reasoning_chunk("think"), corrupt, _content_chunk("ok")]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == chunks[0]
    assert outs[1] == b""  # unparseable data line freezes conservatively
    assert outs[2] == chunks[1] + chunks[2]
    assert b"".join(outs) == b"".join(chunks)


def test_consume_classifier_reasoning_finish_reason_freezes():
    chunks = [_reasoning_chunk("think"), _finish_chunk("stop"), _DONE]
    clf = _tee_classifier()
    outs = [clf.consume(c) for c in chunks]
    assert outs[0] == chunks[0]
    assert outs[1] == b"" and outs[2] == b""
    assert clf.flush_held() == chunks[1] + chunks[2]
    assert clf.finish() == "replay"  # no calls, no content: terminal replay


def test_consume_classifier_reasoning_done_freezes():
    chunks = [_reasoning_chunk("think"), _DONE]
    clf = _tee_classifier()
    assert clf.consume(chunks[0]) == chunks[0]
    assert clf.consume(chunks[1]) == b""
    assert clf.flush_held() == _DONE
    assert clf.finish() == "replay"


def test_consume_classifier_reasoning_buffered_tee_disabled_holds_all():
    tool = _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}')
    chunks = [_reasoning_chunk("think"), tool, _finish_chunk("tool_calls"), _DONE]
    clf = _tee_classifier(tee_forward_enabled=False)
    assert [clf.consume(c) for c in chunks] == [b"", b"", b"", b""]
    assert clf._reasoning_seen  # tracked, but never releases bytes
    assert not clf.forwarded
    assert clf.finish() == "suppress"
    assert clf.flush_held() == b"".join(chunks)


def test_consume_classifier_reasoning_custom_field_detected():
    chunks = [
        _reasoning_chunk("mine", field="my_reasoning"),
        _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    clf = _tee_classifier(reasoning_fields=("my_reasoning",))
    forwarded = b"".join(clf.consume(c) for c in chunks)
    assert b"mine" in forwarded
    assert clf._reasoning_seen
    assert clf.finish() == "suppress"


def test_consume_classifier_reasoning_empty_string_values_not_seen():
    chunks = [
        _reasoning_chunk(""),
        _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    clf = _tee_classifier()
    assert [clf.consume(c) for c in chunks] == [b""] * 4
    assert not clf._reasoning_seen
    assert not clf.forwarded
    assert clf.finish() == "suppress"


def test_consume_classifier_reasoning_disjoint_fields_forward_nothing():
    # Configured fields absent from the stream: nothing forwards, silent suppress.
    chunks = [
        _reasoning_chunk("hidden", field="my_reasoning"),
        _tool_chunk(0, "call_1", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    clf = _tee_classifier()
    assert [clf.consume(c) for c in chunks] == [b""] * 4
    assert not clf.forwarded
    assert clf.finish() == "suppress"


def test_consume_classifier_reasoning_collected_reassemble_parity_with_feed():
    tool = _tool_chunk(0, "call_9", "infinitum_memory_search", '{"query": "db"')
    args = _tool_chunk(0, None, None, ', "limit": 5}')
    chunks = [
        _reasoning_chunk("think"),
        tool,
        args,
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    tee = _tee_classifier()
    for chunk in chunks:
        tee.consume(chunk)
    plain = memory_tools.StreamClassifier(_OURS)
    for chunk in chunks:
        plain.feed(chunk)
    assert tee.collected_bytes() == b"".join(chunks)
    assert tee.collected_bytes() == plain.collected_bytes()
    assert memory_tools.reassemble_stream_tool_calls(tee.calls) == (
        memory_tools.reassemble_stream_tool_calls(plain.calls)
    )
    (call,) = memory_tools.reassemble_stream_tool_calls(tee.calls)
    assert call["id"] == "call_9"
    assert call["function"]["arguments"] == '{"query": "db", "limit": 5}'


def test_qa_s_query_from_messages_truncates_tool_blobs():
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app):
            query = app.state.runtime.compiler.query_from_messages(
                [
                    {"role": "user", "content": "which database"},
                    {"role": "tool", "tool_call_id": "c1", "content": "x" * 8192},
                ]
            )
            assert "x" * 500 in query
            assert "x" * 501 not in query


# --- Live/buffered streaming integration matrix (todo 4) ------------------------
#
# Async tests are plain async defs (collected via asyncio_mode="auto").
#
# Pins the two-phase route end to end: Phase A (round-1 decision in the route
# body) never forwards pre-decision bytes in either mode; Phase B (attempts 2+)
# is where "live" tees visible reasoning lines and "buffered" holds everything
# until the round decision. The plan's scenario-1 phrasing ("client sees
# round-1 reasoning") is physically impossible under Phase A, so the compatible
# construction — same adaptation todo 3 made for t3 — is call 1 = suppressed
# tool round, call 2 = the round whose reasoning reaches the client. The
# observable rule is identical: reasoning of a round that turns terminal
# streams live; suppressed rounds leak nothing.
#
# Async scenarios drive the ASGI app DIRECTLY instead of via httpx's
# ASGITransport: ASGITransport (httpx 0.28.1, measured) awaits the whole app
# call and then hands back ONE buffered fragment, which erases both fragment
# ordering and arrival timing — the plan's fragment-count fallback (>=4) is
# therefore unusable. The raw call records each http.response.body send with a
# monotonic timestamp, exactly what a real ASGI server does, so liveness is
# pinned by WHEN bytes hit the wire, not merely by which bytes exist.

_THINK_MARKER = b"incremental-live-think"
_FINAL_MARKER = b"incremental-final-answer"


async def _stream_runtime(tmp: str):
    """App + runtime wired for raw ASGI driving (lifespan is never entered,
    so build_runtime is called manually and the worker stays stopped; mirrors
    tests/test_learning_defer.py._proxy_app)."""
    from infinitum.runtime import build_runtime

    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.memory.tools_enabled = True
    cfg.learning.enabled = False
    cfg.upstream.passthrough_authorization = False
    app = create_app(cfg)
    rt = await build_runtime(cfg)
    app.state.runtime = rt
    return app, rt


async def _asgi_stream(app, body: dict, session: str, disconnect_after: float | None = None):
    """POST the chat route through the raw ASGI surface; return (status, [(t, frag)]).

    Each http.response.body send is timestamped with time.monotonic() at send
    time. When disconnect_after is set, http.disconnect is delivered after that
    many seconds — the signal a real ASGI server sends when the client goes
    away mid-stream (receive() first hands over the request body once, then
    parks until the disconnect event fires).
    """
    import asyncio
    import time

    payload = json.dumps({**body, "stream": True}).encode()
    t0 = time.monotonic()
    frags: list[tuple[float, bytes]] = []
    start: dict[str, int] = {}
    disconnect = asyncio.Event()
    gave_body = False

    async def receive() -> dict:
        nonlocal gave_body
        if not gave_body:
            gave_body = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.wait_for(disconnect.wait(), 30)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        kind = message.get("type")
        if kind == "http.response.start":
            start["status"] = message["status"]
        elif kind == "http.response.body" and message.get("body"):
            frags.append((time.monotonic() - t0, message["body"]))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"infinitum.test"),
            (b"content-type", b"application/json"),
            (b"x-infinitum-session-id", session.encode()),
        ],
        "client": ("test", 1234),
        "server": ("infinitum.test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    if disconnect_after is not None:
        await asyncio.sleep(disconnect_after)
        disconnect.set()
    await asyncio.wait_for(task, 10)
    return start.get("status"), frags


def test_stream_live_reasoning_streams_and_tool_round_stays_suppressed():
    # Scenario 1: live default, think->our-tool->answer. Round 1 (our tool
    # call) is suppressed silently in the route body; round 2 reasons live and
    # answers. Client bytes == raw round 2: round-2 reasoning text present,
    # every tool byte (name/args/tool_calls/round-1 [DONE]) absent.
    round1 = [
        _reasoning_chunk("suppressed-round-hidden-think"),
        _tool_chunk(0, "call_live1", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    round2 = [
        _reasoning_chunk("client-visible-live-think"),
        _content_chunk("PostgreSQL 17."),
        _finish_chunk("stop"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(round1, None), (round2, None)])
            response = _chat(client, "live-matrix-1", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert response.content == b"".join(round2)
            assert "client-visible-live-think" in text
            assert "PostgreSQL 17." in text
            assert "suppressed-round-hidden-think" not in text
            assert "infinitum_memory_search" not in text
            assert "call_live1" not in text
            assert "tool_calls" not in text
            # Only round 2's sentinel reaches the client.
            assert text.count("[DONE]") == 1
            assert upstream.calls == 2
            events = _events(client, "live-matrix-1")
            assert sum(e["event_type"] == "memory.tool_call" for e in events) == 1
            assistants = [e for e in events if e["event_type"] == "message.assistant"]
            assert len(assistants) == 1
            # The assistant event is the final answer ONLY: no reasoning text
            # in content and none smuggled into metadata.
            assert assistants[0]["content"] == "PostgreSQL 17."
            assert "client-visible-live-think" not in json.dumps(assistants[0])


@pytest.mark.asyncio
async def test_stream_live_reasoning_arrives_before_final_content():
    # Scenario 2 (INCREMENTALITY PROOF — the feature's raison d'être; end-state
    # byte tests cannot prove it). Timing-probe design: the plan's
    # fragment-count fallback was measured unusable (ASGITransport delivers
    # fragments=1, batching the whole call), so the ASGI app is driven directly
    # and every http.response.body send is timestamped. Round 2 yields the
    # reasoning line, sleeps 0.3s, then answers. Live tee: the reasoning bytes
    # hit the wire ~0.3s BEFORE the final-content bytes. Regression (tee off =>
    # Phase B replays the round as one blob at end-of-round): both markers
    # first appear in the same final fragment and the delta collapses to ~0
    # — red. The ordering assert (reasoning byte-offset before content
    # offset) holds in either design; the monotonic-delta assert (>=0.15s
    # margin against the 0.3s sleep) is the liveness pin.
    import asyncio

    round1 = [
        _tool_chunk(0, "call_live2", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _stream_runtime(tmp)
        calls: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            if len(calls) == 1:

                async def round1_body():
                    for chunk in round1:
                        yield chunk

                return httpx.Response(200, content=round1_body())

            async def round2_body():
                yield _reasoning_chunk("incremental-live-think")
                await asyncio.sleep(0.3)
                yield _content_chunk("incremental-final-answer")
                yield _finish_chunk("stop")
                yield _DONE

            return httpx.Response(200, content=round2_body())

        rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            status, frags = await _asgi_stream(
                app,
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "What database standard do we use?"}],
                },
                "live-incremental",
            )
            assert status == 200
            joined = b"".join(frag for _, frag in frags)
            # Suppressed round 1 still leaks zero tool bytes.
            assert b"tool_calls" not in joined and b"call_live2" not in joined
            # Byte-offset ordering within the received stream.
            assert joined.index(_THINK_MARKER) < joined.index(_FINAL_MARKER)
            # Timing: first fragment whose cumulative stream contains each marker.
            seen_at: dict[bytes, float] = {}
            cumulative = b""
            for t, frag in frags:
                cumulative += frag
                for marker in (_THINK_MARKER, _FINAL_MARKER):
                    if marker not in seen_at and marker in cumulative:
                        seen_at[marker] = t
            assert seen_at[_FINAL_MARKER] - seen_at[_THINK_MARKER] >= 0.15
            assert len(calls) == 2
            # Counter drains after the completed stream (scenario: drain check).
            assert rt.active_requests.value == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_stream_reasoning_free_stream_byte_identical_across_modes(mode):
    # Scenario 3: an identical scripted stream in which NO configured
    # reasoning field appears anywhere (our tool chunk + final content, no
    # reasoning deltas) produces byte-identical client bytes under both modes:
    # stream_reasoning only governs reasoning tee, nothing else.
    round1 = [
        _tool_chunk(0, "call_parity", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    round2 = [_content_chunk("he"), _content_chunk("llo"), _finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            app.state.runtime.config.memory.stream_reasoning = mode
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(round1, None), (round2, None)])
            response = _chat(client, f"parity-{mode}", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(round2)
            assert upstream.calls == 2
            assistants = [
                e
                for e in _events(client, f"parity-{mode}")
                if e["event_type"] == "message.assistant"
            ]
            assert len(assistants) == 1
            assert assistants[0]["content"] == "hello"


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_stream_disjoint_reasoning_fields_stays_silent(mode):
    # Scenario 4: the stream carries reasoning_content values but the config
    # names only my_reasoning. Neither the Phase-A round nor a Phase-B round
    # can see them, so both think->tool rounds stay fully silent (today's
    # behavior) and the client bytes are byte-identical in both modes.
    think = _reasoning_chunk("hidden-by-config-think", field="reasoning_content")
    tool = _tool_chunk(
        0, "call_disjoint", "infinitum_memory_search", '{"query": "db"}'
    )
    suppressed = [think, tool, _finish_chunk("tool_calls"), _DONE]
    answer = [
        _content_chunk("answer after disjoint rounds"),
        _finish_chunk("stop"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            mem = app.state.runtime.config.memory
            mem.stream_reasoning = mode
            mem.reasoning_delta_fields = ["my_reasoning"]
            _seed_memory(client)
            upstream = _SseUpstream(
                app.state.runtime,
                [(list(suppressed), None), (list(suppressed), None), (answer, None)],
            )
            response = _chat(client, f"disjoint-{mode}", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(answer)
            text = response.content.decode()
            assert "hidden-by-config-think" not in text
            assert "tool_calls" not in text
            assert upstream.calls == 3


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_stream_buffered_replay_is_byte_identical_to_live_tee(mode):
    # Scenario 5: the suppressed round's reasoning stays silent in BOTH modes
    # (leak-free guarantee), and the terminal round — reasoning as terminal
    # content — yields byte-identical client bytes whether live teed it out or
    # buffered replayed it at end of round.
    suppressed = [
        _reasoning_chunk("suppressed-round-hidden-think"),
        _tool_chunk(0, "call_silent", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    terminal = [
        _reasoning_chunk("terminal-round-visible-think"),
        _content_chunk("PostgreSQL 17."),
        _finish_chunk("stop"),
        _DONE,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            app.state.runtime.config.memory.stream_reasoning = mode
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(suppressed, None), (terminal, None)])
            response = _chat(client, f"replay-parity-{mode}", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(terminal)
            text = response.content.decode()
            assert "suppressed-round-hidden-think" not in text
            assert "terminal-round-visible-think" in text
            assert "tool_calls" not in text
            assert upstream.calls == 2
            assistants = [
                e
                for e in _events(client, f"replay-parity-{mode}")
                if e["event_type"] == "message.assistant"
            ]
            assert assistants[0]["content"] == "PostgreSQL 17."


def test_stream_reasoning_only_round_forwarded_once_without_duplication():
    # Scenario 6: a reasoning-only round (no content, no tool calls) is a
    # terminal replay under the default live mode. The raw round must reach the
    # client EXACTLY once — thinking forwarded, finish/[DONE] flushed at end,
    # no synthesized bytes anywhere. Byte equality against the raw stream plus
    # a marker-count pin the no-duplication contract. Parity variant: today's
    # no-thinking equivalent [_finish_chunk("stop"), _DONE] must produce the
    # identical assistant-event shape; completed()/_record_completion gate the
    # learn-job on learning_enabled AND stream_complete only, never on content,
    # so both variants enqueue identically under any learning config (with
    # learning off here, neither does).
    reasoning_round = [
        _reasoning_chunk("only-round-think", field="reasoning_content"),
        _finish_chunk("stop"),
        _DONE,
    ]
    empty_round = [_finish_chunk("stop"), _DONE]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _SseUpstream(app.state.runtime, [(reasoning_round, None)])
            response = _chat(client, "reasoning-only", extra={"stream": True})
            assert response.status_code == 200
            assert response.content == b"".join(reasoning_round)
            assert response.content.decode().count("only-round-think") == 1
            assert upstream.calls == 1
            assistants = [
                e
                for e in _events(client, "reasoning-only")
                if e["event_type"] == "message.assistant"
            ]
            assert len(assistants) == 1
            assert assistants[0]["content"] == ""
            assert "tool_call_chunks" not in assistants[0]["metadata"]
            assert assistants[0]["metadata"]["stream_complete"] is True

            _SseUpstream(app.state.runtime, [(empty_round, None)])
            response2 = _chat(client, "reasoning-empty", extra={"stream": True})
            assert response2.content == b"".join(empty_round)
            assistants2 = [
                e
                for e in _events(client, "reasoning-empty")
                if e["event_type"] == "message.assistant"
            ]
            assert len(assistants2) == 1
            assert assistants2[0]["content"] == assistants[0]["content"]
            assert assistants2[0]["metadata"] == assistants[0]["metadata"]


def test_stream_buffered_midround_error_after_content_yields_sse_error():
    # Scenario 7a: buffered equivalent of t3 (which pinned the live shape).
    # Round 1 suppresses; round 2's content line triggers the passthrough
    # flush — consume() releases held bytes at the DECISION regardless of tee
    # mode — so once the iterator then dies, bytes are already on the wire and
    # the failure must surface as one in-stream SSE error event, unrecorded.
    # Open failures with zero forwarded bytes stay t2/qa_k territory, and
    # Phase A is mode-agnostic, so round-1 error parity holds in both modes.
    round1 = [
        _tool_chunk(0, "call_buferr", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    visible = _content_chunk("visible before the wire died")
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            app.state.runtime.config.memory.stream_reasoning = "buffered"
            _seed_memory(client)
            upstream = _SseUpstream(
                app.state.runtime, [(round1, None), ([visible], httpx.ConnectError("boom"))]
            )
            response = _chat(client, "buf-mid-err", extra={"stream": True})
            assert response.status_code == 200
            text = response.content.decode()
            assert "visible before the wire died" in text
            assert "tool_calls" not in text
            assert 'data: {"error"' in text
            assert text.rstrip().endswith("data: [DONE]")
            assert upstream.calls == 2
            events = _events(client, "buf-mid-err")
            assert not any(e["event_type"] == "message.assistant" for e in events)


def test_stream_buffered_round2_open_4xx_verbatim():
    # Scenario 7b: buffered parity with t2 — round 2 opens with an HTTP 4xx
    # inside the Phase-B generator; the route's prefetch converts it into the
    # same plain verbatim Response a round-1 4xx produces (the
    # _VerbatimResponse/prefetch path is mode-agnostic) and records nothing.
    round1 = [
        _tool_chunk(0, "call_buf4xx", "infinitum_memory_search", '{"query": "db"}'),
        _finish_chunk("tool_calls"),
        _DONE,
    ]
    bodies: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            app.state.runtime.config.memory.stream_reasoning = "buffered"
            _seed_memory(client)

            async def handler(request: httpx.Request) -> httpx.Response:
                bodies.append(json.loads(request.content))
                if len(bodies) == 1:

                    async def round1_body():
                        for chunk in round1:
                            yield chunk

                    return httpx.Response(200, content=round1_body())
                return httpx.Response(429, content=b'{"error":{"message":"limited"}}')

            app.state.runtime.upstream.client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            response = _chat(client, "buf-open-4xx", extra={"stream": True})
            assert response.status_code == 429
            assert response.content == b'{"error":{"message":"limited"}}'
            assert len(bodies) == 2
            events = _events(client, "buf-open-4xx")
            assert not any(e["event_type"] == "message.assistant" for e in events)


@pytest.mark.asyncio
async def test_stream_disconnect_drains_counter():
    # Scenario 8 (+9 counter drain): client disconnect mid-terminal-round.
    # WHICH assert (the plan allows exactly one of two): counter drain only.
    # Reason: the in-process raw-ASGI disconnect CANCELS the streaming task,
    # and the `await completed(...)` in the response generator's finally is
    # itself cancelled before the partial assistant event lands (observed on
    # this starlette), so a mid-round partial event is NOT guaranteeable from
    # inside the process. The static guarantee lives in the code: _counted's
    # finally decrements the counter under GeneratorExit/CancelledError alike,
    # and _terminal_stream/_rounds_stream finallys await completed() exactly
    # once for a terminal round — under a real ASGI server (uvicorn closes
    # the body iterator with aclose() instead of cancelling) that records the
    # partial with stream_complete False.
    import asyncio

    with tempfile.TemporaryDirectory() as tmp:
        app, rt = await _stream_runtime(tmp)

        async def slow_body():
            yield _content_chunk("Partial answer ")
            await asyncio.sleep(0.4)
            yield _content_chunk("never delivered tail")
            yield _finish_chunk("stop")
            yield _DONE

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=slow_body())

        rt.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            status, frags = await _asgi_stream(
                app,
                {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                "disc-parity",
                disconnect_after=0.15,
            )
            assert status == 200
            joined = b"".join(frag for _, frag in frags)
            assert b"Partial answer " in joined
            # The stream was genuinely mid-round at close (tail lands at 0.4s).
            assert b"never delivered tail" not in joined
            assert rt.active_requests.value == 0
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://infinitum.test"
            ) as client:
                health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["active_requests"] == 0
        finally:
            await rt.upstream.client.aclose()
            await rt.db.close()
