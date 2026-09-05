"""Unit-tier tests for the read-only memory drill-down tools (Batch 1)."""

import json
import tempfile
from types import SimpleNamespace

import httpx
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
# Live-verified on vLLM/Qwen: stripping our defs is not enough, the automatic
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
    # Tool-choice-aware SSE fake (mirrors verified vLLM): honors
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
            assert response.headers["x-infinitum-memory-tool-rejects"] == "1"
            assert response.headers["x-infinitum-memory-tool-calls"] == "1"
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
