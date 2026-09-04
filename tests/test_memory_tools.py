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


def test_qa_h_four_round_cap_forwards_last_unrecorded():
    call = ("infinitum_memory_search", '{"query": "db"}', "call_h")
    replies = [_tool_reply([call]) for _ in range(memory_tools.MAX_ITERATIONS)]
    with tempfile.TemporaryDirectory() as tmp:
        app = _chat_app(tmp)
        with TestClient(app) as client:
            _seed_memory(client)
            upstream = _ScriptedUpstream(app.state.runtime, replies)
            response = _chat(client, "qa-h")
            assert response.status_code == 200
            assert response.json() == replies[-1]
            assert upstream.calls == memory_tools.MAX_ITERATIONS
            events = _events(client, "qa-h")
            assert not any(e["event_type"] == "message.assistant" for e in events)
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
