import json
import sqlite3
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig, RequestContextConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Event, Memory, RequestContext
from infinitum.request_context import RequestContextResolver, project_id_from_cwd
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient


def test_request_context_resolves_opencode_and_headroom_aliases():
    resolver = RequestContextResolver(RequestContextConfig())

    explicit = resolver.resolve(
        {
            "X-OpenCode-User": "adam",
            "X-OpenCode-Project": "memory-runtime",
            "X-OpenCode-Directory": "/home/adam/src/memory-runtime/../memory-runtime",
        }
    )
    assert explicit.user_id == "adam"
    assert explicit.project_id == "memory-runtime"
    assert explicit.cwd == "/home/adam/src/memory-runtime"
    assert explicit.project_derived_from_cwd is False

    derived = resolver.resolve(
        {
            "x-headroom-user-id": "adam",
            "x-headroom-cwd": "/home/adam/src/infinitum",
        }
    )
    assert derived.user_id == "adam"
    assert derived.project_id == project_id_from_cwd("/home/adam/src/infinitum")
    assert derived.project_derived_from_cwd is True


def test_request_context_header_priority_prefers_infinitum_headers():
    resolver = RequestContextResolver(RequestContextConfig())
    resolved = resolver.resolve(
        {
            "x-infinitum-user-id": "canonical-user",
            "x-context-user-id": "legacy-user",
            "x-opencode-user-id": "opencode-user",
            "x-headroom-user-id": "headroom-user",
            "x-infinitum-project-id": "canonical-project",
            "x-context-project-id": "legacy-project",
            "x-opencode-project": "opencode-project",
        }
    )
    assert resolved.user_id == "canonical-user"
    assert resolved.project_id == "canonical-project"


@pytest.mark.asyncio
async def test_retrieval_softly_boosts_same_project_without_filtering_global_memory():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.memory.minimum_retrieval_score = 0.0
        db = Database(cfg.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(cfg.embeddings)
        try:
            event_a = await db.add_event(
                Event(
                    session_id="s1",
                    user_id="adam",
                    project_id="project-a",
                    cwd="/src/a",
                    event_type="message.user",
                    role="user",
                    content="The API uses PostgreSQL.",
                )
            )
            event_b = await db.add_event(
                Event(
                    session_id="s2",
                    user_id="adam",
                    project_id="project-b",
                    cwd="/src/b",
                    event_type="message.user",
                    role="user",
                    content="The API uses PostgreSQL.",
                )
            )
            memory_a = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="database",
                    content="The API uses PostgreSQL.",
                    importance=0.7,
                    confidence=0.9,
                    source_event_ids=[event_a.id],
                )
            )
            memory_b = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="database",
                    content="The API uses PostgreSQL.",
                    importance=0.7,
                    confidence=0.9,
                    source_event_ids=[event_b.id],
                )
            )
            retriever = MemoryRetriever(db, embeddings, cfg)
            results = await retriever.search(
                "Which database does the API use?",
                limit=10,
                request_context=RequestContext(
                    user_id="adam", project_id="project-b", cwd="/src/b"
                ),
            )
            ids = [item.memory.id for item in results]
            assert memory_a.id in ids
            assert memory_b.id in ids
            assert ids.index(memory_b.id) < ids.index(memory_a.id)
            same_project = next(item for item in results if item.memory.id == memory_b.id)
            other_project = next(item for item in results if item.memory.id == memory_a.id)
            assert same_project.project_affinity_score == 1.0
            assert same_project.cwd_affinity_score == 1.0
            assert same_project.affinity_bonus > other_project.affinity_bonus
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_existing_v013_database_is_migrated_in_place():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                request_id TEXT,
                event_type TEXT NOT NULL,
                role TEXT,
                content TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE requests (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

        db = Database(path)
        await db.connect()
        try:
            event_cols = {
                row["name"] for row in await db.fetchall("PRAGMA table_info(events)")
            }
            request_cols = {
                row["name"] for row in await db.fetchall("PRAGMA table_info(requests)")
            }
            assert {"user_id", "project_id", "cwd"} <= event_cols
            assert {"user_id", "project_id", "cwd"} <= request_cols
        finally:
            await db.close()


@pytest.mark.parametrize("debug_header", ["x-infinitum-debug", "x-context-debug"])
def test_api_persists_context_and_strips_consumed_headers_upstream(debug_header):
    captured_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update({k.lower(): v for k, v in request.headers.items()})
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        cfg.upstream.passthrough_authorization = False
        app = create_app(cfg)
        with TestClient(app) as client:
            runtime = app.state.runtime
            runtime.upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "x-opencode-user": "adam",
                    "x-opencode-project": "infinitum",
                    "x-opencode-directory": "/home/adam/infinitum",
                    "x-session-id": "ses_from_opencode",
                    debug_header: "true",
                },
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert response.status_code == 200
            assert response.headers["x-infinitum-resolved-user-id"] == "adam"
            assert response.headers["x-infinitum-resolved-project-id"] == "infinitum"
            assert "x-opencode-user" not in captured_headers
            assert "x-opencode-project" not in captured_headers
            assert "x-opencode-directory" not in captured_headers
            assert debug_header not in captured_headers

            events = client.get(
                "/events", params={"user_id": "adam", "project_id": "infinitum"}
            ).json()
            assert len(events) >= 3
            assert all(event["session_id"] == "ses_from_opencode" for event in events)
            assert all(event["user_id"] == "adam" for event in events)
            assert all(event["project_id"] == "infinitum" for event in events)
            assert all(event["cwd"] == "/home/adam/infinitum" for event in events)


def test_upstream_can_emit_canonical_headroom_headers():
    cfg = AppConfig()
    cfg.request_context.forward_to_headroom = True
    upstream = UpstreamClient(cfg)
    try:
        resolved = RequestContext(
            user_id="adam",
            project_id="project-a",
            cwd="/home/adam/project-a",
        )
        headers = upstream.build_headers(
            {
                "x-opencode-user": "spoof-me-only-as-input",
                "x-headroom-user-id": "also-input",
            },
            resolved,
        )
        assert headers["x-headroom-user-id"] == "adam"
        assert headers["x-headroom-project-id"] == "project-a"
        assert headers["x-headroom-cwd"] == "/home/adam/project-a"
    finally:
        # AsyncClient close is exercised in runtime tests; no event loop here.
        pass
