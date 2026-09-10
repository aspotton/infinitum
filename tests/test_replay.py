"""Live replay (`python -m benchmarks.replay`) driven against an in-process app.

The instance runs with learning disabled, so every learned-memory expectation
and probe WARNs BY DESIGN: replay never mutates beyond chat turns and never
drains the learning queue (that is the offline runner's job). The client
injection hook is the httpx.Client parameter on replay_scenarios; TestClient
IS an httpx.Client, so the test needs no real socket.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

from benchmarks.corpus import Expectation, Probe, Scenario, ScenarioContext, Turn
from benchmarks.replay import main as replay_main
from benchmarks.replay import replay_scenarios
from infinitum.app import create_app
from infinitum.config import AppConfig

_PRESEED = "Sentry is the error tracking tool for all services."


def _scenario(name: str = "replay-smoke") -> Scenario:
    return Scenario(
        name=name,
        description="Two-turn smoke scenario for the live replay command.",
        context=ScenarioContext(user_id="eval", project_id=name),
        turns=[
            Turn(
                user="We decided to use PostgreSQL for the billing service.",
                assistant="Noted - PostgreSQL for billing.",
                expect=Expectation(created=["PostgreSQL for the billing service"]),
            ),
            Turn(
                user="What database does the billing service use?",
                assistant="PostgreSQL.",
                probe=Probe(query="billing database choice", must_include=["billing"]),
            ),
        ],
    )


@contextlib.contextmanager
def _live_instance(tmp: str):
    """App with learning disabled and a canned foreground upstream."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-replay",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "replay-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/replay.db"
    cfg.learning.enabled = False
    cfg.upstream.passthrough_authorization = False
    app = create_app(cfg)
    with TestClient(app) as client:
        app.state.runtime.upstream.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        seeded = client.post(
            "/memory",
            json={
                "memory_type": "fact",
                "topic": "observability",
                "content": _PRESEED,
                "importance": 0.6,
                "confidence": 0.8,
            },
        )
        assert seeded.status_code == 200
        yield client


def test_replay_warns_by_design_and_exits_zero_when_not_strict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lines: list[str] = []
        with _live_instance(tmp) as client:
            code = replay_scenarios([_scenario()], client, emit=lines.append)
        assert code == 0

        warns = [line for line in lines if line.startswith("WARN")]
        assert len(warns) == 2, warns
        assert "replay-smoke turn 0" in warns[0] and "created" in warns[0]
        assert "PostgreSQL for the billing service" in warns[0]
        assert "replay-smoke turn 1" in warns[1] and "probe_must_include" in warns[1]

        # Turn-1 diff table renders the pre-seeded memory as a new (+) row;
        # turn 2 changes nothing, so its table says so.
        assert "replay-smoke turn 0:" in lines and "replay-smoke turn 1:" in lines
        assert any(
            line.lstrip().startswith("+") and _PRESEED[:20] in line for line in lines
        )
        assert any("no memory changes" in line for line in lines)

        # Both chat turns reached the server and were recorded as events.
        conn = sqlite3.connect(f"{tmp}/replay.db")
        events = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        conn.close()
        assert events >= 4


def test_replay_strict_exits_one_when_warnings_printed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lines: list[str] = []
        with _live_instance(tmp) as client:
            code = replay_scenarios([_scenario()], client, strict=True, emit=lines.append)
        assert code == 1
        assert len([line for line in lines if line.startswith("WARN")]) == 2


def test_replay_aborts_each_scenario_on_connect_error_without_traceback() -> None:
    lines: list[str] = []
    with httpx.Client(
        base_url="http://127.0.0.1:1", timeout=httpx.Timeout(5.0, connect=2.0)
    ) as client:
        code = replay_scenarios(
            [_scenario("alpha"), _scenario("beta")], client, emit=lines.append
        )
    errors = [line for line in lines if line.startswith("ERROR scenario")]
    assert len(errors) == 2, errors
    assert "alpha" in errors[0] and "beta" in errors[1]
    assert code == 0


def test_replay_cli_unknown_scenario_exits_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        replay_main(["--scenario", "no-such-scenario"])
    assert excinfo.value.code == 2
