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

from benchmarks.corpus import Expectation, Probe, RankPair, Scenario, ScenarioContext, Turn
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


def test_replay_model_override_and_default() -> None:
    """``model=`` replaces the turn-request model; default stays infinitum-replay."""

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.content:
            captured.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    def turn_models(client: httpx.Client, **kwargs: object) -> list[str]:
        captured.clear()
        replay_scenarios([_scenario("model-flag")], client, emit=lambda _: None, **kwargs)
        return [body["model"] for body in captured if "messages" in body]

    with httpx.Client(
        base_url="http://instance", transport=httpx.MockTransport(handler)
    ) as client:
        assert turn_models(client, model="qwen") == ["qwen", "qwen"]
        assert turn_models(client) == ["infinitum-replay", "infinitum-replay"]


def test_replay_cli_timeout_flag_builds_patient_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--timeout`` sets the client read timeout; connect stays capped at 5s."""

    captured: list[dict] = []

    class _StubClient:
        """Records construction kwargs; every request fails like a dead port."""

        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

        def __enter__(self) -> _StubClient:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def post(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("stub")

        def get(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("stub")

    monkeypatch.setattr(httpx, "Client", _StubClient)
    assert replay_main([]) == 0
    assert captured[0]["timeout"] == httpx.Timeout(120.0, connect=5.0)
    assert replay_main(["--timeout", "7"]) == 0
    assert captured[1]["timeout"] == httpx.Timeout(7.0, connect=5.0)


def _demote_scenario() -> Scenario:
    """One-turn scenario whose probe asserts must_demote on an expired row."""

    return Scenario(
        name="replay-demote",
        description="Probe asserts the current view demotes the expired audit row.",
        context=ScenarioContext(user_id="eval", project_id="replay-demote"),
        turns=[
            Turn(
                user="What ran the 2023 encryption audit?",
                assistant="VaultPress ran it, valid through 2024-06-30.",
                probe=Probe(query="2023 encryption audit", must_demote=["VaultPress"]),
            )
        ],
    )


def _demote_client(current: float, every: float, bodies: list[dict]) -> httpx.Client:
    """Client whose /memory/search returns one row at a view-dependent score."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/memory/search":
            return httpx.Response(200, json=[])
        body = json.loads(request.content)
        bodies.append(body)
        score = every if body.get("temporal_view") == "all" else current
        row = {"memory": {"content": "The 2023 encryption audit ran on VaultPress"}, "score": score}
        return httpx.Response(200, json=[row])

    return httpx.Client(base_url="http://instance", transport=httpx.MockTransport(handler))


def test_replay_must_demote_posts_all_view_followup_and_warns_when_scores_equal() -> None:
    """Inert factor (equal scores) => one WARN line and --strict exits 1."""
    bodies: list[dict] = []
    lines: list[str] = []
    with _demote_client(0.3, 0.3, bodies) as client:
        code = replay_scenarios(
            [_demote_scenario()], client, strict=True, emit=lines.append
        )
    # First POST omits temporal_view (implicit current), second is the "all" follow-up.
    assert [body.get("temporal_view") for body in bodies] == [None, "all"]
    warns = [line for line in lines if line.startswith("WARN")]
    assert len(warns) == 1, warns
    assert "probe_must_demote" in warns[0] and "VaultPress" in warns[0]
    assert code == 1


def test_replay_must_demote_passes_when_all_view_scores_higher() -> None:
    """X 0.70 demotion visible (all > current) => no WARN, exits 0 even strict."""
    bodies: list[dict] = []
    lines: list[str] = []
    with _demote_client(0.21, 0.3, bodies) as client:
        code = replay_scenarios(
            [_demote_scenario()], client, strict=True, emit=lines.append
        )
    assert len(bodies) == 2
    assert not [line for line in lines if line.startswith("WARN")]
    assert code == 0


def _rank_scenario() -> Scenario:
    """One-turn scenario whose probe asserts Memcached ranks below Redis."""
    return Scenario(
        name="replay-rank",
        description="Probe asserts result-list ordering in the probed view.",
        context=ScenarioContext(user_id="eval", project_id="replay-rank"),
        turns=[
            Turn(
                user="What runs the session store?",
                assistant="Redis now; Memcached before that.",
                probe=Probe(
                    query="session store",
                    must_rank_below=[RankPair(memory="Memcached", below="Redis")],
                ),
            )
        ],
    )


def _rank_client(order: list[str], bodies: list[dict]) -> httpx.Client:
    """Client whose /memory/search returns the given content order."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/memory/search":
            return httpx.Response(200, json=[])
        bodies.append(json.loads(request.content))
        rows = [
            {"memory": {"content": f"The session store ran on {name}"}, "score": 0.5 - i * 0.1}
            for i, name in enumerate(order)
        ]
        return httpx.Response(200, json=rows)

    return httpx.Client(base_url="http://instance", transport=httpx.MockTransport(handler))


def test_replay_must_rank_below_warns_when_order_is_wrong_and_skips_followup() -> None:
    """Memcached printed first => one WARN, strict exits 1, and NO all-view
    follow-up: ordering is proven by the probed view alone."""
    bodies: list[dict] = []
    lines: list[str] = []
    with _rank_client(["Memcached", "Redis"], bodies) as client:
        code = replay_scenarios(
            [_rank_scenario()], client, strict=True, emit=lines.append
        )
    assert len(bodies) == 1 and "temporal_view" not in bodies[0]
    warns = [line for line in lines if line.startswith("WARN")]
    assert len(warns) == 1, warns
    assert "probe_must_rank_below" in warns[0]
    assert code == 1


def test_replay_must_rank_below_passes_when_ordered() -> None:
    """Redis first, Memcached second => pair satisfied, no WARN, strict exits 0."""
    bodies: list[dict] = []
    lines: list[str] = []
    with _rank_client(["Redis", "Memcached"], bodies) as client:
        code = replay_scenarios(
            [_rank_scenario()], client, strict=True, emit=lines.append
        )
    assert len(bodies) == 1
    assert not [line for line in lines if line.startswith("WARN")]
    assert code == 0
