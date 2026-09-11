"""Deterministic offline scenario runner for the evaluation quality loop.

Replays a conversation against the full in-process runtime (real durable
learning queue, scripted upstreams) and scores each turn. Determinism is defined
over the pass/fail records alone; ids/timestamps in the snapshot are report-only.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database

from .corpus import (
    Expectation,
    Probe,
    Scenario,
    ScenarioContext,
    Turn,
    demotion_violations,
    rank_pair_label,
    ranking_violations,
)

_EXTRACTION_MARKER = "Extract durable memories"
_POLL_SECONDS = 0.05
_DRAIN_TIMEOUT = 15.0


class RunnerError(Exception):
    """The run failed: a learner error, a stuck/failed job, or a drain timeout."""


class RunnerState:
    """Flag the scripted extractor sets on an unresolvable call.

    The extractor must never raise (that would trigger the worker's bounded
    retry storm), so it records the failure here and the drain aborts on it.
    """

    def __init__(self) -> None:
        self.message: str | None = None


@dataclass(frozen=True, slots=True)
class ExpectationRecord:
    turn_index: int
    kind: str
    target: str
    passed: bool


@dataclass(frozen=True)
class ScenarioResult:
    scenario_name: str
    records: tuple[ExpectationRecord, ...]
    snapshot: tuple[dict[str, Any], ...] = field(default=(), compare=False)
    total_tokens: int = field(default=0, compare=False)

    @property
    def ok(self) -> bool:
        return all(record.passed for record in self.records)


def _envelope(memories: list[dict[str, Any]]) -> dict[str, Any]:
    content = json.dumps({"memories": memories})
    message = {"role": "assistant", "content": content}
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _match_turn(turns: list[Turn], prompt: str) -> Turn | None:
    matches = [t for t in turns if t.user and t.user in prompt]
    return max(matches, key=lambda t: len(t.user)) if matches else None


def _candidate_json(spec: Any, supersedes: list[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "memory_type": spec.memory_type,
        "topic": spec.topic,
        "content": spec.content,
        "importance": spec.importance,
        "confidence": spec.confidence,
        "operation_hint": spec.operation,
        "supersedes_memory_ids": supersedes,
        "explicit_correction": spec.explicit_correction,
    }
    if spec.valid_from is not None:
        entry["valid_from"] = spec.valid_from
    if spec.valid_until is not None:
        entry["valid_until"] = spec.valid_until
    return entry


async def _resolve_supersedes(db: Database, substring: str, state: RunnerState) -> list[str]:
    active = await db.list_active_memories()
    hits = [m.id for m in active if substring in m.content]
    if len(hits) == 1:
        return hits
    if state.message is None:
        state.message = (
            f"supersedes_match {substring!r} matched {len(hits)} active memories "
            "(expected exactly one)"
        )
    return []


def _make_learn_fn(scenario: Scenario, db: Database, state: RunnerState):
    async def learn_fn(**kwargs: Any) -> dict[str, Any]:
        prompt = "\n".join(
            str(m.get("content", "")) for m in kwargs.get("messages", []) if isinstance(m, dict)
        )
        turn = _match_turn(scenario.turns, prompt) if _EXTRACTION_MARKER in prompt else None
        if turn is None:
            if state.message is None:
                state.message = (
                    "extraction prompt matched no scenario turn"
                    if _EXTRACTION_MARKER in prompt
                    else f"unexpected learning call, no extraction marker: {prompt[:160]!r}"
                )
            return _envelope([])
        memories: list[dict[str, Any]] = []
        for spec in turn.learn:
            supersedes: list[str] = []
            if spec.operation == "supersede" and spec.supersedes_match:
                supersedes = await _resolve_supersedes(db, spec.supersedes_match, state)
                if state.message is not None:
                    return _envelope([])
            memories.append(_candidate_json(spec, supersedes))
        return _envelope(memories)

    return learn_fn


def _make_fg_handler(fg: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "chatcmpl-eval",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "eval-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": fg["assistant"]},
                "finish_reason": "stop",
            }],
        })

    return handler


def _reader(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    return conn


def _done_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='done'").fetchone()["n"])


def _drain_jobs(
    conn: sqlite3.Connection, state: RunnerState, *, timeout: float = _DRAIN_TIMEOUT
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if state.message is not None:
            raise RunnerError(state.message)
        rows = conn.execute("SELECT status, last_error FROM jobs").fetchall()
        statuses = [r["status"] for r in rows]
        if "failed" in statuses:
            error = next(r["last_error"] for r in rows if r["status"] == "failed")
            raise RunnerError(f"learning job failed during drain: {error}")
        if not any(s in ("pending", "running") for s in statuses):
            return
        if time.monotonic() >= deadline:
            leftover = [s for s in statuses if s in ("pending", "running")]
            raise RunnerError(f"drain timed out after {timeout:g}s; leftover jobs: {leftover}")
        time.sleep(_POLL_SECONDS)


def _active_counts(conn: sqlite3.Connection) -> dict[str, int]:
    query = "SELECT id, observation_count FROM memories WHERE status='active'"
    return {row["id"]: int(row["observation_count"]) for row in conn.execute(query)}


def _eval_expectation(
    conn: sqlite3.Connection,
    turn_index: int,
    expect: Expectation,
    baseline_counts: dict[str, int],
) -> list[ExpectationRecord]:
    rows = conn.execute("SELECT id, content, status, observation_count FROM memories").fetchall()

    def _hits(sub: str, status: str) -> list[sqlite3.Row]:
        return [r for r in rows if r["status"] == status and sub in r["content"]]

    def _rec(kind: str, sub: str, passed: bool) -> ExpectationRecord:
        return ExpectationRecord(turn_index, kind, sub, passed)

    records: list[ExpectationRecord] = []
    for sub in expect.created:
        records.append(_rec("created", sub, bool(_hits(sub, "active"))))
    for sub in expect.ignored:
        records.append(_rec("ignored", sub, not _hits(sub, "active")))
    for sub in expect.superseded:
        records.append(_rec("superseded", sub, bool(_hits(sub, "superseded"))))
    for sub in expect.reinforced:
        hits = _hits(sub, "active")
        prior = baseline_counts.get(hits[0]["id"]) if len(hits) == 1 else None
        # Fail (not vacuously pass) unless the target existed at baseline and grew.
        grew = prior is not None and int(hits[0]["observation_count"]) > prior
        records.append(_rec("reinforced", sub, grew))
    return records


def _eval_probe(
    client: TestClient, headers: dict[str, str], turn_index: int, probe: Probe
) -> list[ExpectationRecord]:
    payload: dict[str, Any] = {"query": probe.query, "limit": 20}
    if probe.temporal_view != "current" or probe.as_of is not None:
        payload["temporal_view"] = probe.temporal_view
        if probe.as_of is not None:
            payload["as_of"] = probe.as_of
    response = client.post("/memory/search", json=payload, headers=headers)
    response.raise_for_status()
    items = response.json()
    contents = [item["memory"]["content"] for item in items]

    def _rec(kind: str, sub: str, passed: bool) -> ExpectationRecord:
        return ExpectationRecord(turn_index, kind, sub, passed)

    records: list[ExpectationRecord] = []
    for sub in probe.must_include:
        records.append(_rec("probe_must_include", sub, any(sub in c for c in contents)))
    for sub in probe.must_not_include:
        records.append(_rec("probe_must_not_include", sub, not any(sub in c for c in contents)))
    if probe.must_demote:
        # Same query through the same search path, all-view this time: only
        # the follow-up score being strictly higher proves the demotion.
        all_payload = {"query": probe.query, "limit": 20, "temporal_view": "all"}
        all_response = client.post("/memory/search", json=all_payload, headers=headers)
        all_response.raise_for_status()
        failures = demotion_violations(probe.must_demote, items, all_response.json())
        for sub in probe.must_demote:
            records.append(_rec("probe_must_demote", sub, sub not in failures))
    rank_failures = set(ranking_violations(probe.must_rank_below, items))
    for pair in probe.must_rank_below:
        label = rank_pair_label(pair)
        records.append(_rec("probe_must_rank_below", label, label not in rank_failures))
    return records


def _context_headers(context: ScenarioContext) -> dict[str, str]:
    return {
        "X-Infinitum-User-ID": context.user_id,
        "X-Infinitum-Project-ID": context.project_id,
    }


def _snapshot(conn: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
    query = (
        "SELECT id, content, status, created_at, updated_at, observation_count "
        "FROM memories ORDER BY id"
    )
    return tuple(dict(row) for row in conn.execute(query))


def _total_tokens(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(SUM(tokens), 0) AS n FROM request_memories").fetchone()
    return int(row["n"])


def run_scenario(scenario: Scenario) -> ScenarioResult:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/eval.db"
        cfg = AppConfig()
        cfg.memory.database_path = db_path
        cfg.learning.enabled = True
        cfg.learning.poll_interval_seconds = _POLL_SECONDS
        cfg.learning.topic_summaries = False
        cfg.upstream.passthrough_authorization = False

        app = create_app(cfg)
        state = RunnerState()
        fg = {"assistant": ""}
        headers = _context_headers(scenario.context)
        records: list[ExpectationRecord] = []

        with TestClient(app) as client:
            runtime = app.state.runtime
            runtime.upstream.client = httpx.AsyncClient(
                transport=httpx.MockTransport(_make_fg_handler(fg))
            )
            runtime.upstream.learning_chat_completion = AsyncMock(
                side_effect=_make_learn_fn(scenario, runtime.db, state)
            )
            conn = _reader(db_path)
            try:
                for turn_index, turn in enumerate(scenario.turns):
                    fg["assistant"] = turn.assistant
                    body = {
                        "model": "eval-model",
                        "messages": [{"role": "user", "content": turn.user}],
                        "stream": False,
                    }
                    response = client.post("/v1/chat/completions", json=body, headers=headers)
                    response.raise_for_status()
                    baseline = _done_count(conn)
                    counts0 = _active_counts(conn)
                    _drain_jobs(conn, state)
                    if _done_count(conn) <= baseline:
                        raise RunnerError(f"turn {turn_index}: no learning job terminal")
                    records.extend(_eval_expectation(conn, turn_index, turn.expect, counts0))
                    if turn.probe is not None:
                        records.extend(_eval_probe(client, headers, turn_index, turn.probe))
                snapshot = _snapshot(conn)
                total_tokens = _total_tokens(conn)
            finally:
                conn.close()

    return ScenarioResult(scenario.name, tuple(records), snapshot, total_tokens)
