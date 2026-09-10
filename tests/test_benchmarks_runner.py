"""Tests for the deterministic offline scenario runner (todo 3).

The runner drives the full in-process app (foreground proxy + durable learning
queue) with a scripted upstream and a scripted extractor, then evaluates
per-turn learning expectations and retrieval probes. Every Python command runs
from the repo root so the `benchmarks` namespace resolves.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from benchmarks.corpus import Scenario
from benchmarks.runner import RunnerError, RunnerState, ScenarioResult, _drain_jobs, run_scenario


def _inline_scenario() -> Scenario:
    """A 2-turn scenario: one created fact + one ignored temp fact, then a probe."""

    return Scenario.model_validate(
        {
            "name": "inline-runner",
            "description": "inline two-turn runner determinism scenario",
            "context": {"user_id": "eval", "project_id": "inline-runner"},
            "turns": [
                {
                    "user": (
                        "Our standard database is Postgres 17 for all new services. "
                        "Also just today I'm testing a connection pooling experiment on staging."
                    ),
                    "assistant": (
                        "Got it - Postgres 17 is the standard for new services; "
                        "I'll ignore today's pooling experiment noise."
                    ),
                    "learn": [
                        {
                            "operation": "new",
                            "memory_type": "fact",
                            "topic": "database",
                            "content": "Standard database is Postgres 17",
                            "importance": 0.7,
                            "confidence": 0.9,
                        }
                    ],
                    "expect": {
                        "created": ["Postgres 17"],
                        "ignored": ["connection pooling experiment"],
                    },
                },
                {
                    "user": "What database do we standardize new services on?",
                    "assistant": "You standardize new services on Postgres 17.",
                    "learn": [],
                    "probe": {
                        "query": "standard database new services",
                        "must_include": ["Postgres 17"],
                        "must_not_include": ["connection pooling experiment"],
                    },
                },
            ],
        }
    )


def _bad_supersede_scenario() -> Scenario:
    """Turn 2 supersedes a substring that never exists -> runner must fail fast."""

    return Scenario.model_validate(
        {
            "name": "bad-supersede",
            "description": "supersedes_match resolves against nothing",
            "context": {"user_id": "eval", "project_id": "bad-supersede"},
            "turns": [
                {
                    "user": "We cache sessions in Redis for now.",
                    "assistant": "Noted.",
                    "learn": [
                        {
                            "operation": "new",
                            "memory_type": "decision",
                            "topic": "cache",
                            "content": "Sessions cached in Redis",
                            "importance": 0.6,
                            "confidence": 0.8,
                        }
                    ],
                    "expect": {"created": ["Sessions cached in Redis"]},
                },
                {
                    "user": "Actually let's move session caching to Memcached.",
                    "assistant": "Updated.",
                    "learn": [
                        {
                            "operation": "supersede",
                            "memory_type": "decision",
                            "topic": "cache",
                            "content": "Sessions cached in Memcached",
                            "importance": 0.6,
                            "confidence": 0.9,
                            "explicit_correction": True,
                            "supersedes_match": "this phrase never exists anywhere",
                        }
                    ],
                    "expect": {},
                },
            ],
        }
    )


def _jobs_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.isolation_level = None  # autocommit: every SELECT sees the latest WAL snapshot
    conn.row_factory = sqlite3.Row
    return conn


def _seed_jobs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, job_type TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', payload_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL, run_after TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
        "last_error TEXT, locked_at TEXT)"
    )


# ---------------------------------------------------------------------------


def test_run_scenario_is_deterministic_across_two_runs() -> None:
    scenario = _inline_scenario()

    first = run_scenario(scenario)
    second = run_scenario(scenario)

    # The pass/fail records carry the whole determinism contract (no ids, no
    # wall-clock timestamps), so the two runs must be record-for-record equal.
    assert first.scenario_name == second.scenario_name
    assert first.records == second.records
    assert first == second
    assert first.ok and second.ok
    # Every expectation record passed on both runs.
    assert all(r.passed for r in first.records)

    # The REPORT-ONLY snapshot does carry uuid4 ids + timestamps, which are the
    # very things excluded from equality - prove they differ across runs while
    # the results still compare equal.
    first_ids = {m["id"] for m in first.snapshot}
    second_ids = {m["id"] for m in second.snapshot}
    assert first_ids and second_ids
    assert first_ids != second_ids


def test_run_scenario_records_expected_outcomes() -> None:
    result = run_scenario(_inline_scenario())
    kinds = {(r.kind, r.passed) for r in result.records}
    assert ("created", True) in kinds
    assert ("ignored", True) in kinds
    assert ("probe_must_include", True) in kinds
    assert ("probe_must_not_include", True) in kinds


def test_supersedes_match_nothing_fails_fast_under_20s() -> None:
    started = time.monotonic()
    with pytest.raises(RunnerError) as excinfo:
        run_scenario(_bad_supersede_scenario())
    elapsed = time.monotonic() - started

    assert "this phrase never exists anywhere" in str(excinfo.value)
    assert elapsed < 20.0, "fail-fast must avoid the learner retry storm"


def test_drain_fails_on_failed_job_not_vacuous_pass() -> None:
    conn = _jobs_conn(":memory:")
    _seed_jobs_table(conn)
    conn.execute(
        "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after, last_error) "
        "VALUES ('job_1', 'learn_interaction', 'failed', '{}', '2026-01-01T00:00:00+00:00', "
        "'2026-01-01T00:00:00+00:00', 'boom')"
    )
    with pytest.raises(RunnerError, match="failed"):
        _drain_jobs(conn, RunnerState(), timeout=1.0)


def test_drain_times_out_on_never_claimed_pending_job() -> None:
    conn = _jobs_conn(":memory:")
    _seed_jobs_table(conn)
    conn.execute(
        "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after) "
        "VALUES ('job_1', 'learn_interaction', 'pending', '{}', '2026-01-01T00:00:00+00:00', "
        "'2099-01-01T00:00:00+00:00')"
    )
    started = time.monotonic()
    with pytest.raises(RunnerError, match="timed out"):
        _drain_jobs(conn, RunnerState(), timeout=0.2)
    assert time.monotonic() - started < 5.0


def test_drain_aborts_on_error_flag_set_by_extractor() -> None:
    conn = _jobs_conn(":memory:")
    _seed_jobs_table(conn)
    conn.execute(
        "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after) "
        "VALUES ('job_1', 'learn_interaction', 'pending', '{}', '2026-01-01T00:00:00+00:00', "
        "'2099-01-01T00:00:00+00:00')"
    )
    state = RunnerState()
    state.message = "extractor flagged a bad prompt"
    with pytest.raises(RunnerError, match="bad prompt"):
        _drain_jobs(conn, state, timeout=10.0)


def _mk_result(passed: bool, snapshot_id: str, tokens: int) -> ScenarioResult:
    from benchmarks.runner import ExpectationRecord

    return ScenarioResult(
        scenario_name="a",
        records=(ExpectationRecord(0, "created", "fact one", passed),),
        snapshot=({"id": snapshot_id, "content": "fact one", "status": "active"},),
        total_tokens=tokens,
    )


def test_scenario_result_equality_ignores_snapshot_and_tokens() -> None:
    assert _mk_result(True, "mem_1", 10) == _mk_result(True, "mem_2", 999)


def test_scenario_result_inequality_tracks_pass_fail_records() -> None:
    assert _mk_result(True, "mem_1", 10) != _mk_result(False, "mem_1", 10)
