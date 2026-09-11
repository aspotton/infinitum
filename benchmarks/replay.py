"""Live corpus replay against a running Infinitum instance.

Read-only except for chat turns: POSTs each scripted user message, prints the
per-turn memory diff (new/changed keyed on id -> updated_at), then evaluates
turn expectations and retrieval probes SOFT - mismatches print WARN lines and
never raise. Connect/HTTP errors abort one scenario with a clear message and
continue the rest.

Learning on a live instance is asynchronous, so a WARN here is a soft signal,
not a verdict; deterministic scoring lives in ``python -m benchmarks.run``.
Re-running replay against the same instance is safe: after the first turn of
each scenario prints the store's existing memories as ``+`` rows, later turns
print ``(no memory changes)`` unless the live learner actually wrote memory.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from .corpus import (
    Expectation,
    Probe,
    Scenario,
    demotion_violations,
    load_scenarios,
    rank_pair_label,
    ranking_violations,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8788"
_REPLAY_MODEL = "infinitum-replay"
_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_CONTENT_SNIPPET = 60


def _context_headers(scenario: Scenario) -> dict[str, str]:
    return {"X-Infinitum-User-ID": "eval", "X-Infinitum-Project-ID": scenario.name}


def _post_turn(client: httpx.Client, scenario: Scenario, user: str, model: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": user}],
        },
        headers=_context_headers(scenario),
    )
    response.raise_for_status()


def _get_json(client: httpx.Client, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.get(path, params=params)
    response.raise_for_status()
    results: list[dict[str, Any]] = response.json()
    return results


def _active_memories(client: httpx.Client) -> list[dict[str, Any]]:
    return _get_json(client, "/memory", {"limit": 1000, "status": "active"})


def _brief(memory: dict[str, Any]) -> str:
    snippet = memory["content"][:_CONTENT_SNIPPET]
    return f"{memory['id']} [{memory['memory_type']}/{memory['topic']}] {snippet}"


def _print_diff(
    prev: dict[str, dict[str, Any]],
    current: list[dict[str, Any]],
    emit: Callable[[str], None],
) -> None:
    rows = []
    for memory in current:
        old = prev.get(memory["id"])
        if old is None:
            rows.append(f"  + {_brief(memory)}")
        elif old["updated_at"] != memory["updated_at"]:
            rows.append(f"  ~ {_brief(memory)}")
    emit("\n".join(rows) if rows else "  (no memory changes)")


class _Warner:
    """Counts and prints soft expectation mismatches; never raises."""

    def __init__(self, scenario_name: str, turn_index: int, emit: Callable[[str], None]) -> None:
        self._name = scenario_name
        self._turn = turn_index
        self._emit = emit
        self.count = 0

    def warn(self, kind: str, substring: str, detail: str) -> None:
        self.count += 1
        self._emit(f"WARN {self._name} turn {self._turn} {kind} {substring!r}: {detail}")


def _eval_expectation(
    warner: _Warner,
    expect: Expectation,
    active: list[dict[str, Any]],
    prev: dict[str, dict[str, Any]],
    client: httpx.Client,
) -> None:
    for substring in expect.created:
        if not any(substring in memory["content"] for memory in active):
            warner.warn("created", substring, "no active memory contains it")
    for substring in expect.ignored:
        if any(substring in memory["content"] for memory in active):
            warner.warn("ignored", substring, "found among active memories")
    for substring in expect.reinforced:
        hits = [memory for memory in active if substring in memory["content"]]
        old = prev.get(hits[0]["id"]) if len(hits) == 1 else None
        if old is None:
            warner.warn("reinforced", substring, "no pre-existing active memory grew")
        elif hits[0]["observation_count"] <= old["observation_count"]:
            warner.warn("reinforced", substring, "observation_count did not grow")
    if expect.superseded:
        # Extra read-only listing: the active-only view cannot show superseded rows.
        every = _get_json(client, "/memory", {"limit": 1000})
        for substring in expect.superseded:
            if not any(
                substring in memory["content"] and memory["status"] == "superseded"
                for memory in every
            ):
                warner.warn("superseded", substring, "no superseded memory contains it")


def _eval_probe(
    warner: _Warner, probe: Probe, client: httpx.Client, headers: dict[str, str]
) -> None:
    body: dict[str, Any] = {"query": probe.query, "limit": 20}
    if probe.temporal_view != "current" or probe.as_of is not None:
        body["temporal_view"] = probe.temporal_view
        if probe.as_of is not None:
            body["as_of"] = probe.as_of
    response = client.post("/memory/search", json=body, headers=headers)
    response.raise_for_status()
    items = response.json()
    contents = [item["memory"]["content"] for item in items]
    for substring in probe.must_include:
        if not any(substring in content for content in contents):
            warner.warn("probe_must_include", substring, "not in search results")
    for substring in probe.must_not_include:
        if any(substring in content for content in contents):
            warner.warn("probe_must_not_include", substring, "found in search results")
    if probe.must_demote:
        # Control search: only the all-view score being strictly higher than
        # the probed (current-view) score proves the temporal demotion.
        all_body = {"query": probe.query, "limit": 20, "temporal_view": "all"}
        all_response = client.post("/memory/search", json=all_body, headers=headers)
        all_response.raise_for_status()
        failures = demotion_violations(probe.must_demote, items, all_response.json())
        for substring in probe.must_demote:
            if substring in failures:
                warner.warn("probe_must_demote", substring, failures[substring])
    rank_failures = set(ranking_violations(probe.must_rank_below, items))
    for pair in probe.must_rank_below:
        label = rank_pair_label(pair)
        if label in rank_failures:
            warner.warn(
                "probe_must_rank_below",
                label,
                "not ordered below its peer in the probed view",
            )


def _run_scenario(
    scenario: Scenario, client: httpx.Client, emit: Callable[[str], None], model: str
) -> int:
    headers = _context_headers(scenario)
    prev: dict[str, dict[str, Any]] = {}
    warnings = 0
    for turn_index, turn in enumerate(scenario.turns):
        _post_turn(client, scenario, turn.user, model)
        current = _active_memories(client)
        emit(f"{scenario.name} turn {turn_index}:")
        _print_diff(prev, current, emit)
        warner = _Warner(scenario.name, turn_index, emit)
        _eval_expectation(warner, turn.expect, current, prev, client)
        if turn.probe is not None:
            _eval_probe(warner, turn.probe, client, headers)
        warnings += warner.count
        prev = {memory["id"]: memory for memory in current}
    return warnings


def replay_scenarios(
    scenarios: Iterable[Scenario],
    client: httpx.Client,
    *,
    strict: bool = False,
    model: str = _REPLAY_MODEL,
    emit: Callable[[str], None] = print,
) -> int:
    """Replay scenarios through ``client``; return the process exit code.

    ``client`` is the injection seam: the CLI builds its own httpx.Client from
    ``--base-url``; tests pass a client wired to an in-process ASGI app.
    ``model`` is the request ``model`` field for every turn (some upstreams
    reject unknown model names).
    """
    warnings = 0
    for scenario in scenarios:
        try:
            warnings += _run_scenario(scenario, client, emit, model)
        except httpx.HTTPError as exc:
            emit(f"ERROR scenario {scenario.name}: {exc} (scenario aborted)")
    return 1 if strict and warnings else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.replay",
        description="Replay benchmark scenarios against a running Infinitum instance.",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"instance URL (default {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        metavar="NAME",
        help="scenario to replay; repeatable (default: all scenarios in benchmarks/scenarios)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when any WARN line was printed"
    )
    parser.add_argument(
        "--model",
        default=_REPLAY_MODEL,
        help=f"model name for turn requests (default {_REPLAY_MODEL}; "
        "use your upstream's model when it validates names)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        default=120.0,
        help="per-request read timeout (default 120; connect capped at 5)",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios(_SCENARIOS_DIR)
    if args.scenario:
        by_name = {scenario.name: scenario for scenario in scenarios}
        unknown = [name for name in args.scenario if name not in by_name]
        if unknown:
            parser.error(f"unknown scenario(s) {unknown}; available: {sorted(by_name)}")
        scenarios = [by_name[name] for name in args.scenario]

    # connect stays capped at 5s so a dead host fails fast; read is --timeout.
    with httpx.Client(
        base_url=args.base_url, timeout=httpx.Timeout(args.timeout, connect=5.0)
    ) as client:
        return replay_scenarios(scenarios, client, strict=args.strict, model=args.model)


if __name__ == "__main__":
    sys.exit(main())
