"""CI gate: every golden scenario must pass, plus metrics aggregation tests.

Each YAML under benchmarks/scenarios/ is parametrized into its own test so a
failure names the scenario. Commands must run from the repo root so the
`benchmarks` namespace resolves (it is not an installed package).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.corpus import load_scenario, load_scenarios
from benchmarks.runner import ExpectationRecord, ScenarioResult, run_scenario
from benchmarks.scoring import Metrics, aggregate, build_report

_SCENARIO_DIR = Path("benchmarks/scenarios")
_SCENARIOS = load_scenarios(_SCENARIO_DIR)


@pytest.mark.parametrize(
    "path", sorted(_SCENARIO_DIR.glob("*.y*ml")), ids=lambda p: p.stem
)
def test_scenario_gate(path: Path) -> None:
    """Given a golden scenario file, when replayed offline, then all records pass."""
    result = run_scenario(load_scenario(path))
    failed = [
        f"{r.kind} turn {r.turn_index}: {r.target}" for r in result.records if not r.passed
    ]
    assert result.ok, f"scenario {result.scenario_name} failed {len(failed)} record(s): {failed}"


def _rec(kind: str, passed: bool, turn: int = 0) -> ExpectationRecord:
    return ExpectationRecord(turn, kind, "target", passed)


def test_aggregate_ratios_when_counts_are_mixed() -> None:
    """Given TP=1/FN=1/FP=1 and split probe outcomes, ratios are 0.5/0.5/0.0/1.0."""
    result = ScenarioResult(
        "synthetic",
        (
            _rec("created", True),
            _rec("created", False),
            _rec("ignored", False),
            _rec("probe_must_include", True),
            _rec("probe_must_not_include", False),
        ),
        total_tokens=10,
    )
    metrics = aggregate([result])
    assert metrics == Metrics(
        extraction_precision=pytest.approx(0.5),
        extraction_recall=pytest.approx(0.5),
        retrieval_precision=pytest.approx(0.0),
        retrieval_recall=pytest.approx(1.0),
        total_tokens=10,
    )


def test_aggregate_scores_perfect_when_no_opportunities() -> None:
    """Given zero checks of every class, all metrics are 1.0 by convention."""
    metrics = aggregate([ScenarioResult("empty", (), total_tokens=3)])
    assert (
        metrics.extraction_precision
        == metrics.extraction_recall
        == metrics.retrieval_precision
        == metrics.retrieval_recall
        == 1.0
    )
    assert metrics.total_tokens == 3


def test_build_report_sections_when_aggregating() -> None:
    """Given one failing scenario, the report exposes per_scenario + aggregate."""
    result = ScenarioResult("one", (_rec("created", False),), total_tokens=7)
    report = build_report([result])
    assert report["aggregate"]["extraction_precision"] == 1.0
    assert report["aggregate"]["extraction_recall"] == 0.0
    assert report["aggregate"]["total_tokens"] == 7
    (entry,) = report["per_scenario"]
    assert entry["name"] == "one"
    assert entry["ok"] is False
    assert entry["records"] == [
        {"turn_index": 0, "kind": "created", "target": "target", "passed": False}
    ]
