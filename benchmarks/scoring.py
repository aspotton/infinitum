"""Aggregate metrics over scenario results.

Extraction precision/recall treat the expectation records as a confusion
matrix: passing created/reinforced/superseded records are true positives
(memory learned correctly), FAILING ignored records are false positives
(a fact that should have been ignored was learned), failing
created/reinforced/superseded records are false negatives (missed learning).
Passing ignored records are correct rejections and count in neither.

Retrieval precision/recall come from the probe records: must_not_include
outcomes are the negative checks on the returned set (precision),
must_include outcomes the positive coverage (recall).

Zero-denominator convention: a class with no checks scores 1.0 — no
opportunity to err is scored as perfect, so an all-ignored corpus or a
scenario without probes does not drag the aggregate toward zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .runner import ScenarioResult

_LEARNING_KINDS = frozenset({"created", "reinforced", "superseded"})


@dataclass(frozen=True, slots=True)
class Metrics:
    extraction_precision: float
    extraction_recall: float
    retrieval_precision: float
    retrieval_recall: float
    total_tokens: int


def _ratio(passed: int, total: int) -> float:
    return 1.0 if total == 0 else passed / total


def aggregate(results: Sequence[ScenarioResult]) -> Metrics:
    tp = fp = fn = 0
    inc = inc_pass = not_inc = not_inc_pass = 0
    for result in results:
        for record in result.records:
            if record.kind in _LEARNING_KINDS:
                if record.passed:
                    tp += 1
                else:
                    fn += 1
            elif record.kind == "ignored" and not record.passed:
                fp += 1
            elif record.kind == "probe_must_include":
                inc += 1
                inc_pass += record.passed
            elif record.kind == "probe_must_not_include":
                not_inc += 1
                not_inc_pass += record.passed
    return Metrics(
        extraction_precision=_ratio(tp, tp + fp),
        extraction_recall=_ratio(tp, tp + fn),
        retrieval_precision=_ratio(not_inc_pass, not_inc),
        retrieval_recall=_ratio(inc_pass, inc),
        total_tokens=sum(r.total_tokens for r in results),
    )


def build_report(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    """Terminal/JSON report body: per-scenario pass/fail records + aggregate."""
    metrics = aggregate(results)
    return {
        "per_scenario": [
            {
                "name": r.scenario_name,
                "ok": r.ok,
                "total_tokens": r.total_tokens,
                "records": [asdict(rec) for rec in r.records],
            }
            for r in results
        ],
        "aggregate": {
            **asdict(metrics),
            "scenarios_total": len(results),
            "scenarios_passed": sum(r.ok for r in results),
        },
    }
