"""Report CLI: replay scenarios, print a table, optionally emit JSON.

Usage from the repo root::

    python -m benchmarks.run [--scenario NAME]... [--json PATH] [--scenarios-dir DIR]

Exit code is 1 if any scenario fails (or errors), 0 otherwise. No latency
metric: foreground timing does not exist in the runner, deliberately dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .corpus import Scenario, ScenarioError, load_scenarios
from .runner import RunnerError, ScenarioResult, run_scenario
from .scoring import build_report


def _select(
    scenarios: list[Scenario], names: list[str] | None, parser: argparse.ArgumentParser
) -> list[Scenario]:
    if not names:
        return scenarios
    by_name = {s.name: s for s in scenarios}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}; available: {sorted(by_name)}")
    return [by_name[n] for n in names]


def _print_table(per_scenario: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    width = max(len(str(row["name"])) for row in per_scenario) if per_scenario else 6
    print(f"{'scenario'.ljust(width)}  status  checks   tokens")
    for row in per_scenario:
        passed = sum(rec["passed"] for rec in row["records"])
        status = "PASS" if row["ok"] else "FAIL"
        print(
            f"{row['name']:<{width}}  {status:<4}  "
            f"{passed}/{len(row['records']):<3}  {row['total_tokens']:>8}"
        )
        for rec in row["records"]:
            if not rec["passed"]:
                print(f"  - fail: {rec['kind']} turn {rec['turn_index']}: {rec['target']}")
        if row.get("error"):
            print(f"  - error: {row['error']}")
    print(
        f"\n{aggregate['scenarios_passed']}/{aggregate['scenarios_total']} scenarios passed"
        f" | extraction P={aggregate['extraction_precision']:.3f}"
        f" R={aggregate['extraction_recall']:.3f}"
        f" | retrieval P={aggregate['retrieval_precision']:.3f}"
        f" R={aggregate['retrieval_recall']:.3f}"
        f" | tokens={aggregate['total_tokens']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description=__doc__)
    parser.add_argument("--scenario", action="append", metavar="NAME",
                        help="run only this scenario (repeatable)")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the report JSON here")
    parser.add_argument("--scenarios-dir", default="benchmarks/scenarios")
    args = parser.parse_args(argv)

    try:
        scenarios = load_scenarios(args.scenarios_dir)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    selected = _select(scenarios, args.scenario, parser)

    results: list[ScenarioResult] = []
    errors: list[tuple[str, str]] = []
    for scenario in selected:
        try:
            result = run_scenario(scenario)
        except RunnerError as exc:
            print(f"error: scenario {scenario.name}: {exc}", file=sys.stderr)
            errors.append((scenario.name, str(exc)))
            continue
        results.append(result)

    report = build_report(results)
    report["per_scenario"].extend(
        {"name": name, "ok": False, "total_tokens": 0, "records": [], "error": message}
        for name, message in errors
    )
    _print_table(report["per_scenario"], report["aggregate"])
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["ok"] for row in report["per_scenario"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
