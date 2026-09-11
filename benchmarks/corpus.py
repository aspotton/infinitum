"""Scenario corpus: YAML schema, models, and loaders for benchmark scenarios.

A scenario is one conversation script: turns the runner replays, memory
candidates the scripted extractor should propose per turn, per-turn learning
expectations, and an optional retrieval probe. Field names here are
load-bearing contracts for the runner (todo 3) and metrics (todo 4).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from infinitum.models import MemoryType

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_OPERATIONS = Literal["new", "reinforce", "supersede"]
_TEMPORAL_VIEWS = Literal["all", "current", "as_of"]


class ScenarioError(Exception):
    """A scenario file failed to load; the message names the file and field."""


def _require_iso_date(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be an ISO YYYY-MM-DD string, not {type(value).__name__}"
            " (quote dates in YAML)"
        )
    if not _ISO_DATE_RE.match(value):
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD string, got {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field} is not a real calendar date: {value!r}") from exc
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioContext(StrictModel):
    user_id: str = "eval"
    project_id: str


class CandidateSpec(StrictModel):
    operation: _OPERATIONS
    memory_type: MemoryType
    topic: str
    content: str
    importance: float = 0.5
    confidence: float = 0.7
    valid_from: str | None = None
    valid_until: str | None = None
    explicit_correction: bool = False
    # Substring identifying the active memory to supersede; memory IDs are
    # unknown at authoring time and resolved by the runner at call time.
    supersedes_match: str | None = None

    @field_validator("valid_from", "valid_until")
    @classmethod
    def _check_dates(cls, value: str | None, info) -> str | None:
        return _require_iso_date(value, info.field_name)


class Expectation(StrictModel):
    created: list[str] = Field(default_factory=list)
    reinforced: list[str] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)


class RankPair(StrictModel):
    # `memory` must rank strictly below `below` by result-list position.
    memory: str
    below: str


class Probe(StrictModel):
    query: str
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    must_demote: list[str] = Field(default_factory=list)
    must_rank_below: list[RankPair] = Field(default_factory=list)
    temporal_view: _TEMPORAL_VIEWS = "current"
    as_of: str | None = None

    @field_validator("as_of")
    @classmethod
    def _check_as_of(cls, value: str | None) -> str | None:
        return _require_iso_date(value, "as_of")


def _scored_by_content(items: list[dict[str, Any]], entry: str) -> float | None:
    """Score of the first search result whose content contains ``entry`` (must_include style)."""
    for item in items:
        if entry in item["memory"]["content"]:
            score: float = item["score"]
            return score
    return None


def demotion_violations(
    entries: list[str],
    probed_items: list[dict[str, Any]],
    all_view_items: list[dict[str, Any]],
) -> dict[str, str]:
    """must_demote entries the two search result sets fail to prove demoted.

    Maps each failing entry to a self-naming reason; an entry passes only when
    the probed-view results contain a memory whose content includes it AND the
    temporal_view="all" follow-up search scores that same memory strictly
    higher. Scores are compared post-clamp (min(1.0,...)): a row pinned at 1.0
    in both views fails with a ceiling-named reason instead of the misleading
    strictly-higher failure, because no factor is observable there.
    """
    violations: dict[str, str] = {}
    for entry in entries:
        probed = _scored_by_content(probed_items, entry)
        every = _scored_by_content(all_view_items, entry)
        if probed is None or every is None:
            violations[entry] = "not present in both search result sets"
        elif not every > probed:
            if probed >= 1.0 or every >= 1.0:
                violations[entry] = (
                    "at score ceiling 1.0 - post-clamp comparison cannot prove "
                    "the factor; lower fixture scores"
                )
            else:
                violations[entry] = "current-view score is not below the all-view score"
    return violations


def _position_by_content(items: list[dict[str, Any]], entry: str) -> int | None:
    """Index of the first search result whose content contains ``entry``."""
    for position, item in enumerate(items):
        if entry in item["memory"]["content"]:
            return position
    return None


def rank_pair_label(pair: RankPair) -> str:
    return f"{pair.memory!r} below {pair.below!r}"


def ranking_violations(pairs: list[RankPair], probed_items: list[dict[str, Any]]) -> list[str]:
    """Labels of must_rank_below pairs the probed view fails to order.

    Checked in the probed view only (no all-view follow-up): a pair passes only
    when both entries are found and the first memory containing ``memory``
    sits at a strictly later result-list position than the first containing
    ``below``.
    """
    violations: list[str] = []
    for pair in pairs:
        first = _position_by_content(probed_items, pair.memory)
        second = _position_by_content(probed_items, pair.below)
        if first is None or second is None or not first > second:
            violations.append(rank_pair_label(pair))
    return violations


class Turn(StrictModel):
    user: str
    assistant: str
    learn: list[CandidateSpec] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    probe: Probe | None = None


class Scenario(StrictModel):
    name: str = Field(min_length=1)
    description: str
    context: ScenarioContext
    turns: list[Turn] = Field(min_length=1)


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioError(f"{path}: failed to read/parse YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: scenario document must be a mapping")
    try:
        return Scenario.model_validate(data)
    except ValueError as exc:
        details = "; ".join(_describe(err) for err in _errors(exc))
        raise ScenarioError(f"{path}: {details}") from exc


def _describe(err: dict) -> str:
    field = ".".join(str(part) for part in err["loc"]) or "<root>"
    return f"{field}: {err['msg']}"


def _errors(exc: Exception) -> list[dict]:
    errors = getattr(exc, "errors", None)
    return errors() if callable(errors) else [{"loc": (), "msg": str(exc)}]


def load_scenarios(directory: str | Path) -> list[Scenario]:
    paths = sorted(
        p
        for pattern in ("*.yaml", "*.yml")
        for p in Path(directory).glob(pattern)
        if p.is_file()
    )
    return [load_scenario(p) for p in paths]
