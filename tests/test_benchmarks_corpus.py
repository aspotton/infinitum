"""Tests for the benchmarks scenario corpus schema and YAML loader (todo 1)."""

from pathlib import Path

import pytest
import yaml

from benchmarks.corpus import (
    CandidateSpec,
    Expectation,
    Probe,
    Scenario,
    ScenarioContext,
    ScenarioError,
    Turn,
    load_scenario,
    load_scenarios,
)

VALID_YAML = """\
name: durable-fact
description: A fact stated once is remembered later.
context:
  project_id: durable-fact
turns:
  - user: We use PostgreSQL 17 for the primary database.
    assistant: Got it, PostgreSQL 17 it is.
    learn:
      - operation: new
        memory_type: fact
        topic: database
        content: The team uses PostgreSQL 17.
        importance: 0.8
    expect:
      created: ["PostgreSQL 17"]
    probe:
      query: What database do we use?
      must_include: ["PostgreSQL"]
      must_not_include: ["MySQL"]
      temporal_view: current
"""


def _write(tmp: str, name: str, text: str) -> Path:
    path = Path(tmp) / name
    path.write_text(text, encoding="utf-8")
    return path


def _mutate(data: dict, name: str, tmp: str) -> Path:
    return _write(tmp, name, yaml.safe_dump(data))


def _valid_payload() -> dict:
    return yaml.safe_load(VALID_YAML)


def test_valid_scenario_loads_into_expected_models(tmp_path):
    """Given a valid YAML file, When loaded, Then it equals the hand-built models."""
    path = _write(str(tmp_path), "durable-fact.yaml", VALID_YAML)
    expected = Scenario(
        name="durable-fact",
        description="A fact stated once is remembered later.",
        context=ScenarioContext(user_id="eval", project_id="durable-fact"),
        turns=[
            Turn(
                user="We use PostgreSQL 17 for the primary database.",
                assistant="Got it, PostgreSQL 17 it is.",
                learn=[
                    CandidateSpec(
                        operation="new",
                        memory_type="fact",
                        topic="database",
                        content="The team uses PostgreSQL 17.",
                        importance=0.8,
                    )
                ],
                expect=Expectation(created=["PostgreSQL 17"]),
                probe=Probe(
                    query="What database do we use?",
                    must_include=["PostgreSQL"],
                    must_not_include=["MySQL"],
                ),
            )
        ],
    )
    assert load_scenario(path) == expected


def test_defaults_when_omitted():
    """Given minimal fields, When built directly, Then documented defaults apply."""
    turn = Turn(user="hi", assistant="hello")
    assert turn.learn == []
    assert turn.expect == Expectation()
    assert turn.probe is None
    candidate = CandidateSpec(
        operation="new", memory_type="fact", topic="t", content="c"
    )
    assert candidate.importance == 0.5
    assert candidate.confidence == 0.7
    assert candidate.valid_from is None
    assert candidate.valid_until is None
    assert candidate.explicit_correction is False
    assert candidate.supersedes_match is None
    probe = Probe(query="q", must_include=[], must_not_include=[])
    assert probe.must_demote == []
    assert probe.temporal_view == "current"
    assert probe.as_of is None
    assert ScenarioContext(project_id="p").user_id == "eval"


def test_load_scenarios_returns_sorted_and_empty_dir_ok(tmp_path):
    """Given a dir with yaml files, When loaded, Then sorted by filename; empty dir loads []."""
    _write(str(tmp_path), "b.yaml", VALID_YAML.replace("durable-fact", "b"))
    _write(str(tmp_path), "a.yaml", VALID_YAML.replace("durable-fact", "a"))
    loaded = load_scenarios(str(tmp_path))
    assert [s.name for s in loaded] == ["a", "b"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_scenarios(str(empty)) == []


def test_unknown_operation_names_file_and_field(tmp_path):
    """Given operation: delete, When loaded, Then ScenarioError names the file and 'operation'."""
    data = _valid_payload()
    data["turns"][0]["learn"][0]["operation"] = "delete"
    path = _mutate(data, "bad.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert "bad.yaml" in message
    assert "operation" in message


def test_missing_turns_names_file_and_field(tmp_path):
    """Given no turns key, When loaded, Then ScenarioError names the file and 'turns'."""
    data = _valid_payload()
    del data["turns"]
    path = _mutate(data, "noturns.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "noturns.yaml" in str(excinfo.value)
    assert "turns" in str(excinfo.value)


def test_empty_turns_list_is_rejected(tmp_path):
    """Given turns: [], When loaded, Then ScenarioError names 'turns'."""
    data = _valid_payload()
    data["turns"] = []
    path = _mutate(data, "emptyturns.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "turns" in str(excinfo.value)


def test_non_iso_valid_from_names_file_and_field(tmp_path):
    """Given valid_from '15/01/2024', When loaded, Then ScenarioError names 'valid_from'."""
    data = _valid_payload()
    data["turns"][0]["learn"][0]["valid_from"] = "15/01/2024"
    path = _mutate(data, "baddate.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "baddate.yaml" in str(excinfo.value)
    assert "valid_from" in str(excinfo.value)


def test_impossible_calendar_date_is_rejected(tmp_path):
    """Given valid_until '2024-02-30', When loaded, Then ScenarioError fires despite ISO shape."""
    data = _valid_payload()
    data["turns"][0]["learn"][0]["valid_until"] = "2024-02-30"
    path = _mutate(data, "impossible.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "valid_until" in str(excinfo.value)


def test_invalid_as_of_fails_at_load_time(tmp_path):
    """Given probe.as_of 'tomorrow', When loaded, Then load raises naming 'as_of'."""
    data = _valid_payload()
    data["turns"][0]["probe"]["as_of"] = "tomorrow"
    path = _mutate(data, "badprobe.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "badprobe.yaml" in str(excinfo.value)
    assert "as_of" in str(excinfo.value)


def test_iso_dates_accepted_as_strings(tmp_path):
    """Given ISO strings on valid_from/valid_until/as_of, When loaded, Then they load unchanged."""
    data = _valid_payload()
    candidate = data["turns"][0]["learn"][0]
    candidate["valid_from"] = "2024-01-15"
    candidate["valid_until"] = "2024-06-30"
    data["turns"][0]["probe"]["temporal_view"] = "as_of"
    data["turns"][0]["probe"]["as_of"] = "2024-03-01"
    scenario = load_scenario(_mutate(data, "dated.yaml", str(tmp_path)))
    spec = scenario.turns[0].learn[0]
    assert spec.valid_from == "2024-01-15"
    assert spec.valid_until == "2024-06-30"
    assert scenario.turns[0].probe.as_of == "2024-03-01"


def test_empty_name_names_field(tmp_path):
    """Given name: '', When loaded, Then ScenarioError names the file and 'name'."""
    data = _valid_payload()
    data["name"] = ""
    path = _mutate(data, "noname.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "noname.yaml" in str(excinfo.value)
    assert "name" in str(excinfo.value)


def test_unquoted_yaml_date_rejected_with_clear_message(tmp_path):
    """Given an unquoted YAML date (parsed as date object), When loaded, Then it is rejected."""
    # Unquoted so YAML yields a datetime.date object, not an ISO string.
    path = _write(
        str(tmp_path),
        "unquoted.yaml",
        VALID_YAML.replace(
            "        importance: 0.8\n",
            "        importance: 0.8\n        valid_from: 2024-01-15\n",
        ),
    )
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "valid_from" in str(excinfo.value)


def test_malformed_yaml_names_file(tmp_path):
    """Given broken YAML syntax, When loaded, Then ScenarioError names the file."""
    path = _write(str(tmp_path), "broken.yaml", "name: [unclosed\n")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "broken.yaml" in str(excinfo.value)


def test_unknown_key_is_rejected(tmp_path):
    """Given a typo'd key, When loaded, Then ScenarioError fires naming the file."""
    data = _valid_payload()
    data["turns"][0]["explication"] = "typo"
    path = _mutate(data, "typo.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "typo.yaml" in str(excinfo.value)


def test_must_demote_parses_and_defaults_empty(tmp_path):
    """Given probe.must_demote ['x'], When loaded, Then it parses; absent means []."""
    data = _valid_payload()
    data["turns"][0]["probe"]["must_demote"] = ["PostgreSQL"]
    scenario = load_scenario(_mutate(data, "demote.yaml", str(tmp_path)))
    assert scenario.turns[0].probe.must_demote == ["PostgreSQL"]
    plain = load_scenario(_write(str(tmp_path), "plain.yaml", VALID_YAML))
    assert plain.turns[0].probe.must_demote == []


def test_unknown_probe_sibling_of_must_demote_still_rejected(tmp_path):
    """Given probe.must_promote (a fake sibling of must_demote), When loaded, Then rejected."""
    data = _valid_payload()
    data["turns"][0]["probe"]["must_demote"] = ["PostgreSQL"]
    data["turns"][0]["probe"]["must_promote"] = ["MySQL"]
    path = _mutate(data, "badsibling.yaml", str(tmp_path))
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    assert "badsibling.yaml" in str(excinfo.value)
    assert "must_promote" in str(excinfo.value)
