import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from infinitum.config import AppConfig, load_config


def test_config_environment_expansion():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text("upstream:\n  api_key: ${TEST_CONTEXT_KEY:-fallback}\n")
        os.environ["TEST_CONTEXT_KEY"] = "secret"
        cfg = load_config(path)
        assert cfg.upstream.api_key == "secret"


def test_learning_controls_have_safe_defaults():
    cfg = load_config()
    assert cfg.learning.timeout_seconds == 600.0
    assert cfg.learning.max_tokens == 2048
    assert cfg.learning.extra_body == {}
    assert cfg.learning.topic_summary_debounce_seconds == 30.0
    assert cfg.learning.topic_summary_update_threshold == 5
    assert cfg.learning.topic_summary_max_changed_memories == 24
    assert cfg.learning.topic_summary_context_memories == 8
    assert cfg.learning.topic_summary_bootstrap_max_memories == 32
    assert cfg.learning.topic_summary_max_tokens == 1024
    assert cfg.learning.topic_summary_fallback_memories == 12


def test_reinforcement_controls_have_safe_defaults():
    cfg = load_config()
    assert cfg.memory.reinforce_similarity == 0.86
    assert cfg.memory.reinforce_semantic_similarity == 0.90
    assert cfg.memory.reinforce_hint_min_score == 0.55
    assert cfg.memory.reinforce_hint_min_lexical == 0.40
    assert cfg.memory.reinforce_hint_min_semantic == 0.72


def test_stream_reasoning_defaults():
    cfg = AppConfig()
    assert cfg.memory.stream_reasoning == "live"
    assert cfg.memory.reasoning_delta_fields == ["reasoning", "reasoning_content"]


def test_stream_reasoning_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        AppConfig(memory={"stream_reasoning": "bogus"})
