from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.getenv(name, default)

    return _ENV_RE.sub(repl, value)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8788
    log_level: str = "info"


class UpstreamConfig(BaseModel):
    base_url: str = "http://localhost:4000/v1"
    api_key: str = ""
    passthrough_authorization: bool = True
    timeout_seconds: float = 300.0


class MemoryConfig(BaseModel):
    enabled: bool = True
    database_path: str = "./infinitum.db"
    retrieve_candidates: int = 50
    minimum_retrieval_score: float = 0.18
    inject_max_memories: int = 24
    dedup_similarity: float = 0.88
    # Fast deterministic reinforcement for nearly identical wording.
    reinforce_similarity: float = 0.86
    # When embeddings are enabled, semantically equivalent memories can
    # reinforce even when their wording differs substantially.
    reinforce_semantic_similarity: float = 0.90
    # A learner-proposed reinforce hint may use a lower similarity bar, but only
    # after exact memory type + topic compatibility and a credible retrieval match.
    reinforce_hint_min_score: float = 0.55
    reinforce_hint_min_lexical: float = 0.40
    reinforce_hint_min_semantic: float = 0.72
    supersede_similarity_floor: float = 0.30
    freshness_half_life_days: float = 120.0
    # Exposes infinitum_memory_search/infinitum_memory_get to the model when a
    # memory block is injected; default off for upstream compatibility.
    tools_enabled: bool = False


class ContextConfig(BaseModel):
    model_context_window: int = 262_144
    reserve_output_tokens: int = 32_768
    reserve_free_tokens: int = 16_384
    max_memory_tokens: int = 100_000
    memory_message_role: Literal["system", "developer"] = "system"


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    timeout_seconds: float = 60.0


class LearningConfig(BaseModel):
    enabled: bool = True
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    # Learning is a background, non-streaming model call and may legitimately
    # take longer than foreground proxy requests on local/reasoning models.
    timeout_seconds: float = 600.0
    # Bound extraction/summary generations. Without this, some OpenAI-compatible
    # servers may use a very large model default and hold the HTTP response open
    # until the client's read timeout expires.
    max_tokens: int = 2048
    # Extra top-level fields merged into background Chat Completions requests.
    # This is useful for OpenAI-compatible servers with vendor-specific knobs,
    # e.g. vLLM/Qwen chat_template_kwargs to disable thinking for extraction.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    poll_interval_seconds: float = 1.0
    max_attempts: int = 5
    topic_summaries: bool = True
    topic_summary_min_memories: int = 3
    # Topic summaries are maintained incrementally rather than regenerated from
    # the full topic after every interaction. Changes are debounced so bursts of
    # related learning produce one summary update instead of one call per turn.
    topic_summary_debounce_seconds: float = 30.0
    topic_summary_update_threshold: int = 5
    topic_summary_max_changed_memories: int = 24
    topic_summary_context_memories: int = 8
    topic_summary_bootstrap_max_memories: int = 32
    topic_summary_max_tokens: int = 1024
    # If an upstream returns no final content (common when a reasoning model
    # exhausts its token budget before producing an answer), build a bounded
    # deterministic summary from active memories rather than retrying the same
    # doomed job repeatedly.
    topic_summary_fallback_memories: int = 12


class RequestContextConfig(BaseModel):
    enabled: bool = True
    derive_project_from_cwd: bool = True

    # Canonical Infinitum headers first, followed by compatibility aliases
    # useful with OpenCode, Headroom, and a future LiteLLM edge. These are soft
    # hints in V0.2.0, not authenticated identity.
    user_headers: list[str] = Field(
        default_factory=lambda: [
            "x-infinitum-user-id",
            "x-context-user-id",
            "x-opencode-user-id",
            "x-opencode-user",
            "x-headroom-user-id",
            "x-litellm-user-id",
        ]
    )
    project_headers: list[str] = Field(
        default_factory=lambda: [
            "x-infinitum-project-id",
            "x-context-project-id",
            "x-opencode-project-id",
            "x-opencode-project",
            "x-headroom-project-id",
        ]
    )
    cwd_headers: list[str] = Field(
        default_factory=lambda: [
            "x-infinitum-cwd",
            "x-context-cwd",
            "x-opencode-directory",
            "x-opencode-cwd",
            "x-headroom-cwd",
        ]
    )

    # Context stays globally visible in V0.2.0. These bounded bonuses only
    # reorder already-relevant memories toward evidence observed in the current
    # user/project/CWD. They never make an otherwise irrelevant memory eligible.
    user_affinity_bonus: float = 0.03
    project_affinity_bonus: float = 0.07
    cwd_affinity_bonus: float = 0.01

    # Consumed identity hints are stripped before normal upstream forwarding.
    # When Infinitum is directly in front of Headroom, opt in to emitting
    # canonical x-headroom-* headers from the resolved context.
    forward_to_headroom: bool = False


class RetrievalWeights(BaseModel):
    semantic: float = 0.45
    lexical: float = 0.18
    importance: float = 0.14
    confidence: float = 0.10
    freshness: float = 0.05
    topic: float = 0.08

    @model_validator(mode="after")
    def validate_sum(self) -> "RetrievalWeights":
        total = sum(
            [
                self.semantic,
                self.lexical,
                self.importance,
                self.confidence,
                self.freshness,
                self.topic,
            ]
        )
        if total <= 0:
            raise ValueError("retrieval weights must sum to a positive value")
        return self


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    request_context: RequestContextConfig = Field(default_factory=RequestContextConfig)
    retrieval_weights: RetrievalWeights = Field(default_factory=RetrievalWeights)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.getenv("INFINITUM_CONFIG") or os.getenv("CONTEXT_RUNTIME_CONFIG")
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    else:
        data = {}

    config = AppConfig.model_validate(_expand_env(data))

    # Upgrade convenience for pre-Infinitum installs: when database_path was
    # never explicitly configured, prefer an existing legacy database rather
    # than silently starting with an empty ./infinitum.db. Explicit config
    # always wins.
    memory_data = data.get("memory") if isinstance(data, dict) else None
    explicit_database_path = isinstance(memory_data, dict) and "database_path" in memory_data
    if not explicit_database_path:
        legacy = Path("./context-runtime.db")
        current = Path(config.memory.database_path)
        if legacy.exists() and not current.exists():
            config.memory.database_path = str(legacy)

    return config
