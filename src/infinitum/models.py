from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MemoryType = Literal["fact", "decision", "preference", "goal", "procedure", "lesson", "episodic"]
MemoryStatus = Literal["active", "superseded", "contested", "archived"]


def _coerce_iso_or_none(value: Any) -> str | None:
    """Keep an ISO date/datetime string, NULL anything else.

    Used by the temporal field validators below. It never raises: an
    unparseable value (a bare word, a number, garbage) becomes ``None`` so a
    candidate carrying a malformed date is still created, only without the
    temporal annotation. Losing a whole candidate to a bad date string is worse
    than dropping the date.
    """

    if value is None:
        return None
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except ValueError:
            return None
        return value
    return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"




class RequestContext(BaseModel):
    """Soft provenance/affinity context resolved from request headers.

    These fields are not an authorization boundary in the global-memory V0.1
    line. They become inputs to hard scope filtering only in the future scoped
    memory architecture.
    """

    user_id: str | None = None
    project_id: str | None = None
    cwd: str | None = None
    project_derived_from_cwd: bool = False

    def compact(self) -> dict[str, str | bool]:
        result: dict[str, str | bool] = {}
        if self.user_id:
            result["user_id"] = self.user_id
        if self.project_id:
            result["project_id"] = self.project_id
        if self.cwd:
            result["cwd"] = self.cwd
        if self.project_derived_from_cwd:
            result["project_derived_from_cwd"] = True
        return result

    @property
    def is_empty(self) -> bool:
        return not (self.user_id or self.project_id or self.cwd)


class Event(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str
    user_id: str | None = None
    project_id: str | None = None
    cwd: str | None = None
    request_id: str | None = None
    event_type: str
    role: str | None = None
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    memory_type: MemoryType = "fact"
    topic: str = "general"
    content: str
    status: MemoryStatus = "active"
    importance: float = 0.5
    confidence: float = 0.7
    observation_count: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_event_ids: list[str] = Field(default_factory=list)
    # Temporal validity. ISO date/datetime strings, always optional. ``observed_at``
    # is derived (earliest source event) and not extraction-settable.
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None

    @field_validator("valid_from", "valid_until", "observed_at", mode="before")
    @classmethod
    def _validate_temporal(cls, value: Any) -> str | None:
        return _coerce_iso_or_none(value)


class MemoryCandidate(BaseModel):
    memory_type: MemoryType = "fact"
    topic: str = "general"
    content: str
    importance: float = 0.5
    confidence: float = 0.7
    operation_hint: Literal["new", "reinforce", "supersede"] = "new"
    # Optional target chosen by the extraction model from the bounded nearby
    # memories it was shown. Deterministic code still validates compatibility
    # and similarity before applying the reinforcement.
    reinforces_memory_id: str | None = None
    supersedes_memory_ids: list[str] = Field(default_factory=list)
    explicit_correction: bool = False
    reason: str = ""
    # Optional temporal bounds the extraction model may propose.
    valid_from: str | None = None
    valid_until: str | None = None

    @field_validator("valid_from", "valid_until", mode="before")
    @classmethod
    def _validate_temporal(cls, value: Any) -> str | None:
        return _coerce_iso_or_none(value)


class ScoredMemory(BaseModel):
    memory: Memory
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    topic_score: float = 0.0
    freshness_score: float = 0.0
    user_affinity_score: float = 0.0
    project_affinity_score: float = 0.0
    cwd_affinity_score: float = 0.0
    affinity_bonus: float = 0.0


class TopicSummary(BaseModel):
    topic: str
    summary: str
    memory_count: int
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryCreateRequest(BaseModel):
    memory_type: MemoryType = "fact"
    topic: str = "general"
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=200)
    include_archived: bool = False
