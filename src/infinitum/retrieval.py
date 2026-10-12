from __future__ import annotations

import logging
from datetime import UTC, datetime, timezone
from typing import Literal

import numpy as np

from .config import AppConfig
from .database import Database
from .embeddings import EmbeddingClient
from .models import Memory, RequestContext, ScoredMemory
from .text import (
    freshness_score,
    lexical_similarity,
    normalize_text,
    tokens,
    topic_similarity,
)

log = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b) / denom)))


def _temporal_bound(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Normalize a stored ISO validity bound to an aware UTC datetime.

    A date-only ``YYYY-MM-DD`` string is midnight UTC as a start bound and
    ``23:59:59`` UTC as an end bound (``end_of_day``), so a fact valid until
    today stays current through today. Naive datetimes are assumed UTC; full
    datetimes compare as stored. An unparseable value is treated as no bound
    (NULL), mirroring the coerce-never-raise policy of the Memory validators:
    the model path never stores garbage, only raw SQL writes could.
    """
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if end_of_day and len(value) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed


# Demotion factor for naturally-expired rows under the "current" temporal view.
# Decision record: 0.70 is the highest factor that reliably loses a
# head-to-head against an equal-base active row while keeping a strong
# mis-dated static fact visible; a deeper supersession-tier factor was
# dropped as unreachable (supersede_memory sets status='superseded' and the
# candidate set is active-only).
EXPIRED_FACTOR = 0.70


def validity_factor(valid_until: str | None, now: datetime) -> float:
    """Multiplicative current-view demotion for naturally-expired rows.

    Returns EXPIRED_FACTOR iff ``valid_until`` is a strictly-past bound under
    the same end-of-day normalization the as_of filter uses, else 1.0 — a
    bound exactly equal to ``now`` is not yet expired. Parsing is delegated
    wholesale to ``_temporal_bound``, so odd ISO forms behave identically.
    """
    until = _temporal_bound(valid_until, end_of_day=True)
    return EXPIRED_FACTOR if until is not None and until < now else 1.0


def _is_high_authority(memory: Memory) -> bool:
    """High-authority persistent goals/decisions that survive wording mismatch."""
    return memory.memory_type in {"goal", "decision"} and memory.importance >= 0.85


class MemoryRetriever:
    def __init__(self, db: Database, embeddings: EmbeddingClient, config: AppConfig):
        self.db = db
        self.embeddings = embeddings
        self.config = config
        self._warned_fallbacks: set[tuple[str, str]] = set()

    async def search(
        self,
        query: str,
        limit: int | None = None,
        request_context: RequestContext | None = None,
        temporal_view: Literal["all", "current", "as_of"] = "all",
        as_of: str | None = None,
    ) -> list[ScoredMemory]:
        """Hybrid-scored search over active memories.

        ``temporal_view`` never enters base_score or the relevance gates. The
        default ``"all"`` applies no temporal handling and is byte-identical
        to pre-temporal behavior for every existing caller; the compiler will
        compile with ``"current"``. ``"current"`` scores rows normally and,
        AFTER the gates pass, multiplies naturally-expired ACTIVE rows
        (``valid_until`` strictly in the past) by ``EXPIRED_FACTOR``: expired
        rows are demoted into shadow of live ones, never hidden, so a strong
        mis-dated fact can still surface while an equal active peer wins
        head-to-head. ``"as_of"`` stays a hard eligibility filter applied
        before scoring (eligibility before ranking, AGENTS.md invariant 10):
        it keeps only rows whose validity window covers the given ISO
        date/datetime (a date-only ``as_of`` counts as 00:00:00 UTC that day,
        so "as of today" agrees with the current view). Rows with any
        non-active status (superseded, archived, contested) are retrievable
        under NO view: the candidate set stays ``list_active_memories``;
        supersession history is inspected via memory detail, never retrieval.

        Limitation: a naturally expiring memory (``valid_until`` passing with
        no row write) does not move the session-block invalidation watermark,
        which is MAX(updated_at)-based (``Database.memory_state_watermark``,
        database.py:780). A session-pinned compiled block can therefore
        keep showing a naturally-expired fact until the next memory write
        bumps the watermark.
        """
        as_of_dt: datetime | None = None
        if temporal_view == "as_of":
            if as_of is None:
                raise ValueError("temporal_view 'as_of' requires an as_of value")
            as_of_dt = _temporal_bound(as_of)
            if as_of_dt is None:
                raise ValueError(f"invalid as_of: {as_of!r}")
        elif temporal_view not in ("all", "current"):
            raise ValueError(f"unknown temporal_view: {temporal_view!r}")

        if not self.config.memory.enabled or not query.strip():
            return []

        result_limit = limit if limit is not None else self.config.memory.retrieve_candidates
        candidate_limit = max(result_limit, self.config.memory.retrieve_candidates)
        now = datetime.now(timezone.utc)
        # The "as_of" view only reaches here with a validated bound (the block
        # above raises otherwise); other views never read the bound, so `now`
        # stands in to keep the type total.
        # N2 dispatch: the candidate fast path's completeness proof rests on
        # the 0.08 default relevance floor (the phrase-window arms' 4x length
        # window derives SequenceMatcher ratio >= 0.4 from that floor at
        # lexical weight 0.20). A floor below 0.08 widens the window the arms
        # would need beyond 4x, so the floor clause is a completeness
        # precondition, not a performance tweak, and the byte-identical full
        # scan is the honest path there [plan r2 blocker fix]. Likewise at or
        # below full_scan_threshold the store is corpus-gate territory and the
        # verbatim path runs.
        active_count = await self.db.count_active()
        if (
            active_count <= self.config.memory.full_scan_threshold
            or self.config.memory.minimum_relevance_score < 0.08
        ):
            return await self._full_scan(
                query,
                result_limit,
                candidate_limit,
                request_context,
                temporal_view,
                as_of_dt if as_of_dt is not None else now,
                now,
            )
        return await self._candidate_scan(
            query,
            result_limit,
            candidate_limit,
            request_context,
            temporal_view,
            as_of_dt if as_of_dt is not None else now,
            now,
        )

    async def _candidate_scan(
        self,
        query: str,
        result_limit: int,
        candidate_limit: int,
        request_context: RequestContext | None,
        temporal_view: Literal["all", "current", "as_of"],
        as_of_dt: datetime,
        now: datetime,
    ) -> list[ScoredMemory]:
        """Candidate-union fast path: scores only a provably-complete shortlist.

        The arms (``Database.build_retrieval_candidates`` plus the semantic
        membership arm below) cover every way a memory can clear the 0.08
        relevance gate: semantic >= floor (membership arm), token intersection
        (FTS arm), zero-intersection SequenceMatcher ratio >= 0.4 (the 4x
        non-whitespace length windows, whose >8192-char OR-clause also admits
        the lexical-opaque long rows the scorer's phrase cap cannot read), or
        high-authority importance (arm c). Anything else cannot score, so the
        shortlist is a superset of what the full scan would surface and the
        shared scorer settles the rest.

        Learning-path proof: this gate applies to every retriever.search
        consumer (foreground, learner nearby-set, _apply matching, drill-down
        tools, POST /memory/search) yet cannot weaken any reinforcement path
        that can fire today. EVIDENCE-DOMINANCE: every reinforcement firing
        condition requires a relevance signal that IS the same variable this
        gate checks — in _apply, lexical_similarity(candidate.content,
        memory.content) is literally this search's `lexical` for
        query=candidate.content, and the thresholds (lexical >= 0.86/0.40,
        semantic >= 0.90/0.72) all dominate this 0.08 net, so nothing that
        passes a reinforcement guard can fail this gate. A smaller nearby
        set can only ENABLE reinforcements previously blocked by a
        zero-signal compatible[0] match (same guards, smaller pool — never
        weaker). Secondary observation only: with embeddings off the
        hint-path score guard 0.55 sits near the weighted ceiling, but the
        bump and affinity can exceed it, so this is not load-bearing.
        What does change: the extractor's nearby set loses zero-signal
        ambient memories — a quality improvement, not a regression.

        Any degeneracy (arms unavailable/saturated, union over ceiling) or
        arm-building error retreats to the byte-identical full scan via
        ``_fallback_with_reason``, warned at most once per (reason, watermark).
        """
        ceiling = max(self.config.memory.full_scan_threshold, candidate_limit * 3)
        union: list[str] = []
        query_vec: np.ndarray | None = None
        fallback_reason: str | None = None
        try:
            # Union of both tokenizations (regex tokens and the bonus's
            # whitespace chunks) so no term the bonus FTS query would match is
            # missing from the candidate arm.
            terms = sorted({*tokens(query), *query.replace('"', " ").split()})
            arm_ids = await self.db.build_retrieval_candidates(
                terms=terms, qnorm=normalize_text(query), limit=ceiling
            )
            if arm_ids is None:
                fallback_reason = "arms-unavailable-or-saturated"
            else:
                query_vec = await self.embeddings.embed(query)
                union = list(dict.fromkeys([*arm_ids, *await self._semantic_candidates(query_vec)]))
                if len(union) > ceiling:
                    fallback_reason = "union-exceeds-ceiling"
        except Exception:
            fallback_reason = "arm-error"
        if fallback_reason is not None:
            return await self._fallback_with_reason(
                fallback_reason,
                query,
                result_limit,
                candidate_limit,
                request_context,
                temporal_view,
                as_of_dt,
                now,
            )

        log.debug("retrieval fast path: mode=candidates n=%d", len(union))
        rows = await self.db.list_memories_by_ids_active(union)
        if not rows:
            return []
        fts_ids = set(await self.db.fts_memory_ids(query, limit=candidate_limit * 3))
        embedding_map = await self.db.get_embeddings([m.id for m in rows]) if query_vec is not None else {}
        affinity_map = await self.db.memory_context_affinity(
            [m.id for m in rows], request_context
        )
        return await self._score_memories(
            query,
            rows,
            fts_ids,
            query_vec,
            embedding_map,
            affinity_map,
            result_limit,
            temporal_view,
            as_of_dt,
            now,
        )

    async def _semantic_candidates(
        self, query_vec: np.ndarray | None, floor: float | None = None
    ) -> list[str]:
        """Ids whose stored vector clears ``floor`` against the query vector.

        Isolated so the todo-5 watermark cache can replace this bulk fetch
        without touching dispatch. One whole-store vector read (``None`` is
        the fetch-all marker; an empty list would hit the same fetch-all
        branch in ``Database.get_embeddings``, never an empty result). The
        model/dim filter and zero-norm treatment mirror the scorer's own
        guards (cosine 0.0), so this arm is a superset of the scorer's
        semantic gate clears; the ``floor - 1e-9`` margin absorbs 1-ULP drift
        between the matrix dot form and the per-row ``cosine_similarity``.
        """
        if query_vec is None:
            return []
        threshold = (
            self.config.memory.minimum_relevance_score - 1e-9
            if floor is None
            else floor
        )
        rows = await self.db.get_embeddings(None)
        model = self.config.embeddings.model
        dim = int(query_vec.shape[0])
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for memory_id, (row_model, vector) in rows.items():
            if row_model != model or int(vector.shape[0]) != dim:
                continue
            ids.append(memory_id)
            vectors.append(vector)
        if not vectors:
            return []
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        q = np.asarray(query_vec, dtype=np.float64).ravel()
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        cos = (matrix / norms) @ (q / q_norm)
        return [memory_id for memory_id, value in zip(ids, cos) if value >= threshold]

    async def _fallback_with_reason(
        self,
        reason: str,
        query: str,
        result_limit: int,
        candidate_limit: int,
        request_context: RequestContext | None,
        temporal_view: Literal["all", "current", "as_of"],
        as_of_dt: datetime,
        now: datetime,
    ) -> list[ScoredMemory]:
        """Full-scan retreat, warned at most once per (reason, watermark)."""
        key = (reason, await self.db.memory_state_watermark())
        if key not in self._warned_fallbacks:
            self._warned_fallbacks.add(key)
            log.warning(
                "retrieval candidate path fell back to the full-corpus scan"
                " (reason=%s); scoring every active memory costs one full scan"
                " per request until memory state changes",
                reason,
            )
        return await self._full_scan(
            query, result_limit, candidate_limit, request_context, temporal_view, as_of_dt, now
        )

    async def _full_scan(
        self,
        query: str,
        result_limit: int,
        candidate_limit: int,
        request_context: RequestContext | None,
        temporal_view: Literal["all", "current", "as_of"],
        as_of_dt: datetime,
        now: datetime,
    ) -> list[ScoredMemory]:
        """Full-corpus scan behind search: scores every active memory.

        Body relocated verbatim from search (pre-N2 behavior) so the threshold
        dispatch can share this scorer; the ``now`` and ``as_of_dt`` locals are
        computed by search and passed in. The shared scoring loop itself lives
        in ``_score_memories``.
        """
        # ponytail: hard 5000-memory scan cap per retrieval pass, silent above it;
        # upgrade path: progressive/paginated retrieval (docs/ROADMAP.md). See docs/ARCHITECTURE.md.
        active = await self.db.list_active_memories(limit=5000)
        if not active:
            return []

        fts_ids = set(await self.db.fts_memory_ids(query, limit=candidate_limit * 3))
        query_vec = await self.embeddings.embed(query)
        embedding_map = await self.db.get_embeddings([m.id for m in active]) if query_vec is not None else {}
        affinity_map = await self.db.memory_context_affinity(
            [m.id for m in active], request_context
        )
        return await self._score_memories(
            query,
            active,
            fts_ids,
            query_vec,
            embedding_map,
            affinity_map,
            result_limit,
            temporal_view,
            as_of_dt,
            now,
        )

    async def _score_memories(
        self,
        query: str,
        memories: list[Memory],
        fts_ids: set[str],
        query_vec: np.ndarray | None,
        embedding_map: dict[str, tuple[str, np.ndarray]],
        affinity_map: dict[str, tuple[float, float, float]],
        result_limit: int,
        temporal_view: Literal["all", "current", "as_of"],
        as_of_dt: datetime,
        now: datetime,
    ) -> list[ScoredMemory]:
        """Shared scorer: the relocated, unedited pre-N2 scoring loop.

        Both ``_full_scan`` and the candidate fast path call this with their
        own candidate sets; gates, weights, temporal handling, affinity
        placement, and tie-break are byte-identical between the two paths by
        construction (one body, zero copies).
        """
        weights = self.config.retrieval_weights
        weight_total = sum(
            [
                weights.semantic,
                weights.lexical,
                weights.importance,
                weights.confidence,
                weights.freshness,
                weights.topic,
            ]
        )
        scored: list[ScoredMemory] = []
        for memory in memories:
            # Hard temporal eligibility filter, before scoring (pre-ranking
            # eligibility, AGENTS.md invariant 10). Only "as_of" filters here;
            # "current" demotes expired rows after the gates instead, and
            # "all" skips temporal handling entirely. Default "all" is
            # byte-identical to pre-temporal behavior.
            if temporal_view == "as_of":
                start = _temporal_bound(memory.valid_from)
                until = _temporal_bound(memory.valid_until, end_of_day=True)
                if (start is not None and start > as_of_dt) or (
                    until is not None and until <= as_of_dt
                ):
                    continue
            factor = (
                validity_factor(memory.valid_until, now)
                if temporal_view == "current"
                else 1.0
            )
            lexical = lexical_similarity(query, memory.content)
            if memory.id in fts_ids:
                lexical = min(1.0, lexical + 0.15)
            topic = topic_similarity(query, memory.topic)
            semantic = 0.0
            if query_vec is not None and memory.id in embedding_map:
                model, vector = embedding_map[memory.id]
                if model == self.config.embeddings.model:
                    semantic = max(0.0, cosine_similarity(query_vec, vector))
            age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400.0)
            fresh = freshness_score(age_days, self.config.memory.freshness_half_life_days)
            base_score = (
                weights.semantic * semantic
                + weights.lexical * lexical
                + weights.importance * memory.importance
                + weights.confidence * memory.confidence
                + weights.freshness * fresh
                + weights.topic * topic
            ) / weight_total

            # Keep high-authority-like persistent goals/decisions from vanishing
            # solely because their wording differs from today's query.
            if _is_high_authority(memory):
                base_score += 0.04

            # Relevance eligibility gate: drop memories with no genuine query
            # signal (semantic/lexical/topic all below the zero-signal net).
            # lexical already carries the +0.15 FTS bonus above, so FTS hits clear.
            # Learning-path proof: this gate applies to every retriever.search
            # consumer (foreground, learner nearby-set, _apply matching, drill-down
            # tools, POST /memory/search) yet cannot weaken any reinforcement path
            # that can fire today. EVIDENCE-DOMINANCE: every reinforcement firing
            # condition requires a relevance signal that IS the same variable this
            # gate checks — in _apply, lexical_similarity(candidate.content,
            # memory.content) is literally this search's `lexical` for
            # query=candidate.content, and the thresholds (lexical >= 0.86/0.40,
            # semantic >= 0.90/0.72) all dominate this 0.08 net, so nothing that
            # passes a reinforcement guard can fail this gate. A smaller nearby
            # set can only ENABLE reinforcements previously blocked by a
            # zero-signal compatible[0] match (same guards, smaller pool — never
            # weaker). Secondary observation only: with embeddings off the
            # hint-path score guard 0.55 sits near the weighted ceiling, but the
            # bump and affinity can exceed it, so this is not load-bearing.
            # What does change: the extractor's nearby set loses zero-signal
            # ambient memories — a quality improvement, not a regression.
            if not (
                max(semantic, lexical, topic) >= self.config.memory.minimum_relevance_score
                or _is_high_authority(memory)
            ):
                continue

            # Affinity must not make an irrelevant memory eligible by itself.
            # It only reorders memories that already cleared the global relevance
            # floor. This preserves V0.1's global-memory semantics.
            if base_score < self.config.memory.minimum_retrieval_score:
                continue

            user_affinity, project_affinity, cwd_affinity = affinity_map.get(
                memory.id, (0.0, 0.0, 0.0)
            )
            rcfg = self.config.request_context
            affinity_bonus = (
                user_affinity * rcfg.user_affinity_bonus
                + project_affinity * rcfg.project_affinity_bonus
                + cwd_affinity * rcfg.cwd_affinity_bonus
            )
            score = (base_score + affinity_bonus) * factor

            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=min(1.0, score),
                    semantic_score=semantic,
                    lexical_score=lexical,
                    topic_score=topic,
                    freshness_score=fresh,
                    user_affinity_score=user_affinity,
                    project_affinity_score=project_affinity,
                    cwd_affinity_score=cwd_affinity,
                    affinity_bonus=affinity_bonus,
                )
            )

        scored.sort(key=lambda item: (-item.score, item.memory.id))
        return scored[:result_limit]
