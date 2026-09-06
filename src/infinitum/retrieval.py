from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .config import AppConfig
from .database import Database
from .embeddings import EmbeddingClient
from .models import Memory, RequestContext, ScoredMemory
from .text import freshness_score, lexical_similarity, topic_similarity


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b) / denom)))


def _is_high_authority(memory: Memory) -> bool:
    """High-authority persistent goals/decisions that survive wording mismatch."""
    return memory.memory_type in {"goal", "decision"} and memory.importance >= 0.85


class MemoryRetriever:
    def __init__(self, db: Database, embeddings: EmbeddingClient, config: AppConfig):
        self.db = db
        self.embeddings = embeddings
        self.config = config

    async def search(
        self,
        query: str,
        limit: int | None = None,
        request_context: RequestContext | None = None,
    ) -> list[ScoredMemory]:
        if not self.config.memory.enabled or not query.strip():
            return []

        result_limit = limit if limit is not None else self.config.memory.retrieve_candidates
        candidate_limit = max(result_limit, self.config.memory.retrieve_candidates)
        active = await self.db.list_active_memories(limit=5000)
        if not active:
            return []

        fts_ids = set(await self.db.fts_memory_ids(query, limit=candidate_limit * 3))
        query_vec = await self.embeddings.embed(query)
        embedding_map = await self.db.get_embeddings([m.id for m in active]) if query_vec is not None else {}
        affinity_map = await self.db.memory_context_affinity(
            [m.id for m in active], request_context
        )

        now = datetime.now(timezone.utc)
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
        for memory in active:
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
            score = base_score + affinity_bonus

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
