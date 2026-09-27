"""Pure clustering primitives for periodic consolidation.

This module is intentionally dependency-pure: it imports nothing from
``database``, ``upstream``, or ``config`` so benchmarks can reuse it. All
thresholds are passed in by the caller (the consolidator wires
``memory.dedup_similarity`` and ``memory.reinforce_semantic_similarity``).
"""

from __future__ import annotations

import math

from infinitum.models import Memory
from infinitum.text import lexical_similarity


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; 0.0 when either is zero-length.

    Vectors of mismatched length are compared over their common prefix.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_clusters(
    memories: list[Memory],
    embeddings: dict[str, list[float]] | None,
    lexical_floor: float,
    semantic_floor: float,
) -> list[list[Memory]]:
    """Group near-duplicate memories via greedy union-find clustering.

    An edge joins a pair iff they share ``memory_type`` AND either
    ``lexical_similarity(a.content, b.content) >= lexical_floor`` or both ids
    have embeddings whose cosine ``>= semantic_floor``. Transitive links merge
    into one cluster. Only multi-member clusters are returned; members are
    sorted by id and clusters are sorted by their first member id, so the
    output is fully deterministic regardless of input order.
    """
    parent = list(range(len(memories)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(memories)):
        a = memories[i]
        for j in range(i + 1, len(memories)):
            b = memories[j]
            if a.memory_type != b.memory_type:
                continue
            if lexical_similarity(a.content, b.content) >= lexical_floor:
                union(i, j)
                continue
            if embeddings:
                va = embeddings.get(a.id)
                vb = embeddings.get(b.id)
                if va is not None and vb is not None and _cosine(va, vb) >= semantic_floor:
                    union(i, j)

    groups: dict[int, list[Memory]] = {}
    for i, mem in enumerate(memories):
        groups.setdefault(find(i), []).append(mem)
    clusters = [sorted(group, key=lambda m: m.id) for group in groups.values() if len(group) > 1]
    clusters.sort(key=lambda c: c[0].id)
    return clusters
