"""Unit tests for the pure clustering primitive in infinitum.consolidation."""

import math

from infinitum.consolidation import build_clusters
from infinitum.models import Memory


def _mem(mid: str, memory_type: str = "fact", content: str = "some content") -> Memory:
    return Memory(id=mid, memory_type=memory_type, content=content)


def test_near_identical_same_type_forms_one_cluster():
    a = _mem("mem_a", "fact", "The team decided to use SQLite for persistence")
    b = _mem("mem_b", "fact", "The team decided to use SQLite for persistence layer")
    clusters = build_clusters([a, b], None, 0.88, 0.90)
    assert len(clusters) == 1
    assert [m.id for m in clusters[0]] == ["mem_a", "mem_b"]


def test_same_content_different_type_does_not_cluster():
    # Invariant-4 type gate: identical text never merges across memory_type.
    a = _mem("mem_a", "fact", "We use Postgres for all storage")
    b = _mem("mem_b", "decision", "We use Postgres for all storage")
    assert build_clusters([a, b], None, 0.88, 0.90) == []


def test_semantic_floor_merges_at_092_not_085():
    # Lexically far apart; only embeddings can bridge the pair.
    a = _mem("mem_a", "fact", "We standardized the storage engine on Postgres")
    b = _mem("mem_b", "fact", "Postgres was chosen as the durable database")
    sin92 = math.sqrt(1 - 0.92**2)
    sin85 = math.sqrt(1 - 0.85**2)
    merged = build_clusters(
        [a, b], {a.id: [1.0, 0.0], b.id: [0.92, sin92]}, 0.88, 0.90
    )
    assert len(merged) == 1
    not_merged = build_clusters(
        [a, b], {a.id: [1.0, 0.0], b.id: [0.85, sin85]}, 0.88, 0.90
    )
    assert not_merged == []


def test_three_way_transitive_merge_is_one_cluster():
    # a~b and b~c clear the lexical floor; a vs c alone does not.
    a = _mem("mem_a", "fact", "Alpha beta gamma delta epsilon")
    b = _mem("mem_b", "fact", "Alpha beta gamma delta epsilon zeta")
    c = _mem("mem_c", "fact", "Alpha beta gamma delta epsilon zeta eta")
    clusters = build_clusters([a, b, c], None, 0.88, 0.90)
    assert len(clusters) == 1
    assert [m.id for m in clusters[0]] == ["mem_a", "mem_b", "mem_c"]


def test_input_order_permutation_yields_identical_output():
    a = _mem("mem_a", "fact", "The team decided to use SQLite for persistence")
    b = _mem("mem_b", "fact", "The team decided to use SQLite for persistence layer")
    c = _mem("mem_c", "lesson", "Retries amplified the outage window")
    d = _mem("mem_d", "lesson", "Retries amplified the outage window badly")
    forward = build_clusters([a, b, c, d], None, 0.88, 0.90)
    reversed_ = build_clusters([d, c, b, a], None, 0.88, 0.90)
    rotated = build_clusters([c, d, a, b], None, 0.88, 0.90)
    assert len(forward) == 2
    assert forward == reversed_ == rotated
    assert [[m.id for m in cl] for cl in forward] == [["mem_a", "mem_b"], ["mem_c", "mem_d"]]


def test_degenerate_inputs_return_no_clusters():
    assert build_clusters([], None, 0.88, 0.90) == []
    lone = _mem("mem_a", "fact", "Only memory in the store")
    assert build_clusters([lone], None, 0.88, 0.90) == []
    # Missing embeddings for one member falls back to lexical-only.
    a = _mem("mem_a", "fact", "We standardized the storage engine on Postgres")
    b = _mem("mem_b", "fact", "Postgres was chosen as the durable database")
    assert build_clusters([a, b], {a.id: [1.0, 0.0]}, 0.88, 0.90) == []
