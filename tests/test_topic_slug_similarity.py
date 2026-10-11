"""Pinned case table for topic_slug_similarity (N1 plan todo 1).

Every value below was hand-derived from the plan formula and re-verified by
review rounds 1-3; the 0.5/0.5 blend, the both-sided digit veto, and the
numeric-token exclusion are load-bearing. Case (vii) is the fixture-separation
margin pin: if it ever reaches >= 0.85, a formula/weight change would silently
remap existing test fixtures, so it must fail HERE first.
"""

import pytest

from infinitum.text import topic_slug_similarity, topic_slug_tokens


def test_tokens_split_and_drop_empty() -> None:
    assert topic_slug_tokens("Web-App_Deploy /v2") == ["web", "app", "deploy", "v2"]
    assert topic_slug_tokens("") == []
    assert topic_slug_tokens("--  --") == []


def test_undated_vs_dated_variant_scores_one() -> None:
    # Numeric tokens are excluded from scoring, so both sides reduce to
    # {a, b, c, execution, start}: intersection 5, union 5, min 5 -> 1.0.
    # The veto does not fire because the undated side has no digit sequences.
    assert topic_slug_similarity(
        "a-b-c-execution-start", "a-b-c-execution-start-2026-09-24"
    ) == pytest.approx(1.0)
    assert topic_slug_similarity(
        "a-b-c-execution-start", "a-b-c-execution-start-2026-09-24"
    ) >= 0.90


def test_digit_veto_kills_date_distinct_slugs() -> None:
    # Both sides carry digit sequences and the multisets differ
    # ([2026, 09, 24] vs [2026, 10, 01]) -> hard 0.0.
    assert topic_slug_similarity(
        "runbook-start-2026-09-24", "runbook-start-2026-10-01"
    ) == 0.0


def test_prefix_equivalence_collapses_morphological_variants() -> None:
    # deploy ~ deployment (prefix "deploy", length 6 >= 4): every token pairs,
    # intersection 3, union 3 -> 1.0.
    assert topic_slug_similarity("web-app-deploy", "web-app-deployment") == pytest.approx(1.0)
    assert topic_slug_similarity("web-app-deploy", "web-app-deployment") >= 0.85


def test_single_letter_suffix_stays_below_floor() -> None:
    # {session, store, a} vs {session, store, b}: "a"/"b" are scoring members
    # but neither identical nor a >= 4-char prefix, so intersection 2,
    # union 3 + 3 - 2 = 4, min 3 -> 0.5*(2/4) + 0.5*(2/3) = 0.5833... < 0.85.
    score = topic_slug_similarity("session-store-a", "session-store-b")
    assert score == pytest.approx(0.5 * (2 / 4) + 0.5 * (2 / 3))
    assert score < 0.85


def test_empty_slugs_score_zero() -> None:
    assert topic_slug_similarity("", "x") == 0.0
    assert topic_slug_similarity("", "") == 0.0


def test_similarity_is_symmetric() -> None:
    pairs = [
        ("a-b-c-execution-start", "a-b-c-execution-start-2026-09-24"),
        ("runbook-start-2026-09-24", "runbook-start-2026-10-01"),  # vetoed pair
        ("web-app-deploy", "web-app-deployment"),
    ]
    for a, b in pairs:
        assert topic_slug_similarity(a, b) == topic_slug_similarity(b, a)


def test_margin_pin_session_store_variants_stay_below_floor() -> None:
    # {session, store} vs {session, store, a}: intersection 2, union 3,
    # min 2 -> 0.5*(2/3) + 0.5*(2/2) = 0.8333... < 0.85. This thin margin is
    # what keeps existing test fixtures from remapping at the default floor.
    score = topic_slug_similarity("session store", "session-store-a")
    assert score == pytest.approx(0.5 * (2 / 3) + 0.5 * (2 / 2))
    assert score < 0.85
