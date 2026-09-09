"""Characterization tests for the bounded phrase term in lexical_similarity.

_reference is a verbatim inline copy of the pre-fix scorer (src/infinitum/text.py
lines 11-29 at time of writing) so these tests pin the OLD behavior: short-vs-short
scores must stay byte-identical, and huge/lopsided pairs may only drift within the
0.005 phrase-weight*bound ceiling while the O(n*m) SequenceMatcher scan is skipped.
"""

import re
import time
from difflib import SequenceMatcher

import pytest

from infinitum.text import lexical_similarity, tokens

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+.#:/-]*")


def _reference(a: str, b: str) -> float:
    ta = {m.group(0).lower() for m in _WORD_RE.finditer(a)}
    tb = {m.group(0).lower() for m in _WORD_RE.finditer(b)}
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    jaccard = intersection / union if union else 0.0
    containment = intersection / max(1, min(len(ta), len(tb)))
    na = " ".join(a.lower().strip().split())
    nb = " ".join(b.lower().strip().split())
    phrase = SequenceMatcher(None, na, nb).ratio()
    return min(1.0, 0.45 * jaccard + 0.35 * containment + 0.20 * phrase)


def test_short_pairs_unchanged() -> None:
    pairs = [
        ("We decided to use PostgreSQL for the database", "database choice postgres"),
        ("The user prefers dark mode in all editors", "preference editor theme"),
        ("Decision: adopt SQLite WAL mode for the runtime database", "sqlite wal decision"),
        ("Project uses pnpm workspaces", "We use pnpm workspaces for the monorepo packages"),
        ("note note note note alpha", "note beta note note gamma"),
        ("café naïve über", "naive cafe"),
        ("", "hello world"),
        ("   ", "hello   world"),
    ]
    for a, b in pairs:
        assert lexical_similarity(a, b) == pytest.approx(_reference(a, b), abs=0.0)


def test_lopsided_pair_within_phrase_bound() -> None:
    q = " ".join(f"note{i}" for i in range(15000))
    m = ("We decided to keep the schema additive and reversible. " * 25) + " note42 note7"
    assert abs(lexical_similarity(q, m) - _reference(q, m)) <= 0.005 + 1e-9


def test_huge_query_scan_bounded() -> None:
    # RED before the fix: ~165s measured against a production DB copy shape
    # (each pair runs the full O(n*m) SequenceMatcher scan).
    # GREEN after the fix: ~0.03s (the bound-skip drops the phrase term).
    query = " ".join(f"frag{i}" for i in range(24000))
    memories = [
        " ".join(
            [f"frag{(j * 97 + k * 7) % 24000}" for k in range(10)]
            + [f"m{j}w{k}" for k in range(170)]
        )
        for j in range(1000)
    ]
    start = time.perf_counter()
    for m in memories:
        lexical_similarity(query, m)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"huge-query scan took {elapsed:.1f}s"


def test_tokens_are_hashable_frozenset() -> None:
    # Passes only after todo 2 (lru_cache on tokens requires a hashable return).
    assert isinstance(tokens("x y"), frozenset)
