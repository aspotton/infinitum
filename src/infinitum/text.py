"""Scoring primitives shared by retrieval, reinforcement, and topic matching.

Normalization and tokenization, bounded lexical/topic similarity, freshness
decay, and slug-aware topic-slug equivalence (``topic_slug_similarity``) used
by topic canonicalization. Retrieval-facing scorers (``lexical_similarity``,
``topic_similarity``) are independent of the slug scorer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+.#:/-]*")

_PHRASE_WEIGHT = 0.20
_PHRASE_EPS = 0.005
# Backstop for mid-size lopsided pairs the bound-skip misses; the real
# ceiling on learn-query length is learning.max_tokens.
_PHRASE_MAX_CHARS = 8192

@lru_cache(maxsize=64)
def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

@lru_cache(maxsize=64)
def tokens(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _WORD_RE.finditer(text))

def lexical_similarity(a: str, b: str) -> float:
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta) + len(tb) - intersection   # exact |ta | tb|, no set alloc
    jaccard = intersection / union if union else 0.0
    containment = intersection / max(1, min(len(ta), len(tb)))
    na = normalize_text(a)
    nb = normalize_text(b)
    # ratio() <= 2*min/(la+lb); when that bound cannot move the score by
    # _PHRASE_EPS, skip the O(n*m) scan entirely (huge-query vs small memory).
    if _PHRASE_WEIGHT * (2 * min(len(na), len(nb)) / (len(na) + len(nb) or 1)) < _PHRASE_EPS:
        phrase = 0.0
    else:
        phrase = SequenceMatcher(None, na[:_PHRASE_MAX_CHARS], nb[:_PHRASE_MAX_CHARS]).ratio()
    return min(1.0, 0.45 * jaccard + 0.35 * containment + _PHRASE_WEIGHT * phrase)


def dedup_similarity(a: str, b: str) -> float:
    return lexical_similarity(a, b)


def topic_similarity(query: str, topic: str) -> float:
    if not topic:
        return 0.0
    return lexical_similarity(query, topic)


_SLUG_SPLIT_RE = re.compile(r"[-_ /]+")
_DIGITS_RE = re.compile(r"\d+")
_PREFIX_MIN = 4


def topic_slug_tokens(slug: str) -> list[str]:
    """Split a topic slug into lowercase tokens on ``-_`` , space, or slash.

    Empty tokens are dropped; the caller-facing contract is order-preserving
    split-then-filter.
    """
    return [tok for tok in _SLUG_SPLIT_RE.split(slug.lower()) if tok]


def _token_equivalent(x: str, y: str) -> bool:
    """True when two slug tokens count as the same for scoring.

    Identical tokens, or one a prefix of the other with shared prefix length
    >= 4 (``deploy`` ~ ``deployment``; a 1-char token never pairs by prefix).
    """
    if x == y:
        return True
    shorter, longer = (x, y) if len(x) <= len(y) else (y, x)
    return len(shorter) >= _PREFIX_MIN and longer.startswith(shorter)


# ponytail: lexical-only ceiling (slug tokens + digit veto); the upgrade path
# is embedding-based topic clustering if slug variants ever evade this.
def topic_slug_similarity(a: str, b: str) -> float:
    """Deterministic 0..1 equivalence between two topic slugs.

    Rules (fixed by the N1 review rounds; do not "fix" the weights):

    1. ``0.0`` when either slug yields no tokens.
    2. Hard digit veto: ``0.0`` only when BOTH slugs carry digit sequences
       (``\\d+``) and the multisets differ — date-distinct slugs never merge,
       while an undated slug vs a dated slug is NOT vetoed.
    3. Purely numeric tokens are excluded from scoring (the veto already ruled
       on digit content), and tokens pair GREEDY PAIRWISE via
       ``_token_equivalent`` — each token matches at most one counterpart, no
       transitive collapse of prefix chains.
    4. Score = ``0.5*jaccard + 0.5*containment`` over the pairwise-collapsed
       token sets. The 0.5/0.5 blend is deliberate: a containment-heavier
       blend pushes ``session store`` vs ``session-store-a`` over the
       canonicalization floor and would remap existing fixtures.
    """
    tokens_a = [tok for tok in topic_slug_tokens(a) if not tok.isdigit()]
    tokens_b = [tok for tok in topic_slug_tokens(b) if not tok.isdigit()]
    if not tokens_a or not tokens_b:
        return 0.0
    digits_a = _DIGITS_RE.findall(a)
    digits_b = _DIGITS_RE.findall(b)
    if digits_a and digits_b and sorted(digits_a) != sorted(digits_b):
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    common = set_a & set_b
    remaining_a = set_a - common
    remaining_b = set_b - common
    # Greedy pairwise: exact matches are forced (identical tokens have unique
    # counterparts), then each leftover token takes the first (sorted)
    # prefix-equivalent leftover.
    paired = 0
    for tok in sorted(remaining_a):
        for other in sorted(remaining_b):
            if _token_equivalent(tok, other):
                remaining_b.discard(other)
                paired += 1
                break
    intersection = len(common) + paired
    union = len(set_a) + len(set_b) - intersection
    jaccard = intersection / union if union else 0.0
    containment = intersection / max(1, min(len(set_a), len(set_b)))
    return 0.5 * jaccard + 0.5 * containment


def freshness_score(age_days: float, half_life_days: float) -> float:
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def first_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"} and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def join_nonempty(values: Iterable[str], sep: str = "\n") -> str:
    return sep.join(v for v in values if v)
