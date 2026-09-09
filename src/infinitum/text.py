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
