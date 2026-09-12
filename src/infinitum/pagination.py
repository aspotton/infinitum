"""Opaque keyset-pagination cursor codec shared by the list endpoints.

A cursor is an urlsafe-base64 JSON pair ``[sort_value, tiebreak_id]`` naming
the last row of the previous page. It is a position hint, not a security
token: decode validates structure only and fails loudly with ValueError.
"""
from __future__ import annotations

import base64
import binascii
import json


def encode_cursor(sort_value: str, tiebreak_id: str) -> str:
    raw = json.dumps([sort_value, tiebreak_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid cursor") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not all(isinstance(item, str) for item in payload)
    ):
        raise ValueError("invalid cursor")
    return payload[0], payload[1]
