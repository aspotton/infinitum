import base64

import pytest

from infinitum.pagination import decode_cursor, encode_cursor


def test_cursor_roundtrip():
    """Given a sort value and tiebreaker id, When encoding then decoding a
    cursor, Then the original pair comes back unchanged.
    """
    pair = ("2026-01-01T00:00:00+00:00", "mem_9")
    assert decode_cursor(encode_cursor(*pair)) == pair


def test_decode_rejects_garbage():
    """Given non-base64 and base64-but-wrong-shape payloads, When decoding,
    Then each raises ValueError: a dict, a one-element list, a list of
    non-strings, and a bare string JSON scalar are all rejected.
    """
    garbage = [
        "!!!",
        base64.urlsafe_b64encode(b'{"a":1}').decode(),
        base64.urlsafe_b64encode(b'["one"]').decode(),
        base64.urlsafe_b64encode(b"[1,2]").decode(),
        base64.urlsafe_b64encode(b'""').decode(),
    ]
    for cursor in garbage:
        with pytest.raises(ValueError):
            decode_cursor(cursor)
