"""Storage-level tests for event compression (issue #32).

Todo 1: the additive ``events.content_blob BLOB`` column and the
``meta(key, value)`` table. Todo 2: the transparent lzma write/read round-trip
in ``add_event`` / ``_row_to_event`` — rows above the 2048-character threshold
are stored as ``content=''`` plus a header-tagged blob and must reconstruct
byte-identically; a foreign codec header must fail loud.
"""

import json
import sqlite3
import tempfile

import pytest

from infinitum.database import Database
from infinitum.models import Event

# The pre-change events table: the exact 11-column shape initialize() used to
# create before content_blob existed.
OLD_EVENTS_SCHEMA = """
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,
    project_id TEXT,
    cwd TEXT,
    request_id TEXT,
    event_type TEXT NOT NULL,
    role TEXT,
    content TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def _build_old_schema_db(path: str) -> list[tuple[str, str]]:
    """Create a pre-content_blob database with two event rows; return them."""
    rows = [
        ("evt-one", "first event content"),
        ("evt-two", "second event content with unicode: ✅ mémoire"),
    ]
    conn = sqlite3.connect(path)
    conn.execute(OLD_EVENTS_SCHEMA)
    conn.executemany(
        "INSERT INTO events (id, session_id, event_type, role, content,"
        " metadata_json, created_at) VALUES (?, 's', 'message.user', 'user',"
        " ?, '{}', '2026-01-01T00:00:00+00:00')",
        rows,
    )
    conn.commit()
    conn.close()
    return rows


def _events_probe(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


async def test_old_schema_database_when_initialized_gains_content_blob_and_meta():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        expected = _build_old_schema_db(path)

        db = Database(path)
        await db.connect()

        probe = _events_probe(path)
        columns = {
            row["name"]: row["type"]
            for row in probe.execute("PRAGMA table_info(events)")
        }
        assert "content_blob" in columns
        assert columns["content_blob"] == "BLOB"

        tables = {
            row["name"]
            for row in probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "meta" in tables

        stored = [
            (row["id"], row["content"])
            for row in probe.execute("SELECT id, content FROM events ORDER BY id")
        ]
        assert stored == expected
        probe.close()
        await db.close()


async def test_second_initialize_when_column_present_is_a_noop():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        expected = _build_old_schema_db(path)

        db = Database(path)
        await db.connect()
        await db.initialize()  # second boot on the same open connection
        await db.close()

        probe = _events_probe(path)
        columns = {row["name"] for row in probe.execute("PRAGMA table_info(events)")}
        assert "content_blob" in columns
        stored = [
            (row["id"], row["content"])
            for row in probe.execute("SELECT id, content FROM events ORDER BY id")
        ]
        assert stored == expected
        probe.close()


async def test_reopen_after_migration_when_column_present_does_not_rerun_alter():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        _build_old_schema_db(path)

        db = Database(path)
        await db.connect()
        await db.close()

        # Reopen the already-migrated database: initialize() must neither
        # ALTER nor raise, and the column set must be unchanged.
        db2 = Database(path)
        await db2.connect()
        probe = _events_probe(path)
        columns = {row["name"] for row in probe.execute("PRAGMA table_info(events)")}
        assert "content_blob" in columns
        count = probe.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2
        probe.close()
        await db2.close()


# --- Todo 2: transparent lzma write/read round-trip -------------------------


def _big_content(n: int) -> str:
    """Deterministic n-char string with enough variety to compress."""
    return "".join(chr(ord("a") + (i % 26)) for i in range(n))


async def test_large_content_round_trips_when_written_via_add_event():
    """3000-char content: list_events is byte-identical AND raw storage is a blob."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        content = _big_content(3000)

        db = Database(path)
        await db.connect()
        await db.add_event(
            Event(session_id="s", event_type="request.received", content=content)
        )

        events = await db.list_events()
        assert [e.content for e in events] == [content]

        probe = _events_probe(path)
        row = probe.execute(
            "SELECT content, content_blob IS NOT NULL AS has_blob,"
            " hex(substr(content_blob, 1, 1)) AS head FROM events"
        ).fetchone()
        assert row["content"] == ""
        assert row["has_blob"] == 1
        assert row["head"] == "01"  # _LZMA_HEADER on disk
        probe.close()
        await db.close()


async def test_row_at_threshold_when_content_is_2048_chars_stays_plain():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        content = "x" * 2048

        db = Database(path)
        await db.connect()
        await db.add_event(Event(session_id="s", event_type="message.user", content=content))

        probe = _events_probe(path)
        row = probe.execute(
            "SELECT content, content_blob IS NULL AS is_plain FROM events"
        ).fetchone()
        assert row["is_plain"] == 1
        assert row["content"] == content
        probe.close()
        await db.close()


async def test_row_over_threshold_when_content_is_2049_chars_is_compressed():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        content = "x" * 2049

        db = Database(path)
        await db.connect()
        await db.add_event(Event(session_id="s", event_type="message.user", content=content))

        events = await db.list_events()
        assert [e.content for e in events] == [content]

        probe = _events_probe(path)
        row = probe.execute(
            "SELECT content, content_blob IS NOT NULL AS has_blob FROM events"
        ).fetchone()
        assert row["content"] == ""
        assert row["has_blob"] == 1
        probe.close()
        await db.close()


async def test_unicode_json_payload_round_trips_when_compressed():
    """CJK/emoji JSON written with ensure_ascii=False survives the blob exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        content = json.dumps(
            {"messages": [{"role": "user", "content": "記憶テスト ✅ 🚀 mémoire" * 200}]},
            ensure_ascii=False,
        )
        assert len(content) > 2048  # the payload must actually cross the threshold

        db = Database(path)
        await db.connect()
        await db.add_event(
            Event(session_id="s", event_type="request.received", content=content)
        )

        events = await db.list_events()
        assert [e.content for e in events] == [content]
        assert json.loads(events[0].content) == json.loads(content)
        await db.close()


async def test_large_content_round_trips_when_read_after_reopen():
    """Fresh DB, write then close/reopen: the row still reconstructs byte-identically."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        content = _big_content(3000)

        db = Database(path)
        await db.connect()
        await db.add_event(Event(session_id="s", event_type="request.received", content=content))
        await db.close()

        db2 = Database(path)
        await db2.connect()
        events = await db2.list_events()
        assert [e.content for e in events] == [content]
        await db2.close()


async def test_foreign_codec_header_when_read_raises_value_error_twice():
    """A crafted 0x02-header blob fails loud, identically on every read attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/runtime.db"
        db = Database(path)
        await db.connect()
        await db.close()

        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO events (id, session_id, event_type, content, content_blob,"
            " metadata_json, created_at) VALUES"
            " ('evt-bad', 's', 'request.received', ?, ?, '{}',"
            " '2026-01-01T00:00:00+00:00')",
            ("", b"\x02not-lzma-at-all"),
        )
        conn.commit()
        conn.close()

        db2 = Database(path)
        await db2.connect()
        with pytest.raises(ValueError, match="unknown event content codec"):
            await db2.list_events()
        # Repeated reads of the same poisoned row raise identically.
        with pytest.raises(ValueError, match="unknown event content codec"):
            await db2.list_events()
        await db2.close()
