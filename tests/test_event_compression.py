"""Storage-level tests for the event-compression schema migration (issue #32).

Todo 1 scope only: the additive ``events.content_blob BLOB`` column and the
``meta(key, value)`` table. No compression logic exists yet, so migrated rows
must be byte-identical to what was written under the old schema.
"""

import sqlite3
import tempfile

from infinitum.database import Database

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
        # re-ALTER nor raise, and the column set must be unchanged.
        db2 = Database(path)
        await db2.connect()
        probe = _events_probe(path)
        columns = {row["name"] for row in probe.execute("PRAGMA table_info(events)")}
        assert "content_blob" in columns
        count = probe.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2
        probe.close()
        await db2.close()
