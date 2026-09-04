from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import Event, Memory, RequestContext, TopicSummary, new_id, utc_now


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Database:
    """Small async facade around a single SQLite connection.

    SQLite work is serialized through an asyncio lock and executed in a worker
    thread. This keeps the V0.1 dependency surface small while remaining safe
    for normal single-process FastAPI use.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self.fts_enabled = False

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._connect_sync)
        await self.initialize()

    def _connect_sync(self) -> None:
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    async def close(self) -> None:
        if self._conn is not None:
            async with self._lock:
                await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS events (
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
        CREATE INDEX IF NOT EXISTS idx_events_session_created
            ON events(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_events_request
            ON events(request_id);

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT 'general',
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            importance REAL NOT NULL DEFAULT 0.5,
            confidence REAL NOT NULL DEFAULT 0.7,
            observation_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT,
            superseded_by TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(superseded_by) REFERENCES memories(id)
        );
        CREATE INDEX IF NOT EXISTS idx_memories_status_updated
            ON memories(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_memories_topic_status
            ON memories(topic, status);

        CREATE TABLE IF NOT EXISTS memory_sources (
            memory_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            PRIMARY KEY(memory_id, event_id),
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS topics (
            topic TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            memory_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Incremental topic-summary maintenance. Each row records a memory that
        -- changed since the last successful summary refresh. This is separate
        -- from the job queue so failed/retried summary calls never lose dirty
        -- state and new changes can accumulate while a summary is running.
        CREATE TABLE IF NOT EXISTS topic_updates (
            topic TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(topic, memory_id),
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_topic_updates_topic_created
            ON topic_updates(topic, created_at);

        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT,
            project_id TEXT,
            cwd TEXT,
            query TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_memories (
            request_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            tokens INTEGER NOT NULL,
            PRIMARY KEY(request_id, memory_id),
            FOREIGN KEY(request_id) REFERENCES requests(id) ON DELETE CASCADE,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            run_after TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            locked_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_pending
            ON jobs(status, run_after, created_at);
        """
        await self.executescript(schema)
        await self._ensure_request_context_columns()
        await self._initialize_fts()

    async def _ensure_request_context_columns(self) -> None:
        """Upgrade V0.1.3-and-earlier databases in place.

        CREATE TABLE IF NOT EXISTS does not add columns to an existing SQLite
        table, so header-aware provenance requires a tiny idempotent migration.
        Nullable columns preserve all existing rows as legacy/global context.
        """

        async with self._lock:
            await asyncio.to_thread(self._ensure_request_context_columns_sync)

    def _ensure_request_context_columns_sync(self) -> None:
        assert self._conn is not None
        for table in ("events", "requests"):
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing = {row["name"] for row in rows}
            for column in ("user_id", "project_id", "cwd"):
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_user_created ON events(user_id, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_project_created ON events(project_id, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_user_created ON requests(user_id, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_project_created ON requests(project_id, created_at)"
        )
        self._conn.commit()

    async def _initialize_fts(self) -> None:
        fts = """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED,
            content,
            topic,
            tokenize='unicode61'
        );
        """
        try:
            await self.executescript(fts)
            self.fts_enabled = True
            # Repair/synchronize the index at startup; memory counts are expected
            # to be modest in V0.1 and this keeps triggers unnecessary.
            async with self._lock:
                await asyncio.to_thread(self._rebuild_fts_sync)
        except sqlite3.OperationalError:
            self.fts_enabled = False

    def _rebuild_fts_sync(self) -> None:
        assert self._conn is not None
        self._conn.execute("DELETE FROM memories_fts")
        self._conn.execute(
            "INSERT INTO memories_fts(memory_id, content, topic) "
            "SELECT id, content, topic FROM memories WHERE status='active'"
        )
        self._conn.commit()

    async def _refresh_fts_memory(self, memory_id: str) -> None:
        if not self.fts_enabled:
            return
        async with self._lock:
            await asyncio.to_thread(self._refresh_fts_memory_sync, memory_id)

    def _refresh_fts_memory_sync(self, memory_id: str) -> None:
        assert self._conn is not None
        self._conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
        row = self._conn.execute(
            "SELECT id, content, topic FROM memories WHERE id=? AND status='active'", (memory_id,)
        ).fetchone()
        if row:
            self._conn.execute(
                "INSERT INTO memories_fts(memory_id, content, topic) VALUES (?, ?, ?)",
                (row["id"], row["content"], row["topic"]),
            )
        self._conn.commit()

    async def executescript(self, script: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._executescript_sync, script)

    def _executescript_sync(self, script: str) -> None:
        assert self._conn is not None
        self._conn.executescript(script)
        self._conn.commit()

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, params)

    def _execute_sync(self, sql: str, params: tuple[Any, ...]) -> None:
        assert self._conn is not None
        self._conn.execute(sql, params)
        self._conn.commit()

    async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        async with self._lock:
            return await asyncio.to_thread(self._fetchone_sync, sql, params)

    def _fetchone_sync(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        assert self._conn is not None
        return self._conn.execute(sql, params).fetchone()

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        async with self._lock:
            return await asyncio.to_thread(self._fetchall_sync, sql, params)

    def _fetchall_sync(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        assert self._conn is not None
        return list(self._conn.execute(sql, params).fetchall())

    # Events -----------------------------------------------------------------
    async def add_event(self, event: Event) -> Event:
        await self.execute(
            "INSERT INTO events(id, session_id, user_id, project_id, cwd, request_id, event_type, role, content, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.session_id,
                event.user_id,
                event.project_id,
                event.cwd,
                event.request_id,
                event.event_type,
                event.role,
                event.content,
                json.dumps(event.metadata, separators=(",", ":")),
                _iso(event.created_at),
            ),
        )
        return event

    async def list_events(
        self,
        limit: int = 100,
        session_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await self.fetchall(
            f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT ?", tuple(params)
        )
        return [self._row_to_event(r) for r in rows]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            cwd=row["cwd"],
            request_id=row["request_id"],
            event_type=row["event_type"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=_dt(row["created_at"]),
        )

    # Memories ---------------------------------------------------------------
    async def create_memory(self, memory: Memory) -> Memory:
        await self.execute(
            "INSERT INTO memories(id, memory_type, topic, content, status, importance, confidence, "
            "observation_count, created_at, updated_at, last_accessed_at, superseded_by, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.id,
                memory.memory_type,
                memory.topic,
                memory.content,
                memory.status,
                memory.importance,
                memory.confidence,
                memory.observation_count,
                _iso(memory.created_at),
                _iso(memory.updated_at),
                _iso(memory.last_accessed_at),
                memory.superseded_by,
                json.dumps(memory.metadata, separators=(",", ":")),
            ),
        )
        for event_id in memory.source_event_ids:
            await self.add_memory_source(memory.id, event_id)
        await self._refresh_fts_memory(memory.id)
        return memory

    async def add_memory_source(self, memory_id: str, event_id: str) -> None:
        await self.execute(
            "INSERT OR IGNORE INTO memory_sources(memory_id, event_id) VALUES (?, ?)",
            (memory_id, event_id),
        )

    async def get_memory(self, memory_id: str) -> Memory | None:
        row = await self.fetchone("SELECT * FROM memories WHERE id=?", (memory_id,))
        return await self._row_to_memory(row) if row else None

    async def list_memories(
        self, limit: int = 100, status: str | None = None, topic: str | None = None
    ) -> list[Memory]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if topic:
            clauses.append("topic=?")
            params.append(topic)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await self.fetchall(
            f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?", tuple(params)
        )
        return [await self._row_to_memory(r) for r in rows]

    async def list_active_memories(self, limit: int = 5000) -> list[Memory]:
        return await self.list_memories(limit=limit, status="active")

    async def list_active_topic_memories(self, topic: str, limit: int = 100) -> list[Memory]:
        return await self.list_memories(limit=limit, status="active", topic=topic)

    async def count_active_topic_memories(self, topic: str) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) AS n FROM memories WHERE status='active' AND topic=?", (topic,)
        )
        return int(row["n"]) if row else 0

    async def list_memories_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = await self.fetchall(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
        )
        by_id = {r["id"]: r for r in rows}
        result: list[Memory] = []
        for memory_id in memory_ids:
            row = by_id.get(memory_id)
            if row is not None:
                result.append(await self._row_to_memory(row))
        return result

    async def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        sources = await self.fetchall(
            "SELECT event_id FROM memory_sources WHERE memory_id=? ORDER BY event_id", (row["id"],)
        )
        return Memory(
            id=row["id"],
            memory_type=row["memory_type"],
            topic=row["topic"],
            content=row["content"],
            status=row["status"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            observation_count=int(row["observation_count"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            last_accessed_at=_dt(row["last_accessed_at"]),
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            source_event_ids=[r["event_id"] for r in sources],
        )

    async def archive_memory(self, memory_id: str) -> bool:
        row = await self.fetchone("SELECT id FROM memories WHERE id=?", (memory_id,))
        if not row:
            return False
        await self.execute(
            "UPDATE memories SET status='archived', updated_at=? WHERE id=?",
            (_iso(utc_now()), memory_id),
        )
        await self._refresh_fts_memory(memory_id)
        return True

    async def supersede_memory(self, old_id: str, new_id: str) -> None:
        await self.execute(
            "UPDATE memories SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
            (new_id, _iso(utc_now()), old_id),
        )
        await self._refresh_fts_memory(old_id)

    async def reinforce_memory(
        self,
        memory_id: str,
        *,
        confidence: float,
        importance: float,
        source_event_ids: list[str],
        reinforcement_metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        memory = await self.get_memory(memory_id)
        if not memory:
            return None

        # observation_count is intended to represent distinct supporting
        # observations, not worker retries. If this exact source interaction has
        # already been attached to the memory, make reinforcement idempotent.
        existing_sources = set(memory.source_event_ids)
        new_source_ids = [
            event_id for event_id in source_event_ids if event_id not in existing_sources
        ]
        if source_event_ids and not new_source_ids:
            return memory

        count = memory.observation_count + 1
        # More evidence should make confidence converge upward without allowing a
        # single noisy candidate to overwrite established state.
        new_conf = min(1.0, memory.confidence + (confidence - memory.confidence) / count + 0.02)
        new_importance = min(1.0, max(memory.importance, importance) + 0.01)
        metadata = dict(memory.metadata)
        if reinforcement_metadata:
            metadata["last_reinforcement"] = {
                **reinforcement_metadata,
                "at": _iso(utc_now()),
            }
        await self.execute(
            "UPDATE memories SET observation_count=?, confidence=?, importance=?, "
            "metadata_json=?, updated_at=? WHERE id=?",
            (
                count,
                new_conf,
                new_importance,
                json.dumps(metadata, separators=(",", ":")),
                _iso(utc_now()),
                memory_id,
            ),
        )
        for event_id in new_source_ids if source_event_ids else source_event_ids:
            await self.add_memory_source(memory_id, event_id)
        await self._refresh_fts_memory(memory_id)
        return await self.get_memory(memory_id)

    async def touch_memories(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = _iso(utc_now())
        placeholders = ",".join("?" for _ in memory_ids)
        await self.execute(
            f"UPDATE memories SET last_accessed_at=? WHERE id IN ({placeholders})",
            tuple([now, *memory_ids]),
        )

    async def memory_state_watermark(self) -> str:
        """Single invalidation watermark for session-pinned compiled blocks.

        Deliberately has NO status filter: archive/supersede bump updated_at on
        every row, so archiving a non-newest memory still moves the watermark
        instead of pinning a block that contains an archived memory forever.
        All stored ISO strings share the +00:00 offset, so the lexicographic MAX
        is also the chronological one. Returns "" when both tables are empty.
        """
        row = await self.fetchone(
            "SELECT MAX(u) AS w FROM ("
            "SELECT updated_at AS u FROM memories "
            "UNION ALL SELECT updated_at AS u FROM topics"
            ") AS w"
        )
        return row["w"] if row and row["w"] is not None else ""

    async def fts_memory_ids(self, query: str, limit: int = 100) -> list[str]:
        if not self.fts_enabled or not query.strip():
            return []
        # Use OR semantics to make memory recall broad. Quote-free tokens avoid
        # exposing arbitrary FTS syntax from the caller.
        terms = [t for t in query.replace("\"", " ").split() if t]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms[:20])
        try:
            rows = await self.fetchall(
                "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                (fts_query, limit),
            )
            return [r["memory_id"] for r in rows]
        except sqlite3.OperationalError:
            return []

    async def memory_context_affinity(
        self, memory_ids: list[str], context: RequestContext | None
    ) -> dict[str, tuple[float, float, float]]:
        """Return exact user/project/CWD provenance matches for memories.

        A canonical global memory may have many source events. A match is true
        when *any* supporting event was observed in the current context. This is
        deliberately an affinity signal only; it never filters memory access.
        """

        if not memory_ids or context is None or context.is_empty:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = await self.fetchall(
            f"SELECT ms.memory_id, "
            "MAX(CASE WHEN e.user_id=? THEN 1 ELSE 0 END) AS user_match, "
            "MAX(CASE WHEN e.project_id=? THEN 1 ELSE 0 END) AS project_match, "
            "MAX(CASE WHEN e.cwd=? THEN 1 ELSE 0 END) AS cwd_match "
            "FROM memory_sources ms JOIN events e ON e.id=ms.event_id "
            f"WHERE ms.memory_id IN ({placeholders}) GROUP BY ms.memory_id",
            tuple([context.user_id, context.project_id, context.cwd, *memory_ids]),
        )
        return {
            row["memory_id"]: (
                float(row["user_match"] or 0),
                float(row["project_match"] or 0),
                float(row["cwd_match"] or 0),
            )
            for row in rows
        }

    # Embeddings -------------------------------------------------------------
    async def set_embedding(self, memory_id: str, model: str, vector: np.ndarray) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        await self.execute(
            "INSERT INTO memory_embeddings(memory_id, model, dim, vector, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET model=excluded.model, dim=excluded.dim, "
            "vector=excluded.vector, updated_at=excluded.updated_at",
            (memory_id, model, int(arr.shape[0]), arr.tobytes(), _iso(utc_now())),
        )

    async def get_embeddings(self, memory_ids: list[str] | None = None) -> dict[str, tuple[str, np.ndarray]]:
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            rows = await self.fetchall(
                f"SELECT memory_id, model, dim, vector FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                tuple(memory_ids),
            )
        else:
            rows = await self.fetchall("SELECT memory_id, model, dim, vector FROM memory_embeddings")
        result: dict[str, tuple[str, np.ndarray]] = {}
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32, count=int(row["dim"])).copy()
            result[row["memory_id"]] = (row["model"], vec)
        return result

    # Topics -----------------------------------------------------------------
    async def upsert_topic(self, summary: TopicSummary) -> None:
        await self.execute(
            "INSERT INTO topics(topic, summary, memory_count, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(topic) DO UPDATE SET summary=excluded.summary, "
            "memory_count=excluded.memory_count, updated_at=excluded.updated_at",
            (summary.topic, summary.summary, summary.memory_count, _iso(summary.updated_at)),
        )

    async def list_topics(self, limit: int = 100) -> list[TopicSummary]:
        rows = await self.fetchall("SELECT * FROM topics ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [
            TopicSummary(
                topic=r["topic"],
                summary=r["summary"],
                memory_count=int(r["memory_count"]),
                updated_at=_dt(r["updated_at"]),
            )
            for r in rows
        ]

    async def get_topic(self, topic: str) -> TopicSummary | None:
        row = await self.fetchone("SELECT * FROM topics WHERE topic=?", (topic,))
        if not row:
            return None
        return TopicSummary(
            topic=row["topic"],
            summary=row["summary"],
            memory_count=int(row["memory_count"]),
            updated_at=_dt(row["updated_at"]),
        )

    # Incremental topic-summary dirty state ----------------------------------
    async def mark_topic_dirty(
        self,
        topic: str,
        memory_ids: list[str],
        *,
        model: str,
        debounce_seconds: float,
        update_threshold: int,
    ) -> None:
        if not topic or not memory_ids:
            return
        async with self._lock:
            await asyncio.to_thread(
                self._mark_topic_dirty_sync,
                topic,
                memory_ids,
                model,
                debounce_seconds,
                update_threshold,
            )

    def _mark_topic_dirty_sync(
        self,
        topic: str,
        memory_ids: list[str],
        model: str,
        debounce_seconds: float,
        update_threshold: int,
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        for memory_id in dict.fromkeys(memory_ids):
            self._conn.execute(
                "INSERT INTO topic_updates(topic, memory_id, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(topic, memory_id) DO UPDATE SET created_at=excluded.created_at",
                (topic, memory_id, now_iso),
            )
        count_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM topic_updates WHERE topic=?", (topic,)
        ).fetchone()
        dirty_count = int(count_row["n"]) if count_row else 0
        due = now if dirty_count >= max(1, update_threshold) else datetime.fromtimestamp(
            now.timestamp() + max(0.0, debounce_seconds), timezone.utc
        )
        self._ensure_topic_summary_job_sync(topic, model, due)
        self._conn.commit()

    def _ensure_topic_summary_job_sync(self, topic: str, model: str, due: datetime) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT id, status, payload_json FROM jobs "
            "WHERE job_type='refresh_topic_summary' AND status IN ('pending','running')"
        ).fetchall()
        pending_id: str | None = None
        running = False
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                continue
            if payload.get("topic") != topic:
                continue
            if row["status"] == "pending":
                pending_id = row["id"]
                break
            if row["status"] == "running":
                running = True
        payload_json = json.dumps({"topic": topic, "model": model}, separators=(",", ":"))
        if pending_id:
            # Debounce: each new change pushes a not-yet-due summary later,
            # except hitting the update threshold above makes it immediately due.
            self._conn.execute(
                "UPDATE jobs SET payload_json=?, run_after=? WHERE id=?",
                (payload_json, due.isoformat(), pending_id),
            )
            return
        if running:
            # Dirty rows remain. The running job will schedule a follow-up after
            # it clears only the exact update IDs it successfully summarized.
            return
        job_id = new_id("job")
        now_iso = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after, attempts) "
            "VALUES (?, 'refresh_topic_summary', 'pending', ?, ?, ?, 0)",
            (job_id, payload_json, now_iso, due.isoformat()),
        )

    async def ensure_topic_summary_job(
        self,
        topic: str,
        *,
        model: str,
        debounce_seconds: float,
        update_threshold: int,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._ensure_topic_summary_job_if_dirty_sync,
                topic,
                model,
                debounce_seconds,
                update_threshold,
            )

    def _ensure_topic_summary_job_if_dirty_sync(
        self,
        topic: str,
        model: str,
        debounce_seconds: float,
        update_threshold: int,
    ) -> None:
        assert self._conn is not None
        self._conn.execute("BEGIN IMMEDIATE")
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM topic_updates WHERE topic=?", (topic,)
        ).fetchone()
        count = int(row["n"]) if row else 0
        if count:
            now = datetime.now(timezone.utc)
            due = now if count >= max(1, update_threshold) else datetime.fromtimestamp(
                now.timestamp() + max(0.0, debounce_seconds), timezone.utc
            )
            self._ensure_topic_summary_job_sync(topic, model, due)
        self._conn.commit()

    async def recover_dirty_topic_summary_jobs(
        self,
        *,
        default_model: str = "",
    ) -> int:
        """Requeue dirty topics that have no pending/running summary job.

        Failed summary jobs intentionally leave ``topic_updates`` intact. A code
        upgrade or process restart should therefore be able to resume those
        topics without waiting for another interaction to touch the same topic.
        The most recent summary-job payload is used to recover the model when
        learning.model is not explicitly configured.
        """

        async with self._lock:
            return await asyncio.to_thread(
                self._recover_dirty_topic_summary_jobs_sync, default_model
            )

    def _recover_dirty_topic_summary_jobs_sync(self, default_model: str) -> int:
        assert self._conn is not None
        topics = [
            str(row["topic"])
            for row in self._conn.execute(
                "SELECT DISTINCT topic FROM topic_updates ORDER BY topic"
            ).fetchall()
        ]
        if not topics:
            return 0

        job_rows = self._conn.execute(
            "SELECT status, payload_json, created_at FROM jobs "
            "WHERE job_type='refresh_topic_summary' ORDER BY created_at DESC"
        ).fetchall()
        recovered = 0
        now = datetime.now(timezone.utc)
        self._conn.execute("BEGIN IMMEDIATE")
        for topic in topics:
            model = default_model
            for row in job_rows:
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    continue
                if payload.get("topic") != topic:
                    continue
                if not model:
                    model = str(payload.get("model") or "")
                # A pending/running job already owns this topic.
                if row["status"] in {"pending", "running"}:
                    model = ""
                break
            if not model:
                continue
            self._ensure_topic_summary_job_sync(topic, model, now)
            recovered += 1
        self._conn.commit()
        return recovered

    async def get_topic_updates(self, topic: str, limit: int = 24) -> list[tuple[str, str]]:
        rows = await self.fetchall(
            "SELECT memory_id, created_at FROM topic_updates WHERE topic=? ORDER BY created_at LIMIT ?",
            (topic, limit),
        )
        return [(r["memory_id"], r["created_at"]) for r in rows]

    async def clear_topic_updates(self, topic: str, updates: list[tuple[str, str]]) -> None:
        if not updates:
            return
        # Delete only the exact revision that was summarized. If the same memory
        # was reinforced again while the LLM call was running, mark_topic_dirty
        # refreshed created_at and this DELETE intentionally leaves it dirty.
        async with self._lock:
            await asyncio.to_thread(self._clear_topic_updates_sync, topic, updates)

    def _clear_topic_updates_sync(self, topic: str, updates: list[tuple[str, str]]) -> None:
        assert self._conn is not None
        self._conn.execute("BEGIN IMMEDIATE")
        for memory_id, created_at in updates:
            self._conn.execute(
                "DELETE FROM topic_updates WHERE topic=? AND memory_id=? AND created_at=?",
                (topic, memory_id, created_at),
            )
        self._conn.commit()

    async def count_topic_updates(self, topic: str) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) AS n FROM topic_updates WHERE topic=?", (topic,)
        )
        return int(row["n"]) if row else 0

    # Requests / retrieval audit --------------------------------------------
    async def add_request(
        self,
        request_id: str,
        session_id: str,
        query: str,
        model: str,
        context: RequestContext | None = None,
    ) -> None:
        context = context or RequestContext()
        await self.execute(
            "INSERT INTO requests(id, session_id, user_id, project_id, cwd, query, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                session_id,
                context.user_id,
                context.project_id,
                context.cwd,
                query,
                model,
                _iso(utc_now()),
            ),
        )

    async def add_request_memory(
        self, request_id: str, memory_id: str, rank: int, score: float, tokens: int
    ) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO request_memories(request_id, memory_id, rank, score, tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            (request_id, memory_id, rank, score, tokens),
        )

    # Jobs -------------------------------------------------------------------
    async def enqueue_job(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = new_id("job")
        now = utc_now()
        await self.execute(
            "INSERT INTO jobs(id, job_type, status, payload_json, created_at, run_after, attempts) "
            "VALUES (?, ?, 'pending', ?, ?, ?, 0)",
            (job_id, job_type, json.dumps(payload), _iso(now), _iso(now)),
        )
        return job_id

    async def claim_job(self) -> dict[str, Any] | None:
        async with self._lock:
            return await asyncio.to_thread(self._claim_job_sync)

    def _claim_job_sync(self) -> dict[str, Any] | None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status='pending' AND run_after<=? ORDER BY created_at LIMIT 1", (now,)
        ).fetchone()
        if not row:
            self._conn.commit()
            return None
        self._conn.execute(
            "UPDATE jobs SET status='running', attempts=attempts+1, locked_at=? WHERE id=?",
            (now, row["id"]),
        )
        self._conn.commit()
        return {
            "id": row["id"],
            "job_type": row["job_type"],
            "payload": json.loads(row["payload_json"]),
            "attempts": int(row["attempts"]) + 1,
        }

    async def finish_job(self, job_id: str) -> None:
        await self.execute("UPDATE jobs SET status='done', locked_at=NULL WHERE id=?", (job_id,))

    async def fail_job(self, job_id: str, error: str, *, retry: bool, delay_seconds: float = 5.0) -> None:
        if retry:
            run_after = datetime.now(timezone.utc).timestamp() + delay_seconds
            run_dt = datetime.fromtimestamp(run_after, timezone.utc)
            await self.execute(
                "UPDATE jobs SET status='pending', last_error=?, run_after=?, locked_at=NULL WHERE id=?",
                (error[:4000], _iso(run_dt), job_id),
            )
        else:
            await self.execute(
                "UPDATE jobs SET status='failed', last_error=?, locked_at=NULL WHERE id=?",
                (error[:4000], job_id),
            )
