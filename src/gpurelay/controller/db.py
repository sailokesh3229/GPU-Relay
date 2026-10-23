"""Controller state in SQLite at ``$GPURELAY_HOME/state.db`` (spec §13.1).

Tables: ``jobs``, ``segments``, ``checkpoints``, ``quota_ledger``, ``events``, and
``schema_version``.
Jobs and segments are stored as validated JSON documents next to the indexed columns the
controller queries, so model fields can evolve without table rewrites. Migrations are
plain SQL scripts keyed by version.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gpurelay.models import Job, JobState, Segment, SegmentState, utcnow

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        priority INTEGER NOT NULL,
        submitted_at TEXT NOT NULL,
        data TEXT NOT NULL
    );
    CREATE INDEX jobs_state ON jobs(state);
    CREATE TABLE segments (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs(id),
        idx INTEGER NOT NULL,
        backend TEXT NOT NULL,
        state TEXT NOT NULL,
        data TEXT NOT NULL
    );
    CREATE INDEX segments_job ON segments(job_id);
    CREATE INDEX segments_state ON segments(state);
    CREATE TABLE checkpoints (
        job_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        segment_id TEXT,
        created_at TEXT,
        PRIMARY KEY (job_id, seq)
    );
    CREATE TABLE quota_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backend TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('usage', 'correction')),
        start_ts REAL,
        end_ts REAL,
        hours REAL,
        remaining_h REAL,
        used_h REAL,
        segment_id TEXT,
        note TEXT,
        created_ts REAL NOT NULL
    );
    CREATE INDEX ledger_backend ON quota_ledger(backend);
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        kind TEXT NOT NULL,
        job_id TEXT,
        segment_id TEXT,
        message TEXT NOT NULL
    );
    CREATE INDEX events_job ON events(job_id);
    """,
}

ACTIVE_SEGMENT_STATES = (
    SegmentState.LAUNCHING,
    SegmentState.RUNNING,
    SegmentState.AWAITING_USER,
)


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    backend: str
    kind: str  # "usage" | "correction"
    start_ts: float | None
    end_ts: float | None
    hours: float | None
    remaining_h: float | None
    used_h: float | None
    segment_id: str | None
    note: str | None
    created_ts: float


@dataclass(frozen=True)
class Event:
    id: int
    ts: float
    kind: str
    job_id: str | None
    segment_id: str | None
    message: str


class StateDB:
    """Thread-safe (single connection + lock) access to the controller database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ schema

    def _migrate(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for version in sorted(v for v in _MIGRATIONS if v > current):
            self._conn.execute("BEGIN")
            try:
                # executescript() would COMMIT implicitly; run statements one by one instead.
                for statement in _MIGRATIONS[version].split(";"):
                    if statement.strip():
                        self._conn.execute(statement)
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    @property
    def schema_version(self) -> int:
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return int(row["v"] or 0)

    # ------------------------------------------------------------------ jobs

    def put_job(self, job: Job) -> Job:
        """Insert or replace a job (``updated_at`` is refreshed)."""
        job = job.model_copy(update={"updated_at": utcnow()})
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs (id, state, priority, submitted_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.state.value,
                    job.priority,
                    job.submitted_at.isoformat(),
                    job.model_dump_json(),
                ),
            )
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row["data"]) if row else None

    def find_job(self, prefix: str) -> Job | None:
        """A job by id or unique id prefix (``job-1a2b``)."""
        job = self.get_job(prefix)
        if job is not None:
            return job
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM jobs WHERE id LIKE ? ESCAPE '\\'",
                (prefix.replace("_", "\\_").replace("%", "\\%") + "%",),
            ).fetchall()
        return Job.model_validate_json(rows[0]["data"]) if len(rows) == 1 else None

    def list_jobs(self, states: Iterable[JobState] | None = None) -> list[Job]:
        with self._lock:
            if states is None:
                rows = self._conn.execute("SELECT data FROM jobs ORDER BY submitted_at").fetchall()
            else:
                values = [s.value for s in states]
                marks = ",".join("?" * len(values))
                rows = self._conn.execute(
                    f"SELECT data FROM jobs WHERE state IN ({marks}) ORDER BY submitted_at",
                    values,
                ).fetchall()
        return [Job.model_validate_json(r["data"]) for r in rows]

    # ------------------------------------------------------------------ segments

    def put_segment(self, segment: Segment) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments (id, job_id, idx, backend, state, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    segment.id,
                    segment.job_id,
                    segment.index,
                    segment.backend,
                    segment.state.value,
                    segment.model_dump_json(),
                ),
            )

    def segments(self, job_id: str | None = None) -> list[Segment]:
        with self._lock:
            if job_id is None:
                rows = self._conn.execute("SELECT data FROM segments ORDER BY rowid").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data FROM segments WHERE job_id = ? ORDER BY idx", (job_id,)
                ).fetchall()
        return [Segment.model_validate_json(r["data"]) for r in rows]

    def active_segments(self) -> list[Segment]:
        values = [s.value for s in ACTIVE_SEGMENT_STATES]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT data FROM segments WHERE state IN ({','.join('?' * len(values))}) "
                "ORDER BY rowid",
                values,
            ).fetchall()
        return [Segment.model_validate_json(r["data"]) for r in rows]

    # ------------------------------------------------------------------ checkpoints

    def record_checkpoint(
        self, job_id: str, seq: int, segment_id: str | None, created_at: datetime | None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO checkpoints (job_id, seq, segment_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (job_id, seq, segment_id, created_at.isoformat() if created_at else None),
            )

    def checkpoint_count(self, job_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM checkpoints WHERE job_id = ?", (job_id,)
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ quota ledger

    def add_usage(
        self, backend: str, start_ts: float, end_ts: float, segment_id: str | None, now: float
    ) -> None:
        hours = max(0.0, end_ts - start_ts) / 3600
        with self._lock:
            self._conn.execute(
                "INSERT INTO quota_ledger (backend, kind, start_ts, end_ts, hours, segment_id, "
                "created_ts) VALUES (?, 'usage', ?, ?, ?, ?, ?)",
                (backend, start_ts, end_ts, hours, segment_id, now),
            )

    def add_correction(
        self,
        backend: str,
        *,
        remaining_h: float | None,
        used_h: float | None,
        now: float,
        note: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO quota_ledger (backend, kind, remaining_h, used_h, note, created_ts) "
                "VALUES (?, 'correction', ?, ?, ?, ?)",
                (backend, remaining_h, used_h, note, now),
            )

    def ledger(self, backend: str | None = None) -> list[LedgerEntry]:
        with self._lock:
            if backend is None:
                rows = self._conn.execute("SELECT * FROM quota_ledger ORDER BY id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM quota_ledger WHERE backend = ? ORDER BY id", (backend,)
                ).fetchall()
        return [LedgerEntry(**dict(r)) for r in rows]

    # ------------------------------------------------------------------ events

    def add_event(
        self,
        kind: str,
        message: str,
        *,
        now: float,
        job_id: str | None = None,
        segment_id: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (ts, kind, job_id, segment_id, message) VALUES (?, ?, ?, ?, ?)",
                (now, kind, job_id, segment_id, message),
            )

    def events(self, job_id: str | None = None, limit: int = 100) -> list[Event]:
        with self._lock:
            if job_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
        return [Event(**dict(r)) for r in reversed(rows)]
