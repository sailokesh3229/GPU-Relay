"""Hub commit budget (plan W3b): the Hub allows 128 commits per hour per storage repo.

Every write to Hugging Face storage is a commit, so the runtime and controller must stay well
under that with a job running. Found live: a Kaggle stress run hit the limit (HTTP 429).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from gpurelay.backends.colab_manual import ColabManualBackend
from gpurelay.backends.common import (
    MIN_CHECKPOINT_INTERVAL_S,
    MIN_HEARTBEAT_S,
    session_settings,
)
from gpurelay.backends.kaggle import KaggleBackend
from gpurelay.backends.lightning import LightningBackend
from gpurelay.config import RuntimeSettings
from gpurelay.constants import STATE_DB_FILENAME
from gpurelay.controller.db import StateDB
from gpurelay.controller.lease import RENEW_S, STALE_S
from gpurelay.hosted import BACKUP_S, STATE_KEY, StateBackup
from gpurelay.models import Job, JobKind, utcnow
from gpurelay.retry import RetryPolicy
from gpurelay.runtime.persist import Persister
from gpurelay.storage.layout import checkpoints_prefix, parse_ckpt_seq
from gpurelay.storage.localfs import LocalFSStorage

HUB_COMMITS_PER_HOUR = 128


class CommitCounter(LocalFSStorage):
    """Local storage that records each write the way the Hub would: one commit per call."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.commits: list[str] = []
        self._batch = False

    def _commit(self, what: str) -> None:
        if not self._batch:
            self.commits.append(what)

    def put_bytes(self, data: bytes, remote: str) -> None:
        self._commit(f"put {remote}")
        super().put_bytes(data, remote)

    def put_file(self, local: Path, remote: str) -> None:
        self._commit(f"put {remote}")
        super().put_file(local, remote)

    def delete(self, remote: str) -> None:
        self._commit(f"delete {remote}")
        super().delete(remote)

    def put_many(
        self,
        items: Sequence[tuple[str, Path | bytes]],
        message: str = "gpurelay batch",
        deletes: Iterable[str] = (),
    ) -> None:
        self._commit(f"batch {message}")
        self._batch = True
        try:
            super().put_many(items, message, deletes)
        finally:
            self._batch = False

    def delete_many(self, remotes: Iterable[str], message: str = "gpurelay delete") -> None:
        self._commit(f"delete-many {message}")
        self._batch = True
        try:
            super().delete_many(remotes, message)
        finally:
            self._batch = False


def _stored_seqs(store: LocalFSStorage) -> set[int]:
    keys = store.list(checkpoints_prefix("job-1") + "/")
    return {n for k in keys if (n := parse_ckpt_seq(k, "job-1")) is not None}


def _persist(persister: Persister, tmp_path: Path, seq: int) -> None:
    local = tmp_path / f"snap-{seq}"
    local.mkdir()
    (local / "state.bin").write_bytes(b"x" * 1000 + bytes([seq]))
    result = persister.persist(seq, local, segment_id="seg", safe_point_id="p",
                               created_at=utcnow())  # fmt: skip
    assert result.ok, result.error


def test_each_checkpoint_is_two_commits_including_retention(tmp_path: Path) -> None:
    store = CommitCounter(tmp_path / "store")
    persister = Persister(store, "job-1", zstd_level=3, keep=2, retry=RetryPolicy(attempts=1))
    for seq in range(1, 6):
        before = len(store.commits)
        _persist(persister, tmp_path, seq)
        made = store.commits[before:]
        assert len(made) == 2, made  # data (with old checkpoints deleted), then LATEST
        assert made[0].startswith("batch") and made[1].endswith("LATEST.json")
    assert _stored_seqs(store) == {4, 5}  # retention still keeps exactly `keep` checkpoints


def test_retention_never_deletes_what_latest_still_points_to(tmp_path: Path) -> None:
    """Retention is in the data commit, before LATEST moves: the resumed checkpoint stays."""
    store = CommitCounter(tmp_path / "store")
    first = Persister(store, "job-1", zstd_level=3, keep=2, retry=RetryPolicy(attempts=1))
    for seq in (1, 2, 3):
        _persist(first, tmp_path, seq)
    # A new session resumes from 3 (it only knows LATEST) and writes 4.
    resumed = Persister(store, "job-1", zstd_level=3, keep=2, retry=RetryPolicy(attempts=1),
                        committed=[3])  # fmt: skip
    _persist(resumed, tmp_path, 4)
    assert _stored_seqs(store) == {3, 4}


def test_hub_sessions_get_commit_friendly_settings() -> None:
    hub = session_settings(RuntimeSettings(heartbeat_s=30, min_interval_s=60,
                                           stale_heartbeat_s=300, max_lost_work_s=900))  # fmt: skip
    assert hub.heartbeat_s == MIN_HEARTBEAT_S
    assert hub.stale_heartbeat_s >= 3 * hub.heartbeat_s  # never stale between two beats
    assert hub.min_interval_s == MIN_CHECKPOINT_INTERVAL_S
    tight = session_settings(RuntimeSettings(min_interval_s=60, max_lost_work_s=180))
    assert tight.min_interval_s == 180  # the lost-work limit still wins
    for backend in (KaggleBackend, ColabManualBackend, LightningBackend):
        assert backend.hub_sessions is True


def test_an_hour_with_one_job_running_stays_far_under_the_hub_limit() -> None:
    """Worst case per hour: heartbeats + 2 commits per checkpoint + lease renewals + backups."""
    s = session_settings(RuntimeSettings())
    per_hour = 3600 / s.heartbeat_s + 2 * 3600 / s.min_interval_s + 3600 / RENEW_S
    per_hour += 3600 / BACKUP_S
    assert per_hour <= HUB_COMMITS_PER_HOUR / 2, per_hour  # room for a second session
    assert STALE_S > 2 * RENEW_S


def test_state_backup_uploads_only_changes(tmp_path: Path) -> None:
    store = CommitCounter(tmp_path / "store")
    home = tmp_path / "home"
    home.mkdir()
    db = StateDB(home / STATE_DB_FILENAME)
    db.put_job(Job(name="a", kind=JobKind.SCRIPT, entry="a.py"))
    backup = StateBackup(store, home)
    assert backup() is True and len(store.commits) == 1
    assert backup() is False and backup() is False  # nothing changed: no commit
    db.put_job(Job(name="b", kind=JobKind.SCRIPT, entry="b.py"))
    assert backup() is True and len(store.commits) == 2
    # A restarted server compares with the stored copy before its first upload.
    assert StateBackup(store, home)() is False and len(store.commits) == 2
    assert store.exists(STATE_KEY)
