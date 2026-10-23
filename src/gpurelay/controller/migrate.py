"""Move the controller to another machine (Phase 9 item 6): ``export-state`` / ``import-state``.

The archive holds ``config.toml`` and a consistent copy of ``state.db`` (SQLite backup API,
safe while a controller runs). It never holds secrets: tokens stay in the OS keyring or
environment of each machine. Jobs, checkpoints, and outputs are in the shared storage
already, so nothing else needs to move.
"""

from __future__ import annotations

import sqlite3
import tarfile
import tempfile
from pathlib import Path

from gpurelay.constants import CONFIG_FILENAME, STATE_DB_FILENAME
from gpurelay.errors import GpuRelayError

MEMBERS = (CONFIG_FILENAME, STATE_DB_FILENAME)


class MigrationError(GpuRelayError):
    """The state archive cannot be written or imported."""


def export_state(home: Path, dest: Path) -> list[str]:
    """Write ``dest`` (.tar.gz) with the config and job database of ``home``."""
    written = []
    with tempfile.TemporaryDirectory(prefix="gpurelay-export-") as tmp:
        staged = Path(tmp)
        config = home / CONFIG_FILENAME
        if config.exists():
            (staged / CONFIG_FILENAME).write_bytes(config.read_bytes())
            written.append(CONFIG_FILENAME)
        db = home / STATE_DB_FILENAME
        if db.exists():
            src = sqlite3.connect(db)
            out = sqlite3.connect(staged / STATE_DB_FILENAME)
            try:
                src.backup(out)
            finally:
                out.close()
                src.close()
            written.append(STATE_DB_FILENAME)
        if not written:
            raise MigrationError(f"nothing to export in {home}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:gz") as tar:
            for name in written:
                tar.add(staged / name, arcname=name)
    return written


def _has_jobs(db: Path) -> bool:
    conn = sqlite3.connect(db)
    try:
        return bool(conn.execute("select count(*) from jobs").fetchone()[0])
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def import_state(home: Path, archive: Path, *, force: bool = False) -> list[str]:
    """Install an exported config and job database into ``home``."""
    db = home / STATE_DB_FILENAME
    if db.exists() and _has_jobs(db) and not force:
        raise MigrationError(
            f"{db} already has jobs; import with --force to replace them (export them first)"
        )
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if not members or any(m.name not in MEMBERS or not m.isfile() for m in members):
            raise MigrationError(f"{archive} is not a GPU Relay state archive ({names})")
        home.mkdir(parents=True, exist_ok=True)
        for member in members:
            source = tar.extractfile(member)
            assert source is not None
            target = home / member.name
            tmp = target.with_name(target.name + ".importing")
            tmp.write_bytes(source.read())
            try:
                tmp.replace(target)
            except PermissionError as exc:  # Windows: the database is open elsewhere
                tmp.unlink(missing_ok=True)
                raise MigrationError(
                    f"{target} is in use: stop `gpurelay up` and the dashboard first"
                ) from exc
    for suffix in ("-wal", "-shm"):  # stale SQLite side files of the replaced database
        (home / (STATE_DB_FILENAME + suffix)).unlink(missing_ok=True)
    return names
