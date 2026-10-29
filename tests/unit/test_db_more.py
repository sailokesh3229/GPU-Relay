"""The controller's job database under real use: two processes writing at once (the web app
and the controller), a migration that fails, and every listing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gpurelay.controller import db as db_mod
from gpurelay.controller.db import StateDB
from gpurelay.models import Job, JobKind

WRITER = """
import sys
from pathlib import Path
from gpurelay.controller.db import StateDB
from gpurelay.models import Job, JobKind
db = StateDB(Path(sys.argv[1]))
for i in range(150):
    job = db.put_job(Job(name=f"{sys.argv[2]}-{i}", kind=JobKind.SCRIPT, entry="x.py"))
    db.add_event("submitted", job.name, now=float(i), job_id=job.id)
    db.add_usage("kaggle", float(i), float(i) + 1, None, float(i))
print("done")
"""


def test_two_processes_can_write_at_the_same_time(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    StateDB(path).close()  # created and migrated once
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(path), name],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for name in ("web", "controller")]  # fmt: skip
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0 and "done" in out, err  # no "database is locked"
    db = StateDB(path)
    assert len(db.list_jobs()) == 300
    assert len(db.events(limit=1000)) == 300 and len(db.ledger()) == 300
    assert len(db.ledger("kaggle")) == 300 and db.ledger("modal") == []


def test_a_failed_migration_leaves_the_database_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.db"
    db = StateDB(path)
    version = db.schema_version
    db.put_job(Job(name="kept", kind=JobKind.SCRIPT, entry="x.py"))
    db.close()
    broken = {**db_mod._MIGRATIONS,
              version + 1: "ALTER TABLE jobs ADD COLUMN extra TEXT; THIS IS NOT SQL"}  # fmt: skip
    monkeypatch.setattr(db_mod, "_MIGRATIONS", broken)
    with pytest.raises(Exception, match="syntax error"):
        StateDB(path)
    monkeypatch.undo()
    again = StateDB(path)
    assert again.schema_version == version  # rolled back, not half applied
    columns = {row[1] for row in again._conn.execute("PRAGMA table_info(jobs)")}
    assert "extra" not in columns and [j.name for j in again.list_jobs()] == ["kept"]
