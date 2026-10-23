from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpurelay.models import (
    CheckpointPointer,
    DataSourceKind,
    DataSpec,
    ExitCode,
    Job,
    JobKind,
    JobState,
    RuntimeState,
    StatusFile,
)


def test_exit_codes_match_spec() -> None:
    assert [int(c) for c in ExitCode] == [0, 85, 86, 87, 88]


def test_terminal_job_states() -> None:
    terminal = {s for s in JobState if s.is_terminal}
    assert terminal == {
        JobState.COMPLETED,
        JobState.FAILED_USER,
        JobState.FAILED_INFRA,
        JobState.CANCELLED,
    }


@pytest.mark.parametrize(
    ("spec", "kind", "ref"),
    [
        ("kaggle:owner/dataset", DataSourceKind.KAGGLE, "owner/dataset"),
        ("hf:org/repo", DataSourceKind.HF, "org/repo"),
        ("url:https://example.com/a.zip", DataSourceKind.URL, "https://example.com/a.zip"),
        ("local:C:/data/train", DataSourceKind.LOCAL, "C:/data/train"),
    ],
)
def test_data_spec_parse(spec: str, kind: DataSourceKind, ref: str) -> None:
    parsed = DataSpec.parse(spec)
    assert (parsed.kind, parsed.ref) == (kind, ref)
    assert str(parsed) == spec


@pytest.mark.parametrize(
    "spec", ["nokind", "s3:bucket/x", "kaggle:", "kaggle:justname", "url:ftp://x", "hf:a/b/c"]
)
def test_data_spec_parse_rejects(spec: str) -> None:
    with pytest.raises(ValueError, match="invalid data spec"):
        DataSpec.parse(spec)


def test_job_defaults() -> None:
    job = Job(name="vae", kind=JobKind.SCRIPT, entry="train.py")
    assert job.id.startswith("job-")
    assert job.state is JobState.QUEUED
    assert job.submitted_at.tzinfo is not None


def test_status_file_matches_spec_example() -> None:
    raw = (
        '{"job_id": "j", "segment_id": "s", "state": "running",'
        ' "heartbeat_at": "2026-10-21T12:00:00Z", "last_checkpoint_seq": 7,'
        ' "progress": {"cell": 5, "loop": {"L3": [412, 1000]}},'
        ' "rate": {"unit": "iter", "per_hour": 1450.0}, "message": "..."}'
    )
    status = StatusFile.model_validate_json(raw)
    assert status.state is RuntimeState.RUNNING
    assert status.progress.loop["L3"] == (412, 1000)
    assert StatusFile.model_validate_json(status.model_dump_json()) == status


def test_checkpoint_pointer_requires_checksum_per_file() -> None:
    common = {
        "seq": 3,
        "safe_point_id": "L3:412",
        "created_at": datetime.now(timezone.utc),
        "segment_id": "seg-1",
    }
    ok = CheckpointPointer(files=["meta.json"], sha256s={"meta.json": "ab"}, **common)
    assert ok.seq == 3
    with pytest.raises(ValueError, match="sha256s"):
        CheckpointPointer(files=["meta.json", "rng.pt"], sha256s={"meta.json": "ab"}, **common)
