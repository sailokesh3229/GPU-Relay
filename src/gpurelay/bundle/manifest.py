"""Bundle manifest: what a job consists of (spec §5 "Bundle")."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gpurelay.models import DataSpec, JobKind, utcnow

BUNDLE_FORMAT_VERSION = 2


class StagedData(BaseModel):
    """A data source and where the runtime puts it in the working directory."""

    model_config = ConfigDict(extra="forbid")

    spec: DataSpec
    target: str  # path relative to the working directory
    cache_key: str | None = None  # data-cache/<cache_key>/ for uploaded local data


class BundleManifest(BaseModel):
    """Stored at ``jobs/<job>/bundle/manifest.json`` next to ``bundle.tar.zst``."""

    model_config = ConfigDict(extra="forbid")

    format: int = BUNDLE_FORMAT_VERSION
    job_id: str
    kind: JobKind
    entry: str  # script/notebook path relative to the working directory
    files: dict[str, str] = Field(default_factory=dict)  # relative path -> sha256
    requirements: str | None = None  # relative path of the requirements file, if any
    data: list[StagedData] = Field(default_factory=list)
    runtime_wheel: str | None = None  # file name under bundle/
    runtime_version: str | None = None
    content_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    def compute_hash(self) -> str:
        """Hash of everything that defines the job's behaviour (not ids or timestamps)."""
        payload = {
            "kind": self.kind.value,
            "entry": self.entry,
            "files": dict(sorted(self.files.items())),
            "requirements": self.requirements,
            "data": [d.model_dump(mode="json") for d in self.data],
            "runtime_version": self.runtime_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
