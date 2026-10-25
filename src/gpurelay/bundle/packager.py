"""Job bundles (spec §5, §11.3): upload a job's code once; materialize it in each session.

Storage layout::

    jobs/<job>/bundle/manifest.json      BundleManifest
    jobs/<job>/bundle/bundle.tar.zst     the job's files (entry, extra files, requirements)
    jobs/<job>/bundle/<runtime wheel>    pinned gpurelay runtime (installed by cloud sessions)
    data-cache/<sha256>/...              ``local:`` data, uploaded once, content-addressed
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import zstandard

from gpurelay.bundle.manifest import BundleManifest, StagedData
from gpurelay.bundle.wheel import build_runtime_wheel
from gpurelay.errors import ChecksumMismatchError, GpuRelayError, StorageNotFoundError
from gpurelay.models import DataSourceKind, DataSpec, JobKind
from gpurelay.pathutil import safe_join
from gpurelay.storage.base import Storage, put_all, sha256_file
from gpurelay.storage.layout import bundle_prefix

ARCHIVE_NAME = "bundle.tar.zst"
DATA_CACHE = "data-cache"


class BundleError(GpuRelayError):
    """The job cannot be bundled (missing file, unsupported data source, ...)."""


def _manifest_key(job_id: str) -> str:
    return f"{bundle_prefix(job_id)}/manifest.json"


def _archive_key(job_id: str) -> str:
    return f"{bundle_prefix(job_id)}/{ARCHIVE_NAME}"


# ====================================================================== archive


def _pack(files: Mapping[str, Path]) -> bytes:
    """Deterministic tar (sorted, fixed metadata) compressed with zstd."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, path in sorted(files.items()):
            data = Path(path).read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return zstandard.ZstdCompressor(level=10).compress(raw.getvalue())


def _unpack(blob: bytes) -> dict[str, bytes]:
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 34)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise BundleError(f"unsafe path in bundle: {name}")
            extracted = tar.extractfile(member)
            assert extracted is not None
            out[name] = extracted.read()
    return out


# ====================================================================== data


def _dir_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        rel = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(rel.encode() + b"\0" + sha256_file(file).encode() + b"\n")
    return digest.hexdigest()


def _stage_spec(storage: Storage, spec: DataSpec, base_dir: Path) -> StagedData:
    if spec.kind is DataSourceKind.LOCAL:
        source = Path(spec.ref)
        if not source.is_absolute():
            source = (base_dir / source).resolve()
        if not source.exists():
            raise BundleError(f"data {spec}: {source} does not exist")
        key = _dir_hash(source)
        prefix = f"{DATA_CACHE}/{key}"
        if not storage.list(prefix + "/"):
            files = (
                [source]
                if source.is_file()
                else sorted(p for p in source.rglob("*") if p.is_file())
            )

            def key_for(file: Path) -> str:
                rel = file.name if source.is_file() else file.relative_to(source).as_posix()
                return f"{prefix}/{rel}"

            items: list[tuple[str, Path | bytes]] = [(key_for(f), f) for f in files]
            put_all(storage, items, message=f"gpurelay data {key[:12]}")
        return StagedData(spec=spec, target=source.name, cache_key=key)
    if spec.kind is DataSourceKind.URL:
        name = spec.ref.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0] or "download"
        return StagedData(spec=spec, target=name)
    # kaggle:<owner>/<slug> and hf:<owner>/<name>: staged by the session under <slug>/<name>.
    return StagedData(spec=spec, target=spec.ref.split("/", 1)[1])


# ====================================================================== upload / download


def upload_bundle(
    storage: Storage,
    job_id: str,
    entry: Path,
    extra_files: Mapping[str, Path] | None = None,
    *,
    requirements: Path | None = None,
    data: list[DataSpec] | None = None,
    include_wheel: bool = True,
) -> BundleManifest:
    """Package ``entry`` (+ extra files, requirements, data) and upload it for ``job_id``."""
    entry = Path(entry)
    if not entry.is_file():
        raise BundleError(f"{entry} does not exist")
    kind = JobKind.NOTEBOOK if entry.suffix == ".ipynb" else JobKind.SCRIPT
    files: dict[str, Path] = {entry.name: entry, **dict(extra_files or {})}
    req_name = None
    if requirements is not None:
        req_name = "requirements.txt" if requirements.name != entry.name else requirements.name
        files[req_name] = requirements
    for rel, path in files.items():
        if not Path(path).is_file():
            raise BundleError(f"{rel}: {path} does not exist")
    staged = [_stage_spec(storage, spec, entry.parent) for spec in (data or [])]

    import gpurelay

    manifest = BundleManifest(
        job_id=job_id,
        kind=kind,
        entry=entry.name,
        files={rel: sha256_file(Path(p)) for rel, p in sorted(files.items())},
        requirements=req_name,
        data=staged,
        runtime_version=gpurelay.__version__,
    )
    with tempfile.TemporaryDirectory(prefix="gpurelay-bundle-") as tmp:
        archive = Path(tmp) / ARCHIVE_NAME
        archive.write_bytes(_pack(files))
        items: list[tuple[str, Path | bytes]] = [(_archive_key(job_id), archive)]
        if include_wheel:
            wheel = build_runtime_wheel(Path(tmp))
            items.append((f"{bundle_prefix(job_id)}/{wheel.name}", wheel))
            manifest.runtime_wheel = wheel.name
        manifest.content_hash = manifest.compute_hash()
        # The manifest goes last: a session treats the bundle as present once it exists.
        items.append((_manifest_key(job_id), manifest.model_dump_json(indent=2).encode()))
        put_all(storage, items, message=f"gpurelay {job_id} bundle")
    return manifest


def read_manifest(storage: Storage, job_id: str) -> BundleManifest:
    raw = storage.get_bytes(_manifest_key(job_id))
    if raw is None:
        raise StorageNotFoundError(f"bundle manifest for {job_id}")
    return BundleManifest.model_validate_json(raw)


def _files(storage: Storage, job_id: str, manifest: BundleManifest) -> dict[str, bytes]:
    blob = storage.get_bytes(_archive_key(job_id))
    if blob is None:
        raise StorageNotFoundError(f"bundle archive for {job_id}")
    files = _unpack(blob)
    for rel, sha in manifest.files.items():
        if rel not in files or hashlib.sha256(files[rel]).hexdigest() != sha:
            raise ChecksumMismatchError(f"bundle file {rel} does not match its manifest")
    return files


def read_bundle_file(storage: Storage, job_id: str, rel: str) -> bytes:
    """Content of one bundled file (e.g. the source notebook for output assembly)."""
    manifest = read_manifest(storage, job_id)
    files = _files(storage, job_id, manifest)
    if rel not in files:
        raise StorageNotFoundError(f"bundle file {rel} of {job_id}")
    return files[rel]


def download_bundle(storage: Storage, job_id: str, dest: Path) -> BundleManifest:
    """Write the bundle's files into ``dest`` (verified) and return the manifest."""
    manifest = read_manifest(storage, job_id)
    for rel, data in _files(storage, job_id, manifest).items():
        target = safe_join(Path(dest), rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def _stage_kaggle(ref: str, target: Path) -> None:
    """Kaggle datasets: the attached mount on Kaggle (VERIFY: mount layout), else a download."""
    slug = ref.split("/", 1)[1]
    for mount in (
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets") / ref,
    ):
        if mount.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(mount, target_is_directory=True)
            except OSError:
                shutil.copytree(mount, target)
            return
    from gpurelay.backends.kaggle import OfficialKaggleClient

    client = OfficialKaggleClient()
    client._api.dataset_download_files(ref, path=str(target), unzip=True, quiet=True)


def stage_data(storage: Storage, manifest: BundleManifest, work_dir: Path) -> list[str]:
    """Materialize the job's data in the working directory; return the staged paths."""
    staged: list[str] = []
    for item in manifest.data:
        target = Path(work_dir) / item.target
        if item.cache_key is not None:
            prefix = f"{DATA_CACHE}/{item.cache_key}/"
            keys = storage.list(prefix)
            single = len(keys) == 1 and keys[0][len(prefix) :] == item.target
            for key in keys:
                rel = key[len(prefix) :]
                storage.get_file(key, target if single else target / rel)
        elif item.spec.kind is DataSourceKind.KAGGLE:
            _stage_kaggle(item.spec.ref, target)
        elif item.spec.kind is DataSourceKind.HF:
            from huggingface_hub import snapshot_download

            from gpurelay.credentials import hf_token

            snapshot_download(
                item.spec.ref, repo_type="dataset", local_dir=str(target), token=hf_token()
            )
        elif item.spec.kind is DataSourceKind.URL:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(item.spec.ref) as response, open(target, "wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
        staged.append(item.target)
    return staged


__all__ = [
    "BundleError",
    "download_bundle",
    "read_bundle_file",
    "read_manifest",
    "stage_data",
    "upload_bundle",
]
