"""Build a pure-Python wheel of the *installed* ``gpurelay`` package (the pinned runtime).

Backend sessions (Kaggle, Colab) install this wheel so they run exactly the runtime the
controller submitted with. The wheel is assembled directly (it is pure Python), with fixed
timestamps so identical sources give identical bytes.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import zipfile
from pathlib import Path

import gpurelay

_FIXED_DATE = (2020, 1, 1, 0, 0, 0)
_EXCLUDED_PARTS = {"__pycache__"}


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def wheel_name(version: str | None = None) -> str:
    return f"gpurelay-{version or gpurelay.__version__}-py3-none-any.whl"


def build_runtime_wheel(dest_dir: Path) -> Path:
    """Write the runtime wheel into ``dest_dir`` and return its path."""
    version = gpurelay.__version__
    package_dir = Path(gpurelay.__file__).resolve().parent
    dist_info = f"gpurelay-{version}.dist-info"
    try:
        requires = importlib.metadata.requires("gpurelay") or []
    except importlib.metadata.PackageNotFoundError:
        requires = []
    metadata = "\n".join(
        [
            "Metadata-Version: 2.1",
            "Name: gpurelay",
            f"Version: {version}",
            "Requires-Python: >=3.10",
            *[f"Requires-Dist: {req}" for req in requires],
            "",
        ]
    )
    wheel_meta = (
        "Wheel-Version: 1.0\nGenerator: gpurelay\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    entry_points = "[console_scripts]\ngpurelay = gpurelay.cli:app\n"

    files: list[tuple[str, bytes]] = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_dir() or _EXCLUDED_PARTS & set(path.parts) or path.suffix == ".pyc":
            continue
        rel = path.relative_to(package_dir.parent).as_posix()
        files.append((rel, path.read_bytes()))
    files += [
        (f"{dist_info}/METADATA", metadata.encode()),
        (f"{dist_info}/WHEEL", wheel_meta.encode()),
        (f"{dist_info}/entry_points.txt", entry_points.encode()),
    ]
    record = "".join(f"{name},{_record_hash(data)},{len(data)}\n" for name, data in files)
    record += f"{dist_info}/RECORD,,\n"
    files.append((f"{dist_info}/RECORD", record.encode()))

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / wheel_name(version)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return target
