"""Build the Windows desktop app: ``dist/GPU Relay/GPU Relay.exe`` and a zip to share.

    python packaging/build_windows.py

Needs the desktop extra and PyInstaller in the environment (pip install -e ".[desktop]"
pyinstaller). One folder, not one file: it starts faster and antivirus tools flag it less.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "GPU Relay"


def main() -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    import gpurelay
    from gpurelay.desktop.icon import save_ico

    build = ROOT / "build" / "desktop"
    icon = save_ico(build / "gpurelay.ico")
    sep = ";" if sys.platform == "win32" else ":"
    args = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--windowed",
        "--name", NAME, "--icon", str(icon),
        "--distpath", str(ROOT / "dist"), "--workpath", str(build / "work"),
        "--specpath", str(build),
        # The package and its data (web app pages, session templates).
        "--collect-submodules", "gpurelay", "--collect-data", "gpurelay",
        "--copy-metadata", "gpurelay",
        # Sources for the runtime wheel that GPU sessions install (bundle/wheel.py).
        "--add-data", f"{ROOT / 'src' / 'gpurelay'}{sep}gpurelay_src/gpurelay",
        # Imported by name at runtime, so PyInstaller cannot see them.
        "--collect-submodules", "uvicorn", "--collect-submodules", "keyring",
        "--collect-submodules", "huggingface_hub", "--collect-data", "kaggle",
        "--collect-submodules", "kaggle", "--hidden-import", "pystray._win32",
        "--copy-metadata", "huggingface_hub", "--copy-metadata", "kaggle",
        "--copy-metadata", "keyring", "--copy-metadata", "pydantic",
        "--collect-submodules", "modal", "--collect-data", "modal", "--copy-metadata", "modal",
        "--exclude-module", "torch", "--exclude-module", "tensorflow",
        "--exclude-module", "matplotlib", "--exclude-module", "tkinter",
        # ML libraries run inside GPU sessions, never in the desktop app.
        "--exclude-module", "transformers", "--exclude-module", "accelerate",
        "--exclude-module", "lightning", "--exclude-module", "pytorch_lightning",
        "--exclude-module", "lightning_sdk", "--exclude-module", "keras",
        "--exclude-module", "jax", "--exclude-module", "scipy", "--exclude-module", "sklearn",
        str(ROOT / "packaging" / "gpurelay_desktop.py"),
    ]  # fmt: skip
    subprocess.run(args, check=True, cwd=ROOT)
    app_dir = ROOT / "dist" / NAME
    shutil.copy2(ROOT / "README.md", app_dir / "README.md")
    archive = ROOT / "dist" / f"GPU-Relay-{gpurelay.__version__}-windows-x64.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                zf.write(path, Path(NAME) / path.relative_to(app_dir))
    print(f"built {app_dir / (NAME + '.exe')}")
    print(f"zip   {archive} ({archive.stat().st_size / 2**20:.1f} MB)")
    return archive


if __name__ == "__main__":
    main()
