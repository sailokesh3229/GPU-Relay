"""The desktop app's answers and platform credentials (plan G1): answers in
``<home>/platforms.json`` (``config.toml`` would mark setup as done before it is), credentials
in the OS keyring under their own names (``platform.<id>.<field>``).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurelay.config import CLOUD_FIELDS, CloudSettings, Config, PlatformsSettings, load_config
from gpurelay.config import save_config as write_config
from gpurelay.constants import CONFIG_FILENAME
from gpurelay.errors import GpuRelayError

FILENAME = "platforms.json"


@dataclass
class LocalPlatformStore:
    home: Path
    get_secret: Callable[[str, str], str | None]
    set_secret: Callable[[str, str], bool]
    delete_secret: Callable[[str], None]
    kaggle_present: Callable[[], bool]
    modal_present: Callable[[], bool]
    clock: Callable[[], float] = time.time
    reconfigured: Callable[[], None] = lambda: None  # restart the controller with new settings

    @property
    def path(self) -> Path:
        return self.home / FILENAME

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except ValueError as exc:  # never overwrite answers we could not read
            raise GpuRelayError(f"{self.path} is damaged: {exc}") from None
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{FILENAME}.", dir=self.home)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, self.path)

    def load(self) -> PlatformsSettings:
        return PlatformsSettings.model_validate(self._read().get("settings") or {})

    def save(self, settings: PlatformsSettings) -> None:
        data = self._read()
        data["settings"] = settings.model_dump(mode="json", exclude_none=True)
        self._write(data)

    def saved(self) -> dict[str, float]:
        """Credentials recorded as saved that the keyring still holds."""
        recorded: dict[str, float] = self._read().get("saved") or {}
        return {n: t for n, t in recorded.items() if self.get_secret(n, "") is not None}

    def put(self, values: Mapping[str, str]) -> None:
        data = self._read()
        recorded = dict(data.get("saved") or {})
        for name, value in values.items():
            if not self.set_secret(name, value):
                raise GpuRelayError("no keyring on this computer: credentials cannot be kept "
                                    "safely, so none were saved")  # fmt: skip
            recorded[name] = self.clock()
        data["saved"] = recorded
        self._write(data)

    def remove(self, names: Iterable[str]) -> None:
        data = self._read()
        recorded = dict(data.get("saved") or {})
        for name in names:
            self.delete_secret(name)
            recorded.pop(name, None)
        data["saved"] = recorded
        self._write(data)

    def _config(self) -> Config | None:
        path = self.home / CONFIG_FILENAME
        return load_config(path) if path.exists() else None

    def connected(self) -> dict[str, bool]:
        config = self._config()
        if config is None:
            return {}
        backends = config.backends
        return {
            "kaggle": backends.kaggle.enabled and self.kaggle_present(),
            "modal": backends.modal.enabled and self.modal_present(),
            "colab": backends.colab_manual.enabled,
        }

    def clouds(self) -> dict[str, CloudSettings] | None:
        config = self._config()
        return None if config is None else {p: config.backends.cloud(p) for p in CLOUD_FIELDS}

    def save_clouds(self, clouds: dict[str, CloudSettings]) -> None:
        config = self._config()
        if config is None:
            raise GpuRelayError("finish setting up GPU Relay first")
        for platform_id, settings in clouds.items():
            setattr(config.backends, CLOUD_FIELDS[platform_id], settings)
        write_config(config, self.home / CONFIG_FILENAME)
        self.reconfigured()

    def setup_link(self, platform_id: str) -> str:
        return "./#/setup"

    def next_path(self) -> str:
        return "./#/settings" if (self.home / CONFIG_FILENAME).exists() else "./#/setup"
