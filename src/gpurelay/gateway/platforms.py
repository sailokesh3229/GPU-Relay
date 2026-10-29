"""A website user's answers and platform credentials (plan G1): answers in their settings
(``Config.platforms``), credentials sealed in the vault under ``platform.<id>.<field>``.
Those names are not job tokens (:data:`gpurelay.gateway.userenv.SECRET_NAMES`), so no
process acting for the user receives them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from gpurelay.config import CLOUD_FIELDS, CloudSettings, PlatformsSettings
from gpurelay.gateway.space import UserSpaces
from gpurelay.gateway.userenv import KAGGLE, MODAL_SECRET
from gpurelay.platforms.catalog import SECRET_PREFIX

SETUP_STEPS = {"kaggle": "/setup#kaggle", "modal": "/setup#modal", "colab": "/setup#colab"}


@dataclass
class WebsitePlatformStore:
    spaces: UserSpaces
    user_id: str
    start_path: Callable[[], str]  # where a signed-in user goes next
    after_change: Callable[[str], None] = lambda user_id: None  # restarts their GPU Relay

    def load(self) -> PlatformsSettings:
        return self.spaces.config(self.user_id).platforms

    def save(self, settings: PlatformsSettings) -> None:
        config = self.spaces.config(self.user_id)  # read fresh: other settings stay as they are
        config.platforms = settings
        self.spaces.save_config(self.user_id, config)

    def saved(self) -> dict[str, float]:
        return {n: t for n, t in self.spaces.saved(self.user_id).items()
                if n.startswith(SECRET_PREFIX)}  # fmt: skip

    def put(self, values: Mapping[str, str]) -> None:
        _platform_only(values)
        self.spaces.put_secrets(self.user_id, values)

    def remove(self, names: Iterable[str]) -> None:
        names = tuple(names)
        _platform_only(names)
        self.spaces.remove_secrets(self.user_id, names)

    def connected(self) -> dict[str, bool]:
        config, saved = self.spaces.config(self.user_id), self.spaces.saved(self.user_id)
        backends = config.backends
        return {
            "kaggle": KAGGLE in saved and backends.kaggle.enabled,
            "modal": MODAL_SECRET in saved and backends.modal.enabled,
            "colab": backends.colab_manual.enabled,
        }

    def clouds(self) -> dict[str, CloudSettings] | None:
        config = self.spaces.config(self.user_id)
        return {p: config.backends.cloud(p) for p in CLOUD_FIELDS}

    def save_clouds(self, clouds: dict[str, CloudSettings]) -> None:
        config = self.spaces.config(self.user_id)
        for platform_id, settings in clouds.items():
            setattr(config.backends, CLOUD_FIELDS[platform_id], settings)
        self.spaces.save_config(self.user_id, config)
        self.after_change(self.user_id)  # their GPU Relay restarts with the new platforms

    def setup_link(self, platform_id: str) -> str:
        return SETUP_STEPS.get(platform_id, "/setup")

    def next_path(self) -> str:
        return self.start_path()


def _platform_only(names: Iterable[str]) -> None:
    """Never let this page overwrite or remove a job token (Hugging Face, Kaggle, Modal)."""
    if any(not n.startswith(SECRET_PREFIX) for n in names):
        raise ValueError("only platform credentials are kept here")
