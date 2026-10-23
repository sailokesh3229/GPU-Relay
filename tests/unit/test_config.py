from __future__ import annotations

from pathlib import Path

import pytest

from gpurelay.config import (
    Config,
    default_config_path,
    gpurelay_home,
    load_config,
    save_config,
)
from gpurelay.errors import ConfigError
from gpurelay.models import BudgetModel


def test_home_respects_env_override(relay_home: Path) -> None:
    assert gpurelay_home() == relay_home
    assert default_config_path() == relay_home / "config.toml"


def test_missing_file_gives_defaults() -> None:
    assert not default_config_path().exists()
    assert load_config() == Config()


def test_defaults_round_trip() -> None:
    path = save_config(Config())
    assert path == default_config_path()
    assert load_config() == Config()


def test_modified_values_round_trip(tmp_path: Path) -> None:
    cfg = Config()
    cfg.backends.kaggle.enabled = True
    cfg.backends.kaggle.username = "someone"
    cfg.backends.kaggle.weekly_gpu_hours = 22.5
    cfg.backends.kaggle.secrets_dataset = "someone/gpurelay-secrets"
    cfg.backends.colab_manual.enabled = True
    cfg.backends.local.gpu_label = "RTX 3060"
    cfg.storage.kind = "localfs"
    cfg.storage.localfs_root = tmp_path / "store"
    cfg.notify.ntfy_topic = "my-topic"
    cfg.runtime.max_lost_work_s = 600

    path = tmp_path / "custom.toml"
    save_config(cfg, path)
    loaded = load_config(path)

    assert loaded == cfg
    assert loaded.backends.kaggle.budget_model is BudgetModel.WEEKLY_ROLLING
    assert loaded.backends.colab_manual.budget_model is BudgetModel.UNKNOWN_OPPORTUNISTIC


def test_save_leaves_no_temp_files() -> None:
    path = save_config(Config())
    assert sorted(p.name for p in path.parent.iterdir()) == ["config.toml"]


def test_invalid_toml_raises_config_error() -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config()


def test_invalid_value_raises_config_error() -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("[runtime]\nmax_lost_work_s = -5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_lost_work_s"):
        load_config()


# ---------------------------------------------------------------- hard rules (spec §3)


def _write(text: str) -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_second_account_section_rejected() -> None:
    """Hard rule 1: exactly one credential set per backend type."""
    _write('[backends.kaggle]\nusername = "a"\n\n[backends.kaggle2]\nusername = "b"\n')
    with pytest.raises(ConfigError, match="kaggle2"):
        load_config()


def test_accounts_list_rejected() -> None:
    _write('[backends.kaggle]\naccounts = ["a", "b"]\n')
    with pytest.raises(ConfigError, match="accounts"):
        load_config()


@pytest.mark.parametrize("field", ["key", "api_key", "token"])
def test_credentials_not_accepted_in_config(field: str) -> None:
    """Secrets live in the keyring / env vars, never in config.toml."""
    _write(f'[backends.kaggle]\n{field} = "s3cret"\n')
    with pytest.raises(ConfigError, match=field):
        load_config()


def test_storage_token_not_accepted_in_config() -> None:
    _write('[storage]\nhf_token = "hf_xxx"\n')
    with pytest.raises(ConfigError, match="hf_token"):
        load_config()


def test_lightning_needs_the_terms_review() -> None:
    """Hard rule 4: enabling Lightning also requires terms_reviewed (docs/verify.md #10)."""
    _write("[backends.lightning]\nenabled = true\n")
    with pytest.raises(ConfigError, match="terms_reviewed"):
        load_config()
    cfg = Config()
    with pytest.raises(ValueError, match="terms_reviewed"):
        cfg.backends.lightning.enabled = True
    _write("[backends.lightning]\nenabled = true\nterms_reviewed = true\n")
    assert load_config().backends.lightning.enabled
