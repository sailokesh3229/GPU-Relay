"""Project-wide constants and default values.

Platform-specific defaults (Kaggle/Colab limits) live in :mod:`gpurelay.config`
next to their ``# VERIFY:`` notes, because they are user-editable guesses.
"""

from __future__ import annotations

from typing import Final

APP_NAME: Final = "gpurelay"

# ---------------------------------------------------------------- local home / config
ENV_HOME: Final = "GPURELAY_HOME"
# Opens the launcher notebook in Colab (it is in this project's public repository): it reads
# the user's Hugging Face token from their Colab secret, finds the job waiting for Colab in
# their storage repo, and runs its resume notebook when they press Run all.
COLAB_LAUNCHER_URL: Final = "https://colab.research.google.com/github/sailokesh3229/GPU-Relay/blob/main/colab/open_job.ipynb"
DEFAULT_HOME_DIRNAME: Final = ".gpurelay"
CONFIG_FILENAME: Final = "config.toml"
STATE_DB_FILENAME: Final = "state.db"
CONFIG_SCHEMA_VERSION: Final = 1

# ---------------------------------------------------------------- runtime defaults
DEFAULT_REPLAY_THRESHOLD_S: Final = 30.0  # spec §7.1
DEFAULT_OVERHEAD_BUDGET: Final = 0.03  # spec §8.2
DEFAULT_MIN_CHECKPOINT_INTERVAL_S: Final = 120  # spec §8.2
DEFAULT_MAX_LOST_WORK_S: Final = 900  # spec §2.1 / §8.2
DEFAULT_SAFETY_MARGIN_S: Final = 300  # spec §8.2
DEFAULT_ZSTD_LEVEL: Final = 3  # spec §8.1
DEFAULT_KEEP_CHECKPOINTS: Final = 2  # spec §8.1
HEARTBEAT_INTERVAL_S: Final = 60  # spec §12.2
STALE_HEARTBEAT_S: Final = 300  # spec §12.2

# ---------------------------------------------------------------- scheduler defaults
DEFAULT_MIN_SEGMENT_H: Final = 0.5  # spec §13.4
DEFAULT_MAX_INFRA_RETRIES: Final = 3  # spec §13.5
DEFAULT_POLL_INTERVAL_S: Final = 30
BUDGET_LOW_FRACTION: Final = 0.10  # spec §13.6

# ---------------------------------------------------------------- storage layout (spec §11.3)
JOBS_PREFIX: Final = "jobs"
DATA_CACHE_PREFIX: Final = "data-cache"
LATEST_FILENAME: Final = "LATEST.json"
STATUS_FILENAME: Final = "status.json"

# Marker embedded in temporary file names during atomic writes. Remote keys may not
# contain it, and listings never return keys that do.
TMP_MARKER: Final = ".gpurelay-tmp-"
