# Changelog

## 1.0.0 — 2026-10-25

The first complete release: relay ML jobs across free GPU tiers from a desktop app, an
always-on website or the command line.

### Apps
- **Windows desktop app** (`GPU Relay.exe`, no Python needed for cloud GPUs):
  - native window, tray icon with notifications;
  - setup wizard for Hugging Face, Kaggle, Modal and Colab;
  - two modes: run the controller on this computer, or show your always-on server;
  - single instance, optional start with Windows and Start menu entry, update check, clean
    removal.
- **Always-on website** (`gpurelay serve`, one-click Render blueprint):
  - owner login, public read-only view or a viewer password;
  - the job database is backed up to your private storage and restored after restarts;
  - live at https://gpu-relay-9uuh.onrender.com.
- **Web app screens** (approved designs): Jobs, Job detail with relay timeline and live log,
  Submit with a resumability check, GPU budget, Settings.

### Platforms
- **Modal** (new): GPU sandboxes through the official SDK, with a dollar budget ($30 a month,
  about 50 T4-hours). Switched on only after you confirm its terms.
- **Your own NVIDIA GPU** (new): detected with `nvidia-smi` and counted at its real speed.
- Kaggle (automatic), Colab (manual, never automated), Lightning AI (built in, off until
  its terms are reviewed). Paperspace was checked and is not used, because its free notebooks
  are public.

### Scheduling
- **Budgets in any unit:** hours a week, dollars or credits a month, hours a day.
- **GPU speed counts** (T4 = 1.0 … H100 = 9).
- **Order of use:** credits that expire first unless much slower, then the fastest GPU, then
  the most hours left.

### Reliability
- **Hugging Face commit budget:** a checkpoint is 2 commits (was 4); bundles and outputs are
  1 commit each; heartbeats every 5 minutes. About 48 commits an hour with a job running,
  against the limit of 128.
- **Graceful controller stop on every platform** (a stop file): the lease is released and
  the next controller takes over at once.
- **Controller downtime** is no longer charged as GPU time.
- **Hardening campaign:** adversarial, property-based, fault-injection, path-safety and stress
  tests; 14 bugs fixed.

### Tests
- About 500 unit tests and 13 integration test files (kill-and-resume equivalence for
  scripts, notebooks, PyTorch, Keras, Hugging Face and Lightning jobs).
- Live checks on Kaggle, the Render server and the packaged desktop app.
