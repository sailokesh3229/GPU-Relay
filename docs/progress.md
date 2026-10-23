# Progress log

## Phase 0 — Skeleton and foundations

**Built**

- `pyproject.toml`: src layout (hatchling), console script `gpurelay`, extras
  `torch`, `tf`, `hf`, `lightning`, `kaggle`, `dev`; ruff, mypy (strict, pydantic plugin), and
  pytest (+ `pytest-timeout`, ML markers) configuration. Version single-sourced from
  `gpurelay/__init__.py`.
- CI (`.github/workflows/ci.yml`): lint job (ruff check, ruff format check, mypy) and unit tests
  on Ubuntu + Windows × Python 3.10 / 3.12, core install only (no torch).
- `constants.py`: defaults from the spec (checkpoint interval, heartbeat, scheduler, storage layout
  names).
- `errors.py`: exception hierarchy (`ConfigError`, `HardRuleViolation`, `StorageError`,
  `StorageNotFoundError`, `InvalidRemotePathError`, `NotYetImplementedError`).
- `models.py`: enums (job/segment/runtime states, exit codes 0/85/86/87/88, backend run state,
  resumability level, budget model, data source kind) and pydantic records (`Job`, `Segment`,
  `DataSpec` with `--data` parsing, `CheckpointPointer` = `LATEST.json`, `BackendCaps`,
  `ReadyStatus`, `LaunchHandle`, `StatusFile` = `status.json`).
- `config.py`: pydantic config with TOML load/save at `$GPURELAY_HOME/config.toml`
  (default `~/.gpurelay`), atomic save. Hard rules enforced by the schema:
  - one fixed section per backend type; unknown sections such as `[backends.kaggle2]` are
    rejected (one account per service);
  - unknown keys are rejected everywhere, so credentials (`key`, `token`, ...) cannot be put in the
    config file;
  - `backends.lightning.enabled = true` is rejected (hard rule 4).
- `storage/base.py` (`Storage` protocol, key/prefix validation) and `storage/localfs.py`
  (`LocalFSStorage`): writes go to a temp sibling, are fsynced, then `os.replace`d; retries on
  transient Windows `PermissionError`; temp files hidden from `list`.
- `cli.py` (typer): all spec §15 commands registered with their options; each prints
  `not yet implemented (phase N)` and exits 1. `--version` works. `python -m gpurelay` works.

**Tests** (`tests/unit/`): config round-trip and hard-rule rejections; model parsing
(`--data` specs, `status.json` spec example, `LATEST.json` checksum coverage); LocalFS
put/get/list/delete/exists, key validation, failed write keeps old content, concurrent
reader never sees a partial file; CLI command set matches spec, `--help` for every command,
each stub reports its phase.

**Decisions / deviations**

- Project lives at the repository root (the spec's `gpurelay/` folder is the repo itself).
- Dependencies for later phases (`huggingface_hub`, `nbformat`, `IPython`, `keyring`, `plyer`,
  `numpy` for tests) are added in the phase that first uses them, keeping the Phase 0 install small.
- Default storage kind is `hf_hub` per spec; it has no repo until `gpurelay init` (Phase 6). Tests
  and local runs set `storage.kind = "localfs"`.
- Kaggle `accelerator` is a free string (not an enum) because the accepted identifiers are a
  `VERIFY` item.
- Simulated-backend settings are not in the config schema yet; they arrive with the Simulated
  backend (Phase 2/5).

**Deferred**: everything from Phase 1 onwards.

**Open `VERIFY` items**: see `docs/verify.md` (11 items, all open).

## Phase 1 — Checkpoint core

**Built**

- `runtime/rng.py`: capture/restore Python `random`, NumPy global RNG, torch CPU + all CUDA
  devices, TF global generator. Lazy: only libraries already imported are captured; libraries in
  a saved state are imported on restore.
- `runtime/state.py`: variable classification (skip / torch-aware / plain), capture to a
  directory, restore with the **load-into-live-object** rule (spec §7.5) and whole-object fallback;
  unpicklable values are skipped and reported by name.
- `runtime/checkpoint.py`: `CheckpointManager` (snapshot → background persist, one persist at a
  time, stall timing, seq continuity across segments, stale local snapshot cleanup) and
  `load_latest_checkpoint` (verify checksums, fall back to older checkpoints, `LoadedCheckpoint.restore`).
- `runtime/persist.py`: zstd compression with streaming SHA-256, upload with retry/backoff,
  stored-checksum verification + re-upload on mismatch, `manifest.json` + `LATEST.json` commit,
  retention of the last 2 committed checkpoints. Measures raw/stored bytes, compress/upload
  time, throughput (inputs for the Phase 2 adaptive interval).
- `runtime/env.py`: environment fingerprint stored in `meta.json`.
- `retry.py`: `RetryPolicy` (exponential backoff, cap, jitter; never retries not-found or
  `BaseException`). Uses a **private** RNG for jitter so retries never shift the user's random stream.
- `storage/layout.py` (remote key helpers); `LocalFSStorage.get_file` now streams;
  `LocalFSStorage.sha256` + optional `ChecksumStorage` protocol for cheap upload verification.
- `docs/checkpoint-layout.md`: protocol, remote layout, differences from spec §8.3, limitations.
- CI: new `test-torch` job (CPU torch); core jobs run the whole suite with ML tests skipped.

**Tests** (130 total; 123 pass + 3 skip in a core-only venv without torch/numpy)

- Equivalence (`tests/integration/test_checkpoint_torch.py`): MLP + AdamW + StepLR + GradScaler
  using dropout, torch/NumPy/Python RNGs; checkpoint mid-run, resume in a fresh namespace with
  differently initialised objects and scrambled RNG → losses, weights, optimizer state, LR, and
  grad scale **bit-identical** to the uninterrupted run. Also: resume every 3 steps (4 segments),
  whole-object fallback with no live objects, optimizer ↔ parameter identity after restore.
- RNG round-trips (Python, NumPy, torch CPU incl. dropout masks).
- Faults: in-process kill during snapshot and during persist; **real process hard kill**
  (`os._exit`) during snapshot and during persist → `LATEST.json` still names the previous
  checkpoint, which restores correctly, and the next segment continues; transient storage
  outage retried; prolonged outage → persist fails, previous commit kept, recovers later;
  corrupted upload detected and re-uploaded; corrupted/missing file → fallback to previous;
  corrupted `LATEST.json` → manifests; everything corrupted → `CheckpointCorruptError`;
  uncommitted manifest ignored; retention; never two persists at once.

**Decisions / deviations** (details in `docs/checkpoint-layout.md`)

- Snapshot writes uncompressed files locally; persist compresses every non-JSON file
  (`*.zst`) and hashes the stored bytes.
- Added per-checkpoint `manifest.json` so fallback works even if `LATEST.json` is damaged.
- Sequence numbers are never reused across segments.
- Torch-aware objects are saved both as state dicts and (when picklable) as whole objects in one
  `torch.save` call (storages deduplicated); restore prefers loading into live objects.

**Deferred**

- Working-file tracking (`files.tar.zst`), output capture (`outputs/`), framework checkpoints
  (`framework/`): Phase 2/3 via the `extra=` hook.
- Adaptive checkpoint interval and forced-checkpoint condition: Phase 2 (measurements are
  already exposed in `SnapshotInfo` / `PersistResult`).
- CUDA RNG/device mapping exercised only on real GPU backends (Phase 6).

**Open `VERIFY` items**: 12 (added #12: restoring tensors saved on `cuda:N` with fewer GPUs).

## Phase 2 — Script runtime with SDK, output capture, watchdog, Simulated backend

**Built**

- `sdk.py` (re-exported as `import gpurelay as relay`): `state`, `range`, `chunks`,
  `save_result`, `load_results`, `safe_point`, `is_active`; plain passthrough outside GPU Relay.
  Contract and resume semantics documented in `docs/sdk.md`.
- `runtime/loops.py`: position tracker. Loops/safe points are identified by order of
  occurrence (nested contexts included); resume = re-execute in replay mode, skip finished
  loops, fast-forward the interrupted one, restore at the target safe point; divergence →
  `ResumeMismatchError` (exit 88).
- `runtime/session.py`: per-segment runtime (checkpoint at safe points, handoff, restore into
  live objects with in-place update of registered containers, chunk results, output chain).
- `runtime/capture.py`: stdout/stderr tee with byte offsets (recording off during replay);
  `pyplot.show` patched lazily through an import hook (no matplotlib import unless the user
  imports it) to save figures as blobs.
- `runtime/workfiles.py` + `runtime/blobs.py`: working-file tracking (mtime/size → sha256),
  content-addressed blob store; files restored before replay and reverted at the resume point.
- `runtime/watchdog.py`: adaptive interval `clamp(t_snap/overhead_budget, min, max_lost_work)`,
  forced handoff before the deadline, SIGTERM/SIGINT → handoff at next safe point (second signal
  aborts), safe-point-gap warning.
- `runtime/status.py`: `status.json` heartbeat. `runtime/executor_script.py` + `runtime/entry.py`:
  run the script as `__main__` compiled from its real path (original line numbers), exit codes
  0/85/86/87/88, runtime errors raised through user code as `BaseException`s so `except
  Exception` cannot swallow them, CUDA OOM hint.
- `backends/base.py` (Protocol), `backends/local.py` (subprocess per segment, fresh working
  dir, `PYTHONHASHSEED=0`), `backends/simulated.py` (hard kill after fixed/random delay, short
  deadlines, fake GPU labels, kill records).
- `bundle/manifest.py` + `bundle/packager.py`: minimal bundle (files + manifest); Phase 5 packs it.
- `controller/assemble.py`: merged log (truncate each segment at its committed offset),
  figures, files, `report.json`; uploaded to `jobs/<job>/final/`.
- `controller/local_runner.py` + `gpurelay run-local [--simulate-kill-every] [--simulate-deadline]
  [--out] [-q]`: relays segments until completion; no-progress guard; streams console output.
- Checkpoint layer extended with job-level uploads and blobs committed before `LATEST.json`.
- `storage/factory.py` (`localfs:` URIs), `pathutil.long_path` (Windows paths > 260 chars),
  `durations.py`.

**Verification**

- Plan "done when": `gpurelay run-local tests/corpus/sdk_mlp.py --simulate-kill-every 20s`
  (60 epochs, slowed) → 2 hard kills, 3 segments; merged log, metrics, figure PNG, and model
  weights **identical** to the uninterrupted `run-local`.
- Equivalence harness `tests/integration/equivalence.py`; tests (`test_equivalence_sdk.py`):
  chunked eval with random hard kills / forced handoffs / both, MLP (torch) with kills + handoffs
  — all identical to the uninterrupted run; chunked eval relay output identical to a **plain
  `python` run**; lost work per kill ≤ `max_lost_work_s` + one safe-point gap + 1 s slack.
- Failure modes: user exception → `failed_user`, not retried, traceback shows the script's own
  line numbers without runtime frames; `sys.exit(3)` → user error; divergent control flow →
  exit 88.
- Tracker property test: a program with nested loops, explicit safe points and `break` is
  resumed from **every** safe point and always reproduces the uninterrupted run.
- 179 tests pass (≈2 min, mostly real subprocess runs); 170 pass + 5 skip in a core-only venv.

**Decisions / deviations**

- Handoff condition adds the last safe-point gap: `time_left < gap + t_snap + S/bw + margin`
  (the spec's formula alone overran the deadline by one iteration).
- Logs are stored with `\n` line endings on all platforms.
- Runtime settings `min_interval_s`, `max_lost_work_s`, `safety_margin_s`, `heartbeat_s` are
  floats (tests use sub-second values). `run-local` shrinks intervals automatically when
  simulating short sessions.
- Explicit `relay.safe_point()` inside an iteration: on resume the iteration is re-executed in
  replay mode up to that point; values computed before it must be registered to be correct.

**Deferred**

- Automatic loop instrumentation, framework hooks, analyzer: Phase 3. Notebooks: Phase 4.
- Blob garbage collection; mid-iteration DataLoader resume; capturing fd-level output.

**Environment note**: on this Windows machine a session needs ~7 s before user code runs
(Python + torch import + first-use init), so simulated sessions shorter than ~10 s cannot make
progress with torch jobs (the no-progress guard reports this).

**Open `VERIFY` items**: 13 (added #13: SIGTERM at session limits).

## Phase 3 — Automatic instrumentation, framework hooks, analyzer v1

**Built** (design: `docs/automatic-resume.md`)

- `runtime/instrument.py`: splits a script into top-level statements, rewrites every outermost
  `for` loop reachable from module level (incl. inside `if`/`with`/`try`/`match`) into
  `__relay__.loop(loop_id, ITER)` and compiles the transformed AST directly (original line
  numbers); records written/mutated names, call arguments, `global` names, side effects,
  uninstrumented loops (`while`, `async for`, loops in called functions), definitions.
- Runtime now executes scripts **statement by statement** (notebook-style replay-and-restore,
  spec §7.1, applied to statements): replay prefix of cheap side-effect-free statements +
  definitions; other statements before the resume point are skipped and their results come from
  the snapshot; statement boundaries are safe points; skipped statements that built torch objects
  are re-executed (when cheap) so optimizers bind to the live model; values restored before the
  target statement and again (with RNG and files) at the resume point.
- `runtime/loops.py`: statement-scoped positions with loop ids (divergence detection), passthrough
  mode for setup replay, framework events; eager fast-forward before restore (slice
  sequences/ranges/arrays, consume other iterators and log it).
- `runtime/hooks/` (`hf.py`, `lightning.py`, `keras.py`), installed lazily on import via
  `runtime/importhooks.py`, version-gated (warn + no hook otherwise):
  - transformers 4/5 `Trainer.train`: saves when due, aligned to logging steps with lookahead;
    forced saves go to a private dir with output muted; resume via `resume_from_checkpoint`.
  - Lightning 2 `Trainer.fit`: `Checkpoint` callback at train-epoch end; resume via `ckpt_path`.
  - Keras 3 `Model.fit`: `BackupAndRestore` + relay callback; RNG re-applied at the first resumed
    epoch.
- Output continuity: Python `warnings` shown in earlier segments are not repeated; output of
  relay-initiated saves is muted; unpicklable variables are reported once per name.
- `analyzer/static.py`, `analyzer/report.py`, `gpurelay analyze PATH [--requirements F] [--json P]
  [--no-json]`: per-block FULL/PARTIAL with covering layer, overall FULL/PARTIAL/RESTART_ONLY,
  risks (unpicklable state, nondeterminism, side effects, uninstrumented loops), GPU-time
  suggestions (AMP, DataLoader workers/pin_memory, FID feature caching, frequent evaluation),
  imports missing from requirements. `run-local` refuses RESTART_ONLY jobs unless
  `--accept-restart-only`.
- New unmodified corpus: `mlp.py` (1), `vae_fid.py` (3), `hf_trainer.py` (4), `lightning_tiny.py`
  (5), `keras_tiny.py` (6), `chunked_eval.py` (7), `while_loop.py` (8); expected analyses in
  `tests/corpus/expected/`.

**Verification** (223 tests, ≈6 min; 209 pass + 10 skip in a core-only venv)

- Plan "done when": **unmodified VAE + chunked-FID** survives random hard kills (2, 3, and 5 kills
  in development runs; CI test uses seed 5) with identical log, figure, metrics, and weights.
- Unmodified corpus 1, 3, 4, 5, 6, 7 equivalent under the Simulated backend (kills and/or
  handoffs); corpus 8 reported `PARTIAL` and completes correctly with the `while` block restarted.
- Traceback line numbers through instrumented code match the original file (unit + integration).
- Analyzer snapshot tests for every corpus item; `main()` pattern → RESTART_ONLY.
- Hooks degrade on unsupported major versions (fake-module tests).
- CI: new `test-frameworks` job (transformers, lightning, keras on torch backend).

**Bugs found and fixed while verifying**

- Optimizer/model reference split after resume when the optimizer's statement was skipped → the
  constructor-replay rule above.
- HF: progress bars from relay-forced saves leaked into the log; one logged loss changed after
  each resume (HF's loss accumulator is not checkpointed) → muted forced saves, logging-step
  alignment with lookahead based on measured step/save times.
- Lightning: relay callback ran before the module's `on_train_epoch_end` (lost the epoch's print)
  → `Checkpoint` subclass; Lightning library warnings repeated after resume → job-wide warning
  dedupe.
- Keras: `fit` setup consumes RNG before the first resumed epoch's shuffle → RNG re-applied there.

**Decisions / deviations**

- Resume granularity is the top-level statement (plus loop iterations inside it) rather than
  re-running the whole script as in Phase 2; snapshot = names written by non-replayed statements +
  torch objects (a superset of spec §7.3's loop-body variables, needed when finished loops are
  skipped).
- Framework hooks save at step (HF) or epoch (Lightning, Keras) granularity; epoch-long Lightning/
  Keras epochs can exceed `max_lost_work_s` (the watchdog warns).
- `relay.save_result`/SDK unchanged; SDK loops are never double-instrumented.

**Deferred**: notebooks (Phase 4), `--probe` (Phase 8), loops inside functions / `while`
(Phase 9), content-deduplicated variable storage (large frozen models are re-saved each checkpoint).

**Environment note**: framework sessions need 10-17 s before training starts on this machine, so
framework equivalence tests use forced handoffs with ≥16 s sessions.

**Open `VERIFY` items**: 16 (added #14-16: hook behaviour with platform-preinstalled transformers,
Keras 3 + TensorFlow backend, Lightning versions).

## Phase 4 — Notebook support

**Built**

- `runtime/executor_notebook.py`: code cells run through IPython `InteractiveShell.run_cell`
  in one `__main__` namespace (magics, `!cmd`, `display()`, last-expression results). A
  recording display hook, display publisher, and traceback handler capture per-cell outputs
  (`stream`, `display_data`, `execute_result`, `error`). Matplotlib figures left open at the end
  of a cell are captured as `image/png`, like the inline backend. Runtime control exceptions
  (handoff, abort, divergence) are registered as IPython custom exceptions so they are not
  swallowed.
- Cells are resumable units exactly like script statements: short side-effect-free cells are
  replayed, long ones skipped and restored from the snapshot, and an interrupted cell resumes
  inside its instrumented loop (cell-level AST transformer). Imports/defs/classes of skipped
  cells are still executed. Package-install cells (`%pip install`, `!pip install`, conda, apt)
  are always replayed (environment setup); other shell commands are side effects (never
  replayed).
- Per-cell outputs are stored as blobs referenced from every checkpoint; the interrupted cell
  keeps its outputs up to the checkpoint. Assembly writes `final/output.ipynb` (the submitted
  notebook with outputs; execution counts renumbered 1..N).
- Analyzer: `analyze` accepts notebooks (one block per code cell; shell commands, package
  installs; findings reported as `cell N, line L`). `run-local` accepts notebooks.
- Corpus: `mlp_notebook.ipynb` (corpus 2), `vae_fid_notebook.ipynb` (corpus 3, notebook version),
  `chunked_eval_notebook.ipynb` (no torch).
- IPython and nbformat are now core dependencies.

**Verification**

- Corpus 2 and the notebook version of corpus 3 survive random hard kills with identical
  merged logs, files, figures, and `output.ipynb` cell outputs (execution counts ignored; object
  reprs with memory addresses normalized, since those differ between any two runs).
- Killed inside the long training cell → the next segment's log starts at the next epoch
  (resumed inside the loop, not from the cell start).
- Unit tests: cell planning (magics, setup vs shell side effects, definitions code), runtime
  instrumenter loop ids == planned loop ids, output-notebook assembly; analyzer snapshots for the
  notebooks.

**Bug found and fixed**: a skipped cell mixing a magic with imports left its modules undefined
after resume → top-level imports/defs of skipped cells are always re-executed.

## Phase 5 — Controller, scheduler, quota ledger, estimator

**Built** (`docs/controller.md`)

- `controller/db.py`: SQLite state (`jobs`, `segments`, `checkpoints`, `quota_ledger`, `events`,
  migrations via `schema_version`).
- `bundle/`: `bundle.tar.zst` (deterministic) + manifest with content hash + pinned runtime wheel
  (built from the installed package, reproducible bytes, installable) + `local:` data uploaded
  once to `data-cache/<sha256>/` and staged into the working directory (excluded from file
  tracking); `url:` data downloaded in the session.
- `controller/quota.py` (weekly rolling / monthly / unknown / unlimited; live usage; user
  corrections; next reset), `controller/estimator.py` (outermost-loop rate from `status.json`, or
  the `--estimate` hint; "fits this week / spans N weeks" message), `controller/scheduler.py`
  (pure policy of spec §13.4), `controller/daemon.py` (tick loop: requests, segment polling,
  crash detection, ledger, assembly, scheduling, budget-low warnings), `controller/notify.py`
  (desktop via plyer when installed + ntfy.sh), `controller/views.py` (status tables),
  `controller/registry.py` (backends from config; simulated backends configurable for testing).
- Runtime: `jobs/<job>/control.json` polled by the heartbeat → pause = handoff at the next safe
  point; status rate now measures the outermost loop.
- CLI: `submit` (analyze → RESTART_ONLY gate → bundle → queue → estimate), `up [--once]
  [--until-idle] [--interval]`, `status [job]`, `logs [-f]`, `quota [show|set]`,
  `pause/resume/cancel`, `fetch [--to]`.

**Verification**

- Scheduler/controller unit tests with a fake clock and fake backends: budget exhaustion →
  `waiting_for_quota` (with the earliest reset) → still waiting inside the window → resumes after
  the reset; deadline never exceeds remaining budget; priority/submission order; preference for
  most budget, `--prefer`, same GPU on ties; concurrency limits; one live segment per job; manual
  backend only → `awaiting_user` (+ running once the session reports in); user failures not
  retried; infra failures retried then `failed_infra`; crashes with progress not counted; pause →
  control flag → paused → resume → cancel; budget-low notification; ledger usage == segment
  durations.
- `status` snapshot test (per-backend remaining budget, job progress and ETA).
- **End to end**: three jobs (script, SDK script, notebook) × three Simulated backends (different
  budgets and GPU labels; one hard-kills sessions, one forces handoffs) through the real
  controller → all complete with outputs equivalent to uninterrupted runs, segments of
  different jobs ran in parallel, no job ever had two live segments, ledger totals match segment
  durations per backend.
- Manual CLI run (submit ×2 with priorities, `up --until-idle`, `status`, `logs`, `fetch`,
  `quota set`) against a config with simulated backends.

**Decisions / deviations**

- Jobs/segments stored as JSON documents (+ indexed columns) rather than wide tables.
- A crash that committed a new checkpoint does not count toward `max_infra_retries`.
- Local backend = unlimited budget; Kaggle/Colab config sections are ignored with a warning until
  phases 6/7.
- Requirements are recorded in the bundle; installing them in cloud sessions (respecting
  preinstalled CUDA builds) is done by the Kaggle/Colab bootstraps (phases 6/7).

**Deferred**: controller restart re-attaching local sessions, `up --detach` (Phase 8);
`kaggle:`/`hf:` data (Phase 6).

## Phase 6 — Hugging Face storage and Kaggle backend

**Built**

- `storage/hf_hub.py`: private dataset repo; reads bypass the Hub cache; errors mapped to
  storage errors (404 → not found, 429/5xx/network → retried with backoff); `BatchStorage`
  (`put_many` / `delete_many` = one commit), `sha256` from LFS metadata (download for small
  files), `used_bytes`, `squash_history`. `hf:<repo>` storage URIs; token from `HF_TOKEN` or the
  OS keyring (`credentials.py`).
- Checkpoint persister: batched path for `BatchStorage` (blobs + uploads + checkpoint files +
  manifest in one write, verified, then `LATEST.json`) = **2 Hub commits per checkpoint**;
  retention deletes in batches. LocalFS implements the batch API too (both paths tested).
- `backends/kaggle.py` + `templates/kaggle_bootstrap.py.tmpl`: one private kernel per job, a
  new version per segment, `session_timeout_seconds` = segment deadline, runtime deadline 120 s
  earlier (boot allowance), CPU-only option (`enable_gpu = false`), accelerator via
  `machine_shape`, duplicate-session guard (a busy backend is not an infra failure), status
  mapping, console log fetch. The official client is imported only after credentials are
  detected (it exits the process otherwise). Heartbeats are at least 120 s on Hub-backed
  sessions.
- Bootstrap: reads the token from the private secrets dataset, installs the pinned runtime
  wheel from storage, and runs the runtime with `--install-requirements` (keeps the image's
  torch/TF unless a different major.minor is pinned: `runtime/requirements.py`).
- Data: `kaggle:` (attached + mounted on Kaggle, downloaded elsewhere) and `hf:`
  (`snapshot_download`) sources.
- `gpurelay init`: creates/verifies the **private** storage repo (refuses public), stores the
  token in the keyring, checks Kaggle credentials (one account per service), and creates the
  private `gpurelay-secrets` dataset.
- `tests/corpus/kaggle_smoke.py` (deterministic on any machine), `docs/manual-tests.md`.

**Verification so far**

- Unit tests with fakes (no network): HF storage semantics, one commit per batch, rate-limit
  retries, full checkpoint save/load through `HFHubStorage` (2 commits), LFS/downloaded
  checksums; Kaggle metadata always private with the secrets dataset attached, bootstrap
  compiles and contains no secret (with `HF_TOKEN` set in the environment), duplicate guard,
  status mapping, readiness checks; init (private repo, refuses public, secrets dataset, one
  account); requirements policy.
- All earlier integration tests re-run after the persister change: pass (run in groups).
- Storage check (manual test step 2): job-93483bc15ed3 completed on the local backend with the
  private Hub repo as storage.
- **Kaggle smoke test (2026-10-22)**: job-12b247cf374c (`tests/corpus/kaggle_smoke.py`, 40 steps,
  CPU sessions, `max_session_s = 600`) ran as 4 Kaggle sessions (kernel versions 4-7) with 3
  handoffs at safe points (final checkpoint seq 22). `result.json` and the 40-line log are
  **byte-identical** to a local uninterrupted run (`SMOKE_SLEEP=0`).
- Bugs found by the smoke test and fixed (each with a regression test):
  - Kaggle answers 403 (not 404) for a kernel that does not exist yet → treated as missing.
  - `init`'s session-token prompt was ambiguous and a Kaggle token was stored in the secrets
    dataset (sessions failed with HF 401). The prompt now asks for a Hugging Face token, the
    token is checked against the Hub (prefix, `whoami`, repo access), `init
    --rotate-kaggle-secret` replaces it (new dataset version, old versions deleted), and the
    bootstrap stops with a clear message if the secret is not an `hf_` token.
  - Two `gpurelay up` processes could run at once (an orphaned controller kept running after
    its shell was stopped). `up` now holds an OS file lock (`controller/lock.py`), released by
    the OS even if the process is killed.
- Not covered: GPU sessions (accelerator identifiers, VERIFY #17), Kaggle's own timeout,
  framework hooks on the Kaggle image (VERIFY #14-16).

**Decisions / deviations**

- Templates use `string.Template` (`.tmpl`), not Jinja (`.j2`): no extra dependency.
- The deadline is computed inside the session from its own start (queue time on Kaggle does
  not shorten the segment).
- Kaggle sessions are not cancelled through the API (no public call); pause/cancel use
  `control.json` (handoff at the next safe point).
- `huggingface_hub` and `keyring` are now core dependencies; `kaggle` stays an extra
  (`>=2.0`).

## Phase 7 — Colab manual backend

**Built**

- `backends/colab_manual.py` (`automated = False`, hard rule 2): writes one resume notebook per
  job (`sessions/colab_manual/resume_<job>.ipynb`) and, per segment, a request in storage
  (`jobs/<job>/manual_session.json`: segment id, deadline, start checkpoint). It starts,
  drives, and connects to nothing; `poll` is always `unknown` (progress comes from
  `status.json`); `cancel` withdraws the request.
- `templates/colab_resume.py.tmpl` (percent format → nbformat): reads the Hugging Face token
  from Colab user secrets (`GPURELAY_HF_TOKEN`, configurable), checks the request (refuses if
  there is none, if this segment already ran or is running in another tab, or if the job moved
  on), installs the job's pinned runtime wheel, runs the runtime with the same environment as
  the other backends and streams its output; interrupting the cell asks the job to hand off.
  No secrets in the notebook.
- `gpurelay colab <job>`: refreshes the notebook and prints its path, whether the controller is
  waiting for you, and the steps.
- Controller: a manual segment's GPU time and deadline start when the user starts it (not at
  the request); a request nobody started records no usage. The registry adds the Colab slot
  (unknown budget, one session).
- `backends/common.py`: Hub session settings shared by Kaggle and Colab.

**Verification**

- Unit: the notebook is valid nbformat, generic per job, contains no secret (with `HF_TOKEN`
  set), and quotes config values safely; the notebook's own check cells run against fake
  Colab/Hub modules (starts the requested segment; refuses: no request, ran elsewhere, running
  in another tab, rerun, stale; bad or missing secret); request/cancel; readiness; registry
  slot; `gpurelay colab`; controller manual timing and no usage for unstarted requests.
- Compliance (hard rule 2): the backend module imports nothing that can start processes, open
  browsers or sockets, or reach the network (`subprocess`, `webbrowser`, `socket`,
  selenium/playwright, SSH/tunnel libraries, HTTP clients, ...) and mentions no SSH, tunnel, or
  webdriver; the notebook contains no remote-access code; `automated` is False.
- Integration (`tests/integration/test_colab_relay.py`): the Simulated backend hands off, its
  budget runs out, the controller requests Colab; the test runs the generated notebook's check
  cells and then the runtime as the notebook would → the job completes with outputs equivalent
  to an uninterrupted run; running the notebook again refuses. Stable over repeated runs.
- **Real smoke test (2026-10-22)**: job-54b80887c0d7 (`tests/corpus/kaggle_smoke.py`, 40 steps)
  ran segment 0 on Kaggle (CPU, handed off at checkpoint 2); with Kaggle's budget set to 0 the
  controller asked for Colab and notified; the owner uploaded the notebook and pressed Run all
  ~10 min later; segment 1 ran in Colab from checkpoint 2 to completion (checkpoint 11).
  `result.json` and the 40-line log are **byte-identical** to an uninterrupted local run.
  Colab GPU time was counted from the user's start (0.30 h), not from the request.

**Decisions / deviations**

- The notebook runs the runtime as a child process of the notebook cell (like
  `!python job.py`), not inside the kernel itself: same entry point and environment as every
  other backend (`PYTHONHASHSEED=0` cannot be set inside a running kernel; signal handling and
  exit codes), so resumed results stay identical. It still runs only while the user's cell
  runs, with its output in the notebook.
- Template is a `string.Template` percent-format script (`.py.tmpl`), not `.ipynb.j2`.
- The notebook is generic: the segment to run comes from the request in storage, so one
  notebook serves every Colab segment of a job (spec §10.5).
- Not done: moving a job that waits for the user to an automated backend whose budget comes
  back meanwhile (it keeps waiting for Colab until the user runs it, pauses, or cancels).

**Open `VERIFY` items**: added #25-27 (Colab secrets access, session clock start, image
details); #8 stays open until the smoke test.

## Phase 8 — Dashboard, probe, hardening, README

**Built**

- `gpurelay dashboard` (extra `gpurelay[dashboard]`: FastAPI + uvicorn): one self-contained page
  (no build step, no external resources, data inserted as text only) on `127.0.0.1:8765`. Jobs
  with progress bars and ETA, per-job segment timeline, segments, events, live log tail, GPU
  budget per backend, a "Colab action needed" banner, controller running/stopped, and
  pause/resume/cancel (POST needs a custom header, so other websites cannot trigger it).
  JSON API: `/api/overview`, `/api/jobs/<id>`, `/api/jobs/<id>/log`, `POST /api/jobs/<id>/<action>`.
- `gpurelay analyze --probe [--probe-time 60s]` (`analyzer/probe.py`): runs the job here under
  the real runtime against a throwaway store, hands off at the deadline, and reports startup
  time, outermost-loop rate, projected total on this machine, peak RAM of the process tree
  (`psutil`, now a core dependency), checkpoint size, and device.
- Hardening:
  - The controller survives storage/API outages: a failing tick is logged and retried with
    backoff (up to 8x the interval); after 5 failures in a row the user is notified once.
    One broken segment no longer stalls polling of the others.
  - Clock-skew tolerance: a heartbeat is stale only when it has not *changed* for the
    threshold, measured on the controller's own clock (remote clocks can be off by hours).
  - Controller restart recovery (tested): a session that outlives the controller is followed
    to completion; one that died with it is detected, relaunched, and resumed.
  - `merged_log` (used by `logs` and the dashboard) reads only the checkpoint's `meta.json`
    instead of downloading the whole checkpoint state.
  - The per-segment environment record is refreshed at the end of the segment, so GPUs used
    through torch appear in reports (it was taken before the job imported torch).
  - Clearer messages: `gpurelay init --local` (no accounts) and a "no storage configured" hint
    naming both setups; completion message `outputs ready: gpurelay fetch <job>`.
- README: install, local quick start with `examples/quickstart.py`, free-GPU setup, how it
  works, hard rules, limitations, commands.

**Bug found by the full regression run (fixed)**

- Names bound inside compound top-level statements (`try: import matplotlib.pyplot as plt`,
  `if ...: from x import f`) were lost on resume when that statement was skipped (modules and
  functions are never pickled), giving `NameError` (seen in the VAE+FID corpus after a kill
  landed after its `try` block). The state capture now records such bindings by name
  (modules, and functions/classes of importable modules) and imports them again on restore
  when replay did not recreate them. Regression corpus `tests/corpus/conditional_imports.py`
  fails without the fix and passes with it.

**Verification**

- Unit (295): dashboard API (page self-contained, overview, detail, log tail, actions and the
  header guard), clock skew (behind / in sync / ahead), outage backoff and alert, per-segment
  isolation, controller-running probe, meta-only log read, import bindings, `init --local`.
- Integration: controller restart mid-job (session survives / dies), probe (rate and
  projection within 35 %, completed job, failing job), Colab relay, controller e2e, and the
  full corpus regression (automatic resume incl. VAE+FID, SDK, notebooks, torch checkpoints):
  all pass, run one file at a time.
- Dashboard checked in a browser with a demo home (four jobs on two Simulated backends with
  handoffs, a controller restart mid-job): rendering, timeline, log, budgets.
- **Done-when check:** in a fresh virtualenv (`pip install .`, fresh `GPURELAY_HOME`) the
  README quick start works as written: `init --local`, `analyze`, `run-local
  --simulate-kill-every 7s` (4 segments, 3 kills), `submit` / `up --until-idle` / `status` /
  `fetch`; `result.json` and the log are identical to a plain `python examples/quickstart.py`.

**Decisions / deviations**

- The dashboard is an optional extra (`fastapi`, `uvicorn`); the core install stays small.
- No login on the dashboard: it listens on localhost and warns when bound elsewhere.
- Not done: moving a job that waits for Colab back to an automated backend whose budget
  returned (Phase 7 note), mid-epoch DataLoader resume and nested-loop safe points (Phase 9).


## Phase 9 — Stretch items

All six items are built, each with its own tests. Items 4 and 5 still need a live test on the
owner's accounts.

**1. Chunk-parallel execution** (`relay.chunks(..., parallel=True)`; `runtime/parallel.py`)

- The main session walks chunks in order. Worker segments, launched by the controller on
  spare automated backends (`[scheduler] max_parallel_workers`, default 2), resume from a
  main checkpoint taken inside the region and take chunks from the end.
- Coordination happens through storage only: per-chunk claims, which a session keeps alive
  by refreshing them on each heartbeat and which are judged on the reader's own clock, and
  done markers with the result and the chunk's printed output. Workers never write
  checkpoints or the job's status.
- Each chunk is seeded from (seed, chunk id), with and without GPU Relay, so results do not
  depend on placement. Output from chunks computed elsewhere is merged in chunk order.
- Tested with the real controller and three Simulated backends, with and without kills:
  workers ran concurrently and computed chunks, and the log and results are byte-identical
  to a sequential run and to plain `python`.

**2. Nested-loop safe points and mid-epoch DataLoader resume**

- `[runtime] loop_depth` (default 2) instruments nested `for` loops.
- **Adaptive tracking:** a nested loop whose first 32 iterations average under 0.5 ms is not
  tracked for the rest of its run (overhead dropped from 2.3 µs to 0.15 µs per iteration).
- **Sampler-driven loops (torch DataLoader):** the RNG state at the start of iteration is
  checkpointed and restored before the iterator is recreated, so the shuffle order is the
  same. Single-process map-style loaders skip finished batches without loading them.
- Tested: a torch job whose sessions are all shorter than an epoch completes with identical
  output. The test fails if the RNG restore is disabled.

**3. `while` loops and loops inside functions** (opt-in `instrument_while`,
`instrument_functions`)

- **`while` loops** become a tracked `for` with the condition checked each iteration.
- **Function loops:** the function's assigned locals are checkpointed and written back into
  the running frame on resume (Python 3.13+ `f_locals`; `PyFrame_LocalsToFast` before that).
- The analyzer uses the configured options.
- Tested: equivalence under kills or handoffs (pure Python and torch). The pure-Python
  function test fails without the frame write-back.

**4. Lightning AI backend** (`backends/lightning.py`, extra `lightning-ai`)

- The terms of service and the official SDK were reviewed (`docs/verify.md` #10). The SDK
  is official, and every name used was checked against lightning-sdk 2026.10.13. The terms
  are ambiguous about automated job submission, so the backend needs `enabled = true` and
  `terms_reviewed = true`.
- The token goes in the job's environment, never in the command.
- The bootstrap enforces a hard time limit, because Lightning has none of its own.
- Tested with a fake client. **Pending:** a live run on the owner's account.

**5. Google Drive storage** (`storage/gdrive.py`, extra `gdrive`, `gpurelay init --gdrive`)

- Uses the `drive.file` scope, so GPU Relay sees only files it created.
- A write to an existing key replaces that file's content in place. Duplicate names from a
  race resolve to the newest file. 404 means not found; rate limits are retried.
- Tested against an in-memory Drive (the storage contract plus a checkpoint round trip).
  Request parameters were validated against the real `google-api-python-client`.
- Kaggle, Colab, and Lightning sessions still need Hugging Face storage.
- **Pending:** a live run with the owner's OAuth client.

**6. Always-on controller** (`controller/lease.py`, `controller/migrate.py`, `deploy/`,
`docs/always-on.md`)

- **Cross-machine lease** in storage: a second controller refuses to start while the lease
  is renewed, takes over a stale one, and `--takeover` makes the old holder stop at its
  next renewal.
- **`gpurelay export-state` / `import-state`** move the config and a consistent copy of the
  job database, never secrets.
- **Dockerfile and systemd unit.** The dashboard can serve the Colab notebook for download.
- Tested:
  - lease unit tests;
  - the image built and ran the quickstart inside a Linux container, with results
    identical to plain Python;
  - two containers on one shared store: the second refused to start after 111 s, then
    `--takeover` made the first stop itself.

**Fixes found while finishing**

- Kaggle `accelerator` default `gpu_p100` is not a valid `machine_shape`. The official SDK
  lists `NvidiaTeslaT4` and `NvidiaTeslaP100`, so the default is now `NvidiaTeslaT4`.
- Hub HTTP request lines leaked into `gpurelay up` output: huggingface_hub 2.x logs through
  the `httpx2` logger, which is now quieted too.
- Two timing-sensitive integration tests (torch start-up versus kill windows) now use
  deadline handoffs. They still fail without the features they test.

- Random generator *objects* (`random.Random`, NumPy `Generator`/`RandomState`,
  `torch.Generator`) created in a replayed setup statement and advanced in the loop without
  reassignment were not checkpointed, so their stream restarted after a resume (found by the
  Kaggle GPU smoke test). They are now always snapshotted and restored into the live object
  (other holders, e.g. a DataLoader, keep a valid reference). Regression corpus
  `rng_objects.py` fails without the fix.

**Kaggle GPU smoke test (2026-10-22)**: `tests/corpus/gpu_smoke.py` (60 CUDA steps, an
integer checksum from a `torch.Generator`, and a training loss) ran on 2x Tesla T4
(`NvidiaTeslaT4`) as two Kaggle sessions with a handoff; all 60 step lines are identical to a
CPU run. The laptop slept for 53 minutes during an earlier run: the session finished on its
own and the controller picked it up on wake.

**Verification:** 347 unit tests; every integration file passes (run one file at a time).
`test_keras_unmodified_handoffs` is timing-sensitive under heavy load (failed once in a batch
run, passed 3/3 alone).

## Hardening campaign (2026-10-23)

Goal: find loopholes with new kinds of tests, then fix every bug they found. Each fix has a
regression test that fails without it.

**New test kinds**

- *Adversarial scripts* (`tests/integration/test_adversarial.py`): nine unmodified scripts
  written to break resume. They cover:
  - lists that grow while being iterated;
  - state kept in closures;
  - objects mutated only through their own methods;
  - unusual control flow;
  - iterator kinds;
  - globals and attributes;
  - working files;
  - very tight loops;
  - early exit.

  Each runs under random hard kills and must match a plain run.
- *Property tests* (Hypothesis, `tests/unit/test_properties.py`):
  - the loop tracker resumes correctly from every safe point of random loop nests;
  - LocalFS and Google Drive storage behave like a dict model;
  - key validation;
  - state snapshot round trip;
  - duration parsing;
  - storage layout.
- *Fault injection* (`tests/integration/test_faults.py` with
  `faulty_site/sitecustomize.py`):
  - every storage call inside the sessions fails transiently at random while sessions are
    also killed;
  - separately, the latest checkpoint is corrupted between sessions, and the next session
    must fall back to the previous one.

  Both jobs must still finish identically.
- *Path-safety tests* (`tests/unit/test_safe_paths.py`): `..`, absolute paths, drive letters
  and backslashes in bundle, file-manifest and output names are all rejected.
- *Stress* (`tests/integration/test_stress.py`):
  - twelve jobs of four kinds on three Simulated backends, one of which kills at random;
  - the controller is restarted twice;
  - every job must finish with outputs equal to an uninterrupted run, and no job may ever
    have two live main segments.
- *Linux run* (WSL, Python 3.12): the full suite also runs on the Linux code path
  (`PyFrame_LocalsToFast`), the same one Kaggle and Colab use.

**Bugs found and fixed**

| Found by | Bug | Fix |
|---|---|---|
| adversarial | A `for` over a list that grows inside the loop stopped at the original length after a resume | Mutable sequences are iterated by index, re-reading the length |
| adversarial | State held in a closure (counter in a nested function) restarted after a resume | Stateful `__main__` closures are snapshotted and their cells refilled in place |
| adversarial | Objects changed only by method calls (`stats.record(x)`) were not snapshotted | Method receivers that are user objects are snapshotted; more mutating method names are recognised |
| adversarial | Very tight loops slowed down a lot from per-iteration tracking | After a probe, fast loops offer a safe point only every *stride* iterations (about every 50 ms) |
| Hypothesis | Storage prefix ending in a partial `.` or `..` component was rejected | Only full path components are checked |
| Hypothesis | LocalFS `delete` on a directory raised an error | It is a no-op, as on the Hub |
| Linux run | Notebooks without matplotlib installed failed: figure capture imported it unconditionally | Figure capture only uses matplotlib if the job already imported it |
| path tests | A crafted bundle or manifest could write outside the work or output directory (`../x`) | `safe_join` rejects such paths everywhere files are written |
| review | The Kaggle bootstrap pasted ids into source as raw strings | Values are inserted with `repr()` |
| review | The dashboard accepted any `Host` header (DNS rebinding) | Requests with a non-local `Host` get 421; the bind host is allowed explicitly |
| review | `gpurelay up` would use a public Hub repo | `up` refuses a public store, and a missing repo gets a clear error |
| review | Wrong file types, syntax errors, corrupt or missing archives, and a missing `--gdrive` file gave raw tracebacks | Each one gives a one-line error |
| stress | `relay.load_results()` failed with "no relay.chunks() loop has run yet" when a resume skipped the (long) chunks loop statement | The last and named result buckets are kept in checkpoint metadata. Regression: `test_load_results_after_a_resume_past_the_chunks_loop` (plain and parallel) |
| fault injection | One transient storage error while a session started (bundle download, checkpoint pointer read, worker assignment) ended the session as an infrastructure failure; a few in a row failed the job | Every setup step is retried with backoff. Regression: `tests/unit/test_entry_setup.py` (every storage call fails once) |

**Test tuning:** timing-sensitive tests now use deadline handoffs or larger windows so they
pass on both a loaded Windows laptop and Linux. Each one still fails without the feature it
tests. The Keras test now runs 24 epochs (was 16): with a warm start the job fit in one 16 s
session and never handed off.

**Note:** on Windows, paths longer than 260 characters break checkpoint persists. Normal
session paths are about 130 characters; only very deep custom work directories hit this. It
cannot happen on Kaggle or Colab (Linux).

**Verification (2026-10-23):**
- Windows:
  - 378 unit tests pass;
  - all 13 integration files pass (55 tests), one file at a time;
  - the stress test passed 4/4 in a row after the chunks fix (it failed about 1 run in 3
    before);
  - the fault tests passed 5/5 after the setup-retry fix.
- Linux (WSL, Python 3.12), with every fix in place:
  - 374 unit tests pass;
  - all 13 integration files pass (39 passed);
  - 3 unit and 13 integration tests are skipped because torch, Keras, HF and Lightning
    are not installed there (they pass on Windows).
