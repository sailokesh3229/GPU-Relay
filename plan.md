# GPU Relay — Implementation Plan

Companion to `spec.md`. The spec says **what** to build; this file says **in what order** and **how each step is verified**.

## How to use this plan (instructions for Claude Code)

1. Read `spec.md` fully before starting. Section 3 (hard rules) overrides any later instruction that conflicts with it.
2. Work **one phase at a time**. At the end of each phase: all tests green, `ruff` and `mypy` clean, a short `docs/progress.md` entry (what was built, what is deferred, open `VERIFY` items). Then stop and summarise for the owner before starting the next phase.
3. **Test-first for runtime and checkpoint code.** Every resumption feature ships with kill-and-resume equivalence tests on the corpus (spec §18).
4. Never call real external services in automated tests. Kaggle, HF Hub, and Colab are exercised only through fakes/mocks in CI, and through documented manual smoke tests.
5. When an external platform's behavior is uncertain, code against the interface, make it configurable, and leave a `# VERIFY:` comment plus an entry in `docs/verify.md`. Do not invent API fields.
6. Keep optional ML libraries (torch, tensorflow, transformers, lightning) as lazy imports. The core must install and pass non-ML tests without them; ML tests are marked (`@pytest.mark.torch`, etc.) and skipped when the library is absent.
7. Prefer small, readable modules over cleverness. Public functions get docstrings and type hints.

Suggested first prompt to Claude Code:
> Read spec.md and plan.md. Implement Phase 0 only, run the tests, update docs/progress.md, and stop with a summary.

---

## Phase 0 — Skeleton and foundations

**Build**
- `pyproject.toml` (src layout, optional extras: `torch`, `tf`, `hf`, `lightning`, `kaggle`, `dev`), `ruff`, `mypy`, `pytest` config, CI workflow (lint + unit tests).
- `config.py` (pydantic models + TOML load/save at `~/.gpurelay/config.toml`, overridable by `GPURELAY_HOME`), `models.py` (Job, Segment, Checkpoint, BackendCaps, enums for states and exit codes), `constants.py`, `errors.py`.
- `storage/base.py` + `storage/localfs.py` with atomic small-file writes.
- `cli.py` with `typer`: all commands from spec §15 registered; unimplemented ones print "not yet implemented (phase N)".
- `docs/progress.md`, `docs/verify.md`.

**Verify**
- Unit tests: config round-trip; LocalFS put/get/list/delete/exists; atomic write (reader never sees a partial file); CLI `--help` for every command.

**Done when:** `pip install -e .[dev]` works on a clean venv without torch; tests and linters pass.

---

## Phase 1 — Checkpoint core

**Build**
- `runtime/rng.py`: capture/restore Python, NumPy, torch CPU + all CUDA devices, TF (lazy).
- `runtime/state.py`: classify variables (torch-aware via `state_dict`, plain via cloudpickle, skip list), capture to a directory, restore with **load-into-live-object** rule (spec §7.5), report unpicklable names.
- `runtime/checkpoint.py` + `runtime/persist.py`: two-phase checkpoint (snapshot to local tmp → background persist with zstd → checksum verify → commit `LATEST.json`), retention of last 2, single persist at a time, stall timing.
- Checkpoint loader: read `LATEST.json`, verify checksums, fall back to previous checkpoint on mismatch.

**Verify**
- Round-trip tests: model + optimizer + scheduler + GradScaler restored into fresh live objects, then training continues identically to an uninterrupted run (CPU, deterministic).
- Optimizer ↔ parameter linkage preserved after restore (step changes the live model).
- RNG round-trip: same random draws after restore.
- Fault tests: kill (simulate via exception/thread abort) during snapshot and during persist → `LATEST.json` still points to the previous valid checkpoint; corrupted file → fallback; storage write failures → retry with backoff.

**Done when:** all above pass; documented checkpoint layout matches spec §8.3.

---

## Phase 2 — Script runtime with SDK, output capture, watchdog, Simulated backend

**Build**
- `sdk.py`: `relay.state`, `relay.range`, `relay.chunks`, `relay.save_result`, `relay.load_results`, `relay.safe_point`; no-op passthrough outside GPU Relay.
- `runtime/capture.py`: stdout/stderr tee with byte offsets; matplotlib `show` patch saving figures; offsets recorded per checkpoint.
- `runtime/watchdog.py`: deadline, forced-checkpoint condition (spec §8.2), SIGTERM handling.
- Adaptive checkpoint interval (spec §8.2) with measured `t_snap`, size, throughput.
- `runtime/status.py`: `status.json` heartbeat.
- `runtime/executor_script.py` + `runtime/entry.py`: run a script with the SDK active; exit codes per spec §12.3; user-code exceptions → `failed_user` with original tracebacks.
- `backends/base.py`, `backends/local.py`, `backends/simulated.py` (hard kill at time T / random, short deadlines, labels for fake GPU models).
- `controller/assemble.py` (output assembly only, used directly by tests): log truncation at checkpoint offsets + concatenation, figure merge, files export, `report.json`.
- `gpurelay run-local` with `--simulate-kill-every` / `--simulate-deadline` (single job, loops segments locally until done).

**Verify**
- Equivalence harness (`tests/integration/equivalence.py`) introduced here: uninterrupted vs (random kills | forced handoffs | both).
- Corpus items written with the SDK (a copy of corpus 1 and 7 using `relay.*`) pass equivalence: identical weights, metric lists, merged log text, figure count.
- Lost work per hard kill ≤ configured `max_lost_work_s` in tests (use small values).

**Done when:** `gpurelay run-local tests/corpus/sdk_mlp.py --simulate-kill-every 20s` finishes with outputs identical to the uninterrupted run.

---

## Phase 3 — Automatic instrumentation for scripts + framework hooks + analyzer v1

**Build**
- `runtime/instrument.py`: AST transform of the outermost `for` loop in long top-level blocks → `__relay__.loop(loop_id, iter)`; mutated-variable detection; compile transformed AST directly (line numbers preserved).
- `runtime/loops.py`: iteration counting, safe points, fast-forward on resume (slice for range/sequences; consume for iterators, logged).
- `runtime/hooks/hf.py`, `lightning.py`, `keras.py` (version-checked; degrade with warning).
- `analyzer/static.py`, `analyzer/report.py`, `analyzer/suggestions.py`: resumability levels, risks, GPU-time suggestions (spec §14); `gpurelay analyze`.

**Verify**
- **Unmodified** corpus 1, 3, 4, 5, 6, 7 pass equivalence under the Simulated backend.
- Corpus 8 (`while` loop) is reported `PARTIAL` and still completes correctly (restart of that block).
- Traceback line numbers in instrumented code match the original file (test with a deliberate error).
- Analyzer snapshot tests on each corpus item (expected `analysis.json`).

**Done when:** the owner's miniaturized VAE + chunked-FID corpus item survives random kills with identical outputs, with zero code changes.

---

## Phase 4 — Notebook support

**Build**
- `runtime/executor_notebook.py`: execute cells via IPython `InteractiveShell.run_cell` in one namespace; record per-cell status, duration, outputs (stream, display data, errors); magics supported.
- Replay-and-restore on resume (spec §7.1): replay short completed cells, restore snapshot, skip long completed cells, resume interrupted cell via Phase 3 instrumentation (cell-level AST).
- Side-effect detection in analyzer for cells (downloads, deletes, external writes) → never replayed automatically.
- Final `output.ipynb` assembly with `nbformat`, preserving outputs of skipped cells.

**Verify**
- Corpus 2 (notebook MLP) and a notebook version of corpus 3 pass equivalence; final notebook outputs equal the uninterrupted run's (ignoring execution counts/timestamps).
- Interrupted inside a long training cell → resumes inside the loop, not from the cell start.

**Done when:** notebooks and scripts have feature parity for the corpus.

---

## Phase 5 — Controller, scheduler, quota ledger, estimator

**Build**
- `controller/db.py` (SQLite schema from spec §13.1, migrations via simple version table).
- `bundle/packager.py` + `manifest.py`: bundle with code, extra files, requirements, data specs, pinned runtime wheel (built from the current package), content hash.
- `controller/quota.py`: budget models (`weekly_rolling`, `monthly`, `unknown_opportunistic`), usage records, user corrections, next reset time.
- `controller/estimator.py`: rate from `status.json`, remaining GPU-hours, "fits this week / spans N weeks" message.
- `controller/scheduler.py`: policy from spec §13.4 including parallel jobs, `waiting_for_quota`, `awaiting_user`, retries for `failed_infra`, no retries for `failed_user`.
- `controller/daemon.py` (`gpurelay up`): poll loop, crash detection via stale heartbeat + backend state, segment bookkeeping, assembly on completion.
- `controller/notify.py`: desktop + optional ntfy.
- CLI: `submit`, `status` (with remaining budget per backend always shown), `logs`, `quota`, `pause`, `resume`, `cancel`, `fetch`.

**Verify**
- Scheduler unit tests with fake clock: budget exhaustion → wait → reset → resume; preference ordering; concurrency limits; one job never has two live segments.
- End-to-end with 2–3 Simulated backends (different budgets and fake GPU labels): three jobs submitted, run in parallel, survive kills and handoffs, all complete with equivalent outputs; ledger totals match segment durations.
- `status` output snapshot test shows per-backend remaining budget and job ETA.

**Done when:** a full multi-job, multi-backend relay works locally end to end.

---

## Phase 6 — Real storage and Kaggle backend

**Build**
- `storage/hf_hub.py`: private repo, batched commits, retries/backoff, used-space reporting, pruning.
- `backends/kaggle.py` + templates (`kernel-metadata.json.j2`, `kaggle_bootstrap.py.j2`): push/status/output via official package; `is_private: true`; secrets via private dataset; preflight duplicate-run guard; requirements policy (don't reinstall preinstalled CUDA frameworks unless pinned).
- `gpurelay init`: create/verify HF repo, verify Kaggle credentials, guide creation of the private secrets dataset; store credentials in keyring.
- `docs/manual-tests.md`: step-by-step Kaggle smoke test.

**Verify**
- Unit tests with mocked `kaggle` and `huggingface_hub` clients (generated metadata, bootstrap content contains no secrets, status mapping, duplicate guard).
- Manual smoke test (owner runs it): tiny corpus job on Kaggle with `max_session_s` forced to ~10 min → at least one real handoff to a Local or second Kaggle segment (sequential, same single account) → identical outputs. Record findings and resolve `VERIFY` items in `docs/verify.md`.

**Done when:** the owner has completed the manual Kaggle smoke test successfully.

---

## Phase 7 — Colab manual backend

**Build**
- `backends/colab_manual.py` + `colab_resume.ipynb.j2`: one resume notebook per job; token from Colab user secrets; installs runtime wheel from bundle; executes the job in the notebook kernel; prints progress.
- Scheduler integration: `awaiting_user` + notification with clear instructions; `gpurelay colab <job>`.
- Compliance check (test): the Colab backend module contains no browser automation, SSH, tunnel, or remote-execution code paths; `automated = False`.

**Verify**
- Unit tests for notebook generation (valid nbformat, no secrets embedded).
- Manual smoke test: owner opens the generated notebook in Colab, runs it, job continues from the Kaggle/Local checkpoint and finishes with equivalent outputs.

**Done when:** a job relays Kaggle → Colab (manual) → completes.

---

## Phase 8 — Polish: dashboard, suggestions, robustness

**Build**
- `gpurelay dashboard`: local web UI (FastAPI + a single static page, no build step): jobs, segments timeline, progress, ETA, budget per backend, live log tail, "Colab action needed" banner.
- Analyzer `--probe` (quick dry-run for rate and memory estimate).
- Hardening: storage outage handling, controller restart recovery from SQLite, clock skew tolerance for heartbeats, clearer error messages.
- README with quick start, hard rules, and limitations (numeric equivalence across GPU models, uninstrumented loop types).

**Verify**
- API tests for dashboard endpoints; controller restart test mid-job; full corpus regression run.

**Done when:** a new user can follow the README from install to a completed relayed job.

---

## Phase 9 — Stretch (only after Phase 8, one at a time, each with its own tests)

1. Chunk-parallel execution across backends for `relay.chunks(..., parallel=True)`; merge results in order.
2. Nested-loop safe points and mid-epoch DataLoader resume.
3. Instrumentation of `while` loops and loops inside functions (opt-in).
4. Lightning AI backend — **only after** its ToS and official SDK are verified and recorded in `docs/verify.md`.
5. Google Drive storage.
6. Always-on controller on a small free CPU VM.

---

## Multi-user website (phases M1-M5, one at a time; the owner verifies each)

Replaces the single owner login. Anyone can sign up (up to a limit), connect their own
accounts, and relay their own jobs; nobody sees anyone else's jobs or tokens.

Design. The Kaggle, Modal and Hugging Face libraries read credentials for the whole process,
so users never share a process: a **gateway** (accounts, token vault, setup checklist) runs
one **per-user child** (that user's web app + controller, with only that user's tokens in its
environment, its home and state backup in that user's own storage repo) and forwards the
user's requests to it. The server never runs submitted code (only the static analyzer).

- **M1 Accounts.** Users in Postgres (`DATABASE_URL`; SQLite for tests and local runs).
  Email + password (scrypt), signed session cookies bound to a per-user epoch (password
  change or sign-out-everywhere invalidates them), rate limits, sign-up limit
  (`GPURELAY_MAX_USERS`, default 3) and optional invite code. Sign-up, sign-in, sign-out,
  account page with the setup checklist. Owner/viewer passwords and the public view go.
- **M2 Token vault + web setup.** Each user's tokens encrypted at rest (key only in the
  server's environment), checked against the platform before saving, never shown again;
  replace or delete any time. Setup wizard on the website (HF, Kaggle, Modal, Colab, limits).
- **M3 Per-user controllers.** The gateway starts a user's child on sign-in or while their
  jobs run, forwards `/api/*` to it, stops it when idle (memory: 2-3 active users on the free
  plan), restores and backs up its state in the user's own repo.
- **M4 Account care.** Email verification and password reset (SMTP settings), delete account
  (tokens, sessions, children; the user's repo stays theirs), per-user job limits.
- **M5 Deploy.** Postgres on a free host, Render settings, moving the current owner's jobs
  into their account, README and docs.

---

## Phase checklist summary

| Phase | Outcome | Key proof |
|---|---|---|
| 0 | Skeleton, config, LocalFS, CLI | clean install without torch |
| 1 | Safe two-phase checkpoints | kill-during-persist keeps last valid checkpoint |
| 2 | SDK runtime + Simulated backend | SDK scripts identical after random kills |
| 3 | Zero-change scripts + hooks + analyzer | unmodified VAE+FID corpus identical after kills |
| 4 | Notebooks | notebook outputs identical after kills |
| 5 | Controller, budget, parallel jobs | 3 jobs × 3 simulated backends complete |
| 6 | HF storage + Kaggle | real Kaggle handoff smoke test |
| 7 | Colab (manual) | Kaggle → Colab relay completes |
| 8 | Dashboard + hardening | README quick start works end to end |
| 9 | Stretch features | per-feature tests |
| M1-M5 | Multi-user website | two users never see each other's jobs or tokens |
