# GPU Relay — Product Specification

> Run any ML workload (training, metric computation, evaluation, preprocessing) to completion on **free GPU tiers**, by checkpointing automatically and relaying the job across platforms when a session or quota limit is reached — with outputs identical (or numerically equivalent) to an uninterrupted run.

Status: v1 design. Audience: the implementer (Claude Code) and the project owner. Read this whole file before writing code, then follow `plan.md` phase by phase.

---

## 1. Problem

Free GPU platforms (Kaggle, Google Colab, Lightning AI, …) give limited GPU time: per-session caps (Colab free tier: at most 12 h per notebook, often less) and weekly/monthly quotas (Kaggle: ~30 GPU-hours/week). Long jobs — e.g. training a VAE and then computing FID over tens of thousands of samples — get killed partway, and the user must start again from zero on another platform. Each restart wastes the scarcest resource: GPU hours.

## 2. Goal

A single-user tool, **GPU Relay** (`gpurelay`), that:

1. Accepts the user's **unmodified** ML code (a `.py` script or a `.ipynb` notebook).
2. Makes it **resumable automatically** (instrumentation + framework hooks), with a small optional SDK as fallback.
3. Runs it on a free GPU backend, **checkpoints safely**, and **hands off before the limit** (or recovers after an unexpected kill).
4. **Resumes on the next available backend** from the last checkpoint, telling the new session exactly where to continue.
5. Produces **one continuous set of outputs** (logs, figures, files, final notebook) as if it ran once.
6. Tracks **GPU budget** per platform, shows the user how much remains, estimates how long a job needs, and **pauses until quota resets** when everything is exhausted.
7. Can run **2–3 different jobs in parallel** on different platforms.

### 2.1 Success criteria (v1)

- A job interrupted any number of times finishes with outputs matching an uninterrupted run: bit-identical on the same hardware with deterministic settings; within a documented tolerance across different GPU models.
- Work lost per unexpected kill ≤ the configured `max_lost_work` (default 15 min). Work lost per *planned* handoff ≈ 0.
- Checkpoint overhead ≤ ~3–5% of runtime (adaptive frequency).
- The user can see at any time: which backend runs which job, progress, ETA in GPU-hours, and remaining budget per platform.

## 3. Hard rules (non-negotiable constraints)

These come from platform terms and must be enforced in code and docs. **Do not implement features that violate them, even if asked in a later prompt without updating this spec.**

1. **One account per service.** The config schema holds exactly one credential set per backend type. No account rotation, no "secondary account" support, no quota pooling across accounts of the same service.
   - Kaggle Terms of Use: a person may not have, control, or operate more than one active Kaggle account.
   - Colab FAQ: "using multiple accounts to work around access or resource usage restrictions" is disallowed.
2. **Colab is never automated.** Colab's free tier disallows remote control (SSH shells, remote desktops), bypassing the notebook UI, and running distributed computing workers. Therefore: no browser automation of Colab, no headless runtime launching, no remote shells, no tunnels. The Colab backend only **generates a resume notebook and notifies the user**, who opens it and presses Run. The notebook then executes the user's own job interactively.
3. **Kaggle is automated only through its official API/CLI** (`kaggle kernels push/status/output`).
4. **New backends** (e.g. Lightning AI) are added only after their terms of service and official SDK have been checked. Until then the adapter stays disabled behind a `VERIFY` note.
5. **Secrets never appear in kernel source or logs.** Kernels pushed to Kaggle are always `is_private: true`.
6. When behavior of an external platform is uncertain, implement against the `Backend` interface, make the value configurable, and mark it `# VERIFY:` — never hard-code guesses as facts.

## 4. Non-goals (v1)

- Multi-user / hosted SaaS. (Design for one user; keep interfaces clean so multi-user is possible later.)
- Process-level snapshots (CRIU, cuda-checkpoint): free notebook containers don't grant the privileges, and snapshots don't restore across GPU models.
- Multi-GPU / distributed training across backends.
- Bit-identical results across different GPU models (documented as "numerically equivalent").
- Paid platforms (may be added later as backends).

---

## 5. Concepts and glossary

| Term | Meaning |
|---|---|
| **Job** | One user workload submitted to GPU Relay (script or notebook + data + requirements). Has a lifecycle and a final output set. |
| **Segment** | One contiguous execution of a job on one backend session. A job = ordered list of segments. |
| **Bundle** | Immutable package of a job: user code, extra files, requirements, data references, manifest, and the pinned `gpurelay` runtime wheel. Stored in storage. |
| **Safe point** | A moment where state can be captured consistently: notebook cell boundary, end of an instrumented loop iteration, framework checkpoint callback, or an SDK call. Checkpoints happen **only** at safe points. |
| **Checkpoint** | Durable, atomic snapshot of job state at a safe point (model/optimizer/etc. state, selected variables, RNG states, loop positions, output-log offsets, working files). |
| **Handoff** | Planned stop: the runtime checkpoints at the next safe point before a deadline and exits with code `HANDOFF`. |
| **Crash** | Unplanned stop (killed, disconnected). Recovery resumes from the last committed checkpoint. |
| **Backend** | Adapter for a compute platform (Kaggle, Colab-manual, Local, Simulated, later Lightning). |
| **Storage** | Durable shared store reachable from every backend and the controller (default: private Hugging Face Hub repo; LocalFS for tests). |
| **Controller** | Long-running process on the user's machine: scheduler, quota ledger, job state DB, backend polling, notifications. |
| **Runtime** | Python package code that runs **inside** the backend session: executes user code, instruments it, checkpoints, captures outputs, watches the deadline. |
| **Resumability level** | `FULL`, `PARTIAL`, or `RESTART_ONLY` — reported by the analyzer before launch. |

---

## 6. Architecture

```
 User's machine                                         Free GPU backends
┌──────────────────────────────────────┐            ┌──────────────────────────┐
│ CLI (gpurelay ...)                   │            │ Kaggle kernel (auto)     │
│ Analyzer  ── resumability report     │            │   bootstrap.py           │
│ Bundler   ── job bundle              │  push /    │   └─ gpurelay.runtime    │
│ Controller (daemon)                  │  status    │ Colab notebook (manual)  │
│   Scheduler · Quota ledger ·         │◄─────────► │   └─ gpurelay.runtime    │
│   Estimator · State DB (SQLite) ·    │            │ Local / Simulated (tests)│
│   Notifier · Dashboard (later)       │            └───────────┬──────────────┘
└──────────────────┬───────────────────┘                        │
                   │        ┌─────────────────────────────┐     │
                   └───────►│ Storage (HF Hub private repo)│◄───┘
                            │ bundles · checkpoints ·      │
                            │ status.json · logs · outputs │
                            └─────────────────────────────┘
```

Data flow for one job:

1. `gpurelay analyze` → resumability report + GPU-time suggestions.
2. `gpurelay submit` → bundle uploaded to storage; job queued in SQLite.
3. Scheduler picks a backend with budget → backend adapter launches a segment.
4. Runtime in the session: downloads bundle + latest checkpoint → restores → executes → checkpoints at safe points (snapshot, then background persist) → heartbeats `status.json`.
5. Near the session deadline: watchdog requests handoff → checkpoint at next safe point → exit `HANDOFF`.
6. Controller sees `HANDOFF` (or crash) → records segment, updates ledger → schedules the next segment elsewhere (or waits for quota).
7. On `COMPLETED`: controller assembles final outputs (merged log, output notebook, figures, files, metrics) → notifies user → `gpurelay fetch`.

---

## 7. Resumption strategy (layered)

The runtime tries layers in this order and the analyzer reports which layer covers each part of the code.

### 7.1 Layer A — Notebook cells: replay-and-restore

For `.ipynb` jobs, the runtime executes cells itself (IPython `InteractiveShell.run_cell` in one shared namespace, magics supported) and records per cell: status, duration, outputs.

On resume:
1. **Replay** every completed cell whose recorded duration < `replay_threshold_s` (default 30 s) and that the analyzer did not flag as side-effectful (e.g. downloads, file deletions). This recreates imports, classes, functions, model/optimizer objects, data loaders.
2. **Restore** the saved variable snapshot on top (see 7.5 restore order).
3. **Skip** completed long cells; their effects are in the restored state and restored working files.
4. Resume the interrupted cell using Layer B/C if it was instrumented; otherwise re-run it from its start (reported as `PARTIAL`).
5. Saved outputs of skipped cells are placed into the final output notebook unchanged.

### 7.2 Layer B — Framework hooks (zero user changes)

Detected by imports; implemented by patching entry points at runtime:

| Framework | Hook |
|---|---|
| Hugging Face `transformers.Trainer` | Redirect `output_dir` checkpoints into the relay checkpoint area; set save strategy by time; on resume pass `resume_from_checkpoint`. |
| PyTorch Lightning `Trainer.fit` | Add/configure `ModelCheckpoint(save_last=True)` into relay area; on resume pass `ckpt_path`. |
| Keras `Model.fit` | Append `tf.keras.callbacks.BackupAndRestore(backup_dir=relay area)`. |

Each framework's own checkpoint becomes part of the relay checkpoint (files are included in the persist step). Hook implementations must check installed library versions and degrade to "not covered" with a warning on unknown versions.

### 7.3 Layer C — Automatic loop instrumentation (AST)

For scripts and for long notebook cells:

- Parse with `ast`; transform; **compile the transformed AST directly** (never `unparse` + re-parse) so tracebacks keep the user's original line numbers.
- **v1 scope:** the outermost `for` loop of each long-running top-level block (script top level or cell top level). Nested-loop safe points are a later phase.
- Transform `for x in ITER:` → `for x in __relay__.loop("<loop_id>", ITER):` where `loop_id` is a stable hash of file + line + source of the loop header.
- `__relay__.loop` yields items, counts completed iterations, offers a **safe point at the end of each iteration**, and on resume **fast-forwards**: for `range`/`enumerate(range)`/sequences it slices; for general iterators it consumes and discards items (data loading cost only; logged).
- **Variables to snapshot** at that safe point: names assigned or mutated inside the loop body (assignments, augmented assignments, `.append/.extend/.update/...` calls, subscript assignment), plus all torch/tf stateful objects in the namespace (modules, optimizers, schedulers, GradScaler, EMA objects).
- `while` loops, loops inside functions, generators, and comprehensions are **not** instrumented in v1 → reported by the analyzer.

### 7.4 Layer D — Manual SDK (fallback, explicit)

For code the automatic layers cannot cover:

```python
import gpurelay as relay

state = relay.state(model=model, optimizer=opt, scheduler=sched, history=history)
for epoch in relay.range(100):          # resumable range
    ...
    relay.safe_point()                   # optional extra safe point

for chunk in relay.chunks(items, size=1000):   # resumable chunked work (FID, eval, preprocessing)
    relay.save_result(chunk.id, compute(chunk)) # partial results survive cutoffs
results = relay.load_results()                  # all chunks, in order
```

The SDK must be a no-op passthrough when run outside GPU Relay (plain `python script.py` still works).

### 7.5 State capture and restore rules

- **Torch-aware:** `nn.Module`, `Optimizer`, LR schedulers, `GradScaler`, and objects exposing `state_dict()/load_state_dict()` are saved via `state_dict` with `torch.save`. On restore they are **loaded into the already-replayed live objects** (preserves optimizer ↔ parameter references). If no live object exists, fall back to unpickling the whole object.
- **Other variables:** `cloudpickle`. Unpicklable values (open files, sockets, live generators, CUDA streams, locks) are skipped with a warning naming them; the analyzer predicts these up front.
- **Never snapshot:** modules, functions/classes defined in user code (recreated by replay), IPython internals, `__relay__` itself.
- **RNG:** Python `random`, NumPy global RNG, `torch` CPU, all CUDA devices, and TensorFlow global seed state where available. Captured at every checkpoint, restored after state restore.
- **DataLoader position:** for mid-epoch resume (later phase) record the number of batches consumed and the sampler's generator state.
- **Working files:** files written by user code under the job working directory (`./` inside the session) are tracked (mtime/size/hash) and included in the checkpoint if changed; restored before replay.

### 7.6 Fallback

If nothing covers a long block: `RESTART_ONLY` for that block. The analyzer says so before launch and the user must confirm (`--accept-restart-only`).

---

## 8. Checkpointing

### 8.1 Two-phase checkpoint

1. **Snapshot (blocking, short):** at a safe point, copy GPU tensors to CPU memory, serialize to the session's local disk under `ckpt-<seq>.tmp/`. Training pauses only for this.
2. **Persist (background thread):** compress (zstd, level configurable, default 3; full precision always for the latest checkpoint) and upload to storage under `jobs/<job_id>/checkpoints/ckpt-<seq>/`, then verify checksums.
3. **Commit:** only after all files are uploaded and verified, write `jobs/<job_id>/checkpoints/LATEST.json` → `{seq, files, sha256s, safe_point_id, created_at, segment_id}`. A checkpoint does not exist until `LATEST.json` points to it.
4. **Retention:** keep the last 2 committed checkpoints; delete older ones after a new commit.
5. If a new snapshot is requested while a persist is still running, wait for it (never two persists at once); record the stall time.

Reasoning: this follows the snapshot/persist split from CheckFreq (FAST'21) and PyTorch's asynchronous checkpointing. Incremental/differential checkpoints and checkpoint quantization (Check-N-Run) are **not** used in v1: dense models change every weight every step, and quantized resume points would break output equivalence.

### 8.2 Checkpoint frequency (adaptive)

Inputs measured at runtime: snapshot time `t_snap`, persist throughput `bw`, checkpoint size `S`, time between safe points.

- `overhead_interval = t_snap / overhead_budget` (default budget 0.03).
- `interval = clamp(overhead_interval, min_interval_s=120, max_lost_work_s=900)`.
- Checkpoint at the first safe point after `interval` has elapsed since the last commit.
- **Forced checkpoint** when `time_to_deadline < t_snap + S/bw + safety_margin` (default margin 300 s) — this triggers handoff.
- If safe points are rarer than `max_lost_work_s` (e.g. one 2-hour epoch), the analyzer and runtime warn and suggest the SDK or nested-loop instrumentation.

### 8.3 Checkpoint contents

```
ckpt-<seq>/
  meta.json          # seq, job_id, segment_id, safe_point_id, cell/loop positions, iteration counts,
                     # output offsets (log bytes, figure count, notebook cell index), env fingerprint
  rng.pt             # all RNG states
  torch_state.pt     # state_dicts of torch-aware objects, keyed by variable name
  vars.pkl.zst       # cloudpickled plain variables
  framework/         # HF / Lightning / Keras native checkpoint files, if any
  files.tar.zst      # changed working files
  outputs/           # log segment, figures, cell outputs up to this safe point
```

---

## 9. Output continuity

- **stdout/stderr:** tee'd to the session console and to `outputs/log-<segment>.txt`. At each checkpoint record the byte offset. On final assembly, each segment's log is **truncated at its last committed checkpoint offset** (work after it was redone by the next segment), then concatenated. Optional marker lines `[gpurelay] resumed on <backend> at <safe point>` can be enabled (default off to keep output identical).
- **Figures:** patch `matplotlib.pyplot.show` and IPython display hooks to save each figure as `figures/<seq>.png` (and keep it in notebook outputs). Same truncation rule by figure count.
- **Notebook output:** final `.ipynb` built with `nbformat` from per-cell outputs across segments.
- **Files:** final working-directory files are exported to `jobs/<job_id>/final/files/`.
- **Equivalence report:** the final bundle includes `report.json`: segments, backends, GPU models, library versions per segment, checkpoints, lost work, total GPU-hours.

---

## 10. Backends

### 10.1 Interface

```python
class Backend(Protocol):
    name: str                       # "kaggle", "colab_manual", "local", "simulated", ...
    automated: bool                 # False for colab_manual
    def capabilities(self) -> BackendCaps: ...   # gpu types, max_session_s, max_concurrent, needs_user_action
    def check_ready(self) -> ReadyStatus: ...    # credentials present, API reachable
    def launch(self, job: Job, segment: Segment, deadline_s: int) -> LaunchHandle: ...
    def poll(self, handle: LaunchHandle) -> BackendRunState: ...  # queued|running|finished|error|unknown
    def cancel(self, handle: LaunchHandle) -> None: ...
    def fetch_console_log(self, handle: LaunchHandle) -> str | None: ...
```

The **source of truth for job progress is `status.json` in storage** (written by the runtime), not the backend's own status. Backend status is used to detect crashes and finished sessions.

### 10.2 Local backend
Runs the runtime in a subprocess on the user's machine (CPU or local GPU). Used for development and real local runs.

### 10.3 Simulated backend (testing)
Like Local, plus fault injection: hard kill (`SIGKILL`) at a given time or random time; graceful deadline (short `max_session_s` to force handoffs); simulated quotas; simulated different "GPU model" labels. Several simulated backends can run concurrently to test parallel scheduling. This is the backbone of the test suite.

### 10.4 Kaggle backend (automated)
- Uses the official `kaggle` package/CLI: generate a per-segment kernel directory with `kernel-metadata.json` (`is_private: true`, `enable_gpu: true`, `enable_internet: true`, `dataset_sources` for data + the private secrets dataset) and a generated `bootstrap.py`; `kaggle kernels push`; poll with `kaggle kernels status`; fetch console output with `kaggle kernels output`.
- **Secrets:** Kaggle UI secrets may not be available to CLI-pushed kernels (`# VERIFY:`). Default approach: a **private Kaggle dataset** owned by the user containing a fine-grained storage token (write access to the one relay repo only). Never embed tokens in `bootstrap.py`.
- `bootstrap.py`: download bundle from storage → `pip install` the pinned runtime wheel from the bundle → `python -m gpurelay.runtime.entry --job <id> --segment <id> --backend kaggle --deadline <epoch_s>`.
- Do not rely on `/kaggle/working` for checkpoints: a session killed at its time limit may not save outputs (`# VERIFY:`). All durable state goes to storage during the run.
- Config values (all `# VERIFY:` and user-editable): `max_session_s` (default 12 h), `weekly_gpu_hours` (default 30), `max_concurrent` (default 1), accelerator choice (`gpu_p100` / `gpu_t4x2`), quota reset model (rolling weekly).
- Preflight: refuse to push if a relay kernel for this job is already `running`/`queued` (avoids duplicate sessions wasting quota).

### 10.5 Colab backend (manual, compliant)
- Generates `resume_<job_id>.ipynb` (once per job; it always resumes from `LATEST.json`, so the same notebook works for every segment). Cells: mount nothing by default; read the storage token from Colab's user secrets (`google.colab.userdata`); install runtime wheel from the bundle; run the job **in the notebook kernel** via the runtime's notebook/script executor; print progress.
- The controller **notifies** the user ("Kaggle budget exhausted; job X can continue on Colab — open resume_X.ipynb and press Run All") and marks the segment `awaiting_user`.
- Deadline: Colab limits are unpublished and fluctuate; use conservative `max_session_s` (default 10 h, `# VERIFY:`) for handoff and rely on periodic checkpoints for surprise disconnects.
- No automation of any kind beyond generating the notebook and detecting progress via `status.json`.

### 10.6 Lightning AI backend (later, gated)
Disabled until ToS and official SDK are verified (hard rule 4). Expected budget per third-party guides: ~80 GPU-h/month (`# VERIFY:`).

---

## 11. Storage

### 11.1 Interface
```python
class Storage(Protocol):
    def put_file(self, local: Path, remote: str) -> None
    def get_file(self, remote: str, local: Path) -> None
    def put_bytes(self, data: bytes, remote: str) -> None
    def get_bytes(self, remote: str) -> bytes | None
    def list(self, prefix: str) -> list[str]
    def delete(self, remote: str) -> None
    def exists(self, remote: str) -> bool
```
`put` must be atomic from the reader's perspective for small JSON files (write temp name, then rename/commit).

### 11.2 Implementations
- **LocalFS** — tests and local-only use.
- **Hugging Face Hub (default)** — one private repo (e.g. `<user>/gpurelay-store`) via `huggingface_hub`; batch uploads with commits; large files via the Hub's large-file handling. Storage limits and rate limits are `# VERIFY:`; the controller shows used space and prunes old checkpoints.
- **Google Drive** — later.

### 11.3 Layout
```
jobs/<job_id>/
  bundle/            bundle.tar.zst, manifest.json, runtime wheel
  checkpoints/       ckpt-<seq>/..., LATEST.json
  status.json        heartbeat + state (see 12.2)
  segments/<seg_id>/ console.txt, env.json
  final/             merged log, output.ipynb, figures/, files/, report.json
data-cache/<hash>/   staged datasets (optional)
```

---

## 12. Runtime (inside the session)

### 12.1 Entry sequence
1. Read args; load bundle; record env fingerprint (Python, CUDA, GPU model, torch/tf/transformers versions) into `segments/<seg>/env.json`.
2. Install requirements: respect preinstalled CUDA builds — do **not** reinstall `torch`/`tensorflow` unless the user pinned a different major/minor version; log any version differences between segments.
3. Stage data (from Kaggle dataset mount, HF dataset, URL, or `data-cache/`).
4. Download `LATEST.json` → if present, restore working files, replay/restore per Section 7.
5. Start watchdog, heartbeat, output capture, persist thread.
6. Execute (script executor or notebook executor) with instrumentation and framework hooks.
7. On normal end: final checkpoint of outputs → `status: completed` → exit 0.

### 12.2 `status.json`
```json
{"job_id": "...", "segment_id": "...", "state": "running|handoff|completed|failed_user|failed_infra",
 "heartbeat_at": "ISO-8601", "last_checkpoint_seq": 7, "progress": {"cell": 5, "loop": {"L3": [412, 1000]}},
 "rate": {"unit": "iter", "per_hour": 1450.0}, "message": "..."}
```
Heartbeat every 60 s. The controller treats a stale heartbeat (> 5 min) with a finished/errored backend session as a crash.

### 12.3 Exit codes
`0` completed · `85` handoff · `86` user-code error (deterministic, do not retry) · `87` infra/runtime error (retry elsewhere) · `88` non-resumable state detected and not accepted.

### 12.4 Watchdog
Knows `deadline = session_start + max_session_s`. Requests handoff when the forced-checkpoint condition (8.2) holds. Also handles `SIGTERM`/`SIGINT`: request checkpoint at next safe point if time allows, else exit.

### 12.5 User-code errors
An exception raised by user code is `failed_user`: stop the job, keep checkpoints, report traceback (with original line numbers). Do not burn quota retrying. `CUDA out of memory` is reported with suggestions (smaller batch, AMP) and is not auto-retried unless the next backend has a larger GPU.

---

## 13. Controller, scheduler, and budget

### 13.1 State
SQLite at `~/.gpurelay/state.db`: `jobs`, `segments`, `checkpoints`, `quota_ledger`, `events`. Config at `~/.gpurelay/config.toml`; credentials via the OS keyring (`keyring` library) or env vars.

### 13.2 Quota ledger
- Per backend: budget model (`weekly_rolling`, `monthly`, `unknown_opportunistic` for Colab), budget hours, usage records (segment start/end GPU time), user corrections (`gpurelay quota set kaggle --remaining 12h`).
- **Remaining budget is always displayed** in `status`, `quota`, and the dashboard, with the next reset time.
- Defaults are labeled as estimates in the UI.

### 13.3 Estimator
- Before first run: optional user hint (`--estimate 20h`).
- After a segment: progress rate from `status.json` (iterations/hour of the outermost loop, framework steps/hour, or cells completed with durations) → remaining GPU-hours = remaining work / rate, adjusted per GPU model when known.
- Output: "Job needs ≈ 34 GPU-h. Fits this week: Kaggle 22 h + Colab (manual) ~12 h" or "Needs ≈ 120 GPU-h → spans ~2 weeks; will pause at budget exhaustion and resume after reset."

### 13.4 Scheduling policy
1. Queue ordered by priority, then submission time.
2. For the next runnable job, choose a backend that: is ready, has remaining budget ≥ `min_segment_h` (default 0.5 h), has a free concurrency slot, and satisfies the job's GPU-memory need if known. Prefer automated backends; among them prefer the one with the most remaining budget; prefer keeping the same GPU model as previous segments when budgets tie (reduces numeric drift).
3. If only Colab is available → `awaiting_user` + notify.
4. If nothing has budget → `waiting_for_quota` with the earliest reset time; auto-resume after reset.
5. **Parallel jobs:** different jobs may run simultaneously on different backends (and up to `max_concurrent` on one backend). One job never runs two segments at once (except chunk-parallel jobs, later phase).
6. Hard rule: one configured account per backend type; the scheduler never treats two accounts of the same service as separate backends.

### 13.5 Job states
`queued → launching → running → (handoff → queued)* → assembling → completed`
Side states: `awaiting_user` (Colab), `waiting_for_quota`, `paused` (user), `failed_user`, `failed_infra` (after `max_infra_retries`, default 3), `cancelled`.

### 13.6 Notifications
Desktop notification (`plyer` or OS-native) + optional `ntfy.sh` topic. Events: needs Colab action, job completed, job failed, waiting for quota, budget low (< 10%).

---

## 14. Analyzer (`gpurelay analyze`)

Static AST analysis (and optional quick dry-run with `--probe` that runs the first N iterations on CPU/local GPU). Reports:

- **Resumability level** per block/cell and overall: `FULL` / `PARTIAL` / `RESTART_ONLY`, with which layer (A–D) covers it.
- **Risks:** unpicklable objects likely in loop state; side-effectful cells (downloads, deletes, writes outside working dir); nondeterminism (`cudnn.benchmark=True`, unseeded RNGs); safe-point gaps longer than `max_lost_work`; `while` loops / loops inside functions not instrumented.
- **GPU-time savings suggestions** (never applied automatically in v1): mixed precision if no `autocast`/`amp`; `DataLoader(num_workers=0)`; `pin_memory`; recomputing features (e.g. Inception features for FID) that could be cached; evaluation every step.
- **Environment:** imports without matching entries in requirements.

Output: human-readable (rich) + `analysis.json`.

---

## 15. CLI

```
gpurelay init                      # config, storage repo, backend credential checks, secrets dataset helper
gpurelay analyze <path> [--probe]
gpurelay submit <path> [--name N] [--priority P] [--requirements F] [--data SPEC]... \
                       [--estimate 20h] [--prefer kaggle] [--accept-restart-only]
gpurelay up                        # start controller daemon (foreground; --detach later)
gpurelay status [job]              # jobs, backend, progress, ETA, budget remaining
gpurelay logs <job> [-f]
gpurelay quota [show|set <backend> --remaining 12h|--used 3h]
gpurelay pause|resume|cancel <job>
gpurelay colab <job>               # path/link to the resume notebook + instructions
gpurelay fetch <job> [--to DIR]    # download final outputs
gpurelay run-local <path> [--simulate-kill-every 5m | --simulate-deadline 10m]   # testing
gpurelay dashboard                 # later phase: local web UI
```

`--data SPEC` forms: `kaggle:<owner>/<dataset>`, `hf:<repo_id>`, `url:<https://...>`, `local:<path>` (uploaded once to `data-cache/`).

---

## 16. Technology choices

Python ≥ 3.10 · `typer` (CLI) · `rich` (terminal UI) · `pydantic` v2 (config/models) · `sqlite3` (state) · `huggingface_hub` · `kaggle` · `IPython` + `nbformat` (notebook execution/output) · `ast` (instrumentation) · `cloudpickle` · `zstandard` · `keyring` · `pytest` (+ `pytest-timeout`) · `ruff` + `mypy` (strict on `gpurelay/` core modules). Torch / TF / transformers / lightning are **optional** imports: runtime modules must import them lazily and work when absent.

## 17. Repository layout

```
gpurelay/
  pyproject.toml
  README.md
  src/gpurelay/
    __init__.py            # public SDK re-exports (relay.state, relay.range, relay.chunks, ...)
    cli.py
    config.py  models.py  errors.py  constants.py
    analyzer/   static.py  probe.py  report.py  suggestions.py
    bundle/     packager.py  manifest.py
    controller/ daemon.py  scheduler.py  quota.py  estimator.py  db.py  notify.py  assemble.py
    backends/   base.py  local.py  simulated.py  kaggle.py  colab_manual.py  lightning.py(disabled)
    storage/    base.py  localfs.py  hf_hub.py
    runtime/    entry.py  executor_script.py  executor_notebook.py  instrument.py  loops.py
                state.py  rng.py  checkpoint.py  persist.py  watchdog.py  capture.py  status.py
                hooks/  hf.py  lightning.py  keras.py
    sdk.py
    templates/  kaggle_bootstrap.py.j2  kernel-metadata.json.j2  colab_resume.ipynb.j2
  tests/
    unit/  integration/  corpus/   # corpus = real ML workloads used for equivalence tests
  docs/
```

## 18. Verification strategy

**Rule: a feature is done only when its kill-and-resume equivalence tests pass.**

- **Corpus** (`tests/corpus/`, all runnable on CPU in seconds–minutes with tiny settings):
  1. Plain PyTorch MLP on synthetic data; records `losses` list; plots it; saves model.
  2. Same as a notebook.
  3. Tiny VAE on synthetic/MNIST-like data + FID-style metric computed in chunks with a small fixed CNN feature extractor (the owner's original use case, miniaturized).
  4. HF `Trainer` with a tiny randomly initialised model.
  5. PyTorch Lightning tiny model.
  6. Keras tiny model.
  7. Chunked evaluation loop over a list (no framework).
  8. Script with a `while` loop (expected `PARTIAL`, verifies analyzer honesty).
- **Equivalence harness:** run each corpus item uninterrupted, then under the Simulated backend with (a) random hard kills, (b) forced handoffs every N seconds, (c) both. Compare: final model weights (exact on CPU with deterministic settings; `allclose` with documented tolerance otherwise), recorded metric lists, merged log text (exact), figure count and image hashes where deterministic, output files hashes.
- **Fault tests:** kill during snapshot; kill during persist (LATEST must still point to previous valid checkpoint); corrupted checkpoint file (checksum mismatch → fall back to previous); storage temporarily failing (retry with backoff).
- **Scheduler tests:** budget exhaustion → `waiting_for_quota` → reset → resume; parallel jobs on multiple simulated backends; Colab-only availability → `awaiting_user`.
- **Real-backend smoke tests** (manual, documented in `docs/manual-tests.md`): tiny job on Kaggle with `max_session_s` forced low to produce a real handoff; Colab resume notebook run by hand.

## 19. Future work (post-v1)

Chunk-parallel execution of independent loops across backends (`relay.chunks(..., parallel=True)` or analyzer-proven independence) · nested-loop and mid-epoch DataLoader safe points · instrumentation of `while` loops and loops in functions · Lightning AI backend (after verification) · Google Drive storage · web dashboard improvements · controller hosted on an always-on small free CPU VM so handoffs continue while the laptop sleeps · multi-user mode · paid backends.
