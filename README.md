# GPU Relay (`gpurelay`)

Run long ML jobs (training, evaluation, metrics such as FID, preprocessing) to completion on
**free GPU tiers**. GPU Relay checkpoints your job automatically, hands it off before a session
or quota limit is hit, and resumes it on another platform (Kaggle, Colab, your own computer).
The final outputs are the same as an uninterrupted run.

You keep writing ordinary scripts or notebooks. Most of them need no changes.

## Install

Python 3.10 or newer.

```bash
git clone <this repository> gpu-relay
cd gpu-relay
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .                 # add extras as needed: ".[kaggle,dashboard]"
```

Extras: `kaggle` (Kaggle backend), `dashboard` (web UI), and `torch`, `tf`, `hf`, `lightning`
if you want to run those jobs on this computer.

## Try it on this computer (no accounts, 2 minutes)

```bash
gpurelay init --local                                    # storage on this computer only
gpurelay analyze examples/quickstart.py                  # will it resume? (FULL = yes)
gpurelay run-local examples/quickstart.py --simulate-kill-every 7s
```

`run-local` hard-kills the job every 7 seconds and resumes it from its last checkpoint. The
final log and `result.json` are identical to running `python examples/quickstart.py`.

The normal workflow uses the controller, which schedules jobs on your backends:

```bash
gpurelay submit examples/quickstart.py --name quickstart
gpurelay up --until-idle          # runs until no job is left (omit --until-idle to keep it running)
gpurelay status                   # in another terminal: progress, ETA, GPU budget
gpurelay fetch <job-id> --to out  # outputs: log.txt, files/, figures/, report.json
```

## Use free GPUs (Kaggle and Colab)

1. **Storage and Kaggle.** Create a Hugging Face token with write access
   (https://huggingface.co/settings/tokens) and a Kaggle API token
   (https://www.kaggle.com/settings/api, saved to `~/.kaggle/access_token`). Then:

   ```bash
   pip install ".[kaggle]"
   gpurelay init
   ```

   This creates a **private** Hugging Face dataset repo for your jobs and checkpoints, stores
   the token in your OS keyring, and creates a **private** Kaggle dataset holding the token
   that Kaggle sessions use (it never appears in kernel code or logs).

2. **Colab (optional, manual).** In `~/.gpurelay/config.toml` set `enabled = true` under
   `[backends.colab_manual]`. In Colab, add a secret named `GPURELAY_HF_TOKEN` (key icon)
   holding the same Hugging Face token.

3. **Run.** `gpurelay submit train.py`, then keep `gpurelay up` running. The controller uses
   Kaggle while it has GPU quota. When only Colab is left, it notifies you; run
   `gpurelay colab <job-id>`, upload the notebook it names to Colab and choose
   *Runtime → Run all*. It continues from the last checkpoint.

4. **Watch.** `gpurelay status`, `gpurelay logs <job-id> -f`, or the web dashboard:
   `pip install ".[dashboard]"` then `gpurelay dashboard` (jobs, timeline, budgets, live log,
   pause/resume/cancel).

Before submitting a real job, `gpurelay analyze train.py --probe` runs it here for a minute
and reports its speed, projected run time, memory, and checkpoint size.

## Web app on a server (always on, behind a login)

`gpurelay serve` runs the controller and the web app together on a server. The web app has
five screens: Jobs, Job detail, Submit, GPU budget and Settings. The owner logs in with a
password for full control. With `GPURELAY_PUBLIC_VIEW=1`, the setting `render.yaml` uses,
anyone with the URL can look without logging in but cannot change anything. Without it,
visitors need the optional viewer password.

To run it on Render's free plan:
1. Push this repo to GitHub.
2. In Render, choose New, then Blueprint, and pick the repo. `render.yaml` sets the service up.
3. Enter the secrets Render asks for: `HF_TOKEN`, `KAGGLE_API_TOKEN` and
   `GPURELAY_OWNER_PASSWORD`. The password needs at least 12 characters.

The server's disk is wiped whenever it restarts or sleeps, so the job database is backed up
to your private storage repo and restored on start. Free services sleep after 15 minutes with
no visitors; opening the URL wakes the service in about a minute. GPU sessions already running
on Kaggle carry on while it sleeps. `deploy/server/Dockerfile` also works on any other
container host.

## How it works

- **Checkpoints at safe points.** The outermost `for` loop of each long block is instrumented
  automatically. At a safe point the runtime snapshots your variables (PyTorch models and
  optimizers, NumPy arrays, plain Python objects), random number generators, and working
  files, then uploads them in the background. Hugging Face Trainer, Lightning, and Keras jobs
  use the framework's own checkpoints.
- **Hand off before the limit.** Each session gets a deadline below the platform's limit. Near
  it, the job checkpoints and exits cleanly; the controller starts the next session elsewhere.
- **Resume by replay.** A new session re-runs your script, skips finished loops, fast-forwards
  to the checkpoint, restores the state, and continues. Output already printed is not
  repeated, so the merged log reads like one run.
- **Explicit API when needed.** `import gpurelay as relay` gives `relay.range`,
  `relay.chunks`, `relay.state`, and `relay.safe_point` (see `docs/sdk.md`). Outside GPU Relay
  they are plain passthroughs.

More: `docs/automatic-resume.md`, `docs/controller.md`, `docs/checkpoint-layout.md`.

## Hard rules

GPU Relay follows the platforms' terms (spec §3):

- One account per service. No account rotation or quota pooling.
- Colab is never automated: GPU Relay only writes a resume notebook; you open it and press Run.
- Kaggle is automated only through its official API; kernels and datasets are always private.
- Secrets never appear in kernel source, notebooks, logs, or `config.toml`.

## Limitations

- **Numeric equivalence across GPU models.** Identical outputs are guaranteed for the same
  hardware and library versions. If a job moves between different GPUs (e.g. P100 → T4), GPU
  kernels can round differently; results stay correct but may differ in the last digits. The
  scheduler prefers the GPU model a job already used.
- **Uninstrumented loops.** `while` loops, loops inside functions, and nested loops are not
  safe points yet. Blocks without a safe point restart from their beginning after an
  interruption (`PARTIAL`), or the whole job restarts (`RESTART_ONLY`, needs
  `--accept-restart-only`). Use `relay.range` / `relay.chunks` / `relay.safe_point` there.
  `gpurelay analyze` tells you which case you are in.
- **Mid-epoch data loaders.** A `DataLoader` epoch is resumed from its start, not mid-epoch.
- **The controller runs on your computer.** Handoffs happen only while `gpurelay up` is
  running; sessions keep working (and checkpointing) if it stops, and it picks them up again
  when restarted.
- **Platform limits are estimates.** Kaggle quota and Colab session limits are configurable
  defaults (`docs/verify.md`); correct them with `gpurelay quota set <backend> --remaining 12h`.
- **Hub storage.** Each checkpoint is two commits to your private Hugging Face repo; very
  frequent checkpoints of very large models are slow there.

## Commands

| command | what it does |
|---|---|
| `gpurelay init [--local]` | config, storage, Kaggle credentials (or local-only setup) |
| `gpurelay analyze <file> [--probe]` | resumability report, risks, GPU-time suggestions |
| `gpurelay submit <file>` | upload a job (script or notebook) |
| `gpurelay up` | run the controller (scheduling, handoffs, notifications) |
| `gpurelay status [job]` / `quota` | progress, ETA, remaining GPU budget |
| `gpurelay logs <job> [-f]` | merged log |
| `gpurelay pause` / `resume` / `cancel <job>` | applied at the next safe point |
| `gpurelay colab <job>` | the Colab resume notebook and steps |
| `gpurelay fetch <job>` | download outputs |
| `gpurelay dashboard` | local web UI |
| `gpurelay run-local <file>` | relay a job on this computer (testing; can simulate kills) |

## Development

```bash
pip install -e ".[dev]"
pytest -m "not integration"      # fast unit tests
pytest -m integration            # kill-and-resume equivalence tests (slower)
ruff check . && ruff format --check .
mypy
```

Status and design notes: `docs/progress.md`, `spec.md`, `plan.md`.
