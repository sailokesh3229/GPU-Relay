# Manual smoke tests (real services)

Automated tests never touch Kaggle, Colab, or the Hugging Face Hub (plan rule 4). These manual
tests do. Record every finding in `docs/verify.md`.

## Phase 6 — Hugging Face storage + Kaggle

**Goal:** a job relays across real Kaggle sessions with a forced ~10 min limit (sequential
sessions on your single account), and its outputs are identical to an uninterrupted run.

The smoke job `tests/corpus/kaggle_smoke.py` needs no GPU and is deterministic on every
machine. It runs 40 steps × 30 s ≈ 20 min, so a 10 min limit forces at least 2 handoffs.

### 0. Prerequisites (once)

1. **Hugging Face:** an account and a token with *write* access
   (https://huggingface.co/settings/tokens).
2. **Kaggle:**
   - An account with a **phone-verified** profile (needed for internet access in sessions).
   - An API token: Settings → API → *Generate New Token*. Save it as
     `C:\Users\<you>\.kaggle\access_token` (or set `KAGGLE_API_TOKEN`).
3. Install the extras: `pip install -e ".[dev,kaggle]"`.

### 1. `gpurelay init`

```powershell
gpurelay init
```

- Paste the HF token when asked (or set `HF_TOKEN` first). init creates the **private**
  dataset repo `<hf-user>/gpurelay-store` and stores the token in Windows Credential Manager.
- When asked for the *token for Kaggle sessions*, preferably create a **fine-grained** token
  with write access to `<hf-user>/gpurelay-store` only, and paste it. Pressing Enter reuses the
  first token.
- init creates the **private** Kaggle dataset `<kaggle-user>/gpurelay-secrets` holding that
  token.

Check by hand:

- https://huggingface.co/datasets/<hf-user>/gpurelay-store shows **Private**.
- https://www.kaggle.com/datasets/<kaggle-user>/gpurelay-secrets shows **Private**.

### 2. Storage check with the local backend (~1 min)

This proves Hub reads and writes before spending any Kaggle time. Set the following in
`%USERPROFILE%\.gpurelay\config.toml`:

```toml
[backends.local]
enabled = true

[backends.kaggle]
enabled = false
```

```powershell
$env:SMOKE_STEPS = "6"; $env:SMOKE_SLEEP = "1"
gpurelay submit tests\corpus\kaggle_smoke.py --name smoke-local
gpurelay up --until-idle --interval 5
gpurelay status
gpurelay fetch <job-id> --to smoke-local
```

Expected: the job completes, and `smoke-local\files\result.json` exists. The repo on the
Hub now has `jobs/<job-id>/...`.

### 3. Real Kaggle relay (~35 min)

Edit `config.toml`:

```toml
[backends.local]
enabled = false

[backends.kaggle]
enabled = true            # set by init
max_session_s = 600       # force ~10 min sessions
enable_gpu = false        # CPU session: no GPU quota used by the smoke test
max_concurrent = 1

[runtime]
min_interval_s = 60       # checkpoint at least every minute or so
```

```powershell
Remove-Item Env:SMOKE_STEPS, Env:SMOKE_SLEEP   # Kaggle uses the defaults (40 x 30 s)
gpurelay submit tests\corpus\kaggle_smoke.py --name smoke-kaggle
gpurelay up --interval 20
```

While it runs (in another terminal):

- `gpurelay status` and `gpurelay status <job-id>` show segments, checkpoints, and budget.
- `gpurelay logs <job-id>` shows the committed log so far.
- https://www.kaggle.com/code/<kaggle-user>/gpurelay-<job-id> shows the kernel's versions
  (one per segment); open the log of each version.

Expected: at least 3 segments. Each one ends with `handoff` (or, rarely, `crashed` if Kaggle
stops it before the handoff). The job then reaches `completed`. Stop `gpurelay up` with
Ctrl+C after that.

### 4. Equivalence check

```powershell
gpurelay fetch <job-id> --to smoke-kaggle
mkdir smoke-ref; Copy-Item tests\corpus\kaggle_smoke.py smoke-ref\
cd smoke-ref; $env:SMOKE_SLEEP = "0"; python kaggle_smoke.py > ref_log.txt; cd ..
python -c "import json; a=json.load(open('smoke-kaggle/files/result.json')); b=json.load(open('smoke-ref/result.json')); print('RESULT IDENTICAL' if a==b else 'DIFFERENT')"
python -c "a=open('smoke-kaggle/log.txt').read(); b=open('smoke-ref/ref_log.txt').read(); print('LOG IDENTICAL' if a==b else 'DIFFERENT')"
```

Expected: `RESULT IDENTICAL` and `LOG IDENTICAL`.

### 5. Record findings (docs/verify.md)

Fill in what you observed:

- **Secrets:** did the secrets dataset mount at `/kaggle/input/...`? (#6, #19)
- **Time limit:** was `session_timeout_seconds` honoured? How long did each session take from
  push to the first heartbeat? (#18, BOOT_ALLOWANCE)
- **Commit rate:** were there Hub rate-limit warnings (`retry ... in`) in the controller output?
  (#9)
- **Kaggle status:** what did the kernel show after a handoff (`complete`)? (#18)
- **GPU variant (optional, uses GPU quota):** set `enable_gpu = true` and
  `accelerator = "NvidiaTeslaT4"` (or `NvidiaTeslaP100`), and repeat with `SMOKE_STEPS` small. Which
  accelerator names work? (#4, #17)

### Troubleshooting

| symptom | cause / fix |
|---|---|
| `kernel push failed: ... phone verification` | verify your phone number on Kaggle (needed for internet in sessions) |
| bootstrap log: `secrets dataset not attached` | the dataset ref in `config.toml` is wrong, or the dataset was deleted: rerun `gpurelay init` |
| bootstrap log: 401 from huggingface, or "does not hold a Hugging Face token" | the secrets dataset holds the wrong token (e.g. the Kaggle `KGAT…` token) or one without write access to the store repo: run `gpurelay init --rotate-kaggle-secret` and paste an `hf_` token (it is checked against the Hub; old dataset versions are deleted) |
| `another controller is already running (pid N)` | a previous `gpurelay up` is still alive (on Windows, closing a wrapper shell may leave it running): stop that process, then start `up` again |
| `not launched: ... already running` | a previous session of this job is still running on Kaggle; wait for it to stop |
| controller output shows `retry` warnings with 429 | Hub rate limit: raise `min_interval_s` / `heartbeat_s` |

## Phase 7 — Colab (manual)

**Goal:** a job relays Kaggle → Colab (you press Run) → completes, with outputs identical to an
uninterrupted run. Uses the Phase 6 smoke job and setup. GPU Relay never touches Colab: you
upload the notebook and run it yourself.

### 0. Config for the test

Back up `~/.gpurelay/config.toml`, then set:

```toml
[backends.local]
enabled = false

[backends.kaggle]
max_session_s = 600     # one short Kaggle session
enable_gpu = false

[backends.colab_manual]
enabled = true
```

### 1. Colab secret (once per Google account)

Open https://colab.research.google.com, open any notebook, click the key icon (*Secrets*) on
the left, and add a secret named `GPURELAY_HF_TOKEN` whose value is your Hugging Face token
(`hf_...`, the one from `gpurelay init`).

### 2. Submit and start the controller

```powershell
gpurelay submit tests\corpus\kaggle_smoke.py --name colab-smoke
gpurelay up --interval 20        # second terminal; keep it running
```

As soon as `gpurelay status` shows the job **running on kaggle**, take away Kaggle's budget so
the next segment has to go to Colab:

```powershell
gpurelay quota set kaggle --remaining 0h
```

### 3. Run the notebook in Colab

When the Kaggle session hands off (~8 min), the controller shows the job `awaiting_user` and
sends a notification. Then:

```powershell
gpurelay colab <job-id>
```

It prints the notebook path (`~/.gpurelay/sessions/colab_manual/resume_<job-id>.ipynb`) and the
steps: *File → Upload notebook*; allow *Notebook access* for the secret when asked;
*Runtime → Change runtime type → GPU* (optional for this CPU-only job); *Runtime → Run all*.

Expected notebook output: `storage token loaded`, `segment seg-…: resuming from checkpoint N`,
`runtime installed`, the job's `step …` lines continuing where Kaggle stopped, then
`[gpurelay] job completed`. The controller logs `user started the session`, then
`job_completed`.

### 4. Checks

- Equivalence: Phase 6 step 4 with this job id. Expected `RESULT IDENTICAL`, `LOG IDENTICAL`.
- *Runtime → Run all* again in Colab: it must stop with `this session already ran (completed)`.
- Optional (interrupt): submit another job, and while it runs in Colab press ■ on the last
  cell. Expected: `interrupted: the job checkpoints ...`, then a handoff; the controller asks
  for a new session (or uses Kaggle once it has budget).

### 5. Clean up and record

- Restore `config.toml` from the backup and give Kaggle its budget back:
  `gpurelay quota set kaggle --remaining 30h` (or your real remaining hours).
- Record in `docs/verify.md`: #25 (did the secret work after granting access?), #26 (did
  Colab's limit ever cut in before the handoff?), #27 (Python version in the env report:
  `gpurelay fetch` → `report.json`), #8 (any session limit you saw).

### Troubleshooting

| notebook stops with | cause / fix |
|---|---|
| `add a Colab secret named GPURELAY_HF_TOKEN` | add the secret (step 1) and allow notebook access when Colab asks |
| `must hold a Hugging Face token` | the secret holds something else (e.g. the Kaggle token): replace it with an `hf_` token |
| `the controller is not asking for a Colab session` | the job is not waiting for Colab (`gpurelay status`); the notification says when it is |
| `the job has moved on ... ran elsewhere` | another backend continued the job after the request; nothing to do |
| `this session is already running` | the notebook is running in another tab or runtime |
| 401/403 from huggingface | the token lacks write access to the store repo |
