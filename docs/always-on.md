# Always-on controller on a small CPU machine

Handoffs happen only while `gpurelay up` runs. If your laptop sleeps, a session that hands off
waits until the laptop wakes. Running the controller on an always-on machine (a small free
cloud VM, a home server, a Raspberry Pi) keeps jobs moving. The controller needs no GPU: about
256 MB of RAM and almost no CPU.

Storage must be shared: Hugging Face (`gpurelay init`) or Google Drive. Local storage only
works if the jobs also run on that machine.

## 1. A machine

Any Linux machine with Python 3.10+ (or Docker) and outbound internet. Free options to check
(their terms and limits change; see `docs/verify.md` #29):

- Oracle Cloud "Always Free" Ampere A1 VM;
- Google Cloud "Always Free" e2-micro (selected US regions).

## 2. Install

Docker (from a clone of this repository):

```bash
docker build -f deploy/Dockerfile -t gpurelay .
```

Or without Docker:

```bash
python3 -m venv ~/gpurelay-venv && ~/gpurelay-venv/bin/pip install "./gpu-relay[kaggle,dashboard]"
```

## 3. Move the state

On the laptop:

```bash
gpurelay export-state gpurelay-state.tgz   # config + job database, no secrets
scp gpurelay-state.tgz vm:
```

On the VM:

```bash
gpurelay import-state gpurelay-state.tgz
# Docker: docker run --rm -v gpurelay-data:/data -v $PWD:/in gpurelay import-state /in/gpurelay-state.tgz
```

## 4. Tokens on the VM

There is usually no keyring on a server, so tokens come from the environment. Create
`~/.config/gpurelay.env` with `chmod 600`:

```
HF_TOKEN=hf_...
KAGGLE_API_TOKEN=KGAT_...
# Google Drive storage instead: GPURELAY_GDRIVE_CREDENTIALS={...json from the keyring...}
```

For notifications, set a private, hard-to-guess [ntfy](https://ntfy.sh) topic in
`config.toml` (`[notify] ntfy_topic = "..."`) and subscribe on your phone: desktop
notifications do not reach you from a server.

## 5. Start it

Stop the controller on the laptop first. Then, on the VM, either:

```bash
docker run -d --name gpurelay --restart unless-stopped -v gpurelay-data:/data \
    --env-file ~/.config/gpurelay.env gpurelay up
```

or install `deploy/gpurelay.service` (systemd) and `sudo systemctl enable --now gpurelay`.

**Only one controller per store.** The controller holds a lease in the storage. A second
`gpurelay up` (on the laptop, say) watches it for up to 3 minutes and refuses to start while
it is being renewed. `gpurelay up --takeover` takes it at once; the other controller stops
at its next renewal (within ~2 minutes). Tested with two containers sharing one store.

## 6. Daily use

Run commands on the VM over SSH (`ssh vm gpurelay status`, copy scripts with `scp` and
`ssh vm gpurelay submit job.py`). For the dashboard, start it on the VM and tunnel it:

```bash
ssh -L 8765:127.0.0.1:8765 vm gpurelay dashboard --no-open
# open http://127.0.0.1:8765 on the laptop
```

When a job needs Colab, the dashboard's banner has a **download the notebook** link.

## 7. Moving back

Stop the VM controller (`docker stop gpurelay` / `systemctl stop gpurelay`), then
`gpurelay export-state` on the VM, `import-state --force` on the laptop, and `gpurelay up`
(or `gpurelay up --takeover` if the VM cannot be reached).
