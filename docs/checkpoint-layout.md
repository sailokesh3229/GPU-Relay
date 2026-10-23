# Checkpoint layout

Implements spec §8. Code: `src/gpurelay/runtime/checkpoint.py`, `persist.py`, `state.py`, `rng.py`.

## Protocol

1. **Snapshot** (blocking, at a safe point): write to `<local_root>/ckpt-<seq>.tmp/`, then rename
   to `ckpt-<seq>/`. A kill here leaves only a local temp dir, which the next
   `CheckpointManager` on that machine deletes.
2. **Persist** (single background worker): for each file, zstd-compress (JSON files are uploaded
   as-is) → upload with retry/backoff → verify the stored SHA-256 (cheaply via
   `ChecksumStorage.sha256` when the storage supports it, otherwise download and hash) →
   re-upload on mismatch.
3. **Commit**: write `ckpt-<seq>/manifest.json`, then atomically replace `LATEST.json` with the
   same content. **A checkpoint exists only once `LATEST.json` points to it.**
4. **Retention**: delete every checkpoint older than the newest `keep` (default 2) committed
   ones, manifests first (so a half-deleted checkpoint is never offered as a fallback).
5. A snapshot requested while a persist is in flight waits for it; the wait is recorded as stall
   time (`CheckpointStats.total_stall_s`).

Sequence numbers continue across segments and are **never reused**: a new segment starts at
`max(any seq seen in storage) + 1`, so a partial upload from a killed segment is never
overwritten in place; retention removes it after the next commit.

## Remote files

```
jobs/<job_id>/checkpoints/
  LATEST.json                  # CheckpointPointer: seq, files, sha256s, safe_point_id, created_at, segment_id
  ckpt-<seq>/
    manifest.json              # same content as LATEST.json at commit time (fallback index)
    meta.json                  # seq, job_id, segment_id, safe_point_id, created_at,
                               # positions (loop/cell/output offsets), state manifest, env fingerprint
    rng.pt.zst                 # pickled RNG states: python, numpy, torch CPU + CUDA, TF global generator
    torch_state.pt.zst         # torch.save({state_dicts, types, objects}) of torch-aware variables
    vars.pkl.zst               # cloudpickled plain variables
    <extra>/...zst             # framework/, outputs/, files — provided by later phases via `extra=`
```

`sha256s` are of the bytes **as stored** (i.e. compressed).

### Runtime metadata and job-level objects (Phase 2)

`meta.json` carries `positions.position` (the resume point, see `docs/sdk.md`) and `runtime`:

```json
{"chain":   [{"segment": "seg-a", "log_bytes": 1834, "figures": ["<sha256>", ...]}, ...],
 "files":   {"model.pt": "<sha256>", "out/metrics.json": "<sha256>"},
 "results": {"L1": {"0": "<sha256>", "1": "<sha256>"}},
 "final":   false}
```

- `chain`: inherited from the checkpoint the segment resumed from, plus this segment's entry.
  Final assembly concatenates each listed segment's log truncated at `log_bytes`.
- `files`: working-directory files that differ from the bundle.
- `results`: `relay.save_result` values per chunks loop.

Objects outside the checkpoint directories (never deleted by checkpoint retention):

```
jobs/<job_id>/
  blobs/<sha256>                 # content-addressed: figures, chunk results, working files
  segments/<seg_id>/log.txt      # segment log (uploaded with each checkpoint, before commit)
  segments/<seg_id>/env.json     # environment fingerprint of the session
  status.json                    # heartbeat (spec §12.2)
  bundle/manifest.json, files/   # Phase 2 bundle form (Phase 5 packs it)
  final/                         # log.txt, figures/, files/, report.json
```

Blobs referenced by a checkpoint are uploaded (and verified) **before** its commit.
Garbage collection of blobs no longer referenced by any retained checkpoint is not done yet.

### Differences from the spec §8.3 sketch

- Every non-JSON file carries a `.zst` suffix (the spec shows it only on `vars.pkl` and
  `files.tar`); compression is applied uniformly in the persist step.
- `manifest.json` is an addition: it lets the loader fall back to the previous checkpoint even
  when `LATEST.json` itself is unreadable.
- `rng.pt` uses plain pickle (not `torch.save`) so it works without torch installed.
- `framework/`, `files.tar.zst`, and `outputs/` are not produced yet; the `extra=` argument of
  `CheckpointManager.snapshot` is the hook later phases use to add them.

## Loading

`load_latest_checkpoint(storage, job_id, dest)`:

- no `LATEST.json` → `None` (nothing committed; uncommitted manifests are ignored);
- otherwise try `LATEST.json`, then older `manifest.json`s newest-first; each candidate is
  downloaded, checksum-verified, and decompressed; the first that verifies wins
  (`fell_back=True` if it is not the one `LATEST.json` names);
- every candidate fails → `CheckpointCorruptError`.

`LoadedCheckpoint.restore(namespace)` restores variables, then RNG state (spec §7.5 order):

- torch-aware objects: `load_state_dict` into the **live** object of the same type (keeps
  optimizer ↔ parameter references); restore order modules → optimizers → others. If there is no
  compatible live object, the whole saved object is used (objects saved together are unpickled
  together, so their references to each other survive).
- plain variables: assigned into the namespace.

## Known limitations (v1)

- Containers of torch objects (e.g. `models = [a, b]`) are pickled as plain values, so after a
  restore they are copies, no longer linked to live optimizers.
- If a model falls back to its saved object while its optimizer loads into a live object (or
  the reverse), the optimizer will not drive the model. A warning is logged.
- Plain variables holding CUDA tensors cannot be restored on a machine without CUDA.
- Checkpoints are unpickled (`torch.load(weights_only=False)`, `pickle`). The trust boundary is
  the user's own private storage; never load checkpoints from untrusted sources.
