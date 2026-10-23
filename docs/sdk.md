# Manual SDK and resume semantics

The SDK (spec §7.4) makes code resumable explicitly. Phase 3 adds automatic loop
instrumentation; the SDK stays available as the fallback.

```python
import gpurelay as relay

history = []
state = relay.state(model=model, optimizer=opt, scheduler=sched, history=history)

for epoch in relay.range(100):          # resumable range: a safe point after every iteration
    ...
    history.append(loss)

for chunk in relay.chunks(items, size=1000):     # resumable chunked work (FID, eval, ...)
    relay.save_result(chunk.id, compute(chunk.items))
results = relay.load_results()                    # every chunk's result, in order

relay.safe_point()                                # optional extra safe point
```

Outside GPU Relay (plain `python script.py`) every call is a cheap passthrough.

## How a resume works

1. A new session downloads the bundle, restores the checkpoint's **working files**, and runs
   the script **from the top in replay mode**: output is not recorded or echoed.
2. Relay loops that had **finished** before the checkpoint are **skipped** (they yield
   nothing). The loop that was **in progress** fast-forwards to the checkpointed iteration.
3. At the checkpoint's safe point the runtime **restores** the registered state (loading
   into the live objects that replay just rebuilt), the RNG states, and working files that
   replay modified; then execution continues **live** and output recording resumes.

Every relay loop and every `relay.safe_point()` is identified by its order of occurrence
(nested loops included). If re-execution diverges before the resume point (e.g. control flow
depends on the wall clock), the segment stops with exit code 88 instead of producing wrong
results.

## Your side of the contract

- **Register everything that must survive a resume**: models, optimizers, schedulers, scalers,
  and accumulators that live across iterations (`history`, `best_loss`, ...).
  - Mutable containers (list, dict, set, NumPy arrays, tensors) are restored **in place**, so
    your variables keep working.
  - Immutable values (numbers, strings, tuples) must be read and written through the state
    object: `state.best_loss = min(state.best_loss, loss)`.
- Code before and between relay loops is **re-executed** on resume; keep it deterministic and
  free of non-idempotent side effects (downloads are fine if they are cached in the working
  directory, which is restored).
- Seed your RNGs as usual; RNG state is checkpointed and restored at the resume point.
- `relay.save_result` values must be picklable (cloudpickle). They are stored once, in a
  content-addressed blob store, not re-uploaded with every checkpoint.

## Parallel chunks (`parallel=True`)

```python
for chunk in relay.chunks(items, size=1000, parallel=True, seed=0):
    relay.save_result(chunk.id, compute(chunk))
results = relay.load_results()          # every chunk, in order
```

The controller may compute chunks of this loop on several backends at the same time
(`[scheduler] max_parallel_workers`, default 2 extra sessions). The main session walks the
chunks in order; **worker** sessions start from a main checkpoint taken inside the loop and
take chunks from the end. Results and printed output of chunks computed elsewhere are merged
in chunk order, so the log reads exactly like a sequential run.

The contract for a parallel loop:

- `items` is a sequence (list, `range`, array, ...).
- A chunk returns data **only** through `relay.save_result`. Variables it changes, files it
  writes, and figures it draws are not merged from workers.
- Every chunk starts with all random number generators seeded from `seed` and its id, and
  after the loop they are seeded from `seed` and the number of chunks. This holds with and
  without GPU Relay, so results never depend on where a chunk ran.
- Two sessions may occasionally compute the same chunk (e.g. after a worker was lost); the
  results are identical, only time is wasted.

## Outputs

- `stdout`/`stderr` (including tracebacks) go to one log per segment; final assembly
  truncates each segment's log at its last committed checkpoint and concatenates them, so the
  merged log equals an uninterrupted run. Line endings are stored as `\n` on every platform.
- `matplotlib.pyplot.show()` saves every open figure as PNG (in order) and closes it.
- Files written under the working directory are exported to `final/files/`.
- Not captured in v1: output written directly to OS file descriptors (C extensions, child
  processes). Library warnings that Python prints once per process may print again after a
  resume.

## Exit codes (spec §12.3)

`0` completed · `85` handoff · `86` user-code error (not retried) · `87` infra/runtime error
(retry elsewhere) · `88` non-resumable (re-execution diverged / state cannot be restored).
