# Automatic resumption of unmodified scripts (Phase 3)

Spec §7.2 (framework hooks) and §7.3 (loop instrumentation), applied with the replay-and-restore
model of §7.1 at the granularity of **top-level statements**.

## How a script runs

`runtime/instrument.py` parses the script and splits it into top-level statements. Each one is
compiled separately (from the transformed AST, so tracebacks keep the original line numbers) and
analysed:

- **Loop instrumentation (Layer C):** every *outermost* `for` loop reachable from module level,
  including inside `if`/`with`/`try`/`match` blocks such as `if __name__ == "__main__":` or
  `with torch.no_grad():`, becomes `for x in __relay__.loop("<loop_id>", ITER):`, with a safe
  point after every iteration. Loop ids hash the file name, line, and loop header.
  `relay.range`/`relay.chunks` loops are left to the SDK. `while` loops, `async for`, and loops
  inside functions are not instrumented in v1.
- **Names written** by the statement: assignments, augmented assignments, `del`, loop and `with`
  targets, attribute and subscript stores, mutating calls (`.append`, `.update`, `.step`,
  in-place tensor methods ending in `_`, ...). Also recorded: names passed as call arguments, and
  names declared `global` in functions.
- **Side effects:** downloads, deletions, subprocesses, writes to absolute paths.

Every top-level statement boundary is also a safe point.

## Replay prefix

A statement is *replayable* when it is cheap (< `replay_threshold_s`, default 30 s) and has no side
effects (directly or in module functions it calls). The **replay prefix** ends at the first
non-replayable statement `P`. Imports and `def`/`class` statements are always replayable.

## What a checkpoint holds

At a safe point in statement `s`, the snapshot captures:

- names written by statements `P..s`;
- call arguments in those statements that are containers or instances of classes defined in the
  script;
- `global`-declared names;
- every torch-stateful object;
- everything registered with `relay.state`.

The checkpoint also records which skipped statements constructed torch objects, plus the
per-statement durations.

## How a resume works

1. Restore working files.
2. Re-execute the replay prefix and definitions as *setup*: output suppressed, loops run to
   completion, no safe points. Skip all other statements before the target.
3. Skipped statements that built torch objects (e.g. `optimizer = Adam(model.parameters())`) are
   re-executed too, after the snapshot's values are in place, when they are loop-free, side-effect
   free and fast. This keeps an optimizer bound to the *live* model.
4. Put the snapshot's variables into the namespace, then re-execute the target statement in replay
   mode up to the resume point: earlier loops in it are skipped, and the interrupted loop
   fast-forwards (sequences and ranges are indexed; other iterators are consumed and discarded).
5. At the resume point: restore variables again (loading torch state into the live objects), then
   RNG states and working files changed by replay. Then continue live.

Divergent control flow before the resume point, or a different loop id at the target, exits with
code 88.

## Framework hooks (Layer B)

Hooks are installed lazily when the framework is imported, and are version-checked (unknown
major versions → warning, no hook). Each training call (`Trainer.train`, `Trainer.fit`,
`Model.fit`) is a resumable *framework event*. The framework's own checkpoint is stored in the
relay checkpoint's `framework/` directory.

| Framework | Save trigger | Resume | Notes |
|---|---|---|---|
| transformers 4.x/5.x `Trainer.train` | callback sets `should_save` when due, **aligned to logging steps**, with a lookahead so handoffs land on a logging step | `resume_from_checkpoint=` | Forced saves go to a private dir with output muted, so `output_dir` and the log match an uninterrupted run. HF's running-loss accumulator is not checkpointed, so an emergency handoff between logging steps changes that one logged loss. |
| Lightning 2.x `Trainer.fit` | `Checkpoint` callback at train-epoch end (runs after the module's hook) | `ckpt_path=` | Epoch granularity. |
| Keras 3 `Model.fit` (any backend) | `BackupAndRestore` + relay callback at epoch end | restored backup dir | RNG re-applied at the first resumed epoch (`fit` setup draws randomness). The `History` returned by a resumed `fit` covers only resumed epochs. |

## Output continuity additions

- Python `warnings` shown in earlier segments are not shown again: Python's once-per-location
  rule, extended to the whole job. The keys are stored in the checkpoint.
- Output produced by relay-initiated framework saves is muted.

## Nested loops, DataLoaders, `while` loops, loops in functions (Phase 9)

Settings under `[runtime]` in `config.toml`:

| setting | default | effect |
|---|---|---|
| `loop_depth` | `2` | levels of nested `for` loops with safe points (`1` = outermost only, the v1 behaviour) |
| `instrument_while` | `false` | `while` loops get safe points |
| `instrument_functions` | `false` | loops inside top-level functions get safe points; the function's locals are checkpointed |

- **Nested loops** (e.g. epochs → batches) resume *inside* the outer iteration: the outer
  iteration is replayed up to the inner loop (output suppressed), the inner loop
  fast-forwards, and state is restored. A nested loop is tracked for its first 32 iterations;
  if those take < 0.5 ms each, the rest of it runs untracked (tracking costs ~2 µs per
  iteration), and the enclosing loop's safe points still apply.
- **Mid-epoch DataLoader resume.** For loops over sampler-driven iterables (torch
  `DataLoader`), the RNG state at the start of iteration is part of the checkpoint and is
  restored before the iterator is recreated, so a shuffled loader yields the same batches. A
  single-process, map-style DataLoader skips finished batches without loading them; others
  are fast-forwarded by loading and discarding.
- **`while` loops** become `for __relay_iter__ in loop(id, forever()): if not cond: break`.
  On resume, finished iterations are skipped and the condition is evaluated again on the
  restored state, so it must depend only on variables (no I/O such as `readline()`).
  `while ... else` is not instrumented.
- **Loops inside functions**: `for` loops (and `while` loops if enabled) inside top-level,
  non-generator, non-async functions. At their safe points the function's assigned locals
  (and its torch objects) are checkpointed; on resume the call is replayed up to the loop and
  the locals are written back into the running function (Python 3.13+: `frame.f_locals`;
  3.10–3.12: CPython `PyFrame_LocalsToFast`). Arguments that the function does not reassign
  come from the replayed call.
- **Imports inside `try`/`if` blocks** that a resume skips are imported again by name.

## Limitations

- `async for`, generator functions, nested functions and methods, and `while ... else` have no
  safe points; such blocks restart from their beginning (PARTIAL). A single such block
  holding all the work makes the job RESTART_ONLY. Use `relay.range`/`relay.chunks` there.
- Values that cannot be pickled (open files, generators, live iterators over DataLoaders) are not
  checkpointed and are reported once per name.
- Checkpoint size: all torch-stateful objects (including frozen feature extractors) and the
  written names are saved at every checkpoint. Content-deduplicated per-variable storage is
  future work.
