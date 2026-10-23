# Controller, scheduler, and GPU budgets (Phase 5)

## Typical session

```bash
gpurelay submit train.py --estimate 20h      # analyze, bundle, upload, queue
gpurelay up                                   # controller (foreground); Ctrl+C leaves sessions running
gpurelay status                               # jobs (progress, ETA) + remaining budget per backend
gpurelay status <job>                         # segments, checkpoints, "fits this week" estimate
gpurelay logs <job> -f                        # committed log so far (final log when done)
gpurelay quota set sim-a --remaining 12h      # correct a backend's budget
gpurelay pause|resume|cancel <job>
gpurelay fetch <job> --to results/
```

Until Hugging Face storage arrives (Phase 6), configure local storage in
`$GPURELAY_HOME/config.toml`:

```toml
[storage]
kind = "localfs"            # files under $GPURELAY_HOME/store
```

## State

`$GPURELAY_HOME/state.db` (SQLite, WAL) holds these tables:

- `jobs` and `segments`: validated JSON documents plus indexed state columns;
- `checkpoints`: committed sequence numbers seen by the controller;
- `quota_ledger`: usage records per segment, plus user corrections;
- `events`: the audit trail;
- `schema_version`: drives migrations.

The CLI writes jobs and requests to the database; the controller applies them on its next tick.

## Job lifecycle (spec §13.5)

```
queued -> running -> (queued after handoff / crash)* -> assembling -> completed
side states: waiting_for_quota, awaiting_user (manual backends), paused,
             failed_user (never retried), failed_infra (after max_infra_retries), cancelled
```

- A crash that made progress (a new checkpoint was committed) is not counted as an infra failure.
  A crash without progress, or an exit code 87, is counted.
- **Pause** writes `jobs/<job>/control.json`. The runtime's heartbeat reads it and hands off at the
  next safe point; the job then becomes `paused`. This works on any backend.
- `status.json` is the source of truth for progress. The backend's own state is only used to
  detect finished and crashed sessions; a stale heartbeat on a "running" session means it hung.

## Budgets (spec §13.2)

| model | window | next reset |
|---|---|---|
| `weekly_rolling` | any 7 days | when enough old usage leaves the window for a minimum segment |
| `monthly` | calendar month (UTC) | 1st of next month |
| `unknown_opportunistic` | — (Colab) | unknown |
| unlimited | local machine | — |

Running segments count as used while they run. Configured defaults are shown as "(est.)" until
the user records a correction.

## Scheduling (spec §13.4)

A pure function, `controller/scheduler.py::plan`:

1. Jobs are taken in priority order, then by submission time.
2. A job goes to an automated backend that is ready, has at least `min_segment_h` of budget left,
   and has a free slot. The choice, in order: the job's `--prefer` backend; then the most remaining
   budget; when budgets are within 0.5 h, the GPU model of the job's last segment.
3. Session deadline = min(backend session cap, remaining budget).
4. If only manual backends can take the job → `awaiting_user`, and the user is notified.
5. If no backend has budget → `waiting_for_quota` until the earliest reset.
6. A job with a live segment is never scheduled again. Different jobs run in parallel.

## Bundles

`jobs/<job>/bundle/` contains:

- `manifest.json`: entry, per-file sha256, requirements, data, content hash, runtime version;
- `bundle.tar.zst`: deterministic tar;
- the runtime wheel, built from the installed package with fixed timestamps.

`local:` data is uploaded once to `data-cache/<sha256>/` and staged into the working directory
under its base name. Staged data is excluded from working-file tracking. `url:` data is
downloaded by the session. `kaggle:` and `hf:` arrive in Phase 6.

## Notifications

Notifications go to the desktop (if `plyer` is installed) and to an optional ntfy.sh topic
(`[notify] ntfy_topic`). Events: job completed, job failed, waiting for quota, Colab action
needed, budget low (< 10%).

## Not yet (later phases)

- Real backends: Kaggle (Phase 6), Colab manual (Phase 7).
- Controller restart recovery for subprocess backends; running local sessions are not
  re-attached after a controller restart (Phase 8 hardening).
- `gpurelay up --detach`.
