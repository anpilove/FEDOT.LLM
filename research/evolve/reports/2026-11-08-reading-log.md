# Reading log for the reliability cycle

Started 2026-11-08 19:09 Europe/Moscow. Keep going until 2026-11-09 03:10 Europe/Moscow,
then unsubscribe `loop-evolve-research`. No model change, no FEDOT automerge,
no quality KEEP, no OpenML registry run.

## Read

- `docs/superpowers/specs/2026-09-30-fedot-budget-outcomes-design.md`
- `fedotllm/agents/evolve/README.md` through the controller split
- `fedotllm/agents/evolve/controller/README.md` through the module map
- FEDOT paper, arXiv:2106.15397: workflow design is bounded by a time limit tau_g. Current FEDOT docs also default `early_stopping_timeout` to 10 minutes, separate from the hour-long side budget.
- `controller/README.md` failure section: a finished watchdog is not retried. `Fedot.timeout` and the side deadline are different budgets. The README does not mention the 10-minute GOLEM stagnation default.
- `execution/process.py` and `_worker_lifetime_guard.py`: the parent records `task_timeout`; the guardian exits 124 for its own cap. Diagnose labels `fit_without_optimizer_evidence` and does not stop the worker. `run_worker` documents that `cancellation_event` yields `cancelled`, not a timeout.
- GOLEM `Fitness` (golem/core/optimisers/fitness/fitness.py): an invalid fitness, including `None`, is never better than a measured fitness. A signaled evaluator leaves no fitness. The worker previously recorded that the same way as a normal nonzero exit.
- Linux reports memory pressure by delivering `SIGKILL`. Our own deadline and cancellation paths already record their outcomes before a missing result file is read, so a remaining `SIGKILL` is external. `SIGSEGV` stays a per-side fault.
- `storage_guard.py` treats errno 28 and 122 as capacity failures. The worker previously recognized only errno 28, so a quota error during fit became `fedot_fit_error`.
- `supervisor.py` returns immediately when the saved ledger is `storage_error`, and a child exit 75 also latches that state. `drain` itself was still willing to select `queued` and `running` jobs.
- GOLEM fitness comparison: an invalid fitness is never better than a measured one, so a failed evaluation is not a successful individual. `compare_pack` previously skipped an invalid lift dataset and could still return `keep=True` from the other datasets.

## Found

- The budget spec says an outer watchdog is `timeout` with `watchdog_fired=true`,
  not host infrastructure. The code had `timeout_kind` but no `watchdog_fired`,
  and every unlabeled timeout became `infrastructure_error`.
- The same spec asks for one retry of a scoreless timeout. Tests
  `test_quality_pair_does_not_retry_timeout_or_deterministic_patch_crash` and
  `test_quality_pair_never_retries_timeout_with_transient_marker` forbid that
  retry. Leave the retry policy alone until a replacement test is designed.
- No in-repo paper bibliography. FEDOT and GOLEM citations still need a pass
  against controller failure modes only.

## Open

- [x] Record `watchdog_fired` and classify that timeout as `watchdog_timeout`, not infrastructure. Pause the queue on it.
- [x] Read the rest of `controller/README.md`. Diagnose reports `fit_without_optimizer_evidence` and does not stop the worker. No deadlock action to remove.
- [x] Keep a parent `task_timeout` when the guardian's monotonic fire time is at or after the task deadline. An earlier guardian fire still stays `watchdog_timeout`.
- [x] Pass `early_stopping_timeout` equal to the declared FEDOT minute budget and record `early_stopping_timeout` separately from `composing_did_not_start`. A valid metric stays `ok`.
- [x] A parent cancellation stays `cancelled` when the guardian exits 124 while an escaped pipe is still draining. It is not also `watchdog_timeout`.
- [x] A worker killed by a signal, with no result file, is `worker_signaled` plus the signal number. It is not `worker_exit` and not a candidate failure.
- [x] A remaining `SIGKILL` sets `infrastructure_error` and pauses the queue. `SIGSEGV` stays `evaluation_error` for that side only.
- [x] A caught `MemoryError` is `memory_error`: per-side `invalid`, not `candidate_error`, and not the `SIGKILL` queue pause.
- [x] Errno 28 and 122 are `storage_error`. A decision that sees that kind pauses and does not call it a candidate failure. The parent still raises `StorageCapacityError`.
- [x] A `StorageCapacityError` from measurement returns the proposal to `queued` with `keep` unset and reason `storage_capacity_exhausted`. It does not stay `running` and it is not a candidate verdict.
- [x] The supervisor does not relaunch a latched `storage_error`. Drain now refuses too, before it selects `queued` or `running` jobs.
- [x] One invalid lift dataset blocks `compare_pack` KEEP. A candidate failure stays a broken patch. Any other missing score stays infrastructure and `keep=False`.
- [x] Quality decision already returns `keep=None` when any dataset is unresolved. A test now locks one measured gain plus one `invalid` dataset.
- [ ] Check `evidence_only` compare: a small positive signal must not become KEEP when another lift dataset is invalid.
