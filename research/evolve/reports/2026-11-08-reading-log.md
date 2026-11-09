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
- GOLEM `EvaluationDispatcher.apply_evaluation_results` drops an individual whose evaluation returned nothing. `population_evaluation_info` warns when half or fewer of the population evaluated. `compare_pack` previously substituted the task sentinel for `status=crash`, so a crashed stock side became a numeric baseline. `confirm_dev` counted only `timeout` and `invalid`, so that crashed seed was skipped and the other seeds could still confirm a metric signal.
- GOLEM `STAGNATION_EVALUATION_PERCENTAGE` is 0.5. `population_evaluation_info` warns when `evaluated / pop_size <= 0.5`, including an empty population. Drain `stock_ok` previously treated every present `status=ok` stock row as a measured pack, and the stock-only CLI did the same without `metric_valid`.
- GOLEM `evaluate_single` returns `None` when `timer.is_time_limit_reached()` before it builds a fitness. The worker previously kept a holdout number as `metric_valid=true` when fit returned after reporting `FedotBudgetExceeded` or `budget exhausted`.
- GOLEM `apply_evaluation_results` keeps only evaluations that are not `None`. A worker that exits at the moment the parent deadline is already reached was recorded as a normal completion, so its result file could stand in for the timeout. Recovery already prefers a valid timeout manifest; that preference was not locked against a newer result file.
- GOLEM `split_individuals_to_evaluate` skips an individual whose `fitness.valid` is already true. A `quality_task_worker_spawned` row for the same attempt used to erase a finished timeout and leave the side interrupted, so the next pair sent it through fit again.
- GOLEM `OptimisationTimer` starts another generation only when the timeout is strictly longer than the time already spent plus the average generation. A transient failure that had already used the whole side budget still started a second attempt with a fresh full timeout.
- GOLEM elitism copies the best individuals into the next population and does not evaluate them again. A warm stock cache still opened a new stock attempt before the cache was read, so recovery could send that identity through fit.
- GOLEM `GraphVerifier` returns false, or raises, when a graph fails a rule, and that graph is not evaluated. A stock cache entry whose worker attempt identity named another side or task was still reused as a stock hit.
- GOLEM `has_no_cycle` rejects a graph that points back into itself before evaluation. A finished journal event could still adopt a result whose `attempt_id` named a different attempt. A stock cache lookup also crashed when the path resolver returned no file, so a disabled cache stopped the pair before the side ran.
- GOLEM `has_no_isolated_nodes` rejects a node with no edge into the rest of the graph, unless the graph is a single node. A finished side event with no `task_id` was stored under `str(None)`, so a queue task of `null` became a measured dataset and the event counted as task progress.
- GOLEM `has_no_self_cycled_nodes` rejects a node that lists itself as an input. A finished side event could still adopt a score whose `worker_paths` resolved to the journal that stored the event.
- GOLEM `has_no_isolated_components` rejects a graph that falls into disconnected pieces. `stock_pack_measured` returned true for attested stock rows when no job asked for those datasets, and an extra dataset beside a requested one still counted as a measured pack.
- GOLEM `has_one_root` accepts a graph only when `root_node` is a single node. Two quality rows for one dataset stored one delta, so a later gain replaced an earlier regression and the proposal could KEEP.
- GOLEM `has_root` rejects a graph whose `root_node` is empty. A quality row with no `source_dataset` could still be the only KEEP root, and a missing key raised `KeyError` instead of leaving the proposal unevaluated.

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
- [x] `evidence_only` does not return `metric_signal` when another lift dataset is invalid. A stock `crash` is not a sentinel baseline. One crashed DEV seed counts as infrastructure and blocks confirmation.
- [x] A quality pack is measured only when every required dataset has `status=ok`, a finite score, and `metric_valid=true`. Half attested, or a bare `ok` score, leaves `stock_ok` false and the stock-only command exits infrastructure.
- [x] A reported FEDOT budget timeout does not keep a later holdout number. The worker and the parent store `status=timeout`, `metric_valid=false`, and `discarded_holdout_score`.
- [x] A completion observed at the deadline stays `task_timeout`. A newer result file next to a parent timeout manifest does not become the recovered metric.
- [x] A finished timeout stays settled when a later spawn or start names the same attempt. That side is reused and is not sent through fit.
- [x] A failure that already used the side timeout does not start another attempt of that length. A short transient failure can still use the one allowed retry.
- [x] A stock cache hit is returned before a worker plan is created. The pair does not launch a stock worker and does not record `started` or `spawned` for that identity.
- [x] A stock cache entry whose attempt identity names another side, task, or attempt is not loaded and is not stored.
- [x] A finished journal event whose `attempt_id`, or the `attempt_id` inside its result, names a different attempt is not recovered as that side's metric. A cache hit may still keep the producer attempt identity. A missing stock-cache path is a miss, not a crash.
- [x] A finished side event with no `task_id` is not task progress and does not make a null queue task measured.
- [x] A finished side whose `worker_paths` resolve to the journal is not recovered as that side's metric.
- [x] `stock_ok` stays false when a measured stock row names a dataset that no job asked for, including a pack with no jobs.
- [x] Two quality rows for the same dataset stay `not_evaluated`. The stored delta is the worse one, so a later gain does not hide the regression.
- [x] A quality row with no `source_dataset` stays `not_evaluated`. It does not KEEP, and a rooted gain beside it does not KEEP either.
- [ ] Read GOLEM `GraphVerifier`: a rule that returns nothing rejects the graph before evaluation. Check that `compare_pack` refuses KEEP when a requested lift id is missing from the stock or patch map, instead of raising `KeyError`.
