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
- [ ] Read the rest of `controller/README.md` and check diagnose/heartbeat against the code.
- [ ] Read `execution/process.py` watchdog versus task timeout and add a missing distinction if the tests do not already lock it.
- [x] Pass `early_stopping_timeout` equal to the declared FEDOT minute budget and record `early_stopping_timeout` separately from `composing_did_not_start`. A valid metric stays `ok`.
- [ ] Read `execution/process.py` watchdog versus task timeout and add a missing distinction if the tests do not already lock it.
- [ ] Compare diagnose heartbeat staleness with the README claim that a missing optimizer marker is not a deadlock.
