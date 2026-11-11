# Offline early screen of archived quality pairs

The screen triple is fixed from registry order before any pair status is read:

- classification: `openml-31-credit-g`
- regression: `openml-regression-moneyball`
- forecasting: `fedot-ts-beer`

A proposal is rejected early only when one of those three pairs has status `regression`. Status `timeout` and a missing metric (`missing`, `missing_metric`, `not_measured`, `invalid`) are counted separately and do not reject. Status `improved` or `ok` on the triple does not KEEP. KEEP requires an improved pair on all 40 registered datasets.

`pairs_saved` is `37` for each early rejection: the other registered datasets would not be run. `disagreements` lists two divergences from the full reading. `false_early_reject` means the screen rejected a proposal whose full reading is not a regression. `screen_missed_regression` means the regression sits outside the triple, so the screen continues and the full reading drops the proposal.

## Command

The archive is a JSON object. Each item of `pairs` has `proposal`, `task_id`, `problem`, and `status`.

```bash
uv run python -m fedotllm.agents.evolve early-screen \
  --pairs path/to/archive.json \
  --output early-screen-report.json
```

## Tests

`uv run ruff check` on the screen module, the evolve CLI, and the new tests: all checks passed.

`uv run pytest tests/unit/agents/test_evolve_early_screen.py -q`: 4 passed.

The tests use synthetic pairs on the real 40-task registry:

- `gain`: every dataset improved. Early decision is `continue`. Full decision is `keep`. No pairs saved.
- `screen-regression`: regression on `openml-31-credit-g`. Early and full decisions are `regression`. 37 pairs saved.
- `late-regression`: regression outside the triple. Early decision stays `continue`. Full decision is `regression`. This is the recorded disagreement, and no pairs are saved.
- `timeout`: timeout on `fedot-ts-beer`. Counted as a timeout pair. Not an early rejection and not KEEP.
- `missing`: `missing_metric` outside the triple. Counted separately. Full decision is not KEEP.
