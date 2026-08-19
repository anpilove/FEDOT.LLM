# Tuning CatBoost never succeeds: search space and default params declare the same settings under different names

## Summary

`PipelineSearchSpace` declares `iterations`, `border_count` and `max_leaves` tunable for `catboost` / `catboostreg`, while `default_operation_params.json` sets `num_trees`, `max_bin` and `grow_policy: SymmetricTree` for the same operations. CatBoost rejects each of those three pairs outright, so every tuning candidate fails to fit.

## Reproduction
```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

_rng = np.random.default_rng(42)
_n = 120
_x = _rng.normal(size=(_n, 6))
train_data = InputData(idx=np.arange(_n), features=_x,
                       target=(_x[:, 0] + 0.5 * _x[:, 1] > 0).astype(int).reshape(-1, 1),
                       task=Task(TaskTypesEnum.classification),
                       data_type=DataTypesEnum.table)
_t = np.arange(200)
_series = np.sin(_t / 7.0) * 10 + _t * 0.05 + _rng.normal(size=200) * 0.2
ts_data = InputData(idx=_t, features=_series, target=_series,
                    task=Task(TaskTypesEnum.ts_forecasting,
                              TsForecastingParams(forecast_length=5)),
                    data_type=DataTypesEnum.ts)
```
```python
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

# `iterations`, `border_count` and `max_leaves` are declared tunable ...
declared = set(PipelineSearchSpace().parameters_per_operation["catboost"])
print(sorted(declared & {"iterations", "border_count", "max_leaves"}))

# ... while default_operation_params.json sets num_trees / max_bin /
# grow_policy=SymmetricTree for the same operation. CatBoost rejects each pair.
pipeline = PipelineBuilder().add_node("catboost").build()
tuner = (TunerBuilder(Task(TaskTypesEnum.classification))
         .with_tuner(SimultaneousTuner)
         .with_metric(ClassificationMetricsEnum.ROCAUC)
         .with_iterations(6)
         .build(train_data))
tuner.tune(pipeline)
print("obtained_metric:", tuner.obtained_metric)   # None — every candidate failed to fit
```

## Expected

The tuner explores the declared space and returns a metric.

## Actual

Every candidate fails with `CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized` (likewise `border_count` vs `max_bin`, and `max_leaves` only working with `grow_policy=Lossguide`). `obtained_metric` is `None`, and the tuner logs *"Return init graph due to the fact that obtained metric is None"* and hands back the untuned pipeline. Nothing surfaces to the caller: tuning silently does nothing.

Reproduced against `310061e` at report time; the check above exits non-zero:
```text
AssertionError: still conflicting: [('iterations', 'num_trees'), ('border_count', 'max_bin')]
```

## Scope

`catboost`, `catboostreg`. Control: the same harness on `rf` returns a real metric.

## Why it matters

A user who tunes CatBoost gets default hyperparameters and no error. The fix is a change to two declarations, not to any algorithm.

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
