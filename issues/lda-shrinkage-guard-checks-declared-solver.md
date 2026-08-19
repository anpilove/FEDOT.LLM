# LDA: the guard that should suppress `shrinkage` on the `svd` solver never fires, because it tests the declared solver, not the effective one

## Summary

`LDAImplementation.check_and_correct_params` reads `self.params.get('solver')`. `lda` has no `solver` entry in `default_operation_params.json`, so for a caller who sets only `shrinkage` the value is `None`, `is_solver_svd` is False, and nothing is corrected — while sklearn's own default solver is `svd`, which is exactly the combination the guard exists to prevent.

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

pipeline = PipelineBuilder().add_node('lda', params={'shrinkage': 0.5}).build()
pipeline.fit(train_data)
# NotImplementedError: shrinkage not supported with 'svd' solver.
```

## Expected

Either the shrinkage is honoured (by moving to a solver that supports it) or it is ignored, as the guard intends.

## Actual

`NotImplementedError` from sklearn reaches the caller.

Reproduced against `310061e` at report time; the check above exits non-zero:
```text
NotImplementedError: shrinkage not supported with 'svd' solver.
```

## Scope

`lda`; `discriminant_analysis.py:78`.

## Why it matters

`shrinkage` is declared tunable, so every value in its declared scope raises. The condition needs the effective solver rather than the explicitly declared one.

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
