# Operations that rewrite their own parameters during fit lose the operations cache (and take downstream nodes with them)

## Summary

Twelve operations replace a declared hyperparameter during `fit` and write the replacement back into `self.params` (via `self.params.update(...)`). `PipelineNode.descriptive_id` embeds those parameters and is the operations-cache key, so the fitted node is stored under an id nobody will look up — and every node downstream of it misses too, because their ids embed the parent's parameters.

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
from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

def build():
    return PipelineBuilder().add_node("lagged").add_node("ridge").build()

cache = OperationsCache()
fitted = build()
fitted.fit(train_data)          # `lagged` fits window_size to the series ...
cache.save_pipeline(fitted)     # ... and writes it back into its own parameters

fresh = build()
cache.try_load_into_pipeline(fresh)
print([n.name for n in fresh.nodes if n.fitted_operation is not None])   # []
```

## Expected

An identical pipeline reloads both nodes from the cache.

## Actual

Nothing is reloaded — neither `lagged` nor `ridge`. Control: `pca -> ridge` reloads both.

Reproduced against `310061e` at report time; the check above exits non-zero:
```text
AssertionError: []
```

## Scope

Measured on 13 operations: `ar`, `catboost`, `catboostreg`, `cut`, `dask_pca`, `diff_filter`, `ets`, `fast_ica`, `kernel_pca`, `lagged`, `polyfit`, `ransac_lin_reg`, `sparse_lagged`. Four of them — `catboost`, `catboostreg`, `lagged`, `sparse_lagged` — do it with their own defaults, without the caller setting anything.

## Why it matters

`lagged` underpins nearly every time-series pipeline FEDOT composes, so it is re-fitted on every evaluation. The neighbouring symptom is already known: `predictions_cache.py` skips any node whose id contains "ransac", with `# TODO: issue#1363`. The operations cache has no such workaround and no test.

A fix that keeps the adaptive behaviour is small: let the operation adapt a private attribute and leave `self.params` holding what the caller declared.

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
