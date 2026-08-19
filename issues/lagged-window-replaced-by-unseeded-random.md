# `lagged` replaces an out-of-range window_size with an unseeded random value, so time-series pipelines are not reproducible by default

## Summary

When the declared `window_size` exceeds `len(series) - forecast_length - 1`, `ts_transformations.py:131` picks `int(random() * max_allowed_window_size)`. The requested value is discarded entirely, and `random()` reads the global `random` module state, which FEDOT seeds only from `Fedot(seed=...)` — default `None`.

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

for _ in range(6):
    p = PipelineBuilder().add_node('lagged', params={'window_size': 252}).add_node('ridge').build()
    node = next(n for n in p.nodes if n.name == 'lagged')
    p.fit(ts_data_of_200_points)
    print(node.parameters['window_size'])
# 158, 97, 48, 111, 149, 22 — a different model each run
```

## Expected

Either clamping to the maximum allowed window (as the branch two lines above already does for the other direction), or at least a reproducible choice.

## Actual

A uniformly random window anywhere in the legal range, differing between runs of identical code.

Reproduced against `310061e` at report time; the check above exits non-zero:
```text
AssertionError: window_size varies between runs: {113, 20, 193, 150}
```

## Scope

`lagged`, `sparse_lagged`.

## Why it matters

Passing `Fedot(seed=...)` does fix it — with `random.seed(0)` all six runs give 163 — so the honest statement is that time-series pipelines are not reproducible out of the box.

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
