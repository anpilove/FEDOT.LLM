# Setting one hyperparameter of a pair makes the operation raise: the other is read as None and passed to sklearn explicitly

## Summary

Several operations collect a fixed pair of parameters with `.get()` and forward both to sklearn. Declaring only one means the other arrives as an explicit `None`, overriding sklearn's own default.

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

# `degree` is declared tunable for poly_features; set it alone:
pipeline = PipelineBuilder().add_node('poly_features', params={'degree': 3}).add_node('ridge').build()
pipeline.fit(train_data)
# InvalidParameterError: The 'interaction_only' parameter of PolynomialFeatures
#   must be an instance of 'bool' ... Got None instead.
```

## Expected

The unset member of the pair keeps sklearn's default.

## Actual

`InvalidParameterError` from inside sklearn.

Reproduced against `310061e` at report time; the check above exits non-zero:
```text
sklearn.utils._param_validation.InvalidParameterError: The 'interaction_only' parameter of PolynomialFeatures must be an instance of 'bool' or an instance of 'numpy.bool_'. Got None instead.
```

## Scope

`poly_features` (`degree` / `interaction_only`, `sklearn_transformations.py:224`) and the four `rfe_*` operations (`n_features_to_select` / `step`, `sklearn_selectors.py`).

Note the tuner does not hit this — it always sets both members. It is the user setting a single hyperparameter by hand who is affected.

## Why it matters

A documented, declared-tunable parameter cannot be used on its own.

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
