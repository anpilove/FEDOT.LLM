# Fedot.get_metrics()` dereferences `self.prediction.predict` before any prediction was made, causing an unhelpful `AttributeError`.

**Файл:** `fedot/api/main.py`

**Почему:** Severity class 3 — this is a public-API unclear error on valid object state; adding an explicit guard is safe, preserves the method signature, and converts an internal `AttributeError` into a clear `ValueError` explaining that metrics need a prior prediction or explicit test target setup.

## Доказательство — падает до патча

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


def test_verified_fedot_api_main_py_452():
    """live | `get_metrics` accesses `self.prediction.predict` without checking if `self.prediction` is None, causing AttributeError when called before `predict`."""
    try:
        from fedot.api.main import Fedot

        model = Fedot(problem='classification', timeout=1, seed=42)
        model.fit(features=train_data.features, target=train_data.target)
        model.get_metrics(target=train_data.target)
    except AttributeError:
        raise AssertionError(
            "still raises AttributeError — the defect this test was built from"
        ) from None
    except (ValueError, TypeError) as exc:
        assert str(exc).strip(), "replacement validation error must explain the problem"
    except Exception as exc:
        raise AssertionError(
            f"unexpected replacement failure: {type(exc).__name__}: {exc}"
        ) from exc
```

## Патч

```diff
diff --git a/fedot/api/main.py b/fedot/api/main.py
index f9a998f..f115561 100644
--- a/fedot/api/main.py
+++ b/fedot/api/main.py
@@ -447,6 +447,9 @@ class Fedot:
         if self.current_pipeline is None:
             raise ValueError(NOT_FITTED_ERR_MSG)
 
+        if self.prediction is None:
+            raise ValueError('Prediction is not available. Call predict() before get_metrics().')
+
         if target is not None:
             if self.test_data is None:
                 self.test_data = InputData(idx=np.arange(len(self.prediction.predict)),
```
