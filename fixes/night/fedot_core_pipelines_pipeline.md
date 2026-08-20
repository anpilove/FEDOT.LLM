# Calling `fit` on an empty pipeline currently crashes with an unhelpful `AttributeError` because `_fit` dereferences `self.root_node` when it is `None`.

**Файл:** `fedot/core/pipelines/pipeline.py`

**Почему:** Severity class 3 — this is a public-interface failure that currently raises an unclear low-level exception on invalid input; adding an explicit guard is safe, preserves signatures/return types, and produces a clear `ValueError` explaining that a pipeline must contain at least one root node before fitting.

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


def test_verified_fedot_core_pipelines_pipeline_py_117():
    """live | `self.root_node` can be `None` for empty pipeline; accessing `.fitted_operation` raises `AttributeError` if `_fit` is called on an empty pipeline."""
    try:
        from fedot.core.pipelines.pipeline import Pipeline
        p = Pipeline()
        p.fit(train_data)
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
diff --git a/fedot/core/pipelines/pipeline.py b/fedot/core/pipelines/pipeline.py
index 9612252..6ae1298 100644
--- a/fedot/core/pipelines/pipeline.py
+++ b/fedot/core/pipelines/pipeline.py
@@ -113,9 +113,13 @@ class Pipeline(GraphDelegate, Serializable):
             in case of the time controlled call
         """
 
+        root_node = self.root_node
+        if root_node is None:
+            raise ValueError(f'{ERROR_PREFIX} Pipeline must contain at least one root node to be fitted')
+
         with Timer() as t:
-            computation_time_update = not self.root_node.fitted_operation or self.computation_time is None
-            train_predicted = self.root_node.fit(
+            computation_time_update = not root_node.fitted_operation or self.computation_time is None
+            train_predicted = root_node.fit(
                 input_data=input_data, predictions_cache=predictions_cache, fold_id=fold_id)
             if computation_time_update:
                 self.computation_time = round(t.minutes_from_start, 3)
```
