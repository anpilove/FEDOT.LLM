# Invalid public `problem` names currently raise an uninformative `KeyError` instead of a clear `ValueError`.

**Файл:** `fedot/api/api_utils/params.py`

**Почему:** Severity class 3 — this is a proven unclear-error defect on invalid input in a public API. The current `except ValueError` can never catch missing dict keys, and the handler also forgets to `raise`, so invalid problems leak a raw `KeyError`. Replacing this with explicit membership validation is safe, keeps the return type unchanged for valid inputs, and produces a helpful error naming the invalid value and accepted options.

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


def test_verified_fedot_api_api_utils_params_py_95():
    """except ValueError never triggers because dict access raises KeyError; wrong exception type and missing raise cause uninformative error for invalid problem."""
    try:
        from fedot import Fedot

        model = Fedot(problem='invalid_problem')
    except KeyError:
        raise AssertionError(
            "still raises KeyError — the defect this test was built from"
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
diff --git a/fedot/api/api_utils/params.py b/fedot/api/api_utils/params.py
index 5107963..f47acbe 100644
--- a/fedot/api/api_utils/params.py
+++ b/fedot/api/api_utils/params.py
@@ -92,10 +92,11 @@ class ApiParams(UserDict):
                      'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                      'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                      }
-        try:
-            return task_dict[problem]
-        except ValueError:
-            ValueError('Wrong type name of the given task')
+        if problem not in task_dict:
+            available_problems = ', '.join(task_dict.keys())
+            raise ValueError(f'Wrong type name of the given task: {problem!r}. '
+                             f'Expected one of: {available_problems}.')
+        return task_dict[problem]
 
     def _check_timeout_vs_generations(self):
         num_of_generations = self.get('num_of_generations')
```
