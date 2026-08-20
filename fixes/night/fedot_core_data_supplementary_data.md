# Accessing `flow_mask` raises `AttributeError` when `features_mask` is `None`, even though `features_mask` is declared optional.

**Файл:** `fedot/core/data/supplementary_data.py`

**Почему:** Severity class 1 — this is a behavioural defect on valid input because `features_mask` is explicitly typed and defaulted as optional, yet the property unconditionally calls `.get()` on it. Returning `None` is safe, preserves the property return shape already implied by absent data, and fixes the public API failure without changing call signatures.

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


def test_verified_fedot_core_data_supplementary_data_py_48():
    """live | AttributeError when `features_mask` is None, because `.get()` is called on None"""
    try:
        _ = train_data.supplementary_data.flow_mask
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
diff --git a/fedot/core/data/supplementary_data.py b/fedot/core/data/supplementary_data.py
index 77943a2..eef6156 100644
--- a/fedot/core/data/supplementary_data.py
+++ b/fedot/core/data/supplementary_data.py
@@ -44,7 +44,9 @@ class SupplementaryData:
         return comp_list
 
     @property
-    def flow_mask(self) -> list:
+    def flow_mask(self) -> Optional[list]:
+        if self.features_mask is None:
+            return None
         return self.features_mask.get('flow_lens')
 
     def define_parents(self, unique_features_masks: np.array, task: TaskTypesEnum):
```
