# What the runtime scan found in FEDOT

Scanned: **65** operations fitted successfully with their own defaults; **14** could not be set up (missing optional dependency or no working pipeline) and are excluded rather than counted as clean.

Every value tried came from the operation's own declared sampling scope (`PipelineSearchSpace`), so nothing here is an input the library calls illegal.

## 1. A declared hyperparameter is silently replaced during fit

**12 operations.** The caller passes a value through the public API, the fitted object holds another one. `PipelineNode.parameters` reports the replacement too, so it is not merely internal.

| operation | parameter | declared | in force after fit |
|---|---|---|---|
| `ar` | `lag_1` | 101.0 | 99 |
| `ar` | `lag_1` | 200.0 | 99 |
| `ar` | `lag_2` | 401.0 | 99 |
| `ar` | `lag_2` | 800.0 | 99 |
| `cut` | `cut_part` | 0.0 | 0.5; 0.5 |
| `dask_pca` | `n_components` | 10 | 6 |
| `dask_pca` | `n_components` | 20 | 6 |
| `diff_filter` | `poly_degree` | 1 | 2; 2 |
| `diff_filter` | `window_size` | 11.5 | 11 |
| `ets` | `error` | 'mul' | 'add'; 'add' |
| `ets` | `trend` | 'mul' | 'add'; 'add' |
| `ets` | `seasonal` | 'mul' | 'add'; 'add' |
| `ets` | `damped_trend` | True | False |
| `ets` | `seasonal_periods` | 2.0 | None; 1 |
| `ets` | `seasonal_periods` | 51.0 | None; 1 |
| `ets` | `seasonal_periods` | 100.0 | None; 1 |
| `fast_ica` | `n_components` | 10 | 6 |
| `fast_ica` | `n_components` | 20 | 6 |
| `kernel_pca` | `n_components` | 10 | 6 |
| `kernel_pca` | `n_components` | 20 | 6 |
| `lagged` | `window_size` | 252 | 192; 192 |
| `lagged` | `window_size` | 500 | 193; 193 |
| `polyfit` | `degree` | 6 | 3; 3 |
| `ransac_lin_reg` | `residual_threshold` | 0.1 | 0.283473813533783; 0.283473813533783; 0.283473813533783 |
| `resample` | `replace` | False | True; True |
| `sparse_lagged` | `window_size` | 252 | 134; 134 |
| `sparse_lagged` | `window_size` | 500 | 94; 94 |
| `sparse_lagged` | `use_svd` | True | False |

### Why this is not cosmetic

**13 operations lose the operations cache entirely.** `PipelineNode.descriptive_id` is the cache key and embeds the node's parameters *and its parents'*. A node that rewrites its own parameters during fit is stored under a key nobody will ever look up — and every node downstream of it misses too.

- `ar`: pipeline `ar` — nothing reloaded, missed: ar
- `catboost`: pipeline `catboost` — nothing reloaded, missed: catboost
- `catboostreg`: pipeline `catboostreg` — nothing reloaded, missed: catboostreg
- `cut`: pipeline `cut` — nothing reloaded, missed: cut
- `dask_pca`: pipeline `rf → dask_pca` — nothing reloaded, missed: rf, dask_pca
- `diff_filter`: pipeline `diff_filter` — nothing reloaded, missed: diff_filter
- `ets`: pipeline `ets` — nothing reloaded, missed: ets
- `fast_ica`: pipeline `rf → fast_ica` — nothing reloaded, missed: rf, fast_ica
- `kernel_pca`: pipeline `rf → kernel_pca` — nothing reloaded, missed: rf, kernel_pca
- `lagged`: pipeline `lagged` — nothing reloaded, missed: lagged
- `polyfit`: pipeline `polyfit` — nothing reloaded, missed: polyfit
- `ransac_lin_reg`: pipeline `ridge → ransac_lin_reg` — nothing reloaded, missed: ridge, ransac_lin_reg
- `sparse_lagged`: pipeline `sparse_lagged` — nothing reloaded, missed: sparse_lagged

### Where in the source

`self.params.update(...)` inside fit, found by grep — the scan is what says which of these actually fire on legal data.

44 call sites in 12 files:
- `fedot/api/api_utils/api_composer.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_filters.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_imbalanced_class.py` — 4
- `fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py` — 6
- `fedot/core/operations/evaluation/operation_implementations/data_operations/ts_transformations.py` — 9
- `fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py` — 9
- `fedot/core/operations/evaluation/operation_implementations/models/discriminant_analysis.py` — 2
- `fedot/core/operations/evaluation/operation_implementations/models/keras.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/models/knn.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/models/svc.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/poly.py` — 1
- `fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py` — 8

## 2. A value from the declared scope crashes or hangs

**55 crashes, 0 hangs.**

| operation | parameter | value | error | dies outside FEDOT |
|---|---|---|---|---|
| `catboost` | `iterations` | 500 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboost` | `iterations` | 5250 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboost` | `iterations` | 10000 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboost` | `max_leaves` | 1 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboost` | `max_leaves` | 50 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboost` | `max_leaves` | 100 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboost` | `border_count` | 1 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboost` | `border_count` | 32768 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboost` | `border_count` | 65535 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `iterations` | 500 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `iterations` | 500 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `iterations` | 5250 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `iterations` | 5250 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `iterations` | 10000 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `iterations` | 10000 | CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized. | core.py |
| `catboostreg` | `max_leaves` | 1 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `max_leaves` | 1 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `max_leaves` | 50 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `max_leaves` | 50 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `max_leaves` | 100 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `max_leaves` | 100 | CatBoostError: catboost/private/libs/options/catboost_options.cpp:999: max_leaves option works only with lossguide tree growi | _catboost.pyx |
| `catboostreg` | `border_count` | 1 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `border_count` | 1 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `border_count` | 32768 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `border_count` | 32768 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `border_count` | 65535 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `catboostreg` | `border_count` | 65535 | CatBoostError: only one of the parameters border_count, max_bin should be initialized. | core.py |
| `gbr` | `loss` | 'ls' | InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'absolute_error', 'squared_ | _param_validation.py |
| `gbr` | `loss` | 'ls' | InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'absolute_error', 'squared_ | _param_validation.py |
| `gbr` | `loss` | 'lad' | InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'absolute_error', 'squared_ | _param_validation.py |
| `gbr` | `loss` | 'lad' | InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'absolute_error', 'squared_ | _param_validation.py |
| `lda` | `shrinkage` | 0.1 | NotImplementedError: shrinkage not supported with 'svd' solver. | discriminant_analysis.py |
| `lda` | `shrinkage` | 0.5 | NotImplementedError: shrinkage not supported with 'svd' solver. | discriminant_analysis.py |
| `lda` | `shrinkage` | 0.9 | NotImplementedError: shrinkage not supported with 'svd' solver. | discriminant_analysis.py |
| `lgbmreg` | `objective` | 'poisson' | LightGBMError: [poisson]: at least one target label is negative | basic.py |
| `lgbmreg` | `objective` | 'poisson' | LightGBMError: [poisson]: at least one target label is negative | basic.py |
| `stl_arima` | `d` | 1 | LinAlgError: LU decomposition error. | _tools.pyx |
| `stl_arima` | `period` | 1 | ValueError: period must be a positive integer >= 2 | _stl.pyx |
| `poly_features` | `degree` | 2 | InvalidParameterError: The 'interaction_only' parameter of PolynomialFeatures must be an instance of 'bool' or an instance of 'numpy. | _param_validation.py |
| `poly_features` | `degree` | 3 | InvalidParameterError: The 'interaction_only' parameter of PolynomialFeatures must be an instance of 'bool' or an instance of 'numpy. | _param_validation.py |
| `poly_features` | `degree` | 5 | InvalidParameterError: The 'interaction_only' parameter of PolynomialFeatures must be an instance of 'bool' or an instance of 'numpy. | _param_validation.py |
| `poly_features` | `interaction_only` | True | InvalidParameterError: The 'degree' parameter of PolynomialFeatures must be an int in the range [0, inf) or an array-like. Got None i | _param_validation.py |
| `poly_features` | `interaction_only` | False | InvalidParameterError: The 'degree' parameter of PolynomialFeatures must be an int in the range [0, inf) or an array-like. Got None i | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.5 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.5 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.7 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.7 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.9 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_lin_reg` | `n_features_to_select` | 0.9 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.5 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.5 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.7 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.7 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.9 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |
| `rfe_non_lin_reg` | `n_features_to_select` | 0.9 | InvalidParameterError: The 'step' parameter of RFE must be an int in the range (0, inf) or a float in the range (0.0, 1.0). Got None  | _param_validation.py |

## 2b. The replacement is not even reproducible

**4 cases.** The operation replaced the declared value, and fitting the same pipeline on the same data a second time produced a *different* replacement. Checked with the value that triggers the replacement rather than with the defaults, because the substituting branch does not run otherwise. Nothing is reseeded between the two fits: FEDOT seeds the global RNG only from `Fedot(seed=...)`, which defaults to `None`.

- `lagged`.`window_size` = 252 → 192 then 178
- `lagged`.`window_size` = 500 → 193 then 101
- `sparse_lagged`.`window_size` = 252 → 134 then 157
- `sparse_lagged`.`window_size` = 500 → 94 then 185

## 3. Metamorphic properties

**1 violations at defect grade** (`repeat_fit` — the same call twice must give the same prediction).

- `resample`: repeat_fit — violated (max delta 0.49000000000000005)

## 4. Not a finding

`not_observable` — 1 parameters across 1 operations are accepted but appear nowhere on the fitted object. Most are renamed on the way through, so this is a lead, not a defect, and is listed here to keep it out of the counts above.

- `glm`: nested_space

## Operations that could not be scanned

- `cgru` — no working pipeline with default params
- `clstm` — no working pipeline with default params
- `multinb` — no working pipeline with default params
- `cnn` — no working pipeline with default params
- `custom` — no working pipeline with default params
- `tabpfn` — no working pipeline with default params
- `tabpfnreg` — no working pipeline with default params
- `topological_features` — no working pipeline with default params
- `text_clean` — no working pipeline with default params
- `cntvect` — no working pipeline with default params
- `tfidf` — no working pipeline with default params
- `word2vec_pretrained` — no working pipeline with default params
- `decompose` — no working pipeline with default params
- `class_decompose` — no working pipeline with default params
---

## Verified by hand on top of the scan

The scan says which operations misbehave. These three consequences were then
confirmed separately, because "a parameter is replaced" and "a user loses
something" are not the same claim.

### Tuning CatBoost in FEDOT never works

`PipelineSearchSpace` declares `iterations`, `border_count` and `max_leaves`
tunable for `catboost` and `catboostreg`. `default_operation_params.json` sets
`num_trees`, `max_bin` and `grow_policy: SymmetricTree` for the same operations.
CatBoost rejects each of those three pairs outright.

Measured with `SimultaneousTuner`, 6 iterations, 200 rows:

```
catboost: obtained_metric = None, init = -0.9549
          7 candidates, every one "Unsuccessful pipeline fit ...
          Exception <only one of the parameters border_count, max_bin
          should be initialized.>"
rf:       obtained_metric = -0.9724, init = -0.9724   (control: a real metric)
```

The tuner then logs *"Return init graph due to the fact that obtained metric is
None"* and hands back the untuned pipeline. Nothing surfaces to the caller. A
user tuning CatBoost gets default hyperparameters and no error.

This is the single most expensive finding here, and no linter can reach it: the
defect is a disagreement between two of the library's own JSON/py declarations,
each of which is valid on its own.

### Setting one of a pair of parameters breaks the operation

```python
poly_params = {k: self.params.get(k) for k in ['degree', 'interaction_only']}
PolynomialFeatures(include_bias=False, **poly_params)
```

`sklearn_transformations.py:224`. Declare only `degree`, and `interaction_only`
is read as `None` and passed explicitly, overriding sklearn's own default —
`InvalidParameterError`. The same three lines appear four more times in
`sklearn_selectors.py` for `n_features_to_select`/`step`.

Note what this is *not*: the tuner sets both members of the pair at once, so it
never trips. This one costs the user who sets a single hyperparameter by hand.

### The operations cache, not just the parameter

`PipelineNode.descriptive_id` is the operations-cache key and embeds the node's
parameters and its parents'. Measured on `ransac_lin_reg → ridge`: after fit,
storing and immediately re-loading an identical pipeline returns **nothing** —
neither node. The control (`pca → ridge`) reloads both.

FEDOT already carries a workaround for the neighbouring symptom —
`predictions_cache.py:27` skips any node whose id contains `"ransac"`, with
`# TODO: issue#1363`, and `test/unit/cache/test_predictions_cache.py` pins that
behaviour. The operations cache has no such workaround and is not covered by a
test.

A correction to an earlier note in this project: the mutation **is** visible
through `PipelineNode.parameters` and it **does** change `descriptive_id`. The
earlier "descriptive_id is not affected" was measured on data where RANSAC's
retry loop never ran.

### A guard that was written for this exact case and misses it

`discriminant_analysis.py:78`:

```python
current_solver = self.params.get('solver')
is_solver_svd = current_solver is not None and current_solver == 'svd'
if is_solver_svd and current_shrinkage is not None:
    self.params.update(shrinkage=None)   # ignore shrinkage
```

`lda` has no `solver` entry in `default_operation_params.json`, so a caller who
sets only `shrinkage` leaves `current_solver` as `None`, the guard evaluates to
`False`, and nothing is corrected. sklearn's own default solver is `'svd'`,
which is precisely the combination the guard exists to prevent —
`NotImplementedError: shrinkage not supported with 'svd' solver` reaches the
caller from inside sklearn.

The condition tests for an *explicitly declared* solver where it needs the
*effective* one. Static analysis cannot see this: the code is well-formed and
the bug is a disagreement with a default that lives in another library.

### It happens with the operations' own defaults

Four operations lose the operations cache without the caller touching anything:
`catboost`, `catboostreg`, `lagged`, `sparse_lagged`. The last pair matters most
— `lagged` underpins nearly every time-series pipeline FEDOT builds. It fits
`window_size` to the length of the series and writes the result back into its
parameters, so a `lagged` node is **never** restored from the cache and the
composer re-fits it on every evaluation.

### `lagged` replaces an out-of-range window with a random one

`ts_transformations.py:131`, reached whenever the declared `window_size` exceeds
`len(series) - forecast_length - 1`:

```python
new = int(random() * max_allowed_window_size)   # from random import random
```

Two separate problems in one line.

*It ignores what was asked for.* A caller who declares 252 on a 200-point series
gets a uniformly random value anywhere in the legal range — measured across six
runs of a `lagged → ridge` pipeline: 158, 97, 48, 111, 149, 22. (Section 2b
above reports different numbers for the same defect: it fits `lagged` alone, and
the value is random either way.) The branch two lines above, for the other
direction, clamps instead; clamping is the obvious correction here too.

*It is not reproducible by default.* `random()` reads the global `random`
module state, which FEDOT seeds only from `Fedot(seed=...)`, and that parameter
defaults to `None`. With `random.seed(0)` set by hand the same six runs all give
163, so the honest statement is: **time-series pipelines are not reproducible
out of the box, and passing a seed does fix it.**
