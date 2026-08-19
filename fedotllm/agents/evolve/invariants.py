"""Detection of behavioural defects in FEDOT operations without an LLM.

Three deterministic probes, all driven by the library's own declarations
(``PipelineSearchSpace`` + ``OperationTypesRepository``), so nothing here is
guessed by a model:

``declared == used``
    A hyperparameter is passed through the public path
    (``PipelineNode(params=...)``), the pipeline is fitted, and the value the
    fitted object actually carries is read back.  A mismatch means the library
    accepted a value and silently worked with another one.  This generalises
    the single confirmed defect we know (RANSAC, issue #1363).

``boundary``
    Every value is taken from the operation's *own* declared sampling scope,
    including both ends of the interval and every categorical choice.  A crash,
    a hang or a foreign library's internal traceback on such a value is a
    defect: the library declared the value legal itself.

``metamorphic``
    Properties that must hold for any sane estimator: permuting feature
    columns, appending a constant column, shuffling rows, or duplicating the
    sample must not change predictions (up to tolerance) at a fixed seed.

Two more pieces exist because of patches that satisfied a finding without
fixing anything:

``tuning_check``
    When the finding says a parameter cannot be used, the operation has to end
    up tunable.  An agent once "fixed" a CatBoost parameter conflict by no
    longer passing the parameter, which cleared every other gate and turned a
    loud error into a silent one.

the behaviour fingerprint in ``build_defect_test``
    Pinning the declared value alone is satisfied most cheaply by deleting the
    correction that replaced it.  The generated test therefore also pins a
    checksum of the predictions taken on the untouched checkout -- and only
    when that checksum reproduces in a fresh process, twice, because some
    operations draw from the global RNG and have no stable value to pin.

The module is executed by the FEDOT checkout's own interpreter
(``FEDOTLLM_REPO_PYTHON``); the driver in :mod:`fedotllm.agents.evolve.loop`
only consumes the JSON it prints.  Each operation runs in its own subprocess so
a hang or a hard crash is a *result*, not a lost run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

SEED = 42


def _np():
    import numpy as np

    return np


def make_data(kind: str):
    """Small, entirely legal datasets.  Nothing adversarial lives here --
    awkward inputs are the job of ``probe.py``; here we only need something the
    operation can be fitted on."""
    np = _np()
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    rng = np.random.default_rng(SEED)
    if kind == "classification":
        n = 120
        x = rng.normal(size=(n, 6))
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        task = Task(TaskTypesEnum.classification)
        return InputData(idx=np.arange(n), features=x, target=y.reshape(-1, 1),
                         task=task, data_type=DataTypesEnum.table)
    if kind == "regression":
        n = 120
        x = rng.normal(size=(n, 6))
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=n) * 0.3
        task = Task(TaskTypesEnum.regression)
        return InputData(idx=np.arange(n), features=x, target=y.reshape(-1, 1),
                         task=task, data_type=DataTypesEnum.table)
    if kind == "regression_outliers":
        # RANSAC-style filters only take their retry branch when the plain fit
        # leaves too few inliers, so one dataset with heavy contamination is
        # needed to reach that code at all.
        n = 120
        x = rng.normal(size=(n, 6))
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=n) * 0.3
        idx = rng.choice(n, size=n // 3, replace=False)
        y[idx] += rng.normal(size=idx.size) * 50
        task = Task(TaskTypesEnum.regression)
        return InputData(idx=np.arange(n), features=x, target=y.reshape(-1, 1),
                         task=task, data_type=DataTypesEnum.table)
    if kind == "ts":
        n = 200
        t = np.arange(n)
        series = np.sin(t / 7.0) * 10 + t * 0.05 + rng.normal(size=n) * 0.2
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=5))
        return InputData(idx=t, features=series, target=series,
                         task=task, data_type=DataTypesEnum.ts)
    raise ValueError(kind)


DATA_FOR_TASK = {
    "classification": ["classification"],
    "regression": ["regression", "regression_outliers"],
    "ts_forecasting": ["ts"],
    "clustering": ["classification"],
}


# --------------------------------------------------------------------------- #
# pipeline templates
# --------------------------------------------------------------------------- #

def _meta(op_id: str):
    from fedot.core.repository.operation_types_repository import OperationTypesRepository

    for kind in ("model", "data_operation"):
        for op in OperationTypesRepository(kind).operations:
            if op.id == op_id:
                return kind, op
    return None, None


def candidate_chains(op_id: str, task_kind: str) -> List[List[str]]:
    """Node chains to try, cheapest first.  The first one that fits with the
    operation's own defaults becomes the harness for that operation."""
    kind, meta = _meta(op_id)
    if meta is None:
        return []
    from fedot.core.repository.dataset_types import DataTypesEnum

    out_table = DataTypesEnum.table in meta.output_types
    in_ts = DataTypesEnum.ts in meta.input_types
    tail_model = {"classification": "rf", "regression": "ridge",
                  "ts_forecasting": "ridge", "clustering": "kmeans"}[task_kind]

    chains: List[List[str]] = []
    if task_kind == "ts_forecasting":
        if in_ts:
            chains.append([op_id])
            if out_table and kind == "data_operation":
                chains.append([op_id, tail_model])
        else:
            chains.append(["lagged", op_id])
            chains.append(["lagged", op_id, tail_model])
    else:
        if kind == "model":
            chains.append([op_id])
        chains.append([op_id, tail_model])
    return chains


def build_and_fit(chain: List[str], params: Optional[Dict[str, Any]], data):
    """Fit ``chain`` with ``params`` applied to its *first* node (the operation
    under test) and return the fitted pipeline."""
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    builder = PipelineBuilder()
    for i, name in enumerate(chain):
        builder = builder.add_node(name, params=params if i == 0 else None)
    pipeline = builder.build()
    pipeline.fit(data)
    return pipeline


# --------------------------------------------------------------------------- #
# reading back the value actually in force
# --------------------------------------------------------------------------- #

def observed_values(node, name: str) -> Dict[str, Any]:
    """Every place the fitted node exposes ``name``.

    ``PipelineNode.parameters`` only echoes what was requested -- the RANSAC
    defect is invisible there -- so the fitted object and any estimator it
    wraps are read as well."""
    seen: Dict[str, Any] = {}
    fitted = getattr(node, "fitted_operation", None)
    if fitted is None:
        return seen

    impl_params = getattr(fitted, "params", None)
    if impl_params is not None and hasattr(impl_params, "to_dict"):
        d = impl_params.to_dict()
        if name in d:
            seen["implementation.params"] = d[name]

    # `.operation` and `.model` are the two names FEDOT wrappers use for the
    # estimator they delegate to. Reading them matters: `implementation.params`
    # only echoes what was declared, so a parameter can look correct there while
    # never reaching the estimator at all -- which is exactly what a patch that
    # "fixes" a conflict by dropping the parameter produces.
    for label, obj in (("fitted", fitted),
                       ("fitted.operation", getattr(fitted, "operation", None)),
                       ("fitted.model", getattr(fitted, "model", None))):
        if obj is None:
            continue
        if hasattr(obj, "get_params"):
            try:
                gp = obj.get_params(deep=False)
            except Exception:
                gp = {}
            if name in gp:
                seen[label + ".get_params"] = gp[name]
        if hasattr(obj, name):
            try:
                seen[label + "." + name] = getattr(obj, name)
            except Exception:
                pass
    return seen


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return a == b
    return a == b


# --------------------------------------------------------------------------- #
# values to try -- taken from the library's own declared sampling scope
# --------------------------------------------------------------------------- #

def candidate_values(spec: Dict[str, Any]) -> List[Any]:
    scope = spec.get("sampling-scope")
    kind = spec.get("type")
    if kind == "categorical":
        choices = scope[0] if scope and isinstance(scope[0], (list, tuple)) else scope
        return list(choices)[:5]
    if not scope or len(scope) < 2:
        return []
    lo, hi = scope[0], scope[1]
    if kind == "discrete":
        lo, hi = int(lo), int(hi)
        mid = int((lo + hi) // 2)
        return sorted({lo, mid, hi})
    lo, hi = float(lo), float(hi)
    mid = (lo + hi) / 2.0
    return sorted({lo, mid, hi})


# --------------------------------------------------------------------------- #
# metamorphic properties
# --------------------------------------------------------------------------- #

def metamorphic_checks(op_id: str, chain: List[str], data_kind: str) -> List[Dict[str, Any]]:
    """Properties that must hold for any sane estimator.

    Only ``repeat_fit`` is graded as a defect.  The other three are recorded as
    observations, and that grading is a measured result rather than caution:

    * duplicating the whole sample is not a no-op -- doubling the rows halves
      the relative weight of every penalty term, and ``ridge`` moves.  The
      property is not implemented at all for that reason.
    * permuting columns and appending a constant column both move ``rf`` by
      ~0.08 in predicted probability, because a forest draws ``max_features``
      columns at random per split: the column set changes, so the draw changes.
      That is documented sklearn behaviour, not a FEDOT defect.
    * row order legitimately matters to bootstrap and mini-batch estimators.

    A property that fires on correct behaviour is worse than no property, so
    these three are reported for a human to read, never as findings.
    """
    np = _np()
    out: List[Dict[str, Any]] = []
    if data_kind not in ("classification", "regression", "ts"):
        return out

    base = make_data(data_kind)
    try:
        ref = np.asarray(build_and_fit(chain, None, base).predict(base).predict, dtype=float)
    except Exception:
        return out

    def compare(label: str, fit_data, predict_data=None, severity: str = "defect"):
        try:
            fitted = build_and_fit(chain, None, fit_data)
            got = np.asarray(fitted.predict(predict_data if predict_data is not None
                                            else fit_data).predict, dtype=float)
        except Exception as exc:
            out.append({"property": label, "severity": severity, "status": "error",
                        "error": type(exc).__name__ + ": " + str(exc)[:200]})
            return
        if ref.shape != got.shape:
            out.append({"property": label, "severity": severity, "status": "shape",
                        "detail": f"{ref.shape} != {got.shape}"})
            return
        delta = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        scale = float(np.max(np.abs(ref))) if ref.size else 1.0
        tol = 1e-6 * max(1.0, scale)
        out.append({"property": label, "severity": severity,
                    "status": "violated" if delta > tol else "ok",
                    "max_delta": delta, "tolerance": tol})

    from copy import deepcopy

    # 1. determinism: the very same call twice. Run for every data kind --
    #    time-series operations were originally skipped here, and that gap cost
    #    us: `lagged` picks a replacement window with an unseeded `random()`
    #    and had to be found by reading the source instead.
    compare("repeat_fit", make_data(data_kind))

    if data_kind == "ts":
        # The remaining properties are transformations of a feature table and
        # have no meaning for a single series.
        return out

    # 2. the model must not depend on which column is which
    perm = deepcopy(base)
    order = np.array([3, 1, 5, 0, 4, 2])
    perm.features = base.features[:, order]
    compare("column_permutation", perm, severity="observation")

    # 3. a column that never varies carries no information
    const = deepcopy(base)
    const.features = np.hstack([base.features, np.ones((base.features.shape[0], 1))])
    compare("constant_column", const, severity="observation")

    # 4. row order.  Reported as an observation, not as a defect: bootstrap and
    #    mini-batch estimators may legitimately depend on it even at a fixed seed.
    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(base.features.shape[0])
    shuf = deepcopy(base)
    shuf.features = base.features[order]
    shuf.target = base.target[order]
    shuf.idx = np.arange(shuf.features.shape[0])
    compare("row_shuffle", shuf, predict_data=base, severity="observation")

    return out


# --------------------------------------------------------------------------- #
# consequence: does the operations cache still work?
# --------------------------------------------------------------------------- #

def cache_check(chain: List[str], params: Optional[Dict[str, Any]], data) -> Dict[str, Any]:
    """Fit, store in the operations cache, then look the very same pipeline up.

    ``PipelineNode.descriptive_id`` is the cache key and it embeds the node's
    parameters *and* its parents'.  An operation that rewrites its own
    parameters during fit is therefore stored under a key nobody will ever ask
    for -- and it takes every node downstream of it with it, because their ids
    embed the parent's parameters too.  This turns a parameter mismatch from a
    cosmetic complaint into a measurable loss.
    """
    from fedot.core.caching.operations_cache import OperationsCache

    cache = OperationsCache()
    fitted = build_and_fit(chain, params, data)
    cache.save_pipeline(fitted)

    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    builder = PipelineBuilder()
    for i, name in enumerate(chain):
        builder = builder.add_node(name, params=params if i == 0 else None)
    fresh = builder.build()
    cache.try_load_into_pipeline(fresh)
    hits = [n.name for n in fresh.nodes if n.fitted_operation is not None]
    return {"nodes": [n.name for n in fresh.nodes], "hits": sorted(hits)}


# --------------------------------------------------------------------------- #
# worker: everything for one operation
# --------------------------------------------------------------------------- #

def run_operation(op_id: str, do_metamorphic: bool) -> Dict[str, Any]:
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace

    kind, meta = _meta(op_id)
    result: Dict[str, Any] = {"operation": op_id, "kind": kind,
                              "findings": [], "setup": None, "metamorphic": []}
    if meta is None:
        result["setup"] = "unknown operation"
        return result

    space = PipelineSearchSpace().parameters_per_operation.get(op_id, {})
    task_kinds = [t.value for t in meta.task_type]

    harness: Optional[Tuple[List[str], str]] = None
    for task_kind in task_kinds:
        for data_kind in DATA_FOR_TASK.get(task_kind, []):
            data = make_data(data_kind)
            for chain in candidate_chains(op_id, task_kind):
                try:
                    build_and_fit(chain, None, data)
                except Exception:
                    continue
                harness = (chain, data_kind)
                break
            if harness:
                break
        if harness:
            break

    if harness is None:
        result["setup"] = "no working pipeline with default params"
        return result

    chain, data_kind = harness
    result["chain"] = chain
    result["data"] = data_kind

    data_kinds = [data_kind]
    if data_kind == "regression":
        data_kinds.append("regression_outliers")

    for pname, spec in space.items():
        for value in candidate_values(spec):
            for dk in data_kinds:
                data = make_data(dk)
                try:
                    pipeline = build_and_fit(chain, {pname: value}, data)
                except Exception as exc:
                    result["findings"].append({
                        "kind": "boundary_crash", "param": pname, "value": value,
                        "data": dk, "declared_scope": spec.get("sampling-scope"),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                        "foreign": _is_foreign(exc),
                    })
                    continue
                node = [n for n in pipeline.nodes if n.name == op_id]
                if not node:
                    continue
                node = node[0]
                seen = observed_values(node, pname)
                if not seen:
                    result["findings"].append({
                        "kind": "not_observable", "param": pname, "value": value,
                        "data": dk})
                    continue
                bad = {k: v for k, v in seen.items() if not _equal(v, value)}
                if bad:
                    result["findings"].append({
                        "kind": "declared_not_used", "param": pname, "value": value,
                        "data": dk, "observed": {k: repr(v) for k, v in bad.items()},
                        "declared_echo": repr(node.parameters.get(pname))})

                    # Is the replacement at least reproducible? Determinism has to
                    # be checked with the value that triggers the replacement, not
                    # with the defaults -- the branch that picks a substitute may
                    # never run otherwise. `lagged` draws its substitute from an
                    # unseeded `random()` and only shows up here.
                    again = _refit_value(chain, {pname: value}, make_data(dk), op_id, pname)
                    first = node.parameters.get(pname)
                    if again is not _MISSING and not _equal(again, first):
                        result["findings"].append({
                            "kind": "nondeterministic_replacement", "param": pname,
                            "value": value, "data": dk,
                            "observed": [repr(first), repr(again)]})

    # Consequence check.  Run with the operation's own defaults, and once more
    # with each parameter value that was found to be rewritten -- a rewrite that
    # only happens for exotic values still costs the whole pipeline its cache.
    probe_params: List[Optional[Dict[str, Any]]] = [None]
    seen_params: set = set()
    for f in result["findings"]:
        if f["kind"] == "declared_not_used" and f["param"] not in seen_params:
            seen_params.add(f["param"])
            probe_params.append({f["param"]: f["value"]})
    for params in probe_params:
        for dk in data_kinds:
            try:
                report = cache_check(chain, params, make_data(dk))
            except Exception as exc:
                result.setdefault("cache", []).append(
                    {"params": params, "data": dk, "error": type(exc).__name__ + ": " + str(exc)[:200]})
                continue
            missed = [n for n in report["nodes"] if n not in report["hits"]]
            entry = {"params": params, "data": dk, **report, "missed": missed}
            result.setdefault("cache", []).append(entry)
            if missed:
                result["findings"].append({
                    "kind": "cache_miss_after_fit", "params": params, "data": dk,
                    "missed_nodes": missed, "chain": report["nodes"]})

    if do_metamorphic:
        result["metamorphic"] = metamorphic_checks(op_id, chain, data_kind)
    return result


_MISSING = object()


def _refit_value(chain: List[str], params: Dict[str, Any], data, op_id: str, pname: str):
    """Fit the same pipeline again and read the same parameter back.

    Nothing is reseeded between the two fits on purpose: the question is whether
    a caller who runs the same code twice gets the same model, and FEDOT seeds
    the global RNG only when `Fedot(seed=...)` is given, which defaults to None.
    """
    try:
        pipeline = build_and_fit(chain, params, data)
    except Exception:
        return _MISSING
    node = next((n for n in pipeline.nodes if n.name == op_id), None)
    if node is None:
        return _MISSING
    return node.parameters.get(pname, _MISSING)


def _is_foreign(exc: BaseException) -> Optional[str]:
    """True when the traceback dies outside FEDOT -- an internal error of a
    third-party library reaching the user unchanged."""
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return None
    last = tb[-1].filename
    if "/fedot/" in last.replace(os.sep, "/"):
        return None
    return last


# --------------------------------------------------------------------------- #
# acceptance: can the operation actually be tuned?
# --------------------------------------------------------------------------- #

def tuning_check(op_id: str, iterations: int = 6) -> Dict[str, Any]:
    """Tune one operation for a handful of iterations and report the metric.

    This exists because of a patch that passed every other gate. The agent was
    shown "`iterations` is declared tunable for catboost and every value in the
    scope raises CatBoostError", and it removed `iterations` from the arguments
    handed to CatBoost. The error disappeared, the test passed, the runtime
    probe stayed green -- and the caller's value is now discarded in silence,
    which is a worse defect than the loud one it replaced.

    No inspection of the plumbing catches that: `implementation.params` still
    reports the declared value, because only the estimator stopped receiving it.
    What does catch it is asking for the property the defect is actually about.
    Tuning still yields nothing (`border_count` and `max_leaves` collide too),
    so `obtained_metric` stays `None` and the patch is refused.
    """
    from golem.core.tuning.simultaneous import SimultaneousTuner

    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import (ClassificationMetricsEnum,
                                                          RegressionMetricsEnum)
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    kind, meta = _meta(op_id)
    if meta is None:
        return {"operation": op_id, "ok": False, "reason": "unknown operation"}
    task_kinds = [t.value for t in meta.task_type]
    task_kind = "classification" if "classification" in task_kinds else task_kinds[0]
    data_kind = DATA_FOR_TASK.get(task_kind, ["regression"])[0]
    metric = (ClassificationMetricsEnum.ROCAUC if task_kind == "classification"
              else RegressionMetricsEnum.RMSE)

    data = make_data(data_kind)
    chains = candidate_chains(op_id, task_kind)
    if not chains:
        return {"operation": op_id, "ok": False, "reason": "no pipeline"}

    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    builder = PipelineBuilder()
    for name in chains[0]:
        builder = builder.add_node(name)
    pipeline = builder.build()
    try:
        tuner = (TunerBuilder(Task(TaskTypesEnum(task_kind)))
                 .with_tuner(SimultaneousTuner)
                 .with_metric(metric)
                 .with_iterations(iterations)
                 .build(data))
        tuner.tune(pipeline)
    except Exception as exc:
        return {"operation": op_id, "ok": False,
                "reason": type(exc).__name__ + ": " + str(exc)[:200]}
    obtained = getattr(tuner, "obtained_metric", None)
    return {"operation": op_id, "ok": obtained is not None,
            "obtained_metric": None if obtained is None else float(obtained),
            "reason": "" if obtained is not None
            else "every tuning candidate failed to fit; the tuner returned the "
                 "initial pipeline without saying why"}


# --------------------------------------------------------------------------- #
# bridge: a finding -> a repo-relative file and a test that fails today
# --------------------------------------------------------------------------- #

# Docstrings inside the generated test are written as comments on purpose: the
# template itself lives in a triple-quoted string.
TEST_TEMPLATE = '''
import numpy as np

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _data():
{data_code}


def _pipeline():
    builder = PipelineBuilder()
{chain_code}
    return builder.build()


def {test_name}():
    # `{op}` accepts {param}={value!r}; that must still be the value in force
    # after fit. Anything else means the library silently worked with another.
    pipeline = _pipeline()
    node = next(n for n in pipeline.nodes if n.name == "{op}")
    pipeline.fit(_data())
    assert node.parameters["{param}"] == {value!r}, (
        "declared {param}={value!r}, node reports "
        + repr(node.parameters["{param}"]) + " after fit")


def {test_name}_cache():
    # `descriptive_id` is the operations-cache key and embeds the node's
    # parameters and its parents'. A node that rewrites its own parameters
    # during fit is stored under a key nobody looks up, and takes every node
    # downstream of it with it.
    cache = OperationsCache()
    fitted = _pipeline()
    fitted.fit(_data())
    cache.save_pipeline(fitted)

    fresh = _pipeline()
    cache.try_load_into_pipeline(fresh)
    hits = sorted(n.name for n in fresh.nodes if n.fitted_operation is not None)
    assert hits == sorted(n.name for n in fresh.nodes), (
        "identical pipeline missed the operations cache, hits: " + repr(hits))
'''

DATA_CODE = {
    "regression": (
        "    rng = np.random.default_rng(42)\n"
        "    n = 120\n"
        "    features = rng.normal(size=(n, 6))\n"
        "    target = 2 * features[:, 0] - features[:, 1] + rng.normal(size=n) * 0.3\n"
        "    return InputData(idx=np.arange(n), features=features,\n"
        "                     target=target.reshape(-1, 1),\n"
        "                     task=Task(TaskTypesEnum.regression),\n"
        "                     data_type=DataTypesEnum.table)"
    ),
    "regression_outliers": (
        "    rng = np.random.default_rng(42)\n"
        "    n = 120\n"
        "    features = rng.normal(size=(n, 6))\n"
        "    target = 2 * features[:, 0] - features[:, 1] + rng.normal(size=n) * 0.3\n"
        "    outliers = rng.choice(n, size=n // 3, replace=False)\n"
        "    target[outliers] += rng.normal(size=outliers.size) * 50\n"
        "    return InputData(idx=np.arange(n), features=features,\n"
        "                     target=target.reshape(-1, 1),\n"
        "                     task=Task(TaskTypesEnum.regression),\n"
        "                     data_type=DataTypesEnum.table)"
    ),
    "classification": (
        "    rng = np.random.default_rng(42)\n"
        "    n = 120\n"
        "    features = rng.normal(size=(n, 6))\n"
        "    target = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)\n"
        "    return InputData(idx=np.arange(n), features=features,\n"
        "                     target=target.reshape(-1, 1),\n"
        "                     task=Task(TaskTypesEnum.classification),\n"
        "                     data_type=DataTypesEnum.table)"
    ),
    "ts": (
        "    from fedot.core.repository.tasks import TsForecastingParams\n"
        "    rng = np.random.default_rng(42)\n"
        "    n = 200\n"
        "    t = np.arange(n)\n"
        "    series = np.sin(t / 7.0) * 10 + t * 0.05 + rng.normal(size=n) * 0.2\n"
        "    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=5))\n"
        "    return InputData(idx=t, features=series, target=series,\n"
        "                     task=task, data_type=DataTypesEnum.ts)"
    ),
}


CRASH_TEST_TEMPLATE = '''
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _data():
{data_code}


def {test_name}():
    # `{param}` is declared tunable for `{op}` in PipelineSearchSpace, with
    # {value!r} inside its own declared sampling scope. Fitting with it currently
    # dies inside {where} with:
    #   {error}
    # Every value in the scope fails the same way, so the parameter cannot be
    # used at all -- this is not a data-dependent edge case.
    builder = PipelineBuilder()
{chain_code}
    builder.build().fit(_data())
'''


def unusable_parameters(res: Dict[str, Any], space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parameters that crash for *every* value in their declared scope.

    The stricter rule is deliberate. `lgbmreg` crashes on ``objective='poisson'``
    because the targets happen to be negative, and `stl_arima` crashes on
    ``period=1`` alone -- both are about the data, not about the parameter being
    unusable. Requiring the whole scope to fail keeps those out.
    """
    crashes: Dict[str, List[Dict[str, Any]]] = {}
    for f in res.get("findings", []):
        if f.get("kind") == "boundary_crash":
            crashes.setdefault(f["param"], []).append(f)
    out = []
    for param, items in crashes.items():
        spec = space.get(param)
        if not spec:
            continue
        expected = candidate_values(spec)
        failed = {repr(i["value"]) for i in items}
        if expected and all(repr(v) in failed for v in expected):
            out.append(items[0])
    return out


GOLDEN_SNIPPET = '''

def {test_name}_behaviour_unchanged():
    # Pinning the declared value is not the whole requirement. The cheapest way
    # to satisfy that assertion alone is to delete the correction that replaced
    # the value -- measured: the agent commented out `self.params.update(...)`
    # in `PolyfitImplementation` and every other gate let it through, leaving an
    # out-of-range degree to reach `np.polyfit`.
    #
    # So the fingerprint below was taken on the untouched checkout, with the
    # correction in force. A patch that separates "what the caller declared"
    # from "what the operation works with" keeps it; a patch that removes the
    # correction changes it.
    pipeline = _pipeline()
    pipeline.fit(_data())
    predicted = np.asarray(pipeline.predict(_data()).predict, dtype=float)
    fingerprint = float(np.round(np.nansum(np.abs(predicted)), 6))
    assert fingerprint == {golden!r}, (
        "predictions changed: the adaptive behaviour must stay, only the "
        "bookkeeping of the declared value may change")
'''


def prediction_fingerprint(chain: List[str], params: Dict[str, Any], data_kind: str):
    """Stable summary of what the pipeline predicts, or None if it is not stable.

    Fitted twice on purpose: an operation that picks a substitute at random
    (`lagged` does) has no golden value to pin, and pinning one would produce a
    test that fails for reasons no patch can fix.
    """
    np = _np()
    seen = []
    for _ in range(2):
        try:
            pipeline = build_and_fit(chain, params, make_data(data_kind))
            predicted = np.asarray(pipeline.predict(make_data(data_kind)).predict,
                                   dtype=float)
        except Exception:
            return None
        seen.append(float(np.round(np.nansum(np.abs(predicted)), 6)))
    return seen[0] if seen[0] == seen[1] else None


def implementation_symbol(op_id: str, chain: List[str], data_kind: str) -> Optional[str]:
    """Name of the class that actually holds the parameters.

    The file alone is not enough. `boostings_implementations.py` is 350 lines
    with four near-identical wrappers; given only the file, the agent diagnosed
    the defect correctly and then failed to land a patch nine attempts running,
    because its quoted OLD block never matched. The lint templates hit the same
    wall and were fixed by showing the exact symbol -- same fix here."""
    try:
        pipeline = build_and_fit(chain, None, make_data(data_kind))
    except Exception:
        return None
    node = next((n for n in pipeline.nodes if n.name == op_id), None)
    fitted = getattr(node, "fitted_operation", None)
    if fitted is None or not hasattr(fitted, "params"):
        return None
    return type(fitted).__name__


def implementation_file(op_id: str, chain: List[str], data_kind: str) -> Optional[str]:
    """Source file of the class that actually holds the parameters.

    Resolved by fitting rather than by reading repository metadata: what a node
    ends up carrying in ``fitted_operation`` is decided by the evaluation
    strategy at runtime, and only that object does the rewriting."""
    import inspect

    try:
        pipeline = build_and_fit(chain, None, make_data(data_kind))
    except Exception:
        return None
    node = next((n for n in pipeline.nodes if n.name == op_id), None)
    fitted = getattr(node, "fitted_operation", None)
    if fitted is None or not hasattr(fitted, "params"):
        return None
    try:
        path = inspect.getfile(type(fitted))
    except TypeError:
        return None
    marker = os.sep + "fedot" + os.sep
    idx = path.rfind(marker)
    return path[idx + 1:].replace(os.sep, "/") if idx >= 0 else None


def build_defect_test(op_id: str, chain: List[str], data_kind: str,
                      param: str, value: Any) -> Dict[str, str]:
    chain_code = "\n".join(
        '    builder = builder.add_node("' + name + '"'
        + (', params={"' + param + '": ' + repr(value) + '})' if i == 0 else ')')
        for i, name in enumerate(chain))
    test_name = "test_" + op_id + "_keeps_declared_" + param
    code = TEST_TEMPLATE.format(
        data_code=DATA_CODE[data_kind], chain_code=chain_code,
        test_name=test_name, op=op_id, param=param, value=value)
    golden = prediction_fingerprint(chain, {param: value}, data_kind)
    if golden is not None:
        code += GOLDEN_SNIPPET.format(test_name=test_name, golden=golden)
    return {"test_name": test_name, "test_code": code}


def build_crash_test(op_id: str, chain: List[str], finding: Dict[str, Any]) -> Dict[str, str]:
    param, value = finding["param"], finding["value"]
    chain_code = "\n".join(
        '    builder = builder.add_node("' + name + '"'
        + (', params={"' + param + '": ' + repr(value) + '})' if i == 0 else ')')
        for i, name in enumerate(chain))
    where = os.path.basename(finding.get("foreign") or "FEDOT")
    test_name = "test_" + op_id + "_accepts_declared_" + param
    code = CRASH_TEST_TEMPLATE.format(
        data_code=DATA_CODE[finding["data"]], chain_code=chain_code,
        test_name=test_name, op=op_id, param=param, value=value, where=where,
        error=finding.get("error_type", "") + ": " + finding.get("error", "")[:150])
    return {"test_name": test_name, "test_code": code}


def strip_fingerprint(test_code: str, test_name: str) -> str:
    """Remove the behaviour-fingerprint test, keeping the other two intact."""
    head, sep, _ = test_code.partition("\n\ndef " + test_name + "_behaviour_unchanged")
    return (head + "\n") if sep else test_code


def verify_fails_on_pristine(items: List[Dict[str, Any]], python: str,
                             repo: str) -> List[Dict[str, Any]]:
    """Keep only the defects whose test really fails on the untouched checkout.

    Same discipline as the lint templates, and it earns its keep: of 23 tests
    built from the scan, 22 fail and one (`resample.replace`) passes, because
    that operation is non-deterministic and the run that produced the finding
    was not the run that reproduces it. A test that does not fail first proves
    nothing about the patch that follows.
    """
    kept: List[Dict[str, Any]] = []
    tmp = os.path.join(repo, "test", "unit", "test_fedotllm_invariant_tmp.py")
    rel = "test/unit/test_fedotllm_invariant_tmp.py"

    def run() -> Optional[str]:
        try:
            proc = subprocess.run([python, "-m", "pytest", rel, "-q"], cwd=repo,
                                  capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None
        return proc.stdout

    try:
        for item in items:
            with open(tmp, "w") as fh:
                fh.write(item["test_code"])
            out = run()
            if out is None:
                continue

            # The fingerprint has to hold on the untouched checkout, in a fresh
            # process, twice. Measured why: computing it inside the scanning
            # process gave 221.230299 for `ransac_lin_reg` and every standalone
            # run gives 219.116627 -- RANSAC draws its subsets from the global
            # numpy RNG, so the value depends on whatever was fitted before it.
            # A golden value that is wrong from the start fails every patch,
            # including the correct one, so it is dropped instead.
            if "behaviour_unchanged" in item["test_code"]:
                unstable = "behaviour_unchanged" in (out or "")
                if not unstable:
                    out2 = run()
                    unstable = out2 is None or "behaviour_unchanged" in out2
                if unstable:
                    item = dict(item, test_code=strip_fingerprint(
                        item["test_code"], item["test_name"]))
                    with open(tmp, "w") as fh:
                        fh.write(item["test_code"])
                    out = run() or ""

            if " failed" in out or " error" in out:
                kept.append(item)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return kept


def bridge(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn scan results into (file, failing test) pairs the agent can act on."""
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace

    spaces = PipelineSearchSpace().parameters_per_operation
    out: List[Dict[str, Any]] = []
    for res in results:
        if not res.get("chain"):
            continue
        op, chain = res["operation"], res["chain"]

        # A parameter that cannot be used at all outranks one that is merely
        # replaced: the user gets a foreign library's traceback, not a value.
        for f in unusable_parameters(res, spaces.get(op, {})):
            rel = implementation_file(op, chain, f["data"])
            if rel is None:
                continue
            item = build_crash_test(op, chain, f)
            item.update({"file": rel, "operation": op, "param": f["param"],
                         "value": f["value"], "kind": "unusable_parameter",
                         "symbol": implementation_symbol(op, chain, f["data"]) or "",
                         "observed": {"error": f.get("error_type", "")}})
            out.append(item)

        rewritten = [f for f in res.get("findings", [])
                     if f.get("kind") == "declared_not_used"]
        if not rewritten:
            continue
        f = rewritten[0]
        rel = implementation_file(op, chain, f["data"])
        if rel is None:
            continue
        item = build_defect_test(op, chain, f["data"], f["param"], f["value"])
        item.update({"file": rel, "operation": op, "param": f["param"],
                     "value": f["value"], "kind": "rewritten_parameter",
                     "symbol": implementation_symbol(op, chain, f["data"]) or "",
                     "observed": f.get("observed", {})})
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def all_operations(python: str, cwd: Optional[str]) -> List[str]:
    code = (
        "import warnings;warnings.filterwarnings('ignore');"
        "from fedot.core.repository.operation_types_repository import OperationTypesRepository as R;"
        "import json;"
        "print(json.dumps([o.id for k in ('model','data_operation') for o in R(k).operations]))"
    )
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, cwd=cwd)
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("cannot list operations: " + out.stderr[-2000:])


def scan(python: str, cwd: Optional[str] = None, only: Optional[Iterable[str]] = None,
         timeout: int = 420, metamorphic: bool = True,
         progress=None) -> List[Dict[str, Any]]:
    ops = list(only) if only else all_operations(python, cwd)
    results = []
    for op in ops:
        cmd = [python, os.path.abspath(__file__), "--worker", op]
        if not metamorphic:
            cmd.append("--no-metamorphic")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=cwd, timeout=timeout)
            payload = None
            for line in reversed(proc.stdout.strip().splitlines()):
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if payload is None:
                payload = {"operation": op, "findings": [], "setup": "worker produced no result",
                           "stderr": proc.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            payload = {"operation": op, "findings": [
                {"kind": "hang", "detail": f"no result in {timeout}s on legal data"}],
                "setup": None}
        results.append(payload)
        if progress:
            progress(payload)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help="run the checks for one operation (internal)")
    ap.add_argument("--no-metamorphic", action="store_true")
    ap.add_argument("--python", default=os.environ.get("FEDOTLLM_REPO_PYTHON", sys.executable))
    ap.add_argument("--repo", default=os.environ.get("FEDOTLLM_REPO_PATH"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--bridge", help="scan JSON to turn into (file, failing test) pairs")
    ap.add_argument("--tuning", help="check that one operation can actually be tuned")
    args = ap.parse_args()

    if args.bridge:
        with open(args.bridge) as fh:
            items = bridge(json.load(fh))
        built = len(items)
        if args.repo:
            items = verify_fails_on_pristine(items, args.python, args.repo)
            print(f"{built} tests built, {len(items)} verified to fail on the "
                  f"untouched checkout")
        target = args.out or (args.bridge + ".bridge.json")
        with open(target, "w") as fh:
            json.dump(items, fh, indent=2, default=str)
        print(f"{len(items)} defects with a ready test -> {target}")
        return 0

    if args.tuning:
        try:
            res = tuning_check(args.tuning)
        except Exception as exc:
            res = {"operation": args.tuning, "ok": False,
                   "reason": "harness error: " + type(exc).__name__ + ": " + str(exc)[:200]}
        print(json.dumps(res, default=str))
        return 0

    if args.worker:
        try:
            res = run_operation(args.worker, not args.no_metamorphic)
        except Exception as exc:  # a crash of the harness itself is not a finding
            res = {"operation": args.worker, "findings": [],
                   "setup": "harness error: " + type(exc).__name__ + ": " + str(exc)[:300]}
        print(json.dumps(res, default=str))
        return 0

    def progress(payload):
        n = len(payload.get("findings", []))
        mm = [m for m in payload.get("metamorphic", []) if m.get("status") not in (None, "ok")]
        print(f"  {payload['operation']:<24} findings={n:<3} metamorphic_issues={len(mm):<3} "
              f"{payload.get('setup') or ''}", flush=True)

    print(f"scanning with {args.python}", flush=True)
    results = scan(args.python, cwd=args.repo, only=args.only,
                   timeout=args.timeout, metamorphic=not args.no_metamorphic,
                   progress=progress)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print("written to " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
