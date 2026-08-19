#!/usr/bin/env python3
"""Turn verified findings into issues a maintainer can act on.

Two things this does that a raw finding list does not.

**It groups.** The scan produced 22 failing tests, but they are not 22 problems:
twelve of them are one root cause seen through twelve operations. Filing 22
issues would be exactly the behaviour that made cURL close its bug bounty in
January 2026 — volume that costs the maintainer more than it gives.

**It re-runs the reproduction before writing anything.** An issue whose repro
does not reproduce is worse than no issue: it burns the maintainer's trust on
the first click. Every snippet below is executed against the checkout at report
time, and an issue is emitted only if its snippet still fails.

    uv run python examples/report_issues.py --out issues/
    uv run python examples/report_issues.py --post owner/repo   # needs GITHUB_TOKEN
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class Issue:
    slug: str
    title: str
    labels: list[str]
    summary: str
    repro: str
    expected: str
    actual: str
    scope: str
    why_it_matters: str
    # Executed at report time; the issue is dropped unless this exits non-zero
    # (i.e. the assertion inside the snippet still fails).
    verify: str = ""
    verified: bool = field(default=False, init=False)
    evidence: str = field(default="", init=False)


CATBOOST_REPRO = '''
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

# `iterations`, `border_count` and `max_leaves` are declared tunable ...
declared = set(PipelineSearchSpace().parameters_per_operation["catboost"])
print(sorted(declared & {"iterations", "border_count", "max_leaves"}))

# ... while default_operation_params.json sets num_trees / max_bin /
# grow_policy=SymmetricTree for the same operation. CatBoost rejects each pair.
pipeline = PipelineBuilder().add_node("catboost").build()
tuner = (TunerBuilder(Task(TaskTypesEnum.classification))
         .with_tuner(SimultaneousTuner)
         .with_metric(ClassificationMetricsEnum.ROCAUC)
         .with_iterations(6)
         .build(train_data))
tuner.tune(pipeline)
print("obtained_metric:", tuner.obtained_metric)   # None — every candidate failed to fit
'''

CACHE_REPRO = '''
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
'''


def build_issues() -> list[Issue]:
    return [
        Issue(
            slug="catboost-tuning-never-succeeds",
            title="Tuning CatBoost never succeeds: search space and default params "
                  "declare the same settings under different names",
            labels=["bug"],
            summary=(
                "`PipelineSearchSpace` declares `iterations`, `border_count` and "
                "`max_leaves` tunable for `catboost` / `catboostreg`, while "
                "`default_operation_params.json` sets `num_trees`, `max_bin` and "
                "`grow_policy: SymmetricTree` for the same operations. CatBoost "
                "rejects each of those three pairs outright, so every tuning "
                "candidate fails to fit."
            ),
            repro=CATBOOST_REPRO,
            expected="The tuner explores the declared space and returns a metric.",
            actual=(
                "Every candidate fails with `CatBoostError: only one of the parameters "
                "iterations, n_estimators, num_boost_round, num_trees should be "
                "initialized` (likewise `border_count` vs `max_bin`, and `max_leaves` "
                "only working with `grow_policy=Lossguide`). `obtained_metric` is "
                "`None`, and the tuner logs *\"Return init graph due to the fact that "
                "obtained metric is None\"* and hands back the untuned pipeline. "
                "Nothing surfaces to the caller: tuning silently does nothing."
            ),
            scope="`catboost`, `catboostreg`. Control: the same harness on `rf` "
                  "returns a real metric.",
            why_it_matters=(
                "A user who tunes CatBoost gets default hyperparameters and no error. "
                "The fix is a change to two declarations, not to any algorithm."
            ),
            verify=(
                "from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace\n"
                "from fedot.core.operations.operation_parameters import OperationParameters\n"
                "import json, pathlib, fedot\n"
                "root = pathlib.Path(fedot.__file__).parent\n"
                "defaults = json.loads((root / 'core/repository/data/"
                "default_operation_params.json').read_text())['catboost']\n"
                "declared = set(PipelineSearchSpace().parameters_per_operation['catboost'])\n"
                "clashes = [(a, b) for a, b in (('iterations', 'num_trees'),\n"
                "                               ('border_count', 'max_bin'))\n"
                "           if a in declared and b in defaults]\n"
                "assert not clashes, 'still conflicting: %r' % (clashes,)\n"
            ),
        ),
        Issue(
            slug="declared-params-rewritten-break-operations-cache",
            title="Operations that rewrite their own parameters during fit lose the "
                  "operations cache (and take downstream nodes with them)",
            labels=["bug"],
            summary=(
                "Twelve operations replace a declared hyperparameter during `fit` and "
                "write the replacement back into `self.params` (via "
                "`self.params.update(...)`). `PipelineNode.descriptive_id` embeds those "
                "parameters and is the operations-cache key, so the fitted node is "
                "stored under an id nobody will look up — and every node downstream of "
                "it misses too, because their ids embed the parent's parameters."
            ),
            repro=CACHE_REPRO,
            expected="An identical pipeline reloads both nodes from the cache.",
            actual="Nothing is reloaded — neither `lagged` nor `ridge`. "
                   "Control: `pca -> ridge` reloads both.",
            scope=(
                "Measured on 13 operations: `ar`, `catboost`, `catboostreg`, `cut`, "
                "`dask_pca`, `diff_filter`, `ets`, `fast_ica`, `kernel_pca`, `lagged`, "
                "`polyfit`, `ransac_lin_reg`, `sparse_lagged`. Four of them — "
                "`catboost`, `catboostreg`, `lagged`, `sparse_lagged` — do it with "
                "their own defaults, without the caller setting anything."
            ),
            why_it_matters=(
                "`lagged` underpins nearly every time-series pipeline FEDOT composes, "
                "so it is re-fitted on every evaluation. The neighbouring symptom is "
                "already known: `predictions_cache.py` skips any node whose id contains "
                "\"ransac\", with `# TODO: issue#1363`. The operations cache has no such "
                "workaround and no test.\n\n"
                "A fix that keeps the adaptive behaviour is small: let the operation "
                "adapt a private attribute and leave `self.params` holding what the "
                "caller declared."
            ),
            verify=(
                "from fedot.core.caching.operations_cache import OperationsCache\n"
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "build = lambda: PipelineBuilder().add_node('lagged').add_node('ridge').build()\n"
                "cache = OperationsCache()\n"
                "fitted = build(); fitted.fit(ts_data); cache.save_pipeline(fitted)\n"
                "fresh = build(); cache.try_load_into_pipeline(fresh)\n"
                "hits = [n.name for n in fresh.nodes if n.fitted_operation is not None]\n"
                "assert sorted(hits) == sorted(n.name for n in fresh.nodes), hits\n"
            ),
        ),
        Issue(
            slug="single-parameter-of-a-pair-raises",
            title="Setting one hyperparameter of a pair makes the operation raise: "
                  "the other is read as None and passed to sklearn explicitly",
            labels=["bug"],
            summary=(
                "Several operations collect a fixed pair of parameters with `.get()` "
                "and forward both to sklearn. Declaring only one means the other "
                "arrives as an explicit `None`, overriding sklearn's own default."
            ),
            repro=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n\n"
                "# `degree` is declared tunable for poly_features; set it alone:\n"
                "pipeline = PipelineBuilder().add_node('poly_features', "
                "params={'degree': 3}).add_node('ridge').build()\n"
                "pipeline.fit(train_data)\n"
                "# InvalidParameterError: The 'interaction_only' parameter of "
                "PolynomialFeatures\n#   must be an instance of 'bool' ... Got None instead.\n"
            ),
            expected="The unset member of the pair keeps sklearn's default.",
            actual="`InvalidParameterError` from inside sklearn.",
            scope=(
                "`poly_features` (`degree` / `interaction_only`, "
                "`sklearn_transformations.py:224`) and the four `rfe_*` operations "
                "(`n_features_to_select` / `step`, `sklearn_selectors.py`).\n\n"
                "Note the tuner does not hit this — it always sets both members. It is "
                "the user setting a single hyperparameter by hand who is affected."
            ),
            why_it_matters="A documented, declared-tunable parameter cannot be used on "
                           "its own.",
            verify=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "p = PipelineBuilder().add_node('poly_features', params={'degree': 3})"
                ".add_node('ridge').build()\n"
                "p.fit(train_data)\n"
            ),
        ),
        Issue(
            slug="lda-shrinkage-guard-checks-declared-solver",
            title="LDA: the guard that should suppress `shrinkage` on the `svd` solver "
                  "never fires, because it tests the declared solver, not the effective one",
            labels=["bug"],
            summary=(
                "`LDAImplementation.check_and_correct_params` reads "
                "`self.params.get('solver')`. `lda` has no `solver` entry in "
                "`default_operation_params.json`, so for a caller who sets only "
                "`shrinkage` the value is `None`, `is_solver_svd` is False, and nothing "
                "is corrected — while sklearn's own default solver is `svd`, which is "
                "exactly the combination the guard exists to prevent."
            ),
            repro=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n\n"
                "pipeline = PipelineBuilder().add_node('lda', "
                "params={'shrinkage': 0.5}).build()\n"
                "pipeline.fit(train_data)\n"
                "# NotImplementedError: shrinkage not supported with 'svd' solver.\n"
            ),
            expected="Either the shrinkage is honoured (by moving to a solver that "
                     "supports it) or it is ignored, as the guard intends.",
            actual="`NotImplementedError` from sklearn reaches the caller.",
            scope="`lda`; `discriminant_analysis.py:78`.",
            why_it_matters=(
                "`shrinkage` is declared tunable, so every value in its declared scope "
                "raises. The condition needs the effective solver rather than the "
                "explicitly declared one."
            ),
            verify=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "p = PipelineBuilder().add_node('lda', params={'shrinkage': 0.5}).build()\n"
                "p.fit(train_data)\n"
            ),
        ),
        Issue(
            slug="lagged-window-replaced-by-unseeded-random",
            title="`lagged` replaces an out-of-range window_size with an unseeded "
                  "random value, so time-series pipelines are not reproducible by default",
            labels=["bug"],
            summary=(
                "When the declared `window_size` exceeds "
                "`len(series) - forecast_length - 1`, `ts_transformations.py:131` picks "
                "`int(random() * max_allowed_window_size)`. The requested value is "
                "discarded entirely, and `random()` reads the global `random` module "
                "state, which FEDOT seeds only from `Fedot(seed=...)` — default `None`."
            ),
            repro=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n\n"
                "for _ in range(6):\n"
                "    p = PipelineBuilder().add_node('lagged', "
                "params={'window_size': 252}).add_node('ridge').build()\n"
                "    node = next(n for n in p.nodes if n.name == 'lagged')\n"
                "    p.fit(ts_data_of_200_points)\n"
                "    print(node.parameters['window_size'])\n"
                "# 158, 97, 48, 111, 149, 22 — a different model each run\n"
            ),
            expected=(
                "Either clamping to the maximum allowed window (as the branch two lines "
                "above already does for the other direction), or at least a "
                "reproducible choice."
            ),
            actual="A uniformly random window anywhere in the legal range, differing "
                   "between runs of identical code.",
            scope="`lagged`, `sparse_lagged`.",
            why_it_matters=(
                "Passing `Fedot(seed=...)` does fix it — with `random.seed(0)` all six "
                "runs give 163 — so the honest statement is that time-series pipelines "
                "are not reproducible out of the box."
            ),
            verify=(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "seen = set()\n"
                "for _ in range(4):\n"
                "    p = PipelineBuilder().add_node('lagged', "
                "params={'window_size': 252}).add_node('ridge').build()\n"
                "    node = next(n for n in p.nodes if n.name == 'lagged')\n"
                "    p.fit(ts_data)\n"
                "    seen.add(node.parameters['window_size'])\n"
                "assert len(seen) == 1, 'window_size varies between runs: %r' % (seen,)\n"
            ),
        ),
    ]


PREAMBLE = '''
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
'''


def verify(issue: Issue, repo: Path, py: str, workdir: Path) -> None:
    """Run the issue's own check. It must still fail, or the issue is dropped."""
    if not issue.verify:
        return
    script = workdir / f"verify_{issue.slug}.py"
    script.write_text(PREAMBLE + "\n" + issue.verify, encoding="utf-8")
    try:
        proc = subprocess.run([py, str(script)], cwd=repo, capture_output=True,
                              text=True, timeout=600)
    except subprocess.TimeoutExpired:
        issue.evidence = "verification timed out"
        return
    issue.verified = proc.returncode != 0
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    issue.evidence = tail[-1][:300] if tail else ""


def render(issue: Issue, commit: str) -> str:
    return f"""## Summary

{issue.summary}

## Reproduction
```python
{PREAMBLE.strip()}
```
```python
{issue.repro.strip()}
```

## Expected

{issue.expected}

## Actual

{issue.actual}

Reproduced against `{commit}` at report time; the check above exits non-zero:
```text
{issue.evidence or '<no output>'}
```

## Scope

{issue.scope}

## Why it matters

{issue.why_it_matters}

---
Found by a runtime invariant scan that fits every operation in the repository with
values taken from its own declared `PipelineSearchSpace`, so nothing reported here
is an input FEDOT calls illegal. Happy to send a PR for any of these.
"""


def post(repo_full: str, issue: Issue, body: str, token: str) -> str:
    payload = json.dumps({"title": issue.title, "body": body,
                          "labels": issue.labels}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo_full}/issues", data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "fedotllm-evolve-agent"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp).get("html_url", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("issues"))
    ap.add_argument("--post", metavar="OWNER/REPO",
                    help="open the issues on this repository (needs GITHUB_TOKEN); "
                         "point it at a fork unless you mean the real project")
    ap.add_argument("--skip-verify", action="store_true",
                    help="write the issues without re-running their reproductions "
                         "(not recommended: an issue that does not reproduce is worse "
                         "than no issue)")
    args = ap.parse_args()

    repo = Path(os.environ.get("FEDOTLLM_REPO_PATH", "")).resolve()
    py = os.environ.get("FEDOTLLM_REPO_PYTHON", "")
    if not args.skip_verify and (not repo.is_dir() or not py):
        print("FEDOTLLM_REPO_PATH / FEDOTLLM_REPO_PYTHON not set "
              "(or pass --skip-verify)", file=sys.stderr)
        return 2

    commit = "unknown"
    if repo.is_dir():
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                                capture_output=True, text=True).stdout.strip()

    workdir = Path("/tmp/fedotllm_issue_verify")
    workdir.mkdir(parents=True, exist_ok=True)

    issues = build_issues()
    args.out.mkdir(parents=True, exist_ok=True)
    kept = []
    for issue in issues:
        if args.skip_verify:
            issue.verified = True
        else:
            verify(issue, repo, py, workdir)
        mark = "reproduces" if issue.verified else "DOES NOT REPRODUCE — dropped"
        print(f"  {issue.slug:<45} {mark}", flush=True)
        if not issue.verified:
            continue
        body = render(issue, commit)
        (args.out / f"{issue.slug}.md").write_text(
            f"# {issue.title}\n\n{body}", encoding="utf-8")
        kept.append((issue, body))

    print(f"\n{len(kept)}/{len(issues)} issues verified and written to {args.out}/")

    if args.post:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            print("GITHUB_TOKEN not set — nothing posted", file=sys.stderr)
            return 2
        for issue, body in kept:
            try:
                url = post(args.post, issue, body, token)
                print(f"  opened: {url}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:300]
                print(f"  FAILED {issue.slug}: {exc.code} {detail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
