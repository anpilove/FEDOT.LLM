from __future__ import annotations

import sys
import uuid
from pathlib import Path

from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.evaluation.eval import run_stock
from fedotllm.agents.evolve.evaluation.fedot_pin_preflight import (
    check_fedot_assets,
    check_runtime_dask,
    resolve_fedot_repo_path_check,
)
from fedotllm.agents.evolve.evaluation.return_best_policy import resolve_return_best_source
from fedotllm.agents.evolve.execution.run_code import fedot_python, run_fedot_snippet


def run_doctor(
    source: Path,
    workspace: Path,
    *,
    evaluator_task: str = "catboost",
    evaluator: bool = True,
) -> dict:
    source = source.resolve()
    workspace = workspace.resolve()
    before = source_fingerprint(source)
    checks: dict[str, dict] = {}

    # Early gates always run together so a pin failure cannot hide missing
    # FEDOTLLM_REPO_PATH / checksum assets / runtime dask.
    path_check = resolve_fedot_repo_path_check(explicit=source)
    checks["fedot_repo_path"] = {
        "ok": path_check.ok,
        "failure_kind": path_check.failure_kind,
        "summary": path_check.summary,
        "details": path_check.checks,
    }

    assets = check_fedot_assets(source)
    checks["manifest"] = {
        "ok": assets.ok,
        "failure_kind": assets.failure_kind,
        "summary": assets.summary,
        **assets.checks.get("manifest_assets", {}),
    }

    runtime_dask = check_runtime_dask()
    checks["runtime_dask"] = {
        "ok": runtime_dask.ok,
        "failure_kind": runtime_dask.failure_kind,
        "summary": runtime_dask.summary,
        "details": runtime_dask.checks,
    }

    decision = resolve_return_best_source(source)
    pin = decision.preflight
    checks["fedot_pin"] = {
        "ok": decision.ok,
        "failure_kind": None if decision.ok else pin.failure_kind,
        "summary": decision.detail,
        "policy": decision.policy,
        "resolved_source": str(decision.source),
        "details": pin.checks,
    }
    if decision.source != source and decision.ok:
        checks["return_best_research_worktree"] = {
            "ok": True,
            "path": str(decision.source),
            "note": "research worktree passes preflight; stock pin remains unchanged",
        }

    early_ok = all(
        bool(checks[name].get("ok"))
        for name in ("fedot_repo_path", "manifest", "runtime_dask", "fedot_pin")
    )
    if not early_ok:
        after = source_fingerprint(source)
        checks["immutable_source"] = {"ok": before == after, "before": before, "after": after}
        return {"ok": False, "checks": checks}

    experiment = None
    try:
        experiment = create_experiment_checkout(
            source,
            workspace,
            run_id=f"doctor-{uuid.uuid4().hex[:8]}",
            candidate_id="smoke",
        )
        disjoint = experiment != source and source not in experiment.parents and experiment not in source.parents
        checks["checkout"] = {
            "ok": disjoint,
            "source": str(source),
            "experiment": str(experiment),
        }
        try:
            interpreter = fedot_python(experiment)
            checks["interpreter"] = {
                "ok": True,
                "path": interpreter,
                "controller_python": sys.executable,
            }
        except OSError as exc:
            checks["interpreter"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        snippet = run_fedot_snippet(
            experiment,
            "import fedot\nprint(fedot.__file__)",
            trace_path=workspace / "trace.jsonl",
            timeout_s=30,
        )
        checks["fedot_import"] = {
            "ok": snippet.status == "ok" and str(experiment) in snippet.stdout,
            "status": snippet.status,
            "output": snippet.output[-1000:],
        }
        if evaluator:
            score = run_stock(
                evaluator_task,
                checkout=experiment,
                split="dev",
                seed=42,
            )
            checks["evaluator"] = {
                "ok": score.status == "ok",
                "task": evaluator_task,
                "status": score.status,
                "score": score.score,
                "detail": score.detail,
                "env_hash": score.env_hash,
            }
        else:
            checks["evaluator"] = {"ok": True, "skipped": True}
    except Exception as exc:
        # Attribute the failure to the first check that has not reported yet
        # instead of overwriting an already successful checkout check.
        failed = next(
            (name for name in ("checkout", "interpreter", "fedot_import", "evaluator") if name not in checks),
            "doctor",
        )
        checks[failed] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if experiment is not None and experiment.exists():
            try:
                discard_experiment_checkout(experiment, workspace=workspace, source=source)
            except Exception as exc:
                checks["cleanup"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    after = source_fingerprint(source)
    checks["immutable_source"] = {"ok": before == after, "before": before, "after": after}
    ok = all(bool(item.get("ok")) for item in checks.values())
    return {"ok": ok, "checks": checks}
