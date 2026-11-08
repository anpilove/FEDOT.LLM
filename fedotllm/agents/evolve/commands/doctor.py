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
from fedotllm.agents.evolve.evaluation.manifest import verify_manifest
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
    if not decision.ok:
        after = source_fingerprint(source)
        checks["immutable_source"] = {"ok": before == after, "before": before, "after": after}
        return {"ok": False, "checks": checks}
    # Doctor continues against the stock checkout; a research worktree is only
    # accepted as evidence that a compatible pin exists for offline experiments.
    if decision.source != source:
        checks["return_best_research_worktree"] = {
            "ok": True,
            "path": str(decision.source),
            "note": "research worktree passes preflight; stock pin remains unchanged",
        }

    try:
        manifest_errors = verify_manifest(source)
    except (OSError, ValueError) as exc:
        manifest_errors = [f"{type(exc).__name__}: {exc}"]
    checks["manifest"] = {"ok": not manifest_errors, "errors": manifest_errors}

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
