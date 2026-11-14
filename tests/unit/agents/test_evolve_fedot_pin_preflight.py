"""Preflight must refuse a FEDOT pin that cannot run EvolveAgent quality workers."""

from __future__ import annotations

from pathlib import Path

from fedotllm.agents.evolve.commands.doctor import run_doctor
from fedotllm.agents.evolve.evaluation.fedot_pin_preflight import (
    FAILURE_KIND_ASSETS_MISSING,
    FAILURE_KIND_PATH_UNRESOLVED,
    FAILURE_KIND_PIN_INCOMPATIBLE,
    FedotPinPreflight,
    check_fedot_assets,
    check_fedot_pin_preflight,
    resolve_fedot_repo_path_check,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _stock_pin(tmp_path: Path) -> Path:
    """Minimal tree shaped like stock FEDOT without return_best / isolated optimizer."""

    root = tmp_path / "FEDOT-stock"
    _write(
        root / "fedot" / "api" / "api_utils" / "api_params_repository.py",
        "default_param_values_dict = dict(with_tuning=True, seed=None)\n",
    )
    _write(
        root / "fedot" / "core" / "operations" / "evaluation" /
        "operation_implementations" / "data_operations" / "sklearn_transformations.py",
        "import dask.array as da\n",
    )
    _write(root / "fedot" / "__init__.py", "")
    return root


def _compatible_pin(tmp_path: Path) -> Path:
    root = tmp_path / "FEDOT-ready"
    _write(
        root / "fedot" / "api" / "api_utils" / "api_params_repository.py",
        "default_param_values_dict = dict(\n"
        "    timeout_policy='legacy',\n"
        "    final_fit_reserve_seconds=0,\n"
        "    max_pipeline_evaluation_time=None,\n"
        ")\n",
    )
    _write(
        root / "fedot" / "core" / "optimisers" / "isolated_gp_optimizer.py",
        "class IsolatedEvoGraphOptimizer:\n    pass\n",
    )
    _write(
        root / "fedot" / "core" / "operations" / "evaluation" /
        "operation_implementations" / "data_operations" / "sklearn_transformations.py",
        "import dask.array as da\n",
    )
    _write(root / "fedot" / "__init__.py", "")
    return root


def test_stock_pin_without_return_best_fails_preflight_with_stable_failure_kind(tmp_path):
    result = check_fedot_pin_preflight(
        _stock_pin(tmp_path),
        require_return_best=True,
        require_isolated_optimizer=True,
        require_dask_import_site=True,
    )
    assert result.ok is False
    assert result.failure_kind == FAILURE_KIND_PIN_INCOMPATIBLE
    assert result.checks["timeout_policy_return_best"]["ok"] is False
    assert result.checks["isolated_gp_optimizer"]["ok"] is False
    assert "timeout_policy" in result.summary


def test_compatible_pin_passes_static_preflight(tmp_path):
    result = check_fedot_pin_preflight(
        _compatible_pin(tmp_path),
        require_return_best=True,
        require_isolated_optimizer=True,
        require_dask_import_site=True,
    )
    assert result.ok is True
    assert result.failure_kind is None
    assert result.checks["timeout_policy_return_best"]["ok"] is True
    assert result.checks["isolated_gp_optimizer"]["ok"] is True
    assert result.checks["dask_import_site"]["ok"] is True


def test_repo_path_check_fails_when_env_and_cache_missing(tmp_path):
    result = resolve_fedot_repo_path_check(
        environ={},
        repo_root=tmp_path / "missing-root",
    )
    assert result.ok is False
    assert result.failure_kind == FAILURE_KIND_PATH_UNRESOLVED
    assert "FEDOTLLM_REPO_PATH" in result.summary


def test_repo_path_check_accepts_explicit_directory(tmp_path):
    source = tmp_path / "FEDOT"
    source.mkdir()
    result = resolve_fedot_repo_path_check(explicit=source)
    assert result.ok is True
    assert result.failure_kind is None


def test_assets_check_reports_missing_real_cases(tmp_path):
    source = _compatible_pin(tmp_path)
    result = check_fedot_assets(source)
    assert result.ok is False
    assert result.failure_kind == FAILURE_KIND_ASSETS_MISSING
    assert "dataset missing" in result.summary or "commit mismatch" in result.summary


def test_doctor_includes_pin_preflight_and_fails_closed(tmp_path, monkeypatch):
    source = _stock_pin(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # Avoid expensive checkout/evaluator path: stub remaining doctor stages and
    # isolate the pin failure from asset/path noise in this unit.
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.check_fedot_assets",
        lambda _source: FedotPinPreflight(ok=True, checks={"manifest_assets": {"ok": True, "errors": []}}, summary="ok"),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.check_runtime_dask",
        lambda: FedotPinPreflight(ok=True, checks={"runtime_dask": {"ok": True}}, summary="ok"),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.create_experiment_checkout",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not checkout after pin fail")),
    )

    report = run_doctor(source, workspace, evaluator=False)
    assert report["ok"] is False
    pin = report["checks"]["fedot_pin"]
    assert pin["ok"] is False
    assert pin["failure_kind"] == FAILURE_KIND_PIN_INCOMPATIBLE
    assert report["checks"]["fedot_repo_path"]["ok"] is True
    assert report["checks"]["manifest"]["ok"] is True


def test_doctor_reports_missing_assets_even_when_pin_also_fails(tmp_path, monkeypatch):
    source = _stock_pin(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.check_runtime_dask",
        lambda: FedotPinPreflight(ok=True, checks={"runtime_dask": {"ok": True}}, summary="ok"),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.create_experiment_checkout",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no checkout on early fail")),
    )

    report = run_doctor(source, workspace, evaluator=False)
    assert report["ok"] is False
    assert report["checks"]["fedot_pin"]["ok"] is False
    assert report["checks"]["manifest"]["ok"] is False
    assert report["checks"]["manifest"]["failure_kind"] == FAILURE_KIND_ASSETS_MISSING
