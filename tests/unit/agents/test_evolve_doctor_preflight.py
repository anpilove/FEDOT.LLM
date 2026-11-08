"""Doctor fails before a quality fit when the FEDOT pin cannot run return_best."""

from pathlib import Path

import pytest

from fedotllm.agents.evolve.commands.doctor import run_doctor
from fedotllm.agents.evolve.commands.preflight import (
    analyze_checkout,
    dependency_checks,
    parse_preflight_stdout,
)

API = """
class Fedot:
    def __init__(
        self,
        problem,
        timeout_policy="legacy",
        final_fit_reserve_seconds=0,
        cv_folds=None,
        max_pipeline_fit_time=None,
        max_pipeline_evaluation_time=None,
    ):
        if timeout_policy not in {"legacy", "return_best"}:
            raise ValueError(timeout_policy)
"""

STOCK_API = """
class Fedot:
    def __init__(self, problem, timeout=None, **composer_tuner_params):
        self.problem = problem
"""


def _package(root: Path, api: str, *, extra: str = "") -> Path:
    package = root / "fedot"
    (package / "api").mkdir(parents=True)
    (package / "__init__.py").write_text('"""fixture"""\n', encoding="utf-8")
    (package / "api" / "__init__.py").write_text("", encoding="utf-8")
    (package / "api" / "main.py").write_text(api, encoding="utf-8")
    if extra:
        (package / "parallel.py").write_text(extra, encoding="utf-8")
    return root


def test_stock_pin_does_not_implement_return_best(tmp_path: Path):
    source = _package(tmp_path / "FEDOT", STOCK_API)
    report = analyze_checkout(source)
    assert report["implements_return_best"] is False
    assert "timeout_policy" in report["missing_parameters"]
    loaded = source / "fedot" / "__init__.py"
    checks = dependency_checks(
        source,
        {"fedot_file": str(loaded), "dask": False, "parameters": ["self", "problem"]},
    )
    assert checks["return_best"]["ok"] is False
    assert checks["dask"]["ok"] is True
    assert checks["dask"]["required"] is False
    assert checks["path"]["ok"] is True


def test_docstring_mention_is_not_an_implementation(tmp_path: Path):
    source = _package(
        tmp_path / "FEDOT",
        STOCK_API + '\n"""timeout_policy return_best is not implemented."""\n',
    )
    assert analyze_checkout(source)["implements_return_best"] is False


def test_return_best_checkout_accepts_the_worker_parameters(tmp_path: Path):
    source = _package(tmp_path / "FEDOT", API)
    report = analyze_checkout(source)
    assert report["implements_return_best"] is True
    assert report["missing_parameters"] == []
    loaded = source / "fedot" / "api" / "main.py"
    checks = dependency_checks(
        source,
        {
            "fedot_file": str(loaded),
            "dask": False,
            "parameters": [
                "self",
                "problem",
                "timeout_policy",
                "final_fit_reserve_seconds",
                "cv_folds",
                "max_pipeline_fit_time",
                "max_pipeline_evaluation_time",
            ],
        },
    )
    assert checks["return_best"]["ok"] is True
    assert checks["path"]["ok"] is True


def test_import_outside_the_checkout_fails_path_and_return_best(tmp_path: Path):
    source = _package(tmp_path / "FEDOT", API)
    other = tmp_path / "site-packages" / "fedot" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("", encoding="utf-8")
    checks = dependency_checks(
        source,
        {"fedot_file": str(other), "dask": True, "parameters": ["timeout_policy"]},
    )
    assert checks["path"]["ok"] is False
    assert checks["return_best"]["ok"] is False
    assert "outside checkout" in checks["path"]["detail"]


def test_dask_import_in_the_pin_requires_the_worker_interpreter(tmp_path: Path):
    source = _package(tmp_path / "FEDOT", API, extra="import dask.distributed\n")
    loaded = source / "fedot" / "__init__.py"
    missing = dependency_checks(
        source,
        {
            "fedot_file": str(loaded),
            "dask": False,
            "parameters": [
                "timeout_policy",
                "final_fit_reserve_seconds",
                "cv_folds",
                "max_pipeline_fit_time",
                "max_pipeline_evaluation_time",
            ],
        },
    )
    present = dependency_checks(
        source,
        {
            "fedot_file": str(loaded),
            "dask": True,
            "parameters": missing["return_best"]
            and [
                "timeout_policy",
                "final_fit_reserve_seconds",
                "cv_folds",
                "max_pipeline_fit_time",
                "max_pipeline_evaluation_time",
            ],
        },
    )
    assert analyze_checkout(source)["imports_dask"] is True
    assert missing["dask"]["ok"] is False
    assert missing["dask"]["required"] is True
    assert present["dask"]["ok"] is True


def test_preflight_marker_parses_the_last_payload():
    raw = 'noise\nEVOLVE_PREFLIGHT {"fedot_file": "/tmp/fedot/__init__.py", "dask": false}\n'
    assert parse_preflight_stdout(raw)["fedot_file"].endswith("__init__.py")
    assert parse_preflight_stdout("no marker") is None
    assert parse_preflight_stdout("EVOLVE_PREFLIGHT {") is None


def test_doctor_skips_evaluator_when_the_pin_lacks_return_best(
    tmp_path: Path, monkeypatch
):
    source = _package(tmp_path / "FEDOT", STOCK_API)
    called = []

    def fail_if_called(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("evaluator must not start after a failed preflight")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.run_stock",
        fail_if_called,
    )
    payload = run_doctor(source, tmp_path / "work", evaluator=True)
    assert called == []
    assert payload["ok"] is False
    checks = payload["checks"]
    assert checks["return_best"]["ok"] is False
    assert checks["path"]["ok"] is True
    assert checks["dask"]["required"] is False
    assert checks["evaluator"]["reason"] == "preflight_failed"
    assert checks["pin"]["ok"] is True
    assert checks["immutable_source"]["ok"] is True
    assert not (tmp_path / "work" / "experiments").exists() or not any(
        (tmp_path / "work" / "experiments").rglob("main.py")
    )


def test_doctor_accepts_a_return_best_pin_without_running_the_evaluator(
    tmp_path: Path, monkeypatch
):
    source = _package(tmp_path / "FEDOT", API)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.verify_manifest",
        lambda _source: [],
    )
    payload = run_doctor(source, tmp_path / "work", evaluator=False)
    checks = payload["checks"]
    assert checks["return_best"]["ok"] is True, checks["return_best"]
    assert checks["dask"]["ok"] is True
    assert checks["path"]["ok"] is True
    assert checks["fedot_import"]["ok"] is True
    assert checks["evaluator"] == {"ok": True, "skipped": True}
    assert checks["pin"]["ok"] is True
    assert payload["ok"] is True


def test_doctor_rejects_a_dask_pin_when_dask_is_not_installed(
    tmp_path: Path, monkeypatch
):
    pytest.importorskip("importlib")
    source = _package(tmp_path / "FEDOT", API, extra="import dask\n")
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.doctor.run_stock",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("evaluator ran")),
    )
    payload = run_doctor(source, tmp_path / "work", evaluator=True)
    assert payload["checks"]["dask"]["ok"] is False
    assert payload["checks"]["dask"]["required"] is True
    assert payload["checks"]["return_best"]["ok"] is True
    assert payload["checks"]["evaluator"]["reason"] == "preflight_failed"
    assert payload["ok"] is False
