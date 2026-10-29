"""Integration checks of diagnostic spec wiring and local parallel fixtures."""

import json
import io
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from fedotllm.agents.evolve.evaluation import fedot_quality
from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry


@pytest.mark.parametrize("process_traces", [False, True])
def test_quality_side_passes_stack_path_to_real_child(
    tmp_path, monkeypatch, process_traces
):
    """Removing the stack path from the worker request loses real stack evidence."""
    checkout = tmp_path / "FEDOT"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot/__init__.py").write_text("# isolated fixture\n")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(artifacts))
    monkeypatch.delenv("EVOLVE_QUALITY_TRACE_PROCESSES", raising=False)
    monkeypatch.delenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("EVOLVE_QUALITY_TRACE_FEDOT", raising=False)
    monkeypatch.delenv("EVOLVE_QUALITY_TRACE_FEDOT_INTERVAL_SECONDS", raising=False)
    if process_traces:
        monkeypatch.setenv("EVOLVE_QUALITY_TRACE_PROCESSES", "1")
        monkeypatch.setenv("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "0.1")
    child = tmp_path / "controlled_child.py"
    child.write_text("""import argparse, json, time
from pathlib import Path
from fedotllm.agents.evolve.evaluation._fedot_quality_worker import _periodic_stack_dumps
p = argparse.ArgumentParser()
p.add_argument("--spec-json")
p.add_argument("--out")
a = p.parse_args()
spec = json.loads(Path(a.spec_json).read_text())
identity = {key: spec.get(key) for key in ("side", "attempt", "attempt_id",
    "registry_identity", "evaluation_identity", "worker_invocation_id")}
identity["task_id"] = spec["source_dataset"]
with _periodic_stack_dumps(spec.get("stack_trace_path"), interval_seconds=0.05) as state:
    time.sleep(0.2)
Path(a.out).write_text(json.dumps({**identity, "status":"crash", "detail":"controlled instrumentation fixture",
    "metric_observations":{"stack_trace_diagnostics":state}}))
""")
    real_run_worker = fedot_quality.run_worker

    def controlled_run_worker(command, **kwargs):
        return real_run_worker([command[0], str(child), *command[3:]], **kwargs)

    monkeypatch.setattr(fedot_quality, "run_worker", controlled_run_worker)
    result = fedot_quality.run_quality_side(
        "smoke-regression",
        checkout=checkout,
        side="patch",
        n_jobs=4,
        registry=smoke_registry(),
        python_executable=sys.executable,
        proposal_id="trace-wiring",
        attempt=1,
    )
    assert result.status == "crash"
    spec_files = list(artifacts.glob("*/quality-spec-*.json"))
    assert len(spec_files) == 1
    spec = json.loads(spec_files[0].read_text())
    assert spec.get("stack_trace_path"), (
        "production request omitted diagnostic artifact path"
    )
    path = Path(spec["stack_trace_path"])
    assert path.parent == artifacts / spec["worker_invocation_id"]
    assert "--- stack dump at " in path.read_text()
    assert "in <module>" in path.read_text()
    assert "controlled_child.py" in path.read_text()
    assert result.metric_observations["stack_trace_diagnostics"]["dump_count"] >= 1
    if process_traces:
        output = Path(result.metric_observations["worker_result_path"])
        trace_directory = output.with_suffix(".process-traces")
        assert spec.get("process_trace_dir") == str(trace_directory)
        assert spec.get("process_trace_manifest_path") == str(
            trace_directory / "manifest.jsonl"
        )
        assert spec.get("stack_dump_interval_seconds") == 0.1
        assert spec.get("process_trace_interval_seconds") == 0.1
    else:
        assert "process_trace_dir" not in spec


PROBE = (
    Path(__file__).resolve().parents[3]
    / "research/evolve/campaigns/2026-10-09/parallel_trace_probe.py"
)


def test_parallel_trace_probe_checks_without_creating_workspace(tmp_path):
    workspace = tmp_path / "check-only"
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mode"] == "check"
    assert not workspace.exists()


def test_parallel_trace_probe_saves_four_overlapping_cases(tmp_path):
    workspace = tmp_path / "parallel"
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--mode", "launch", "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((workspace / "launch_audit.json").read_text())
    assert report["passed"] is True
    assert report["overlap_seconds"] > 0.1
    assert report["purpose"] == "instrumentation_fixture_not_fedot_reproduction"
    assert {row["case"] for row in report["workers"]} == {
        "cpu_loop",
        "wait",
        "native_gil",
        "raised_exception",
    }
    for row in report["workers"]:
        assert row["stack_dump_count"] >= 1
        assert row["expected_function_present"] is True
        assert row["source_line_present"] is True
        assert Path(row["log_path"]).stat().st_size > 0
        assert row["elapsed_seconds"] < 10
    failure = next(
        row for row in report["workers"] if row["case"] == "raised_exception"
    )
    assert failure["returncode"] == 3
    assert "intentional diagnostic exception" in Path(failure["log_path"]).read_text()
    native = next(row for row in report["workers"] if row["case"] == "native_gil")
    assert native["native_call_elapsed_seconds"] >= 1.0
    assert native["native_inside_call_line_guaranteed"] is False
    assert native["sampling_gap_expected"] is True
    again = subprocess.run(
        [sys.executable, str(PROBE), "--mode", "launch", "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert again.returncode != 0
    assert json.loads((workspace / "launch_audit.json").read_text()) == report


def test_worker_main_restores_only_its_owned_logging_handlers(tmp_path, monkeypatch):
    """A repeated worker must not retain captured stdout or discard foreign handlers."""
    from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    foreign = logging.StreamHandler(io.StringIO())
    added_during_evaluation = logging.StreamHandler(io.StringIO())
    root_logger.addHandler(foreign)
    root_logger.setLevel(logging.WARNING)
    original_measure = worker._StageTimings.measure

    def measure_and_add_foreign(self, *args, **kwargs):
        root_logger.addHandler(added_during_evaluation)
        return original_measure(self, *args, **kwargs)

    monkeypatch.setattr(worker._StageTimings, "measure", measure_and_add_foreign)
    spec = tmp_path / "invalid-spec.json"
    spec.write_text(
        json.dumps({"source_dataset": "fixture", "composer_options": {"unexpected": 1}})
    )
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    try:
        assert (
            worker.main(
                ["--spec-json", str(spec), "--out", str(tmp_path / "result.json")]
            )
            != 0
        )
        assert root_logger.handlers == [
            *original_handlers,
            foreign,
            added_during_evaluation,
        ]
        assert root_logger.level == logging.WARNING
        stdout.close()
        logging.warning("foreign logging remains usable after captured stdout closes")
        assert "foreign logging remains usable" in foreign.stream.getvalue()
    finally:
        # Restore the test host even while exercising a leaking RED implementation.
        for handler in list(root_logger.handlers):
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()
        root_logger.setLevel(original_level)


def test_quality_side_passes_opt_in_fedot_trace_directory_to_child(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "FEDOT"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot/__init__.py").write_text("# isolated fixture\n")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(artifacts))
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_FEDOT", "1")
    monkeypatch.setenv("EVOLVE_QUALITY_TRACE_FEDOT_INTERVAL_SECONDS", "0.2")
    child = tmp_path / "trace_env_child.py"
    child.write_text("""import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--spec-json")
p.add_argument("--out")
a = p.parse_args()
spec = json.loads(Path(a.spec_json).read_text())
identity = {key: spec.get(key) for key in ("side", "attempt", "attempt_id",
    "registry_identity", "evaluation_identity", "worker_invocation_id")}
identity["task_id"] = spec["source_dataset"]
Path(a.out).write_text(json.dumps({**identity, "status":"crash", "detail":"trace env fixture",
    "metric_observations":{"spec_trace_dir":spec.get("fedot_trace_dir"),
    "env_trace_dir":os.environ.get("FEDOT_TRACE_DIR"),
    "env_trace_run_id":os.environ.get("FEDOT_TRACE_RUN_ID"),
    "env_trace_interval":os.environ.get("FEDOT_TRACE_INTERVAL_SECONDS")}}))
""")
    real_run_worker = fedot_quality.run_worker
    captured = {}

    def controlled_run_worker(command, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return real_run_worker([command[0], str(child), *command[3:]], **kwargs)

    monkeypatch.setattr(fedot_quality, "run_worker", controlled_run_worker)
    result = fedot_quality.run_quality_side(
        "smoke-regression",
        checkout=checkout,
        side="patch",
        n_jobs=4,
        registry=smoke_registry(),
        python_executable=sys.executable,
        proposal_id="fedot-trace-env",
        attempt=1,
    )

    spec = json.loads(next(artifacts.glob("*/quality-spec-*.json")).read_text())
    observations = result.metric_observations
    expected_dir = fedot_quality.worker_artifact_paths(
        "smoke-regression",
        "patch",
        "fedot-trace-env",
        1,
        root=artifacts,
        worker_invocation_id=spec["worker_invocation_id"],
    )["fedot_trace_dir"]
    assert result.status == "crash"
    assert spec["fedot_trace_dir"] == expected_dir
    assert spec["fedot_trace_interval_seconds"] == 0.2
    assert observations["spec_trace_dir"] == expected_dir
    assert observations["env_trace_dir"] == expected_dir
    assert observations["env_trace_run_id"] == spec["attempt_id"]
    assert observations["env_trace_interval"] == "0.2"
    assert captured["env"]["FEDOT_TRACE_DIR"] == expected_dir
