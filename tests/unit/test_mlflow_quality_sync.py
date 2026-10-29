from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "research/evolve/scripts/mlflow_quality_sync.py"
SPEC = importlib.util.spec_from_file_location("mlflow_quality_sync", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mlflow_quality_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mlflow_quality_sync)


class _FakeClient:
    def __init__(self, **_kwargs):
        self.experiments = {}
        self.runs = {}
        self.metric_history = {}
        self.text_artifacts = {}

    def get_experiment_by_name(self, name):
        return self.experiments.get(name)

    def create_experiment(self, name):
        experiment_id = str(len(self.experiments) + 1)
        self.experiments[name] = SimpleNamespace(experiment_id=experiment_id)
        return experiment_id

    def search_runs(self, *, filter_string, **_kwargs):
        key = filter_string.split("'", 2)[1]
        return [run for run in self.runs.values() if run.data.tags.get("controller_key") == key]

    def create_run(self, *, experiment_id, run_name, tags):
        run_id = f"run-{len(self.runs) + 1}"
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, status="RUNNING", experiment_id=experiment_id),
            data=SimpleNamespace(tags={**tags, "mlflow.runName": run_name}),
        )
        self.runs[run_id] = run
        return run

    def set_tag(self, run_id, key, value):
        self.runs[run_id].data.tags[key] = value

    def get_metric_history(self, run_id, name):
        return self.metric_history.get((run_id, name), [])

    def log_metric(self, run_id, name, value, *, step):
        self.metric_history.setdefault((run_id, name), []).append(
            SimpleNamespace(value=value, step=step)
        )

    def log_text(self, run_id, text, artifact_file):
        self.text_artifacts[(run_id, artifact_file)] = text

    def set_terminated(self, run_id, *, status):
        self.runs[run_id].info.status = status

    def update_run(self, run_id, *, status):
        self.runs[run_id].info.status = status


def test_mlflow_sync_is_idempotent_and_tracks_parent_and_dataset_attempts(monkeypatch):
    fake_client = _FakeClient()
    fake_mlflow = ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda _uri: None
    fake_mlflow.MlflowClient = lambda **kwargs: fake_client
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    summary = {
        "candidates": {
            "candidate-a": {
                "candidate_revision_id": "candidate-a",
                "proposal_id": "proposal-a",
                "title": "Forecast quality hypothesis",
                "queue_status": "running",
                "registry_identity": "registry-v1",
                "hypothesis": {"claim": "Improve forecasts"},
                "quality_scope": {"problem_families": ["ts_forecasting"]},
                "task_ids": ["task-1", "task-2"],
                "planned_task_count": 2,
                "completed_task_pair_count": 1,
                "measured_task_pair_count": 1,
                "datasets": [{"task_id": "task-1"}],
            }
        }
    }
    journal = [
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-a",
            "task_id": "task-1",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "registry-v1",
            "spec": {"metric": "rmse", "higher_is_better": False},
            "result": {"status": "ok", "score": 10.0},
        },
        {
            "event": "quality_task_side_finished",
            "candidate": "candidate-a",
            "task_id": "task-1",
            "side": "patch",
            "attempt": 1,
            "registry_identity": "registry-v1",
            "spec": {"metric": "rmse", "higher_is_better": False},
            "result": {"status": "ok", "score": 9.5},
        },
        {
            "event": "quality_task_side_started",
            "candidate": "candidate-a",
            "task_id": "task-2",
            "side": "stock",
            "attempt": 1,
            "registry_identity": "registry-v1",
            "spec": {"metric": "rmse", "higher_is_better": False},
        },
    ]

    first = mlflow_quality_sync.sync_quality_to_mlflow(
        journal, summary, tracking_uri="sqlite:///quality.db"
    )
    second = mlflow_quality_sync.sync_quality_to_mlflow(
        journal, summary, tracking_uri="sqlite:///quality.db"
    )

    assert first == second == {"parents": 1, "children": 2}
    assert len(fake_client.runs) == 3
    children = [run for run in fake_client.runs.values() if run.data.tags.get("task_id")]
    assert {run.data.tags["task_id"] for run in children} == {"task-1", "task-2"}
    completed_child = next(run for run in children if run.data.tags["task_id"] == "task-1")
    assert completed_child.info.status == "FINISHED"
    assert completed_child.data.tags["metric"] == "rmse"
    gain_history = fake_client.get_metric_history(completed_child.info.run_id, "quality_gain")
    assert len(gain_history) == 1
    assert abs(gain_history[0].value - 0.5) < 1e-9


def test_real_mlflow_sqlite_persists_one_parent_and_one_dataset_run(tmp_path):
    import pytest

    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    summary = {
        "candidates": {
            "proposal-r1": {
                "candidate_revision_id": "proposal-r1",
                "proposal_id": "proposal",
                "title": "Real SQLite smoke test",
                "queue_status": "running",
                "registry_identity": "registry-test",
                "hypothesis": {"claim": "test"},
                "quality_scope": {},
                "task_ids": ["task-1"],
                "planned_task_count": 1,
                "completed_task_pair_count": 1,
                "measured_task_pair_count": 1,
                "datasets": [{"task_id": "task-1"}],
            }
        }
    }
    journal = [
        {
            "event": "quality_task_side_finished",
            "candidate": "proposal-r1",
            "task_id": "task-1",
            "side": side,
            "attempt": 1,
            "registry_identity": "registry-test",
            "spec": {"metric": "rmse", "higher_is_better": False},
            "result": {"status": "ok", "score": score},
        }
        for side, score in (("stock", 2.0), ("patch", 1.5))
    ]

    mlflow_quality_sync.sync_quality_to_mlflow(journal, summary, tracking_uri=tracking_uri)
    mlflow_quality_sync.sync_quality_to_mlflow(journal, summary, tracking_uri=tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("FEDOT quality controller")
    runs = client.search_runs([experiment.experiment_id], max_results=10)
    child = next(run for run in runs if run.data.tags.get("task_id") == "task-1")

    assert len(runs) == 2
    assert child.info.status == "FINISHED"
    assert child.data.metrics["quality_gain"] == 0.5
