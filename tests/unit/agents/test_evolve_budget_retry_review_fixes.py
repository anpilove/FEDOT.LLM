"""Budget and restart regressions without FEDOT training or external resources."""

import sys
from concurrent.futures import Future

import pytest

from fedotllm.agents.evolve.evaluation import fedot_quality as quality
from fedotllm.agents.evolve.evaluation import quality_registry
from fedotllm.agents.evolve.types import ScoreResult


def _submit_immediately(work, **kwargs):
    result = Future()
    try:
        result.set_result(work())
    except BaseException as exc:
        result.set_exception(exc)
    return result


def test_loaded_legacy_registry_defaults_to_a_worker_compatible_profile(monkeypatch):
    raw = quality_registry._load_raw()
    raw["fedot_timeout_policy"] = "legacy"
    raw.pop("max_pipeline_evaluation_policy")
    monkeypatch.setattr(quality_registry, "_load_raw", lambda: raw)

    registry = quality_registry.load_quality_registry()
    spec = quality_registry.job_spec(registry.datasets[0], registry=registry)

    assert spec["max_pipeline_fit_time_seconds"] == 900
    assert spec["max_pipeline_evaluation_time_seconds"] == 900
    assert spec["max_pipeline_evaluation_policy"] == "per-candidate-evaluation-v1"


@pytest.mark.parametrize("side", ["stock", "patch"])
@pytest.mark.parametrize(
    "prior_attempts, expected_attempts", [(0, [1, 2]), (1, [2]), (2, [])]
)
def test_async_transient_retry_obeys_two_total_attempts_after_restart(
    tmp_path,
    monkeypatch,
    side,
    prior_attempts,
    expected_attempts,
):
    registry = quality_registry.load_quality_registry()
    dataset = registry.datasets[0]
    task_id = dataset.task_id
    interrupted = ScoreResult(
        task_id=task_id,
        status="crash",
        score=float("nan"),
        detail="controller interrupted during evaluation",
    )
    successful = ScoreResult(task_id=task_id, status="ok", score=0.8)
    other_side = "patch" if side == "stock" else "stock"
    previous = {(task_id, other_side): successful}
    if prior_attempts:
        previous[(task_id, side)] = interrupted
    attempts = []
    events = []

    def transient_side(_dataset, *, side, attempt, **kwargs):
        attempts.append((side, attempt))
        return ScoreResult(
            task_id=task_id,
            status="crash",
            score=float("nan"),
            detail="ConnectionError: reset",
        )

    monkeypatch.setattr(quality, "fedot_python", lambda checkout: sys.executable)
    monkeypatch.setattr(quality, "_stock_cache_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(quality, "run_quality_side", transient_side)

    row = quality._run_quality_pair_async(
        dataset,
        stock_checkout=tmp_path / "stock",
        patch_checkout=tmp_path / "patch",
        registry=registry,
        submit_compute=_submit_immediately,
        prior_side_results=previous,
        prior_attempt_counts={
            (task_id, side): prior_attempts,
            (task_id, other_side): 1,
        },
        artifact_root=tmp_path / "artifacts",
        on_side_event=lambda spec, event_side, phase, result, attempt: events.append(
            (event_side, phase, attempt)
        ),
    ).result(timeout=1)

    assert attempts == [(side, attempt) for attempt in expected_attempts]
    assert [
        attempt
        for event_side, phase, attempt in events
        if event_side == side and phase == "started"
    ] == expected_attempts
    assert (
        row["stock_result" if side == "stock" else "patched_result"].status == "crash"
    )
    assert (
        row["stock_result" if other_side == "stock" else "patched_result"].score == 0.8
    )


@pytest.mark.parametrize("status", ["ok", "timeout"])
def test_async_resume_reuses_completed_stock_and_runs_unfinished_patch(
    tmp_path,
    monkeypatch,
    status,
):
    registry = quality_registry.load_quality_registry()
    dataset = registry.datasets[0]
    task_id = dataset.task_id
    previous = ScoreResult(
        task_id=task_id, status=status, score=0.8 if status == "ok" else float("nan")
    )
    events = []
    calls = []

    def finish_patch(_dataset, *, side, attempt, **kwargs):
        calls.append((side, attempt))
        return ScoreResult(task_id=task_id, status="ok", score=0.9)

    monkeypatch.setattr(quality, "fedot_python", lambda checkout: sys.executable)
    monkeypatch.setattr(quality, "_stock_cache_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(quality, "run_quality_side", finish_patch)

    row = quality._run_quality_pair_async(
        dataset,
        stock_checkout=tmp_path / "stock",
        patch_checkout=tmp_path / "patch",
        registry=registry,
        submit_compute=_submit_immediately,
        prior_side_results={(task_id, "stock"): previous},
        prior_attempt_counts={(task_id, "stock"): 1},
        artifact_root=tmp_path / "artifacts",
        on_side_event=lambda spec, side, phase, result, attempt: events.append(
            (side, phase, attempt)
        ),
    ).result(timeout=1)

    assert calls == [("patch", 1)]
    assert events == [
        ("stock", "reused", 1),
        ("patch", "started", 1),
        ("patch", "finished", 1),
    ]
    assert row["stock_result"] is previous
    assert row["patched_result"].score == 0.9


def test_quality_resource_check_preserves_storage_error_without_exception_notes(
    tmp_path, monkeypatch
):
    import errno

    from fedotllm.agents.evolve.execution import storage_guard

    class Python310StorageError(storage_guard.StorageCapacityError):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text("VERSION = 1\n")
    error = Python310StorageError(
        errno.ENOSPC, "storage reserve exhausted", str(tmp_path)
    )
    checks = []

    def require_storage(path):
        checks.append(path)
        if len(checks) > 1:
            raise error

    def admitted_worker(command, **kwargs):
        kwargs["resource_check"]()
        raise AssertionError("storage failure must stop admission")

    monkeypatch.setattr(storage_guard, "require_storage", require_storage)
    monkeypatch.setattr(quality, "run_worker", admitted_worker)

    with pytest.raises(storage_guard.StorageCapacityError) as failure:
        quality.run_quality_side(
            quality_registry.load_quality_registry().datasets[0],
            checkout=source,
            side="patch",
            python_executable=sys.executable,
            artifact_root=tmp_path / "artifacts",
            stock_cache_root=tmp_path / "stock-cache",
        )

    assert failure.value is error
    assert error.errno == errno.ENOSPC
    assert len(checks) == 2
