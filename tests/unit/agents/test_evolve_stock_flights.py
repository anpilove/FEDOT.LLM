"""Tests for non-blocking, cancellation-isolated shared stock results."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import threading

import pytest

from fedotllm.agents.evolve.evaluation.stock_flights import StockFlightRegistry


def test_same_key_subscribers_share_one_compute_and_same_result():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    gate = threading.Barrier(3)
    both_returned = threading.Barrier(3)
    started = threading.Event()
    release = threading.Event()
    submitted: list[Future[str]] = []
    submit_lock = threading.Lock()
    results: list[Future[str] | None] = [None, None]

    with (
        ThreadPoolExecutor(max_workers=2) as compute_pool,
        ThreadPoolExecutor(max_workers=2) as callers,
    ):

        def submit(compute):
            with submit_lock:
                future = compute_pool.submit(compute)
                submitted.append(future)
                return future

        def compute() -> str:
            started.set()
            assert release.wait(3), "test did not release the owner compute"
            return "stock-result"

        def subscribe(index: int) -> None:
            gate.wait(timeout=3)
            results[index] = registry.get_or_submit(
                "dataset-key", submit=submit, compute=compute
            )
            both_returned.wait(timeout=3)

        requests = [callers.submit(subscribe, index) for index in range(2)]
        gate.wait(timeout=3)
        try:
            assert started.wait(2), "owner compute was not submitted"
            both_returned.wait(timeout=2)
            for request in requests:
                request.result(timeout=2)
            assert len(submitted) == 1
            assert results[0] is not results[1], (
                "subscriber cancellation needs an independent Future view"
            )
            release.set()
            assert results[0].result(timeout=2) == "stock-result"
            assert results[1].result(timeout=2) == "stock-result"
        finally:
            release.set()


def test_waiting_subscriber_does_not_occupy_executor_thread_or_compute_slot():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    owner_started = threading.Event()
    release_owner = threading.Event()
    unrelated_started = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as compute_pool:

        def submit(compute):
            return compute_pool.submit(compute)

        def owner_compute() -> str:
            owner_started.set()
            assert release_owner.wait(3), "test did not release owner stock fit"
            return "shared-stock"

        def unrelated_compute() -> str:
            unrelated_started.set()
            return "independent-stock"

        owner = registry.get_or_submit("shared", submit=submit, compute=owner_compute)
        try:
            assert owner_started.wait(2), "owner fit did not start"
            follower = registry.get_or_submit(
                "shared",
                submit=submit,
                compute=lambda: pytest.fail("duplicate stock fit"),
            )
            independent = registry.get_or_submit(
                "independent", submit=submit, compute=unrelated_compute
            )
            assert unrelated_started.wait(2), (
                "singleflight follower occupied the only remaining executor worker"
            )
            assert independent.result(timeout=2) == "independent-stock"
            assert not owner.done()
            assert not follower.done()
        finally:
            release_owner.set()
        assert owner.result(timeout=2) == "shared-stock"
        assert follower.result(timeout=2) == "shared-stock"


def test_cancelling_one_subscriber_does_not_cancel_owner_or_other_subscribers():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    owner_started = threading.Event()
    release_owner = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as compute_pool:

        def compute() -> str:
            owner_started.set()
            assert release_owner.wait(3), "test did not release owner compute"
            return "stock-result"

        owner_view = registry.get_or_submit(
            "key", submit=compute_pool.submit, compute=compute
        )
        assert owner_started.wait(2), "owner compute was not started"
        cancelled_view = registry.get_or_submit(
            "key",
            submit=compute_pool.submit,
            compute=lambda: pytest.fail("duplicate compute"),
        )
        surviving_view = registry.get_or_submit(
            "key",
            submit=compute_pool.submit,
            compute=lambda: pytest.fail("duplicate compute"),
        )

        assert cancelled_view.cancel()
        assert not owner_view.cancelled()
        assert not surviving_view.cancelled()
        release_owner.set()
        assert owner_view.result(timeout=2) == "stock-result"
        assert surviving_view.result(timeout=2) == "stock-result"
        with pytest.raises(CancelledError):
            cancelled_view.result()


def test_cancelling_stock_owner_pair_does_not_cancel_shared_fit_for_follower(
    monkeypatch,
):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="owner-cancel-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="owner-cancel-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda value, *_args, **_kwargs: value.task_id,
    )
    monkeypatch.setattr(quality, "_async_stock_flights", quality.StockFlightRegistry())

    owner_started = threading.Event()
    follower_waiting = threading.Event()
    follower_patch_started = threading.Event()
    release_owner = threading.Event()
    owner_cancel = threading.Event()
    owner_fit_observed_cancel = []

    def fake_side(value, *, side, proposal_id, cancellation_event=None, **_kwargs):
        if proposal_id == "owner" and side == "stock":
            owner_started.set()
            assert release_owner.wait(3), "test did not release shared stock fit"
            owner_fit_observed_cancel.append(cancellation_event.is_set())
            status = "timeout" if cancellation_event.is_set() else "ok"
        else:
            if proposal_id == "follower" and side == "patch":
                follower_patch_started.set()
            status = "ok"
        return ScoreResult(
            task_id=value.task_id,
            status=status,
            score=1.0 if status == "ok" else float("nan"),
            metric_observations={"search_ran": True, "metric_valid": status == "ok"},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    scheduler = SharedPairScheduler(max_pairs=2)
    registry = load_quality_registry()

    try:
        owner_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="owner",
            cancellation_event=owner_cancel,
        )
        assert owner_started.wait(2), "shared stock owner did not start"
        follower_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="follower",
            on_side_wait=lambda *_args: follower_waiting.set(),
        )
        assert follower_waiting.wait(1), "follower did not join shared stock fit"
        assert owner_pair.cancel(), "owner proposal pair was not cancelled"
        assert owner_cancel.is_set(), (
            "cancelling the owner pair did not cancel its event"
        )
        release_owner.set()

        follower_result = follower_pair.result(timeout=2)
        assert follower_result["stock_result"].status == "ok"
        assert follower_result["patched_result"].status == "ok"
        assert follower_patch_started.is_set()
        assert owner_fit_observed_cancel == [False], (
            "cancelling one proposal must not cancel shared stock still needed by a follower"
        )
    finally:
        release_owner.set()
        scheduler.shutdown()


def test_owner_exception_reaches_all_subscribers_and_completed_flight_can_retry():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    error = RuntimeError("stock fit failed")
    owner_work: list[Future[str]] = []

    def deferred_submit(_compute):
        future: Future[str] = Future()
        owner_work.append(future)
        return future

    first = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "unused"
    )
    second = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "must-not-run"
    )
    assert len(owner_work) == 1
    owner_work[0].set_exception(error)
    with pytest.raises(RuntimeError, match="stock fit failed") as first_error:
        first.result()
    with pytest.raises(RuntimeError, match="stock fit failed") as second_error:
        second.result()
    assert first_error.value is error
    assert second_error.value is error

    retry = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "retry-result"
    )
    assert len(owner_work) == 2
    owner_work[1].set_result("retry-result")
    assert retry.result(timeout=1) == "retry-result"


def test_cancelling_owner_work_notifies_subscribers_and_removes_flight_for_retry():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    owner_work: list[Future[str]] = []

    def deferred_submit(_compute):
        future: Future[str] = Future()
        owner_work.append(future)
        return future

    owner_view = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "unused"
    )
    follower_view = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "unused"
    )
    assert len(owner_work) == 1
    assert owner_work[0].cancel()
    assert owner_view.cancelled()
    assert follower_view.cancelled()

    retry_view = registry.get_or_submit(
        "key", submit=deferred_submit, compute=lambda: "retry"
    )
    assert len(owner_work) == 2
    owner_work[1].set_result("retry")
    assert retry_view.result(timeout=1) == "retry"


def test_submit_failure_reaches_subscribers_and_does_not_leave_a_stuck_flight():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    error = RuntimeError("executor rejected stock work")

    def reject_submit(_compute):
        raise error

    first = registry.get_or_submit(
        "key", submit=reject_submit, compute=lambda: "unused"
    )
    with pytest.raises(RuntimeError, match="executor rejected stock work") as caught:
        first.result()
    assert caught.value is error

    retry = registry.get_or_submit(
        "key", submit=reject_submit, compute=lambda: "unused"
    )
    with pytest.raises(
        RuntimeError, match="executor rejected stock work"
    ) as retry_error:
        retry.result()
    assert retry_error.value is error


def test_owner_wait_telemetry_failure_does_not_leave_a_stuck_flight():
    registry: StockFlightRegistry[str] = StockFlightRegistry()
    error = OSError("failed to record queue state")

    def failed_wait_event():
        raise error

    with ThreadPoolExecutor(max_workers=1) as compute_pool:
        failed = registry.get_or_submit(
            "key",
            submit=compute_pool.submit,
            compute=lambda: "must-not-run",
            on_waiting=failed_wait_event,
        )
        with pytest.raises(OSError, match="failed to record queue state") as caught:
            failed.result(timeout=1)
        assert caught.value is error

        retry = registry.get_or_submit(
            "key", submit=compute_pool.submit, compute=lambda: "retry-result"
        )
        assert retry.result(timeout=1) == "retry-result"


def test_campaign_pair_scheduler_parks_stock_follower_and_runs_independent_dataset(
    monkeypatch,
):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.types import ScoreResult

    datasets = {
        task_id: QualityDataset(
            task_id=task_id,
            source="fedot_public_ts",
            openml_task=0,
            openml_data=0,
            name=task_id,
            problem="ts_forecasting",
            metric="smape",
            higher_is_better=False,
            min_delta=0.0,
            fold=0,
            repeat=0,
            why_full="test",
        )
        for task_id in ("task-x", "task-y")
    }
    monkeypatch.setattr(
        quality, "get_dataset", lambda task_id, _registry: datasets[task_id]
    )
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda dataset, *_args, **_kwargs: dataset.task_id,
    )

    x_stock_started = threading.Event()
    release_x_stock = threading.Event()
    y_patch_finished = threading.Event()
    lock = threading.Lock()
    calls: list[tuple[str, str]] = []
    attempts: list[int] = []
    phases: list[list[tuple[str, str, str]]] = [[], []]
    stock_flight_ids: list[list[str | None]] = [[], []]
    stock_wait_ids: list[list[str | None]] = [[], []]
    active = 0
    peak = 0

    def fake_side(dataset, *, side, attempt, on_worker_spawned=None, **_kwargs):
        nonlocal active, peak
        attempts.append(attempt)
        if on_worker_spawned is not None:
            on_worker_spawned(12345)
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((dataset.task_id, side))
        try:
            if dataset.task_id == "task-x" and side == "stock":
                x_stock_started.set()
                assert release_x_stock.wait(3), "test did not release stock owner"
            if dataset.task_id == "task-y" and side == "patch":
                y_patch_finished.set()
            return ScoreResult(
                task_id=dataset.task_id,
                status="ok",
                score=1.0,
                metric_observations={"search_ran": True, "metric_valid": True},
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    registry = load_quality_registry()
    scheduler = SharedPairScheduler[str, object](max_pairs=2, initial_groups=2)
    groups = [scheduler.register_group(), scheduler.register_group()]
    results: list[list[dict] | None] = [None, None]
    errors: list[BaseException] = []

    def record_event(index: int):
        def callback(spec, side, phase, _score, _attempt):
            phases[index].append((spec["source_dataset"], side, phase))
            if spec["source_dataset"] == "task-x" and side == "stock":
                stock_flight_ids[index].append(spec.get("stock_flight_id"))

        return callback

    def record_wait(index: int):
        def callback(spec, side, _wait_kind, _elapsed, _attempt):
            if spec["source_dataset"] == "task-x" and side == "stock":
                stock_wait_ids[index].append(spec.get("stock_flight_id"))

        return callback

    def run_group(index: int, task_ids: tuple[str, ...]) -> None:
        try:
            results[index] = quality.run_quality_jobs(
                stock_checkout=Path("/stock"),
                patch_checkout=Path("/patch"),
                task_ids=task_ids,
                n_jobs=4,
                cpu_quota=8,
                registry=registry,
                proposal_id=f"proposal-{index}",
                on_side_event=record_event(index),
                on_side_wait=record_wait(index),
                pair_scheduler=scheduler,
                pair_group_id=groups[index],
            )
        except BaseException as exc:
            errors.append(exc)

    callers = [
        threading.Thread(target=run_group, args=(0, ("task-x", "task-y")), daemon=True),
        threading.Thread(target=run_group, args=(1, ("task-x",)), daemon=True),
    ]
    try:
        for caller in callers:
            caller.start()
        assert x_stock_started.wait(2), "stock owner did not start"
        assert y_patch_finished.wait(2), (
            "independent dataset stayed blocked behind a shared-stock waiter"
        )
        assert calls.count(("task-x", "stock")) == 1
        assert ("task-x", "patch") not in calls
        release_x_stock.set()
        for caller in callers:
            caller.join(timeout=3)
        assert not any(caller.is_alive() for caller in callers)
        assert not errors
        assert calls.count(("task-x", "stock")) == 1
        assert calls.count(("task-x", "patch")) == 2
        owner_flight_id = next(
            value for value in stock_flight_ids[0] if value is not None
        )
        assert stock_wait_ids[1] and all(
            value == owner_flight_id for value in stock_wait_ids[1]
        )
        assert all(value == owner_flight_id for value in stock_flight_ids[1])
        assert peak <= 2
        assert attempts and min(attempts) == 1
        assert all(
            len(rows or []) == expected for rows, expected in zip(results, (2, 1))
        )
        assert all(
            (task, side, "finished") in events
            for events, tasks in zip(phases, (("task-x", "task-y"), ("task-x",)))
            for task in tasks
            for side in ("stock", "patch")
        )
    finally:
        release_x_stock.set()
        scheduler.shutdown()


def test_shared_stock_follower_receives_owner_timeout_after_cleanup(monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    import time

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="deadline-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="deadline-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda dataset, *_args, **_kwargs: dataset.task_id,
    )

    owner_started = threading.Event()
    release_owner = threading.Event()
    follower_waiting = threading.Event()
    follower_patch_started = threading.Event()
    follower_stock_timed_out = threading.Event()

    def fake_side(_dataset, *, side, proposal_id, **_kwargs):
        if side == "stock":
            if proposal_id == "owner":
                owner_started.set()
                assert release_owner.wait(3), (
                    "test did not release owner stock computation"
                )
                # Model the worker's bounded kill/reap/result-publication tail
                # after its own fit deadline has elapsed.
                time.sleep(0.05)
                return ScoreResult(
                    task_id=dataset.task_id,
                    status="timeout",
                    score=float("nan"),
                    detail="owner task timed out after cleanup",
                )
        elif proposal_id == "follower":
            follower_patch_started.set()
        return ScoreResult(
            task_id=dataset.task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    scheduler = SharedPairScheduler[str, object](max_pairs=2)
    registry = replace(load_quality_registry(), timeout_seconds=0.03)
    side_timeout_results: list[str] = []

    def follower_event(_spec, side, phase, score, _attempt):
        if side == "stock" and phase == "finished" and score.status == "timeout":
            side_timeout_results.append(score.detail)
            follower_stock_timed_out.set()

    try:
        owner_pair = quality._run_quality_pair_async(
            "deadline-task",
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="owner",
        )
        assert owner_started.wait(2)
        follower_pair = quality._run_quality_pair_async(
            "deadline-task",
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="follower",
            on_side_event=follower_event,
            on_side_wait=lambda *_args: follower_waiting.set(),
        )
        assert follower_waiting.wait(1)
        # The follower's own side timer and the owner's fit timer share the
        # same start/deadline. It must not publish a competing timeout while
        # the owner is still completing bounded cleanup.
        assert not follower_stock_timed_out.wait(0.08), (
            "follower published a timeout before the shared owner completed"
        )
        assert not owner_pair.done(), "follower timeout cancelled the stock owner"
        release_owner.set()
        owner_result = owner_pair.result(timeout=2)["stock_result"]
        follower_result = follower_pair.result(timeout=2)["stock_result"]
        assert follower_stock_timed_out.is_set()
        assert follower_patch_started.wait(1), (
            "follower did not continue to its patch side"
        )
        assert owner_result.status == "timeout"
        assert follower_result.status == "timeout"
        assert (
            owner_result.detail
            == follower_result.detail
            == "owner task timed out after cleanup"
        )
    finally:
        release_owner.set()
        scheduler.shutdown()
    assert side_timeout_results == ["owner task timed out after cleanup"]


def test_shared_stock_follower_deadline_starts_after_owner_queue_admission(monkeypatch):
    """Waiting behind the compute cap must not spend the owner's fit budget."""
    import time
    from dataclasses import replace
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="queued-shared-stock-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="queued-shared-stock-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda value, *_args, **_kwargs: value.task_id,
    )
    monkeypatch.setattr(quality, "_async_stock_flights", quality.StockFlightRegistry())

    blocker_started = threading.Event()
    release_blocker = threading.Event()
    owner_started = threading.Event()
    follower_waiting = threading.Event()
    follower_stock_timed_out = threading.Event()
    follower_wait_phases: list[tuple[str, float]] = []
    owner_wait_phases: list[tuple[str, float]] = []

    def fake_side(value, *, side, proposal_id, **_kwargs):
        if proposal_id == "owner" and side == "stock":
            owner_started.set()
        return ScoreResult(
            task_id=value.task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    scheduler = SharedPairScheduler(max_pairs=1)
    registry = replace(load_quality_registry(), timeout_seconds=0.03)
    blocker = scheduler.submit_compute(
        lambda _cancel=None: (
            blocker_started.set(),
            release_blocker.wait(2),
            "released",
        )[-1],
        cooperative_cancellation=True,
    )

    def follower_event(_spec, side, phase, score, _attempt):
        if side == "stock" and phase == "finished" and score.status == "timeout":
            follower_stock_timed_out.set()

    def follower_wait(_spec, _side, wait_kind, elapsed, _attempt):
        follower_wait_phases.append((wait_kind, elapsed))
        follower_waiting.set()

    try:
        assert blocker_started.wait(2), "compute slot was not occupied"
        owner_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="owner",
            on_side_wait=lambda _spec,
            _side,
            kind,
            elapsed,
            _attempt: owner_wait_phases.append((kind, elapsed)),
        )
        assert owner_wait_phases and owner_wait_phases[0][0] == "scheduler_queue", (
            "the queued shared-stock owner must report queue wait before FEDOT starts"
        )
        follower_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="follower",
            on_side_event=follower_event,
            on_side_wait=follower_wait,
        )
        assert follower_waiting.wait(1)
        assert follower_wait_phases[0][0] == "scheduler_queue"
        # This elapsed time is queueing only: the shared stock owner has no
        # compute slot yet and FEDOT has not begun its fit.
        time.sleep(0.08)
        assert not follower_stock_timed_out.is_set(), (
            "shared-stock waiter timed out before its owner was admitted"
        )

        release_blocker.set()
        assert blocker.result(timeout=2) == "released"
        assert owner_started.wait(2), "owner was not admitted to the compute pool"
        assert any(kind == "shared_stock_fit" for kind, _ in owner_wait_phases), (
            "owner wait telemetry did not switch to admitted fit time"
        )
        assert any(kind == "shared_stock_fit" for kind, _ in follower_wait_phases), (
            "wait telemetry did not switch from scheduler queue to owner fit"
        )
        assert owner_pair.result(timeout=2)["stock_result"].status == "ok"
        assert follower_pair.result(timeout=2)["stock_result"].status == "ok"
        assert not follower_stock_timed_out.is_set()
    finally:
        release_blocker.set()
        scheduler.shutdown()


def test_subscriber_arriving_after_stock_admission_reports_active_fit_elapsed(
    monkeypatch,
):
    from dataclasses import replace
    from pathlib import Path
    import time

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="already-admitted-stock-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="already-admitted-stock-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda value, *_args, **_kwargs: value.task_id,
    )
    monkeypatch.setattr(quality, "_async_stock_flights", quality.StockFlightRegistry())

    owner_started = threading.Event()
    release_owner = threading.Event()
    wait_events: list[tuple[str, float]] = []

    def fake_side(value, *, side, proposal_id, **_kwargs):
        if proposal_id == "owner" and side == "stock":
            owner_started.set()
            assert release_owner.wait(3), "test did not release admitted owner"
        return ScoreResult(
            task_id=value.task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    scheduler = SharedPairScheduler(max_pairs=2)
    registry = replace(load_quality_registry(), timeout_seconds=0.03)

    try:
        owner_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="owner",
        )
        assert owner_started.wait(2), "stock owner was not admitted"
        time.sleep(0.05)
        follower_pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="follower",
            on_side_wait=lambda _spec,
            _side,
            kind,
            elapsed,
            _attempt: wait_events.append((kind, elapsed)),
        )
        assert wait_events, "subscribing waiter emitted no initial state"
        assert wait_events[0][0] == "shared_stock_fit"
        assert wait_events[0][1] >= 0.03, "wait timer did not start at owner admission"
        release_owner.set()
        assert owner_pair.result(timeout=2)["stock_result"].status == "ok"
        assert follower_pair.result(timeout=2)["stock_result"].status == "ok"
    finally:
        release_owner.set()
        scheduler.shutdown()


def test_spawned_stock_owner_stops_wait_heartbeat_without_stopping_subscriber(
    monkeypatch,
):
    """The actual worker owner retains spawned state while followers wait."""
    from pathlib import Path

    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        get_dataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    registry = load_quality_registry()
    dataset = get_dataset("fedot-ts-beer", registry)
    timers, pending = [], []
    events = {"owner": [], "subscriber": []}

    class ManualTimer:
        def __init__(self, _interval, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(quality.threading, "Timer", ManualTimer)
    monkeypatch.setattr(quality, "_async_stock_flights", StockFlightRegistry())
    monkeypatch.setattr(
        quality, "_async_stock_flight_key", lambda *_args, **_kwargs: "beer"
    )
    monkeypatch.setattr(
        quality, "_stock_cache_path", lambda *_args, **_kwargs: Path("/cache")
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )

    def run_side(value, *, side, on_worker_spawned, **_kwargs):
        on_worker_spawned(123)
        if side == "stock":
            timers[0].callback()
            timers[1].callback()
        return ScoreResult(
            task_id=value.task_id,
            status="ok",
            score=0.75,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", run_side)

    def submit(compute, **_kwargs):
        future = Future()
        pending.append((future, compute))
        return future

    pairs = []
    for proposal in ("owner", "subscriber"):
        pairs.append(
            quality._run_quality_pair_async(
                dataset,
                stock_checkout=Path("/stock"),
                patch_checkout=Path("/patch"),
                submit_compute=submit,
                n_jobs=4,
                registry=registry,
                proposal_id=proposal,
                on_side_event=lambda _spec,
                side,
                phase,
                _result,
                _attempt,
                proposal=proposal: events[proposal].append((side, phase)),
                on_side_wait=lambda _spec,
                side,
                kind,
                _elapsed,
                _attempt,
                proposal=proposal: events[proposal].append((side, kind)),
            )
        )
    assert len(pending) == 1
    stock_future, stock_compute = pending[0]
    stock_result = stock_compute()
    assert events["owner"][-1] == ("stock", "spawned")
    assert events["subscriber"][-1] == ("stock", "shared_stock_fit")
    stock_future.set_result(stock_result)
    for future, compute in pending[1:]:
        future.set_result(compute())
    assert all(pair.result(timeout=1)["stock_result"].status == "ok" for pair in pairs)


def test_stock_without_cache_completes_pair_after_owner_future_finishes(monkeypatch):
    """Unshared stock completion must publish its result and schedule patch."""
    from pathlib import Path

    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        get_dataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    registry = load_quality_registry()
    dataset = get_dataset("fedot-ts-beer", registry)
    pending = []
    phases = []
    monkeypatch.setattr(quality, "_stock_cache_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )
    monkeypatch.setattr(
        quality,
        "run_quality_side",
        lambda value, **_kwargs: ScoreResult(
            task_id=value.task_id,
            status="ok",
            score=0.75,
            metric_observations={"search_ran": True, "metric_valid": True},
        ),
    )

    def submit(compute, **_kwargs):
        future = Future()
        pending.append((future, compute))
        return future

    pair = quality._run_quality_pair_async(
        dataset,
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        submit_compute=submit,
        n_jobs=4,
        registry=registry,
        proposal_id="uncached-stock",
        on_side_event=lambda _spec, side, phase, _result, _attempt: phases.append(
            (side, phase)
        ),
    )
    stock_future, stock_compute = pending[0]
    stock_future.set_result(stock_compute())
    assert stock_future.done()
    assert ("stock", "finished") in phases, "completed stock result was not published"
    assert len(pending) == 2, "completed stock result did not schedule patch"
    patch_future, patch_compute = pending[1]
    patch_future.set_result(patch_compute())
    assert pair.done(), "completed side futures left the pair pending"
    pair_result = pair.result(timeout=1)
    assert pair_result["stock_result"].status == "ok"
    assert pair_result["patched_result"].status == "ok"
    assert not pair_result["stock_result"].metric_observations.get(
        "singleflight_shared"
    )


def test_compute_queue_wait_does_not_consume_side_timeout(monkeypatch):
    """A side gets its timeout after a real compute slot is available."""
    import time
    from dataclasses import replace
    from pathlib import Path

    from fedotllm.agents.evolve.evaluation import fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        get_dataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    registry = replace(load_quality_registry(), timeout_seconds=0.1)
    dataset = get_dataset("fedot-ts-beer", registry)
    observed_deadline_remaining = []
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    monkeypatch.setattr(
        quality, "_stock_cache_path", lambda *_args, **_kwargs: Path("/cache")
    )
    monkeypatch.setattr(
        quality, "_async_stock_flight_key", lambda *_args, **_kwargs: "beer"
    )
    monkeypatch.setattr(quality, "_async_stock_flights", quality.StockFlightRegistry())
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda value, *, n_jobs, registry: {
            "source_dataset": value.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": value.higher_is_better,
        },
    )

    def fake_run_quality_side(
        value, *, side, _side_deadline=None, _side_started=None, **_kwargs
    ):
        observed_deadline_remaining.append(_side_deadline - time.monotonic())
        return ScoreResult(
            task_id=value.task_id,
            status="ok",
            score=0.75,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_run_quality_side)
    scheduler = SharedPairScheduler(max_pairs=1)

    def blocker(_cancel=None):
        blocker_started.set()
        assert release_blocker.wait(2), "test did not release occupied compute slot"
        return "released"

    occupied = scheduler.submit_compute(blocker, cooperative_cancellation=True)
    try:
        assert blocker_started.wait(2), "compute slot was not occupied"
        pair = quality._run_quality_pair_async(
            dataset,
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="queued-side-timeout",
        )
        # Wait longer than this test's side timeout while the task is queued.
        time.sleep(0.15)
        release_blocker.set()
        assert occupied.result(timeout=2) == "released"
        pair_result = pair.result(timeout=2)
        assert pair_result["stock_result"].status == "ok"
        assert pair_result["patched_result"].status == "ok"
        assert len(observed_deadline_remaining) == 2
        assert all(remaining > 0.05 for remaining in observed_deadline_remaining)
    finally:
        release_blocker.set()
        scheduler.shutdown()


def test_shared_stock_timeout_and_owner_completion_publish_one_terminal_event(
    monkeypatch,
):
    from dataclasses import replace
    from pathlib import Path
    import time

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="terminal-race-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="terminal-race-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda dataset, *_args, **_kwargs: dataset.task_id,
    )

    owner_started = threading.Event()
    release_owner = threading.Event()
    owner_compute: Future[ScoreResult] = Future()
    owner_threads = []

    def submit_owner(compute, **_kwargs):
        def run_owner():
            try:
                owner_compute.set_result(compute())
            except BaseException as exc:
                owner_compute.set_exception(exc)

        thread = threading.Thread(target=run_owner, daemon=True)
        owner_threads.append(thread)
        thread.start()
        return owner_compute

    def fake_side(_dataset, *, side, proposal_id, **_kwargs):
        if side == "stock" and proposal_id == "owner":
            owner_started.set()
            assert release_owner.wait(3), "test did not release owner stock computation"
            time.sleep(0.05)
            return ScoreResult(
                task_id=dataset.task_id,
                status="timeout",
                score=float("nan"),
                detail="owner task timed out after cleanup",
            )
        return ScoreResult(
            task_id=dataset.task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    scheduler = SharedPairScheduler[str, object](max_pairs=2)
    registry = replace(load_quality_registry(), timeout_seconds=5.0)
    terminal_events: list[ScoreResult] = []

    def record_follower_event(_spec, side, phase, score, _attempt):
        if side == "stock" and phase == "finished":
            terminal_events.append(score)

    try:
        owner_pair = quality._run_quality_pair_async(
            "terminal-race-task",
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=submit_owner,
            n_jobs=4,
            registry=registry,
            proposal_id="owner",
        )
        follower_pair = quality._run_quality_pair_async(
            "terminal-race-task",
            stock_checkout=Path("/stock"),
            patch_checkout=Path("/patch"),
            submit_compute=scheduler.submit_compute,
            n_jobs=4,
            registry=registry,
            proposal_id="follower",
            on_side_event=record_follower_event,
            on_side_wait=lambda *_args: None,
        )
        assert owner_started.wait(2)

        release_owner.set()
        owner_timeout = ScoreResult(
            task_id=dataset.task_id,
            status="timeout",
            score=float("nan"),
            detail="owner task timed out after cleanup",
        )
        follower_result = follower_pair.result(timeout=2)["stock_result"]
        owner_result = owner_pair.result(timeout=2)["stock_result"]
        assert follower_result.status == owner_result.status == "timeout"
        assert follower_result.detail == owner_result.detail == owner_timeout.detail
        assert len(terminal_events) == 1
        assert terminal_events[0].status == "timeout"
        assert terminal_events[0].detail == owner_timeout.detail
    finally:
        release_owner.set()
        scheduler.shutdown()
        for thread in owner_threads:
            thread.join(timeout=2)


def test_cancel_snapshot_is_synchronized_with_side_finalize(monkeypatch):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.types import ScoreResult

    dataset = QualityDataset(
        task_id="cancel-finalize-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="cancel-finalize-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    source_future: Future[ScoreResult] = Future()
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    finalize_started = threading.Event()
    finalize_finished = threading.Event()
    iteration_errors: list[BaseException] = []
    original_tuple = tuple

    def controlled_tuple(values):
        if (
            isinstance(values, set)
            and values
            and isinstance(next(iter(values)), Future)
        ):
            iterator = iter(values)
            first = next(iterator)
            snapshot_entered.set()
            assert release_snapshot.wait(3), "test did not release active-side snapshot"
            try:
                return (first, *iterator)
            except BaseException as exc:
                iteration_errors.append(exc)
                raise
        return original_tuple(values)

    monkeypatch.setattr(quality, "tuple", controlled_tuple, raising=False)
    monkeypatch.setattr(
        quality,
        "run_quality_side",
        lambda *_args, **_kwargs: ScoreResult(
            task_id=dataset.task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        ),
    )
    monkeypatch.setattr(quality, "_async_stock_flights", quality.StockFlightRegistry())

    pair = quality._run_quality_pair_async(
        "cancel-finalize-task",
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        submit_compute=lambda _compute, **_kwargs: source_future,
        n_jobs=4,
        registry=load_quality_registry(),
        proposal_id="cancel-finalize",
    )

    cancel_thread = threading.Thread(target=pair.cancel, daemon=True)

    def complete_source():
        finalize_started.set()
        source_future.set_result(
            ScoreResult(
                task_id=dataset.task_id,
                status="ok",
                score=1.0,
                metric_observations={"search_ran": True, "metric_valid": True},
            )
        )
        finalize_finished.set()

    try:
        cancel_thread.start()
        assert snapshot_entered.wait(2)
        finalize_thread = threading.Thread(target=complete_source, daemon=True)
        finalize_thread.start()
        assert finalize_started.wait(2)
        assert not finalize_finished.wait(0.05), (
            "finalize mutated active views during cancel snapshot"
        )
        release_snapshot.set()
        cancel_thread.join(timeout=2)
        finalize_thread.join(timeout=2)
        assert not cancel_thread.is_alive()
        assert not finalize_thread.is_alive()
        assert not iteration_errors
        assert pair.cancelled()
    finally:
        release_snapshot.set()


def test_async_quality_side_observes_scheduler_fail_closed_without_mutating_local_event(
    monkeypatch,
):
    from concurrent.futures import CancelledError
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler

    dataset = QualityDataset(
        task_id="cooperative-cancel-task",
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name="cooperative-cancel-task",
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(quality, "_stock_cache_path", lambda *_args, **_kwargs: None)
    local_cancel = threading.Event()
    compute_started = threading.Event()
    observed_tokens = []

    def wait_for_fatal(_dataset, *, cancellation_event, **_kwargs):
        observed_tokens.append(cancellation_event)
        compute_started.set()
        assert cancellation_event.wait(2), (
            "compute did not receive scheduler fatal cancellation"
        )
        raise CancelledError("scheduler failed closed")

    monkeypatch.setattr(quality, "run_quality_side", wait_for_fatal)
    scheduler = SharedPairScheduler[str, object](max_pairs=1)
    pair = quality._run_quality_pair_async(
        "cooperative-cancel-task",
        stock_checkout=Path("/stock"),
        patch_checkout=Path("/patch"),
        submit_compute=scheduler.submit_compute,
        n_jobs=1,
        registry=load_quality_registry(),
        cancellation_event=local_cancel,
        proposal_id="cooperative-cancel",
    )
    fatal = RuntimeError("journal failure")
    try:
        assert compute_started.wait(2)
        scheduler.fail_closed(fatal)
        with pytest.raises(CancelledError):
            pair.result(timeout=2)
        assert len(observed_tokens) == 1
        assert observed_tokens[0] is not local_cancel
        assert observed_tokens[0].is_set()
        assert not local_cancel.is_set()
    finally:
        scheduler.shutdown()


def test_seven_proposals_share_one_stock_then_run_all_patch_sides(monkeypatch):
    from pathlib import Path

    import fedotllm.agents.evolve.evaluation.fedot_quality as quality
    from fedotllm.agents.evolve.evaluation.quality_registry import (
        QualityDataset,
        load_quality_registry,
    )
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
    from fedotllm.agents.evolve.types import ScoreResult

    task_id = "seven-proposal-task"
    dataset = QualityDataset(
        task_id=task_id,
        source="fedot_public_ts",
        openml_task=0,
        openml_data=0,
        name=task_id,
        problem="ts_forecasting",
        metric="smape",
        higher_is_better=False,
        min_delta=0.0,
        fold=0,
        repeat=0,
        why_full="test",
    )
    monkeypatch.setattr(quality, "get_dataset", lambda _task_id, _registry: dataset)
    monkeypatch.setattr(
        quality,
        "job_spec",
        lambda dataset, *, n_jobs, registry: {
            "source_dataset": dataset.task_id,
            "n_jobs": n_jobs,
            "higher_is_better": dataset.higher_is_better,
        },
    )
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: "/fake/python")
    monkeypatch.setattr(
        quality,
        "_async_stock_flight_key",
        lambda dataset, *_args, **_kwargs: dataset.task_id,
    )

    proposal_count = 7
    all_started = threading.Barrier(proposal_count)
    owner_started = threading.Event()
    release_owner = threading.Event()
    all_waiters_subscribed = threading.Event()
    all_coordination_groups_finished = threading.Event()
    all_patches_finished = threading.Event()
    lock = threading.Lock()
    stock_fits = 0
    patch_fits = 0
    waiter_count = 0
    phases: list[list[tuple[str, str, str]]] = [[] for _ in range(proposal_count)]
    results: list[list[dict] | None] = [None] * proposal_count
    errors: list[BaseException] = []

    def fake_side(_dataset, *, side, proposal_id, on_worker_spawned=None, **_kwargs):
        nonlocal stock_fits, patch_fits
        if on_worker_spawned is not None:
            on_worker_spawned(12345)
        if side == "stock":
            with lock:
                stock_fits += 1
            owner_started.set()
            assert release_owner.wait(4), "test did not release stock owner"
        else:
            with lock:
                patch_fits += 1
                if patch_fits == proposal_count:
                    all_patches_finished.set()
        return ScoreResult(
            task_id=task_id,
            status="ok",
            score=1.0,
            metric_observations={"search_ran": True, "metric_valid": True},
        )

    monkeypatch.setattr(quality, "run_quality_side", fake_side)
    registry = load_quality_registry()

    class ObservedScheduler(SharedPairScheduler[str, object]):
        def submit_group(self, *args, **kwargs):
            nonlocal coordination_groups_finished
            group_future = super().submit_group(*args, **kwargs)

            def group_done(_future):
                nonlocal coordination_groups_finished
                with lock:
                    coordination_groups_finished += 1
                    if coordination_groups_finished == proposal_count:
                        all_coordination_groups_finished.set()

            group_future.add_done_callback(group_done)
            return group_future

    coordination_groups_finished = 0
    scheduler = ObservedScheduler(
        max_pairs=proposal_count, initial_groups=proposal_count
    )
    groups = [scheduler.register_group() for _ in range(proposal_count)]

    def on_wait(*_args):
        nonlocal waiter_count
        with lock:
            waiter_count += 1
            if waiter_count == proposal_count - 1:
                all_waiters_subscribed.set()

    def on_event(index: int):
        def record(spec, side, phase, _score, _attempt):
            if side == "stock" and phase == "started":
                all_started.wait(timeout=3)
            phases[index].append((spec["source_dataset"], side, phase))

        return record

    def run_proposal(index: int) -> None:
        try:
            results[index] = quality.run_quality_jobs(
                stock_checkout=Path("/stock"),
                patch_checkout=Path("/patch"),
                task_ids=(task_id,),
                n_jobs=4,
                cpu_quota=28,
                registry=registry,
                proposal_id=f"proposal-{index}",
                on_side_event=on_event(index),
                on_side_wait=on_wait,
                pair_scheduler=scheduler,
                pair_group_id=groups[index],
            )
        except BaseException as exc:
            errors.append(exc)

    callers = [
        threading.Thread(target=run_proposal, args=(index,), daemon=True)
        for index in range(proposal_count)
    ]
    try:
        for caller in callers:
            caller.start()
        assert owner_started.wait(2), "stock owner did not start"
        assert all_waiters_subscribed.wait(2), (
            "six proposals did not subscribe to stock owner"
        )
        assert all_coordination_groups_finished.wait(2), (
            "pair coordination callbacks waited on their stock Futures"
        )
        assert scheduler.active_pair_slots == 0, (
            "stock waiters retained proposal pair slots"
        )
        assert scheduler.active_compute_workers == 1
        with lock:
            assert stock_fits == 1
            assert patch_fits == 0, "patch began before stock had an outcome"
        release_owner.set()
        assert all_patches_finished.wait(3), (
            "not every proposal reached patch evaluation"
        )
        for caller in callers:
            caller.join(timeout=3)
        assert not any(caller.is_alive() for caller in callers)
        assert not errors
        assert all(len(rows or []) == 1 for rows in results)
        assert scheduler.peak_compute_workers <= proposal_count
        with lock:
            assert stock_fits == 1
            assert patch_fits == proposal_count
        for events in phases:
            assert (task_id, "stock", "started") in events
            assert (task_id, "stock", "finished") in events
            assert (task_id, "patch", "started") in events
            assert (task_id, "patch", "finished") in events
    finally:
        release_owner.set()
        scheduler.shutdown()
