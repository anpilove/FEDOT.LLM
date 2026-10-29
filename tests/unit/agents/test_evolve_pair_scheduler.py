"""Concurrency contract for the shared quality pair budget."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler


def test_shared_pair_scheduler_never_exceeds_seven_slots_and_uses_all_available_slots():
    scheduler = SharedPairScheduler[int, int](max_pairs=7)
    lock = threading.Lock()
    active = 0
    peak = 0
    started = 0
    seven_started = threading.Event()
    release = threading.Event()

    def run_one(value: int) -> int:
        nonlocal active, peak, started
        with lock:
            active += 1
            started += 1
            peak = max(peak, active)
            if active == 7:
                seven_started.set()
        assert release.wait(5), "test barrier was not released"
        with lock:
            active -= 1
        return value * 2

    try:
        result = scheduler.submit_group(tuple(range(10)), run_one)
        assert seven_started.wait(3), (
            "scheduler did not fill its configured pair budget"
        )
        with lock:
            assert active == 7
            assert started == 7
        assert scheduler.active_pair_slots == 7
        assert scheduler.max_pairs == 7
        release.set()
        assert result.result(timeout=5) == [value * 2 for value in range(10)]
        assert peak == 7
        assert started == 10
        assert scheduler.active_pair_slots == 0
    finally:
        release.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_round_robins_waiting_proposal_after_slot_release():
    scheduler = SharedPairScheduler[str, str](max_pairs=2)
    a_first_started = threading.Event()
    a_second_started = threading.Event()
    a_third_started = threading.Event()
    b_started = threading.Event()
    release_a_first = threading.Event()
    release_a_second = threading.Event()
    release_a_third = threading.Event()

    def run_a(task: str) -> str:
        if task == "a0":
            a_first_started.set()
            assert release_a_first.wait(5)
        elif task == "a1":
            a_second_started.set()
            assert release_a_second.wait(5)
        elif task == "a2":
            a_third_started.set()
            assert release_a_third.wait(5)
        return task

    def run_b(task: str) -> str:
        b_started.set()
        return task

    try:
        a_result = scheduler.submit_group(("a0", "a1", "a2", "a3"), run_a)
        assert a_first_started.wait(2)
        assert a_second_started.wait(2)
        b_result = scheduler.submit_group(("b0",), run_b)

        release_a_first.set()
        assert a_third_started.wait(2)
        release_a_second.set()
        assert b_started.wait(2), (
            "queued proposal did not receive the next released slot"
        )
        release_a_third.set()

        assert b_result.result(timeout=2) == ["b0"]
        assert a_result.result(timeout=2) == ["a0", "a1", "a2", "a3"]
    finally:
        release_a_first.set()
        release_a_second.set()
        release_a_third.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_waits_for_initial_proposals_before_dispatching():
    scheduler = SharedPairScheduler[str, str](max_pairs=2, initial_groups=2)
    a_started = threading.Event()
    b_started = threading.Event()
    release = threading.Event()

    def run_a(task: str) -> str:
        a_started.set()
        assert release.wait(3)
        return task

    def run_b(task: str) -> str:
        b_started.set()
        assert release.wait(3)
        return task

    try:
        a_group = scheduler.register_group()
        a_result = scheduler.submit_group(("a0", "a1"), run_a, group_id=a_group)
        assert not a_started.wait(0.05), (
            "first proposal dispatched before its peers were ready"
        )

        b_group = scheduler.register_group()
        b_result = scheduler.submit_group(("b0",), run_b, group_id=b_group)
        assert a_started.wait(2)
        assert b_started.wait(2), (
            "initial dispatch did not share slots across proposals"
        )

        release.set()
        assert a_result.result(timeout=2) == ["a0", "a1"]
        assert b_result.result(timeout=2) == ["b0"]
    finally:
        release.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_skipped_initial_proposal_releases_dispatch_barrier():
    scheduler = SharedPairScheduler[str, str](max_pairs=2, initial_groups=2)
    a_started = threading.Event()
    release = threading.Event()

    def run_a(task: str) -> str:
        a_started.set()
        assert release.wait(3)
        return task

    try:
        a_group = scheduler.register_group()
        a_result = scheduler.submit_group(("a0",), run_a, group_id=a_group)
        assert not a_started.wait(0.05)

        skipped_group = scheduler.register_group()
        scheduler.skip_group(skipped_group)
        assert a_started.wait(2), (
            "a proposal that produces no evaluation kept the barrier closed"
        )

        release.set()
        assert a_result.result(timeout=2) == ["a0"]
    finally:
        release.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_handles_task_completed_before_callback_registration(
    monkeypatch,
):
    import fedotllm.agents.evolve.evaluation.pair_scheduler as scheduler_module

    class ImmediateExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, fn, *args):
            future = Future()
            try:
                future.set_result(fn(*args))
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(scheduler_module, "ThreadPoolExecutor", ImmediateExecutor)
    scheduler = SharedPairScheduler[str, str](max_pairs=1)
    result: list[list[str]] = []
    runner = threading.Thread(
        target=lambda: result.append(
            scheduler.run_group(("instant",), lambda item: item)
        ),
        daemon=True,
    )

    runner.start()
    runner.join(timeout=0.5)
    assert not runner.is_alive(), (
        "completed future callback deadlocked trying to reacquire scheduler lock"
    )
    assert result == [["instant"]]
    scheduler.shutdown()


def test_shared_pair_scheduler_dispatches_when_initial_peer_has_no_work():
    scheduler = SharedPairScheduler[str, str](max_pairs=1, initial_groups=2)
    started = threading.Event()
    release = threading.Event()

    def run_one(task: str) -> str:
        started.set()
        assert release.wait(3)
        return task

    try:
        active_group = scheduler.register_group()
        active_result = scheduler.submit_group(
            ("work",), run_one, group_id=active_group
        )
        empty_group = scheduler.register_group()
        empty_result = scheduler.submit_group((), run_one, group_id=empty_group)

        assert empty_result.result(timeout=1) == []
        assert started.wait(1), (
            "empty initial group opened the barrier without dispatching ready work"
        )
        release.set()
        assert active_result.result(timeout=1) == ["work"]
    finally:
        release.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_fails_closed_on_proposal_preparation_error():
    scheduler = SharedPairScheduler[str, str](max_pairs=1, initial_groups=2)
    started: list[str] = []
    preparation_error = OSError(28, "No space left on device")

    try:
        failed_group = scheduler.register_group()
        waiting_group = scheduler.register_group()
        waiting_result = scheduler.submit_group(
            ("must-not-start",),
            lambda item: started.append(item) or item,
            group_id=waiting_group,
        )

        scheduler.fail_closed(preparation_error)
        scheduler.skip_group(failed_group)

        with pytest.raises(OSError, match="No space left"):
            waiting_result.result(timeout=1)
        assert started == []
    finally:
        scheduler.shutdown()


def test_shared_pair_scheduler_waits_until_running_compute_callback_exits():
    scheduler = SharedPairScheduler[str, str](max_pairs=1)
    started = threading.Event()
    release = threading.Event()

    def compute():
        started.set()
        assert release.wait(3)
        return "finished"

    try:
        future = scheduler.submit_compute(compute)
        assert started.wait(1)
        waiter_done = threading.Event()

        def wait_for_idle():
            scheduler.wait_for_compute_idle(timeout=2)
            waiter_done.set()

        waiter = threading.Thread(target=wait_for_idle, daemon=True)
        waiter.start()
        assert not waiter_done.wait(0.05), (
            "idle barrier returned while a worker still used its checkout"
        )
        release.set()
        assert waiter_done.wait(1), (
            "idle barrier did not wake after the worker completed"
        )
        assert future.result(timeout=1) == "finished"
    finally:
        release.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_does_not_start_admitted_task_after_fail_closed(
    monkeypatch,
):
    import fedotllm.agents.evolve.evaluation.pair_scheduler as scheduler_module

    class DeferredExecutor:
        def __init__(self, **_kwargs):
            self.pending = []

        def submit(self, fn, *args):
            future = Future()
            self.pending.append((fn, args, future))
            return future

        def shutdown(self, **_kwargs):
            return None

    executor = DeferredExecutor()
    monkeypatch.setattr(
        scheduler_module, "ThreadPoolExecutor", lambda **_kwargs: executor
    )
    scheduler = SharedPairScheduler[str, str](max_pairs=1, initial_groups=1)
    started: list[str] = []
    error = OSError(28, "No space left on device")

    try:
        group_id = scheduler.register_group()
        result = scheduler.submit_group(
            ("late-worker",),
            lambda task: started.append(task) or task,
            group_id=group_id,
        )
        assert len(executor.pending) == 1

        scheduler.fail_closed(error)
        _fn, _args, worker_future = executor.pending.pop()

        with pytest.raises(OSError, match="No space left"):
            result.result(timeout=1)
        assert worker_future.cancelled()
        assert started == []
    finally:
        scheduler.shutdown()


def test_shared_pair_scheduler_opens_start_barrier_after_bounded_wait():
    scheduler = SharedPairScheduler[str, str](
        max_pairs=1, initial_groups=2, initial_group_wait_seconds=0.02
    )
    started = threading.Event()
    release = threading.Event()

    def run_one(task: str) -> str:
        started.set()
        assert release.wait(3)
        return task

    try:
        group_id = scheduler.register_group()
        result = scheduler.submit_group(("ready-work",), run_one, group_id=group_id)
        assert started.wait(1), (
            "missing initial proposal prevented the bounded start gate from opening"
        )
        release.set()
        assert result.result(timeout=1) == ["ready-work"]
    finally:
        release.set()
        scheduler.shutdown()


def test_pair_scheduler_fails_closed_after_worker_exception():
    scheduler = SharedPairScheduler[str, str](max_pairs=2)
    fail_now = threading.Event()
    sibling_started = threading.Event()
    release_sibling = threading.Event()
    started: list[str] = []
    started_lock = threading.Lock()

    def run_a(task: str) -> str:
        with started_lock:
            started.append(task)
        if task == "fail":
            assert fail_now.wait(3)
            raise RuntimeError("durable event write failed")
        if task == "sibling":
            sibling_started.set()
            assert release_sibling.wait(3)
        return task

    def run_b(task: str) -> str:
        with started_lock:
            started.append(task)
        return task

    try:
        a_result = scheduler.submit_group(("fail", "sibling", "queued-a"), run_a)
        assert sibling_started.wait(2)
        b_result = scheduler.submit_group(("queued-b",), run_b)
        fail_now.set()
        deadline = time.monotonic() + 2
        while scheduler.fatal_error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert scheduler.fatal_error is not None, (
            "scheduler did not observe the worker failure"
        )
        release_sibling.set()

        with pytest.raises(RuntimeError, match="durable event write failed"):
            a_result.result(timeout=2)
        with pytest.raises(RuntimeError, match="durable event write failed"):
            b_result.result(timeout=2)
        assert set(started) == {"fail", "sibling"}
    finally:
        fail_now.set()
        release_sibling.set()
        scheduler.shutdown()


def test_shared_pair_scheduler_can_submit_compute_without_pair_callbacks_waiting_for_it():
    scheduler = SharedPairScheduler[str, str](max_pairs=2)
    owner_started = threading.Event()
    release_owner = threading.Event()
    independent_started = threading.Event()

    def owner_compute():
        owner_started.set()
        assert release_owner.wait(3)
        return "owner"

    def independent_compute():
        independent_started.set()
        return "independent"

    try:
        owner = scheduler.submit_compute(owner_compute)
        assert owner_started.wait(2)
        independent = scheduler.submit_compute(independent_compute)
        assert independent_started.wait(2), "compute executor did not use its free slot"
        assert independent.result(timeout=1) == "independent"
    finally:
        release_owner.set()
        scheduler.shutdown()
    assert owner.result(timeout=1) == "owner"


def test_fail_closed_cooperatively_cancels_compute_and_prevents_queued_start():
    scheduler = SharedPairScheduler[str, str](max_pairs=1)
    user_cancel = threading.Event()
    active_started = threading.Event()
    cancellation_observed = threading.Event()
    queued_started = threading.Event()

    def active_compute(scheduler_cancel):
        active_started.set()
        if scheduler_cancel.wait(timeout=3):
            cancellation_observed.set()
        return "stopped"

    try:
        active = scheduler.submit_compute(
            active_compute,
            cancellation_event=user_cancel,
            cooperative_cancellation=True,
        )
        assert active_started.wait(2)
        queued = scheduler.submit_compute(
            lambda _cancel: queued_started.set() or "queued",
            cooperative_cancellation=True,
        )

        fatal = OSError(28, "journal persistence failed")
        scheduler.fail_closed(fatal)

        assert cancellation_observed.wait(2), (
            "active compute did not see fatal cancellation"
        )
        assert active.result(timeout=1) == "stopped"
        assert queued.cancelled(), "queued compute was not best-effort cancelled"
        assert not queued_started.is_set(), "queued compute started after fail-closed"
        assert not user_cancel.is_set(), (
            "scheduler mutated a caller-owned cancellation event"
        )
        rejected = scheduler.submit_compute(
            lambda _cancel: "late", cooperative_cancellation=True
        )
        with pytest.raises(OSError, match="journal persistence failed"):
            rejected.result(timeout=1)
    finally:
        scheduler.shutdown()
