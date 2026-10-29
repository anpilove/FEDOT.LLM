"""Fair bounded scheduler for complete stock/patch dataset pairs."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Event, RLock, Timer
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


class _CombinedCancellationEvent:
    """Expose scheduler fatal state without mutating the caller's Event."""

    def __init__(self, local: Event | None, scheduler: Event) -> None:
        self._local = local
        self._scheduler = scheduler
        self._wake = Event()

    def is_set(self) -> bool:
        return self._scheduler.is_set() or bool(self._local and self._local.is_set())

    def set(self) -> None:
        if self._local is not None:
            self._local.set()
        self._wake.set()

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        self._wake.wait(timeout)
        return self.is_set()

    def _scheduler_cancelled(self) -> None:
        self._wake.set()


@dataclass
class _Group(Generic[T, R]):
    key: int
    items: tuple[T, ...]
    run_one: Callable[[T], R]
    result: Future[list[R]]
    values: list[R | None]
    next_index: int = 0
    active: int = 0
    queued: bool = False


class SharedPairScheduler(Generic[T, R]):
    """Share a fixed number of pair workers across proposal-scoped queues.

    Each group is a proposal's ordered dataset-pair list. Groups are served in
    round-robin order. Campaign callers may make ``run_one`` a short
    coordination callback that returns a Future, then use ``submit_compute``
    for stock and patch work. Both use this scheduler's one bounded executor,
    while a shared-stock subscriber can release its pair slot immediately.
    """

    def __init__(
        self,
        max_pairs: int,
        *,
        initial_groups: int = 0,
        initial_group_wait_seconds: float = 10.0,
    ) -> None:
        if (
            isinstance(max_pairs, bool)
            or not isinstance(max_pairs, int)
            or max_pairs <= 0
        ):
            raise ValueError("max_pairs must be a positive integer")
        if (
            isinstance(initial_groups, bool)
            or not isinstance(initial_groups, int)
            or initial_groups < 0
        ):
            raise ValueError("initial_groups must be a non-negative integer")
        if (
            isinstance(initial_group_wait_seconds, bool)
            or not isinstance(initial_group_wait_seconds, (int, float))
            or initial_group_wait_seconds < 0
        ):
            raise ValueError("initial_group_wait_seconds must be a non-negative number")
        self.max_pairs = max_pairs
        self.initial_groups = initial_groups
        self._executor = ThreadPoolExecutor(
            max_workers=max_pairs, thread_name_prefix="evolve-quality-pair"
        )
        # Future.add_done_callback runs inline when a future is already done.
        # A reentrant lock keeps that fast-completion path from self-deadlocking.
        self._lock = RLock()
        self._compute_idle = Condition(self._lock)
        self._groups: dict[int, _Group] = {}
        self._futures: set[Future] = set()
        self._ready: deque[int] = deque()
        self._next_key = 0
        self._initial_ids: set[int] = set()
        self._initial_ready: set[int] = set()
        self._registered_ids: set[int] = set()
        self._submitted_ids: set[int] = set()
        self._initial_barrier_open = initial_groups == 0
        self._initial_timer: Timer | None = None
        self._active = 0
        self._active_compute = 0
        self._peak_compute = 0
        self._compute_futures: set[Future] = set()
        self._compute_cancel_tokens: set[_CombinedCancellationEvent] = set()
        self._fatal_cancellation = Event()
        self._fatal: BaseException | None = None
        self._closed = False
        if initial_groups and initial_group_wait_seconds:
            self._initial_timer = Timer(
                initial_group_wait_seconds, self._open_initial_barrier
            )
            self._initial_timer.daemon = True
            self._initial_timer.start()

    @property
    def active_pair_slots(self) -> int:
        """Number of pair callbacks currently occupying scheduler slots."""

        with self._lock:
            return self._active

    @property
    def queued_pair_tasks(self) -> int:
        """Number of pair tasks waiting for a scheduler slot."""

        with self._lock:
            return len(self._ready)

    @property
    def active_compute_workers(self) -> int:
        """Actual stock/patch callbacks currently using the bounded pool."""

        with self._lock:
            return self._active_compute

    @property
    def peak_compute_workers(self) -> int:
        """Highest observed concurrent stock/patch callback count."""

        with self._lock:
            return self._peak_compute

    def register_group(self) -> int:
        """Reserve a proposal's place before its patch is prepared."""

        with self._lock:
            if self._closed:
                raise RuntimeError("pair scheduler is shut down")
            key = self._next_key
            self._next_key += 1
            self._registered_ids.add(key)
            if len(self._initial_ids) < self.initial_groups:
                self._initial_ids.add(key)
            return key

    def submit_group(
        self,
        items: tuple[T, ...] | list[T],
        run_one: Callable[[T], R],
        *,
        group_id: int | None = None,
    ) -> Future[list[R]]:
        """Register one proposal's work without blocking its caller."""

        values = tuple(items)
        result: Future[list[R]] = Future()
        with self._lock:
            if self._closed:
                result.set_exception(RuntimeError("pair scheduler is shut down"))
                return result
            if self._fatal is not None:
                result.set_exception(self._fatal)
                return result
            key = self._register_group_locked() if group_id is None else group_id
            if key in self._submitted_ids:
                result.set_exception(
                    ValueError(f"proposal group {key} was already submitted")
                )
                return result
            if key not in self._registered_ids:
                result.set_exception(
                    ValueError(f"proposal group {key} was not registered")
                )
                return result
            self._submitted_ids.add(key)
            self._mark_initial_ready_locked(key)
            if not values:
                result.set_result([])
                self._dispatch_locked()
                return result
            group = _Group(
                key=key,
                items=values,
                run_one=run_one,
                result=result,
                values=[None] * len(values),
            )
            self._groups[key] = group
            self._queue_locked(group)
            self._dispatch_locked()
        return result

    def run_group(
        self, items: tuple[T, ...] | list[T], run_one: Callable[[T], R]
    ) -> list[R]:
        """Run one proposal-scoped list, preserving its task order."""

        return self.submit_group(items, run_one).result()

    def run_registered_group(
        self,
        group_id: int,
        items: tuple[T, ...] | list[T],
        run_one: Callable[[T], R],
    ) -> list[R]:
        """Run a group reserved before proposal-specific preparation began."""

        return self.submit_group(items, run_one, group_id=group_id).result()

    def submit_compute(
        self,
        compute: Callable[..., R],
        *,
        cancellation_event: Event | None = None,
        cooperative_cancellation: bool = False,
    ) -> Future[R]:
        """Submit one actual stock or patch computation to the shared cap.

        Pair coordination callbacks must return promptly and must not wait on
        a Future submitted here; doing so could deadlock the bounded pool. A
        cooperative callback opts into a composite Event: ``is_set`` observes
        caller cancellation or scheduler fail-closed, while ``set`` mutates
        only the caller-owned Event.
        """

        with self._lock:
            if self._closed:
                future: Future[R] = Future()
                future.set_exception(RuntimeError("pair scheduler is shut down"))
                return future
            if self._fatal is not None:
                future = Future()
                future.set_exception(self._fatal)
                return future
            token = (
                _CombinedCancellationEvent(cancellation_event, self._fatal_cancellation)
                if cooperative_cancellation
                else None
            )
            if token is not None:
                self._compute_cancel_tokens.add(token)

            def tracked_compute() -> R:
                with self._lock:
                    self._active_compute += 1
                    self._peak_compute = max(self._peak_compute, self._active_compute)
                try:
                    return compute(token) if token is not None else compute()
                finally:
                    with self._lock:
                        self._active_compute -= 1

            try:
                future = self._executor.submit(tracked_compute)
            except BaseException:
                if token is not None:
                    self._compute_cancel_tokens.discard(token)
                raise
            self._compute_futures.add(future)
            future.add_done_callback(
                lambda completed, cancel_token=token: self._compute_done(
                    completed, cancel_token
                )
            )
            return future

    def _compute_done(
        self,
        completed: Future,
        token: _CombinedCancellationEvent | None,
    ) -> None:
        with self._lock:
            self._compute_futures.discard(completed)
            if token is not None:
                self._compute_cancel_tokens.discard(token)
            if not self._compute_futures:
                self._compute_idle.notify_all()

    def wait_for_compute_idle(self, timeout: float | None = None) -> bool:
        """Wait until no callback can still be using a proposal checkout.

        Cancelling a Future does not stop a callback that is already running.
        Proposal cleanup uses this barrier after fail-closed cancellation so
        the checkout stays available until every spawned FEDOT worker exits.
        """

        with self._compute_idle:
            return self._compute_idle.wait_for(
                lambda: not self._compute_futures,
                timeout=timeout,
            )

    def skip_group(self, group_id: int) -> None:
        """Release an initial registration when a proposal needs no evaluation."""

        with self._lock:
            if group_id in self._submitted_ids:
                return
            if group_id not in self._registered_ids:
                raise ValueError(f"proposal group {group_id} was not registered")
            self._submitted_ids.add(group_id)
            self._mark_initial_ready_locked(group_id)
            self._dispatch_locked()

    def fail_closed(self, error: BaseException) -> None:
        """Stop dispatch after a proposal-wide orchestration or persistence error."""

        with self._lock:
            self._fail_closed_locked(error)

    def _fail_closed_locked(self, error: BaseException) -> None:
        if self._fatal is None:
            self._fatal = error
        self._fatal_cancellation.set()
        for token in tuple(self._compute_cancel_tokens):
            token._scheduler_cancelled()
        self._ready.clear()
        self._initial_barrier_open = True
        if self._initial_timer is not None:
            self._initial_timer.cancel()
        for group in self._groups.values():
            group.queued = False
            group.next_index = len(group.items)
        # Cancel executor entries that have not started. Opted-in running
        # compute callbacks see the scheduler token; run_worker uses it to
        # clean up only that callback's owned process group.
        for future in tuple(self._futures):
            future.cancel()
        # Best-effort cancellation prevents queued compute from starting. A
        # running callback observes the scheduler-owned cancellation token.
        for future in tuple(self._compute_futures):
            future.cancel()
        self._finish_ready_groups_locked()

    @property
    def fatal_error(self) -> BaseException | None:
        """Return the shared fail-closed error, if any."""

        with self._lock:
            return self._fatal

    def _queue_locked(self, group: _Group) -> None:
        if group.next_index < len(group.items) and not group.queued:
            group.queued = True
            self._ready.append(group.key)

    def _register_group_locked(self) -> int:
        key = self._next_key
        self._next_key += 1
        self._registered_ids.add(key)
        if len(self._initial_ids) < self.initial_groups:
            self._initial_ids.add(key)
        return key

    def _mark_initial_ready_locked(self, key: int) -> None:
        if key not in self._initial_ids:
            return
        self._initial_ready.add(key)
        if len(self._initial_ready) >= self.initial_groups:
            self._initial_barrier_open = True
            if self._initial_timer is not None:
                self._initial_timer.cancel()

    def _open_initial_barrier(self) -> None:
        with self._lock:
            if self._closed or self._fatal is not None:
                return
            self._initial_barrier_open = True
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        if not self._initial_barrier_open:
            return
        while self._active < self.max_pairs and self._ready and self._fatal is None:
            key = self._ready.popleft()
            group = self._groups.get(key)
            if group is None:
                continue
            group.queued = False
            index = group.next_index
            if index >= len(group.items):
                continue
            group.next_index += 1
            group.active += 1
            self._active += 1
            if group.next_index < len(group.items):
                self._queue_locked(group)
            try:
                future = self._executor.submit(group.run_one, group.items[index])
            except BaseException as exc:
                group.active -= 1
                self._active -= 1
                self._fail_closed_locked(exc)
                return
            self._futures.add(future)
            future.add_done_callback(
                lambda completed, group_key=key, item_index=index: self._task_done(
                    group_key, item_index, completed
                )
            )

    def _task_done(self, key: int, index: int, completed: Future[R]) -> None:
        with self._lock:
            self._futures.discard(completed)
            group = self._groups.get(key)
            self._active -= 1
            if group is not None:
                group.active -= 1
                try:
                    group.values[index] = completed.result()
                except BaseException as exc:
                    if self._fatal is None:
                        self._fail_closed_locked(exc)
                self._finish_ready_groups_locked()
            if self._fatal is not None:
                self._finish_ready_groups_locked()
            else:
                self._dispatch_locked()

    def _finish_ready_groups_locked(self) -> None:
        for key, group in tuple(self._groups.items()):
            if group.active:
                continue
            if self._fatal is not None:
                if not group.result.done():
                    group.result.set_exception(self._fatal)
                del self._groups[key]
            elif group.next_index == len(group.items):
                if not group.result.done():
                    group.result.set_result(list(group.values))
                del self._groups[key]

    def shutdown(self, *, wait: bool = True) -> None:
        """Close the scheduler after all submitted proposal groups finish."""

        with self._lock:
            self._closed = True
            if self._initial_timer is not None:
                self._initial_timer.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=False)
