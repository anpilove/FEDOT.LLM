"""Asynchronous singleflight coordination for shared stock computations.

Subscribers receive independent ``Future`` views of one owner computation. A
subscriber can cancel its own view without cancelling the owner or other
subscribers. The owner computation must be submitted to a bounded worker
executor and the submit callback must return promptly; this registry never
waits for a result in the caller's thread.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, InvalidStateError
from functools import wraps
from threading import RLock
from time import monotonic
from typing import Callable, Generic, Hashable, TypeVar


T = TypeVar("T")
OwnerStartedCallback = Callable[[float], None]


class StockFlightRegistry(Generic[T]):
    """Share one submitted computation per key without blocking subscribers.

    ``get_or_submit`` atomically publishes an in-flight result before asking
    ``submit`` to schedule the owner. Concurrent callers for the same key get
    separate cancellation-safe Future views of that result. The first caller's
    ``compute`` is the only one submitted. Completed, failed, and cancelled
    flights are removed so a later call can retry the key.

    The registry does not own an executor. Callers provide the bounded compute
    pool through ``submit`` so stock and patched worker calls can share one
    resource limit.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._flights: dict[Hashable, Future[T]] = {}
        self._started_at: dict[Future[T], float] = {}
        self._start_callbacks: dict[Future[T], list[OwnerStartedCallback]] = {}

    def get_or_submit(
        self,
        key: Hashable,
        *,
        submit: Callable[[Callable[[], T]], Future[T]],
        compute: Callable[[], T],
        on_waiting: Callable[[], None] | None = None,
        on_owner_started: OwnerStartedCallback | None = None,
    ) -> Future[T]:
        """Return a cancellation-isolated Future view and submit one owner.

        ``submit`` should be a prompt operation such as ``executor.submit``;
        it must return the Future for ``compute`` rather than synchronously
        running ``compute``. Submission errors are delivered to all current
        subscribers and the flight is removed for a later retry.
        """

        with self._lock:
            owner_future = self._flights.get(key)
            is_owner = owner_future is None
            if owner_future is None:
                owner_future = Future()
                self._flights[key] = owner_future
                self._start_callbacks[owner_future] = []
            started_at = self._started_at.get(owner_future)
            if on_owner_started is not None and started_at is None:
                self._start_callbacks.setdefault(owner_future, []).append(
                    on_owner_started
                )
            subscriber = self._subscriber_view(owner_future)

        if on_owner_started is not None and started_at is not None:
            on_owner_started(started_at)
        if on_waiting is not None:
            try:
                on_waiting()
            except BaseException as exc:
                if is_owner:
                    self._finish_exception(key, owner_future, exc)
                else:
                    subscriber.set_exception(exc)
                return subscriber
        if is_owner:
            try:

                @wraps(compute)
                def start_notifying_compute(*args, **kwargs):
                    self._mark_started(key, owner_future, monotonic())
                    return compute(*args, **kwargs)

                work_future = submit(start_notifying_compute)
                if not isinstance(work_future, Future):
                    raise TypeError(
                        "stock compute submitter must return concurrent.futures.Future"
                    )
            except BaseException as exc:
                self._finish_exception(key, owner_future, exc)
            else:
                work_future.add_done_callback(
                    lambda completed: self._finish_from_work(
                        key, owner_future, completed
                    )
                )
        return subscriber

    def _mark_started(
        self,
        key: Hashable,
        owner_future: Future[T],
        started_at: float,
    ) -> None:
        with self._lock:
            if self._flights.get(key) is not owner_future:
                return
            if owner_future in self._started_at:
                return
            self._started_at[owner_future] = started_at
            callbacks = self._start_callbacks.pop(owner_future, [])
        for callback in callbacks:
            try:
                callback(started_at)
            except Exception:
                # Deadline observers must never prevent the stock computation.
                continue

    @staticmethod
    def _subscriber_view(owner_future: Future[T]) -> Future[T]:
        subscriber: Future[T] = Future()

        def copy_outcome(completed: Future[T]) -> None:
            if subscriber.done():
                return
            if completed.cancelled():
                subscriber.cancel()
                return
            try:
                value = completed.result()
            except CancelledError:
                subscriber.cancel()
            except BaseException as exc:
                try:
                    subscriber.set_exception(exc)
                except InvalidStateError:
                    # The individual subscriber can cancel concurrently.
                    pass
            else:
                try:
                    subscriber.set_result(value)
                except InvalidStateError:
                    # The individual subscriber can cancel concurrently.
                    pass

        owner_future.add_done_callback(copy_outcome)
        return subscriber

    def _finish_from_work(
        self,
        key: Hashable,
        owner_future: Future[T],
        work_future: Future[T],
    ) -> None:
        if work_future.cancelled():
            self._finish_cancelled(key, owner_future)
            return
        try:
            result = work_future.result()
        except CancelledError:
            self._finish_cancelled(key, owner_future)
        except BaseException as exc:
            self._finish_exception(key, owner_future, exc)
        else:
            self._finish_result(key, owner_future, result)

    def _remove_flight(self, key: Hashable, owner_future: Future[T]) -> bool:
        with self._lock:
            if self._flights.get(key) is not owner_future:
                return False
            del self._flights[key]
            self._started_at.pop(owner_future, None)
            self._start_callbacks.pop(owner_future, None)
            return True

    def _finish_result(self, key: Hashable, owner_future: Future[T], result: T) -> None:
        if not self._remove_flight(key, owner_future):
            return
        owner_future.set_result(result)

    def _finish_exception(
        self, key: Hashable, owner_future: Future[T], error: BaseException
    ) -> None:
        if not self._remove_flight(key, owner_future):
            return
        owner_future.set_exception(error)

    def _finish_cancelled(self, key: Hashable, owner_future: Future[T]) -> None:
        if not self._remove_flight(key, owner_future):
            return
        owner_future.cancel()
