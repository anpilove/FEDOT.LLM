"""Fixed POSIX guardian for one already-launched worker process group.

This script never executes commands or discovers processes. The controller
passes only its worker's new-session PGID and two inherited pipe descriptors.
Closing the parent pipe without ``complete\n`` terminates that exact group.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import time


def terminate_owned_worker_group(worker_pgid: int) -> int:
    """Kill the controller-supplied worker group; tolerate a Darwin exit race."""
    try:
        os.killpg(worker_pgid, signal.SIGKILL)
    except ProcessLookupError:
        return 0
    except PermissionError as exc:
        # On Darwin, signaling a just-exited/reaped leader's group can report
        # EPERM during the exit race. Check only this explicit PGID: if it is
        # gone, cleanup succeeded; if it still exists, surface the failure.
        try:
            os.killpg(worker_pgid, 0)
        except ProcessLookupError:
            return 0
        except OSError as probe_error:
            print(
                f"worker_group_cleanup_failed pgid={worker_pgid} "
                f"signal_error={type(exc).__name__}:{exc} "
                f"probe_error={type(probe_error).__name__}:{probe_error}",
                file=sys.stderr, flush=True,
            )
            return 1
        print(
            f"worker_group_cleanup_failed pgid={worker_pgid} "
            f"signal_error={type(exc).__name__}:{exc} group_still_exists=true",
            file=sys.stderr, flush=True,
        )
        return 1
    except OSError as exc:
        print(
            f"worker_group_cleanup_failed pgid={worker_pgid} "
            f"signal_error={type(exc).__name__}:{exc}",
            file=sys.stderr, flush=True,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-pgid", required=True, type=int)
    parser.add_argument("--parent-fd", required=True, type=int)
    parser.add_argument("--ready-fd", required=True, type=int)
    parser.add_argument("--watchdog-timeout-seconds", type=float)
    args = parser.parse_args()
    if args.worker_pgid <= 1 or args.worker_pgid == os.getpgrp():
        parser.error("worker PGID must identify a separate, owned session")
    try:
        os.write(args.ready_fd, b"R")
    finally:
        os.close(args.ready_fd)
    message = b""
    deadline = (
        None if args.watchdog_timeout_seconds is None
        else time.monotonic() + args.watchdog_timeout_seconds
    )
    if deadline is not None and args.watchdog_timeout_seconds <= 0:
        parser.error("watchdog timeout must be positive")
    try:
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([args.parent_fd], [], [], remaining)
            if not readable:
                cleanup_status = terminate_owned_worker_group(args.worker_pgid)
                return 124 if cleanup_status == 0 else cleanup_status
            chunk = os.read(args.parent_fd, 64)
            if not chunk:
                return terminate_owned_worker_group(args.worker_pgid)
            message += chunk
            if message == b"complete\n":
                return 0
    finally:
        os.close(args.parent_fd)


if __name__ == "__main__":
    raise SystemExit(main())
