"""Subprocess worker. Invoked with PYTHONPATH=fedot_checkout:repo_root."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        from research.evolve.metric_agent.scorer import score_task

        payload = score_task(args.task)
    except Exception as exc:
        import traceback

        payload = {
            "task_id": args.task,
            "status": "invalid",
            "score": float("nan"),
            "traceback": traceback.format_exc(),
            "detail": f"{type(exc).__name__}: {exc}",
            "n_train": 0,
        }
    payload["duration_s"] = time.perf_counter() - started
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0 if payload.get("status") != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
