"""End-to-end controller queue acceptance at the supported seven-pair cap."""

from __future__ import annotations

import json
import threading

import pytest

from fedotllm.agents.evolve.controller import quality_executor
from fedotllm.agents.evolve.controller import quality_queue
from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
from fedotllm.agents.evolve.types import Decision, PatchCandidate


def _enqueue_queue(tmp_path, count: int):
    source = tmp_path / "FEDOT"
    rel = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
    target = source / rel
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    workspace = tmp_path / "campaign"
    for index in range(count):
        quality_queue.enqueue_quality_job(
            workspace,
            candidate=PatchCandidate(f"improvement-{index}", file_path=rel),
            patch_text=(
                f"--- a/{rel}\n+++ b/{rel}\n@@\n-value = 1\n+value = {index + 2}\n"
            ),
            hint="forecasting improvement",
            priority="normal",
        )
    return source, workspace


@pytest.mark.parametrize(
    ("cpu_quota", "max_pair_slots", "expected_slots"),
    [(28, None, 7), (32, 8, 8)],
)
def test_quality_queue_drains_fifteen_proposals_through_configured_pair_cap(
    tmp_path, monkeypatch, cpu_quota, max_pair_slots, expected_slots
):
    source, workspace = _enqueue_queue(tmp_path, 15)
    lock = threading.Lock()
    active_proposals: set[str] = set()
    peak_active_proposals = 0
    first_wave_full = threading.Event()
    release_first_wave = threading.Event()
    started: list[str] = []

    def measure(
        _source,
        _experiment,
        *,
        candidate,
        pair_scheduler: SharedPairScheduler,
        pair_group_id: int,
        **_kwargs,
    ):
        nonlocal peak_active_proposals
        candidate_id = candidate.candidate_id
        with lock:
            assert candidate_id not in active_proposals
            active_proposals.add(candidate_id)
            started.append(candidate_id)
            peak_active_proposals = max(peak_active_proposals, len(active_proposals))
            if len(active_proposals) == expected_slots:
                first_wave_full.set()
        try:
            # Exercise the same campaign-shared pair scheduler used by a real
            # proposal; the barrier keeps the first wave concurrently admitted.
            pair_scheduler.run_registered_group(
                pair_group_id,
                (candidate_id,),
                lambda item: item,
            )
            if candidate_id in {
                f"improvement-{index}" for index in range(expected_slots)
            }:
                assert release_first_wave.wait(3), (
                    "test did not release the first proposal wave"
                )
            return Decision(False, "no improvement", 0.0, stage="quality")
        finally:
            with lock:
                active_proposals.remove(candidate_id)

    monkeypatch.setattr(quality_executor, "measure_fedot_quality", measure)
    result_holder = []
    errors = []

    def drain():
        try:
            kwargs = {
                "continue_on_error": True,
                "n_jobs": 4,
                "cpu_quota": cpu_quota,
            }
            if max_pair_slots is not None:
                kwargs["max_pair_slots"] = max_pair_slots
            result_holder.append(
                quality_executor.drain_quality_queue(source, workspace, **kwargs)
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=drain, daemon=True)
    runner.start()
    try:
        assert first_wave_full.wait(3), (
            "queue did not admit the configured initial proposal wave"
        )
        with lock:
            assert len(active_proposals) == expected_slots
            assert peak_active_proposals == expected_slots
        release_first_wave.set()
        runner.join(timeout=8)
        assert not runner.is_alive(), "queue did not drain all fifteen proposals"
        assert not errors
        assert len(result_holder) == 1

        result = result_holder[0]
        assert len(started) == 15
        assert set(started) == {f"improvement-{index}" for index in range(15)}
        assert {job["candidate_id"] for job in result["jobs"]} == {
            f"improvement-{index}" for index in range(15)
        }
        assert len(result["jobs"]) == 15
        assert all(job["status"] == "drop" for job in result["jobs"])

        events = [
            json.loads(line)
            for line in (workspace / "quality_jobs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        batch = next(
            event
            for event in events
            if event.get("event") == "quality_scheduler_batch_started"
        )
        assert batch["parallel_proposals"] == expected_slots
        assert batch["dataset_pair_slots"] == expected_slots
        assert batch["cpu_slots"] == expected_slots * 4
        assert batch["cpu_quota"] == cpu_quota
        assert batch["n_jobs_per_side"] == 4
        assert batch["policy"] == "fair-shared-pair-pool-v3"
    finally:
        release_first_wave.set()
        runner.join(timeout=4)
