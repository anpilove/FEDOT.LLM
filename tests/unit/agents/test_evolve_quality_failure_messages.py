"""Terminal execution failures must never be described as a quality rejection."""

import pytest

from fedotllm.agents.evolve.controller.quality_queue import quality_result_message


@pytest.mark.parametrize("status", ["fedot_budget_timeout", "unverified_metric"])
def test_missing_verified_metrics_do_not_produce_a_rejection_message(status):
    reason = "metric not measured or verified"
    message = quality_result_message(
        {
            "candidate_id": "proposal-a",
            "status": status,
            "reason": reason,
            "keep": None,
            "quality_outcome": "not_evaluated",
            "task_ids": ["dataset-a"],
        }
    )

    assert "proposal-a" in message
    assert reason in message
    assert "не завершена" in message
    assert "Итогового вердикта качества нет" in message
    assert "Не принимаем" not in message
