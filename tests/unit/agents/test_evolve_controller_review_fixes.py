"""Executor lifecycle regressions runnable with stdlib unittest on Python 3.10."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fedotllm.agents.evolve.controller import quality_executor as executor
from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
from fedotllm.agents.evolve.types import Decision, PatchCandidate


class LegacyStorageError(OSError):
    """Expose the same missing optional exception API as Python 3.10."""

    def __getattribute__(self, name):
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


class ExecutorLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        relative = "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
        target = self.source / relative
        target.parent.mkdir(parents=True)
        target.write_text("value = 1\n")
        self.workspace = self.root / "campaign"
        enqueue_quality_job(
            self.workspace,
            candidate=PatchCandidate("proposal-a", file_path=relative),
            patch_text=f"--- a/{relative}\n+++ b/{relative}\n@@\n-value = 1\n+value = 2\n",
            hint="forecasting improvement",
            priority="normal",
        )
        self.job_path = self.workspace / "quality_queue/proposal-a.json"

    def saved_job(self):
        return json.loads(self.job_path.read_text())

    def test_completed_proposal_cleans_its_checkout_on_supported_python(self):
        with patch.object(executor, "measure_fedot_quality", return_value=Decision(
            False, "no quality improvement", 0.0, stage="quality"
        )):
            result = executor.drain_quality_queue(self.source, self.workspace)

        self.assertEqual(result["jobs"][0]["status"], "drop")
        self.assertEqual(self.saved_job()["status"], "drop")
        self.assertFalse(list((self.workspace / "experiments/quality-drain").glob("*")))
        self.assertEqual((self.source / self.saved_job()["file_path"]).read_text(), "value = 1\n")

    def test_storage_error_survives_cleanup_error_without_add_note(self):
        original = LegacyStorageError(28, "measurement journal full")
        with patch.object(executor, "measure_fedot_quality", side_effect=original), patch.object(
            executor, "discard_experiment_checkout", side_effect=OSError("checkout cleanup failed")
        ):
            with self.assertRaises(LegacyStorageError) as caught:
                executor.drain_quality_queue(self.source, self.workspace)

        self.assertIs(caught.exception, original)
        self.assertEqual(self.saved_job()["status"], "queued")
        self.assertIn("executor_interrupted", self.saved_job()["reason"])

    def test_storage_error_survives_recovery_write_error_without_add_note(self):
        original = LegacyStorageError(28, "measurement journal full")
        write_status = executor.write_job_status

        def fail_recovery(path, status, extra=None):
            if status == "queued":
                raise OSError(28, "queue recovery disk full")
            return write_status(path, status, extra)

        with patch.object(executor, "measure_fedot_quality", side_effect=original), patch.object(
            executor, "write_job_status", side_effect=fail_recovery
        ):
            with self.assertRaises(LegacyStorageError) as caught:
                executor.drain_quality_queue(self.source, self.workspace)

        self.assertIs(caught.exception, original)
        self.assertEqual(self.saved_job()["status"], "running")


if __name__ == "__main__":
    unittest.main()
