"""Failure reporting must preserve the primary exception on Python 3.10."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fedotllm.agents.evolve.storage import journal


class LegacyWriteError(OSError):
    def __getattribute__(self, name):
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


class ExceptionCompatibilityTests(unittest.TestCase):
    def test_partial_journal_write_error_survives_failed_rollback(self):
        for error_class in (OSError, LegacyWriteError):
            with self.subTest(error_class=error_class.__name__), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "quality_jobs.jsonl"
                history = b'{"event":"previous durable result"}\n'
                path.write_bytes(history)
                original = error_class(28, "journal write disk full")
                real_write = journal.os.write
                writes = 0

                def partial_write(descriptor, data):
                    nonlocal writes
                    writes += 1
                    if writes == 1:
                        return real_write(descriptor, data[:8])
                    raise original

                with patch.object(journal.os, "write", side_effect=partial_write), patch.object(
                    journal.os, "ftruncate", side_effect=OSError("rollback unavailable")
                ):
                    with self.assertRaises(error_class) as caught:
                        journal.append_journal(path, {"event": "new result"})

                self.assertIs(caught.exception, original)
                self.assertTrue(path.read_bytes().startswith(history))
                self.assertGreater(path.stat().st_size, len(history))
                if hasattr(original, "add_note"):
                    self.assertIn("rollback unavailable", original.__notes__[-1])

if __name__ == "__main__":
    unittest.main()
