"""Generated proofs must be honest: fail on the defect, pass only on a real fix.

A template is the one thing in the loop the agent cannot rewrite, so it carries
the whole weight of the acceptance decision. If it is satisfiable by something
that is not a fix, every gate behind it is decoration.
"""

import subprocess
import sys
import types

import pytest

from fedotllm.agents.evolve import templates
from fedotllm.agents.evolve.templates import build_test, module_path

SHARED_ATTR = "class C:\n    keys = {'a', 'b'}\n"
FIXED_ATTR = "class C:\n    def __init__(self):\n        self.keys = {'a', 'b'}\n"
GAMED_ATTR = "class C:\n    @classmethod\n    def keys(cls):\n        return {'a', 'b'}\n"

CONSTRUCTED_DEFAULT = "class T:\n    pass\n\n\ndef build(task=T()):\n    return task\n"
FIXED_DEFAULT = (
    "class T:\n    pass\n\n\ndef build(task=None):\n"
    "    if task is None:\n        task = T()\n    return task\n"
)


def _run(tmp_path, module_src: str, test_code: str) -> str:
    (tmp_path / "mod.py").write_text(module_src, encoding="utf-8")
    (tmp_path / "test_generated.py").write_text(test_code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_generated.py", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return "failed" if " failed" in proc.stdout else "passed"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "fedot").mkdir()
    return tmp_path


class TestClassAttributeProof:
    def _build(self, repo, src):
        (repo / "fedot" / "mod.py").write_text(src, encoding="utf-8")
        gen = build_test(repo, "RUF012", "fedot/mod.py", 2)
        assert gen is not None
        return gen.test_code.replace("from fedot.mod import", "from mod import")

    def test_fails_on_the_defect(self, repo, tmp_path):
        code = self._build(repo, SHARED_ATTR)
        assert _run(tmp_path, SHARED_ATTR, code) == "failed"

    def test_passes_on_a_real_fix(self, repo, tmp_path):
        code = self._build(repo, SHARED_ATTR)
        assert _run(tmp_path, FIXED_ATTR, code) == "passed"

    def test_rejects_the_classmethod_workaround(self, repo, tmp_path):
        """Wrapping the attribute hides the shared object and breaks every reader."""
        code = self._build(repo, SHARED_ATTR)
        assert _run(tmp_path, GAMED_ATTR, code) == "failed"

    def test_targets_the_attribute_on_the_reported_line(self, repo):
        src = "class C:\n    first = []\n    second = {}\n"
        (repo / "fedot" / "mod.py").write_text(src, encoding="utf-8")
        gen = build_test(repo, "RUF012", "fedot/mod.py", 3)
        assert gen is not None
        assert gen.target == "C.second"


class TestConstructedDefaultProof:
    def _build(self, repo, src):
        (repo / "fedot" / "mod.py").write_text(src, encoding="utf-8")
        gen = build_test(repo, "B008", "fedot/mod.py", 5)
        assert gen is not None
        return gen.test_code.replace("from fedot.mod import", "from mod import")

    def test_fails_on_the_defect(self, repo, tmp_path):
        code = self._build(repo, CONSTRUCTED_DEFAULT)
        assert _run(tmp_path, CONSTRUCTED_DEFAULT, code) == "failed"

    def test_passes_once_the_default_is_none(self, repo, tmp_path):
        code = self._build(repo, CONSTRUCTED_DEFAULT)
        assert _run(tmp_path, FIXED_DEFAULT, code) == "passed"

    def test_targets_the_default_at_the_reported_column(self, repo):
        signature = "def build(first=T(), second=T()):"
        src = "class T:\n    pass\n\n\n" + signature + "\n    return first\n"
        (repo / "fedot" / "mod.py").write_text(src, encoding="utf-8")
        column = signature.index("second=T") + len("second=") + 1
        gen = build_test(repo, "B008", "fedot/mod.py", 5, column)
        assert gen is not None
        assert gen.target == "build(second=…)"


def test_module_path_maps_to_import_path():
    assert module_path("fedot/core/data/data.py") == "fedot.core.data.data"


def test_ruff_ansi_does_not_break_finding_paths(repo, monkeypatch):
    raw = (
        "\x1b[1mfedot/mod.py\x1b[0m\x1b[36m:\x1b[0m2\x1b[36m:\x1b[0m12"
        "\x1b[36m:\x1b[0m \x1b[1;31mRUF012\x1b[0m Mutable class attribute"
    )
    monkeypatch.setattr(
        templates.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(stdout=raw, stderr="", returncode=1),
    )

    assert templates.collect_findings(repo) == [
        "fedot/mod.py:2:12: RUF012 Mutable class attribute"
    ]
