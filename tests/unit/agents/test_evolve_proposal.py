"""Parsing and application of evolution proposals.

These cover the failure that cost a benchmark run: a two-hunk answer plus a full
pytest function overflowed the token ceiling, the trailing ``<<<END>>>`` was cut,
and the test silently vanished — every run died with "missing: ['test_code']".
"""

import subprocess
import sys
import types

import pytest

from fedotllm.agents.evolve import loop
from fedotllm.agents.evolve.loop import (
    EvolveResult,
    Proposal,
    apply_source,
    build_audit,
    infer_anchor,
    parse_delimited,
    parse_proposal,
    proposal_as_delimited,
    require_clean_repo,
    reset_evolution_changes,
)

HEADER = """FILE: fedot/core/data/data.py
TEST_FILE: test/unit/test_data.py
TEST_NAME: test_x
PROBLEM: constructed default argument
RATIONALE: two coordinated sites
"""

ONE_HUNK = """<<<OLD>>>
def f(task=Task()):
<<<NEW>>>
def f(task=None):
"""

SECOND_HUNK = """<<<OLD>>>
    return task
<<<NEW>>>
    if task is None:
        task = Task()
    return task
"""

TEST_BLOCK = """<<<TEST>>>
def test_x():
    assert True
"""


class TestParseDelimited:
    def test_single_hunk_still_parses(self):
        p = parse_delimited(HEADER + ONE_HUNK + TEST_BLOCK + "<<<END>>>")
        assert len(p.hunks) == 1
        assert p.old_code == "def f(task=Task()):"
        assert p.new_code == "def f(task=None):"

    def test_two_hunks_are_kept_separate(self):
        p = parse_delimited(HEADER + ONE_HUNK + SECOND_HUNK + TEST_BLOCK + "<<<END>>>")
        assert len(p.hunks) == 2
        assert p.hunks[1][0] == "    return task"
        assert "task = Task()" in p.hunks[1][1]

    def test_old_new_mirror_the_first_hunk(self):
        """Call sites that predate multi-hunk support must keep working."""
        p = parse_delimited(HEADER + ONE_HUNK + SECOND_HUNK + TEST_BLOCK + "<<<END>>>")
        assert (p.old_code, p.new_code) == p.hunks[0]

    def test_test_survives_a_missing_end_marker(self):
        """A truncated answer loses <<<END>>> first; it must not lose the test."""
        p = parse_delimited(HEADER + ONE_HUNK + SECOND_HUNK + TEST_BLOCK)
        assert len(p.hunks) == 2
        assert "def test_x()" in p.test_code
        parse_proposal(HEADER + ONE_HUNK + SECOND_HUNK + TEST_BLOCK)  # must not raise

    def test_end_marker_never_leaks_into_the_test(self):
        p = parse_delimited(HEADER + ONE_HUNK + TEST_BLOCK + "<<<END>>>")
        assert "<<<END>>>" not in p.test_code

    def test_round_trip_preserves_every_hunk(self):
        p = parse_delimited(HEADER + ONE_HUNK + SECOND_HUNK + TEST_BLOCK + "<<<END>>>")
        assert parse_delimited(proposal_as_delimited(p)).hunks == p.hunks


SOURCE = '''"""Module docstring."""


def build(task=None):
    """Build a thing."""
    return task
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "fedot" / "core").mkdir(parents=True)
    (tmp_path / "fedot" / "core" / "m.py").write_text(SOURCE, encoding="utf-8")
    return tmp_path


def _proposal(hunks):
    return Proposal(file_path="fedot/core/m.py", hunks=hunks)


class TestApplySource:
    def test_applies_every_hunk(self, repo):
        assert apply_source(
            repo,
            _proposal(
                [
                    ("def build(task=None):", "def build(task=None, extra=1):"),
                    ("    return task", "    return task, extra"),
                ]
            ),
        )
        text = (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8")
        assert "extra=1" in text and "return task, extra" in text

    def test_partial_application_is_refused(self, repo):
        """Half of a two-site fix is worse than none — that is how a default
        became None while the body that should build it never appeared."""
        before = (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8")
        assert not apply_source(
            repo,
            _proposal(
                [
                    ("def build(task=None):", "def build(task=None, extra=1):"),
                    ("    nonexistent line", "    whatever"),
                ]
            ),
        )
        assert (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8") == before

    def test_multi_hunk_ast_fallback_cannot_drop_a_hunk(self, repo):
        path = repo / "fedot" / "core" / "m.py"
        before = path.read_text(encoding="utf-8")
        p = Proposal(
            file_path="fedot/core/m.py",
            hunks=[
                (
                    "def build(task=None):\n    \"\"\"Build a thing.\"\"\"\n    return task",
                    "def build(task=None):\n    \"\"\"Build safely.\"\"\"\n    return task",
                ),
                ("missing coordinated change", "required second change"),
            ],
        )

        assert not apply_source(repo, p)
        assert path.read_text(encoding="utf-8") == before

    def test_ambiguous_hunk_is_refused(self, repo):
        path = repo / "fedot" / "core" / "m.py"
        path.write_text(SOURCE + "\n\ndef other(task=None):\n    return task\n", encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        assert not apply_source(repo, _proposal([("    return task", "    return None")]))
        assert path.read_text(encoding="utf-8") == before

    def test_syntax_breaking_patch_is_refused(self, repo):
        path = repo / "fedot" / "core" / "m.py"
        before = path.read_text(encoding="utf-8")
        assert not apply_source(repo, _proposal([("    return task", "    return (")]))
        assert path.read_text(encoding="utf-8") == before

    def test_anchor_disambiguates_a_repeated_snippet(self, repo):
        """FEDOT's data.py repeats the same defective parameter line five times;
        the enclosing symbol is what tells the two apart."""
        path = repo / "fedot" / "core" / "m.py"
        path.write_text(
            "def a(task=None):\n"
            "    return task\n"
            "\n"
            "def b(task=None):\n"
            "    return task\n",
            encoding="utf-8",
        )
        p = Proposal(
            file_path="fedot/core/m.py",
            anchor="b",
            hunks=[("    return task", "    return task or 1")],
        )
        assert apply_source(repo, p)
        text = path.read_text(encoding="utf-8")
        assert text.count("return task or 1") == 1
        assert text.split("def b")[0].count("return task or 1") == 0  # untouched in a

    def test_wrong_anchor_is_refused(self, repo):
        path = repo / "fedot" / "core" / "m.py"
        path.write_text(
            "def a(task=None):\n    return task\n\ndef b(task=None):\n    return task\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        p = Proposal(
            file_path="fedot/core/m.py",
            anchor="nonexistent",
            hunks=[("    return task", "    return task or 1")],
        )
        assert not apply_source(repo, p)
        assert path.read_text(encoding="utf-8") == before

    def test_legacy_proposal_without_hunks_still_applies(self, repo):
        p = Proposal(
            file_path="fedot/core/m.py",
            old_code="    return task",
            new_code="    return None",
        )
        assert apply_source(repo, p)
        assert "return None" in (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8")

    def test_infers_the_anchor_before_caller_regression_lookup(self, repo):
        p = Proposal(
            file_path="fedot/core/m.py",
            problem="the `build` function mishandles task",
        )
        assert infer_anchor(repo, p) == "build"

    def test_direct_test_consumers_are_regression_targets(self, repo):
        test_path = repo / "test" / "integration" / "test_tuning.py"
        test_path.parent.mkdir(parents=True)
        test_path.write_text(
            "from fedot.core.m import SharedModel\n\n"
            "def test_class_contract():\n"
            "    assert SharedModel.public_config\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)

        assert loop.find_caller_regression_tests(
            repo, "SharedModel.__init__", "fedot/core/m.py"
        ) == ["test/integration/test_tuning.py::test_class_contract"]


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


class TestRepoLifecycle:
    @staticmethod
    def _init(repo):
        _git(repo, "init", "-q")
        (repo / "tracked.py").write_text("original\n", encoding="utf-8")
        _git(repo, "add", "tracked.py")
        _git(
            repo,
            "-c", "user.name=Evolve Test",
            "-c", "user.email=evolve@example.invalid",
            "commit", "-qm", "initial",
        )

    def test_dirty_checkout_is_refused(self, tmp_path):
        self._init(tmp_path)
        (tmp_path / "notes.txt").write_text("user work\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="clean disposable checkout"):
            require_clean_repo(tmp_path)

    def test_reset_removes_only_files_created_by_the_agent(self, tmp_path):
        self._init(tmp_path)
        (tmp_path / "tracked.py").write_text("patched\n", encoding="utf-8")
        generated = tmp_path / "tests" / "test_generated.py"
        generated.parent.mkdir()
        generated.write_text("def test_x(): pass\n", encoding="utf-8")
        user_file = tmp_path / "notes.txt"
        user_file.write_text("keep\n", encoding="utf-8")

        reset_evolution_changes(tmp_path, {"tests/test_generated.py"})

        assert (tmp_path / "tracked.py").read_text(encoding="utf-8") == "original\n"
        assert not generated.exists()
        assert user_file.read_text(encoding="utf-8") == "keep\n"

    def test_exception_restores_the_clean_checkout(self, tmp_path, monkeypatch):
        self._init(tmp_path)

        def crash(_inference, repo, _workspace, _python):
            (repo / "tracked.py").write_text("half-applied\n", encoding="utf-8")
            (repo / "generated.py").write_text("temporary\n", encoding="utf-8")
            raise RuntimeError("gate crashed")

        monkeypatch.setattr(loop, "_run_evolution_loop", crash)
        with pytest.raises(RuntimeError, match="gate crashed"):
            loop.run_evolution_loop(object(), tmp_path, tmp_path / "workspace")

        assert (tmp_path / "tracked.py").read_text(encoding="utf-8") == "original\n"
        assert not (tmp_path / "generated.py").exists()
        assert _git(tmp_path, "status", "--porcelain").stdout == ""

    def test_audit_diff_includes_a_new_untracked_test(self, tmp_path):
        self._init(tmp_path)
        test_path = tmp_path / "test" / "unit" / "test_generated.py"
        test_path.parent.mkdir(parents=True)
        test_path.write_text("def test_x():\n    assert True\n", encoding="utf-8")
        proposal = Proposal(
            file_path="tracked.py",
            test_file="test/unit/test_generated.py",
            test_name="test_x",
        )
        result = EvolveResult(
            proposal=proposal,
            changed=["tracked.py", "test/unit/test_generated.py"],
        )

        audit = build_audit(tmp_path, result, "test/model")

        assert "+++ test/unit/test_generated.py" in audit
        assert "def test_x():" in audit


def test_full_green_loop_uses_the_external_proof(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    source = tmp_path / "fedot" / "x.py"
    source.parent.mkdir()
    (source.parent / "__init__.py").write_text("", encoding="utf-8")
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(tmp_path, "add", "fedot/__init__.py", "fedot/x.py")
    _git(
        tmp_path,
        "-c", "user.name=Evolve Test",
        "-c", "user.email=evolve@example.invalid",
        "commit", "-qm", "initial",
    )

    proof = (
        "from fedot.x import value\n\n\n"
        "def test_value_returns_two():\n"
        "    assert value() == 2\n"
    )
    ready = types.SimpleNamespace(
        rule="B008",
        target="value",
        test_name="test_value_returns_two",
        test_code=proof,
    )
    model_proposal = Proposal(
        file_path="fedot/x.py",
        test_file="test/model_owned.py",
        test_name="test_value_returns_two",
        test_code="def test_value_returns_two():\n    assert True",
        problem="B008 behavioural defect",
        rationale="return the expected value",
        hunks=[("    return 1", "    return 2")],
        old_code="    return 1",
        new_code="    return 2",
    )
    inference = types.SimpleNamespace(
        config=types.SimpleNamespace(provider="test", model_name="stub")
    )

    monkeypatch.setattr(loop, "scout_pick", lambda *_a, **_k: ("fedot/x.py", "proof"))
    monkeypatch.setattr(loop, "proven_defect_for", lambda *_a, **_k: ready)
    monkeypatch.setattr(loop, "invariant_defect_for", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "module_usage_section", lambda *_a, **_k: "")
    monkeypatch.setattr(loop, "ask_proposal", lambda *_a, **_k: model_proposal)
    monkeypatch.setattr(loop, "find_regression_tests", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "find_caller_regression_tests", lambda *_a, **_k: [])
    monkeypatch.setattr(loop, "NUM_CANDIDATES", 1)
    monkeypatch.setattr(loop, "MAX_FIX_TRIES", 1)
    monkeypatch.setattr(loop, "AUTOML_GATE", False)
    monkeypatch.setattr(loop, "PROBE_GATE", False)
    monkeypatch.setattr(loop, "TUNING_GATE", False)
    monkeypatch.setattr(loop, "VALUE_GATE", True)
    monkeypatch.setattr(loop, "JOURNAL_ENABLED", False)

    result = loop.run_evolution_loop(
        inference,
        tmp_path,
        tmp_path / "workspace",
        venv_python=sys.executable,
    )

    assert result.success is True
    assert result.proposal.test_code == proof
    assert "return 2" in source.read_text(encoding="utf-8")
    assert (tmp_path / "test/unit/test_fedotllm_evolve_proof.py").is_file()


class TestMarkerAndElision:
    """Two ways a replacement can be silently destructive."""

    def test_end_marker_does_not_leak_into_replacement(self):
        """It leaked once and cost a full run: the patch stopped being Python."""
        raw = HEADER + ONE_HUNK + "<<<END>>>"
        p = parse_delimited(raw)
        assert "<<<END>>>" not in p.new_code
        assert len(p.hunks) == 1

    def test_elided_replacement_is_refused(self, repo):
        """A bare `...` stands for code the model did not repeat — applying it deletes that code."""
        path = repo / "fedot" / "core" / "m.py"
        before = path.read_text(encoding="utf-8")
        p = Proposal(
            file_path="fedot/core/m.py",
            hunks=[("def build(task=None):", "def build(task=None):\n    ...\n    return 1")],
        )
        assert not apply_source(repo, p)
        assert path.read_text(encoding="utf-8") == before


class TestForgivingMatch:
    """Models re-type a quote instead of copying it; the block must still be found."""

    SRC = (
        "class A:\n"
        "    def f(self, task=1,\n"
        "          size=2):\n"
        '        """Doc.\n'
        "\n"
        "        Args:\n"
        "            task: a task\n"
        '        """\n'
        "        return task\n"
    )

    def _repo(self, tmp_path):
        (tmp_path / "fedot" / "core").mkdir(parents=True)
        (tmp_path / "fedot" / "core" / "m.py").write_text(self.SRC, encoding="utf-8")
        return tmp_path

    def test_wrong_indentation_still_matches(self, tmp_path):
        """Every line correct, all shifted one space — a whole run was lost to this."""
        repo = self._repo(tmp_path)
        p = Proposal(
            file_path="fedot/core/m.py",
            hunks=[("     def f(self, task=1,\n           size=2):", "    def f(self, task=None,\n          size=2):")],
        )
        assert apply_source(repo, p)
        assert "task=None" in (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8")

    def test_dropped_blank_lines_still_match(self, tmp_path):
        """Quoting a docstring back, the model omits its empty lines."""
        repo = self._repo(tmp_path)
        quoted = '        """Doc.\n        Args:\n            task: a task\n        """'
        p = Proposal(file_path="fedot/core/m.py", hunks=[(quoted, '        """Doc."""')])
        assert apply_source(repo, p)
        assert "Args:" not in (repo / "fedot" / "core" / "m.py").read_text(encoding="utf-8")
