"""Classification of runtime probe results.

The probe is what makes a target worth fixing: lint-only rounds scored 100%
success with zero real repairs. What counts as a finding therefore has to be
exact — especially the control cases, which are the only thing stopping a
"fix" that simply refuses all input.
"""

import subprocess
import types
from pathlib import Path

import pytest

from fedotllm.agents.evolve import loop, probe
from fedotllm.agents.evolve.probe import (
    Finding,
    _classify,
    compare_probes,
    probe_section,
)

CATBOOST_ERROR = (
    "Initial pipeline fit was failed due to: "
    "catboost/private/libs/target/target_converter.cpp:404: "
    "Target contains only one unique value."
)


class TestClassify:
    def test_success_is_not_a_finding(self):
        assert _classify({"case": "control_plain", "must_pass": True, "ok": True}) is None

    def test_broken_control_is_the_most_severe(self):
        f = _classify(
            {
                "case": "control_plain",
                "must_pass": True,
                "ok": False,
                "exc": "ValueError",
                "message": "boom",
                "location": "api/x.py:10",
            }
        )
        assert f is not None
        assert (f.kind, f.severity) == ("control_broken", 1)

    def test_leaked_third_party_error_is_a_finding(self):
        """Refusing awkward input is fine; quoting a foreign C++ file is not."""
        f = _classify(
            {
                "case": "single_class_target",
                "must_pass": False,
                "ok": False,
                "exc": "ValueError",
                "message": CATBOOST_ERROR,
                "location": "api/api_utils/assumptions/assumptions_handler.py:92",
            }
        )
        assert f is not None
        assert (f.kind, f.severity) == ("leaked_foreign_error", 2)

    def test_clear_own_refusal_is_not_a_finding(self):
        """FEDOT saying plainly why it refuses is correct behaviour."""
        assert (
            _classify(
                {
                    "case": "three_rows",
                    "must_pass": False,
                    "ok": False,
                    "exc": "ValueError",
                    "message": "Not enough rows to build a pipeline: 3 given, 10 required.",
                    "location": "api/main.py:50",
                }
            )
            is None
        )

    def test_unclear_exception_type_is_a_finding(self):
        f = _classify(
            {
                "case": "three_rows",
                "must_pass": False,
                "ok": False,
                "exc": "KeyError",
                "message": "'target'",
                "location": "api/main.py:50",
            }
        )
        assert f is not None
        assert (f.kind, f.severity) == ("unclear_error", 3)


class TestProbeSection:
    def test_no_findings_produces_nothing(self):
        """Never announce a clean runtime as if it were a result."""
        assert probe_section([]) == ""

    def test_section_names_case_and_location(self):
        text = probe_section(
            [
                Finding(
                    case="single_class_target",
                    kind="leaked_foreign_error",
                    severity=2,
                    exc="ValueError",
                    message=CATBOOST_ERROR,
                    location="api/api_utils/assumptions/assumptions_handler.py:92",
                )
            ]
        )
        assert "single_class_target" in text
        assert "assumptions_handler.py:92" in text

    def test_section_respects_the_limit(self):
        findings = [
            Finding(f"case{i}", "unclear_error", 3, "KeyError", "m", "a.py:1")
            for i in range(12)
        ]
        text = probe_section(findings, limit=3)
        assert "case2" in text and "case5" not in text


class TestCompareProbes:
    """The runtime gate: what counts as an improvement and what is cheating."""

    @staticmethod
    def _f(case, kind="leaked_foreign_error", severity=2):
        return Finding(case, kind, severity, "ValueError", "m", "a.py:1")

    def test_resolved_finding_is_accepted(self):
        ok, resolved, why = compare_probes([self._f("single_class_target")], [])
        assert ok and resolved == ["single_class_target"]
        assert "resolved" in why

    def test_breaking_ordinary_data_is_refused(self):
        """The cheapest way to make a defect vanish is to reject valid input."""
        ok, _, why = compare_probes(
            [self._f("single_class_target")],
            [self._f("control_plain", kind="control_broken", severity=1)],
        )
        assert not ok
        assert "ordinary data" in why

    def test_new_defect_is_refused(self):
        ok, _, why = compare_probes([], [self._f("inf_column")])
        assert not ok
        assert "inf_column" in why

    def test_unchanged_runtime_passes_without_credit(self):
        before = [self._f("single_class_target")]
        ok, resolved, why = compare_probes(before, list(before))
        assert ok and resolved == []
        assert "no runtime defect resolved" in why


def test_probe_gate_without_a_baseline_fails_closed(monkeypatch):
    monkeypatch.setattr(probe, "read_pristine_findings", lambda _repo: None)
    ok, resolved, why = loop.run_probe_gate(Path("/repo"), "python")
    assert ok is False
    assert resolved == []
    assert "unavailable" in why


def test_probe_timeout_is_not_reported_as_a_clean_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("probe", 1)
        ),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        probe.run_probe(tmp_path, "python")


def test_probe_process_failure_is_not_reported_as_a_clean_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout="", stderr="worker crashed", returncode=1
        ),
    )
    with pytest.raises(RuntimeError, match="produced no result"):
        probe.run_probe(tmp_path, "python")
