"""The agent's right to say "there is nothing here worth fixing".

Every run used to have to produce something, and over 150 runs that something
was, 26 times, a reworded error message. The rule is enforced here in code
because asking for it in the prompt was measured not to work: three separate
requirements were ignored by every model tried, `gpt-4o` included.
"""

import types
from pathlib import Path

import pytest

from fedotllm.agents.evolve import loop
from fedotllm.agents.evolve.loop import Proposal


class _Inference:
    """Records whether the loop ever reached the model."""

    def __init__(self):
        self.calls = 0
        self.config = types.SimpleNamespace(provider="test", model_name="stub")

    def query(self, *_args, **_kwargs):
        self.calls += 1
        return "PICK: fedot/x.py\nWHY: because"


@pytest.fixture
def loop_without_a_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "scout_pick", lambda *a, **k: ("fedot/x.py", "because"))
    monkeypatch.setattr(loop, "read_source_file", lambda *a, **k: "source")
    monkeypatch.setattr(loop, "module_usage_section", lambda *a, **k: "")
    monkeypatch.setattr(loop, "resolve_repo_python", lambda *a, **k: "python")
    monkeypatch.setattr(loop, "build_audit", lambda *a, **k: "# audit")
    monkeypatch.setattr(loop, "JOURNAL_ENABLED", False)
    return tmp_path


def _no_evidence(monkeypatch):
    monkeypatch.setattr(loop, "proven_defect_for", lambda *a, **k: None)
    monkeypatch.setattr(loop, "invariant_defect_for", lambda *a, **k: None)


def test_preexisting_evidence_overrides_the_model_test():
    proposal = Proposal(
        test_file="test/test_model_choice.py",
        test_name="trusted_name",
        test_code="def trusted_name():\n    assert patch_specific_condition",
    )
    ready = types.SimpleNamespace(
        test_name="trusted_name",
        test_code="def trusted_name():\n    assert original_proof",
    )

    bound = loop.bind_evidence_test(proposal, ready=ready)

    assert bound.test_file == "test/unit/test_fedotllm_evolve_proof.py"
    assert bound.test_name == ready.test_name
    assert bound.test_code == ready.test_code


def test_evidence_not_model_wording_determines_severity():
    ready = types.SimpleNamespace(rule="B008")
    severity, _ = loop.classify_accepted_severity(
        ready, None, [], "harmless cosmetic wording"
    )
    assert severity == 1


def test_mutable_class_attribute_is_not_automatically_a_runtime_defect():
    ready = types.SimpleNamespace(rule="RUF012")
    severity, _ = loop.classify_accepted_severity(
        ready, None, [], "the model calls this a critical shared-state bug"
    )
    assert severity == 3


def test_without_evidence_the_run_abstains_before_spending_a_token(
        loop_without_a_repo, monkeypatch):
    _no_evidence(monkeypatch)
    monkeypatch.setattr(loop, "VALUE_GATE", True)
    called = {"proposal": False}
    monkeypatch.setattr(loop, "ask_proposal",
                        lambda *a, **k: called.__setitem__("proposal", True))

    inference = _Inference()
    result = loop.run_evolution_loop(
        inference, Path("/nonexistent"), loop_without_a_repo, venv_python="python")

    assert result.abstained is True
    assert result.success is False
    assert "no pre-existing evidence" in result.abstain_reason
    assert called["proposal"] is False, "the model must not be asked for a patch"


def test_with_the_gate_off_the_same_run_proceeds_to_the_model(
        loop_without_a_repo, monkeypatch):
    _no_evidence(monkeypatch)
    monkeypatch.setattr(loop, "VALUE_GATE", False)
    seen = {"proposal": False}

    def _ask(*_a, **_k):
        seen["proposal"] = True
        raise RuntimeError("stop here -- reaching the model is the whole assertion")

    monkeypatch.setattr(loop, "ask_proposal", _ask)
    with pytest.raises(RuntimeError):
        loop.run_evolution_loop(_Inference(), Path("/nonexistent"),
                                loop_without_a_repo, venv_python="python")
    assert seen["proposal"] is True


def test_evidence_lets_the_run_start(loop_without_a_repo, monkeypatch):
    monkeypatch.setattr(loop, "VALUE_GATE", True)
    monkeypatch.setattr(loop, "proven_defect_for", lambda *a, **k: None)
    monkeypatch.setattr(loop, "invariant_defect_for", lambda *a, **k: {
        "file": "fedot/x.py", "operation": "op", "param": "p", "value": 1,
        "test_name": "t", "test_code": "assert True", "observed": {"impl": "2"}})
    seen = {"proposal": False}

    def _ask(*_a, **_k):
        seen["proposal"] = True
        raise RuntimeError("stop here")

    monkeypatch.setattr(loop, "ask_proposal", _ask)
    with pytest.raises(RuntimeError):
        loop.run_evolution_loop(_Inference(), Path("/nonexistent"),
                                loop_without_a_repo, venv_python="python")
    assert seen["proposal"] is True


def test_an_abstention_is_not_reported_as_a_success(loop_without_a_repo, monkeypatch):
    _no_evidence(monkeypatch)
    monkeypatch.setattr(loop, "VALUE_GATE", True)
    monkeypatch.setattr(loop, "ask_proposal", lambda *a, **k: None)
    result = loop.run_evolution_loop(_Inference(), Path("/nonexistent"),
                                     loop_without_a_repo, venv_python="python")
    assert (result.success, result.abstained) == (False, True)
    assert result.proposal.problem == "", "an abstention must not carry a made-up problem"


class TestTuningGate:
    """A patch that stops the error by discarding the caller's parameter must
    not be accepted. Measured: one such patch cleared all five earlier gates.
    """

    def test_no_metric_means_the_gate_refuses(self, monkeypatch):
        monkeypatch.setattr(loop, "run_cmd", lambda *a, **k: loop.CommandResult(
            "x", 0, '{"operation": "catboost", "ok": false, "obtained_metric": null,'
                    ' "reason": "every tuning candidate failed to fit"}'))
        ok, summary = loop.run_tuning_gate(Path("/repo"), "python", "catboost")
        assert ok is False
        assert "still cannot be tuned" in summary

    def test_a_metric_means_the_gate_passes(self, monkeypatch):
        monkeypatch.setattr(loop, "run_cmd", lambda *a, **k: loop.CommandResult(
            "x", 0, '{"operation": "rf", "ok": true, "obtained_metric": -1.0}'))
        ok, summary = loop.run_tuning_gate(Path("/repo"), "python", "rf")
        assert ok is True

    def test_a_gate_that_cannot_run_fails_closed(self, monkeypatch):
        monkeypatch.setattr(loop, "run_cmd",
                            lambda *a, **k: loop.CommandResult("x", 1, "boom"))
        ok, summary = loop.run_tuning_gate(Path("/repo"), "python", "rf")
        assert (ok, "unavailable" in summary) == (False, True)
