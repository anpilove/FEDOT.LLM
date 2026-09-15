"""Refining localization must preserve the public claim and all acceptance gates."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

import pytest

from fedotllm.agents.evolve.agents import verifier
from fedotllm.agents.evolve.discovery.targets import apply_verified_target, resolve_verification_target
from fedotllm.agents.evolve.execution import run_code
from fedotllm.agents.evolve.types import PatchSite, VerificationResult


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "fedot").mkdir(parents=True)
    (root / "fedot/__init__.py").write_text("")
    (root / "fedot/a.py").write_text(
        'class Output:\n'
        '    def save(self):\n'
        '        """Return rows in their original order."""\n'
        '        return [2, 1]\n'
        '\n'
        'def unrelated():\n'
        '    return 0\n'
    )
    (root / "fedot/b.py").write_text('def other():\n    return 1\n')
    return root


def lead():
    return PatchSite("execution", "fedot/a.py", 7,
                     why="Output export must preserve row order", hypothesis_kind="correctness")


def test_refinement_preserves_identity_and_revalidates_source(source):
    refined, target = resolve_verification_target(source, lead(), file_path="fedot/a.py", line=4)
    assert refined.line == 4 and refined.why == lead().why
    assert target["symbol"] == "Output.save" and target["original_line"] == 7
    result = VerificationResult("verified_bug", resolved_target=target)
    assert apply_verified_target(source, lead(), result) == refined
    (source / "fedot/a.py").write_text((source / "fedot/a.py").read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        apply_verified_target(source, lead(), result)


@pytest.mark.parametrize("kwargs", [
    {"file_path": "fedot/b.py", "line": 2},
    {"file_path": "../outside.py", "line": 2},
    {"line": 100}, {"line": 5},
    {"line": 4, "symbol": "unrelated"},
])
def test_invalid_refinement_fails_closed(source, kwargs):
    with pytest.raises(ValueError):
        resolve_verification_target(source, lead(), **kwargs)


def test_trace_uses_actual_refined_method_in_child_process(source, monkeypatch):
    monkeypatch.setattr(run_code, "fedot_python", lambda root: sys.executable)
    code = "from fedot.a import Output\nassert Output().save() == [1, 2], 'row order'"
    for line, expected in [(7, False), (4, True)]:
        _, target = resolve_verification_target(source, lead(), line=line)
        result = run_code.run_fedot_snippet(source, code, trace_target={
            "file_path": target["file_path"], "symbol": target["symbol"],
        })
        assert result.status == "runtime_error" and result.target_reached is expected


@pytest.mark.parametrize("audit_verdict", ["supported", "inconclusive", "unsupported"])
def test_real_verifier_refines_trace_and_audit_but_keeps_original_claim(source, tmp_path, monkeypatch, audit_verdict):
    monkeypatch.setattr(run_code, "fedot_python", lambda root: sys.executable)
    monkeypatch.setattr(verifier, "docs_runtime", lambda *a: "")
    monkeypatch.setattr(verifier, "callers_runtime", lambda *a: "")
    class Inference:
        def create(self, prompt, schema):
            if schema is verifier.ContractSupportAudit:
                assert "Output.save" in prompt
                assert "Original public suspicion: Output export must preserve row order" in prompt
                assert "same public behavior" in prompt
                return schema(verdict=audit_verdict, reason="disclosed regression fixture")
            return schema(action="verify_bug", file_path="fedot/a.py", line=4,
                          claim="Output.save reverses rows",
                          reproduction_code="from fedot.a import Output\nassert Output().save() == [1, 2], 'row order'")
    work = tmp_path / "verification"
    work.mkdir()
    result = verifier.verify_lead(source, lead(), inference=Inference(), workspace=work,
                                  max_model_calls=1, correctness_only=True)
    assert result.status == {"supported": "verified_bug", "inconclusive": "inconclusive", "unsupported": "rejected"}[audit_verdict]
    if audit_verdict == "supported":
        assert result.stock_probe.target_reached is True
        assert result.resolved_target["symbol"] == "Output.save"
        saved = json.loads((work / "verifications/a-7/verification.json").read_text())
        assert saved["resolved_target"] == result.resolved_target
        assert saved["resolved_target"]["original_line"] == lead().line


def test_unreached_refined_target_never_reaches_contract_audit(source, monkeypatch):
    monkeypatch.setattr(run_code, "fedot_python", lambda root: sys.executable)
    class Inference:
        def create(self, prompt, schema):
            assert schema is not verifier.ContractSupportAudit
            return schema(action="verify_bug", file_path="fedot/a.py", line=4,
                          reproduction_code="from fedot.a import Output\nassert False, 'setup failed'\nOutput().save()")
    result = verifier.verify_lead(source, lead(), inference=Inference(), max_model_calls=1, correctness_only=True)
    assert not result.proceed and result.status != "verified_bug"


def test_resume_preserves_validated_target(source, tmp_path):
    from fedotllm.agents.evolve.storage.replay import load_resume_branch
    refined, target = resolve_verification_target(source, lead(), line=4)
    rows = [
        {"event": "verification", "hypothesis_id": "h", "lead": asdict(refined),
         **asdict(VerificationResult("verified_bug", reproduction_code="assert False", resolved_target=target))},
        {"event": "decision", "candidate": "c", "hypothesis_id": "h", "lead": asdict(refined)},
    ]
    (tmp_path / "journal.jsonl").write_text("\n".join(map(json.dumps, rows)))
    branch = load_resume_branch(tmp_path, candidate_id="c")
    assert branch["verification"].resolved_target == target
    assert apply_verified_target(source, branch["lead"], branch["verification"]) == refined


@pytest.mark.parametrize("metric_only", [False, True])
def test_controller_propagates_target_before_fixer_and_metric_plan(source, tmp_path, monkeypatch, metric_only):
    from fedotllm.agents.evolve.controller import campaign, metric_study
    from fedotllm.agents.evolve.types import EvolveAgentConfig, EvolveRunPolicy, ScoreResult
    from fedotllm.agents.evolve.types import TestResult as EvolveTestResult
    monkeypatch.setattr(run_code, "fedot_python", lambda root: sys.executable)
    _, target = resolve_verification_target(source, lead(), line=4)
    checked = VerificationResult("verified_bug", claim=lead().why,
        reproduction_code="from fedot.a import Output\nassert Output().save() == [1, 2]",
        resolved_target=target)
    monkeypatch.setattr(campaign, "verify_lead", lambda *a, **k: checked)
    baseline = ScoreResult("catboost", "ok", .8)
    monkeypatch.setattr(campaign, "measure_stock", lambda *a, **k: {"catboost": baseline})
    monkeypatch.setattr(campaign, "run_stock", lambda *a, **k: baseline)
    monkeypatch.setattr(campaign, "measure_fedot_tests", lambda *a, **k: EvolveTestResult("passed", 0))
    seen = []
    def preregister(refined, stock, tasks, path, **kwargs):
        assert refined.line == 4 and stock["catboost"] is baseline
        seen.append("preregister")
        return {"target_task": "catboost"}
    monkeypatch.setattr(metric_study, "preregister", preregister)
    class ReachedFixer(BaseException):
        pass
    def fixer(checkout, refined, **kwargs):
        assert refined.line == 4 and refined.why == lead().why
        assert seen == ["preregister"]  # Both acceptance paths freeze their benchmark before Fixer.
        raise ReachedFixer
    monkeypatch.setattr(campaign, "fix_lead", fixer)
    work = tmp_path / "controller"
    with pytest.raises(ReachedFixer):
        campaign.run_once(checkout=source, workspace=work, inference=object(), verifier_inference=object(),
            resume_lead=lead(), lift_ids=("catboost",), protect_ids=("catboost",),
            max_leads=1, max_revisions=1,
            policy=EvolveRunPolicy(verify_manifest=False, fedot_quality_jobs=False),
            config=EvolveAgentConfig(metric_only=metric_only,
                                    metric_study_path=str(tmp_path / "study")))
    journal = [json.loads(line) for line in (work / "journal.jsonl").read_text().splitlines()]
    event = next(row for row in journal if row["event"] == "verified_target_refined")
    assert event["original_lead"]["line"] == 7 and event["lead"]["line"] == 4
    verification = next(row for row in journal if row["event"] == "verification")
    assert verification["resolved_target"] == target
    assert event["hypothesis_id"] == verification["hypothesis_id"]
