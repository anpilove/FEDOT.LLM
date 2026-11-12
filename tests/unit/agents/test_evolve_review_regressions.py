"""Regression tests for the September acceptance/retrieval audit.

Deterministic harness tests are not a prospective LLM performance measurement.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from fedotllm.agents.evolve.agents import verifier
from fedotllm.agents.evolve.benchmark import hidden_controls as hidden
from fedotllm.agents.evolve.storage import replay
from fedotllm.agents.evolve.types import (
    Decision, EvolveRunPolicy, PatchCandidate, PatchEdit, MatchSite, ScoreResult,
    SnippetResult, TestResult as EvolveTestResult,
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "fedot").mkdir(parents=True)
    (root / "fedot/__init__.py").write_text("")
    (root / "fedot/a.py").write_text(
        '"""Module contract."""\n'
        'class Relevant:\n'
        '    """Relevant class contract."""\n'
        '    def __init__(self):\n'
        '        """Correct constructor contract."""\n'
        '        self.value = 1\n'
    )
    (root / "fedot/b.py").write_text(
        'class Composer:\n'
        '    def __init__(self):\n'
        '        """Unrelated constructor contract."""\n'
        '        self.value = 99\n'
    )
    return root


def test_target_context_uses_exact_file_class_method_and_docstrings(source):
    symbol, code, docs = verifier._target_contract_context(source, "fedot/a.py", 6)
    assert symbol == "Relevant.__init__"
    assert "self.value = 1" in code
    assert "99" not in code
    assert "Module contract" in docs
    assert "Relevant class contract" in docs
    assert "Unrelated" not in docs


@pytest.mark.parametrize("path,line", [
    ("fedot/missing.py", 1), ("../outside.py", 1),
    ("fedot/a.py", 99), ("fedot/a.py", 1),
])
def test_unresolved_context_is_inconclusive_without_model_call(source, path, line):
    audit = verifier._audit_contract_support(
        source, MatchSite("execution", path, line),
        verifier.VerificationProposal(action="verify_bug"), inference=None,
    )
    assert audit.verdict == "inconclusive"


def test_audit_never_globally_resolves_constructor(source, monkeypatch):
    queries = []
    from fedotllm.agents.evolve.discovery import research_tools

    monkeypatch.setattr(research_tools, "symbol_runtime", lambda *a, **k: pytest.fail("global lookup"))
    monkeypatch.setattr(verifier, "docs_runtime", lambda root, q: queries.append(q) or "")
    monkeypatch.setattr(verifier, "callers_runtime", lambda root, q: queries.append(q) or "")

    class Inference:
        def create(self, prompt, schema):
            assert "Relevant.__init__" in prompt
            assert "Unrelated constructor" not in prompt
            assert "Missing evidence is not proof" in prompt
            return schema(verdict="inconclusive", reason="contract not established")

    result = verifier._audit_contract_support(
        source, MatchSite("execution", "fedot/a.py", 6),
        verifier.VerificationProposal(action="verify_bug"), inference=Inference(),
    )
    assert result.verdict == "inconclusive"
    assert queries == ["Relevant", "Relevant"]


def test_linked_contracts_resolve_return_alias_and_inherited_helper(source):
    (source / "fedot/a.py").write_text(
        "from fedot.types import Output as Result\nfrom fedot.base import Parent\n"
        "class Relevant(Parent):\n"
        "    def predict(self, data) -> Result:\n"
        "        return self.convert(data)\n")
    (source / "fedot/types.py").write_text(
        "class Output:\n    def consume(self):\n        assert len(self.idx) == len(self.predict)\n")
    (source / "fedot/base.py").write_text(
        "class Parent:\n    def convert(self, data):\n        return data\n")
    text = verifier._linked_contract_context(source, "fedot/a.py", 5)
    assert "fedot/types.py" in text and "len(self.idx) == len(self.predict)" in text
    assert "fedot/base.py" in text and "def convert" in text
    assert "Unrelated" not in text


def test_contract_audit_receives_actual_setup_failure(source, monkeypatch):
    monkeypatch.setattr(verifier, "docs_runtime", lambda *a: "")
    monkeypatch.setattr(verifier, "callers_runtime", lambda *a: "")
    class Inference:
        def create(self, prompt, schema):
            assert "actual setup failure" in prompt
            assert "method docstring need not repeat" in prompt
            return schema(verdict="inconclusive", reason="probe setup failed")
    verdict = verifier._audit_contract_support(source, MatchSite("execution", "fedot/a.py", 6),
        verifier.VerificationProposal(action="verify_bug"), inference=Inference(),
        stock_probe=SnippetResult("runtime_error", "", stderr="actual setup failure"))
    assert verdict.verdict == "inconclusive"


def test_acceptance_pytest_never_uses_discovery_fail_limit(source, monkeypatch):
    from fedotllm.agents.evolve.evaluation import judge
    calls = []
    monkeypatch.setattr(judge, "pytest_snapshot", lambda root, **kw: calls.append(kw) or EvolveTestResult("passed", 0))
    judge.measure_fedot_tests(source)
    assert calls == [{"maxfail": 0}]


def test_early_pytest_stop_is_incomplete_not_a_baseline(source, monkeypatch):
    import subprocess
    from fedotllm.agents.evolve.discovery import signals
    (source / "test/unit").mkdir(parents=True)
    monkeypatch.setattr(signals.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=[], returncode=1, stdout="FAILED test/unit/a.py::test_a\n!!!! stopping after 8 failures !!!!", stderr=""))
    result = signals.pytest_result(source)
    assert result.status == "incomplete" and not result.completed


def test_baseline_cache_identifies_test_data_and_empty_directories(source):
    from fedotllm.agents.evolve.evaluation.judge import _baseline_test_fingerprint
    before = _baseline_test_fingerprint(source)
    data = source / "examples/data"
    (data / "empty").mkdir(parents=True)
    empty = _baseline_test_fingerprint(source)
    assert empty != before
    (data / "input.csv").write_text("x\n1\n")
    one = _baseline_test_fingerprint(source)
    (data / "input.csv").write_text("x\n2\n")
    assert one != empty and _baseline_test_fingerprint(source) != one


def test_empty_fixture_hydrated_only_in_disposable_test_copy(source):
    import io
    import tarfile
    from fedotllm.agents.evolve.evaluation.test_contracts import run_with_test_repairs
    folder = source / "examples/data/multimodal"
    folder.mkdir(parents=True)
    archive_path = folder.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("multimodal/example.json")
        payload = b'{"votes":1}'
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    def run(tree):
        assert tree != source
        assert (tree / "examples/data/multimodal/example.json").read_bytes() == payload
        return EvolveTestResult("passed", 0)
    assert run_with_test_repairs(source, run).status == "passed"
    assert not list(folder.iterdir())


def test_fixture_archive_cannot_escape_expected_directory(source):
    import io
    import tarfile
    from fedotllm.agents.evolve.evaluation.test_contracts import run_with_test_repairs
    folder = source / "examples/data/multimodal"
    folder.mkdir(parents=True)
    with tarfile.open(folder.with_suffix(".tar.gz"), "w:gz") as archive:
        member = tarfile.TarInfo("../escape.py")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        run_with_test_repairs(source, lambda *a: pytest.fail("unsafe runner"))


def test_audit_inconclusive_propagates_through_real_verifier(source, monkeypatch):
    monkeypatch.setattr(verifier, "run_fedot_snippet", lambda root, code, **kwargs: SnippetResult(
        "runtime_error", code, stderr="AssertionError: value contract", exit_code=1, target_reached=True,
    ))

    prompts: list[str] = []

    class Inference:
        def create(self, prompt, schema):
            if schema is verifier.ContractSupportAudit:
                return schema(verdict="inconclusive", reason="missing lifecycle contract")
            prompts.append(prompt)
            return schema(
                action="verify_bug", claim="value contract",
                reproduction_code="from fedot.a import Relevant\nx = Relevant()\nassert x.value == 2",
            )

    result = verifier.verify_lead(
        source, MatchSite("execution", "fedot/a.py", 6, hypothesis_kind="correctness"),
        inference=Inference(), max_model_calls=3, correctness_only=True,
    )
    assert result.status == "inconclusive"
    assert not result.proceed
    assert "unsupported public-contract" not in result.detail
    assert len(prompts) == 2
    assert "Assert the failed behavior" in prompts[1]


def test_annotated_int_or_recognizes_a_zero_seed():
    from fedotllm.agents.evolve.agents.verifier import annotated_int_or_coerces_zero

    source = (
        "def train_test_data_setup(random_seed: int = 42, shuffle: bool = False):\n"
        "    random_seed = (random_seed or 42) if shuffle else None\n"
        "    return random_seed\n"
    )
    assert "random_seed" in annotated_int_or_coerces_zero(source, 2)
    assert annotated_int_or_coerces_zero(source, 3) == ""
    untouched = source.replace("random_seed: int = 42", "random_seed: str = '42'")
    assert annotated_int_or_coerces_zero(untouched, 2) == ""


def test_one_inconclusive_correctness_lead_is_resumed(tmp_path):
    seed = {
        "file_path": "fedot/core/data/data_split.py",
        "line": 198,
        "why": "seed 0 becomes 42",
        "hypothesis_kind": "correctness",
        "channel": "llm",
        "mechanism": "random_seed or 42",
        "proposed_change": "use None instead of truthiness",
    }
    repeated = {
        "file_path": "fedot/other.py",
        "line": 4,
        "hypothesis_kind": "correctness",
    }
    twice = {
        "file_path": "fedot/twice.py",
        "line": 2,
        "hypothesis_kind": "correctness",
        "why": "still open",
    }
    rows = [
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": seed,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": repeated,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": repeated,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": repeated,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": twice,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": twice,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "rejected", "lead": {
            "file_path": "fedot/done.py", "line": 9, "hypothesis_kind": "correctness",
        }, "reproduction": {"status": "quality_hypothesis"}},
    ]
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(map(json.dumps, rows)))
    resumed = replay.unfinished_correctness_leads(findings, limit=2)
    assert [(item.file_path, item.line) for item in resumed] == [
        ("fedot/twice.py", 2),
        ("fedot/core/data/data_split.py", 198),
    ]
    parked = {
        "file_path": "fedot/core/data/data_split.py",
        "line": 198,
        "why": "seed 0 becomes 42",
        "hypothesis_kind": "correctness",
        "proposed_change": "use None instead of truthiness",
    }
    rows.append({
        "record_type": "finding",
        "outcome": "infrastructure_error",
        "lead": parked,
        "tests": {"status": "passed"},
        "dev": {"reason": "regression pca->catboost timeout_or_invalid"},
        "reproduction": {"status": "verified_bug"},
    })
    findings.write_text("\n".join(map(json.dumps, rows)))
    resumed = replay.unfinished_correctness_leads(findings, limit=3)
    assert ("fedot/core/data/data_split.py", 198) in {
        (item.file_path, item.line) for item in resumed
    }
    seed_lead = next(item for item in resumed if item.line == 198)
    assert seed_lead.hypothesis_kind == "correctness"
    assert seed_lead.proposed_change.startswith("use None")
    noisy = {
        "file_path": "fedot/noisy.py",
        "line": 390,
        "hypothesis_kind": "correctness",
        "why": "imputer state",
    }
    rows.extend([
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": noisy,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": noisy,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "outcome": "unverified_no_patch", "lead": noisy,
         "reproduction": {"status": "rejected"}},
    ])
    findings.write_text("\n".join(map(json.dumps, rows)))
    resumed = replay.unfinished_correctness_leads(findings, limit=1)
    assert [(item.file_path, item.line) for item in resumed] == [
        ("fedot/core/data/data_split.py", 198),
    ]
    proven = next(row for row in rows if row.get("outcome") == "infrastructure_error")
    proven.update({
        "candidate_id": "seed-patch",
        "patch_hash": "seed-hash",
        "source_hash": "src",
        "evaluation_protocol_hash": "proto",
        "edits": [{
            "file_path": "fedot/core/data/data_split.py",
            "old_code": "random_seed or 42",
            "new_code": "42 if random_seed is None else random_seed",
        }],
        "reproduction": {
            "status": "verified_bug",
            "code": "assert seed == 0",
            "claim": "seed 0 is aliased",
        },
    })
    rows.append({
        "record_type": "finding",
        "outcome": "unverified_no_patch",
        "lead": parked,
        "reproduction": {"status": "inconclusive"},
    })
    rows.append({
        "record_type": "finding",
        "outcome": "rejected",
        "patch_hash": "kept-hash",
        "source_hash": "src",
        "evaluation_protocol_hash": "proto",
        "lead": {
            "file_path": "fedot/other.py",
            "line": 1,
            "hypothesis_kind": "quality_hypothesis",
        },
        "dev": {"reason": "tests_regressed"},
    })
    findings.write_text("\n".join(map(json.dumps, rows)))
    resumed = replay.unfinished_correctness_leads(findings, limit=1)
    assert [(item.file_path, item.line) for item in resumed] == [
        ("fedot/core/data/data_split.py", 198),
    ]
    verification, candidate = replay.proven_correctness_replay(findings)[
        ("fedot/core/data/data_split.py", 198)
    ]
    assert verification.status == "verified_bug"
    assert verification.reproduction_code == "assert seed == 0"
    assert "is None" in candidate.new_code
    assert replay.tried_patch_hashes_from_findings(
        findings, source_hash="src", evaluation_protocol_hash="proto",
    ) == {"kept-hash"}
    rows.append({
        "record_type": "finding",
        "outcome": "infrastructure_error",
        "lead": parked,
        "tests": {"status": "passed"},
        "dev": {"reason": "regression pca->catboost timeout_or_invalid"},
        "reproduction": {"status": "verified_bug"},
    })
    findings.write_text("\n".join(map(json.dumps, rows)))
    resumed = replay.unfinished_correctness_leads(findings, limit=3)
    assert ("fedot/core/data/data_split.py", 198) not in {
        (item.file_path, item.line) for item in resumed
    }
    assert ("fedot/noisy.py", 390) not in {
        (item.file_path, item.line) for item in resumed
    }


def test_finished_correctness_sites_stay_out_of_the_next_catalog(tmp_path):
    from fedotllm.agents.evolve.controller.session import kept_patch_report

    seed = {
        "file_path": "fedot/core/data/data_split.py",
        "line": 198,
        "hypothesis_kind": "correctness",
    }
    imputer = {
        "file_path": "fedot/sklearn_transformations.py",
        "line": 390,
        "hypothesis_kind": "correctness",
    }
    open_lead = {
        "file_path": "fedot/open.py",
        "line": 3,
        "hypothesis_kind": "correctness",
    }
    rows = [
        {"record_type": "finding", "source_hash": "src", "outcome": "correctness_keep",
         "lead": seed, "reproduction": {"status": "verified_bug"}},
        {"record_type": "finding", "source_hash": "src", "outcome": "unverified_no_patch",
         "lead": imputer, "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "source_hash": "src", "outcome": "unverified_no_patch",
         "lead": imputer, "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "source_hash": "src", "outcome": "unverified_no_patch",
         "lead": imputer, "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "source_hash": "src", "outcome": "unverified_no_patch",
         "lead": open_lead, "reproduction": {"status": "inconclusive"}},
        {"record_type": "finding", "source_hash": "other", "outcome": "rejected",
         "lead": {"file_path": "fedot/other.py", "line": 1, "hypothesis_kind": "correctness"},
         "reproduction": {"status": "rejected"}},
    ]
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(map(json.dumps, rows)))
    assert replay.closed_correctness_sites(findings, source_hash="src") == {
        ("fedot/core/data/data_split.py", 198),
        ("fedot/sklearn_transformations.py", 390),
    }
    rows.append({
        "record_type": "finding",
        "source_hash": "src",
        "outcome": "rejected_tests",
        "lead": {
            "file_path": "fedot/sklearn_transformations.py",
            "line": 316,
            "hypothesis_kind": "quality_hypothesis",
        },
        "tests": {"status": "test_failures"},
        "dev": {"reason": "fedot_tests_regressed test_imputation_with_binary_correct"},
    })
    rows.append({
        "record_type": "finding",
        "source_hash": "src",
        "outcome": "unverified_no_patch",
        "lead": {
            "file_path": "fedot/sklearn_transformations.py",
            "line": 316,
            "hypothesis_kind": "quality_hypothesis",
        },
        "dev": {"reason": "cannot_fix"},
    })
    findings.write_text("\n".join(map(json.dumps, rows)))
    assert ("fedot/sklearn_transformations.py", 316) in replay.closed_hunt_sites(
        findings, source_hash="src"
    )
    workspace = tmp_path / "run"
    patch_dir = workspace / "correctness_fixes"
    patch_dir.mkdir(parents=True)
    patch = patch_dir / "seed.patch"
    patch.write_text(
        "--- stock/fedot/core/data/data_split.py\n"
        "+++ patched/fedot/core/data/data_split.py\n"
        "-    random_seed = (random_seed or 42) if shuffle else None\n"
        "+    random_seed = (42 if random_seed is None else random_seed) if shuffle else None\n",
        encoding="utf-8",
    )
    report = kept_patch_report(
        workspace,
        {"artifacts": {"correctness_patches": ["correctness_fixes/seed.patch"]}},
    )
    assert report["file"] == "fedot/core/data/data_split.py"
    assert report["patch"] == str(patch)
    assert "is None" in report["diff"]


def test_probes_that_never_reach_the_target_stay_open_for_one_more_attempt(tmp_path):
    lead = {
        "file_path": "fedot/categorical_encoders.py",
        "line": 100,
        "hypothesis_kind": "correctness",
    }
    row = {
        "record_type": "finding",
        "source_hash": "src",
        "outcome": "unverified_no_patch",
        "lead": lead,
        "reproduction": {
            "status": "inconclusive",
            "detail": "verification step budget exhausted",
            "observed": (
                "Bug was not verified: probe did not execute the claimed "
                "target method; an earlier setup failure"
            ),
        },
    }
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(json.dumps(row) for _ in range(4)))
    assert ("fedot/categorical_encoders.py", 100) not in replay.closed_correctness_sites(
        findings, source_hash="src",
    )
    resumed = replay.unfinished_correctness_leads(findings, limit=1)
    assert [(item.file_path, item.line) for item in resumed] == [
        ("fedot/categorical_encoders.py", 100),
    ]
    findings.write_text("\n".join(json.dumps(row) for _ in range(5)))
    assert ("fedot/categorical_encoders.py", 100) in replay.closed_correctness_sites(
        findings, source_hash="src",
    )
    assert replay.unfinished_correctness_leads(findings, limit=1) == []
    earlier = {
        "record_type": "finding",
        "source_hash": "src",
        "outcome": "unverified_no_patch",
        "lead": lead,
        "reproduction": {
            "status": "inconclusive",
            "detail": "contract support inconclusive: crash before the assertion",
        },
    }
    findings.write_text("\n".join(json.dumps(item) for item in [earlier, *([row] * 5)]))
    assert ("fedot/categorical_encoders.py", 100) in replay.closed_correctness_sites(
        findings, source_hash="src",
    )
    assert replay.unfinished_correctness_leads(findings, limit=1) == []


def test_healthy_contract_rejection_stays_open_until_the_target_ran(tmp_path):
    missed = {
        "file_path": "fedot/sklearn_transformations.py",
        "line": 432,
        "hypothesis_kind": "correctness",
    }
    reached = {
        "file_path": "fedot/sklearn_transformations.py",
        "line": 457,
        "hypothesis_kind": "correctness",
    }
    explicit = {
        "file_path": "fedot/core/pipelines/pipeline.py",
        "line": 457,
        "hypothesis_kind": "correctness",
    }
    detail = (
        "stock satisfied the proposed healthy contract; no independent "
        "failure or grounded comparative quality hypothesis remained"
    )
    rows = [
        {
            "record_type": "finding",
            "source_hash": "src",
            "outcome": "unverified_no_patch",
            "lead": missed,
            "reproduction": {"status": "rejected", "detail": detail},
        },
        {
            "record_type": "finding",
            "source_hash": "src",
            "outcome": "unverified_no_patch",
            "lead": reached,
            "reproduction": {
                "status": "rejected",
                "detail": detail,
                "target_reached": True,
            },
        },
        {
            "record_type": "finding",
            "source_hash": "src",
            "outcome": "unverified_no_patch",
            "lead": explicit,
            "reproduction": {
                "status": "rejected",
                "detail": "unsupported public-contract precondition: documented rewrite",
            },
        },
    ]
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(map(json.dumps, rows)))
    closed = replay.closed_correctness_sites(findings, source_hash="src")
    assert ("fedot/sklearn_transformations.py", 432) not in closed
    assert ("fedot/sklearn_transformations.py", 457) in closed
    assert ("fedot/core/pipelines/pipeline.py", 457) in closed


def test_context_failure_is_not_a_site_cooldown_or_completed_hypothesis(tmp_path):
    lead = {"file_path": "fedot/a.py", "line": 6, "mechanism": "check value"}
    rows = [
        {"record_type": "run", "event": "run_start", "run_id": "r", "run_number": 1},
        {"record_type": "finding", "run_id": "r", "lead": lead,
         "reproduction": {"status": "inconclusive"}},
        {"record_type": "run", "event": "run_end", "run_id": "r", "immutable_source": True},
    ]
    findings = tmp_path / "findings.jsonl"
    findings.write_text("\n".join(map(json.dumps, rows)))
    assert replay.recent_completed_sites_from_findings(findings) == set()
    assert replay.recent_completed_hypotheses_from_findings(findings) == []
    (tmp_path / "scoreboard.jsonl").write_text(json.dumps({
        "event": "attempt", "lead": lead, "reason": "verification_inconclusive: context missing",
    }))
    assert replay.tried_sites(tmp_path) == set()
    rows[1]["reproduction"]["status"] = "rejected"
    findings.write_text("\n".join(map(json.dumps, rows)))
    assert replay.recent_completed_sites_from_findings(findings) == {("fedot/a.py", 6)}


@pytest.mark.parametrize("reached", [False, True])
def test_snippet_trace_distinguishes_setup_failure_from_target(source, monkeypatch, reached):
    from fedotllm.agents.evolve.execution import run_code
    import sys
    monkeypatch.setattr(run_code, "fedot_python", lambda root: sys.executable)
    code = "from fedot.a import Relevant\n"
    if reached:
        code += "Relevant()\n"
    code += "raise AssertionError('probe failed')\n"
    result = run_code.run_fedot_snippet(source, code, trace_target={
        "file_path": "fedot/a.py", "symbol": "Relevant.__init__",
    })
    assert result.status == "runtime_error"
    assert result.target_reached is reached


def test_verifier_rejects_assertion_before_target(source, monkeypatch):
    monkeypatch.setattr(verifier, "run_fedot_snippet", lambda root, code, **kwargs: SnippetResult(
        "runtime_error", code, stderr="AssertionError: setup failed", exit_code=1, target_reached=False,
    ))

    class Inference:
        def create(self, prompt, schema):
            assert schema is not verifier.ContractSupportAudit
            return schema(action="verify_bug", claim="value contract",
                          reproduction_code="from fedot.a import Relevant\nassert False, 'setup failed'")

    result = verifier.verify_lead(
        source, MatchSite("execution", "fedot/a.py", 6, hypothesis_kind="correctness"),
        inference=Inference(), max_model_calls=2, correctness_only=True,
    )
    assert not result.proceed
    assert result.status != "verified_bug"


def _accepted(new_value=2):
    return {
        "event": "decision", "candidate": "p", "correctness_keep": True,
        "edits": [asdict(PatchEdit("fedot/a.py", "self.value = 1", f"self.value = {new_value}"))],
    }


def test_dev_success_is_not_final_acceptance_and_final_uses_ablated_edits(tmp_path):
    dev = {**_accepted(), "correctness_keep": False, "keep": True}
    final = {**dev, "event": "final", "keep_final": False}
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(map(json.dumps, [dev, final])))
    assert hidden._accepted_decisions(tmp_path) == []
    final.update(keep_final=True, edits=_accepted(3)["edits"])
    journal.write_text("\n".join(map(json.dumps, [dev, final])))
    assert hidden._accepted_decisions(tmp_path) == [final]


@pytest.mark.parametrize("observation,status,confirmed,false_accept,execution_ok", [
    ("correct", "ok", True, False, True),
    ("wrong", "ok", False, True, True),
    ("", "runtime_error", False, True, True),
    ("", "timeout", False, None, False),
    ("", "ok", False, None, False),
])
def test_independent_oracle_can_detect_false_accepts_and_losses(
    source, tmp_path, monkeypatch, observation, status, confirmed, false_accept, execution_ok,
):
    monkeypatch.setattr(hidden, "run_fedot_snippet", lambda root, code: SnippetResult(
        status, code, stdout=f"EVOLVE_OBSERVATION={observation}" if observation else "",
    ))
    decision = _accepted()
    result, = hidden._independent_assessment(
        source, tmp_path / "audit", [decision], probe="PRIVATE_PROBE",
        healthy_observation="correct", healthy_case=False,
    )
    assert result["controller_accepted"] is True
    assert result["confirmed_fix"] is confirmed
    assert result["false_accept"] is false_accept
    assert result["execution_ok"] is execution_ok
    assert decision["correctness_keep"] is True  # never rewritten by oracle
    assert "self.value = 1" in (source / "fedot/a.py").read_text()


def test_hidden_harness_calls_production_entry_for_mutation_and_healthy(
    source, tmp_path, monkeypatch,
):
    from fedotllm.agents.evolve.controller import campaign
    control = hidden.HiddenControl(
        "case", "component", "private symptom", "public value contract",
        "fedot/a.py", "self.value = 1", "self.value = 0", "Relevant", "PRIVATE_PROBE",
    )
    monkeypatch.setattr(hidden, "run_fedot_snippet", lambda root, code: SnippetResult(
        "ok", code, stdout="EVOLVE_OBSERVATION=" + (
            "wrong" if "self.value = 0" in (root / "fedot/a.py").read_text() else "correct"
        ),
    ))
    calls = []

    class Inference:
        def create(self, prompt, schema):
            assert "PRIVATE_PROBE" not in prompt
            assert "private symptom" not in prompt
            if schema is hidden.FileShortlist:
                return schema(selected_indices=[0])
            return schema(selected_index=0, line=6, mechanism="value assignment")

    def controller(**kwargs):
        # Adapter/wiring test: expensive production execution is replaced here;
        # real controller behavior is covered by the existing integration suite.
        calls.append(kwargs)
        assert "probe" not in kwargs and "resume_verification" not in kwargs
        assert kwargs["policy"].confirm_and_ablate
        assert kwargs["policy"].evaluate_final
        root = kwargs["checkout"]
        work = kwargs["workspace"]
        work.mkdir(parents=True)
        if "self.value = 0" in (root / "fedot/a.py").read_text():
            bad = _accepted(7)
            bad["edits"][0]["old_code"] = "self.value = 0"
            (work / "journal.jsonl").write_text(json.dumps(bad))
            return Decision(False, "correctness_keep", None, correctness_keep=True)
        (work / "journal.jsonl").write_text(json.dumps({"event": "decision", "keep": False}))
        return Decision(False, "verification_rejected", None)

    monkeypatch.setattr(campaign, "run_once", controller)
    # The private probe distinguishes the correct reference from the bad repair.
    monkeypatch.setattr(hidden, "run_fedot_snippet", lambda root, code: SnippetResult(
        "ok", code, stdout="EVOLVE_OBSERVATION=" + (
            "correct" if "self.value = 1" in (root / "fedot/a.py").read_text() else "wrong"
        ),
    ))
    result = hidden.run_hidden_control_benchmark(
        source, tmp_path / "benchmark", inference=Inference(),
        controls=(control,), catalog=("fedot/a.py",),
    )
    assert len(calls) == 2
    assert result["metrics"]["false_accepts"] == 1
    assert result["metrics"]["healthy_rejections"] == 1
    assert result["metrics"]["autonomous_confirmed_fixes"] == 0
    assert result["execution_ok"] is True
    assert result["ok"] is False
    assert result["source_unchanged"] is True


def test_empty_control_set_cannot_pass(source, tmp_path):
    result = hidden.run_hidden_control_benchmark(
        source, tmp_path / "empty", inference=None, controls=(),
    )
    assert not result["ok"]


@pytest.mark.parametrize("patched_value,false_accept", [(2, False), (3, True)])
def test_real_controller_decision_is_judged_by_separate_stricter_oracle(
    source, tmp_path, monkeypatch, patched_value, false_accept,
):
    from fedotllm.agents.evolve.controller import campaign

    # Deliberately imperfect model verification: value > 1 is weaker than the
    # independent oracle's value == 2. Do not feed the strict oracle to the agent.
    class Inference:
        def create(self, prompt, schema):
            assert "PRIVATE_ORACLE" not in prompt
            if schema is verifier.ContractSupportAudit:
                return schema(verdict="supported", reason="test double accepts weak contract")
            return schema(
                action="verify_bug", claim="constructor value should exceed one",
                reproduction_code="from fedot.a import Relevant\nx = Relevant()\nassert x.value > 1",
            )

    def fixer(checkout, *args, **kwargs):
        candidate = PatchCandidate(
            "repair", edits=[PatchEdit(
                "fedot/a.py", "self.value = 1", f"self.value = {patched_value}",
            )],
        )
        # The real Fixer applies its patch before returning the candidate.
        assert hidden.apply_patch(checkout, candidate)
        return candidate

    monkeypatch.setattr(campaign, "fix_lead", fixer)
    scores = {"catboost": ScoreResult("catboost", "ok", 0.8)}
    monkeypatch.setattr(campaign, "measure_stock", lambda *a, **k: scores)
    monkeypatch.setattr(campaign, "measure_patched", lambda *a, **k: scores)
    monkeypatch.setattr(campaign, "measure_fedot_tests", lambda *a, **k: EvolveTestResult("passed", 0))
    work = tmp_path / "real-controller"
    inference = Inference()
    decision = campaign.run_once(
        checkout=source, workspace=work, inference=inference, verifier_inference=inference,
        resume_lead=MatchSite("execution", "fedot/a.py", 6, hypothesis_kind="correctness"),
        lift_ids=("catboost",), protect_ids=("catboost",),
        max_leads=1, max_revisions=1, policy=EvolveRunPolicy(verify_manifest=False, fedot_quality_jobs=False),
    )
    assert decision.correctness_keep
    accepted = hidden._accepted_decisions(work)
    assert len(accepted) == 1
    results = hidden._independent_assessment(
        source, tmp_path / "independent", accepted,
        probe="from fedot.a import Relevant\n# PRIVATE_ORACLE\nprint('EVOLVE_OBSERVATION=' + str(Relevant().value))",
        healthy_observation="2", healthy_case=False,
    )
    assert results[0]["controller_accepted"] is True
    assert results[0]["execution_ok"] is True
    assert results[0]["false_accept"] is false_accept
    assert results[0]["confirmed_fix"] is not false_accept


def test_importless_probe_resume_names_the_missing_import(tmp_path):
    findings = tmp_path / "findings.jsonl"
    findings.write_text(json.dumps({
        "record_type": "finding",
        "outcome": "unverified_no_patch",
        "lead": {
            "file_path": "fedot/sklearn_transformations.py",
            "line": 391,
            "hypothesis_kind": "correctness",
            "why": "binary ids accumulate across transform",
        },
        "reproduction": {
            "status": "inconclusive",
            "observed": "Probe rejected: probe must import FEDOT",
            "detail": "verification step budget exhausted",
        },
    }) + "\n")
    resumed = replay.unfinished_correctness_leads(findings, limit=1)
    assert [(item.file_path, item.line) for item in resumed] == [
        ("fedot/sklearn_transformations.py", 391),
    ]
    assert "must import fedot" in resumed[0].why
