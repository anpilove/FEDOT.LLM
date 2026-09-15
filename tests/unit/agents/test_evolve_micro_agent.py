from __future__ import annotations

import shutil
from pathlib import Path

from fedotllm.agents.evolve.benchmark.micro import (
    MicroCase,
    MicroCaseOracle,
    MicroCasePrompt,
)
from fedotllm.agents.evolve.types import SnippetResult


def test_micro_discovery_pools_hide_balanced_target_positions():
    from fedotllm.agents.evolve.benchmark.micro import micro_cases
    from fedotllm.agents.evolve.benchmark.micro_discovery import discovery_pools

    pools = discovery_pools()
    positions = []
    for case in micro_cases():
        pool = pools[case.prompt.case_id]
        assert len(pool) == len(set(pool)) == 4
        assert pool.count(case.prompt.file_path) == 1
        positions.append(pool.index(case.prompt.file_path))
    assert positions[:4] == [0, 1, 2, 3]
    assert positions[4] == 0
    assert positions[5] == 1
    assert positions[6] == 1


def test_micro_localization_accepts_combined_symbol_answer():
    from fedotllm.agents.evolve.benchmark.micro import micro_cases
    from fedotllm.agents.evolve.benchmark.micro_agent import (
        LocalizationProposal,
        _localization_matches,
    )

    case = next(
        case
        for case in micro_cases()
        if case.prompt.case_id == "polyfit_parameter_identity"
    )
    proposal = LocalizationProposal(
        symbol=("PolyfitImplementation._correct_degree / PolyfitImplementation.degree"),
        line=1,
        mechanism="Both symbols participate in the same parameter contract.",
    )
    assert _localization_matches(case, proposal)


def test_monolith_micro_agent_keeps_controller_oracle_out_of_prompt(
    tmp_path: Path, monkeypatch
):
    from fedotllm.agents.evolve.benchmark import micro_agent
    from fedotllm.agents.evolve.benchmark.micro_agent import (
        MonolithProposal,
        run_micro_agent_benchmark,
    )

    source = tmp_path / "source"
    target = source / "fedot/a.py"
    target.parent.mkdir(parents=True)
    target.write_text("def public_value():\n    return 1\n", encoding="utf-8")
    (source / "fedot/__init__.py").write_text("", encoding="utf-8")
    case = MicroCase(
        prompt=MicroCasePrompt(
            case_id="private_case",
            file_path="fedot/a.py",
            symptom="The public value must be two.",
        ),
        oracle=MicroCaseOracle(
            symbols=("public_value",),
            stock_observation={"value": 1},
            patched_observation={"value": 2},
        ),
        behavior_probe="PRIVATE_CONTROLLER_PROBE",
    )

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []
            self.usage = {"requests": 0}

        def create(self, prompt, schema):
            self.prompts.append(prompt)
            assert schema is MonolithProposal
            return MonolithProposal(
                symbol="public_value",
                line=1,
                mechanism="The implementation returns the wrong public value.",
                proposed_change="Return two.",
                old_code="    return 1",
                new_code="    return 2",
                rationale="Match the public contract.",
            )

    def create_checkout(src, workspace, **_kwargs):
        checkout = workspace / "checkout"
        shutil.copytree(src, checkout)
        return checkout

    def discard_checkout(checkout, **_kwargs):
        shutil.rmtree(checkout)

    def run_probe(checkout, _code):
        value = 2 if "return 2" in (checkout / "fedot/a.py").read_text() else 1
        return SnippetResult(
            "ok",
            "probe",
            stdout=f'EVOLVE_OBSERVATION={{"value": {value}}}\n',
        )

    monkeypatch.setattr(micro_agent, "micro_cases", lambda: (case,))
    monkeypatch.setattr(micro_agent, "create_experiment_checkout", create_checkout)
    monkeypatch.setattr(micro_agent, "discard_experiment_checkout", discard_checkout)
    monkeypatch.setattr(micro_agent, "run_fedot_snippet", run_probe)
    inference = Inference()

    result = run_micro_agent_benchmark(
        source,
        tmp_path / "work",
        inference=inference,
        architecture="monolith",
    )

    assert result["ok"] is True
    assert result["counts"]["passed_cases"] == 1
    assert "PRIVATE_CONTROLLER_PROBE" not in inference.prompts[0]
    assert '"value": 2' not in inference.prompts[0]
    assert "public_value" in inference.prompts[0]
    assert target.read_text(encoding="utf-8").endswith("return 1\n")
