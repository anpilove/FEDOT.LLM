from __future__ import annotations

from pathlib import Path

import pytest

from fedotllm.agents.evolve.benchmark.micro import (
    MicroCaseResult,
    MicroStageResult,
    micro_cases,
    model_facing_context,
    run_stock_microbenchmark,
)
from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src


def test_micro_case_catalog_is_private_and_diverse():
    cases = micro_cases()

    assert len(cases) == 7
    assert len({case.prompt.case_id for case in cases}) == len(cases)
    paths = [case.prompt.file_path for case in cases]
    assert len(set(paths)) >= len(cases) - 1
    assert max(paths.count(path) for path in set(paths)) <= 2
    for case in cases:
        context = model_facing_context(case)
        assert case.prompt.symptom in context
        assert case.prompt.file_path in context
        assert case.behavior_probe not in context
        assert not any(symbol in context for symbol in case.oracle.symbols)
        assert "old_code" not in context
        assert "new_code" not in context


def test_hidden_controls_are_three_private_cross_component_mutations():
    from fedotllm.agents.evolve.benchmark.hidden_controls import _CONTROLS as controls

    assert len(controls) == 3
    assert len({case.component for case in controls}) == 3
    assert len({case.file_path for case in controls}) == 3
    assert all(case.control_id.startswith("heldout-") for case in controls)
    assert all(case.old_code != case.defective_code for case in controls)
    # Public symptoms must not disclose the private target or exact correction.
    assert all(case.file_path not in case.symptom for case in controls)
    assert all(case.responsible_symbol not in case.symptom for case in controls)


def test_fresh_hidden_controls_are_disjoint_and_keep_oracles_private():
    from fedotllm.agents.evolve.benchmark.hidden_controls import (
        _CONTROLS as old,
        _FRESH_CONTROLS as fresh,
        _FRESH_V2_CONTROLS as fresh_v2,
    )

    assert len(fresh) == 3
    assert {case.control_id for case in fresh}.isdisjoint(
        case.control_id for case in old
    )
    assert {case.file_path for case in fresh} != {case.file_path for case in old}
    assert all(case.old_code != case.defective_code for case in fresh)
    assert all(case.file_path not in case.symptom for case in fresh)
    assert all(case.responsible_symbol not in case.symptom for case in fresh)

    assert len(fresh_v2) == 3
    assert {case.control_id for case in fresh_v2}.isdisjoint(
        {case.control_id for case in (*old, *fresh)}
    )
    assert all(case.old_code != case.defective_code for case in fresh_v2)
    assert all(case.file_path not in case.symptom for case in fresh_v2)
    assert all(case.responsible_symbol not in case.symptom for case in fresh_v2)


def test_micro_stage_results_require_pipeline_order():
    row = MicroCaseResult("case")
    with pytest.raises(ValueError, match="expected stage 'stock_probe'"):
        row.add(MicroStageResult("verification", "passed"))

    row.add(MicroStageResult("stock_probe", "passed"))
    row.add(MicroStageResult("localization", "passed"))
    row.add(MicroStageResult("verification", "failed", reason="not reproduced"))
    assert row.ok is False


def test_stock_microbenchmark_reproduces_all_private_contracts():
    source = resolve_fedot_src()
    result = run_stock_microbenchmark(source)

    assert result.ok is True
    assert result.source_hash
    assert {case.case_id for case in result.cases} == {
        "partial_poly_params",
        "lda_effective_solver",
        "lagged_reproducibility",
        "polyfit_parameter_identity",
        "nonfinite_target_preprocessing",
        "merge_parent_index_alignment",
        "single_column_multits_lagged",
    }
    assert all(case.stages[0].stage == "stock_probe" for case in result.cases)
    assert all(case.stages[0].passed for case in result.cases)
    assert result.as_dict()["component"] == "micro-fast"


def test_stock_microbenchmark_does_not_write_fedot(tmp_path: Path):
    source = resolve_fedot_src()
    before = {
        path.relative_to(source): path.stat().st_mtime_ns
        for path in source.joinpath("fedot").rglob("*.py")
    }

    run_stock_microbenchmark(source)

    after = {
        path.relative_to(source): path.stat().st_mtime_ns
        for path in source.joinpath("fedot").rglob("*.py")
    }
    assert after == before
