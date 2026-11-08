from pathlib import Path

from fedotllm.agents.evolve.benchmark.micro import run_stock_microbenchmark
from fedotllm.agents.evolve.execution.runtime_contract import (
    checkout_runtime_contract,
    measure_smoke_aa,
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "FEDOT"
    package = source / "fedot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""stub pin."""\n', encoding="utf-8")
    return source


def test_smoke_aa_repeat_is_measurable_and_not_a_quality_claim(tmp_path: Path):
    source = _source(tmp_path)
    report = measure_smoke_aa(source, tmp_path / "workspace")

    assert report["quality_claim"] is False
    assert report["identical"] is True
    assert report["abs_delta"] == 0
    assert report["values"][0] == report["values"][1] > 0
    assert all(row["contract"]["ok"] for row in report["repeats"])
    assert not (tmp_path / "workspace" / "experiments").exists() or not any(
        (tmp_path / "workspace" / "experiments").rglob(".evolve-agent-checkout.json")
    )


def test_checkout_contract_rejects_source_and_shadowed_import(tmp_path: Path):
    source = _source(tmp_path)
    rejected = checkout_runtime_contract(
        source=source,
        checkout=source,
        workspace=tmp_path / "workspace",
        fedot_file="/tmp/site-packages/fedot/__init__.py",
    )

    assert rejected["ok"] is False
    assert "checkout_not_disjoint" in rejected["issues"]
    assert "checkout_marker_missing" in rejected["issues"]
    assert "fedot_file_outside_checkout" in rejected["issues"]


def test_microbenchmark_workspace_uses_a_disjoint_checkout(tmp_path: Path):
    source = _source(tmp_path)
    before = (source / "fedot" / "__init__.py").read_bytes()

    result = run_stock_microbenchmark(source, tmp_path / "micro")

    assert result.ok is False
    assert result.checkout_contract["ok"] is True
    assert (source / "fedot" / "__init__.py").read_bytes() == before
    assert not (tmp_path / "micro" / "experiments" / "microbench" / "stock").exists()
    assert not any((tmp_path / "micro").rglob(".evolve-agent-checkout.json"))
