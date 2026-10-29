from __future__ import annotations

from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src


def test_source_context_maps_implementation_to_public_operation_and_params():
    source = resolve_fedot_src()
    source_context = build_fedot_context(
        source,
        (
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/sklearn_transformations.py"
        ),
        symbol="PolyFeaturesImplementation.__init__",
    )

    assert source_context is not None
    assert source_context.symbol == "PolyFeaturesImplementation.__init__"
    assert source_context.bases[:2] == (
        "PolyFeaturesImplementation",
        "EncodedInvariantImplementation",
    )
    operation = next(item for item in source_context.operations if item.operation_id == "poly_features")
    assert operation.implementation == "PolyFeaturesImplementation"
    assert {"degree", "interaction_only"} <= set(operation.declared_params)
    assert "pipeline data transformation implementation" in source_context.lifecycle
    assert any(item.kind == "operation_dispatch" for item in source_context.callers)
    assert any("test_data_operations" in item.file_path for item in source_context.tests)


def test_source_context_follows_base_to_related_public_operations():
    source = resolve_fedot_src()
    source_context = build_fedot_context(
        source,
        (
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/ts_transformations.py"
        ),
        symbol="LaggedImplementation._check_and_correct_window_size",
    )

    assert source_context is not None
    operations = {item.operation_id: dict(item.defaults) for item in source_context.operations}
    assert operations["lagged"] == {"window_size": 0}
    assert operations["sparse_lagged"]["window_size"] == 0
    assert source_context.bases[0] == "LaggedImplementation"
    assert "fit stage" in source_context.lifecycle
    assert "transform stage" in source_context.lifecycle


def test_source_context_is_deterministic_compact_and_source_grounded():
    source = resolve_fedot_src()
    path = (
        "fedot/core/operations/evaluation/operation_implementations/"
        "models/discriminant_analysis.py"
    )
    first = build_fedot_context(source, path, symbol="LDAImplementation.check_and_correct_params")
    second = build_fedot_context(source, path, symbol="LDAImplementation.check_and_correct_params")

    assert first == second
    assert first is not None
    rendered = first.render(max_chars=2_000)
    assert len(rendered) <= 2_000
    assert "LDAImplementation.check_and_correct_params" in rendered
    assert "shrinkage" in rendered
    assert "oracle" not in rendered.lower()
    assert "benchmark" not in rendered.lower()


def test_file_level_context_uses_outline_and_missing_symbol_falls_back():
    source = resolve_fedot_src()
    path = (
        "fedot/core/operations/evaluation/operation_implementations/"
        "models/ts_implementations/poly.py"
    )
    source_context = build_fedot_context(source, path, symbol="does_not_exist")

    assert source_context is not None
    assert source_context.kind == "file"
    assert source_context.symbol == ""
    assert "class PolyfitImplementation" in source_context.source


def test_source_context_rejects_paths_outside_fedot():
    source = resolve_fedot_src()

    assert build_fedot_context(source, "../pyproject.toml") is None
    assert build_fedot_context(source, "test/unit/pipelines/test_pipeline.py") is None
