from __future__ import annotations

from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src


def test_context_card_maps_implementation_to_public_operation_and_params():
    source = resolve_fedot_src()
    card = build_fedot_context(
        source,
        (
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/sklearn_transformations.py"
        ),
        symbol="PolyFeaturesImplementation.__init__",
    )

    assert card is not None
    assert card.symbol == "PolyFeaturesImplementation.__init__"
    assert card.bases[:2] == (
        "PolyFeaturesImplementation",
        "EncodedInvariantImplementation",
    )
    operation = next(item for item in card.operations if item.operation_id == "poly_features")
    assert operation.implementation == "PolyFeaturesImplementation"
    assert {"degree", "interaction_only"} <= set(operation.declared_params)
    assert "pipeline data transformation implementation" in card.lifecycle
    assert any(item.kind == "operation_dispatch" for item in card.callers)
    assert any("test_data_operations" in item.file_path for item in card.tests)


def test_context_card_follows_base_to_related_public_operations():
    source = resolve_fedot_src()
    card = build_fedot_context(
        source,
        (
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/ts_transformations.py"
        ),
        symbol="LaggedImplementation._check_and_correct_window_size",
    )

    assert card is not None
    operations = {item.operation_id: dict(item.defaults) for item in card.operations}
    assert operations["lagged"] == {"window_size": 0}
    assert operations["sparse_lagged"]["window_size"] == 0
    assert card.bases[0] == "LaggedImplementation"
    assert "fit stage" in card.lifecycle
    assert "transform stage" in card.lifecycle


def test_context_card_is_deterministic_compact_and_source_grounded():
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


def test_file_level_card_uses_outline_and_missing_symbol_falls_back():
    source = resolve_fedot_src()
    path = (
        "fedot/core/operations/evaluation/operation_implementations/"
        "models/ts_implementations/poly.py"
    )
    card = build_fedot_context(source, path, symbol="does_not_exist")

    assert card is not None
    assert card.kind == "file"
    assert card.symbol == ""
    assert "class PolyfitImplementation" in card.source


def test_context_card_rejects_paths_outside_fedot():
    source = resolve_fedot_src()

    assert build_fedot_context(source, "../pyproject.toml") is None
    assert build_fedot_context(source, "test/unit/pipelines/test_pipeline.py") is None
