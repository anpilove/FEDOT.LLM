"""What a proposed property is allowed to lean on.

The agent's job is to say what MUST be true; recalling how FEDOT builds a
pipeline is not part of that and it is not good at it. Measured: of 16 freely
written checks, 13 hardcoded the operation and the rest invented imports that do
not exist (`from fedot.core.operations.model import RANSACLinReg`,
`OperationFactory.create_operation`). These helpers remove API recall from the
task and leave the hypothesis.

Imported both by the test suite and, flat, by the subprocess that runs a
property inside the FEDOT checkout — hence the double import below.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (re-exported to the property)

try:  # inside FEDOT.LLM
    from fedotllm.agents.evolve import invariants as _I
except ImportError:  # inside the FEDOT checkout, run flat by the property runner
    import invariants as _I


class Skipped(Exception):
    """The property does not apply here — not evidence either way."""

def make_data(kind="regression"):
    """Small legal datasets: classification, regression, regression_outliers, ts."""
    return _I.make_data(kind)


def fit_operation(operation, params=None, kind="auto"):
    """Fit `operation` inside a working pipeline and return (pipeline, node, data).

    Picks a pipeline and a dataset the operation actually supports, and raises
    Skipped when there is none -- an operation that cannot be constructed is not
    evidence for or against any property.
    """
    kind_, meta = _I._meta(operation)
    if meta is None:
        raise Skipped("unknown operation " + str(operation))
    tasks = [t.value for t in meta.task_type]
    kinds = [kind] if kind != "auto" else None
    for task in tasks:
        for dk in (kinds or _I.DATA_FOR_TASK.get(task, [])):
            data = _I.make_data(dk)
            for chain in _I.candidate_chains(operation, task):
                try:
                    pipeline = _I.build_and_fit(chain, params, data)
                except Exception:
                    continue
                node = next((n for n in pipeline.nodes if n.name == operation), None)
                if node is not None:
                    return pipeline, node, data
    raise Skipped("no working pipeline for " + str(operation)
                  + (" with params " + repr(params) if params else ""))


_UNSET = object()


def in_force(node, name, default=None):
    """The value of `name` actually in force on the fitted object.

    Returns `default` when the parameter appears nowhere. Raises AssertionError
    if the fitted object disagrees with itself about the value -- that is a
    defect in its own right and must not be hidden by picking one of them.

    A single value, not a mapping, and that is the whole point. The previous
    version returned {where: value}; the agent reached for `.get(name)`, got
    None every time, compared None to the declared value and reported a defect
    that does not exist. It was the only property to survive screening in that
    run, and it was wrong -- an API that invites one specific mistake will
    collect it.
    """
    seen = _I.observed_values(node, name)
    if not seen:
        return default
    values = list(seen.values())
    first = values[0]
    for other in values[1:]:
        assert _I._equal(first, other), (
            "the fitted object reports %r in different places: %r" % (name, seen))
    return first


def observed_everywhere(node, name):
    """{where: value} for the rare check that cares about the locations."""
    return _I.observed_values(node, name)


def _space(operation):
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace

    return PipelineSearchSpace().parameters_per_operation.get(operation, {})


def declared_parameters(operation):
    """{name: [values worth trying]} — sample points, NOT the whole legal range.

    Both ends of the declared interval plus the middle, or every categorical
    choice. Use these as values to pass to `fit_operation`; use `is_legal` to
    ask whether some other value is allowed.

    The distinction is not pedantry. This function used to be documented as
    "{name: [legal values]}", and two separate properties in one run did
    `value in declared_parameters(op)[name]` and reported a defect because
    `min_samples=0.4` is not one of the three sampled points — while being
    entirely legal inside [0.1, 0.9]. Both survived control screening, because
    a harness that misleads misleads every operation equally.
    """
    return {name: _I.candidate_values(spec) for name, spec in _space(operation).items()}


def is_legal(operation, name, value):
    """Does the library's own declared scope allow `value` for `name`?

    Interval membership for numbers, set membership for categories. Returns
    False when the parameter is not declared tunable at all — ask
    `declared_parameters` first if you need to tell those cases apart.
    """
    spec = _space(operation).get(name)
    if not spec:
        return False
    scope = spec.get("sampling-scope")
    if spec.get("type") == "categorical":
        choices = scope[0] if scope and isinstance(scope[0], (list, tuple)) else scope
        return value in (choices or [])
    if not scope or len(scope) < 2:
        return False
    try:
        return float(scope[0]) <= float(value) <= float(scope[1])
    except (TypeError, ValueError):
        return False
