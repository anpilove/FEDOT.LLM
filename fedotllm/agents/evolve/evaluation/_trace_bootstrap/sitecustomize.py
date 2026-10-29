"""Install opt-in compute-child stacks on standard Python interpreter startup."""

import os
import sys


def _run_original_sitecustomize():
    """Preserve the startup hook that this opt-in PYTHONPATH entry shadows.

    Execute it under its real module name/file so self-imports and package/path
    policy behave normally. Its exceptions intentionally propagate to Python's
    standard site.execsitecustomize handler; only our diagnostics are fail-open.
    """
    from importlib.machinery import PathFinder
    from importlib.util import module_from_spec

    guard_name = "_evolve_trace_sitecustomize_forwarding"
    if getattr(sys, guard_name, False):
        return
    bootstrap_directory = os.path.realpath(os.path.dirname(__file__))
    search_path = [
        entry
        for entry in sys.path
        if not isinstance(entry, str)
        or os.path.realpath(entry or os.curdir) != bootstrap_directory
    ]
    spec = PathFinder.find_spec("sitecustomize", search_path)
    if spec is None or spec.loader is None:
        return
    previous_guard = getattr(sys, guard_name, None)
    setattr(sys, guard_name, True)
    try:
        original = module_from_spec(spec)
        sys.modules["sitecustomize"] = original
        spec.loader.exec_module(original)
    finally:
        if previous_guard is None:
            delattr(sys, guard_name)
        else:
            setattr(sys, guard_name, previous_guard)


_run_original_sitecustomize()


if os.environ.get("EVOLVE_TRACE_DIRECTORY"):
    try:
        from fedotllm.agents.evolve.evaluation.child_traces import (
            install_child_trace_from_environment,
        )

        install_child_trace_from_environment()
    except Exception as exc:
        try:
            print(
                f"child stack diagnostic bootstrap warning: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
