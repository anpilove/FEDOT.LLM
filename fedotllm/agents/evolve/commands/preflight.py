"""Pin and dependency checks before a quality run.

The quality registry calls ``Fedot`` with ``timeout_policy="return_best"``.
Stock FEDOT 0.7.5 rejects that keyword inside ``ApiParams``. Doctor has to
report that mismatch, a missing ``dask`` import, and a shadowed ``fedot``
path before any hour-long fit starts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

PREFLIGHT_MARKER = "EVOLVE_PREFLIGHT "
WORKER_PARAMETERS = (
    "timeout_policy",
    "final_fit_reserve_seconds",
    "cv_folds",
    "max_pipeline_fit_time",
    "max_pipeline_evaluation_time",
)
PREFLIGHT_SNIPPET = """
import importlib.util
import inspect
import json
from pathlib import Path
payload = {
    "dask": importlib.util.find_spec("dask") is not None,
}
try:
    import fedot
    payload["fedot_file"] = str(Path(fedot.__file__).resolve())
    from fedot.api.main import Fedot
    payload["parameters"] = sorted(inspect.signature(Fedot.__init__).parameters)
except Exception as exc:
    payload["import_error"] = f"{type(exc).__name__}: {exc}"
print("EVOLVE_PREFLIGHT " + json.dumps(payload, sort_keys=True))
""".strip()


def analyze_checkout(checkout: Path) -> dict:
    """Read the checkout source. Do not import it into the controller."""

    checkout = checkout.resolve()
    package = checkout / "fedot"
    api = package / "api" / "main.py"
    parameters = _init_parameters(api) if api.is_file() else None
    missing = [
        name
        for name in WORKER_PARAMETERS
        if parameters is None or name not in parameters
    ]
    implements = (
        parameters is not None and not missing and _literal_used(package, "return_best")
    )
    return {
        "package": package.is_dir() and (package / "__init__.py").is_file(),
        "parameters": sorted(parameters) if parameters else [],
        "implements_return_best": implements,
        "missing_parameters": missing,
        "imports_dask": _imports_module(package, "dask") if package.is_dir() else False,
    }


def parse_preflight_stdout(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(PREFLIGHT_MARKER):
            continue
        try:
            payload = json.loads(line[len(PREFLIGHT_MARKER) :])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def dependency_checks(
    checkout: Path,
    runtime: dict | None,
    *,
    probe_error: str = "",
) -> dict:
    """Combine source analysis with the interpreter that will run the worker."""

    static = analyze_checkout(checkout)
    checkout = checkout.resolve()
    fedot_file = ""
    parameters: list[str] = []
    dask_importable = False
    import_error = probe_error
    if runtime:
        fedot_file = str(runtime.get("fedot_file") or "")
        parameters = [str(item) for item in runtime.get("parameters") or []]
        dask_importable = bool(runtime.get("dask"))
        if runtime.get("import_error") and not import_error:
            import_error = str(runtime["import_error"])

    path_ok = False
    path_detail = import_error or "preflight probe did not report fedot.__file__"
    if fedot_file:
        resolved = Path(fedot_file).resolve()
        package = (checkout / "fedot").resolve()
        path_ok = resolved.is_file() and (
            resolved == (package / "__init__.py") or package in resolved.parents
        )
        path_detail = (
            fedot_file if path_ok else f"import resolved outside checkout: {fedot_file}"
        )

    loaded_policy = path_ok and "timeout_policy" in parameters
    return_best_ok = bool(static["implements_return_best"]) and loaded_policy
    if not static["implements_return_best"]:
        missing = (
            ", ".join(static["missing_parameters"]) or "timeout_policy/return_best"
        )
        return_best_detail = f"pinned FEDOT does not implement return_best ({missing})"
    elif not path_ok:
        return_best_detail = "loaded FEDOT is not the pinned checkout"
    elif "timeout_policy" not in parameters:
        return_best_detail = "loaded Fedot.__init__ does not accept timeout_policy"
    else:
        return_best_detail = (
            "checkout implements return_best and the loaded API accepts it"
        )

    if static["imports_dask"] and dask_importable:
        dask_detail = "checkout imports dask and the worker interpreter can import it"
    elif static["imports_dask"] and runtime is None and import_error:
        dask_detail = (
            "preflight probe failed before dask was measured: " + import_error
        )
    elif static["imports_dask"]:
        dask_detail = (
            "checkout imports dask but the worker interpreter cannot import it"
        )
    else:
        dask_detail = "checkout does not import dask"
    return {
        "return_best": {
            "ok": return_best_ok,
            "detail": return_best_detail,
            "missing_parameters": list(static["missing_parameters"]),
        },
        "dask": {
            "ok": (not static["imports_dask"]) or dask_importable,
            "required": bool(static["imports_dask"]),
            "importable": dask_importable,
            "detail": dask_detail,
        },
        "path": {
            "ok": path_ok,
            "fedot_file": fedot_file,
            "checkout": str(checkout),
            "detail": path_detail,
        },
    }


def _init_parameters(api_main: Path) -> set[str] | None:
    try:
        tree = ast.parse(api_main.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Fedot":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                args = item.args
                names = {
                    arg.arg for arg in (*args.args, *args.kwonlyargs, *args.posonlyargs)
                }
                return names
    return None


def _literal_used(package: Path, literal: str) -> bool:
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr):
                continue
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Constant) and child.value == literal:
                    return True
    return False


def _imports_module(package: Path, module: str) -> bool:
    prefix = f"{module}."
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == module or name.startswith(prefix) for name in names):
                return True
    return False
