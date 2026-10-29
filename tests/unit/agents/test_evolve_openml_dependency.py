"""A normal installation must include the quality loader's required OpenML API."""

from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, also supported by pytest's dependency.
    import tomli as tomllib


def test_openml_is_a_required_locked_dependency_for_quality_loading():
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    openml = next((item for item in requirements if item.name == "openml"), None)
    assert openml is not None, (
        "quality loading requires OpenML in a normal installation"
    )
    assert openml.marker is None, "OpenML must be installed on every supported platform"

    lock = tomllib.loads((root / "uv.lock").read_text())
    package = next(item for item in lock["package"] if item["name"] == "openml")
    assert package["version"] in openml.specifier
    library = next(item for item in lock["package"] if item["name"] == "fedotllm")
    assert {"name": "openml"} in library["dependencies"]
