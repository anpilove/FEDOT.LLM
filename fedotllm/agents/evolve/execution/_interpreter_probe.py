"""Print a stdlib-only identity of the selected Python and installed ML packages."""

from importlib import metadata
import json
import platform
import sys


def main() -> None:
    packages = {}
    for name in ("fedot", "numpy", "pandas", "scikit-learn", "statsmodels", "golem",
                 "thegolem", "catboost", "xgboost", "lightgbm", "giotto-ph"):
        try:
            # FEDOT's `golem` import belongs to the `thegolem` distribution.
            packages[name] = metadata.version("thegolem" if name == "golem" else name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps({
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": packages,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
