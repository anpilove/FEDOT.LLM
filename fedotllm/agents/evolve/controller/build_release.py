"""Build the deployable controller module, without FEDOT, datasets or credentials."""

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.controller.release import controller_version
from fedotllm.agents.evolve.execution.guard import repo_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = repo_root()
    version = controller_version()
    destination = args.output.resolve() / version
    destination.mkdir(parents=True, exist_ok=False)
    files = {
        path.relative_to(root).as_posix(): path
        for path in (root / "fedotllm").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".yaml", ".yml", ".md"}
        and "__pycache__" not in path.parts
        and "cache" not in path.relative_to(root / "fedotllm").parts
    }
    files["README.md"] = Path(__file__).with_name("README.md")
    manifest = {
        "name": "FEDOT.LLM Evolve Controller",
        "module": "fedotllm.agents.evolve.controller",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "controller_sha256": version,
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        },
        "entry_point": "python -m fedotllm.agents.evolve.controller",
    }
    manifest_path = destination / "release.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive = destination / "controller.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
        tar.add(manifest_path, arcname="release.json", recursive=False)
    result = {
        "controller_sha256": version,
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "installer": str(Path(__file__).with_name("install.py")),
        "installer_sha256": hashlib.sha256(
            Path(__file__).with_name("install.py").read_bytes()
        ).hexdigest(),
    }
    (destination / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
