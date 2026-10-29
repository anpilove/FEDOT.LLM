"""Install a checksum-pinned controller release; default mode only checks.

Standalone stdlib script for reviewed SSH delivery. Never launches benchmarks,
reads processes, changes FEDOT, removes data, or overwrites an existing module.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True


def _write(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "install"), default="check")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--archive-inputs", action="store_true")
    args = parser.parse_args(argv)
    if not args.root.is_absolute() or args.root == Path("/"):
        parser.error("root must be an explicit absolute project directory")
    if len(args.controller_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.controller_sha256
    ):
        parser.error("invalid controller SHA256")
    root = args.root
    evolve = root / "evolve"
    current = evolve / "controller"
    releases = evolve / "releases"
    runs = evolve / "runs"
    release = releases / args.controller_sha256
    code = release / "controller"
    for path in (root, evolve, releases, runs):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            parser.error(f"unexpected installation path: {path}")
    if os.path.lexists(current) or os.path.lexists(release):
        parser.error("controller or release already exists; refusing to overwrite")
    raw = args.archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.archive_sha256:
        parser.error("archive hash mismatch")
    files = {}
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in files
                or member.name != path.as_posix()
            ):
                parser.error(f"unsafe archive member: {member.name}")
            if not (
                member.name.startswith("fedotllm/")
                or member.name in {"README.md", "release.json"}
            ):
                parser.error(f"unexpected archive member: {member.name}")
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files["release.json"])
    expected_files = manifest.get("files") or {}
    if set(expected_files) != set(files) - {"release.json"}:
        parser.error("release file manifest mismatch")
    for name, data in files.items():
        if (
            name != "release.json"
            and hashlib.sha256(data).hexdigest() != expected_files[name]
        ):
            parser.error(f"release file hash mismatch: {name}")
    digest = hashlib.sha256()
    for name in sorted(files):
        if name.startswith("fedotllm/") and Path(name).suffix in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
        }:
            digest.update(name.removeprefix("fedotllm/").encode())
            digest.update(files[name])
    if (
        digest.hexdigest() != args.controller_sha256
        or manifest["controller_sha256"] != args.controller_sha256
    ):
        parser.error("controller hash mismatch")
    if args.archive_inputs:
        for source in (args.archive, Path(__file__)):
            if (
                source.parent.resolve() != Path.home().resolve()
                or not source.name.startswith("_evolve-controller-")
            ):
                parser.error(
                    "archive-inputs accepts only the named installer staging files in the user's home"
                )
    audit = {
        "name": "FEDOT.LLM Evolve Controller",
        "time": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "arguments": vars(args) | {"root": str(root), "archive": str(args.archive)},
        "installer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controller_sha256": args.controller_sha256,
        "archive_sha256": args.archive_sha256,
        "affected_paths": [str(current), str(release), str(runs)],
        "result": "checked",
        "benchmark_started": False,
    }
    if args.mode == "install":
        release.mkdir(parents=True, exist_ok=False)
        audit_path = release / "installation.json"
        audit["result"] = "installing"
        _write(audit_path, audit)
        try:
            code.mkdir()
            for name, data in files.items():
                target = code / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            runs.mkdir(exist_ok=True)
            shutil.copyfile(args.archive, release / "controller.tar.gz")
            # The module path is stable; each child resolves its immutable release.
            current.symlink_to(code.relative_to(evolve), target_is_directory=True)
            audit["result"] = "installed"
            if args.archive_inputs:
                args.archive.replace(release / "controller.tar.gz")
                Path(__file__).replace(release / "installer.py")
                audit["archived_inputs"] = [
                    str(release / "controller.tar.gz"),
                    str(release / "installer.py"),
                ]
            _write(audit_path, audit)
        except BaseException as exc:
            audit.update(
                result="installation_failed", error=f"{type(exc).__name__}: {exc}"
            )
            try:
                _write(audit_path, audit)
            except OSError as persist_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"installation audit failed: {persist_error}")
            raise
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
