"""Minimal child process for trusted EvolveAgent snippets."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


from fedotllm.agents.evolve.execution.snippet_policy import snippet_runtime_guards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--target-symbol")
    parser.add_argument("--target-result", type=Path)
    args = parser.parse_args(argv)
    checkout = args.checkout.resolve()

    import fedot

    loaded = Path(fedot.__file__).resolve()
    if checkout != loaded and checkout not in loaded.parents:
        raise RuntimeError(f"fedot imported outside experiment: {loaded}")
    reached = False
    target_file = str(args.target_file.resolve()) if args.target_file else None

    def trace(frame, event, arg):
        nonlocal reached
        if event == "call" and frame.f_code.co_filename == target_file:
            qualname = getattr(frame.f_code, "co_qualname", None)
            if qualname is None:
                # Python < 3.11 has no co_qualname; fall back to the bare name so a
                # ``Class.method`` target still matches instead of silently never firing.
                qualname = args.target_symbol if frame.f_code.co_name == args.target_symbol.rsplit(".", 1)[-1] else frame.f_code.co_name
            symbol = qualname.replace(".<locals>", "")
            if symbol == args.target_symbol or symbol.startswith(args.target_symbol + "."):
                reached = True
                sys.settrace(None)
        return None

    try:
        if target_file and args.target_symbol:
            sys.settrace(trace)
        with snippet_runtime_guards(checkout):
            runpy.run_path(str(args.script), run_name="__main__")
    finally:
        sys.settrace(None)
        if args.target_result:
            args.target_result.write_text(json.dumps({"target_reached": reached}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
