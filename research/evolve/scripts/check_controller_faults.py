"""Fixed fault acceptance entry point; --mode run explicitly creates artifacts."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if __name__ == "__main__":
    from fedotllm.agents.evolve.controller.fault_checks import main

    raise SystemExit(main())
