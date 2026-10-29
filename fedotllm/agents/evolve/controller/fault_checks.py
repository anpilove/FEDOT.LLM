"""Run a fixed controller fault suite in a new workspace; default reads only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

SUITE = (
    'tests/unit/agents/test_evolve_worker_lifecycle.py',
    'tests/unit/agents/test_evolve_quality_cancellation.py',
    'tests/unit/agents/test_evolve_quality_reliability.py',
    'tests/unit/agents/test_evolve_quality_worker_diagnostics.py',
    'tests/unit/agents/test_evolve_controller_smoke.py',
    'tests/unit/agents/test_evolve_storage_guard.py',
    'tests/unit/agents/test_evolve_supervisor.py',
    'tests/unit/agents/test_evolve_quality_cache_policy.py',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run'), default='check')
    parser.add_argument('--fedot-source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--expected-controller-sha256', required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[4]
    source, workspace = args.fedot_source.resolve(), args.workspace.resolve()
    if not args.fedot_source.is_absolute() or not args.workspace.is_absolute():
        parser.error('source and workspace must be absolute')
    if not (source / 'fedot').is_dir():
        parser.error('FEDOT source is missing')
    if workspace.exists() or args.workspace.is_symlink() or not workspace.parent.is_dir():
        parser.error('workspace must be new with an existing parent')
    for protected in (root, source):
        if workspace == protected or protected in workspace.parents or workspace in protected.parents:
            parser.error('workspace must be separate from the library and FEDOT source')
    from fedotllm.agents.evolve.controller.release import controller_version
    from fedotllm.agents.evolve.execution.checkout import source_fingerprint
    from fedotllm.agents.evolve.execution.process import clean_subprocess_env, run_worker
    from fedotllm.agents.evolve.storage.journal import write_json_atomic
    version = controller_version(root)
    if version != args.expected_controller_sha256:
        parser.error('controller hash mismatch')
    if importlib.util.find_spec('pytest') is None or any(not (root / name).is_file() for name in SUITE):
        parser.error('fixed acceptance suite or pytest is unavailable; no packages installed')
    before = source_fingerprint(source)
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '--basetemp', str(workspace / 'fixtures'), *(str(root / name) for name in SUITE)]
    audit = {'purpose':'controller_fault_acceptance', 'mode':args.mode,
             'time':datetime.now(timezone.utc).isoformat(), 'controller_sha256':version,
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_sha256':before, 'command':command, 'affected_paths':[str(workspace)],
             'passed':False, 'result':'checked; no tests executed', 'watchdog_seconds':180}
    if args.mode == 'run':
        workspace.mkdir()
        audit['result'] = 'started'
        write_json_atomic(workspace / 'acceptance.json', audit)
        env = clean_subprocess_env(source, repo_root=root, extra={
            'PYTHONUNBUFFERED':'1', 'EVOLVE_AGENT_TMP':str(workspace / 'worker_artifacts')})
        try:
            outcome = run_worker(command, cwd=workspace, env=env, timeout=180, log_path=workspace / 'tests.log')
        except BaseException as exc:
            audit.update(result='failed', error=f'{type(exc).__name__}: {exc}',
                         finished_at=datetime.now(timezone.utc).isoformat())
            try:
                audit['source_unchanged'] = source_fingerprint(source) == before
                write_json_atomic(workspace / 'acceptance.json', audit)
            except BaseException as audit_error:
                if hasattr(exc, 'add_note'):
                    exc.add_note(f'failure audit could not be saved: {audit_error}')
            raise
        unchanged = source_fingerprint(source) == before
        audit.update(returncode=outcome.returncode, timed_out=outcome.timed_out,
                     source_unchanged=unchanged,
                     passed=outcome.returncode == 0 and not outcome.timed_out and unchanged,
                     result='finished', finished_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(workspace / 'acceptance.json', audit)
        (workspace / 'tests.rc').write_text(str(outcome.returncode) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if args.mode == 'check' or audit['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
