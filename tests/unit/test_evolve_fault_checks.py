import json
import subprocess
import sys

from fedotllm.agents.evolve.controller.release import controller_version
from fedotllm.agents.evolve.execution.guard import repo_root


def command(tmp_path):
    source = tmp_path / 'FEDOT'
    (source / 'fedot').mkdir(parents=True)
    (source / 'fedot/version.py').write_text('VERSION = 1\n')
    return [sys.executable, str(repo_root() / 'research/evolve/scripts/check_controller_faults.py'),
            '--fedot-source', str(source), '--workspace', str(tmp_path / 'fault-check'),
            '--expected-controller-sha256', controller_version(repo_root())]


def test_fault_check_defaults_to_read_only(tmp_path):
    result = subprocess.run(command(tmp_path), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit['mode'] == 'check'
    assert '-p' in audit['command'] and 'no:cacheprovider' in audit['command']
    assert not (tmp_path / 'fault-check').exists()


def test_fault_check_rejects_wrong_code_before_writes(tmp_path):
    args = command(tmp_path)
    args[-1] = '0' * 64
    result = subprocess.run(args + ['--mode', 'run'], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'hash mismatch' in result.stderr
    assert not (tmp_path / 'fault-check').exists()


def test_fault_check_records_startup_failure_and_keeps_original_error(tmp_path, monkeypatch):
    import pytest
    from fedotllm.agents.evolve.controller import fault_checks
    from fedotllm.agents.evolve.execution import process
    args = command(tmp_path)[2:]
    def denied(*_args, **_kwargs):
        raise OSError('fixture worker startup denied')
    monkeypatch.setattr(process, 'run_worker', denied)
    with pytest.raises(OSError, match='fixture worker startup denied'):
        fault_checks.main(args + ['--mode', 'run'])
    audit = json.loads((tmp_path / 'fault-check/acceptance.json').read_text())
    assert audit['result'] == 'failed'
    assert audit['error'] == 'OSError: fixture worker startup denied'
    assert audit['source_unchanged']


def test_fault_suite_runs_with_logs_outside_immutable_release(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import fault_checks
    from fedotllm.agents.evolve.execution import process
    args = command(tmp_path)[2:]
    calls = []
    def completed(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return process.WorkerOutcome(0, 'fixed suite completed', '')
    monkeypatch.setattr(process, 'run_worker', completed)
    assert fault_checks.main(args + ['--mode', 'run']) == 0
    cmd, options = calls[0]
    assert options['cwd'] == tmp_path / 'fault-check'
    assert all(__import__('pathlib').Path(item).is_absolute() for item in cmd[-len(fault_checks.SUITE):])
