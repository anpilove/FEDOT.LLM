"""Real fixture children cover stopping quality work after storage failure."""

import json
import sys
import threading
import time
from concurrent.futures import CancelledError
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedotllm.agents.evolve.evaluation import fedot_quality as quality
from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry
from fedotllm.agents.evolve.execution.process import run_worker as real_run_worker


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    (path / "fedot").mkdir(parents=True)
    (path / "fedot" / "__init__.py").write_text("VERSION = 1\n")
    return path


@pytest.fixture
def concurrent_workers(tmp_path, monkeypatch, source):
    registry = replace(load_quality_registry(), watchdog_seconds=1)
    ids = tuple(dataset.task_id for dataset in registry.datasets[:4])
    fixture = tmp_path / "quality_cancellation_fixture.py"
    fixture.write_text("""import json, sys, time
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text()); output = Path(sys.argv[2]); root = Path(sys.argv[3])
first, second = sys.argv[4:6]
task = spec["source_dataset"]
with (root / "launches.jsonl").open("a") as log:
    log.write(json.dumps({"task": task, "side": spec["side"]}) + "\\n")
if task == second:
    (root / "second_started").touch()
    while True:
        (root / "heartbeat").write_text(str(time.monotonic())); time.sleep(.03)
if task == first:
    deadline = time.monotonic() + 3
    while not (root / "second_started").exists() and time.monotonic() < deadline:
        time.sleep(.01)
output.write_text(json.dumps({"status": "ok", "score": .5, "seed": 42, "n_train": 10,
                             "detail": "", "traceback": "", "duration_s": .1,
                             "metric_observations": {"metric_valid": True, "search_ran": True}}))
""")
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(quality, "fedot_python", lambda _: sys.executable)

    def run_fixture(command, **kwargs):
        spec = command[command.index("--spec-json") + 1]
        output = command[command.index("--out") + 1]
        return real_run_worker(
            [sys.executable, str(fixture), spec, output, str(tmp_path), ids[0], ids[1]],
            **kwargs,
        )

    monkeypatch.setattr(quality, "run_worker", run_fixture)
    return registry, ids


@pytest.mark.parametrize("paired", [True, False])
def test_persistence_failure_cancels_running_workers_before_more_datasets_start(
    tmp_path, source, monkeypatch, concurrent_workers, paired
):
    # Waiting for every future after a storage error would leave the second
    # fixture computing for its 3h watchdog and launch queued work afterward.
    registry, ids = concurrent_workers
    original = OSError(28, "fixture result storage full")

    def fail_persistence(*args, **kwargs):
        raise original

    started = time.monotonic()
    with pytest.raises(OSError) as caught:
        if paired:

            def side_event(spec, side, event, result, attempt):
                if event == "finished" and spec["source_dataset"] == ids[0]:
                    raise original

            quality.run_quality_jobs(
                stock_checkout=source,
                patch_checkout=source,
                task_ids=ids,
                n_jobs=4,
                cpu_quota=8,
                registry=registry,
                on_side_event=side_event,
            )
        else:
            monkeypatch.setattr(quality, "_store_stock_cache", fail_persistence)
            quality.run_quality_stock_jobs(
                stock_checkout=source,
                task_ids=ids,
                n_jobs=4,
                cpu_quota=8,
                registry=registry,
            )
    assert caught.value is original
    assert time.monotonic() - started < 3
    launches = [
        json.loads(line)
        for line in (tmp_path / "launches.jsonl").read_text().splitlines()
    ]
    assert {entry["task"] for entry in launches} == set(ids[:2])
    assert all(entry["side"] == "stock" for entry in launches)
    heartbeat = (tmp_path / "heartbeat").read_text()
    time.sleep(0.15)
    assert (tmp_path / "heartbeat").read_text() == heartbeat


def test_shared_async_pair_error_drains_manually_cancelled_future_callbacks(
    source, monkeypatch
):
    registry = load_quality_registry()
    ids = tuple(dataset.task_id for dataset in registry.datasets[:2])
    scheduler = quality.SharedPairScheduler[str, str](max_pairs=2)
    pending: Future = Future()
    original = OSError("fixture scheduler failure")

    def async_pair(dataset, **_kwargs):
        future: Future = Future()
        if dataset == ids[0]:
            future.set_exception(original)
            return future
        return pending

    monkeypatch.setattr(quality, "_run_quality_pair_async", async_pair)
    started = time.monotonic()
    try:
        with pytest.raises(OSError, match="fixture scheduler failure") as caught:
            quality.run_quality_jobs(
                stock_checkout=source,
                patch_checkout=source,
                task_ids=ids,
                n_jobs=4,
                cpu_quota=8,
                registry=registry,
                pair_scheduler=scheduler,
            )
        assert caught.value is original
        assert pending.cancelled()
        assert scheduler.fatal_error is original
        assert time.monotonic() - started < 2
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "n_jobs,cpu_quota", [(0, 28), (-4, 28), (4, 0), (4, -1), (4, 3)]
)
def test_invalid_resource_profile_is_rejected_before_task_launch(
    monkeypatch, n_jobs, cpu_quota
):
    def must_not_launch(*args, **kwargs):
        pytest.fail("invalid resource profile launched quality work")

    monkeypatch.setattr(quality, "run_quality_pair", must_not_launch)
    with pytest.raises(ValueError):
        quality.run_quality_jobs(
            stock_checkout=Path("source"),
            patch_checkout=Path("patch"),
            task_ids=(load_quality_registry().datasets[0].task_id,),
            n_jobs=n_jobs,
            cpu_quota=cpu_quota,
        )


def test_dataset_parallelism_never_exceeds_seven_even_with_large_quota(monkeypatch):
    active = maximum = 0
    lock = threading.Lock()

    def fixture_pair(task_id, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"task_id": task_id}

    monkeypatch.setattr(quality, "run_quality_pair", fixture_pair)
    ids = tuple(dataset.task_id for dataset in load_quality_registry().datasets[:12])
    rows = quality.run_quality_jobs(
        stock_checkout=Path("source"),
        patch_checkout=Path("patch"),
        task_ids=ids,
        n_jobs=4,
        cpu_quota=1000,
    )
    assert maximum <= 7
    assert [row["task_id"] for row in rows] == list(ids)


def test_nonzero_structured_connection_failure_retains_diagnostics_and_one_retry(
    tmp_path, source, monkeypatch
):
    fixture = tmp_path / "quality_failure_fixture.py"
    fixture.write_text("""import json, sys
from pathlib import Path
output = Path(sys.argv[1]); root = Path(sys.argv[2])
spec = json.loads(Path(sys.argv[3]).read_text())
identity = {key: spec.get(key) for key in (
    "side", "attempt", "attempt_id", "registry_identity", "evaluation_identity", "worker_invocation_id")}
identity["task_id"] = spec["source_dataset"]
with (root / "attempts").open("a") as log: log.write("attempt\\n")
output.write_text(json.dumps({**identity, "status": "crash", "score": None, "detail": "provider connection failed",
                             "traceback": "ConnectionError: upstream unreachable", "failed_stage": "predict",
                             "failure_kind": "evaluation_error", "last_stage": "predict",
                             "stage_timings": {"predict": {"status": "failed"}},
                             "metric_observations": {}}))
raise SystemExit(2)
""")
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(quality, "fedot_python", lambda _: sys.executable)

    def run_fixture(command, **kwargs):
        output = command[command.index("--out") + 1]
        return real_run_worker(
            [
                sys.executable,
                str(fixture),
                output,
                str(tmp_path),
                command[command.index("--spec-json") + 1],
            ],
            **kwargs,
        )

    monkeypatch.setattr(quality, "run_worker", run_fixture)
    task_id = load_quality_registry().datasets[0].task_id
    row = quality.run_quality_pair(
        task_id, stock_checkout=source, patch_checkout=source
    )
    assert row["stock_result"].status == "crash"
    assert "ConnectionError" in row["stock_result"].traceback
    assert row["stock_result"].metric_observations["failed_stage"] == "predict"
    assert row["stock_result"].metric_observations["failure_kind"] == "evaluation_error"
    assert (
        row["stock_result"].metric_observations["stage_timings"]["predict"]["status"]
        == "failed"
    )
    assert row["stock_result"].metric_observations["worker_return_code"] == 2
    assert (
        row["stock_result"]
        .metric_observations["stage_status_path"]
        .endswith(".stages.json")
    )
    assert row["patched_result"].status == "crash"
    assert len((tmp_path / "attempts").read_text().splitlines()) == 4


def test_timeout_retains_stage_sidecar_diagnostics(tmp_path, source, monkeypatch):
    fixture = tmp_path / "quality_stage_timeout_fixture.py"
    fixture.write_text("""import json, subprocess, sys, time
from pathlib import Path
output = Path(sys.argv[1])
output.with_suffix(".stages.json").write_text(json.dumps({"last_stage": "fit", "event": "start", "updated_at": "2026-10-01T00:00:00Z",
    "stage_timings": {"fit": {"status": "running", "started_at": "2026-10-01T00:00:00Z"}}}))
child = output.parent / "quality_child.py"
child.write_text("import sys, time\\nfrom pathlib import Path\\np=Path(sys.argv[1])\\nwhile True:\\n p.open('a').write('.')\\n time.sleep(.02)\\n")
subprocess.Popen([sys.executable, str(child), str(output.parent / "heartbeat")])
print("STAGE fit started", flush=True)
time.sleep(30)
""")
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(quality, "fedot_python", lambda _: sys.executable)

    def run_fixture(command, **kwargs):
        output = command[command.index("--out") + 1]
        kwargs.update(timeout=0.2, watchdog_timeout=1.0)
        return real_run_worker([sys.executable, str(fixture), output], **kwargs)

    monkeypatch.setattr(quality, "run_worker", run_fixture)
    registry = replace(load_quality_registry(), timeout_seconds=5, watchdog_seconds=10)
    result = quality.run_quality_side(
        registry.datasets[0], checkout=source, side="stock", registry=registry
    )
    assert result.status == "timeout"
    assert result.metric_observations["timeout_kind"] == "task_timeout"
    assert result.metric_observations["timeout_seconds"] == 5
    assert result.metric_observations["watchdog_seconds"] == 10
    assert result.metric_observations["last_stage"] == "fit"
    assert result.metric_observations["stage_timings"]["fit"]["status"] == "running"
    sidecar = Path(result.metric_observations["stage_status_path"])
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["last_stage"] == "fit"
    worker_log = Path(result.metric_observations["worker_log_path"])
    assert worker_log.exists()
    assert "STAGE fit started" in worker_log.read_text()
    heartbeat = (
        Path(result.metric_observations["worker_result_path"]).parent / "heartbeat"
    )
    previous = heartbeat.stat().st_size
    time.sleep(0.1)
    assert heartbeat.stat().st_size == previous


@pytest.mark.parametrize(
    ("timeout_kind", "expected_duration"),
    [("task_timeout", 17), ("watchdog_timeout", 43)],
)
def test_quality_side_uses_separate_limits_and_persists_timeout_source(
    tmp_path, source, monkeypatch, timeout_kind, expected_duration
):
    registry = replace(load_quality_registry(), timeout_seconds=17, watchdog_seconds=43)
    dataset = registry.datasets[0]
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "artifacts"))
    monkeypatch.setattr(quality, "_fedot_commit", lambda _checkout: "fixture-commit")
    monkeypatch.setattr(quality, "_source_sha256", lambda _checkout: "fixture-source")
    monkeypatch.setattr(quality, "fedot_python", lambda _checkout: sys.executable)
    monkeypatch.setattr(
        quality, "interpreter_identity", lambda *_args, **_kwargs: "fixture-runtime"
    )
    monkeypatch.setattr(
        quality, "clean_subprocess_env", lambda *_args, **_kwargs: {"PATH": ""}
    )
    seen = {}

    def timed_out_worker(_cmd, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            returncode=None,
            stdout="fit started",
            stderr="",
            timed_out=True,
            timeout_kind=timeout_kind,
            cancelled=False,
        )

    monkeypatch.setattr(quality, "run_worker", timed_out_worker)
    result = quality.run_quality_side(
        dataset,
        checkout=source,
        side="patch",
        registry=registry,
    )
    assert 0 < seen["timeout"] <= 17
    assert seen["watchdog_timeout"] == 43
    assert result.status == "timeout"
    assert result.metric_observations["timeout_kind"] == timeout_kind
    assert 0 <= result.duration_s < expected_duration
    assert result.metric_observations["elapsed_seconds"] == result.duration_s
    assert result.metric_observations["timeout_limit_seconds"] == expected_duration
    assert result.metric_observations["timeout_seconds"] == 17
    assert result.metric_observations["watchdog_limit_seconds"] == 43
    assert result.metric_observations["watchdog_seconds"] == 43


def test_timeout_snapshot_copies_active_fit_and_last_optimizer_evidence(tmp_path):
    status_path = tmp_path / "quality-out-fit.stages.json"
    status_path.write_text(
        json.dumps(
            {
                "event": "heartbeat",
                "updated_at": "2026-10-06T00:00:00+00:00",
                "proposal_id": "revision-4",
                "attempt_id": "revision-4:traffic:stock:1",
                "task_id": "traffic",
                "side": "stock",
                "attempt": 1,
                "active_stage": "fit",
                "stage_started_at": "2026-10-05T23:00:00+00:00",
                "stage_elapsed_seconds": 3600,
                "last_optimizer_event_at": "2026-10-05T23:10:00+00:00",
                "last_fedot_log_line": "EvoGraphOptimizer - Generation num: 1 size: 4",
            }
        )
    )

    snapshot = quality._stage_status_observations(status_path)

    assert snapshot["stage_status_path"] == str(status_path)
    assert snapshot["active_stage"] == "fit"
    assert snapshot["stage_started_at"] == "2026-10-05T23:00:00+00:00"
    assert snapshot["stage_elapsed_seconds"] == 3600
    assert snapshot["last_optimizer_event_at"] == "2026-10-05T23:10:00+00:00"
    assert "Generation num: 1" in snapshot["last_fedot_log_line"]


def test_coordinator_interrupt_is_not_replaced_by_sibling_cancellation(monkeypatch):
    original = KeyboardInterrupt("fixture interruption")

    def waiting_pair(task_id, *, cancellation_event, **kwargs):
        cancellation_event.wait(1)
        raise CancelledError("sibling stopped")

    def interrupt_wait(futures):
        raise original

    monkeypatch.setattr(quality, "run_quality_pair", waiting_pair)
    monkeypatch.setattr(quality, "as_completed", interrupt_wait)
    ids = tuple(dataset.task_id for dataset in load_quality_registry().datasets[:2])
    with pytest.raises(KeyboardInterrupt) as caught:
        quality.run_quality_jobs(
            stock_checkout=Path("source"),
            patch_checkout=Path("patch"),
            task_ids=ids,
            n_jobs=4,
            cpu_quota=8,
        )
    assert caught.value is original
