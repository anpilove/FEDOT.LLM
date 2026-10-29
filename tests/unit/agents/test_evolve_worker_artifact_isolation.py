"""Physical workers never consume a previous run's files or cache namespace."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from fedotllm.agents.evolve.evaluation import fedot_quality as quality
from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    # Exercise a real subprocess and its real spec/result files, without ML.
    script = tmp_path / "isolation_protocol.py"
    script.write_text("""import json, os, sys, time
from pathlib import Path
spec = json.loads(Path(sys.argv[sys.argv.index('--spec-json') + 1]).read_text())
out = Path(sys.argv[sys.argv.index('--out') + 1])
identity = {k: spec.get(k) for k in ('side', 'attempt', 'attempt_id', 'registry_identity', 'evaluation_identity', 'worker_invocation_id')}
identity['task_id'] = spec['source_dataset']
mode_path = Path(__file__).with_name('isolation-mode')
mode = mode_path.read_text() if mode_path.exists() else 'ok'
if mode == 'delay' and spec['side'] == 'stock':
    time.sleep(0.2)
if mode == 'timeout':
    time.sleep(30)
if mode.startswith('stale'):
    identity['worker_invocation_id'] = 'previous-invocation'
payload = dict(identity, status='crash' if mode == 'stale-crash' else 'ok', score=0.75, metric_observations={'metric_valid': True})
if mode == 'stale-crash':
    payload.update(failure_kind='storage_error', detail='No space left on device')
payload['metric_observations']['child_artifact_root'] = os.environ.get('EVOLVE_AGENT_TMP')
payload['metric_observations']['child_stock_cache_root'] = os.environ.get('EVOLVE_QUALITY_STOCK_CACHE')
payload['metric_observations']['dataset_environment'] = {name: os.environ.get(name) for name in ('EVOLVE_OPENML_CACHE', 'EVOLVE_QUALITY_DATA_CACHE', 'SKLEARN_DATA', 'EVOLVE_QUALITY_DATA_OFFLINE')}
out.write_text(json.dumps(payload))
""")
    monkeypatch.setenv("FEDOTLLM_REPO_PYTHON", sys.executable)
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path / "global-artifacts"))
    monkeypatch.setenv("EVOLVE_QUALITY_STOCK_CACHE", str(tmp_path / "global-cache"))
    monkeypatch.setattr(quality, "WORKER_MODULE", "isolation_protocol")
    return tmp_path


def test_repeating_same_logical_attempt_uses_fresh_physical_paths(protocol):
    first = quality.run_quality_side(
        "fedot-ts-beer", checkout=protocol, side="patch", proposal_id="same"
    )
    second = quality.run_quality_side(
        "fedot-ts-beer", checkout=protocol, side="patch", proposal_id="same"
    )
    assert first.status == second.status == "ok"
    paths = [Path(r.metric_observations["worker_result_path"]) for r in (first, second)]
    assert paths[0] != paths[1]
    assert all(p.is_file() for p in paths)
    assert (
        first.metric_observations["worker_invocation_id"]
        != second.metric_observations["worker_invocation_id"]
    )


@pytest.mark.parametrize("mode", ["stale-ok", "stale-crash"])
def test_stale_structured_payload_rejected_before_storage_detection(
    protocol, monkeypatch, mode
):
    # Trusting any status before checking the nonce can attribute an old disk
    # failure to the new physical worker and stop the whole controller.
    (protocol / "isolation-mode").write_text(mode)
    result = quality.run_quality_side("fedot-ts-beer", checkout=protocol, side="patch")
    assert result.status == "invalid"
    assert result.metric_observations["failure_kind"] == "worker_identity_mismatch"
    assert (
        "worker_invocation_id" in result.metric_observations["identity_mismatch_fields"]
    )


def test_concurrent_explicit_roots_isolate_artifacts_and_stock_cache(protocol):
    def evaluate(name):
        return quality.run_quality_side(
            "fedot-ts-beer",
            checkout=protocol,
            side="stock",
            artifact_root=protocol / name / "worker_artifacts",
            stock_cache_root=protocol / name / "stock_cache",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(evaluate, ("one", "two"))
    for name, result in zip(("one", "two"), (first, second)):
        assert result.status == "ok"
        observations = result.metric_observations
        assert Path(observations["worker_result_path"]).is_relative_to(
            protocol / name / "worker_artifacts"
        )
        assert observations["child_stock_cache_root"] == str(
            protocol / name / "stock_cache"
        )
        assert Path(observations["child_artifact_root"]).is_relative_to(
            protocol / name / "worker_artifacts"
        )
        assert list((protocol / name / "stock_cache").glob("*.json"))
    assert not (protocol / "global-cache").exists()


def test_controller_records_one_plan_from_start_through_finished(protocol):
    from fedotllm.agents.evolve.controller.quality_executor import measure_fedot_quality
    from fedotllm.agents.evolve.storage.journal import read_jsonl

    journal = protocol / "workspace" / "quality_jobs.jsonl"
    journal.parent.mkdir()
    measure_fedot_quality(
        protocol,
        protocol,
        journal=journal,
        task_ids=("fedot-ts-beer",),
        n_jobs=1,
        cpu_quota=1,
    )
    events = read_jsonl(journal, strict=True)
    for side in ("stock", "patch"):
        lifecycle = [e for e in events if e.get("side") == side]
        assert [e["event"] for e in lifecycle] == [
            "quality_task_side_started",
            "quality_task_worker_spawned",
            "quality_task_side_finished",
        ]
        paths = lifecycle[0]["worker_paths"]
        assert paths == lifecycle[1]["worker_paths"] == lifecycle[2]["worker_paths"]
        assert paths["worker_invocation_id"] == lifecycle[2]["worker_invocation_id"]
        assert Path(paths["worker_result_path"]).is_relative_to(
            journal.parent / "worker_artifacts"
        )
        assert Path(paths["worker_result_path"]).is_file()


def test_checked_dataset_cache_routes_reach_the_physical_worker(protocol, monkeypatch):
    expected = {
        "EVOLVE_OPENML_CACHE": str(protocol / "openml"),
        "EVOLVE_QUALITY_DATA_CACHE": str(protocol / "splits"),
        "SKLEARN_DATA": str(protocol / "sklearn"),
        "EVOLVE_QUALITY_DATA_OFFLINE": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    result = quality.run_quality_side("fedot-ts-beer", checkout=protocol, side="patch")
    assert result.status == "ok"
    assert result.metric_observations["dataset_environment"] == expected


@pytest.mark.parametrize("status", ["ok", "crash"])
def test_recovery_uses_persisted_invocation_and_requires_nonce(tmp_path, status):
    from fedotllm.agents.evolve.controller.quality_executor import (
        _recover_durable_worker_result,
    )

    plan = quality.create_worker_artifact_plan(
        "fedot-ts-beer",
        "patch",
        "proposal",
        1,
        artifact_root=tmp_path / "original-run" / "worker_artifacts",
    )
    result_path = Path(plan["worker_result_path"])
    result_path.parent.mkdir(parents=True)
    spec = {"n_jobs": 4, "timeout_seconds": 5400, "fedot_search_budget_seconds": 4500}
    identity = dict(
        proposal_id="proposal",
        task_id="fedot-ts-beer",
        source_dataset="fedot-ts-beer",
        side="patch",
        attempt=1,
        attempt_id="proposal:fedot-ts-beer:patch:1",
        registry_identity="registry",
        evaluation_identity={"source": "exact"},
        worker_invocation_id=plan["worker_invocation_id"],
    )
    event = dict(
        identity,
        candidate="proposal",
        spec=spec,
        worker_paths=plan,
        worker_result_path=str(result_path),
        stage_status_path=plan["stage_status_path"],
    )
    payload = dict(
        identity,
        status=status,
        score=0.75 if status == "ok" else None,
        failure_kind="fedot_fit_error" if status == "crash" else "",
        metric_observations={
            "metric_valid": status == "ok",
            "metric_outcome": "not_measured",
        },
    )
    result_path.write_text(json.dumps(payload))
    stage_path = Path(plan["stage_status_path"])
    stage_path.write_text(json.dumps(dict(identity, **spec)))

    def recover():
        return _recover_durable_worker_result(
            event,
            candidate_id="proposal",
            task_id="fedot-ts-beer",
            side="patch",
            attempt=1,
            registry_identity="registry",
        )

    recovered = recover()
    assert recovered is not None
    assert recovered.status == status
    assert (
        recovered.metric_observations["worker_invocation_id"]
        == plan["worker_invocation_id"]
    )
    payload["worker_invocation_id"] = "previous-invocation"
    result_path.write_text(json.dumps(payload))
    assert recover() is None
    payload["worker_invocation_id"] = identity["worker_invocation_id"]
    result_path.write_text(json.dumps(payload))
    stage_path.write_text(
        json.dumps({**identity, **spec, "worker_invocation_id": "previous-invocation"})
    )
    assert recover() is None


def test_parent_timeout_manifest_and_result_carry_exact_invocation(protocol):
    from dataclasses import replace

    (protocol / "isolation-mode").write_text("timeout")
    result = quality.run_quality_side(
        "fedot-ts-beer",
        checkout=protocol,
        side="patch",
        registry=replace(load_quality_registry(), watchdog_seconds=0.15),
    )
    assert result.status == "timeout"
    observations = result.metric_observations
    manifest = json.loads(Path(observations["timeout_manifest_path"]).read_text())
    assert observations["worker_invocation_id"] == manifest["worker_invocation_id"]
    assert (
        observations["worker_paths"]["worker_invocation_id"]
        == manifest["worker_invocation_id"]
    )
    assert (
        observations["worker_paths"]["worker_result_path"]
        == manifest["worker_result_path"]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_shared_stock_result_retains_original_worker_plan(protocol, asynchronous):
    import threading

    (protocol / "isolation-mode").write_text("delay")
    barrier = threading.Barrier(2)
    events = []
    compute = ThreadPoolExecutor(max_workers=2)

    def evaluate(name):
        def record(spec, side, phase, result, attempt):
            events.append((name, side, phase, dict(spec["worker_paths"])))
            if side == "stock" and phase == "started":
                barrier.wait(timeout=3)

        kwargs = {}
        if asynchronous:
            kwargs["submit_compute"] = lambda work, **_options: compute.submit(work)
        runner = (
            quality._run_quality_pair_async
            if asynchronous
            else quality.run_quality_pair
        )
        result = runner(
            "fedot-ts-beer",
            stock_checkout=protocol,
            patch_checkout=protocol,
            proposal_id=name,
            on_side_event=record,
            artifact_root=protocol / "shared-artifacts",
            stock_cache_root=protocol / "shared-cache",
            **kwargs,
        )
        return result.result(timeout=5) if asynchronous else result

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate, ("owner-or-waiter-a", "owner-or-waiter-b")))
    compute.shutdown()
    stocks = [r["stock_result"].metric_observations for r in rows]
    assert sum(bool(s.get("singleflight_shared")) for s in stocks) == 1
    assert stocks[0]["worker_paths"] == stocks[1]["worker_paths"]
    spawns = [e for e in events if e[1:3] == ("stock", "spawned")]
    assert len(spawns) == 1
    assert spawns[0][3] == stocks[0]["worker_paths"]
    starts = [e[3] for e in events if e[1:3] == ("stock", "started")]
    assert starts[0]["worker_invocation_id"] != starts[1]["worker_invocation_id"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_resuming_finished_sides_reuses_original_paths_and_nonce(
    protocol, asynchronous
):
    from fedotllm.agents.evolve.controller.quality_executor import measure_fedot_quality
    from fedotllm.agents.evolve.storage.journal import read_jsonl
    from fedotllm.agents.evolve.types import PatchCandidate

    journal = protocol / "resume-workspace" / "quality_jobs.jsonl"
    journal.parent.mkdir()
    from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler

    scheduler = SharedPairScheduler(max_pairs=1) if asynchronous else None
    kwargs = dict(
        pair_scheduler=scheduler,
        journal=journal,
        candidate=PatchCandidate("original-proposal"),
        task_ids=("fedot-ts-beer",),
        n_jobs=1,
        cpu_quota=1,
    )
    measure_fedot_quality(protocol, protocol, **kwargs)
    before = read_jsonl(journal, strict=True)
    measure_fedot_quality(protocol, protocol, **kwargs)
    after = read_jsonl(journal, strict=True)
    if scheduler is not None:
        scheduler.shutdown()
    resumed = after[len(before) :]
    assert not any(
        e.get("event") in {"quality_task_side_started", "quality_task_worker_spawned"}
        for e in resumed
    )
    for side in ("stock", "patch"):
        original = next(
            e
            for e in before
            if e.get("side") == side and e.get("event") == "quality_task_side_finished"
        )
        reused = next(
            e
            for e in resumed
            if e.get("side") == side and e.get("event") == "quality_task_side_reused"
        )
        assert reused["worker_paths"] == original["worker_paths"]
        assert reused["artifact_state"] == "reused"
        assert reused["worker_invocation_id"] == original["worker_invocation_id"]
        assert reused["spec"]["worker_paths"] == original["spec"]["worker_paths"]
