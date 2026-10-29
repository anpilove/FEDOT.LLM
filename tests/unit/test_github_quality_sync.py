from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "research/evolve/scripts/github_quality_sync.py"
SPEC = importlib.util.spec_from_file_location("github_quality_sync", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
github_quality_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(github_quality_sync)


def test_issue_sync_creates_then_updates_only_controller_comment(monkeypatch):
    monkeypatch.setattr(github_quality_sync.shutil, "which", lambda _name: "/usr/bin/gh")
    list_calls = 0
    mutations = []

    def fake_run(command, **_kwargs):
        nonlocal list_calls
        if command[1:4] == ["auth", "status", "--hostname"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[1] == "api" and "--paginate" in command:
            list_calls += 1
            return SimpleNamespace(returncode=0, stdout="" if list_calls == 1 else "42\n", stderr="")
        if command[1] == "api":
            mutations.append(command)
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(github_quality_sync.subprocess, "run", fake_run)
    summary = {
        "generated_at": "2026-09-28T00:00:00+00:00",
        "candidates": {
            "candidate-r1": {
                "issue_url": "https://github.com/org/repo/issues/123",
                "title": "Improve forecasting",
                "patch_revision": "abc123",
                "hypothesis": {"claim": "Improve RMSE"},
                "queue_status": "running",
                "completed_task_pair_count": 1,
                "planned_task_count": 8,
                "measured_task_pair_count": 1,
                "datasets": [
                    {
                        "task_id": "fedot-ts-beer",
                        "metric": "rmse",
                        "stock_score": 4.0,
                        "patched_score": 3.5,
                        "delta": 0.5,
                    }
                ],
            }
        },
    }

    created = github_quality_sync.sync_issue_comments(summary)
    updated = github_quality_sync.sync_issue_comments(summary)

    assert created["created"] == 1
    assert updated["updated"] == 1
    assert len(mutations) == 2
    assert mutations[0][2:4] == ["--method", "POST"]
    assert mutations[1][2:4] == ["--method", "PATCH"]
    assert github_quality_sync.MARKER in mutations[0][-1]
    assert "fedot-ts-beer" in mutations[0][-1]


def test_issue_sync_skips_gracefully_without_github_auth(monkeypatch):
    monkeypatch.setattr(github_quality_sync.shutil, "which", lambda _name: "/usr/bin/gh")
    monkeypatch.setattr(
        github_quality_sync.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="not logged in"),
    )
    result = github_quality_sync.sync_issue_comments(
        {
            "candidates": {
                "candidate": {"issue_url": "https://github.com/org/repo/issues/1"}
            }
        }
    )
    assert result["reason"] == "gh_unauthenticated"
    assert result["skipped"] == 1
