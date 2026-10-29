"""Update one controller-owned GitHub issue comment per linked proposal issue."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import defaultdict
from typing import Any

MARKER = "<!-- fedot-quality-controller -->"
_ISSUE_URL = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/issues/(\d+)(?:/.*)?$")


def _issue_body(candidates: list[tuple[str, dict[str, Any]]], generated_at: str) -> str:
    lines = [MARKER, "## FEDOT: проверка предложений улучшений", "", f"Обновлено: {generated_at}", ""]
    for candidate_id, meta in candidates:
        lines.extend(
            [
                f"### {meta.get('title') or candidate_id}",
                f"- Ревизия патча: `{meta.get('patch_revision') or 'unknown'}`",
                f"- Гипотеза: {(meta.get('hypothesis') or {}).get('claim') or 'не указана'}",
                f"- Проверка: {meta.get('completed_task_pair_count', 0)}/"
                f"{meta.get('planned_task_count', len(meta.get('task_ids') or []))} датасетов; "
                f"измерено пар: {meta.get('measured_task_pair_count', 0)}",
                f"- Итог: `{meta.get('derived_quality_outcome') or meta.get('quality_outcome') or meta.get('queue_status') or meta.get('status') or 'pending'}`",
            ]
        )
        datasets = meta.get("datasets") or []
        if datasets:
            lines.extend(
                [
                    "",
                    "| Датасет | Метрика | Baseline | Изменение | Эффект |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in datasets:
                lines.append(
                    f"| `{row.get('task_id') or '—'}` | {row.get('metric') or '—'} | "
                    f"{row.get('stock_score') if row.get('stock_score') is not None else '—'} | "
                    f"{row.get('patched_score') if row.get('patched_score') is not None else '—'} | "
                    f"{row.get('delta') if row.get('delta') is not None else '—'} |"
                )
        if meta.get("reason"):
            lines.append(f"Причина: {meta['reason']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def sync_issue_comments(summary: dict[str, Any]) -> dict[str, Any]:
    """Create or update only comments carrying our marker; never edit human comments."""

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for candidate_id, meta in (summary.get("candidates") or {}).items():
        issue_url = str(meta.get("issue_url") or "")
        if issue_url:
            grouped[issue_url].append((candidate_id, meta))
    if not grouped:
        return {"created": 0, "updated": 0, "skipped": 0, "reason": "no_issue_urls"}
    gh = shutil.which("gh")
    if not gh:
        return {"created": 0, "updated": 0, "skipped": len(grouped), "reason": "gh_not_installed"}
    auth = subprocess.run(
        [gh, "auth", "status", "--hostname", "github.com"],
        capture_output=True,
        text=True,
        check=False,
    )
    if auth.returncode != 0:
        return {"created": 0, "updated": 0, "skipped": len(grouped), "reason": "gh_unauthenticated"}

    counts = {"created": 0, "updated": 0, "skipped": 0}
    for issue_url, candidates in grouped.items():
        match = _ISSUE_URL.match(issue_url)
        if not match:
            counts["skipped"] += 1
            continue
        owner, repo, number = match.groups()
        endpoint = f"repos/{owner}/{repo}/issues/{number}/comments"
        listed = subprocess.run(
            [gh, "api", "--paginate", endpoint, "--jq", f'.[] | select(.body | contains("{MARKER}")) | .id'],
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            counts["skipped"] += 1
            continue
        ids = [line.strip() for line in listed.stdout.splitlines() if line.strip().isdigit()]
        body = _issue_body(candidates, str(summary.get("generated_at") or "unknown"))
        if ids:
            result = subprocess.run(
                [gh, "api", "--method", "PATCH", f"repos/{owner}/{repo}/issues/comments/{ids[0]}", "-f", f"body={body}"],
                capture_output=True,
                text=True,
                check=False,
            )
            counts["updated" if result.returncode == 0 else "skipped"] += 1
        else:
            result = subprocess.run(
                [gh, "api", "--method", "POST", endpoint, "-f", f"body={body}"],
                capture_output=True,
                text=True,
                check=False,
            )
            counts["created" if result.returncode == 0 else "skipped"] += 1
    return counts
