#!/usr/bin/env python3
"""Turn a night's artefacts into a report whose every number is checkable.

Nothing here is written from memory or from a model's summary: the counts come
from the JSON the stages wrote, and a stage that did not run is reported as not
run rather than as zero. That distinction is the point — "no defects found" and
"the verifier never got there" look identical in a hand-written summary and
mean opposite things.

    uv run python examples/night_report.py --out pipeline_out --report NIGHT_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

CLASS_NAMES = {1: "критический", 2: "обычный", 3: "смелл", 4: "косметика"}


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def defect_class(row: dict[str, Any]) -> str:
    if row.get("status") != "confirmed":
        return "excluded"
    if row.get("kind") in {"declared_not_used", "nondeterministic", "timeout"}:
        return "critical"
    return "ordinary"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="pipeline output dir")
    ap.add_argument("--report", type=Path, default=Path("NIGHT_REPORT.md"))
    args = ap.parse_args()
    out = args.out

    passes = [load(out / f"triage{i}.json") for i in (1, 2)]
    passes = [p for p in passes if p is not None]
    leads = load(out / "leads.json")
    verified = load(out / "verified.json")
    state = load(out / "NIGHT_STATE.json")
    journal = jsonl(out / "repair" / "evolution_journal.jsonl")
    bench = load(out / "repair" / "bench.json")

    L = ["# Ночной прогон EvolveAgent", ""]

    L += ["## Воронка", "", "| ступень | сколько | чем решается |",
          "|---|---|---|"]
    if passes:
        total = len(passes[0])
        L.append(f"| находки линтера, показанные агенту | {total} | ruff, детерминированно |")
        for n, rows in enumerate(passes, 1):
            c = Counter(r.get("verdict") for r in rows)
            L.append(f"| проход читателя {n}: live / extra / inert | "
                     f"{c.get('live', 0)} / {c.get('extra', 0)} / {c.get('inert', 0)} "
                     f"| дешёвая модель |")
    else:
        L.append("| читатель | не запускался | — |")

    if leads is None:
        L.append("| объединение проходов | не запускалось | — |")
    else:
        agreed = sum(1 for r in leads if r.get("agreed"))
        L.append(f"| подозрений после объединения | {len(leads)} "
                 f"(оба прохода: {agreed}) | union_leads.py |")

    if verified is None:
        L.append("| проверяющий | **не запускался** | — |")
    else:
        c = Counter(r.get("status") for r in verified)
        L.append(f"| подтверждено публичным вызовом | **{c.get('confirmed', 0)}** | запуск скрипта |")
        L.append(f"| только внутренний путь | {c.get('internal only', 0)} | запуск скрипта |")
        L.append(f"| упало не в том файле | {c.get('wrong site', 0)} | трассировка не называет файл подозрения |")
        L.append(f"| опровергнуто | {c.get('refuted', 0)} | скрипт отработал без ошибки |")
        L.append(f"| не проверяется запуском | {c.get('not testable', 0)} | модель отказалась писать скрипт |")
        other = sum(v for k, v in c.items() if k not in
                    {"confirmed", "internal only", "refuted", "not testable", "wrong site"})
        if other:
            L.append(f"| прочие исходы | {other} | см. verified.json |")

    L += ["", "## Починка", ""]
    attempts = state.get("attempts", []) if isinstance(state, dict) else []
    if attempts:
        accepted = [row for row in attempts if row.get("success")]
        fixed_ids = set(state.get("fixed", []))
        classes = Counter(
            defect_class(row)
            for row in (verified or [])
            if f"{row.get('file')}:{row.get('line')}" in fixed_ids
        )
        covered_ids = {
            row["id"] for row in attempts if row.get("covered_by_prior_fix")
        }
        verified_by_id = {
            f"{row.get('file')}:{row.get('line')}": row
            for row in (verified or [])
        }
        L += [
            f"Подтверждённых дефектов закрыто: **{len(fixed_ids)} из "
            f"{sum(1 for row in (verified or []) if row.get('status') == 'confirmed')}**. "
            f"Принято патчей: **{len(accepted)}**; ещё **{len(covered_ids)}** "
            "строки verifier закрыты теми же общими патчами.",
            "",
            "| класс | закрыто |",
            "|---|---:|",
            f"| критический | {classes.get('critical', 0)} |",
            f"| обычный | {classes.get('ordinary', 0)} |",
            f"| косметический | {classes.get('cosmetic', 0)} |",
            "",
            "| доказанный дефект | исход | стоимость | commit |",
            "|---|---|---:|---|",
        ]
        attempts_by_id: dict[str, list[dict[str, Any]]] = {}
        for row in attempts:
            attempts_by_id.setdefault(row["id"], []).append(row)
        for item_id, item_attempts in attempts_by_id.items():
            evidence = verified_by_id.get(item_id, {})
            successful = next((row for row in item_attempts if row.get("success")), None)
            covered = any(row.get("covered_by_prior_fix") for row in item_attempts)
            outcome = "патч принят" if successful else "закрыт общим патчем" if covered else "отклонён"
            detail = (evidence.get("why") or item_id)[:100]
            item_cost = sum(float(row.get("cost_usd", 0)) for row in item_attempts)
            L.append(
                f"| `{item_id}` — {detail} | {outcome} | "
                f"${item_cost:.4f} | "
                f"`{successful.get('commit', '—') if successful else '—'}` |"
            )
    elif not journal:
        L += ["Чинильщик не запускался.", ""]
    else:
        ok = [r for r in journal if r.get("success")]
        cls = Counter(r.get("severity") for r in ok)
        L += [f"Кругов: **{len(journal)}**, принято патчей: **{len(ok)}**.", "",
              "| класс | принято |", "|---|---|"]
        for k in sorted(CLASS_NAMES):
            L.append(f"| {k} — {CLASS_NAMES[k]} | {cls.get(k, 0)} |")
        valuable = cls.get(1, 0) + cls.get(2, 0)
        L += ["", f"Ценных (класс 1–2): **{valuable}**. Это и есть мера успеха — "
              "не доля успешных кругов, которую агент может завысить, выбирая "
              "лёгкие цели.", ""]
        L += ["| файл | класс | что исправлено |", "|---|---|---|"]
        for r in ok:
            L.append(f"| `{r.get('file', '')}` | {r.get('severity')} | "
                     f"{str(r.get('problem', ''))[:110]} |")

    L += ["", "## Деньги", ""]
    usage_paths = {
        "читатель, проход 1": out / "triage1.usage.json",
        "читатель, проход 2": out / "triage2.usage.json",
        "проверяющий": out / "verified.usage.json",
    }
    costs = {}
    for name, path in usage_paths.items():
        usage = load(path)
        costs[name] = float((usage or {}).get("cost_usd", 0))
        if usage is None:
            L.append(f"- {name}: **не измерено**")
        else:
            L.append(
                f"- {name}: **${costs[name]:.4f}**, "
                f"{usage.get('requests', 0)} LLM-запросов"
            )
    if attempts:
        fixer_cost = sum(float(row.get("cost_usd", 0)) for row in attempts)
    elif isinstance(bench, list):
        fixer_cost = sum(float(row.get("cost_usd", 0)) for row in bench)
    else:
        fixer_cost = 0.0
    costs["чинильщик"] = fixer_cost
    L.append(f"- чинильщик: **${fixer_cost:.4f}**")
    measured = all(path.is_file() for path in usage_paths.values())
    total = sum(costs.values())
    qualifier = "" if measured else " (без неизмеренных стадий)"
    L += ["", f"**Итого: ${total:.4f}{qualifier}.**"]

    L += [
        "",
        "## Модели по ролям",
        "",
        "- Читатель: `deepseek/deepseek-v4-flash` — дешёвая массовая "
        "классификация, два независимых прохода.",
        "- Проверяющий: `deepseek/deepseek-v4-flash` — пишет короткий скрипт; "
        "вердикт всё равно выносит интерпретатор.",
        "- Чинильщик: `openai/gpt-5.4` — дорогая модель используется только "
        "после публичного воспроизведения.",
    ]

    L += ["", "## Чего этот прогон не показывает", "",
          "- Дефекты, до которых нельзя дойти публичным вызовом, отброшены "
          "намеренно — путь в коде есть, пользователь туда не попадёт.",
          "- Для `KeyError`, `AttributeError` и `IndexError` proof требует "
          "нормального завершения либо содержательного `ValueError`/`TypeError`; "
          "качество текста сверх непустого сообщения автоматически не оценивается.",
          "- Подозрения, которые модель отказалась превращать в скрипт, не "
          "опровергнуты. Они просто не проверены."]

    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
