#!/usr/bin/env bash
# The whole research pipeline: linter -> reader twice -> verifier -> fixer.
# Every stage is resumable from files in OUT; the fixer stops at its hard budget.
#
#   FEDOTLLM_REPO_PATH=/path/to/FEDOT examples/run_pipeline.sh [outdir]
set -euo pipefail

: "${FEDOTLLM_REPO_PATH:?set it to the FEDOT checkout under test}"
: "${FEDOTLLM_LLM_API_KEY:?set it, e.g. from ~/.config/fedotllm/or_key2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/pipeline_out}"
PY="${FEDOTLLM_CONTROLLER_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${FEDOTLLM_CHEAP_MODEL:-deepseek/deepseek-v4-flash}"
FIXER_MODEL="${FEDOTLLM_FIXER_MODEL:-openai/gpt-5.4}"
BUDGET="${FEDOTLLM_NIGHT_BUDGET:-3}"
MAX_FIXES="${FEDOTLLM_NIGHT_MAX_FIXES:-10}"
export FEDOTLLM_PYTHON="${FEDOTLLM_REPO_PYTHON:?set python from the FEDOT checkout}"
mkdir -p "$OUT"

if [[ -n "$(git -C "$FEDOTLLM_REPO_PATH" status --porcelain --untracked-files=all)" ]]; then
  echo "target checkout is dirty; refusing to delete or overwrite user changes" >&2
  exit 2
fi

if [[ -n "${CLAUDE_JOB_DIR:-}" ]]; then
  [[ -s "$OUT/triage1.json" || ! -s "$CLAUDE_JOB_DIR/tmp/PASS1.json" ]] \
    || cp "$CLAUDE_JOB_DIR/tmp/PASS1.json" "$OUT/triage1.json"
  [[ -s "$OUT/triage2.json" || ! -s "$CLAUDE_JOB_DIR/tmp/PASS2.json" ]] \
    || cp "$CLAUDE_JOB_DIR/tmp/PASS2.json" "$OUT/triage2.json"
fi

if [[ ! -s "$OUT/triage1.json" || ! -s "$OUT/triage2.json" ]]; then
echo "== reader, pass 1 and 2 in parallel =="
# Two passes of the same cheap model, unioned below. Recall is what matters
# here; the verifier is what removes the noise, and it charges nothing to say no.
"$PY" "$ROOT/examples/triage_lint.py" --whole-file --model "$MODEL" --workers "${FEDOTLLM_WORKERS:-8}" \
      --out "$OUT/triage1.md" > "$OUT/triage1.log" 2>&1 &
P1=$!
"$PY" "$ROOT/examples/triage_lint.py" --whole-file --model "$MODEL" --workers "${FEDOTLLM_WORKERS:-8}" \
      --out "$OUT/triage2.md" > "$OUT/triage2.log" 2>&1 &
P2=$!
wait $P1 $P2
else
echo "== reader: reusing completed passes =="
fi

if [[ ! -s "$OUT/leads.json" ]]; then
echo "== union =="
"$PY" "$ROOT/examples/union_leads.py" \
      "$OUT/triage1.json" "$OUT/triage2.json" --out "$OUT/leads.json"
else
echo "== union: reusing $OUT/leads.json =="
fi

if [[ ! -s "$OUT/verified.json" ]]; then
echo "== verifier =="
"$PY" "$ROOT/examples/verify_leads.py" --leads "$OUT/leads.json" \
      --model "$MODEL" --out "$OUT/verified.json"
else
echo "== verifier: reusing $OUT/verified.json =="
fi

echo "== probe baseline =="
"$PY" "$ROOT/examples/warm_probe.py" "$FEDOTLLM_REPO_PATH" "$PY"

echo "== fixer =="
"$PY" "$ROOT/examples/night_fixer.py" \
      --verified "$OUT/verified.json" \
      --out "$OUT" \
      --model "$FIXER_MODEL" \
      --max-fixes "$MAX_FIXES" \
      --budget "$BUDGET" \
      --commit

echo
echo "== report =="
"$PY" "$ROOT/examples/night_report.py" --out "$OUT" --report "$OUT/NIGHT_REPORT.md"
