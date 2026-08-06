#!/usr/bin/env bash
# Apply the Phase-2 overlap/evaluator fixes to the teammate harness.
# Usage: bash apply_to_teammate.sh <harness-root>
#   <harness-root> = the dir that contains chinatravel/ and eval_tpc.py
#   (in the teammate zip that is .../tpc-qwen36-official-api/harness).
set -euo pipefail
ROOT="${1:?usage: bash apply_to_teammate.sh <harness-root-containing-chinatravel>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CT="$ROOT/chinatravel"
ENR="$CT/agent/tpc_agent_penguins/enrich"
[ -d "$ENR" ] || { echo "ERR: $ENR not found — pass the dir containing chinatravel/"; exit 1; }

echo "[1/3] de-overlap module -> enrich/enrich_deoverlap.py"
cp "$HERE/files/enrich/enrich_deoverlap.py" "$ENR/enrich_deoverlap.py"
cp "$HERE/files/enrich/enrich_fixspace.py" "$ENR/enrich_fixspace.py"
cp "$HERE/files/enrich/enrich_mustpoi.py" "$ENR/enrich_mustpoi.py"

echo "[2/3] evaluator sync (new meal/chronology/scoring, 764614c)"
cp "$HERE/files/evaluator/symbol_verification/commonsense_constraint.py" "$CT/symbol_verification/commonsense_constraint.py"
cp "$HERE/files/evaluator/evaluation/commonsense_constraint.py" "$CT/evaluation/commonsense_constraint.py"
[ -f "$ROOT/eval_tpc.py" ] && cp "$HERE/files/evaluator/eval_tpc.py" "$ROOT/eval_tpc.py" && echo "    eval_tpc.py updated"

echo "[3/3] runner.py: add deoverlap stage (first)"
if patch -p1 --dry-run -d "$ROOT" < "$HERE/runner.py.patch" >/dev/null 2>&1; then
    patch -p1 -d "$ROOT" < "$HERE/runner.py.patch"
    echo "    patched via runner.py.patch"
else
    echo "    patch didn't apply cleanly -> installing full runner_new.py"
    cp "$HERE/runner_new.py" "$ENR/runner.py"
fi

echo "--- verify ---"
python3 -m py_compile "$ENR/enrich_deoverlap.py" "$ENR/runner.py" \
    "$CT/symbol_verification/commonsense_constraint.py" "$CT/evaluation/commonsense_constraint.py" && echo "py_compile OK"
grep -q "Repeated Meal Types in One Day" "$CT/symbol_verification/commonsense_constraint.py" && echo "evaluator: NEW meal/chronology check present ✓"
grep -q '"deoverlap"' "$ENR/runner.py" && echo "runner: deoverlap stage present ✓"
grep -q '"fixspace"' "$ENR/runner.py" && echo "runner: fixspace stage present ✓"
echo "DONE. Re-run the planner (SGLang) and re-score under the new evaluator."
