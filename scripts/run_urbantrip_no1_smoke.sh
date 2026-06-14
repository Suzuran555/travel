#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CHINATRAVEL_PYTHON:-}" ]]; then
  PY="$CHINATRAVEL_PYTHON"
elif [[ -x /home/elljames/miniconda3/envs/chinatravel/bin/python ]]; then
  PY=/home/elljames/miniconda3/envs/chinatravel/bin/python
else
  PY=python
fi

QUERY_UID="${1:-20250320174446059265}"
SPLIT="${URBANTRIP_SPLIT:-TPC_IJCAI_2026_phase1_smoke3}"
TIMEOUT="${URBANTRIP_TIMEOUT:-300}"
METHOD="UrbanTripOptimizedV2_TPCLLM_en_oracletranslation"

echo "Using Python: $PY"
echo "Running UrbanTripOptimizedV2 for $QUERY_UID from split $SPLIT"

"$PY" run_tpc.py \
  --splits "$SPLIT" \
  --index "$QUERY_UID" \
  --agent UrbanTripOptimizedV2 \
  --llm TPCLLM \
  --oracle_translation \
  --timeout "$TIMEOUT" \
  --lang en

echo "Evaluating $METHOD on $SPLIT"
"$PY" eval_tpc.py \
  --splits "$SPLIT" \
  --method "$METHOD" \
  --lang en
