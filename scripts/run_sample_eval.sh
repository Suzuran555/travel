#!/usr/bin/env bash
# Run + evaluate a small UID sample with a given URBANTRIP_KWARGS override.
# Usage: TAG=phaseA KWARGS='{"...":true}' UID_FILE=scripts/phase_ab_sample_uids.txt \
#        scripts/run_sample_eval.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chinatravel

PY="${CHINATRAVEL_PYTHON:-$CONDA_PREFIX/bin/python}"
WORKERS="${WORKERS:-6}"
SPLIT="TPC_IJCAI_2026_phase1"
TAG="${TAG:?set TAG}"
UID_FILE="${UID_FILE:?set UID_FILE}"
export URBANTRIP_KWARGS="${KWARGS:?set KWARGS}"

LOG_DIR="$ROOT/logs/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation/${SPLIT}_sample_${TAG}"
OUT_DIR="$ROOT/submission_full_v7"
RES_DIR="$ROOT/results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RES_DIR"

TOTAL="$(wc -l < "$UID_FILE")"
echo "=== Sample run [$TAG]: $TOTAL queries, ${WORKERS} workers, 350s ==="
echo "URBANTRIP_KWARGS=$URBANTRIP_KWARGS"
echo "Logs: $LOG_DIR"

xargs -P "$WORKERS" -I {} \
  env ROOT="$ROOT" PY="$PY" SPLIT="$SPLIT" LOG_DIR="$LOG_DIR" URBANTRIP_KWARGS="$URBANTRIP_KWARGS" \
  bash -c '
    set -euo pipefail
    uid="$1"
    "$PY" "$ROOT/run_tpc.py" \
      --splits "$SPLIT" \
      --index "$uid" \
      --agent UrbanTripOptimizedV5 \
      --llm TPCLLM \
      --oracle_translation \
      --timeout 350 \
      --lang en \
      --skip 0 \
      >"${LOG_DIR}/${uid}.log" 2>&1
    echo "done: $uid"
  ' _ {} < "$UID_FILE"

echo "=== Generate finished for sample [$TAG] ==="
