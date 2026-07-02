#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chinatravel

PY="${CHINATRAVEL_PYTHON:-$CONDA_PREFIX/bin/python}"
WORKERS="${WORKERS:-7}"
SPLIT="TPC_IJCAI_2026_phase1"
TAG="v13"
SPLIT_FILE="$ROOT/chinatravel/evaluation/default_splits/${SPLIT}.txt"
LOG_DIR="$ROOT/logs/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation/${SPLIT}_350s_${TAG}"
OUT_DIR="$ROOT/submission_full_v7"
RES_DIR="$ROOT/results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation"
RUNNER_LOG="$OUT_DIR/phase1_1000_350s_${TAG}_runner.log"

# v13: bundle search + geo anchor + stability extras + bed_number filter +
# budget-aware bundle weights + cost-augmented intercity beam + POI-relax fallback.
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"debug":false}'

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RES_DIR"

TOTAL="$(wc -l < "$SPLIT_FILE")"
echo "=== Generate: $TOTAL queries, ${WORKERS} workers, 350s (${TAG}) ===" | tee "$RUNNER_LOG"
echo "URBANTRIP_KWARGS=$URBANTRIP_KWARGS" | tee -a "$RUNNER_LOG"
echo "Logs: $LOG_DIR" | tee -a "$RUNNER_LOG"
echo "Results: $RES_DIR" | tee -a "$RUNNER_LOG"

format_duration() {
  local seconds="$1"
  printf "%02d:%02d:%02d" "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

render_progress() {
  local done="$1"
  local total="$2"
  local started="$3"
  local workers="$4"
  local now elapsed percent filled empty eta eta_text

  now="$(date +%s)"
  elapsed="$((now - started))"
  if [[ "$total" -gt 0 ]]; then
    percent="$((done * 100 / total))"
    filled="$((done * 30 / total))"
  else
    percent=100
    filled=30
  fi
  empty="$((30 - filled))"

  if [[ "$done" -gt 0 && "$done" -lt "$total" ]]; then
    eta="$(((elapsed * (total - done)) / done))"
    eta_text="$(format_duration "$eta")"
  elif [[ "$done" -ge "$total" ]]; then
    eta_text="00:00:00"
  else
    eta_text="--:--:--"
  fi

  bar="$(printf "%*s" "$filled" "" | tr ' ' '#')$(printf "%*s" "$empty" "" | tr ' ' '-')"

  printf "\r[%s] %3d%%  %d/%d done  elapsed %s  eta %s  active %d" \
    "$bar" "$percent" "$done" "$total" "$(format_duration "$elapsed")" "$eta_text" "$workers"
}

PROGRESS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/run_v13_progress.XXXXXX")"
cleanup_progress() {
  rm -rf "$PROGRESS_DIR"
}
trap cleanup_progress EXIT

xargs -P "$WORKERS" -I {} \
  env ROOT="$ROOT" PY="$PY" SPLIT="$SPLIT" LOG_DIR="$LOG_DIR" PROGRESS_DIR="$PROGRESS_DIR" URBANTRIP_KWARGS="$URBANTRIP_KWARGS" \
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
    touch "${PROGRESS_DIR}/${uid}.done"
  ' _ {} < "$SPLIT_FILE" &
GEN_PID="$!"
STARTED="$(date +%s)"

while kill -0 "$GEN_PID" 2>/dev/null; do
  DONE_MARKERS="$(find "$PROGRESS_DIR" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)"
  ACTIVE_WORKERS="$( (pgrep -P "$GEN_PID" 2>/dev/null || true) | wc -l | tr -d ' ')"
  render_progress "$DONE_MARKERS" "$TOTAL" "$STARTED" "$ACTIVE_WORKERS"
  sleep 5
done

set +e
wait "$GEN_PID"
GEN_STATUS="$?"
set -e

DONE_MARKERS="$(find "$PROGRESS_DIR" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)"
render_progress "$DONE_MARKERS" "$TOTAL" "$STARTED" 0
echo

if [[ "$GEN_STATUS" -ne 0 ]]; then
  echo "Generate failed after $DONE_MARKERS / $TOTAL queries. Check logs in $LOG_DIR" | tee -a "$RUNNER_LOG" >&2
  exit "$GEN_STATUS"
fi

DONE="$(find "$RES_DIR" -maxdepth 1 -name '*.json' | wc -l)"
echo "Generate finished: $DONE / $TOTAL" | tee -a "$RUNNER_LOG"

echo "=== Eval ===" | tee -a "$RUNNER_LOG"
"$PY" "$ROOT/eval_tpc.py" \
  --splits "$SPLIT" \
  --method UrbanTripOptimizedV5_TPCLLM_en_oracletranslation \
  --lang en \
  2>&1 | tee "$OUT_DIR/phase1_1000_eval_350s_${TAG}.log"

echo "=== Failure analysis ===" | tee -a "$RUNNER_LOG"
"$PY" "$ROOT/scripts/analyze_phase1_failures.py" \
  --split "$SPLIT" \
  --lang en \
  --log "$OUT_DIR/phase1_1000_eval_350s_${TAG}.log" \
  --out "$OUT_DIR/phase1_1000_eval_350s_${TAG}_summary.json"

echo "ALL_DONE" | tee -a "$RUNNER_LOG"
