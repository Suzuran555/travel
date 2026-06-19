#!/usr/bin/env bash
# UrbanTripOptimizedV5 — Phase 1 generate (parallel) + official eval.
#
# Method name MUST match run_tpc.py / eval_tpc.py (see URBANTRIP_NO1_QUICKSTART.md):
#   UrbanTripOptimizedV5_TPCLLM_en_oracletranslation
#
# Usage (from repo root, conda env chinatravel active):
#   bash scripts/run_urbantrip_v5_phase1.sh status
#   bash scripts/run_urbantrip_v5_phase1.sh generate          # 1000 queries, 8 workers
#   JOBS=16 bash scripts/run_urbantrip_v5_phase1.sh generate
#   bash scripts/run_urbantrip_v5_phase1.sh eval              # eval only
#   bash scripts/run_urbantrip_v5_phase1.sh all               # generate then eval
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- fixed TPC knobs (override only via URBANTRIP_* env, never shell $LANG) ---
AGENT="${URBANTRIP_AGENT:-UrbanTripOptimizedV5}"
LLM="${URBANTRIP_LLM:-TPCLLM}"
LOCALE="${URBANTRIP_LOCALE:-en}"          # zh | en only
SPLIT="${URBANTRIP_SPLIT:-TPC_IJCAI_2026_phase1}"
TIMEOUT="${URBANTRIP_TIMEOUT:-330}"
JOBS="${URBANTRIP_JOBS:-$(nproc 2>/dev/null || echo 4)}"

case "$LOCALE" in
  zh|en) ;;
  *)
    echo "ERROR: URBANTRIP_LOCALE must be zh or en (got: $LOCALE)" >&2
    exit 1
    ;;
esac

if [[ -n "${CHINATRAVEL_PYTHON:-}" ]]; then
  PY="$CHINATRAVEL_PYTHON"
elif [[ -x /home/elljames/miniconda3/envs/chinatravel/bin/python ]]; then
  PY=/home/elljames/miniconda3/envs/chinatravel/bin/python
else
  PY=python3
fi

# Mirror run_tpc.py lines 71-81 exactly.
method_name() {
  local m="${AGENT}_${LLM}"
  if [[ "$LOCALE" == "en" ]]; then
    m="${m}_en"
  fi
  m="${m}_oracletranslation"
  echo "$m"
}

METHOD="$(method_name)"
SPLIT_FILE="$ROOT/chinatravel/evaluation/default_splits/${SPLIT}.txt"
RES_DIR="$ROOT/results/${METHOD}"
LOG_DIR="$ROOT/logs/${METHOD}/${SPLIT}"
SCORE_FILE="$ROOT/eval_res/splits_${SPLIT}/${METHOD}_scores.json"

mkdir -p "$RES_DIR" "$LOG_DIR" "$(dirname "$SCORE_FILE")"

cmd_status() {
  local total done logs workers
  total="$(wc -l < "$SPLIT_FILE")"
  done="$(find "$RES_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
  logs="$(find "$LOG_DIR" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)"
  workers="$(pgrep -cf "run_tpc.py.*--agent ${AGENT}" 2>/dev/null || true)"
  workers="${workers:-0}"

  echo "Agent:    $AGENT"
  echo "Method:   $METHOD"
  echo "Locale:   $LOCALE  (use URBANTRIP_LOCALE, not shell \$LANG)"
  echo "Split:    $SPLIT ($total queries)"
  echo "Results:  $done / $total"
  echo "          $RES_DIR"
  echo "Logs:     $logs files in $LOG_DIR"
  echo "Workers:  $workers run_tpc.py process(es)"

  if [[ "$done" -ge "$total" ]]; then
    echo "Generate: DONE"
  elif [[ "$workers" -gt 0 ]]; then
    echo "Generate: RUNNING"
  else
    echo "Generate: STOPPED (incomplete)"
    local bad
    bad="$(grep -l 'error:' "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)"
    if [[ -n "$bad" ]]; then
      echo "Last error sample ($bad):"
      tail -5 "$bad"
    fi
  fi

  if [[ -f "$SCORE_FILE" ]]; then
    echo "Eval:     $SCORE_FILE"
    cat "$SCORE_FILE"
  elif [[ -f "$ROOT/your_tpc_scores.json" ]]; then
    echo "Eval:     $ROOT/your_tpc_scores.json (legacy)"
    tail -1 "$ROOT/your_tpc_scores.json"
  else
    echo "Eval:     not run yet"
  fi
}

cmd_generate() {
  echo "=== Generate: $METHOD on $SPLIT (${JOBS} workers) ==="
  echo "Python:  $PY"
  echo "Timeout: ${TIMEOUT}s/query"
  echo "Output:  $RES_DIR"
  echo "Logs:    $LOG_DIR"
  echo

  xargs -P "$JOBS" -I {} \
    env ROOT="$ROOT" PY="$PY" SPLIT="$SPLIT" AGENT="$AGENT" LLM="$LLM" \
        LOCALE="$LOCALE" TIMEOUT="$TIMEOUT" LOG_DIR="$LOG_DIR" \
    bash -c '
    set -euo pipefail
    uid="$1"
    log="${LOG_DIR}/${uid}.log"
    "$PY" "$ROOT/run_tpc.py" \
      --splits "$SPLIT" \
      --index "$uid" \
      --agent "$AGENT" \
      --llm "$LLM" \
      --oracle_translation \
      --timeout "$TIMEOUT" \
      --lang "$LOCALE" \
      --skip 1 \
      >"$log" 2>&1
  ' _ {} < "$SPLIT_FILE"

  local done total
  total="$(wc -l < "$SPLIT_FILE")"
  done="$(find "$RES_DIR" -maxdepth 1 -name '*.json' | wc -l)"
  echo
  echo "Generate finished: $done / $total JSON in $RES_DIR"
  if [[ "$done" -lt "$total" ]]; then
    echo "WARNING: missing $((total - done)) results — run 'bash $0 status' and check logs." >&2
    return 1
  fi
}

cmd_eval() {
  echo "=== Eval: $METHOD on $SPLIT ==="
  local done total
  total="$(wc -l < "$SPLIT_FILE")"
  done="$(find "$RES_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
  if [[ "$done" -lt "$total" ]]; then
    echo "ERROR: only $done / $total results in $RES_DIR — run generate first." >&2
    return 1
  fi

  # Same as URBANTRIP_NO1_QUICKSTART.md / run_urbantrip_no1_smoke.sh
  "$PY" "$ROOT/eval_tpc.py" \
    --splits "$SPLIT" \
    --method "$METHOD" \
    --lang "$LOCALE" \
    | tee "$ROOT/eval_res/splits_${SPLIT}/${METHOD}_eval.log"

  # eval_tpc.py appends one JSON line to your_tpc_scores.json — keep a clean copy.
  if [[ -f "$ROOT/your_tpc_scores.json" ]]; then
    tail -1 "$ROOT/your_tpc_scores.json" >"$SCORE_FILE"
    echo
    echo "Scores saved: $SCORE_FILE"
    cat "$SCORE_FILE"
  fi
}

usage() {
  sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Commands: status | generate | eval | all (default)"
}

ACTION="${1:-all}"
case "$ACTION" in
  status)   cmd_status ;;
  generate) cmd_generate ;;
  eval)     cmd_eval ;;
  all)
    cmd_generate
    cmd_eval
    ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
