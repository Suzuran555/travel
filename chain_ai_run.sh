#!/usr/bin/env bash
# FULL AI-RUN CHAIN (stage 1 with Qwen3.6-27B translation, no oracle).
# Runs concurrently with the 4-shard pretranslate workers: plans each uid as
# soon as its translation is cached, unloads the LLM when translation is done,
# then runs the enrichment battery on the new source and the official eval.
# The AI source lives in results/UrbanTripOptimizedV6_ollama-qwen3.6-27b_en —
# fully separate from the oracle submission set. NO oracle-donor merging.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
TCACHE="cache/translation_Qwen3.6-27B_reflect"
AIRES="results/UrbanTripOptimizedV6_ollama-qwen3.6-27b_en"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# planner flag stack: the config that produced the 99.95 oracle source
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"transit_signal_min_duration":true,"transit_geo_fallback":true,"transit_geo_feasibility":true,"transit_weight_floor":3.0,"debug":false}'
mkdir -p "$AIRES" run_logs

plan_ready_uids() {  # translated but not yet planned
  local P="$1"
  comm -23 \
    <(ls "$TCACHE" 2>/dev/null | sed 's/\.json$//' | sort) \
    <(ls "$AIRES" 2>/dev/null | sed 's/\.json$//' | sort) \
    | head -400 | xargs -P "$P" -I {} "$PY" run_tpc.py --splits "$SPLIT" \
        --index {} --agent UrbanTripOptimizedV6 --llm ollama-qwen3.6-27b \
        --lang en --timeout 350 --skip 1 >> run_logs/ai_planner.log 2>&1
}

echo "=== [1/4] incremental planning while translation runs ==="
while true; do
  t=$(ls "$TCACHE" 2>/dev/null | wc -l | tr -d ' ')
  p=$(ls "$AIRES" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%H:%M') translated=$t planned=$p"
  if [ "$t" -ge 1000 ] && [ "$p" -ge "$t" ]; then break; fi
  if [ "$p" -lt "$t" ]; then
    # 4-wide while the LLM holds ~40GB; full width once translation is done
    W=4; [ "$t" -ge 1000 ] && W=10
    plan_ready_uids "$W"
  else
    sleep 120
  fi
done

echo "=== [2/4] translation+planning complete -> unload LLM ==="
ollama stop qwen3.6:27b 2>/dev/null || true

echo "=== [3/4] enrichment battery on the AI source ==="
ENRICH_RES="$AIRES" SKIP_PLANNER=1 \
  STAGES="endday gapmeal att repair gapattr endattr dav2 travelday bfstack" \
  bash run_full_pipeline.sh > run_logs/ai_battery.log 2>&1 || true
for i in $(seq 0 7); do ENRICH_RES="$AIRES" $PY enrich_fillerswap.py --shard "$i/8" --apply > "run_logs/ai_filler_sh$i.log" 2>&1 & done; wait
for i in $(seq 0 7); do ENRICH_RES="$AIRES" SEQSWAP_ALLOW_RESULTS=1 $PY enrich_seqswap.py --shard "$i/8" --apply > "run_logs/ai_seq_sh$i.log" 2>&1 & done; wait
ENRICH_RES="$AIRES" $PY enrich_lateattr.py --apply > run_logs/ai_lateattr.log 2>&1 || true

echo "=== [4/4] official eval of the AI source ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_ollama-qwen3.6-27b_en \
  --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== AI RUN DONE ==="
