#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
PY=.venv/bin/python
WORKERS="${WORKERS:-10}"
SPLIT="TPC_IJCAI_2026_phase1"
SPLIT_FILE="chinatravel/evaluation/default_splits/${SPLIT}.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"enable_compact_dwell":true,"debug":false}'
echo "bias+compact full run: $URBANTRIP_KWARGS"
cat "$SPLIT_FILE" | xargs -P "$WORKERS" -I {} \
  $PY run_tpc.py --splits "$SPLIT" --index {} --agent UrbanTripOptimizedV6 --llm TPCLLM \
    --lang en --oracle_translation --timeout 350 --skip 0 > run_logs/patched_compact_full.log 2>&1
echo "ALL BIAS+COMPACT PLANNER DONE"
