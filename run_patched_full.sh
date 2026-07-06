#!/usr/bin/env bash
# Full patched planner run (travel-day timing bias ON) into a SCRATCH dir,
# leaving the enriched live results untouched until we compare.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
set -uo pipefail
PY=.venv/bin/python
WORKERS="${WORKERS:-10}"
SPLIT="TPC_IJCAI_2026_phase1"
SPLIT_FILE="chinatravel/evaluation/default_splits/${SPLIT}.txt"
RES_DIR="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"debug":false}'
echo "patched full run: WORKERS=$WORKERS, 350s, bias ON"
echo "$URBANTRIP_KWARGS"
cat "$SPLIT_FILE" | xargs -P "$WORKERS" -I {} \
  $PY run_tpc.py --splits "$SPLIT" --index {} --agent UrbanTripOptimizedV6 --llm TPCLLM \
    --lang en --oracle_translation --timeout 350 --skip 0 > run_logs/patched_full.log 2>&1
echo "ALL PATCHED PLANNER DONE"
