#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"; SPLIT_FILE="chinatravel/evaluation/default_splits/${SPLIT}.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"debug":false}'
echo "=== [1/6] weight-0.30 planner ==="
cat "$SPLIT_FILE" | xargs -P 10 -I {} $PY run_tpc.py --splits "$SPLIT" --index {} --agent UrbanTripOptimizedV6 --llm TPCLLM --lang en --oracle_translation --timeout 350 --skip 0 > run_logs/weight30_planner.log 2>&1
echo "=== [2/6] fresh eval ==="; $PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== [3/6] endday ==="; bash run_endday_parallel.sh >/dev/null 2>&1
echo "=== [4/6] gapmeal ==="; bash run_gapmeal_parallel.sh >/dev/null 2>&1
echo "=== [5/6] att(metro) + repair ==="; bash run_att_parallel.sh >/dev/null 2>&1; $PY repair_fails.py --apply 2>&1 | grep -E "^fixed"
rm -rf _ARCHIVE_V6_weight30; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_weight30
echo "=== [6/6] FINAL eval ==="; $PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== WEIGHT30 CHAIN DONE ==="
