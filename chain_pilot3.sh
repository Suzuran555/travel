#!/usr/bin/env bash
# ATT PILOT ROUND 3 — variant C: min-duration + geo-fallback + geo-feasibility
# + transit_weight_floor 3.0, on the post-round-2 residual. Per-uid gated merge.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
SCRATCH="/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/84a5018c-680f-44fc-88e1-0f42be32ced7/scratchpad"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
BATTERY="endday gapmeal att repair gapattr endattr dav2 travelday bfstack"

echo "=== [1/5] prepilot3 backup + target refresh ==="
rm -rf _ARCHIVE_V6_prepilot3; cp -r "$ENRICH_RES" _ARCHIVE_V6_prepilot3
sed "s|_ARCHIVE_V6_top5|results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation|" "$SCRATCH/att_top5_analysis.py" > "$SCRATCH/att_refresh3.py"
$PY "$SCRATCH/att_refresh3.py" > run_logs/att_refresh3.log 2>&1
$PY -c "import json; [print(t['uid']) for t in json.load(open('$SCRATCH/att_pilot_targets.json')) if t['uid'] != '20250323010327713880']" > pilot3_uids.txt
echo "targets: $(wc -l < pilot3_uids.txt)"

echo "=== [2/5] variant C planner (min-dur + geo-fallback + geo-feasibility + floor 3.0) ==="
( export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"transit_signal_min_duration":true,"transit_geo_fallback":true,"transit_geo_feasibility":true,"transit_weight_floor":3.0,"debug":false}'
  cat pilot3_uids.txt | xargs -P 8 -I {} $PY run_tpc.py --splits "$SPLIT" --index {} --agent UrbanTripOptimizedV6 --llm TPCLLM --lang en --oracle_translation --timeout 350 --skip 0 > run_logs/pilot3_planner.log 2>&1 )

echo "=== [3/5] battery ==="
SKIP_PLANNER=1 STAGES="$BATTERY" bash run_full_pipeline.sh > run_logs/pilot3_battery.log 2>&1

echo "=== [4/5] gated merge vs prepilot3 ==="
for i in $(seq 0 5); do $PY merge_soft.py --apply --donors _ARCHIVE_V6_prepilot3 --shard "$i/6" > "run_logs/pilot3merge_sh$i.log" 2>&1 & done; wait
grep -h "swap" run_logs/pilot3merge_sh*.log | tail -6

echo "=== [5/5] FINAL eval + archive ==="
rm -rf _ARCHIVE_V6_pilot3; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot3
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== PILOT3 DONE ==="
