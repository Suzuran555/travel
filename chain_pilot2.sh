#!/usr/bin/env bash
# ATT PILOT ROUND 2 — two signal variants, per-uid best-of-three merge.
# A = transit_signal_min_duration; B = A + transit_geo_fallback.
# Merge keeps, per uid: best passing plan among {prepilot2, A-enriched, B-enriched}.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
SCRATCH="/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/84a5018c-680f-44fc-88e1-0f42be32ced7/scratchpad"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
BASE_KW='"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"transit_signal_min_duration":true'
BATTERY="endday gapmeal att repair gapattr endattr dav2 travelday bfstack"

run_planner() { # run_planner <kwargs-json> <log>
  ( export URBANTRIP_KWARGS="$1"
    cat pilot2_uids.txt | xargs -P 8 -I {} $PY run_tpc.py --splits "$SPLIT" --index {} \
      --agent UrbanTripOptimizedV6 --llm TPCLLM --lang en --oracle_translation \
      --timeout 350 --skip 0 > "run_logs/$2" 2>&1 )
}

echo "=== [1/7] prepilot2 backup + target refresh ==="
rm -rf _ARCHIVE_V6_prepilot2; cp -r "$ENRICH_RES" _ARCHIVE_V6_prepilot2
sed "s|_ARCHIVE_V6_top5|results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation|" "$SCRATCH/att_top5_analysis.py" > "$SCRATCH/att_refresh2.py"
$PY "$SCRATCH/att_refresh2.py" > run_logs/att_refresh2.log 2>&1
$PY -c "import json; [print(t['uid']) for t in json.load(open('$SCRATCH/att_pilot_targets.json')) if t['uid'] != '20250323010327713880']" > pilot2_uids.txt
echo "targets: $(wc -l < pilot2_uids.txt)"

echo "=== [2/7] variant A planner (min-duration) ==="
run_planner "{${BASE_KW},\"debug\":false}" pilot2A_planner.log
echo "=== [3/7] variant A battery + snapshot ==="
SKIP_PLANNER=1 STAGES="$BATTERY" bash run_full_pipeline.sh > run_logs/pilot2A_battery.log 2>&1
rm -rf _ARCHIVE_V6_pilot2A; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot2A

echo "=== [4/7] restore base ==="
rm -rf "$ENRICH_RES"; cp -r _ARCHIVE_V6_prepilot2 "$ENRICH_RES"

echo "=== [5/7] variant B planner (min-duration + geo-fallback) ==="
run_planner "{${BASE_KW},\"transit_geo_fallback\":true,\"debug\":false}" pilot2B_planner.log
echo "=== [6/7] variant B battery + 3-way merge ==="
SKIP_PLANNER=1 STAGES="$BATTERY" bash run_full_pipeline.sh > run_logs/pilot2B_battery.log 2>&1
for i in $(seq 0 5); do $PY merge_soft.py --apply --donors _ARCHIVE_V6_prepilot2,_ARCHIVE_V6_pilot2A --shard "$i/6" > "run_logs/pilot2merge_sh$i.log" 2>&1 & done; wait
grep -h "swap" run_logs/pilot2merge_sh*.log | tail -8

echo "=== [7/7] FINAL eval + archive ==="
rm -rf _ARCHIVE_V6_pilot2; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot2
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== PILOT2 DONE ==="
