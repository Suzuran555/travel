#!/usr/bin/env bash
# ENDGAME: DAV/ATT waves -> checkpoint -> ATT planner pilot (flag-on re-plan of all
# ATT<1 uids) -> idempotent battery -> per-uid gated merge vs prepilot -> final eval.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
SCRATCH="/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/84a5018c-680f-44fc-88e1-0f42be32ced7/scratchpad"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== [1/9] gapattr v2 ==="
for i in $(seq 0 9); do $PY enrich_gapattr.py --apply --shard "$i/10" > "run_logs/gapattr_v2_sh$i.log" 2>&1 & done; wait
grep -h "^tried" run_logs/gapattr_v2_sh*.log

echo "=== [2/9] endattr ==="
for i in $(seq 0 9); do $PY enrich_endattr.py --apply --shard "$i/10" > "run_logs/endattr_sh$i.log" 2>&1 & done; wait
grep -h "tried" run_logs/endattr_sh*.log

echo "=== [3/9] att taxi-first sweep ==="
for i in $(seq 0 9); do $PY enrich_att.py --apply --shard "$i/10" > "run_logs/att_swap2_sh$i.log" 2>&1 & done; wait
grep -h "swaps" run_logs/att_swap2_sh*.log

echo "=== [4/9] checkpoint eval (post-waves) ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
rm -rf _ARCHIVE_V6_waves; cp -r "$ENRICH_RES" _ARCHIVE_V6_waves

echo "=== [5/9] refresh ATT pilot targets ==="
sed "s|_ARCHIVE_V6_top5|results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation|" "$SCRATCH/att_top5_analysis.py" > "$SCRATCH/att_refresh.py"
$PY "$SCRATCH/att_refresh.py" > run_logs/att_refresh.log 2>&1
$PY -c "import json; [print(t['uid']) for t in json.load(open('$SCRATCH/att_pilot_targets.json')) if t['uid'] != '20250323010327713880']" > pilot_uids.txt
echo "pilot targets: $(wc -l < pilot_uids.txt)"

echo "=== [6/9] prepilot backup ==="
rm -rf _ARCHIVE_V6_prepilot; cp -r "$ENRICH_RES" _ARCHIVE_V6_prepilot

echo "=== [7/9] pilot planner (enable_transit_time_score) ==="
( export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"debug":false}'
  cat pilot_uids.txt | xargs -P 8 -I {} $PY run_tpc.py --splits "$SPLIT" --index {} --agent UrbanTripOptimizedV6 --llm TPCLLM --lang en --oracle_translation --timeout 350 --skip 0 > run_logs/pilot_planner.log 2>&1 )
echo "planner done"

echo "=== [8/9] battery re-enrichment (idempotent) + gated merge vs prepilot ==="
SKIP_PLANNER=1 STAGES="endday gapmeal att repair gapattr travelday bfstack" bash run_full_pipeline.sh > run_logs/pilot_battery.log 2>&1
tail -20 run_logs/pilot_battery.log
for i in $(seq 0 5); do $PY merge_soft.py --apply --donors _ARCHIVE_V6_prepilot --shard "$i/6" > "run_logs/pilotmerge_sh$i.log" 2>&1 & done; wait
grep -h "swap" run_logs/pilotmerge_sh*.log | tail -8

echo "=== [9/9] FINAL eval + archive ==="
rm -rf _ARCHIVE_V6_pilot; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== ENDGAME DONE ==="
