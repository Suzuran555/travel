#!/usr/bin/env bash
# PILOT 3 RESUME after power-off (2026-07-08): stages 3-5 of chain_pilot3.sh.
# Stage 1 (prepilot3 backup, 158 targets) and stage 2 (variant C planner) completed
# before the outage; live dir verified 1000 valid JSONs, all targets present.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
BATTERY="endday gapmeal att repair gapattr endattr dav2 travelday bfstack"

echo "=== [3/5] checkpoint planned state + battery (resume) ==="
rm -rf _ARCHIVE_V6_pilot3_planned; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot3_planned
SKIP_PLANNER=1 STAGES="$BATTERY" bash run_full_pipeline.sh > run_logs/pilot3_battery.log 2>&1

echo "=== [4/5] gated merge vs prepilot3 ==="
for i in $(seq 0 5); do $PY merge_soft.py --apply --donors _ARCHIVE_V6_prepilot3 --shard "$i/6" > "run_logs/pilot3merge_sh$i.log" 2>&1 & done; wait
grep -h "swap" run_logs/pilot3merge_sh*.log | tail -6

echo "=== [5/5] FINAL eval + archive ==="
rm -rf _ARCHIVE_V6_pilot3; cp -r "$ENRICH_RES" _ARCHIVE_V6_pilot3
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== PILOT3 RESUME DONE ==="
