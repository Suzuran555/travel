#!/usr/bin/env bash
# WAVE 5 — last-sliver squeeze from 99.94990 to >99.95 (needs +0.02 ATT plan-units).
# The old battery + the new fixers re-run on the post-wave-4 geometry (rebooked
# airports, moved POIs) where fresh second-order opportunities exist. All stages
# are passes()-gated and atomic; snapshot taken first.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== [1/4] snapshot 99.9499 -> _ARCHIVE_V6_wave4 ==="
rm -rf _ARCHIVE_V6_wave4; cp -r "$ENRICH_RES" _ARCHIVE_V6_wave4

echo "=== [2/4] old battery on new geometry ==="
SKIP_PLANNER=1 STAGES="endday gapmeal att repair gapattr endattr dav2 travelday bfstack" bash run_full_pipeline.sh > run_logs/wave5_battery.log 2>&1

echo "=== [3/4] fillerswap + seqswap round 2 ==="
for i in $(seq 0 7); do $PY enrich_fillerswap.py --shard "$i/8" --apply > "run_logs/wave5_filler_sh$i.log" 2>&1 & done; wait
for i in $(seq 0 7); do SEQSWAP_ALLOW_RESULTS=1 $PY enrich_seqswap.py --shard "$i/8" --apply > "run_logs/wave5_seq_sh$i.log" 2>&1 & done; wait

echo "=== [4/4] FINAL eval ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== WAVE5 DONE ==="
