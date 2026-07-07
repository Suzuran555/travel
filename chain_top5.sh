#!/usr/bin/env bash
# Top-5 post-process chain on the 99.09 base (workflow top5-patch-builders, all reviews: ship).
# Order: merge donors in first, then enrich everything, bfstack --stack last (covers residual only).
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== [1/6] merge_soft (import better donors from 98.95 archive) ==="
N=6; for i in $(seq 0 $((N-1))); do $PY merge_soft.py --apply --shard "$i/$N" > "run_logs/mergesoft_sh$i.log" 2>&1 & done; wait
grep -ho "would swap.*\|swapped.*" run_logs/mergesoft_sh*.log | tail -6

echo "=== [2/6] enrich_att incl. station legs ==="
N=10; for i in $(seq 0 $((N-1))); do $PY enrich_att.py --apply --shard "$i/$N" > "run_logs/att_station_sh$i.log" 2>&1 & done; wait
grep -h "swaps" run_logs/att_station_sh*.log

echo "=== [3/6] enrich_gapattr ==="
for i in $(seq 0 9); do $PY enrich_gapattr.py --apply --shard "$i/10" > "run_logs/gapattr_sh$i.log" 2>&1 & done; wait
grep -h "^tried" run_logs/gapattr_sh*.log

echo "=== [4/6] enrich_travelday_meals ==="
bash run_travelday_parallel.sh > run_logs/travelday_master.log 2>&1
grep -h "tried" run_logs/travelday_sh*.log

echo "=== [5/6] enrich_bfstack --stack ==="
for i in $(seq 0 9); do $PY enrich_bfstack.py --apply --stack --shard "$i/10" > "run_logs/bfstack_sh$i.log" 2>&1 & done; wait
grep -h "^tried" run_logs/bfstack_sh*.log

rm -rf _ARCHIVE_V6_top5; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_top5
echo "=== [6/6] FINAL eval ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== TOP5 CHAIN DONE ==="
