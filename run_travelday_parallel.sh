#!/usr/bin/env bash
# Travel-day REAL meal enrichment (DDR): 10 sharded workers over affected uids only.
# The script itself pre-filters to plans with fixable travel-day slots and shards
# over THAT list, so each worker only touches its own subset.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
N=10
export ENRICH_RES=results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pids=()
for i in $(seq 0 $((N-1))); do
  .venv/bin/python enrich_travelday_meals.py --apply --shard "$i/$N" > "run_logs/travelday_sh$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $N travelday workers: ${pids[*]}"
wait
echo "ALL TRAVELDAY WORKERS DONE"
grep -h "^tried" run_logs/travelday_sh*.log
