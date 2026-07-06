#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
N=10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pids=()
for i in $(seq 0 $((N-1))); do
  .venv/bin/python enrich_endday.py --apply --shard "$i/$N" > "run_logs/endday_sh$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $N endday workers: ${pids[*]}"
wait
echo "ALL ENDDAY WORKERS DONE"
