#!/usr/bin/env bash
# Parallel full Phase-1 run of UrbanTripOptimizedV5 (TPCLLM, oracle DSL).
# Round-robin shards the 1000 uids across N single-threaded workers.
# All workers write to the same results/<method>/ dir; --skip 1 skips done uids.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
N=${N:-12}
SPLITDIR=chinatravel/evaluation/default_splits
FULL="$SPLITDIR/TPC_IJCAI_2026_phase1.txt"

# Keep each worker single-threaded so 6 procs don't oversubscribe the BLAS pools.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# Build round-robin shard split files.
for i in $(seq 0 $((N-1))); do : > "$SPLITDIR/TPC_IJCAI_2026_phase1_sh$i.txt"; done
idx=0
while IFS= read -r uid || [ -n "$uid" ]; do
  [ -z "$uid" ] && continue
  echo "$uid" >> "$SPLITDIR/TPC_IJCAI_2026_phase1_sh$((idx % N)).txt"
  idx=$((idx+1))
done < "$FULL"
echo "sharded $idx uids across $N workers"

pids=()
for i in $(seq 0 $((N-1))); do
  .venv/bin/python run_tpc.py --splits "TPC_IJCAI_2026_phase1_sh$i" \
    --agent UrbanTripOptimizedV5 --llm TPCLLM --oracle_translation \
    --timeout 330 --lang en --skip 1 > "run_logs/v5_sh$i.log" 2>&1 &
  pids+=($!)
done
echo "launched workers: ${pids[*]}"
wait
echo "ALL WORKERS DONE; results=$(ls results/UrbanTripOptimizedV5_TPCLLM_en_oracletranslation/*.json 2>/dev/null | wc -l | tr -d ' ')/1000"
