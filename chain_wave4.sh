#!/usr/bin/env bash
# WAVE 4 — apply the three reviewed residual fixers to the live dir, sequentially.
# Order matters: rebook (flights/structure) -> fillerswap (POI re-selection) ->
# seqswap (transport re-moding on the final geometry). Each is passes()-gated
# per plan with atomic writes, so any interruption leaves only valid plans.
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== [1/4] backup -> _ARCHIVE_V6_prewave4 ==="
rm -rf _ARCHIVE_V6_prewave4; cp -r "$ENRICH_RES" _ARCHIVE_V6_prewave4

echo "=== [2/4] rebook ==="
$PY enrich_rebook.py --apply > run_logs/wave4_rebook.log 2>&1

echo "=== [3/4] fillerswap (8 shards) ==="
for i in $(seq 0 7); do $PY enrich_fillerswap.py --shard "$i/8" --apply > "run_logs/wave4_filler_sh$i.log" 2>&1 & done; wait

echo "=== [4/4] seqswap (8 shards) ==="
for i in $(seq 0 7); do SEQSWAP_ALLOW_RESULTS=1 $PY enrich_seqswap.py --shard "$i/8" --apply > "run_logs/wave4_seq_sh$i.log" 2>&1 & done; wait

echo "=== intermediate eval ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== WAVE4 (3 fixers) DONE ==="
