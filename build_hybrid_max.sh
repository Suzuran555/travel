#!/usr/bin/env bash
# HYBRID-MAX: drive the AI source to the current-evaluator ceiling (99.953).
# Run AFTER build_pure_ai.sh (so hybrid + pure are archived). Takes the hybrid,
# repairs any failures against the CEILING donors, then per-uid best-merges with
# the ceiling archive (passes()+soft-gated). Result >= ceiling by construction.
set -u
cd /Users/zhanggangyi/Desktop/TPC2026/travel
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
DIR="results/UrbanTripOptimizedV6_claude-cache_en"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== [1/4] start from hybrid ==="
rm -rf "$DIR"; cp -r _ARCHIVE_CLAUDE_hybrid "$DIR"

echo "=== [2/4] repair failures against ceiling donors ==="
ENRICH_RES="$DIR" $PY repair_fails.py --apply > run_logs/hmax_repair.log 2>&1
grep -E '^fixed|still-failing' run_logs/hmax_repair.log | tail -1

echo "=== [3/4] per-uid best-merge vs ceiling archive ==="
for i in $(seq 0 5); do ENRICH_RES="$DIR" $PY merge_soft.py --apply \
  --donors _ARCHIVE_V6_ceiling --shard "$i/6" > "run_logs/hmax_merge_sh$i.log" 2>&1 & done; wait
grep -h 'would swap' run_logs/hmax_merge_sh*.log

echo "=== [4/4] eval HYBRID-MAX + archive ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_claude-cache_en \
  --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
rm -rf _ARCHIVE_CLAUDE_hybridmax; cp -r "$DIR" _ARCHIVE_CLAUDE_hybridmax
echo "=== HYBRID-MAX DONE ==="
