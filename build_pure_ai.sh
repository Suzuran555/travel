#!/usr/bin/env bash
# PURE-AI VARIANT builder. Run AFTER chain_claude_run.sh completes (hybrid eval done).
# Hybrid = Claude plans + oracle-donor restores for eval-failing uids.
# Pure   = hybrid minus donor restores: those uids re-planned from Claude DSL only;
#          failures kept honestly. repair_fails' minimal-EDIT path is kept (it edits
#          the AI plan deterministically, no oracle content) — only DONOR swaps revert.
set -u
cd /Users/zhanggangyi/Desktop/TPC2026/travel
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
DIR="results/UrbanTripOptimizedV6_claude-cache_en"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"transit_signal_min_duration":true,"transit_geo_fallback":true,"transit_geo_feasibility":true,"transit_weight_floor":3.0,"debug":false}'

echo "=== [1/5] archive hybrid ==="
rm -rf _ARCHIVE_CLAUDE_hybrid; cp -r "$DIR" _ARCHIVE_CLAUDE_hybrid

echo "=== [2/5] revert donor-restored uids (delete -> re-plan from Claude DSL) ==="
grep 'restored donor' run_logs/claude_battery.log | awk -F: '{print $1}' | sort -u > run_logs/donor_restored_uids.txt
n=$(wc -l < run_logs/donor_restored_uids.txt | tr -d ' ')
echo "donor-restored uids: $n"
while read -r uid; do rm -f "$DIR/$uid.json"; done < run_logs/donor_restored_uids.txt
xargs -P 12 -I {} $PY run_tpc.py --splits "$SPLIT" --index {} \
  --agent UrbanTripOptimizedV6 --llm claude-cache --lang en --timeout 350 --skip 1 \
  < run_logs/donor_restored_uids.txt > run_logs/pure_replan.log 2>&1
echo "re-planned: $(ls "$DIR" | wc -l)/1000 present"

echo "=== [3/5] battery on pure source (NO repair stage) ==="
ENRICH_RES="$DIR" SKIP_PLANNER=1 \
  STAGES="endday gapmeal att gapattr endattr dav2 travelday bfstack" \
  bash run_full_pipeline.sh > run_logs/pure_battery.log 2>&1 || true

echo "=== [4/5] wave-4 fixers ==="
for i in $(seq 0 7); do ENRICH_RES="$DIR" $PY enrich_fillerswap.py --shard "$i/8" --apply > "run_logs/pure_filler_sh$i.log" 2>&1 & done; wait
for i in $(seq 0 7); do ENRICH_RES="$DIR" SEQSWAP_ALLOW_RESULTS=1 $PY enrich_seqswap.py --shard "$i/8" --apply > "run_logs/pure_seq_sh$i.log" 2>&1 & done; wait
ENRICH_RES="$DIR" $PY enrich_lateattr.py --apply > run_logs/pure_lateattr.log 2>&1 || true

echo "=== [5/5] eval PURE source + archive ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_claude-cache_en \
  --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
rm -rf _ARCHIVE_CLAUDE_pure; cp -r "$DIR" _ARCHIVE_CLAUDE_pure
echo "=== PURE AI BUILD DONE (hybrid archived in _ARCHIVE_CLAUDE_hybrid) ==="
