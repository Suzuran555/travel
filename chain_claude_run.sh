#!/usr/bin/env bash
# CLAUDE-TRANSLATED AI SOURCE - full-speed chain.
# Syncs workflow outputs into the Claude translation cache as they land,
# plans each uid incrementally at full CPU width (no LLM in RAM), then runs
# the battery + wave-4 fixers and the official eval.
# Source dir: results/UrbanTripOptimizedV6_claude-cache_en (fully separate).
cd /Users/zhanggangyi/Desktop/TPC2026/travel
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
WFOUT="/private/tmp/claude-501/-Users-zhanggangyi-Desktop-TPC2026/ccadc8f1-a9ae-4981-8039-90c33e359ae2/scratchpad/claude_nl2dsl/out"
TCACHE="cache/translation_Claude_reflect"
AIRES="results/UrbanTripOptimizedV6_claude-cache_en"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export URBANTRIP_KWARGS='{"use_bundle_search":true,"geo_anchor_must":true,"enable_fallback_hard_repair":true,"enable_metro_only_prune":true,"enable_dynamic_top_k":true,"enable_dfs_memoization":true,"enable_budget_drop_repair":true,"enable_travelday_time_bias":true,"travelday_arr_weight":0.30,"travelday_dep_weight":0.30,"enable_transit_time_score":true,"transit_signal_min_duration":true,"transit_geo_fallback":true,"transit_geo_feasibility":true,"transit_weight_floor":3.0,"debug":false}'
mkdir -p "$AIRES" "$TCACHE" run_logs

echo "=== [1/4] incremental sync + full-width planning ==="
stall=0
while true; do
  cp -n "$WFOUT"/*.json "$TCACHE"/ 2>/dev/null
  t=$(ls "$TCACHE" 2>/dev/null | wc -l | tr -d ' ')
  p=$(ls "$AIRES" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%H:%M') translated=$t planned=$p"
  if [ "$t" -ge 1000 ] && [ "$p" -ge 1000 ]; then break; fi
  todo=$(comm -23 \
    <(ls "$TCACHE" | sed 's/\.json$//' | sort) \
    <(ls "$AIRES" 2>/dev/null | sed 's/\.json$//' | sort) | head -600)
  if [ -n "$todo" ]; then
    stall=0
    echo "$todo" | xargs -P 12 -I {} "$PY" run_tpc.py --splits "$SPLIT" \
      --index {} --agent UrbanTripOptimizedV6 --llm claude-cache \
      --lang en --timeout 350 --skip 1 >> run_logs/claude_planner.log 2>&1
  else
    stall=$((stall+1))
    if [ "$stall" -gt 60 ]; then echo "STALLED: no new translations for 60 cycles"; break; fi
    sleep 60
  fi
done

echo "=== [2/4] enrichment battery on the Claude source ==="
ENRICH_RES="$AIRES" SKIP_PLANNER=1 \
  STAGES="endday gapmeal att repair gapattr endattr dav2 travelday bfstack" \
  bash run_full_pipeline.sh > run_logs/claude_battery.log 2>&1 || true

echo "=== [3/4] wave-4 fixers ==="
for i in $(seq 0 7); do ENRICH_RES="$AIRES" $PY enrich_fillerswap.py --shard "$i/8" --apply > "run_logs/claude_filler_sh$i.log" 2>&1 & done; wait
for i in $(seq 0 7); do ENRICH_RES="$AIRES" SEQSWAP_ALLOW_RESULTS=1 $PY enrich_seqswap.py --shard "$i/8" --apply > "run_logs/claude_seq_sh$i.log" 2>&1 & done; wait
ENRICH_RES="$AIRES" $PY enrich_lateattr.py --apply > run_logs/claude_lateattr.log 2>&1 || true

echo "=== [4/4] official eval (oracle constraints) ==="
$PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_claude-cache_en \
  --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== CLAUDE AI RUN DONE ==="
