#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
# wait for planner
for i in $(seq 1 900); do
  n=$(ps aux | grep '[r]un_tpc.py' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && break
  sleep 30
done
echo "=== PLANNER DONE; plans: $(ls results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation/*.json|wc -l|tr -d ' ') ==="
echo "=== FRESH compact eval ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
rm -rf _ARCHIVE_V6_compact_fresh; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_compact_fresh
echo "=== enrich: endday ==="; bash run_endday_parallel.sh >/dev/null 2>&1
echo "=== enrich: gapmeal ==="; bash run_gapmeal_parallel.sh >/dev/null 2>&1
echo "=== enrich: att ==="; bash run_att_parallel.sh >/dev/null 2>&1
rm -rf _ARCHIVE_V6_compact_enriched; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_compact_enriched
echo "=== FINAL compact+enriched eval ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== COMPACT CHAIN DONE ==="
