#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
# 1) wait for all enrich workers to exit (cap ~3.5h)
for i in $(seq 1 630); do
  n=$(ps aux | grep '[e]nrich_route.py' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && break
  sleep 20
done
echo "=== WORKERS DONE (or cap hit); alive now: $(ps aux | grep '[e]nrich_route.py' | wc -l|tr -d ' ') ==="
echo "=== per-shard summaries ==="
grep -h '^tried' run_logs/enrich_sh*.log 2>/dev/null
echo "=== totals ==="
cat run_logs/enrich_sh*.log 2>/dev/null | grep '^tried' | awk '{k+=$4; m+=$8} END{print "plans improved:",k,"  meals added:",m}'
echo "kept lines total: $(cat run_logs/enrich_sh*.log 2>/dev/null | grep -c 'meal | DAV')"
# 2) snapshot enriched V6
rm -rf _ARCHIVE_V6_enriched_results
cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_enriched_results
echo "=== snapshot -> _ARCHIVE_V6_enriched_results ($(ls _ARCHIVE_V6_enriched_results/*.json|wc -l|tr -d ' ') plans) ==="
# 3) full official eval
echo "=== RUNNING EVAL ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -40
echo "=== EVAL DONE ==="
