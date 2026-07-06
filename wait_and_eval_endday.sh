#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
for i in $(seq 1 720); do
  n=$(ps aux | grep '[e]nrich_endday.py' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && break
  sleep 20
done
echo "=== ENDDAY WORKERS DONE; alive: $(ps aux | grep '[e]nrich_endday.py' | wc -l|tr -d ' ') ==="
echo "=== per-shard summaries ==="
grep -h '^tried' run_logs/endday_sh*.log 2>/dev/null
echo "=== totals ==="
cat run_logs/endday_sh*.log 2>/dev/null | grep '^tried' | awk '{k+=$4; m+=$6} END{print "plans improved:",k,"  meals added:",m}'
echo "kept lines: $(cat run_logs/endday_sh*.log 2>/dev/null | grep -c 'meal | DDR')"
rm -rf _ARCHIVE_V6_endday_results
cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_endday_results
echo "=== snapshot -> _ARCHIVE_V6_endday_results ($(ls _ARCHIVE_V6_endday_results/*.json|wc -l|tr -d ' ') plans) ==="
echo "=== RUNNING EVAL ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -12
echo "=== EVAL DONE ==="
