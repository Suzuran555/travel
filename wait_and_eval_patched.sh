#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
for i in $(seq 1 900); do
  n=$(ps aux | grep '[r]un_tpc.py' | wc -l | tr -d ' ')
  [ "$n" -eq 0 ] && break
  sleep 30
done
echo "=== PATCHED PLANNER DONE; run_tpc alive: $(ps aux | grep '[r]un_tpc.py' | wc -l|tr -d ' ') ==="
echo "plans in dir: $(ls results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation/*.json 2>/dev/null | wc -l|tr -d ' ')"
rm -rf _ARCHIVE_V6_patched_fresh
cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_patched_fresh
echo "=== snapshot fresh patched -> _ARCHIVE_V6_patched_fresh ==="
echo "=== EVAL FRESH PATCHED (compare to baseline 96.86: DDR 67.35 DAV 89.86 ATT 88.41 FPR 99.4) ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -12
echo "=== FRESH EVAL DONE ==="
