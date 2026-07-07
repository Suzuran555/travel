#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
echo "=== metro ATT swap on repaired base ==="; bash run_att_parallel.sh >/dev/null 2>&1
rm -rf _ARCHIVE_V6_repaired_metro; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_repaired_metro
echo "=== EVAL (98.56 + repair + metro) ==="
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== FINAL EVAL DONE ==="
