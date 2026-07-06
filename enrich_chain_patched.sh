#!/usr/bin/env bash
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
echo "=== [1/4] endday ==="; bash run_endday_parallel.sh
echo "=== [2/4] gapmeal ==="; bash run_gapmeal_parallel.sh
echo "=== [3/4] att swap ==="; bash run_att_parallel.sh
echo "=== [4/4] snapshot + eval ==="
rm -rf _ARCHIVE_V6_patched_enriched
cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_patched_enriched
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -12
echo "=== CHAIN EVAL DONE ==="
