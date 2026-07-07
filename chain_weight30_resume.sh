#!/usr/bin/env bash
# Resume of chain_weight30.sh after session break: steps 1-3 done
# (planner fresh eval 97.71, endday complete, gapmeal killed ~2min in).
cd /Users/zhanggangyi/Desktop/TPC2026/travel
export ENRICH_RES="results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation"
PY=.venv/bin/python
SPLIT="TPC_IJCAI_2026_phase1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "=== [4/6] gapmeal (resume) ==="; bash run_gapmeal_parallel.sh >/dev/null 2>&1
echo "=== [5/6] att(metro) + repair ==="; bash run_att_parallel.sh >/dev/null 2>&1; $PY repair_fails.py --apply 2>&1 | grep -E "restored|applied|^fixed"
rm -rf _ARCHIVE_V6_weight30; cp -r results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation _ARCHIVE_V6_weight30
echo "=== [6/6] FINAL eval ==="; $PY eval_tpc.py --splits "$SPLIT" --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 | grep -viE "it/s|%\|" | tail -3
echo "=== WEIGHT30 RESUME DONE ==="
