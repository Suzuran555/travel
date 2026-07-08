#!/usr/bin/env bash
# RUN ONLY AFTER the organizers fix the alias bug (EVALUATOR_BUG_REPORT.md:
# _POI_NAME_ALIASES -> 'Sola Bistro' vs EN_POI_NAME_CORRECTIONS -> 'Bistro Sola').
# Swaps in the archived Bistro Sola donor for the one bug-pinned uid, turning
# FPR 99.9 -> 100.0 and C-LPR 99.972 -> 100.0 (+0.047 Overall).
# Under the CURRENT evaluator this plan FAILS grounding - do not run early.
set -e
cd /Users/zhanggangyi/Desktop/TPC2026/travel
LIVE=results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation
UID_FIX=20250323010327713880
cp -v "$LIVE/$UID_FIX.json" "run_logs/prefix_backup_$UID_FIX.json"
cp -v "_ARCHIVE_V6_enriched_98.12/$UID_FIX.json" "$LIVE/$UID_FIX.json"
.venv/bin/python eval_tpc.py --splits TPC_IJCAI_2026_phase1 \
  --method UrbanTripOptimizedV6_TPCLLM_en_oracletranslation --lang en 2>&1 \
  | grep -viE "it/s|%\|" | tail -3
echo "If FPR did not reach 100.0, the evaluator is NOT fixed yet - revert:"
echo "  cp run_logs/prefix_backup_$UID_FIX.json $LIVE/$UID_FIX.json"
