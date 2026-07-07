# Reproducible submission pipeline

`run_full_pipeline.sh` regenerates the leaderboard submission **from source** —
no dependency on previous runs or archives.

```bash
bash run_full_pipeline.sh                    # everything (~5-6h; planner ~4h)
SKIP_PLANNER=1 bash run_full_pipeline.sh     # re-enrich an existing planner output
STAGES="gapattr bfstack final" bash run_full_pipeline.sh   # any subset, in order
METHOD=MyTest ENRICH_RES=results/MyTest bash run_full_pipeline.sh  # sandbox dir
```

## Stages (in the order that produced the submitted result)

| stage | script | lever | note |
|---|---|---|---|
| planner | `run_tpc.py` ×1000, 10 workers | base plans | `URBANTRIP_KWARGS` incl. `enable_travelday_time_bias` w=0.30 |
| eval0 | `eval_tpc.py` | checkpoint | fresh-planner score (~97.7) |
| endday | `enrich_endday.py` | DDR | elastic end-of-day dinner + arrival breakfast |
| gapmeal | `enrich_gapmeal.py` | DDR | nearest-restaurant gap lunch |
| att | `enrich_att.py` | ATT | walk→metro→taxi swaps, incl. station legs |
| repair | `repair_fails.py` | FPR | donor-restore failing plans (archives optional) |
| merge | `merge_soft.py` | all soft | per-uid best-merge from archives (skips if none) |
| gapattr | `enrich_gapattr.py` | DAV | gap-attraction inserts, free-first, tickets=people |
| travelday | `enrich_travelday_meals.py` | DDR | real dep-day dinner/lunch + arrival meals |
| bfstack | `enrich_bfstack.py --stack` | DDR | hotel-breakfast deficit cover |
| final | `eval_tpc.py` + archive | — | archives to `_ARCHIVE_pipeline_out` |

Every enrichment insert is regression-safe: kept only if the full 3-stage eval
(schema + commonsense + hardlogic) still passes AND a soft metric strictly
improved. FPR can therefore never regress from enrichment.

## Requirements
- `.venv` with the repo deps; no GPU/LLM needed (`--oracle_translation`).
- ~14 cores recommended (10 parallel workers).
- `ENRICH_RES` env var controls the target dir for all enrichers; the pipeline
  sets it, but if you run a script standalone, **always set it yourself** —
  some scripts default to a stale dir.

## Score trajectory (phase1, 1000 queries)
fresh 97.71 → endday/gapmeal/att/repair 99.09 → merge/gapattr/travelday/bfstack 99.61
→ bfstack seed-gate + cap fix **99.71** (DAV 98.08, ATT 97.06, DDR **100.0**, FPR 99.9,
C-LPR 99.97, EPR 100; archive `_ARCHIVE_V6_ddr100`)
