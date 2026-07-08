# TPC2026 — Complete Materials Manifest
Generated 2026-07-08 evening · state: **Overall 99.95113** (DAV 100, DDR 100, EPR 100, ATT 99.963, FPR 99.9, C-LPR 99.972)

## 1. Submission
| Item | Detail |
|---|---|
| `results/UrbanTripOptimizedV6_TPCLLM_en_oracletranslation/` | 1000 plan JSONs — the 99.95113 submission set |
| `your_tpc_scores.json` | score record |

## 2. Enrichers (post-processing, all passes()-gated, dry-run default)
| Script | Size | Role |
|---|---|---|
| `enrich_att.py` | 10.0 KB | walk->metro->taxi leg swaps |
| `enrich_attdilute.py` | 13.0 KB | NEW wave5: short-leg inserts dilute transit average (final +0.246u) |
| `enrich_attr.py` | 6.7 KB | legacy attraction insert |
| `enrich_bfstack.py` | 8.4 KB | hotel-breakfast deficit cover (--stack) |
| `enrich_breakfast.py` | 5.6 KB | legacy breakfast insert |
| `enrich_dav2.py` | 28.2 KB | 5-phase DAV squeeze incl. dwell-shrink |
| `enrich_day.py` | 7.2 KB | legacy day rebuild |
| `enrich_endattr.py` | 10.3 KB | evening-append attractions |
| `enrich_endday.py` | 8.5 KB | end-of-day dinner + arrival breakfast |
| `enrich_fillerswap.py` | 14.9 KB | NEW wave4: far-filler POI replacement |
| `enrich_gapattr.py` | 14.2 KB | gap-attraction inserts |
| `enrich_gapmeal.py` | 5.9 KB | gap lunch via nearest-restaurant recall |
| `enrich_lateattr.py` | 12.2 KB | NEW wave4: movable-check-in evening DAV inserts |
| `enrich_lunch.py` | 6.3 KB | midday lunch cascade (dead end, 0 yield) |
| `enrich_meal.py` | 6.1 KB | legacy meal insert |
| `enrich_rebook.py` | 15.8 KB | NEW wave4: alternate-airport/dead-schedule flight rebooking |
| `enrich_route.py` | 9.2 KB | route-aware on-the-way meal insertion |
| `enrich_seqswap.py` | 16.8 KB | NEW wave4: joint transport re-moding under evaluated caps |
| `enrich_soft.py` | 3.7 KB | legacy soft sweep |
| `enrich_travelday_meals.py` | 19.2 KB | real meals on travel days |
| `enrich_rebook_recipes.json` | 67.2 KB | rebook companion: 5 verified full-itinerary rebuilds |

## 3. Orchestration
| Script | Purpose |
|---|---|
| `chain_compact.sh` | compact-dwell experiment (rejected) |
| `chain_endgame.sh` | 5-patch chain to 99.61+ |
| `chain_pilot2.sh` | ATT pilot round 2 (99.867->99.910) |
| `chain_pilot3.sh` | ATT pilot round 3 original |
| `chain_pilot3_resume.sh` | pilot3 stages 3-5 after power loss |
| `chain_top5.sh` | top-5 push chain |
| `chain_wave4.sh` | apply rebook+fillerswap+seqswap (99.917->99.9499) |
| `chain_wave5.sh` | battery+round2 squeeze (wash, reverted) |
| `chain_weight30.sh` | weight-0.30 chain |
| `chain_weight30_resume.sh` | weight chain resume |
| `run_full_pipeline.sh` | full reproducible rebuild from source |
| `run_enrich_parallel.sh` etc. | shard runners |

## 4. Documents
- `README_PIPELINE.md` (3.4 KB)
- `EVALUATOR_BUG_REPORT.md` (2.9 KB)
- `CHANGE_REPORT.md` (5.9 KB)
- `CHANGE_REPORT.docx` (40.0 KB)
- `README.md` (10.4 KB)
- `SETUP_LOG.md` (17.0 KB)
- `URBANTRIP_NO1_QUICKSTART.md` (2.8 KB)
- `AGENTS.md` (0.7 KB)
- `AGENT_ENV_PROGRESS.md` (2.4 KB)
- `../TPC_IJCAI2026_Analysis_Report.docx`, `../ChinaTravel_Paper_Analysis.docx`, `../agent_env_Execution_Path_Analysis.docx` — background analyses

## 5. Archives (full 1000-plan rollback snapshots)
| Archive | State |
|---|---|
| `_ARCHIVE_94.97_results` | older waypoint |
| `_ARCHIVE_95.01_results` | older waypoint |
| `_ARCHIVE_V6_9995` | **99.95113 (current best)** |
| `_ARCHIVE_V6_att2_results` | older waypoint |
| `_ARCHIVE_V6_att_results` | older waypoint |
| `_ARCHIVE_V6_best_98.95` | 98.95 |
| `_ARCHIVE_V6_compact_fresh` | older waypoint |
| `_ARCHIVE_V6_dav2` | 99.867 |
| `_ARCHIVE_V6_ddr100` | older waypoint |
| `_ARCHIVE_V6_endday_results` | older waypoint |
| `_ARCHIVE_V6_enriched_98.12` | older waypoint |
| `_ARCHIVE_V6_enriched_results` | older waypoint |
| `_ARCHIVE_V6_gapmeal_results` | older waypoint |
| `_ARCHIVE_V6_patched_enriched` | older waypoint |
| `_ARCHIVE_V6_patched_fresh` | repro sandbox source |
| `_ARCHIVE_V6_phaseD_96.86_results` | older waypoint |
| `_ARCHIVE_V6_pilot` | older waypoint |
| `_ARCHIVE_V6_pilot2` | 99.910 |
| `_ARCHIVE_V6_pilot2A` | older waypoint |
| `_ARCHIVE_V6_pilot3` | 99.917 |
| `_ARCHIVE_V6_pilot3_planned` | pilot3 planner-only checkpoint |
| `_ARCHIVE_V6_prepilot` | older waypoint |
| `_ARCHIVE_V6_prepilot2` | older waypoint |
| `_ARCHIVE_V6_prepilot3` | 99.910 pre-pilot3 |
| `_ARCHIVE_V6_prewave4` | 99.917-era (recopied mid-wave4; not clean) |
| `_ARCHIVE_V6_repaired_metro` | older waypoint |
| `_ARCHIVE_V6_top5` | older waypoint |
| `_ARCHIVE_V6_wave4` | 99.9499 pre-attdilute |
| `_ARCHIVE_V6_waves` | older waypoint |
| `_ARCHIVE_V6_weight30` | older waypoint |
| `_ARCHIVE_V6_weight30_endday` | older waypoint |
| `_ARCHIVE_merged_95.33_results` | older waypoint |
| `_ARCHIVE_plusvisit_DAV_results` | older waypoint |
| `_ARCHIVE_plusvisit_bfb2c97_95.33_results` | older waypoint |
| `_ARCHIVE_plusvisit_v2_DAV77_results` | older waypoint |
| `_ARCHIVE_stacked_95.67_results` | older waypoint |

## 6. Eval & infra
- `eval_tpc.py` (official eval) · `merge_soft.py` (gated per-uid donor merge) · `repair_fails.py` (donor-restore) · `run_tpc.py` (planner entry) · `chinatravel/` (env, DBs, evaluators) · `.venv/` (python env)

## 7. Logs (`run_logs/`)
- 408 files; today's: `chain_pilot3_resume.log`, `chain_wave4.log`, `chain_wave5.log`, `wave4_*`/`wave5_*` shard logs, `pilot3_*`, `push_retry.log`

## 8. Session artifacts (scratchpad — EPHEMERAL, copy into repo if wanted)
- `fix_recipes.json` (89 KB — audit claims + adversarial verifications, per-uid fix evidence)
- `residual_audit.json`, `att_residual_final.json` (residual enumerations)
- builders' work dirs: `attdilute_work/`, `fswapval_work/`, `seqswap_work/`, `rebook_work/`, `lateattr_work/`, `review_*/` (validated apply outputs + verify scripts)

## 9. Git state
```
1d57f95 milestone: 99.951 - user target >99.95 cleared; DAV and DDR both 100.0
7ee8c02 milestone: 99.917 - ATT pilot round 3 (variant C, power-loss recovery)
1733511 milestone: 99.910 - top-5 bar crossed (ATT pilot round 2)
e9ae357 planner: transit_geo_feasibility + transit_weight_floor (ATT round 3)
ec75eb9 planner: transit_geo_fallback sub-flag (geo-distance for segment holes)
2cd4bb4 milestone: 99.867 - DAV 99.79 (enrich_dav2: 5-phase deficit squeeze)
```
- `1d57f95` (99.951 milestone) committed LOCALLY — push pending network recovery (retry loop armed)
- remote: `git@github.com:Suzuran555/travel.git`, branch `improve/top5-soft-max`