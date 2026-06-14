# UrbanTrip No.1 Quickstart

This checkout is configured to run the UrbanTrip-style top solution through the
official ChinaTravel TPC scripts.

## Environment

Use the existing conda environment, not the shell's default Python:

```bash
source /home/elljames/miniconda3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/elljames/travel
python --version
```

Expected Python is `3.9.x`. The default shell Python is currently `3.13.x` and
does not have the benchmark dependencies installed.

Quick dependency and database checks:

```bash
python -c "import geopy, numpy, pandas, jsonschema, func_timeout; print('deps-ok')"
python -m agent_env.cli call attractions_keys '{"city":"上海"}'
python -c "from chinatravel.environment.world_env import WorldEnv; WorldEnv(lang='en'); print('world-ok')"
```

The required local data is already present under:

- `chinatravel/environment/database/`
- `chinatravel/environment/database_en/`
- `chinatravel/data/en/TPC_IJCAI_2026_phase1_EN/`
- `chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1.txt`

## Smoke Run

Run one official Phase 1 English query with the no.1 local solution variant:

```bash
python run_tpc.py \
  --splits TPC_IJCAI_2026_phase1_smoke3 \
  --index 20250320174446059265 \
  --agent UrbanTripOptimizedV2 \
  --llm TPCLLM \
  --oracle_translation \
  --timeout 300 \
  --lang en
```

Evaluate the smoke split:

```bash
python eval_tpc.py \
  --splits TPC_IJCAI_2026_phase1_smoke3 \
  --method UrbanTripOptimizedV2_TPCLLM_en_oracletranslation \
  --lang en
```

Or use the helper:

```bash
scripts/run_urbantrip_no1_smoke.sh
```

The smoke run should end with `Pass!`; the current smoke3 evaluation is:

```text
Mic.EPR 100.0
Mac.EPR 100.0
C-LPR 100.0
FPR 100.0
Overall Score 97.93439153439154
```

## Full Phase 1 Run

The full 1000-query result set has already been generated at:

```text
results/UrbanTripOptimizedV2_TPCLLM_en_oracletranslation/
```

The existing submission package is:

```text
submission_full_v2/UrbanTripOptimizedV2_phase1.zip
```

The recorded full evaluation is:

```text
MicEPR 99.88
MacEPR 97.0
C-LPR 94.78699551569507
FPR 87.3
Overall 90.11031559339716
```

To regenerate all 1000 queries serially:

```bash
python run_tpc.py \
  --splits TPC_IJCAI_2026_phase1 \
  --agent UrbanTripOptimizedV2 \
  --llm TPCLLM \
  --oracle_translation \
  --timeout 300 \
  --lang en \
  --skip 1
```

To evaluate all 1000 generated plans:

```bash
python eval_tpc.py \
  --splits TPC_IJCAI_2026_phase1 \
  --method UrbanTripOptimizedV2_TPCLLM_en_oracletranslation \
  --lang en
```

`TPCLLM` is an empty local LLM stub for this solution path, so no
`OPENAI_API_KEY` is required. `--oracle_translation` is needed for these
UrbanTrip implementations because they read the official `hard_logic_py` DSL
constraints during planning.
