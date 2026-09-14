# TPCAgent (UrbanTrip)

Antarctic penguins' Phase-2 travel-planning agent for the ChinaTravel benchmark.
A language model translates requests into constraints; the UrbanTrip planner
builds database-backed, multi-day itineraries and verifies repairs.
See the [technical report](tech_report/ijcai26_official/Antarctic%20penguins.tex).

## Requirements

- **Software:** Linux/WSL2, Bash, Python 3.12, and `requirements.txt`.
- **Data:** both `chinatravel/environment/database/` and `database_en/`;
  `phase2_familiar_EN/` and `phase2_heldout_sim_EN/` under
  `chinatravel/data/en/`, with matching files in
  `chinatravel/evaluation/default_splits/`. These datasets are not included
  in Git; supply them before running.
- **Local model serving:** Qwen3.6-27B through SGLang 0.5.10.post1.
  The report used two A800 GPUs (`TP=2`). Allow roughly 60 GB or more of GPU
  memory for BF16 serving, depending on context and concurrency; an 8 GB GPU
  cannot run that configuration.
- **Planner only:** runs on CPU, with no local GPU needed when the model is
  hosted elsewhere. A minimum system-RAM requirement has not been benchmarked.

## Start

Run from the repository root:

```bash
conda activate chinatravel
python --version                 # must be 3.12
python -m pip install -r requirements.txt
```

If that environment has an older Python, switch to a Python 3.12 environment
(such as `chinatravel312`) before installing and running.

With Qwen3.6-27B already served at `http://127.0.0.1:30000/v1`:

```bash
bash scripts/validate_report.sh
```

Or start SGLang and run the same workflow on a prepared GPU host:

```bash
MODEL_DIR=/data/models/Qwen3.6-27B TP=2 bash docker/run_single_instance.sh
```

The launcher requires existing model weights and a SGLang installation.
See [GPU/Docker setup](docker/README.md) for separate serving environments.
Validation runs all 100 simulation queries, then the official evaluator;
plans go to `results/`, with logs and snapshots in `validation_runs/`.

## Less powerful computers

**Use a remote model:** keep the planner on your computer and connect to a
Qwen3.6-27B server. This avoids the local GPU requirement:

```bash
CHINATRAVEL_OPENAI_BASE_URL="https://your-server.example/v1" \
  bash scripts/validate_report.sh
```

Set `CHINATRAVEL_OPENAI_API_KEY` if the server requires authentication.

**Try a smaller local model:** there is no separately validated Lite release,
but the client supports Ollama. After starting Ollama and installing a model
that fits your hardware, replace the placeholders below:

```bash
CHINATRAVEL_OPENAI_BASE_URL= OLLAMA_TAG="<installed-model-tag>" \
CHINATRAVEL_LLM_NAME="<unique-model-label>" \
  python run_tpc.py --index "<query-id>" --method small_model_trial_01_en
```

Choose an ID from `phase2_heldout_sim.txt` and a fresh method name for each
experiment. Smaller or quantized models may change translation quality and
runtime; this is an experimental option, not a reproduction of the report.
The report-validation script requires Qwen3.6-27B.
