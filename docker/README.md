# Run the technical-report configuration

These helpers run the current checkout's TPCAgent. The reference is
`tech_report/ijcai26_official/Antarctic penguins.tex`: Qwen3.6-27B served by
SGLang 0.5.10.post1, thinking disabled, and two A800 GPUs with tensor
parallelism two. Its reported familiarization Overall score was 97.70;
another machine or run is not assumed to reproduce that number.

The validation entrypoint is `scripts/validate_report.sh`. It checks the
runtime, runs all 100 `phase2_heldout_sim` requests with fresh translations,
and scores the completed results against `phase2_familiar` through
`eval_with_oracle.py tpc`, which invokes the unchanged `eval_tpc.py`.
Each validation uses a new method and preserves its result snapshot.

## Existing serving endpoint

From the repository root, activate the project's harness environment:

```bash
conda activate chinatravel
bash scripts/validate_report.sh --preflight-only
bash scripts/validate_report.sh
```

The default endpoint is `http://127.0.0.1:30000/v1`. The endpoint must serve
the model under the exact name `Qwen3.6-27B`. The harness sends thinking-off
requests. `HARNESS_PYTHON` can select a Python executable from another
environment; it is an executable path, not a multi-word shell command.

## One GPU-server instance

`run_single_instance.sh` reuses a healthy local endpoint or starts SGLang
from an already prepared serving environment and checkpoint. It checks
GPU availability, capacity, SGLang's version and `config.json` before
launching. It does not install packages or download model weights.

```bash
conda activate chinatravel
MODEL_DIR=/data/models/Qwen3.6-27B \
  SGLANG_PYTHON=/opt/sglang/bin/python \
  HARNESS_PYTHON="$CONDA_PREFIX/bin/python" \
  TP=2 bash docker/run_single_instance.sh
```

On a platform whose base image is `lmsysorg/sglang:v0.5.10.post1`, set
`SGLANG_PYTHON` to that image's Python. The BF16 checkpoint needs about
60 GB for weights plus runtime headroom; a single 8 GB consumer GPU is
insufficient. `TP` changes the GPU count, not the model's precision.
The script logs a server it starts under `/tmp/tpca-sglang.*/sglang.log`
and stops that server when validation ends. An existing server is reused.

Other settings: `PORT` defaults to 30000 and `SGLANG_START_TIMEOUT` to
1800 seconds. `--preflight-only` delegates to the validation checks without
starting a server. Use `--help` for the launcher options.

## Docker Compose

Install Docker Compose and NVIDIA Container Toolkit on a GPU host. Place
the checkpoint in an existing host directory. The local checkout must also
contain the benchmark databases and the two familiarization query splits;
these are included in the harness build even though they are ignored by Git.

From the repository root:

```bash
MODEL_DIR=/data/models/Qwen3.6-27B TP=2 \
  docker compose -f docker/docker-compose.yml up --build \
    --abort-on-container-exit --exit-code-from harness harness
```

Compose uses the pinned SGLang image for serving and builds the harness
image from this checkout with Python 3.12 and the current lightweight
requirements. It waits for serving health before starting the harness.
The harness shares the serving container's network namespace and writes
results, evaluation artifacts, run logs, caches and `validation_runs/`
snapshots to the corresponding host directories. It does not load an old
submission ZIP or published harness image.

Build the harness alone, without starting a model:

```bash
docker build -f docker/Dockerfile -t tpca-report-harness:local .
```

If serving fails, inspect `docker compose -f docker/docker-compose.yml logs
sglang`. Check the checkpoint path, available GPU memory and `TP`. A run
that cannot translate requests because serving is unavailable does not
validate the report's end-to-end system.
