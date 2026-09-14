# Build from the repository root:
#   docker build -f docker/Dockerfile -t tpca-report-harness .
# Model serving runs separately in the report's SGLang image.
FROM python:3.12-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace/travel

RUN apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-harness-py312.txt ./
RUN pip install --no-cache-dir -r requirements-harness-py312.txt

# Include the current source and locally installed benchmark data. The root
# .dockerignore excludes logs, caches and local credentials, not the data.
COPY chinatravel/ ./chinatravel/
COPY agent_env/*.py ./agent_env/
COPY agent_env/scripts/*.py ./agent_env/scripts/
COPY agent_env/config.toml.tpcagent ./agent_env/config.toml.tpcagent
COPY scripts/validate_report* ./scripts/
COPY run_tpc.py eval_tpc.py eval_exp.py eval_with_oracle.py ./
COPY tech_report/ijcai26_official/ ./tech_report/ijcai26_official/
COPY docker/entrypoint.sh /entrypoint.sh

ENV HARNESS_PYTHON=python \
    CHINATRAVEL_LLM_THINK=0 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "/entrypoint.sh"]
CMD ["test"]
