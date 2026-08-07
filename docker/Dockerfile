# TPC 2026 Phase-2 — organizer-equivalent test image for "Antarctic penguins_v2"
#
# Base = official SGLang matching the organizers' serving stack (0.5.10, CUDA).
# The harness gets its OWN venv so its pinned deps (torch 2.6 etc.) never touch
# SGLang's python environment. Model weights are NOT baked in — mount them:
#   -v /path/to/Qwen3.6-27B:/models/Qwen3.6-27B:ro
#
# Build (on any amd64 box with docker; the A100 server itself is easiest):
#   cd docker && docker build -t <you>/tpc2026-penguins-v2test:latest .
# Push:
#   docker login && docker push <you>/tpc2026-penguins-v2test:latest

FROM lmsysorg/sglang:v0.5.10.post1

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip curl \
 && rm -rf /var/lib/apt/lists/*

# --- the submission package under test (byte-identical to the upload) ------
COPY ["Antarctic penguins_v2.zip", "/tmp/submission.zip"]
RUN mkdir -p /workspace && cd /workspace \
 && unzip -q /tmp/submission.zip \
 && mv "Antarctic penguins_v2" submission \
 && rm /tmp/submission.zip

# --- isolated harness venv (CPU torch: the harness never needs CUDA) -------
RUN python3 -m venv /opt/harness-venv \
 && /opt/harness-venv/bin/pip install --no-cache-dir -U pip \
 && /opt/harness-venv/bin/pip install --no-cache-dir \
      torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && /opt/harness-venv/bin/pip install --no-cache-dir \
      -r /workspace/submission/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/submission
ENTRYPOINT ["/entrypoint.sh"]
# modes: test (default) | serve | bash
CMD ["test"]
