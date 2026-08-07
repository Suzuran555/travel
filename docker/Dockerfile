# TPC 2026 Phase-2 — harness-only test image for "Antarctic penguins_v2"
#
# Two-image design: model serving uses the OFFICIAL lmsysorg/sglang image
# untouched (the organizers' stack, pulled directly on the server); this
# image carries only the submission package + its pinned deps (~2 GB).
# Run it with --network container:<sglang> so the package's hardcoded
# http://127.0.0.1:30000/v1 works with ZERO config overrides.
#
# Build:
#   cd docker && docker build -t <you>/tpc2026-penguins-v2test:latest .

FROM python:3.12-slim

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

# --- harness deps (CPU torch: the harness never needs CUDA) ----------------
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /workspace/submission/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/submission
ENTRYPOINT ["/entrypoint.sh"]
# modes: test (default) | bash
CMD ["test"]
