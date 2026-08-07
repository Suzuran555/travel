#!/usr/bin/env bash
# Harness-only container. Serving runs in the OFFICIAL sglang container;
# share its network namespace (--network container:<name> or compose
# network_mode: "service:sglang") so 127.0.0.1:30000 resolves to it.
#
# Modes:
#   test (default) — wait for the server, run the 100-query held-out sim, score
#   bash           — interactive shell
set -uo pipefail

MODE="${1:-test}"
PORT="${PORT:-30000}"
SUB=/workspace/submission

wait_healthy() {
  echo "[entry] waiting for SGLang at 127.0.0.1:$PORT (model load can take minutes)..."
  for _ in $(seq 1 360); do          # up to 30 min
    if curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "Qwen3.6-27B"; then
      echo "[entry] server healthy."
      return 0
    fi
    sleep 5
  done
  echo "[entry] server never became healthy — is the sglang container running"
  echo "[entry] and does this container share its network namespace?"
  return 1
}

smoke() {
  local resp
  resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}')
  if grep -q "<think>" <<<"$resp"; then
    echo "[entry] WARNING: response contains <think> — thinking is ENABLED."
    echo "[entry] Fix the serving config (see docker/README.md) before trusting scores."
  else
    echo "[entry] thinking-off smoke test passed."
  fi
}

case "$MODE" in
  bash)
    exec bash
    ;;
  test)
    wait_healthy || exit 1
    smoke
    cd "$SUB"
    echo "[entry] === running the 100-query held-out simulation ==="
    python -u agent_env/scripts/solve_script_with_harness.py \
        --split phase2_heldout_sim --limit 100
    rc=$?
    echo "[entry] harness exit code: $rc (expected 0; 'Split summary skipped' is normal)"
    n=$(ls results/TPCAgent_Qwen3.6-27B_en/ 2>/dev/null | wc -l)
    echo "[entry] result files: $n/100"

    echo "[entry] === scoring against the oracle (phase2_familiar) ==="
    python -u eval_with_oracle.py eval_tpc \
        --splits phase2_familiar \
        --method TPCAgent_Qwen3.6-27B_en \
        --preference --lang en | tee /tmp/score.log

    if [ -d /output ]; then
      cp -r results /output/ 2>/dev/null
      cp /tmp/score.log /output/ 2>/dev/null
      echo "[entry] results + score log copied to /output"
    fi
    exit "$rc"
    ;;
  *)
    echo "unknown mode: $MODE (use test | bash)"; exit 2
    ;;
esac
