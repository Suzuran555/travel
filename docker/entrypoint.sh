#!/usr/bin/env bash
# Modes:
#   test  (default) — serve Qwen3.6-27B, run the 100-query held-out sim, score it
#   serve           — just serve the model (organizer surface on :30000)
#   bash            — interactive shell
set -uo pipefail

MODE="${1:-test}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
TP="${TP:-2}"
PORT="${PORT:-30000}"
PY=/opt/harness-venv/bin/python
SUB=/workspace/submission

SERVER_PID=""

start_server() {
  # args passed as array to survive the JSON braces
  local -a extra=("$@")
  echo "[entry] launching SGLang: tp=$TP model=$MODEL_PATH ${extra[*]:-}"
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name Qwen3.6-27B \
    --host 127.0.0.1 --port "$PORT" --tp "$TP" \
    "${extra[@]}" \
    >/tmp/sglang.log 2>&1 &
  SERVER_PID=$!
}

wait_healthy() {
  local tries="${1:-240}"          # 240 x 5s = 20 min for model load
  for _ in $(seq 1 "$tries"); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "Qwen3.6-27B"; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1                      # server process died
    fi
    sleep 5
  done
  return 1
}

boot() {
  # organizers serve with thinking DISABLED; try the server-side switch first,
  # fall back to a plain launch if this SGLang build rejects the flag.
  start_server --chat-template-kwargs '{"enable_thinking": false}'
  if ! wait_healthy 60; then
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[entry] retrying without --chat-template-kwargs (flag unsupported?)"
      start_server
      wait_healthy || { echo "[entry] SGLang failed:"; tail -50 /tmp/sglang.log; exit 1; }
    else
      wait_healthy || { echo "[entry] SGLang failed:"; tail -50 /tmp/sglang.log; exit 1; }
    fi
  fi
  echo "[entry] server healthy."

  # smoke test: thinking must be OFF
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
  serve)
    boot
    echo "[entry] serving on http://127.0.0.1:$PORT/v1 — Ctrl+C to stop."
    tail -f /tmp/sglang.log
    ;;
  test)
    boot
    cd "$SUB"
    echo "[entry] === running the 100-query held-out simulation ==="
    "$PY" -u agent_env/scripts/solve_script_with_harness.py \
        --split phase2_heldout_sim --limit 100
    rc=$?
    echo "[entry] harness exit code: $rc (expected 0; 'Split summary skipped' is normal)"
    n=$(ls results/TPCAgent_Qwen3.6-27B_en/ 2>/dev/null | wc -l)
    echo "[entry] result files: $n/100"

    echo "[entry] === scoring against the oracle (phase2_familiar) ==="
    "$PY" -u eval_with_oracle.py eval_tpc \
        --splits phase2_familiar \
        --method TPCAgent_Qwen3.6-27B_en \
        --preference --lang en | tee /tmp/score.log

    if [ -d /output ]; then
      cp -r results /output/ 2>/dev/null
      cp /tmp/score.log /tmp/sglang.log /output/ 2>/dev/null
      echo "[entry] results + logs copied to /output"
    fi
    exit "$rc"
    ;;
  *)
    echo "unknown mode: $MODE (use test | serve | bash)"; exit 2
    ;;
esac
