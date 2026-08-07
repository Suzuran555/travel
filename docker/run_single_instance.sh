#!/usr/bin/env bash
# All-in-one test for SINGLE-CONTAINER GPU platforms (AutoDL-style: the
# platform runs ONE image as your instance; no nested docker, no compose).
#
# Instance image to select on the platform:  lmsysorg/sglang:v0.5.10.post1
# Then inside the instance:
#   git clone -b phase2/aug5-evaluator-and-deoverlap-fix https://github.com/Suzuran555/travel.git
#   bash travel/docker/run_single_instance.sh
#
# Env knobs: MODEL_DIR (default /root/models/Qwen3.6-27B), TP (default 2),
#            PIP_INDEX (e.g. https://pypi.tuna.tsinghua.edu.cn/simple)
set -uo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${MODEL_DIR:-/root/models/Qwen3.6-27B}"
TP="${TP:-2}"
PORT="${PORT:-30000}"
PIP_INDEX="${PIP_INDEX:-}"
PIPARGS=()
[ -n "$PIP_INDEX" ] && PIPARGS=(-i "$PIP_INDEX")

echo "=== [1/5] model weights (ModelScope, domestic-friendly) ==="
if [ ! -f "$MODEL_DIR/config.json" ]; then
  pip install "${PIPARGS[@]}" -U modelscope >/dev/null
  modelscope download --model Qwen/Qwen3.6-27B --local_dir "$MODEL_DIR"
fi
echo "model at $MODEL_DIR"

echo "=== [2/5] unpack the v2 submission (zip ships in this repo) ==="
if [ ! -d /workspace/submission ]; then
  mkdir -p /workspace
  unzip -q "Antarctic penguins_v2.zip" -d /workspace
  mv "/workspace/Antarctic penguins_v2" /workspace/submission
fi

echo "=== [3/5] harness venv (isolated from sglang's python) ==="
if [ ! -x /opt/harness-venv/bin/python ]; then
  python3 -m venv /opt/harness-venv \
    || { apt-get update && apt-get install -y python3-venv && python3 -m venv /opt/harness-venv; }
  /opt/harness-venv/bin/pip install "${PIPARGS[@]}" -U pip
  /opt/harness-venv/bin/pip install torch==2.6.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    || /opt/harness-venv/bin/pip install "${PIPARGS[@]}" torch==2.6.0
  /opt/harness-venv/bin/pip install "${PIPARGS[@]}" -r /workspace/submission/requirements.txt
fi

echo "=== [4/5] launch SGLang (served name MUST be Qwen3.6-27B) ==="
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "Qwen3.6-27B"; then
  ( python3 -m sglang.launch_server \
      --model-path "$MODEL_DIR" \
      --served-model-name Qwen3.6-27B \
      --host 127.0.0.1 --port "$PORT" --tp "$TP" \
      --chat-template-kwargs '{"enable_thinking": false}' \
      >/tmp/sglang.log 2>&1 || \
    python3 -m sglang.launch_server \
      --model-path "$MODEL_DIR" \
      --served-model-name Qwen3.6-27B \
      --host 127.0.0.1 --port "$PORT" --tp "$TP" \
      >/tmp/sglang.log 2>&1 ) &
  echo "waiting for server (model load takes minutes)..."
  ok=0
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "Qwen3.6-27B"; then ok=1; break; fi
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "server failed to start:"; tail -50 /tmp/sglang.log; exit 1; }
fi
resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16}')
if grep -q "<think>" <<<"$resp"; then
  echo "WARNING: <think> leaked — thinking is ENABLED. Scores untrustworthy."
  echo "Retry harness with: export CHINATRAVEL_LLM_THINK=0"
else
  echo "thinking-off smoke test passed."
fi

echo "=== [5/5] run 100-query held-out sim + score ==="
cd /workspace/submission
/opt/harness-venv/bin/python -u agent_env/scripts/solve_script_with_harness.py \
    --split phase2_heldout_sim --limit 100
rc=$?
echo "harness exit: $rc (0 expected; 'Split summary skipped' line is normal)"
echo "result files: $(ls results/TPCAgent_Qwen3.6-27B_en/ 2>/dev/null | wc -l)/100"

/opt/harness-venv/bin/python -u eval_with_oracle.py eval_tpc \
    --splits phase2_familiar \
    --method TPCAgent_Qwen3.6-27B_en \
    --preference --lang en | tee /workspace/score.log
echo "=== DONE — score above, log at /workspace/score.log ==="
exit "$rc"
