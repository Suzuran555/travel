#!/usr/bin/env bash
# Start the report's local serving stack if necessary, then validate the
# CURRENT checkout. Install serving dependencies and model weights first.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$project_dir"

export HARNESS_PYTHON="${HARNESS_PYTHON:-python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Qwen3.6-27B}"
TP="${TP:-2}"
PORT="${PORT:-30000}"
SGLANG_START_TIMEOUT="${SGLANG_START_TIMEOUT:-1800}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash docker/run_single_instance.sh [validation arguments]' \
        'Uses a healthy local Qwen3.6-27B endpoint or starts SGLang 0.5.10.post1.' \
        'MODEL_DIR: existing model directory (default /root/autodl-tmp/Qwen3.6-27B)' \
        'TP: tensor parallel size (default 2); PORT: local serving port (default 30000)' \
        'HARNESS_PYTHON: harness Python executable (default python)' \
        'SGLANG_PYTHON: serving Python executable (default python3)' \
        '--preflight-only: inspect validation prerequisites without starting a server'
    exit 0
fi

for value in "$TP" "$PORT" "$SGLANG_START_TIMEOUT"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'TP, PORT and SGLANG_START_TIMEOUT must be positive integers.\n' >&2
        exit 2
    fi
done
if (( PORT > 65535 )); then
    printf 'PORT must be at most 65535.\n' >&2
    exit 2
fi

# This helper is deliberately local. An externally configured endpoint can
# be used through scripts/validate_report.sh directly.
export CHINATRAVEL_OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export CHINATRAVEL_OPENAI_MODEL=Qwen3.6-27B
export CHINATRAVEL_LLM_THINK=0

for argument in "$@"; do
    if [[ "$argument" == "--preflight-only" ]]; then
        exec bash scripts/validate_report.sh "$@"
    fi
done

for executable in curl "$HARNESS_PYTHON"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required executable is unavailable: %s\n' "$executable" >&2
        exit 1
    fi
done

server_healthy() {
    curl --fail --silent --max-time 5 "$CHINATRAVEL_OPENAI_BASE_URL/models" \
        | "$HARNESS_PYTHON" -c \
            'import json,sys; sys.exit(not any(m.get("id") == "Qwen3.6-27B" for m in json.load(sys.stdin).get("data", [])))' \
            >/dev/null 2>&1
}

sglang_pid=""
stop_owned_server() {
    if [[ -n "$sglang_pid" ]] && kill -0 "$sglang_pid" 2>/dev/null; then
        kill "$sglang_pid" 2>/dev/null || true
        wait "$sglang_pid" 2>/dev/null || true
    fi
}
trap stop_owned_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if server_healthy; then
    printf 'Using existing local Qwen3.6-27B serving at %s\n' "$CHINATRAVEL_OPENAI_BASE_URL"
elif curl --fail --silent --max-time 5 "$CHINATRAVEL_OPENAI_BASE_URL/models" >/dev/null; then
    printf 'A server is already using %s but does not advertise Qwen3.6-27B. Select the report model or another PORT.\n' \
        "$CHINATRAVEL_OPENAI_BASE_URL" >&2
    exit 1
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf 'No local serving endpoint and nvidia-smi is unavailable. Start the report serving stack on a GPU host.\n' >&2
        exit 1
    fi
    if ! gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"; then
        printf 'The NVIDIA GPU is inaccessible. Check the host driver and container/WSL GPU access.\n' >&2
        exit 1
    fi
    mapfile -t gpu_memories <<< "$gpu_memory"
    if (( ${#gpu_memories[@]} < TP )); then
        printf 'TP=%s requires %s GPUs; nvidia-smi reports %s. The report used two A800 GPUs.\n' \
            "$TP" "$TP" "${#gpu_memories[@]}" >&2
        exit 1
    fi
    total_memory=0
    for (( i=0; i<TP; i++ )); do
        memory="${gpu_memories[$i]//[[:space:]]/}"
        if [[ ! "$memory" =~ ^[0-9]+$ ]]; then
            printf 'Cannot determine GPU memory from nvidia-smi output.\n' >&2
            exit 1
        fi
        total_memory=$((total_memory + memory))
    done
    if (( total_memory < 60000 )); then
        printf 'Selected GPUs provide %s MiB total; the BF16 27B model needs approximately 60 GB plus runtime headroom.\n' \
            "$total_memory" >&2
        exit 1
    fi
    if ! command -v "$SGLANG_PYTHON" >/dev/null 2>&1; then
        printf 'Serving Python is unavailable: %s\n' "$SGLANG_PYTHON" >&2
        exit 1
    fi
    if ! "$SGLANG_PYTHON" -c \
        'from importlib.metadata import version; actual=version("sglang"); print("SGLang", actual); raise SystemExit(actual != "0.5.10.post1")'; then
        printf 'Install SGLang 0.5.10.post1 in the serving environment, or use lmsysorg/sglang:v0.5.10.post1.\n' >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_DIR/config.json" ]]; then
        printf 'Model config is missing: %s/config.json\nSet MODEL_DIR to the existing Qwen3.6-27B checkpoint. No model was downloaded.\n' \
            "$MODEL_DIR" >&2
        exit 1
    fi

    serving_run_dir="$(mktemp -d /tmp/tpca-sglang.XXXXXX)"
    serving_log="$serving_run_dir/sglang.log"
    printf 'Starting SGLang with MODEL_DIR=%s, TP=%s; log: %s\n' "$MODEL_DIR" "$TP" "$serving_log"
    "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --served-model-name Qwen3.6-27B \
        --host 127.0.0.1 --port "$PORT" --tp "$TP" \
        >"$serving_log" 2>&1 &
    sglang_pid=$!
    start_seconds=$SECONDS
    until server_healthy; do
        if ! kill -0 "$sglang_pid" 2>/dev/null; then
            printf 'SGLang exited before it became healthy. See %s\n' "$serving_log" >&2
            tail -n 30 "$serving_log" >&2
            exit 1
        fi
        if (( SECONDS - start_seconds >= SGLANG_START_TIMEOUT )); then
            printf 'SGLang did not become healthy within %s seconds. See %s\n' \
                "$SGLANG_START_TIMEOUT" "$serving_log" >&2
            exit 1
        fi
        sleep 5
    done
fi

bash scripts/validate_report.sh "$@"
