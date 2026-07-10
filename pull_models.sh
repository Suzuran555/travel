#!/usr/bin/env bash
# Network-gated model puller: only spends pull attempts when DNS actually
# resolves, so a dead network cannot burn the retry budget (2026-07-09 outage
# killed 70 attempts against DNS i/o timeouts). Resumes ollama partials.
# Usage: bash pull_models.sh  (pulls 8B first - small - then 27B)
set -u
cd "$(dirname "$0")"

net_ok() { nslookup -timeout=5 registry.ollama.ai >/dev/null 2>&1; }

pull_until_done() {
  local tag="$1" log="$2" n=0
  while true; do
    if ollama list 2>/dev/null | grep -q "$tag"; then
      echo "[$tag] READY after $n attempts"
      return 0
    fi
    if ! net_ok; then
      echo "[$tag] network down, waiting 60s (attempt budget preserved)"
      sleep 60
      continue
    fi
    n=$((n + 1))
    echo "[$tag] pull attempt $n"
    ollama pull "$tag" > "$log" 2>&1
    sleep 10
  done
}

pull_until_done "qwen3:8b" /tmp/ollama_pull8b_log.txt
pull_until_done "qwen3.6:27b" /tmp/ollama_pull_log.txt
echo "ALL MODELS READY"
ollama list
