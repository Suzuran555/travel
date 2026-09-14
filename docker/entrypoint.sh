#!/usr/bin/env bash
# The image contains the current checkout; serving is a separate container.
set -euo pipefail

mode="${1:-test}"
if (( $# > 0 )); then shift; fi
cd /workspace/travel
export HARNESS_PYTHON="${HARNESS_PYTHON:-python}"

case "$mode" in
    test)
        exec bash scripts/validate_report.sh "$@"
        ;;
    bash)
        exec bash "$@"
        ;;
    *)
        printf 'Unknown mode: %s (use test or bash)\n' "$mode" >&2
        exit 2
        ;;
esac
