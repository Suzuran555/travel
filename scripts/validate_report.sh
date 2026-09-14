#!/usr/bin/env bash
# Fresh 100-query report run, followed by the unmodified official evaluator.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HARNESS_PYTHON="${HARNESS_PYTHON:-python}"
METHOD="TPCAgent_validation_$(date -u +%Y%m%dT%H%M%S)_$$_en"
PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --method) METHOD="${2:?--method needs a fresh result directory name}"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/validate_report.sh [--preflight-only] [--method NAME_en]"
            echo "Use Python 3.12 via HARNESS_PYTHON and a Qwen3.6-27B HTTP service."
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$METHOD" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*_en$ ]] || {
    echo "--method must be a single directory name ending in _en" >&2; exit 2;
}
export CHINATRAVEL_OPENAI_BASE_URL="${CHINATRAVEL_OPENAI_BASE_URL:-http://127.0.0.1:30000/v1}"
export CHINATRAVEL_OPENAI_MODEL="${CHINATRAVEL_OPENAI_MODEL:-Qwen3.6-27B}"
export CHINATRAVEL_LLM_NAME=Qwen3.6-27B
export CHINATRAVEL_LLM_THINK=0
export PYTHONDONTWRITEBYTECODE=1

# Run these checks before creating outputs, so a missing model cannot produce
# an apparent successful run consisting solely of deadline-fallback plans.
"$HARNESS_PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('The report harness requires Python 3.12+. Set HARNESS_PYTHON to that interpreter.')
import argparse
import os
import requests
overrides = sorted(key for key in os.environ if key.startswith(('PENGUINS_', 'TPC_', 'URBANTRIP_'))
    or key in {'CHINATRAVEL_LLM_MAX_TOKENS', 'CHINATRAVEL_DISABLE_JSON_FORMAT',
               'CHINATRAVEL_LLM_TIMEOUT'})
if overrides:
    raise SystemExit('Unset experiment overrides before a fresh report run: ' + ', '.join(overrides))
from chinatravel.data.load_datasets import load_query
from chinatravel.agent.load_model import init_llm
from chinatravel.agent.tpc_agent_penguins.tpc_agent import ORACLE_FIELDS
from agent_env.scripts.tpc_agent_runner import _package_imports
from chinatravel.environment.world_env import WorldEnv

ids, queries = load_query(argparse.Namespace(
    splits='phase2_heldout_sim', lang='en', oracle_translation=True))
ref_ids, refs = load_query(argparse.Namespace(
    splits='phase2_familiar', lang='en', oracle_translation=True))
if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(ref_ids):
    raise SystemExit('Expected the same 100 unique UIDs in both report splits.')
for uid in ids:
    if uid not in queries or uid not in refs:
        raise SystemExit(f'Missing query: {uid}')
    if any(str(k).startswith('hard_logic') for k in queries[uid]):
        raise SystemExit('Simulation data contains reference constraints; check split loading.')
    if not refs[uid].get('hard_logic_py'):
        raise SystemExit(f'Missing reference DSL for scoring: {uid}')
    visible_reference = {k: v for k, v in refs[uid].items() if k not in ORACLE_FIELDS}
    if queries[uid] != visible_reference:
        raise SystemExit(f'Simulation/reference request mismatch: {uid}')
client = init_llm('TPCLLM')
canonical, _, _ = _package_imports()
from chinatravel.agent.tpc_agent_penguins.tpc_agent import TPCAgent
if canonical is not TPCAgent:
    raise SystemExit('The harness is not using the canonical report package.')
WorldEnv(lang='en')
WorldEnv(lang='zh')
print('Preflight: both databases, report imports, and 100 paired queries are ready.', flush=True)
if client.openai_model != 'Qwen3.6-27B':
    raise SystemExit('The report configuration requires served model Qwen3.6-27B.')
try:
    response = requests.get(client.openai_base + '/models', timeout=10,
        headers={'Authorization': 'Bearer ' + client.openai_key})
    response.raise_for_status()
    models = response.json().get('data', [])
    matching = [item for item in models if item.get('id') == client.openai_model]
    if not matching:
        raise SystemExit('The endpoint does not advertise Qwen3.6-27B.')
    if any('mock' in str(item.get('owned_by', '')).lower() for item in matching):
        raise SystemExit('A development proxy is running; it is not the report SGLang serving stack.')
    client.max_tokens = 16
    client.timeout = 30
    content = client([{'role': 'user', 'content': 'Reply with OK.'}], one_line=False)
    if not content.strip():
        raise SystemExit('The model health check returned no text.')
except requests.RequestException as exc:
    # Never print a request or headers: the endpoint may require a credential.
    raise SystemExit(f'Model service unavailable ({type(exc).__name__}). '
        'See docker/README.md and docker/run_single_instance.sh.') from None
print('Preflight: Qwen3.6-27B is responding.', flush=True)
PY
[[ "$PREFLIGHT_ONLY" == 0 ]] || exit 0

for target in "results/$METHOD" "cache/$METHOD" "validation_runs/$METHOD" "agent_env/runs/$METHOD"; do
    [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing run: $target" >&2; exit 2; }
done
RUN_DIR="validation_runs/$METHOD"
mkdir -p "$RUN_DIR"
"$HARNESS_PYTHON" - "$RUN_DIR" "$METHOD" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path
run, method = Path(sys.argv[1]), sys.argv[2]
paths = [Path('eval_tpc.py'), Path('eval_with_oracle.py'),
         Path('chinatravel/data/load_datasets.py'), Path('agent_env/config.toml.tpcagent'),
         Path('scripts/validate_report.sh'), Path('chinatravel/evaluation/output_schema.json')]
for directory in ('chinatravel/agent/tpc_agent_penguins', 'chinatravel/evaluation',
                  'chinatravel/symbol_verification', 'agent_env/scripts'):
    paths.extend(Path(directory).rglob('*.py'))
paths.extend(Path('chinatravel/agent/tpc_agent_penguins/data/segments').rglob('*.jsonl'))
for split in ('phase2_familiar', 'phase2_heldout_sim'):
    paths.append(Path('chinatravel/evaluation/default_splits') / (split + '.txt'))
    paths.extend((Path('chinatravel/data/en') / (split + '_EN')).glob('*.json'))
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(run / 'run.json').write_text(json.dumps({
    'method': method, 'python': platform.python_version(), 'fresh_translation_cache': True,
    'inference_split': 'phase2_heldout_sim', 'reference_split': 'phase2_familiar',
    'source_sha256': hashes}, indent=2) + '\n')
PY

# A unique method gives the supervisor a new per-method translation cache and
# result directory. All default report budgets/retries/worker counts remain.
set +e
"$HARNESS_PYTHON" -u agent_env/scripts/solve_script_with_harness.py \
    --config agent_env/config.toml.tpcagent --harness tpcagent --lang en \
    --split phase2_heldout_sim --limit 100 --method "$METHOD" \
    --work-dir "agent_env/runs/$METHOD" 2>&1 | tee "$RUN_DIR/inference.log"
inference_rc=${PIPESTATUS[0]}
set -e

# Capture actual plans before scoring, including an incomplete failed run.
"$HARNESS_PYTHON" - "$RUN_DIR" "$METHOD" "$inference_rc" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path
run, method, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
result_dir = Path('results') / method
ids = Path('chinatravel/evaluation/default_splits/phase2_heldout_sim.txt').read_text().split()
files = list(result_dir.glob('*.json'))
if result_dir.exists():
    shutil.copytree(result_dir, run / 'plans')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
missing = sorted(set(ids) - {p.stem for p in files})
empty = []
for p in files:
    try:
        plan = json.loads(p.read_text())
        if not isinstance(plan, dict) or not plan.get('itinerary'):
            empty.append(p.stem)
    except (ValueError, OSError):
        empty.append(p.stem)
(run / 'plans.json').write_text(json.dumps({
    'inference_exit': rc, 'count': len(files), 'missing': missing,
    'empty_or_invalid': empty, 'sha256': hashes}, indent=2) + '\n')
for p in (run / 'plans').glob('*'):
    if p.is_file():
        p.chmod(0o444)
if missing or empty or rc:
    raise SystemExit(f'Incomplete inference: exit={rc}, missing={len(missing)}, invalid={len(empty)}; snapshot saved.')
print(f'Saved a read-only snapshot of {len(files)} plans.')
PY

# The report wrapper preserves reference DSL, then runs eval_tpc.py via runpy.
score_offset=0
if [[ -f your_tpc_scores.json ]]; then
    score_offset=$(wc -c < your_tpc_scores.json)
fi
"$HARNESS_PYTHON" -u eval_with_oracle.py tpc \
    --splits phase2_familiar --method "$METHOD" --preference --lang en \
    2>&1 | tee "$RUN_DIR/evaluation.log"
"$HARNESS_PYTHON" - "$RUN_DIR" "$score_offset" <<'PY'
import json
import shutil
import sys
from pathlib import Path
run = Path(sys.argv[1])
source = Path('eval_res/splits_phase2_familiar/default')
if source.exists() and any(source.iterdir()):
    shutil.copytree(source, run / 'official_evaluation')
# Isolate only the JSON object appended by this invocation of eval_tpc.
scores = Path('your_tpc_scores.json')
with scores.open('rb') as stream:
    stream.seek(int(sys.argv[2]))
    current_scores = json.loads(stream.read().decode('utf-8'))
(run / 'scores.json').write_text(json.dumps(current_scores, indent=2) + '\n')
for p in run.rglob('*'):
    if p.is_file():
        p.chmod(0o444)
print(f'Full inference and official evaluation finished. Snapshot: {run}')
PY
