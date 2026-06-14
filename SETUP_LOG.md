# ChinaTravel WSL Setup Log

Date: 2026-06-13 Australia/Sydney
Workspace: `/home/liangfei_jin/chinatravel`

## 1. Clone Repository

Command:

```bash
git clone https://github.com/LAMDA-NeSy/ChinaTravel.git /home/liangfei_jin/chinatravel
cd /home/liangfei_jin/chinatravel
git rev-parse HEAD
```

Result:

- Clone succeeded.
- Commit: `a123786043684bb17767b6d207aa415b5ae77dcd`

## 2. Environment Check

Command:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda env list
conda activate chinatravel
python --version
```

Result:

- Reused existing conda environment: `/home/liangfei_jin/miniforge3/envs/chinatravel`
- Python: `Python 3.9.23`

## 3. Data Check

Command:

```bash
test -d /mnt/e/chinatravel/_downloads && find /mnt/e/chinatravel/_downloads -maxdepth 2 -type f
test -d /home/liangfei_jin/chinatravel/chinatravel/environment/database
```

Result:

- `/mnt/e/chinatravel/_downloads` was not present.
- `chinatravel/environment/database` was not present.
- No database or 2025 TPC phase archive was available to unzip.

Required database path:

```bash
/home/liangfei_jin/chinatravel/chinatravel/environment/database/attractions/beijing/attractions.csv
```

## 4. UrbanTrip Champion Entrypoint

Files changed:

- `chinatravel/agent/UrbanTrip/tpc_agent.py`
- `chinatravel/agent/UrbanTrip/tpc_llm.py`
- `chinatravel/agent/load_model.py`
- `run_tpc.py`

Changes:

- Replaced non-package `agent.*` imports inside UrbanTrip with `chinatravel.agent.*`.
- Registered `UrbanTrip` in `init_agent`.
- Added `UrbanTrip` to `run_tpc.py --agent` choices.
- Kept the original `TPCAgent` template unchanged.

No algorithm logic was intentionally changed.

## 5. Requirements Verification

Command:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python --version
pip install -r requirements.txt
```

Result:

- Python: `Python 3.9.23`
- `pip install -r requirements.txt` completed successfully.
- All official pinned requirements were already satisfied in the reused `chinatravel` environment.

## 6. Import and Entrypoint Smoke Tests

Command:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python run_tpc.py --help
```

Result:

- `run_tpc.py` now accepts `--agent {TPCAgent,UrbanTrip}`.

Command:

```bash
python - <<'PY'
import numpy, pandas, jsonschema, openai, datasets, torch
from chinatravel.data import load_datasets
from chinatravel.agent import load_model
from chinatravel.agent.UrbanTrip.tpc_agent import UrbanTrip
PY
```

Result:

- Core dependency imports work.
- Importing `UrbanTrip` reaches database-backed evaluation imports and fails because the sandbox database is missing.
- Error:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/home/liangfei_jin/chinatravel/chinatravel/environment/tools/accommodations/../../database/accommodations/beijing/accommodations.csv'
```

Command:

```bash
python - <<'PY'
from chinatravel.environment.world_env import WorldEnv
print("constructing WorldEnv")
WorldEnv()
print("WorldEnv ok")
PY
```

Result:

- Failed because the official sandbox database is missing.
- Error:

```text
constructing WorldEnv
FileNotFoundError: [Errno 2] No such file or directory:
'/home/liangfei_jin/chinatravel/chinatravel/environment/tools/attractions/../../database/attractions/beijing/attractions.csv'
```

Conclusion:

- Code environment and UrbanTrip entrypoint wiring are in place.
- Official baseline and UrbanTrip runtime remain blocked until database files are provided.

## 7. Dataset Check

Command:

```bash
python - <<'PY'
import argparse
from chinatravel.data.load_datasets import load_query
for split in ["easy", "medium", "human"]:
    args = argparse.Namespace(splits=split, lang="zh", oracle_translation=True)
    idx, data = load_query(args)
    print(split, len(idx), len(data), idx[0])
PY
```

Result:

```text
easy 300 300 e20241028160248698752
medium 150 150 e20241028160842228543
human 154 154 h20241029143447759844
```

2025 TPC phase data:

- No 2025 TPC data archive was found.
- No `tpc_aic_phase1.txt`, `tpc_phase1.txt`, or equivalent downloaded split was present under `chinatravel/evaluation/default_splits/`.
- No 2025 TPC query data directory was present under `chinatravel/data/`.

## 8. Commands Ready After Database Is Provided

Official baseline one-sample smoke test:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
export OPENAI_API_KEY='<provided at runtime>'
python run_exp.py --splits easy --index e20241028160248698752 --agent LLMNeSy --llm deepseek --oracle_translation
```

UrbanTrip one-sample smoke test:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python run_tpc.py --splits easy --index e20241028160248698752 --agent UrbanTrip --llm TPCLLM --oracle_translation --timeout 300
```

Expected UrbanTrip result path:

```bash
/home/liangfei_jin/chinatravel/results/UrbanTrip_EmptyLLM_oracletranslation/e20241028160248698752.json
```

## 9. IJCAI 2026 Phase 1 Data And Smoke Run

Official 2026 Phase 1 assets were downloaded from the competition site:

```bash
mkdir -p /home/liangfei_jin/chinatravel/_downloads
download TPC_IJCAI_2026_phase1_EN.zip
download TPC_IJCAI_2026_phase1.txt
download official_test_0000.zip
```

Result:

```text
TPC_IJCAI_2026_phase1_EN.zip: downloaded
TPC_IJCAI_2026_phase1.txt: downloaded, 1000 ids
official_test_0000.zip: downloaded, example package contains official_test_0000/<uid>.json
```

Data extraction:

```bash
unzip _downloads/TPC_IJCAI_2026_phase1_EN.zip -d chinatravel/data/en
cp _downloads/TPC_IJCAI_2026_phase1.txt chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1.txt
```

Database extraction:

```bash
download database_en.zip from README NJU Drive
download database.zip from README NJU Drive
unzip database_en.zip under chinatravel/environment/
unzip database.zip under chinatravel/environment/
```

Verified key database paths:

```text
chinatravel/environment/database_en/attractions/beijing/attractions.csv
chinatravel/environment/database/accommodations/beijing/accommodations.csv
```

WorldEnv smoke test:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python -c "from chinatravel.environment.world_env import WorldEnv; WorldEnv(lang='en')"
```

Result: exit code 0.

UrbanTrip without oracle translation:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320174446059265 --agent UrbanTrip --llm TPCLLM --timeout 300 --lang en
```

Result:

```text
Process exited but result JSON was {"error": "'hard_logic_py'"}.
Cause: the UrbanTrip implementation expects the official DSL field hard_logic_py.
No algorithm changes were made.
```

UrbanTrip with oracle translation, first three official Phase 1 ids:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320174446059265 --agent UrbanTrip --llm TPCLLM --oracle_translation --timeout 300 --lang en
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320181941699936 --agent UrbanTrip --llm TPCLLM --oracle_translation --timeout 300 --lang en
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320185832139483 --agent UrbanTrip --llm TPCLLM --oracle_translation --timeout 300 --lang en
```

Result:

```text
All three commands completed and generated JSON files under:
results/UrbanTrip_TPCLLM_en_oracletranslation/
Each run log ended with Pass!
No OPENAI_API_KEY was required because TPCLLM is EmptyLLM.
```

Smoke split added for local evaluation only:

```text
chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1_smoke3.txt
```

Evaluation command:

```bash
python eval_tpc.py --splits TPC_IJCAI_2026_phase1_smoke3 --method UrbanTrip_TPCLLM_en_oracletranslation --lang en
```

Issue:

```text
eval_tpc.py appends _en to the method when --lang en is used.
run_tpc.py wrote results/UrbanTrip_TPCLLM_en_oracletranslation.
eval_tpc.py therefore looked for results/UrbanTrip_TPCLLM_en_oracletranslation_en.
```

Environment-level fix:

```bash
ln -s UrbanTrip_TPCLLM_en_oracletranslation results/UrbanTrip_TPCLLM_en_oracletranslation_en
```

Re-run result:

```text
MicEPR: 100.0
MacEPR: 100.0
C-LPR: 100.0
FPR: 100.0
DAV: 80.55555555555554
ATT: 92.38970441977959
DDR: 75.92592592592592
Overall: 97.44355929506305
```

Smoke package command:

```bash
mkdir -p submission_smoke3/UrbanTrip_smoke3
cp results/UrbanTrip_TPCLLM_en_oracletranslation/{20250320174446059265,20250320181941699936,20250320185832139483}.json submission_smoke3/UrbanTrip_smoke3/
cd submission_smoke3
zip -qr UrbanTrip_smoke3.zip UrbanTrip_smoke3
```

Package structure:

```text
UrbanTrip_smoke3/
UrbanTrip_smoke3/20250320174446059265.json
UrbanTrip_smoke3/20250320181941699936.json
UrbanTrip_smoke3/20250320185832139483.json
```

Command quoting note:

```text
Several WSL commands launched from PowerShell failed because PowerShell expanded shell variables such as $id and $(date ...) before WSL received them. These did not modify code or generate valid result files, except empty-id run logs under run_logs/.
```

## 10. UrbanTripOptimized接入与2026 Phase 1小样本验证

接入方式：

```text
保留原 UrbanTrip。
新增并列 agent: UrbanTripOptimized。
来源文件: C:/Users/123/Downloads/tpc_agent_optimized.py
目标文件: chinatravel/agent/UrbanTrip/tpc_agent_optimized.py
类名改动: UrbanTrip -> UrbanTripOptimized
```

注册改动：

```text
chinatravel/agent/load_model.py: 注册 UrbanTripOptimized。
run_tpc.py: --agent choices 加入 UrbanTripOptimized。
```

额外入口健壮性修复：

```text
run_tpc.py 的 results/cache 目录创建改为 os.makedirs(..., exist_ok=True)。
原因：并行 smoke run 时多个进程同时创建同一结果目录，触发 FileExistsError。
该修复不改 agent 算法。
```

Import smoke test：

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python - <<'PY'
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv
kwargs = {
    "method": "UrbanTripOptimized",
    "env": WorldEnv(lang="en"),
    "backbone_llm": init_llm("TPCLLM"),
    "cache_dir": "cache",
    "log_dir": "cache/UrbanTripOptimized_TPCLLM_en_oracletranslation",
    "debug": False,
    "lang": "en",
}
agent = init_agent(kwargs)
print(type(agent).__name__)
print("ok")
PY
```

Result:

```text
UrbanTripOptimized
ok
```

2026 Phase 1 smoke3 commands：

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320174446059265 --agent UrbanTripOptimized --llm TPCLLM --oracle_translation --timeout 300 --lang en
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320181941699936 --agent UrbanTripOptimized --llm TPCLLM --oracle_translation --timeout 300 --lang en
python run_tpc.py --splits TPC_IJCAI_2026_phase1 --index 20250320185832139483 --agent UrbanTripOptimized --llm TPCLLM --oracle_translation --timeout 300 --lang en
```

Result:

```text
3/3 result JSON generated.
0 error JSON.
All three logs ended with Pass!.
Elapsed seconds: 20.06, 17.14, 17.02.
```

本地评测兼容：

```bash
ln -s UrbanTripOptimized_TPCLLM_en_oracletranslation results/UrbanTripOptimized_TPCLLM_en_oracletranslation_en
```

原因：

```text
eval_tpc.py --lang en 会自动把 method 追加 _en。
run_tpc.py 已经把 _en 写进结果目录名。
因此需要 symlink，避免评测脚本读取不存在的双 _en 目录。
```

Smoke3 eval:

```bash
python eval_tpc.py --splits TPC_IJCAI_2026_phase1_smoke3 --method UrbanTripOptimized_TPCLLM_en_oracletranslation --lang en
```

Result:

```text
MicEPR: 100.0
MacEPR: 100.0
C-LPR: 100.0
FPR: 100.0
DAV: 82.22222222222223
ATT: 92.48727880306828
DDR: 75.92592592592592
Overall: 97.53177134756083
```

2026 Phase 1 smoke20 split:

```text
chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1_smoke20.txt
```

Smoke20 run:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1_smoke20 --agent UrbanTripOptimized --llm TPCLLM --oracle_translation --timeout 300 --lang en --skip 1
```

Result:

```text
20/20 result JSON present.
0 missing.
0 error JSON.
17 Pass! lines in optimized_smoke20.log because the first 3 samples were already generated before this --skip 1 run.
Elapsed for the remaining 17 samples: 217.31 seconds.
```

Smoke20 eval:

```bash
python eval_tpc.py --splits TPC_IJCAI_2026_phase1_smoke20 --method UrbanTripOptimized_TPCLLM_en_oracletranslation --lang en
```

Result:

```text
MicEPR: 100.0
MacEPR: 100.0
C-LPR: 100.0
FPR: 100.0
DAV: 80.45833333333334
ATT: 81.37119982434555
DDR: 70.5
Overall: 96.61647665788394
```

Notes:

```text
UrbanTripOptimized still depends on hard_logic_py, so runs use --oracle_translation.
No OPENAI_API_KEY or DeepSeek key was required; TPCLLM is EmptyLLM.
Full 1000-sample run was not started.
```

## 11. IJCAI 2026 Phase 1 Full 1000 Run

User requested full official test run with reduced concurrency:

```text
Use UrbanTripOptimizedV2.
Run all 1000 TPC_IJCAI_2026_phase1 samples.
Use 18 parallel shards, not 28.
Use --oracle_translation.
Use TPCLLM / EmptyLLM, no API key.
```

Initial 28-shard attempt:

```text
An interrupted launch left 26 run_tpc.py shard processes running.
Those processes were stopped before starting the requested 18-shard run.
Partial outputs were kept; result count had reached 110 and later runs used --skip 1.
```

18 shard generation:

```bash
source /home/liangfei_jin/miniforge3/etc/profile.d/conda.sh
conda activate chinatravel
cd /home/liangfei_jin/chinatravel
python - <<'PY'
from pathlib import Path
src = Path("chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1.txt")
ids = src.read_text().splitlines()
n = 18
for i in range(n):
    shard = ids[i::n]
    out = Path(f"chinatravel/evaluation/default_splits/TPC_IJCAI_2026_phase1_shard18_{i:02d}.txt")
    out.write_text("\n".join(shard) + "\n")
    print(out, len(shard))
print("total", len(ids))
PY
```

Shard sizes:

```text
shard18_00 to shard18_09: 56 samples each
shard18_10 to shard18_17: 55 samples each
total: 1000
```

18-way full run:

```bash
mkdir -p run_logs/full_v2_18
/usr/bin/time -f "elapsed_sec=%e" -o run_logs/full_v2_18/full_parallel.time bash -lc '
for i in $(seq -w 0 17); do
  (
    python run_tpc.py \
      --splits TPC_IJCAI_2026_phase1_shard18_${i} \
      --agent UrbanTripOptimizedV2 \
      --llm TPCLLM \
      --oracle_translation \
      --timeout 300 \
      --lang en \
      --skip 1 \
      > run_logs/full_v2_18/shard_${i}.log 2>&1
    echo shard_${i}_exit=$? > run_logs/full_v2_18/shard_${i}.exit
  ) &
done
wait
'
```

Result:

```text
elapsed_sec=2724.34
all 18 shard exit files reported exit=0
1000 JSON files generated
missing_count=0
initial bad_count=38
```

Repair pass 1:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1_bad38_shard18_<id> --agent UrbanTripOptimized --llm TPCLLM --oracle_translation --timeout 300 --lang en
```

Result:

```text
elapsed_sec=337.13
UrbanTripOptimized repaired 6/38 bad JSON files.
Fixed ids:
20250322170959500916
20250322173253258437
20250322200540485890
20250322201313997965
20250323002030174014
20250323003540484362
```

Repair pass 2:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1_bad_remaining32_shard18_<id> --agent UrbanTrip --llm TPCLLM --oracle_translation --timeout 300 --lang en
```

Result:

```text
elapsed_sec=569.39
Original UrbanTrip repaired 26/32 remaining bad JSON files.
Remaining bad_count=6.
```

Repair pass 3:

```text
Patched UrbanTripOptimizedV2 only:
- Guarded an empty activities pop in breakfast backtracking.
- Added safe _visited_contains() to avoid pandas Index / numpy ambiguous truth-value membership errors.
```

Commands:

```bash
python run_tpc.py --splits TPC_IJCAI_2026_phase1_bad_remaining6 --agent UrbanTripOptimizedV2 --llm TPCLLM --oracle_translation --timeout 300 --lang en
python run_tpc.py --splits TPC_IJCAI_2026_phase1_bad_remaining4 --agent UrbanTripOptimizedV2 --llm TPCLLM --oracle_translation --timeout 300 --lang en
```

Result:

```text
remaining6 elapsed_sec=646.42
remaining4 elapsed_sec=137.12
final result_count=1000
missing_count=0
bad_count=0
invalid_count=0
```

Full local evaluation:

```bash
python eval_tpc.py \
  --splits TPC_IJCAI_2026_phase1 \
  --method UrbanTripOptimizedV2_TPCLLM_en_oracletranslation \
  --lang en \
  > run_logs/full_v2_18/eval_full.log 2>&1
```

Result:

```text
elapsed_sec=1105.04
MicEPR: 99.88
MacEPR: 97.0
C-LPR: 94.78699551569507
FPR: 87.3
DAV: 75.627147766323
ATT: 83.39861696733462
DDR: 71.32556955581028
Overall: 90.11031559339716
```

Submission package:

```bash
rm -rf submission_full_v2
mkdir -p submission_full_v2/UrbanTripOptimizedV2_phase1
cp results/UrbanTripOptimizedV2_TPCLLM_en_oracletranslation/*.json submission_full_v2/UrbanTripOptimizedV2_phase1/
cd submission_full_v2
zip -qr UrbanTripOptimizedV2_phase1.zip UrbanTripOptimizedV2_phase1
```

Verified package:

```text
path: /home/liangfei_jin/chinatravel/submission_full_v2/UrbanTripOptimizedV2_phase1.zip
size: 2.4M
zip entry_count: 1001
json_count: 1000
top-level folder: UrbanTripOptimizedV2_phase1/
```
