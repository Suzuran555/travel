# GPU Server Test Runbook — `Antarctic penguins_v2.zip`

Full steps to reproduce the organizers' Phase-2 evaluation flow on a GPU
server and score the result. 在 GPU 服务器上完整复现组织方评测流程并打分。

- Package under test: **`Antarctic penguins_v2.zip`**
  (MD5 `d051547b52b0a9d23a0f16e6ada1a5db`, 941 entries)
- Organizer environment being mirrored: Debian 12, Python 3.12,
  SGLang 0.5.10 serving **Qwen3.6-27B** at `http://127.0.0.1:30000/v1`,
  api_key EMPTY, **thinking disabled**; entry command
  `python agent_env/scripts/solve_script_with_harness.py --method <m> --split <s>`;
  5 h total for 100 queries.

---

## 1. Prerequisites 前置条件

- NVIDIA GPU with enough VRAM for Qwen3.6-27B in BF16 (~60 GB+; organizers
  use 2×A800 — multi-GPU tensor parallel works too).
- CUDA 12.x driver, Python 3.10–3.12, ~80 GB free disk (model + package).

```bash
nvidia-smi                      # confirm GPU + driver
python3 --version
```

## 2. Download the model 下载模型

Pick ONE source (repo id must match what the hub actually hosts — verify):

```bash
# HuggingFace
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.6-27B --local-dir ./models/Qwen3.6-27B

# or ModelScope (国内网络快)
pip install -U modelscope
modelscope download --model Qwen/Qwen3.6-27B --local_dir ./models/Qwen3.6-27B
```

## 3. Serve with SGLang (organizer-aligned) 启动服务

```bash
pip install -U "sglang[all]==0.5.10"

python -m sglang.launch_server \
  --model-path ./models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --host 127.0.0.1 --port 30000 \
  --tp 2                                   # = number of GPUs; use 1 on a single card
```

**Served model name MUST be exactly `Qwen3.6-27B`** — the harness config
checks it verbatim. 服务名必须逐字是 `Qwen3.6-27B`。

Smoke test (thinking must be OFF — output must contain no `<think>`):

```bash
curl -s http://127.0.0.1:30000/v1/models          # must list Qwen3.6-27B
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}'
```

If the reply contains `<think>`, disable thinking at the server
(depending on SGLang build: `--chat-template-kwargs '{"enable_thinking": false}'`,
or the qwen3 non-thinking chat template). As a last resort the harness
also honors `CHINATRAVEL_LLM_THINK=0` — but prefer fixing it server-side,
because the organizers do.

## 4. Unpack the submission + install deps 解压并装依赖

```bash
unzip "Antarctic penguins_v2.zip"
cd "Antarctic penguins_v2"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Run the held-out simulation 跑 held-out 模拟

`phase2_heldout_sim` ships inside the package: it is the familiarization-100
with all oracle fields stripped — byte-identical in *shape* to the formal
held-out data.

```bash
python agent_env/scripts/solve_script_with_harness.py \
  --split phase2_heldout_sim --limit 100
```

Expected 预期:
- 100 result files in `results/TPCAgent_Qwen3.6-27B_en/<uid>.json` — always,
  even on per-query timeouts (deterministic fallback plans).
- Per-query cap 170 s, global soft deadline 4 h 50 m (both built in).
- Final lines: `Split summary skipped (oracle-less data?): KeyError('hard_logic_py')`
  then **exit code 0** — this is correct behaviour on oracle-less data
  (v2 fix; the result files are the deliverable).
- Logs per query: `agent_env/runs/tpcagent/<split>_<uid>/`.

Optional robustness knobs: `--resume` re-runs only uids whose result file is
missing (delete a uid's json to redo just that one). 只想重跑失败的:删掉对应
result json 再加 `--resume`。

## 6. Score against the oracle 打分

The oracle lives in the same package as split `phase2_familiar`
(same 100 uids):

```bash
python eval_with_oracle.py eval_tpc \
  --splits phase2_familiar \
  --method TPCAgent_Qwen3.6-27B_en \
  --preference --lang en
```

Notes: `eval_with_oracle.py` preserves oracle constraint fields for the
evaluator; with `--lang en` the method name must end in `_en` (the harness
default already does). Results are read from `results/<method>/`.

If you used a custom `--method` in step 5, mirror it here. But note the
scorer reads `results/<method>/` while the harness wrote to the method you
passed — keep them identical.

## 7. Interpreting the score 分数解读

| Reference run | Overall | FPR | Meaning |
|---|---|---|---|
| Mac + DashScope API, 170 s cap | 71.26 | 55 | ~45% planner timeouts → fallbacks (hardware-limited) |
| DashScope offline, no cap | 93.54 | 93 | in-run quality ceiling reference |
| Offline repaired plans (report headline) | 99.18 | 100 | not in-run; includes per-uid repairs |

On a real GPU server translation is fast, so the planner gets nearly the
whole 170 s → expect **Overall ≈ 85–93**. If clearly below 80, check in
this order:
1. `<think>` leaking (step 3 smoke test) — thinking not disabled;
2. served model name mismatch (`/v1/models` must say `Qwen3.6-27B`);
3. per-query logs for planner timeouts (`agent_env/runs/tpcagent/.../`);
4. GPU saturation / batch contention if you ran other jobs concurrently.

## 8. No big GPU? API-backed fallback 没有大卡时的替代

`scripts/mock_sglang_openai.py` (this repo, not in the zip) presents the
exact organizer serving surface on port 30000 and forwards to ANY
OpenAI-compatible endpoint. No secrets on disk — env vars only:

```bash
UPSTREAM_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" \
UPSTREAM_MODEL="qwen3.6-27b" \
UPSTREAM_API_KEY="<your key>" \
python scripts/mock_sglang_openai.py --port 30000
# then run steps 4–6 unchanged, zero env overrides on the harness itself
```

## 9. What changed v1 → v2 (exactly 5 files)

1. `chinatravel/agent/tpc_agent/enrich/enrich_dedupmeal.py` (new) — one
   meal type per day; E2E-discovered planner defect, removal-only, gated.
2. `.../enrich/runner.py` — dedupmeal wired as the first battery stage (16 total).
3. `.../enrich/enrich_mustpoi.py` — no-walk queries no longer wrongly ban
   metro (evaluator reports composite rides as 'metro').
4. `agent_env/scripts/solve_script_with_harness.py` — end-of-run aggregate
   evaluation guarded: oracle-less data exits 0 (results are the deliverable).
5. `contact.txt` — team-leader marking.

Ranking rule: final score = max(v1, v2) → v2 is pure upside.
排名取 v1/v2 较高者,v2 只赚不亏。
