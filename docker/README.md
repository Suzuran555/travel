# Docker image — organizer-equivalent test for `Antarctic penguins_v2`

One image = SGLang 0.5.10 (organizer serving stack) + the v2 submission
package + an isolated harness venv + a one-shot test entrypoint.
一个镜像搞定:起服务 → 跑 100 条 held-out 模拟 → 对 oracle 打分。

Model weights are **not** in the image — mount them (60 GB+, BF16).

## 1. Build & push 构建并推送(推荐直接在 A100 服务器上做)

```bash
git clone -b phase2/aug5-evaluator-and-deoverlap-fix \
    https://github.com/Suzuran555/travel.git
cd travel/docker

docker build -t <your-dockerhub-user>/tpc2026-penguins-v2test:latest .
docker login
docker push <your-dockerhub-user>/tpc2026-penguins-v2test:latest
```

Notes:
- Build on an **amd64** machine (the server itself is perfect). From an
  Apple-Silicon Mac you must add `--platform linux/amd64` (slow; not
  recommended).
- The base image (`lmsysorg/sglang:v0.5.10.post1`) is public on Docker Hub,
  so pushing your image mostly cross-mounts those layers instead of
  re-uploading them — only the small delta uploads.

## 2. Get the model onto the server 下载模型(宿主机)

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.6-27B --local-dir /data/models/Qwen3.6-27B
# 或 modelscope:
#   pip install -U modelscope
#   modelscope download --model Qwen/Qwen3.6-27B --local_dir /data/models/Qwen3.6-27B
```

## 3. Run the full test 一键跑测(2×A100 80G)

```bash
mkdir -p ./out
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  -v "$PWD/out":/output \
  -e TP=2 \
  <your-dockerhub-user>/tpc2026-penguins-v2test:latest test
```

What it does 它做什么:
1. serve Qwen3.6-27B on `127.0.0.1:30000` (`--served-model-name Qwen3.6-27B`,
   tp=2), waits until healthy (model load can take minutes);
2. **thinking-off smoke test** — warns loudly if `<think>` leaks;
3. runs `solve_script_with_harness.py --split phase2_heldout_sim --limit 100`
   (the organizer entry command; 170 s/query cap, 4 h 50 m global budget);
4. scores against the oracle (`eval_with_oracle.py eval_tpc
   --splits phase2_familiar --method TPCAgent_Qwen3.6-27B_en
   --preference --lang en`);
5. copies `results/` + score log + sglang log to `./out/`.

Expected 预期: 100/100 result files, harness exit 0 (the final
`Split summary skipped (oracle-less data?)` line is CORRECT behaviour),
**Overall ≈ 85–93**.

## 4. Other modes 其他模式

```bash
# 只起服务(想手工跑命令时):
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  -p 30000:30000 <image> serve

# 进容器调试:
docker run --rm -it --gpus all --ipc=host --shm-size=32g \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro <image> bash
```

Env knobs 环境变量: `TP` (default 2), `MODEL_PATH`
(default `/models/Qwen3.6-27B`), `PORT` (default 30000).

## 5. Troubleshooting 排查

| Symptom | Fix |
|---|---|
| `<think>` warning in step 2 | This SGLang build ignored the chat-template switch. Rerun with the harness-side kill switch: add `-e CHINATRAVEL_LLM_THINK=0` to `docker run`. |
| Overall < 80 | Check `out/sglang.log` speed, per-query logs under `agent_env/runs/tpcagent/` for planner timeouts, GPU contention from other jobs. |
| OOM on load | Confirm both GPUs visible (`--gpus all`, `TP=2`), nothing else on the cards. |
| Score run reads 0 files | `--method` of the scorer must equal the harness results dir name (`TPCAgent_Qwen3.6-27B_en` by default). |

Reference score ladder (see `GPU_TEST_RUNBOOK.md` §7): Mac+API capped 71.26
(hardware-limited, not comparable to the public leaderboard), offline
uncapped 93.54, expected here 85–93.

<!-- build: v2 zip md5 d051547b52b0a9d23a0f16e6ada1a5db -->
