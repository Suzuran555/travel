# Docker GPU test — `Antarctic penguins_v2` (two-image design)

- **Serving** = the OFFICIAL `lmsysorg/sglang:v0.5.10.post1` image, untouched
  (the organizers' stack — pulled straight from Docker Hub, never rebuilt).
- **Harness** = a small (~2 GB) image with only the v2 submission package +
  pinned deps: `zhanggangyi1224/tpc2026-penguins-v2test:latest`.
- The harness container **shares the server container's network namespace**,
  so the package's hardcoded `http://127.0.0.1:30000/v1` works with ZERO
  config overrides. harness 容器共享 serving 容器网络命名空间,包配置零改动。

Model weights are mounted from the host (~60 GB BF16, not in any image).

## 0. One-time: build & push the harness image(已由 CI 自动完成)

CI builds and pushes this automatically on any `docker/**` change
(`.github/workflows/docker-test-image.yml`). Manual equivalent:

```bash
cd travel/docker
docker build -t zhanggangyi1224/tpc2026-penguins-v2test:latest .
docker login && docker push zhanggangyi1224/tpc2026-penguins-v2test:latest
```

## 1. On the GPU server: get the model 下载模型

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.6-27B --local-dir /data/models/Qwen3.6-27B
# 或 modelscope download --model Qwen/Qwen3.6-27B --local_dir /data/models/Qwen3.6-27B
```

## 2. Run the full test 一条命令跑测(2×A100 80G)

```bash
git clone -b phase2/aug5-evaluator-and-deoverlap-fix \
    https://github.com/Suzuran555/travel.git
cd travel/docker
mkdir -p out
MODEL_DIR=/data/models/Qwen3.6-27B TP=2 \
  docker compose up --abort-on-container-exit harness
```

Or without compose 不用 compose 的等价两条命令:

```bash
docker run -d --name sglang --gpus all --ipc=host --shm-size=32g \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  lmsysorg/sglang:v0.5.10.post1 \
  python3 -m sglang.launch_server --model-path /models/Qwen3.6-27B \
    --served-model-name Qwen3.6-27B --host 0.0.0.0 --port 30000 --tp 2

mkdir -p out
docker run --rm --network container:sglang -v "$PWD/out":/output \
  zhanggangyi1224/tpc2026-penguins-v2test:latest test
```

What the harness container does 流程:
1. waits for the server (`/v1/models` must list `Qwen3.6-27B`; model load
   takes minutes);
2. **thinking-off smoke test** — warns loudly if `<think>` leaks;
3. runs the organizer entry command on the packaged held-out sim:
   `solve_script_with_harness.py --split phase2_heldout_sim --limit 100`
   (170 s/query cap, 4 h 50 m global budget, fallback always writes a file);
4. scores vs the oracle: `eval_with_oracle.py eval_tpc --splits
   phase2_familiar --method TPCAgent_Qwen3.6-27B_en --preference --lang en`;
5. copies `results/` + `score.log` to `./out/`.

Expected 预期: 100/100 result files, harness exit 0 (the final
`Split summary skipped (oracle-less data?)` line is CORRECT), and
**Overall ≈ 85–93**.

## 3. Thinking must be OFF 思考模式必须关闭

The organizers serve with thinking disabled. If step 2's smoke test warns
about `<think>`, add the server-side switch to the sglang command (build
dependent), e.g.:

```
--chat-template-kwargs '{"enable_thinking": false}'
```

and restart. Harness-side kill switch as a last resort: add
`-e CHINATRAVEL_LLM_THINK=0` to the harness `docker run`.

## 4. Troubleshooting 排查

| Symptom | Fix |
|---|---|
| harness says server never became healthy | `docker logs sglang` — model path wrong / OOM / still loading. Confirm `--network container:sglang` (or compose `network_mode: "service:sglang"`). |
| `<think>` warning | See §3. |
| Overall < 80 | Check per-query logs `agent_env/runs/tpcagent/...` inside the harness container for planner timeouts; GPU contention. |
| OOM on load | Both GPUs visible (`--gpus all`, `TP=2`), nothing else on the cards. |
| Scorer reads 0 files | scorer `--method` must equal the harness results dir (`TPCAgent_Qwen3.6-27B_en`). |

Reference ladder (details `GPU_TEST_RUNBOOK.md` §7): Mac+API capped 71.26
(hardware-limited; NOT comparable to the public leaderboard), offline
uncapped 93.54, expected here 85–93.

## 5. What changed v1 → v2 (exactly 5 files)

1. `enrich_dedupmeal.py` (new) — one meal type per day (E2E-discovered).
2. `runner.py` — dedupmeal first battery stage (16 total).
3. `enrich_mustpoi.py` — no-walk no longer bans metro.
4. `solve_script_with_harness.py` — oracle-less aggregate eval exits 0.
5. `contact.txt` — team-leader marking.

Ranking = max(v1, v2) → v2 is pure upside.
