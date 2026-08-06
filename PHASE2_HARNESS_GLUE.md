# Phase-2 自定义 agent 接入（`solve_script_with_harness.py` 胶水）

目标：让组织方的固定入口命令

```bash
python agent_env/scripts/solve_script_with_harness.py     # 组织方用 uv run python 执行
```

驱动我们**确定性的 NeSy planner（TPCAgent）**（而不是 OpenCode/Codex 的 agentic LLM），
并把结果写到 `results/<method>/<uid>.json`。已在本仓库 `agent_env/` 落地。

## 组成

1. **`agent_env/scripts/tpc_agent_runner.py`**（新增，核心胶水）
   - `run_tpc_agent(uid, query, lang, tpcagent_config, timeout, cache_dir, log_dir) -> plan`
   - 用 `init_agent({"method":"TPCAgent", "env":WorldEnv(lang), "backbone_llm":TPCLLM(), ...})` 构造一次并复用。
   - 把 SGLang 端点写进 `CHINATRAVEL_OPENAI_BASE_URL/MODEL/API_KEY`（`setdefault`，不覆盖环境已有值）。
   - `func_timeout(cap, agent.run, args=(query,), kwargs=dict(prob_idx=uid, oralce_translation=False))`（保留组织方那个拼写）。
   - **5 小时/100 题预算**：全局软截止（默认 4h50m）+ 每题上限（默认 170s），超时或异常时用 `build_fallback_plan` 输出确定性兜底，保证 `results/<uid>.json` 不缺不空。
   - 导入路径自适应：优先 `chinatravel.agent.tpc_agent`（提交安装名），回退 `tpc_agent_penguins`（开发名）。

2. **`agent_env/config.toml.tpcagent`**（新增，配置模板）：`[run].harness = "tpcagent"` + `[tpcagent]` 段（SGLang base_url/model/api_key、per_query_timeout、total_budget_sec）。运行前 `cp agent_env/config.toml.tpcagent agent_env/config.toml`。

3. **`agent_env/scripts/solve_script_with_harness.py`** 三处最小改动（已在本分支应用；重构到组织方最新例子时照此重打）：

   **改动 1 — `main()` 放行 tpcagent：**
   ```python
   -    if harness not in {"opencode", "codex"}:
   -        raise ValueError("Harness must be one of: opencode, codex.")
   +    if harness not in {"opencode", "codex", "tpcagent"}:
   +        raise ValueError("Harness must be one of: opencode, codex, tpcagent.")
   ```

   **改动 2 — `main()` 为 tpcagent 选 config：**
   ```python
        if harness == "opencode":
            harness_model_arg = choose(args.model, args.opencode_model, None)
            selected_config = opencode_config
   +    elif harness == "tpcagent":
   +        harness_model_arg = choose(args.model, None, None)
   +        selected_config = harness_config
        else:
            harness_model_arg = choose(args.model, args.codex_model, None)
            selected_config = codex_config
   ```

   **改动 3 — `solve_query()` 分派处加分支（放在 `else:` 之前）：**
   ```python
        elif no_run_harness:
            print(...)
            return None
   +    elif harness == "tpcagent":
   +        from agent_env.scripts.tpc_agent_runner import run_tpc_agent
   +        plan = run_tpc_agent(
   +            uid=uid, query=public_query, lang=lang,
   +            tpcagent_config=harness_config, timeout=timeout,
   +            cache_dir=str(PROJECT_ROOT / "cache" / method),
   +            log_dir=str(run_dir),
   +        )
        else:
            if harness == "opencode":
                ...
   ```
   `plan` 随后走 `solve_query` 原有的公共尾部：`write_json(result_path, plan)` + `evaluate_one`。

   **改动 5(官方例子版才需要)— argparse `--harness` 的 `choices` 加 `"tpcagent"`**(travel 版无 choices;例子版 `choices=["opencode", "codex"]` 会在 CLI 显式传 `--harness tpcagent` 时拒绝)。

   **改动 4 — `solve_query()` guard `evaluate_one`（held-out 无 oracle 必需）：**
   组织方澄清（2026-08-06）：正式 held-out 数据**不含任何** oracle/DSL 字段
   （`hard_logic` / `hard_logic_py` / `hard_logic_nl` 等）。但 stock（含最新 `456b60a`）的
   `evaluate_hard_constraints_v2` 会 `symbolic_input_dict[idx]["hard_logic_py"]` →
   `KeyError`，而 `evaluate_one`/`solve_query`/`main` 都**没有** try 包裹，一崩就中断整个 run。
   结果文件在 `evaluate_one` **之前**已写出（`results/<uid>.json` 才是交付物，组织方用他们自己的
   ground-truth 分开评分），所以把这一处包起来即可：
   ```python
        write_json(result_path, plan)
        print(f"Saved plan: {result_path}")
   -    evaluation = evaluate_one(split, uid, query, plan)        # 新例子是 (..., plan, lang)
   +    try:
   +        evaluation = evaluate_one(split, uid, query, plan)    # 新例子是 (..., plan, lang)
   +    except Exception as exc:
   +        print(f"[eval] internal evaluate_one skipped ({type(exc).__name__}: {exc})")
   +        evaluation = {"uid": uid, "split": split, "method": method, "eval_skipped": True}
        write_json(eval_path, evaluation)
   ```
   已在本仓库验证：剥掉 oracle 字段后 `evaluate_one` 抛 `KeyError('hard_logic_py')`，被捕获→返回
   `eval_skipped` stub，**run 不中断、`results/<uid>.json` 照写**。

   **组织方命令与不硬编码（2026-08-06 澄清）：** 正式命令
   `python agent_env/scripts/solve_script_with_harness.py --method <method> --split <split>`，
   CLI 的 `--method/--split` 会覆盖 config（`choose(args.x, run_config.x, default)`）。我方 config
   仅提供默认值、`limit=0`（=全部，不截断），**不硬编码** split / query 数 / method。**前提**：提交包里
   必须有 `agent_env/config.toml`（`harness="tpcagent"`），因为组织方命令不带 `--harness`，
   harness 从 `[run].harness` 读。用 `cp agent_env/config.toml.tpcagent agent_env/config.toml`。

## 运行

```bash
# 组织方在 http://127.0.0.1:30000/v1 起好 SGLang(Qwen3.6-27B) 后：
cp agent_env/config.toml.tpcagent agent_env/config.toml
python agent_env/scripts/solve_script_with_harness.py            # 跑整个 split
python agent_env/scripts/solve_script_with_harness.py --uid <uid>  # 单题冒烟
# 断点续跑：config 里 resume=true，已存在的 results/<uid>.json 会跳过
```
产物：`results/<method>/<uid>.json`。

## 仍需在 Linux + Py3.12 + SGLang 上验证

- 该胶水**未在本机运行过**（macOS，无 SGLang）。提交前必须在组织方同款环境跑通：
  1. 单题冒烟（`--uid`）确认能连 SGLang、翻译、出合法 plan；
  2. 全 100 题确认总时长 ≤ 5h、无缺失文件；用同步后的评测器（`456b60a`）打分确认无回归。
- `TPCAgent`/`TPCLLM` 需装在 `chinatravel/agent/tpc_agent/`（提交安装名）；确认 `load_model.init_agent("TPCAgent")` 能解析，`TPCLLM` 由 runner 直接构造（不依赖 `init_llm` 认识它）。
- 确认 `[run].split` 的名字与组织方 held-out 数据一致（或让其用 `--split` 覆盖）。
- 依赖用 `requirements-harness-py312.txt`（Py3.12，SGLang 服务式，去掉 torch/vllm）。
