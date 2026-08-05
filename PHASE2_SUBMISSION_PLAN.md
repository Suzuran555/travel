# Phase 2 提交计划（依据组织方 2026-08-05 更新邮件）

> 本分支的目标：把在新 Phase-2 评测器下的**已验证修复**落进代码库，并记录提交所需的全部规格与行动项。
> 提交主体建议用**队友的确定性 harness**（更快、已避开堆餐、打包更严谨）；本仓库（travel）作为**本地打分基地 + 可移植零件仓库**。

## 1. 组织方硬性规格（2026-08-05）

- **入口命令（组织方执行）**：`python agent_env/scripts/solve_script_with_harness.py`，必须产出 `results/<method>/<uid>.json`。允许自定义 agent（不强制 OpenCode/Codex）。
- **只有两次正式提交**：`Antarctic penguins_v1.zip` / `Antarctic penguins_v2.zip`。v1 约 3 天出分（无实时日志）→ **尽早交 v1**，v2 留给定向改进。**此前所有上传作废。**
- **运行时预算**：全部 100 题**共 5 小时**（≈180s/题均值），不是每题 5 小时。
- **held-out 测试集**：100 题，用更新后的评测器 + Qwen3.6-27B 打分。
- **评测环境**：Debian 12 / **Python 3.12.13** / PyTorch 2.9.1+cu128 / SGLang 0.5.10 / Transformers 5.3.0 / Qwen3.6-27B（SGLang `http://127.0.0.1:30000/v1`，模型名 `Qwen3.6-27B`，api_key `EMPTY`，`enable_thinking=false`）；16 逻辑 CPU / 112 GiB / 2×A800 80G no-NVLink。
- **新 harness 例子**：`ChinaTravel_harness_example`（完整 ChinaTravel checkout，`requires-python>=3.12`，`uv`/`pyproject`，含 `agent_env/config.toml`）。报告同为 8/10 截止。

## 2. 新评测器下的真实分数（已在 clean-tip `456b60a` 沙箱验证）

对照实验：队友 100 题在干净 **旧** checkout（`d4f90b1`）下复现其报告的 **99.35666081558651（14 位小数一致）**，证明打分环境正确。

| | 旧评测器 | 新评测器 | 本分支修复后 |
|---|---|---|---|
| 队友（熟悉集 100） | 99.357 | **86.84**（FPR 86） | **92.19**（FPR 92，deoverlap 修 6/14） |
| 我们（phase1 1000） | ~100 / 榜 97.64 | **71.16**（FPR 68.4） | 需重跑 planner |

原因：新评测器新增「一天同种正餐≤1」「活动不能重叠」「交通须在上一活动结束后出发」等常识检查，并把失败计划计入偏好均值（DAV/ATT/DDR 与 FPR 绑定）。

## 3. 本分支已落地的修复

1. **评测器同步到新 upstream tip**（`chinatravel/evaluation/commonsense_constraint.py`、`symbol_verification/commonsense_constraint.py`、`eval_tpc.py`）→ enrichment 的 `ctx.passes()` 门控现在按新规则拦截堆餐/重叠插入。
2. **新增 `enrich/enrich_deoverlap.py`** + 接入 `runner.py`（放在 STAGES 最前）：对**基础计划自带**的可修重叠做「只改时间、保留 POI/交通位置」的 rethread 修复，门控保留（仅当整条计划变有效才采纳）。熟悉集实测：修 6/14，**86.84 → 92.19**。
3. **关闭 `bfstack --stack`**（`runner.py`）：堆第 2/3 份早餐在新评测器下必然触发「Repeated Meal Types in One Day」，`passes()` 本就会拒绝，显式关闭避免无效计算与风险。

## 4. 仍需完成的工作（在队友版本上做，需 Linux + Py3.12 + SGLang 才能真跑）

- **重叠的根本修复**：deoverlap 只能修 6/14 的「纯时间」重叠；剩下 8 个是「硬窗口正餐被相邻活动挤占」，需 planner **重排+重算交通**（后处理会破坏 `Invalid Transport information across positions`，因为交通与活动的 POI 端点绑定）。正确做法 = 用新评测器当 gate **重跑 planner/enrichment**（其自身门控会拒绝会产生重叠的搜索/插入）→ 目标回到 **~99.36**（队友软指标上限；DDR ~90.7，无堆餐时真 100 不可达）。
- **重构到新 harness 例子**：去掉队友 bundle 的 Python 3.11 Linux `.venv`（组织方用自己的 Py3.12 环境），改用 `uv`/`pyproject` 表达依赖；让 `python agent_env/scripts/solve_script_with_harness.py` 驱动我们的 planner 并输出 `results/<method>/<uid>.json`（自定义 harness，绕过 opencode）。
- **规则翻译器加 LLM 兜底**：队友翻译器是纯规则（30 条，仅覆盖熟悉集词汇），held-out 遇到新表述会「unmatched」掉分。可移植本仓库的 Qwen NL→DSL 硬化 prompt / `dsl_canonicalizer` 作为未命中兜底。
- **运行时**：队友 ~54s/题 → 100 题 ~90 分钟，安全；我们 V6 有 300–350s 慢题，若用我们的 planner 必须加**每题时限 + 确定性兜底**保证 100 题总计 ≤ 5h。
- **两次机会策略**：v1 = 稳妥可跑版本（尽早交）；读 v1 分后 v2 做一处定向改进。

## 5. 复现与产物

- clean-tip 打分沙箱构建：`git archive 456b60a | tar -x` + 拷入 `chinatravel/data/en`、`environment/database*`、split 文件 + 计划，用 `.venv/bin/python eval_tpc.py --splits <split> --method <m> --lang en`。
- 修复后的队友 100 计划与脚本存于 scratchpad（`teammate_repaired_92_familiar/`、`final_overlap_repair.py`）。
- 评测器回退：`git checkout d4f90b1 -- <3 个评测器文件>`。
