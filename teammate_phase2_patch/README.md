# Phase-2 修复补丁（用于队友 harness `tpc-qwen36-official-api`）

把新 Phase-2 评测器下的修复套到你的 harness。你的版本**已经关掉了 `--stack`**（0 重复餐），所以本补丁只做两件事：

1. **同步评测器到新版**（`764614c` 的 meal / chronology / scoring 检查）——你 bundle 的是旧评测器，导致 enrichment 的 `ctx.passes()` 门控**不会拒绝会产生重叠的插入**，也让 de-overlap 无法触发（旧评测器下重叠计划"通过"）。同步后：所有 enrichment stage 的门控按新规则拦截重叠插入。
2. **新增 `enrich/enrich_deoverlap.py` + 在 `runner.STAGES` 最前面加 `deoverlap` stage**——对**基础计划自带**的重叠做"只改时间、保留 POI/交通位置"的 rethread 修复（门控保留，仅当整条计划变有效才采纳）。

## 为什么需要

你的 100 条熟悉集计划在**新评测器**下 = **Overall 86.84（FPR 86，14/14 掉在 chronology）**（我在干净沙箱验证，旧评测器复现你报告的 99.35666 到 14 位小数）。原因：evaluation 插入的餐/景点在新的"活动不能重叠 / 交通须在上一活动结束后出发"检查下失败。

- **纯后处理 deoverlap** 能修其中 **6/14 → Overall 92.19**（时间可重排的那些）。
- 剩下 8 个是"硬窗口正餐被相邻活动挤占"，后处理修不了（丢活动会破坏交通-位置绑定）。
- **真正回到 ~99.36 的做法**：同步评测器后**重跑 planner/enrichment**——新门控会拒绝产生重叠的搜索/插入，从源头避免。你的 planner 是确定性 + 无 LLM，重跑很快（~54s/题，100 题安全在 5h 内）。

## 应用

```bash
# <harness-root> = 含 chinatravel/ 和 eval_tpc.py 的目录
# （你 zip 里是 .../tpc-qwen36-official-api/harness）
bash apply_to_teammate.sh /path/to/tpc-qwen36-official-api/harness
```
脚本会：装 `enrich_deoverlap.py`、同步 3 个评测器文件、用 `runner.py.patch` 加 deoverlap stage（套不上则整文件替换 `runner_new.py`），并 py_compile + 检查。

## 之后（在 Linux + Py3.12 + SGLang 上）

1. **重跑 planner** 生成 100 条计划（新门控自动避免重叠/堆餐）；
2. 用**同步后的评测器**打分，确认回到 ~99.36、无回归。

## 注意

- **评测器同步会改动"competition-owned"文件** → 你的 `verify_package.sh` 字节一致性校验会报差异。请把 baseline manifest **重新 pin 到新 upstream**（`764614c` 或更新），这本来就是组织方要求的"按最新评测环境自查"。
- 组织方在他们统一环境里评分时可能用**他们自己的**评测器覆盖你 bundle 的——同步到新版仍是对的（保证生成期门控与评分一致）。
- 本补丁的 `enrich_deoverlap.py` 已单独验证：对你 100 条计划做门控修复，精确复现 **6/14 → 92.19**。
- 提交入口需符合新规范：`python agent_env/scripts/solve_script_with_harness.py`。若要接你的确定性 planner，参见我方仓库分支 `phase2/aug5-evaluator-and-deoverlap-fix` 里的 `PHASE2_HARNESS_GLUE.md` + `tpc_agent_runner.py`（自定义 `harness="tpcagent"` 接法，含 5h/100 预算与兜底）。

## 关于 upstream `456b60a` 的完整同步（本补丁**不含**，建议你**跳过**）

upstream 在 `764614c` 之后又到 `456b60a`，多加了 `concept_func` 归一化（+新文件 `concept_labels.py`）、更严的 schema 校验器、以及 environment tool apis 的改动。**我们在自己仓库分支已同步并验证过它，但本补丁没给你套上，原因：**

1. **对评分零影响**：我们 1000 条计划在 `764614c` 与 `456b60a` 下打分**逐字节相同**（`71.16204575651886`，7 指标全同）。这组改动只影响 grounding/schema 的内部表示，不改变任何计划的判定。
2. **和你的自定义交通代码冲突**：`456b60a` 的 `transportation/apis.py` 只新增了一个 `_find_nearest_station`（且 concept_func/评测器**并不调用**它），但**你的 `transportation/apis.py` 是你大改过的（相对 stock 有 188 行差异）**——直接覆盖会毁掉你的自定义交通逻辑。
3. 组织方在他们统一环境里用**他们自己的**评测器评分，你 bundle 的评测器只用于生成期门控——meal/chronology（本补丁已给的 `764614c`）才是真正会改变判定的部分，grounding/schema 同步是可有可无的防御性对齐。

**结论**：跳过 `456b60a` 闭包即可。若你确实想做 grounding/schema 对齐，只把 `concept_labels.py`（新增）、`concept_func.py`、`schema_constraint.py`、以及 accommodations/attractions/poi/restaurants 四个 tool apis 从最新 upstream 拷来，**唯独 `transportation/apis.py` 保留你自己的版本、或手工合并那 ~36 行 stock 改动**——但记住这不会改分。需要这几个文件我可以单独给。

## ⚠️ held-out 无 oracle 字段 + `evaluate_one` guard（**必做**，2026-08-06 组织方澄清）

正式 held-out（100 题）**不含任何** oracle/DSL 字段（`hard_logic` / `hard_logic_py` / `hard_logic_nl`）。而 stock（含最新 GitHub `456b60a`）的 `evaluate_hard_constraints_v2` 会 `symbolic_input_dict[idx]["hard_logic_py"]` **无 guard** → `KeyError`，且 `evaluate_one`/`solve_query`/`main` 都没包 try，**一崩就在第 1 题后中断整个 run**。`results/<uid>.json` 在 `evaluate_one` **之前**已写（那才是交付物，组织方分开用他们的 ground-truth 评分），所以把这一处包起来即可：

```python
     write_json(result_path, plan)
     print(f"Saved plan: {result_path}")
-    evaluation = evaluate_one(split, uid, query, plan)      # 新例子是 (..., plan, lang)
+    try:
+        evaluation = evaluate_one(split, uid, query, plan)  # 新例子是 (..., plan, lang)
+    except Exception as exc:
+        print(f"[eval] internal evaluate_one skipped ({type(exc).__name__}: {exc})")
+        evaluation = {"uid": uid, "split": split, "method": method, "eval_skipped": True}
     write_json(eval_path, evaluation)
```

**注意**：你的 harness zip 里**没有** `agent_env/scripts/solve_script_with_harness.py`（你用了自己的 run 脚本）。而组织方正式命令就是 `python agent_env/scripts/solve_script_with_harness.py --method <m> --split <s>`——所以你**必须把 harness 重构到最新例子**（`ChinaTravel_harness_example`，那里才有 solve_script + agent_env）。上面的 guard 就打在那个 solve_script 上。完整接入（自定义 agent 接入 solve_script 的 3 处编辑 + 这条 guard = 4 处、config、5h/100 预算+兜底）见本包 `harness_glue/PHASE2_HARNESS_GLUE.md` + `tpc_agent_runner.py` + `config.toml.tpcagent`。我们已在本机 Mac 用 Ollama 验证过这套接法端到端可跑。

## 包内容

```
apply_to_teammate.sh                         应用脚本
runner.py.patch                              runner.py 的 unified diff（加 deoverlap stage）
runner_new.py                                套不上时的整文件替换
files/enrich/enrich_deoverlap.py             de-overlap 模块（自包含）
files/evaluator/symbol_verification/commonsense_constraint.py   新 meal+chronology 检查
files/evaluator/evaluation/commonsense_constraint.py            新异常暴露
files/evaluator/eval_tpc.py                  新 scoring（含偏好耦合修正）
harness_glue/PHASE2_HARNESS_GLUE.md          harness 接入指南（4 处编辑，含 evaluate_one guard）
harness_glue/tpc_agent_runner.py             自定义 agent 接入 solve_script（5h/100 预算+兜底）
harness_glue/config.toml.tpcagent            harness="tpcagent" 配置模板
```
