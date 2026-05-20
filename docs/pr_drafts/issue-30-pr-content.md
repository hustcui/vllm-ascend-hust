PR 标题
feat(core): 统一 preempt victim selector 并引入 utility 控制

目标分支
main

来源分支
feat/bidkv-victim-selector-item1-2

PR 正文
## 变更目标
- 将 preempt victim 选人逻辑统一收敛到一个入口，降低多调度器行为漂移风险。
- 在统一入口上落地轻量 utility 排序：U = r / (delta + epsilon)。
- 保持默认行为可回退：关闭 utility 时与现网路径一致。

## 主要改动
- 新增统一 selector 实现与回退逻辑：vllm_ascend/core/victim_selector.py。
- 仅替换“选谁被 preempt”步骤，保留原 preempt 执行链：
  - vllm_ascend/core/recompute_scheduler.py
  - vllm_ascend/core/scheduler_dynamic_batch.py
  - vllm_ascend/core/scheduler_profiling_chunk.py
  - vllm_ascend/patch/platform/patch_balance_schedule.py
- 保留 recompute_scheduler 中 kv_consumer 的特殊路径。
- 打通配置与环境透传：
  - enable_utility_victim_selection
  - utility_kill_switch
  - utility_completion_weight
  - utility_preempt_weight
  - utility_kv_gate
  - utility_cooldown_s
- 将 dynamic-batch 相关 UT 纳入 CI 必跑集合（移除 ignore/blacklist）。

## 实验与验证数据
| 项目 | 命令 | 数据结果 |
|---|---|---|
| 静态检查 | /root/miniconda3/envs/vllm-hust-dev/bin/python -m ruff check (14 个目标文件) | 通过，0 个违规 |
| 单元测试 | /root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -sv tests/ut/core/test_victim_selector.py tests/ut/core/test_recompute_victim_selector.py tests/ut/core/test_utility_victim_config.py tests/ut/core/test_profiling_chunk.py tests/ut/core/test_scheduler_dynamic_batch.py tests/ut/test_ascend_config.py | 55 passed, 3 warnings |
| 性能脚本门禁 | bash -n benchmarks/scripts/run-performance-benchmarks.sh | 语法检查通过，退出码 0 |

补充说明：
- 本次分支未新增 benchmarks/ 或 results/ 下的可跟踪结果文件。
- 本次交付重点是“策略接入 + 配置开关 + CI 与 UT 验证闭环”。
- 若需要完整吞吐/延迟数值表（如 TTFT/TPOT/SLO），建议单独发起一轮固定矩阵基准并在独立结果 PR 发布。

## 风险与回滚
- 风险：高 KV 压力下可能出现驱逐抖动。
- 缓解：默认关闭、kill switch、kv gate、cooldown。
- 回滚：关闭 utility 开关即可立即回到当前稳定路径。

## 关联 Issue
Closes #30

## 附：验证证据文件
docs/pr_drafts/issue-30-validation-evidence.md
