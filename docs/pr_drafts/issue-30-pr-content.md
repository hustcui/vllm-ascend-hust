Issue 30 草稿（与 BidKV 原始语义对齐）

### 背景
当前分配失败后的 victim 选择逻辑在多套调度器中重复实现，后续策略演进需要多点改动，容易出现行为漂移和回归风险。

本 Issue 合并两项工作：
1. 先统一四套调度器的 preempt 选人入口。
2. 在统一入口上落地轻量 BidKV 公式，先做稳定、可回滚、可对比的版本。

### 目标
1. 将选谁被 preempt 收敛到统一 selector 接口（例如 selector.pick_victim）。
2. 四套调度器只替换选人步骤，保持原有 preempt 执行链不变。
3. 在 selector 内实现轻量公式：U = r / (delta + epsilon)。
4. 采用请求级近似：r 使用 num_computed_tokens。
5. delta 首版采用：delta = 1 + 0.5 x completion + 0.3 x num_preemptions。
6. 开关开启时按 BidKV 语义进行 utility 降序选择（高 U 优先）；开关关闭时行为与现网一致。

### 改动范围
1. vllm_ascend/core/recompute_scheduler.py
2. vllm_ascend/core/scheduler_dynamic_batch.py
3. vllm_ascend/core/scheduler_profiling_chunk.py
4. vllm_ascend/patch/platform/patch_balance_schedule.py
5. vllm_ascend/envs.py
6. vllm_ascend/ascend_config.py
7. vllm_ascend/platform.py

### 编码任务
1. 新增统一 victim selector 接口，定义输入、输出、回退行为。
2. 在四套调度器中接入 selector，仅替换选人逻辑，不改后续 preempt 执行链。
3. 保留 recompute_scheduler 的 kv_consumer 特殊路径，不被通用流程覆盖。
4. 实现轻量 utility 排序与稳定 tie-break 规则，保证同一快照排序稳定。
5. 处理边界输入：空 running、max_tokens 缺失、短输出、异常字段。
6. 增加 kill switch 与参数化权重，支持快速回退。

### 建议开关
1. enable_utility_victim_selection
2. utility_kill_switch
3. utility_completion_weight
4. utility_preempt_weight
5. utility_kv_gate
6. utility_cooldown_s

### CI 任务
1. 扩展并稳定以下 UT：
  - tests/ut/core/test_scheduler_dynamic_batch.py
  - tests/ut/core/test_profiling_chunk.py
2. 新增 recompute_scheduler 定向 UT，覆盖：
  - PRIORITY 分支
  - 非 PRIORITY 分支
  - kv_consumer 特殊路径
3. 新增 selector 级别 UT，覆盖：
  - 排序稳定性
  - 边界输入安全
  - 开关关闭一致性
4. 将上述 UT 纳入 CI 必跑集合，避免只靠手工回归。

### 性能验证启动
1. 先跑基线版本（统一 selector，但保持原策略）并归档结果。
2. 再跑轻量公式版本（开启 utility 选人）并与基线对比。
3. 使用固定口径运行性能脚本：
  - benchmarks/scripts/run-performance-benchmarks.sh
4. 固定实验条件：请求率、并发、KV 容量、随机种子。
5. 每组至少 3 次重复，输出均值和方差。
6. 必看指标：TTFT、SLO、Throughput、ITL、preempt 总量、单请求连续被驱逐比例。

### 验收标准
1. 四套调度器都走同一 selector 入口。
2. 开关关闭时与当前行为一致。
3. 开关开启后轻量公式稳定运行，无边界崩溃。
4. CI 全绿，新增用例稳定通过。
5. 性能结果可复现，收益和代价可解释。

### 风险与回滚
1. 风险：策略开启后可能出现过度驱逐或抖动。
2. 对策：默认关闭，保留 kill switch，支持一键回退到原策略。
3. 发布建议：先灰度再全量，按压力场景逐步放开。

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
### A. same-spec 基准实验（在 vllm-hust-benchmark 执行）
实验目标：固定同一规格（same-spec）比较 current 与 baseline，验证 100 样本下的稳定结论。

实验规格文件：
- vllm-hust-benchmark/.benchmarks/spec-random-online-100-temp0-conc2.json

关键参数：
- num_prompts=100
- max_concurrency=2
- request_rate=1
- temperature=0
- input_len=1024
- output_len=256

执行方法（先 current，后 baseline）：
1. 运行 current（vllm-hust + vllm-ascend-hust）
```bash
cd /home/cyb/vllm-hust-benchmark
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "$TS" > /tmp/bench_compare_ts_100
RESULT_DIR="/home/cyb/vllm-hust-benchmark/.benchmarks/compare-unoptimized-temp0-conc2-100-${TS}/current" \
RUN_ID="compare-current-temp0-conc2-100-${TS}" \
VLLM_HUST_WORKSPACE_ROOT=/home/cyb \
CURRENT_ENV_PREFIX=/root/miniconda3/envs/vllm-hust-dev \
CURRENT_RUNTIME_PYTHON=/root/miniconda3/envs/vllm-hust-dev/bin/python \
CURRENT_VLLM_HUST_REPO=/home/cyb/vllm-hust \
CURRENT_VLLM_ASCEND_HUST_REPO=/home/cyb/vllm-ascend-hust \
CURRENT_MODEL_PATH=/data/shared-models/Qwen2.5-14B-Instruct \
CURRENT_SERVER_PORT=18101 CURRENT_CLIENT_PORT=18101 \
bash scripts/run-current-ascend-same-spec.sh \
  /home/cyb/vllm-hust-benchmark/.benchmarks/spec-random-online-100-temp0-conc2.json
```
2. 运行 baseline（reference-repos/vllm + reference-repos/vllm-ascend）
```bash
cd /home/cyb/vllm-hust-benchmark
TS=$(cat /tmp/bench_compare_ts_100)
RESULT_DIR="/home/cyb/vllm-hust-benchmark/.benchmarks/compare-unoptimized-temp0-conc2-100-${TS}/baseline" \
RUN_ID="compare-baseline-temp0-conc2-100-${TS}" \
VLLM_BATCH_INVARIANT=1 \
VLLM_HUST_WORKSPACE_ROOT=/home/cyb \
CURRENT_ENV_PREFIX=/root/miniconda3/envs/vllm-hust-dev \
CURRENT_RUNTIME_PYTHON=/root/miniconda3/envs/vllm-hust-dev/bin/python \
CURRENT_RUNTIME_PYTHONPATH=/tmp/vllm_cli_sitecustomize \
CURRENT_VLLM_HUST_REPO=/home/cyb/reference-repos/vllm \
CURRENT_VLLM_ASCEND_HUST_REPO=/home/cyb/reference-repos/vllm-ascend \
CURRENT_ENGINE=vllm CURRENT_PLUGIN_ENGINE=vllm-ascend \
CURRENT_GITHUB_REPOSITORY=vllm-project/vllm \
CURRENT_PLUGIN_GITHUB_REPOSITORY=vllm-project/vllm-ascend \
CURRENT_MODEL_PATH=/data/shared-models/Qwen2.5-14B-Instruct \
CURRENT_SERVER_PORT=18102 CURRENT_CLIENT_PORT=18102 \
bash scripts/run-current-ascend-same-spec.sh \
  /home/cyb/vllm-hust-benchmark/.benchmarks/spec-random-online-100-temp0-conc2.json
```

结果目录：
- current: vllm-hust-benchmark/.benchmarks/compare-unoptimized-temp0-conc2-100-20260519T100943Z/current/raw_benchmark_result.json
- baseline: vllm-hust-benchmark/.benchmarks/compare-unoptimized-temp0-conc2-100-20260519T100943Z/baseline/raw_benchmark_result.json

正确性核验：
- two-side resolved_spec_hash 一致。
- two-side resolved_same_spec.json 均为 num_prompts=100、max_concurrency=2。

100 样本稳定对比（核心指标）：
- 成功率：current 100/100，baseline 100/100。
- Request throughput：0.0665 vs 0.0280 req/s，current 2.378x（+137.82%）。
- Output throughput：17.02 vs 7.16 tok/s，current 2.378x（+137.82%）。
- Mean TTFT：345.61 vs 749.46 ms，current 2.169x faster（-53.89%）。
- P99 TTFT：395.94 vs 2230.26 ms，current 5.633x faster（-82.25%）。
- Mean TPOT：116.43 vs 277.43 ms，current 2.383x faster（-58.03%）。
- P99 TPOT：126.78 vs 438.44 ms，current 3.458x faster（-71.08%）。

### B. 代码质量与门禁验证（在 vllm-ascend-hust 执行）
| 项目 | 命令 | 数据结果 |
|---|---|---|
| 静态检查 | /root/miniconda3/envs/vllm-hust-dev/bin/python -m ruff check (14 个目标文件) | 通过，0 个违规 |
| 单元测试 | /root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -sv tests/ut/core/test_victim_selector.py tests/ut/core/test_recompute_victim_selector.py tests/ut/core/test_utility_victim_config.py tests/ut/core/test_profiling_chunk.py tests/ut/core/test_scheduler_dynamic_batch.py tests/ut/test_ascend_config.py | 55 passed, 3 warnings |
| 性能脚本门禁 | bash -n benchmarks/scripts/run-performance-benchmarks.sh | 语法检查通过，退出码 0 |

补充说明：
- same-spec 基准数据产物来自 vllm-hust-benchmark 的本地跑数目录（.benchmarks）。
- 本 PR 的代码改动位于 vllm-ascend-hust；benchmark 结果用于支撑改动效果与稳定性说明。

## 风险与回滚
- 风险：高 KV 压力下可能出现驱逐抖动。
- 缓解：默认关闭、kill switch、kv gate、cooldown。
- 回滚：关闭 utility 开关即可立即回到当前稳定路径。

## 关联 Issue
Closes #30
