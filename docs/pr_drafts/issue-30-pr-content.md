PR Title
feat(core): unify preempt victim selector with utility controls

Base Branch
main

Head Branch
feat/bidkv-victim-selector-item1-2

PR Body
## Summary
- Introduce a unified preempt victim selector and wire it into recompute, dynamic batch, profiling chunk, and balance schedulers.
- Implement utility-based victim ranking with U = r / (delta + epsilon).
- Use request-level approximation with r = num_computed_tokens.
- Use delta = 1 + 0.5 x completion + 0.3 x num_preemptions.
- Keep behavior equivalent to current production path when utility is disabled.
- Add rollback controls and tunable parameters for safe rollout.

## What Changed
- Add unified selector implementation and fallback logic in vllm_ascend/core/victim_selector.py.
- Replace only victim-picking path in:
  - vllm_ascend/core/recompute_scheduler.py
  - vllm_ascend/core/scheduler_dynamic_batch.py
  - vllm_ascend/core/scheduler_profiling_chunk.py
  - vllm_ascend/patch/platform/patch_balance_schedule.py
- Keep recompute scheduler kv_consumer special path unchanged.
- Add config/env/platform wiring for:
  - enable_utility_victim_selection
  - utility_kill_switch
  - utility_completion_weight
  - utility_preempt_weight
  - utility_kv_gate
  - utility_cooldown_s
- Include dynamic-batch UT into CI required path by removing ignore/blacklist entries.

## Validation
- python -m ruff check vllm_ascend/core/victim_selector.py vllm_ascend/core/recompute_scheduler.py vllm_ascend/core/scheduler_dynamic_batch.py vllm_ascend/core/scheduler_profiling_chunk.py vllm_ascend/patch/platform/patch_balance_schedule.py vllm_ascend/envs.py vllm_ascend/ascend_config.py vllm_ascend/platform.py tests/ut/core/test_victim_selector.py tests/ut/core/test_recompute_victim_selector.py tests/ut/core/test_utility_victim_config.py tests/ut/core/test_profiling_chunk.py tests/ut/core/test_scheduler_dynamic_batch.py tests/ut/test_ascend_config.py
- python -m pytest -sv tests/ut/core/test_victim_selector.py tests/ut/core/test_recompute_victim_selector.py tests/ut/core/test_utility_victim_config.py tests/ut/core/test_profiling_chunk.py tests/ut/core/test_scheduler_dynamic_batch.py tests/ut/test_ascend_config.py
- bash -n benchmarks/scripts/run-performance-benchmarks.sh
- Evidence file: docs/pr_drafts/issue-30-validation-evidence.md

## Results
- Ruff check passed.
- Pytest passed: 55 passed, 3 warnings.
- Benchmark script syntax check passed.

## Risk and Rollback
- Risk: possible eviction oscillation under sustained KV pressure.
- Mitigation: default-off, utility kill switch, kv gate, cooldown.
- Rollback: disable utility and fall back to current behavior immediately.

## Issue Link
Closes #30
