# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.profiling_config import SERVICE_PROFILING_SYMBOLS_YAML


def test_service_profiling_symbols_include_real_ascend_schedulers():
    expected_symbols = [
        "vllm_ascend.core.recompute_scheduler:RecomputeScheduler.schedule",
        "vllm_ascend.core.scheduler_dynamic_batch:SchedulerDynamicBatch.schedule",
        "vllm_ascend.core.scheduler_profiling_chunk:ProfilingChunkScheduler.schedule",
        "vllm_ascend.patch.platform.patch_balance_schedule:BalanceScheduler.schedule",
    ]
    for symbol in expected_symbols:
        assert symbol in SERVICE_PROFILING_SYMBOLS_YAML


def test_service_profiling_symbols_include_utility_selector_hooks():
    assert (
        "vllm_ascend.core.victim_selector:UnifiedVictimSelector.pick_victim"
        in SERVICE_PROFILING_SYMBOLS_YAML
    )
    assert (
        "vllm_ascend.core.victim_selector:UnifiedVictimSelector.emit_observability_log"
        in SERVICE_PROFILING_SYMBOLS_YAML
    )


def test_service_profiling_symbols_drop_legacy_scheduler_symbol():
    assert "vllm_ascend.core.scheduler:AscendScheduler.schedule" not in SERVICE_PROFILING_SYMBOLS_YAML
