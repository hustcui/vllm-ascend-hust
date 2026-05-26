# SPDX-License-Identifier: Apache-2.0

from vllm_ascend import profiling_config
from vllm_ascend.profiling_config import CONFIG_FILENAME, SERVICE_PROFILING_SYMBOLS_YAML


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


def test_generate_service_profiling_config_refreshes_stale_existing_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "vllm_ascend"
    config_dir.mkdir(parents=True)
    config_file = config_dir / CONFIG_FILENAME
    config_file.write_text("stale-config\n", encoding="utf-8")

    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    output_path = profiling_config.generate_service_profiling_config()

    assert output_path == config_file
    assert config_file.read_text(encoding="utf-8") == SERVICE_PROFILING_SYMBOLS_YAML


def test_generate_service_profiling_config_keeps_existing_file_with_required_symbols(tmp_path, monkeypatch):
    config_dir = tmp_path / "vllm_ascend"
    config_dir.mkdir(parents=True)
    config_file = config_dir / CONFIG_FILENAME
    expected = SERVICE_PROFILING_SYMBOLS_YAML + "\n# custom-local-marker\n"
    config_file.write_text(expected, encoding="utf-8")

    monkeypatch.setattr(profiling_config, "get_config_dir", lambda: config_dir)

    output_path = profiling_config.generate_service_profiling_config()

    assert output_path == config_file
    assert config_file.read_text(encoding="utf-8") == expected
