# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config


@patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
@patch("vllm.config.VllmConfig.__post_init__", MagicMock())
def test_utility_selector_config_defaults(_mock_fix):
    clear_ascend_config()
    vllm_config = VllmConfig()
    ascend_config = init_ascend_config(vllm_config)

    assert ascend_config.enable_utility_victim_selection is False
    assert ascend_config.utility_kill_switch is False
    assert ascend_config.utility_completion_weight == 0.5
    assert ascend_config.utility_preempt_weight == 0.3
    assert ascend_config.utility_kv_gate == 0.0
    assert ascend_config.utility_cooldown_s == 0.0
    assert ascend_config.utility_min_running == 1
    assert ascend_config.utility_snapshot_enabled is False
    assert ascend_config.utility_snapshot_top_k == 3
    assert ascend_config.utility_snapshot_history_size == 32
    exported = ascend_config.get_utility_selector_config_dict()
    assert exported["enable_utility_victim_selection"] is False
    assert exported["utility_snapshot_top_k"] == 3
    assert exported["utility_default_max_tokens"] == 1024

    clear_ascend_config()


@patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
@patch("vllm.config.VllmConfig.__post_init__", MagicMock())
def test_utility_selector_config_validation(_mock_fix):
    clear_ascend_config()

    vllm_config = VllmConfig()
    vllm_config.additional_config = {
        "enable_utility_victim_selection": True,
        "utility_kill_switch": True,
        "utility_completion_weight": 0.8,
        "utility_preempt_weight": 0.2,
        "utility_kv_gate": 0.9,
        "utility_cooldown_s": 2.0,
        "utility_min_running": 2,
        "utility_snapshot_enabled": True,
        "utility_snapshot_top_k": 5,
        "utility_snapshot_history_size": 16,
    }
    ascend_config = init_ascend_config(vllm_config)

    assert ascend_config.enable_utility_victim_selection is True
    assert ascend_config.utility_kill_switch is True
    assert ascend_config.utility_completion_weight == 0.8
    assert ascend_config.utility_preempt_weight == 0.2
    assert ascend_config.utility_kv_gate == 0.9
    assert ascend_config.utility_cooldown_s == 2.0
    assert ascend_config.utility_min_running == 2
    assert ascend_config.utility_snapshot_enabled is True
    assert ascend_config.utility_snapshot_top_k == 5
    assert ascend_config.utility_snapshot_history_size == 16
    exported = ascend_config.get_utility_selector_config_dict()
    assert exported["utility_min_running"] == 2
    assert exported["utility_snapshot_history_size"] == 16

    clear_ascend_config()

    invalid_config = VllmConfig()
    invalid_config.additional_config = {"utility_completion_weight": -0.1}
    try:
        init_ascend_config(invalid_config)
    except ValueError as exc:
        assert "utility_completion_weight" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_completion_weight")

    clear_ascend_config()

    invalid_kv_gate_config = VllmConfig()
    invalid_kv_gate_config.additional_config = {"utility_kv_gate": 1.1}
    try:
        init_ascend_config(invalid_kv_gate_config)
    except ValueError as exc:
        assert "utility_kv_gate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_kv_gate")

    clear_ascend_config()

    invalid_min_running = VllmConfig()
    invalid_min_running.additional_config = {"utility_min_running": 0}
    try:
        init_ascend_config(invalid_min_running)
    except ValueError as exc:
        assert "utility_min_running" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_min_running")

    clear_ascend_config()

    invalid_snapshot_top_k = VllmConfig()
    invalid_snapshot_top_k.additional_config = {"utility_snapshot_top_k": 0}
    try:
        init_ascend_config(invalid_snapshot_top_k)
    except ValueError as exc:
        assert "utility_snapshot_top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_snapshot_top_k")

    clear_ascend_config()

    invalid_snapshot_history_size = VllmConfig()
    invalid_snapshot_history_size.additional_config = {"utility_snapshot_history_size": 0}
    try:
        init_ascend_config(invalid_snapshot_history_size)
    except ValueError as exc:
        assert "utility_snapshot_history_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_snapshot_history_size")

    clear_ascend_config()

    invalid_default_max_tokens = VllmConfig()
    invalid_default_max_tokens.additional_config = {"utility_default_max_tokens": 0}
    try:
        init_ascend_config(invalid_default_max_tokens)
    except ValueError as exc:
        assert "utility_default_max_tokens" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid utility_default_max_tokens")

    clear_ascend_config()