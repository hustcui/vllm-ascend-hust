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
    }
    ascend_config = init_ascend_config(vllm_config)

    assert ascend_config.enable_utility_victim_selection is True
    assert ascend_config.utility_kill_switch is True
    assert ascend_config.utility_completion_weight == 0.8
    assert ascend_config.utility_preempt_weight == 0.2
    assert ascend_config.utility_kv_gate == 0.9
    assert ascend_config.utility_cooldown_s == 2.0

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