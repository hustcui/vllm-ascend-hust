#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.utils.mem_constants import GiB_bytes

from tests.ut.base import TestBase
from vllm_ascend.worker.worker import (
    NPUWorker,
    _format_startup_memory_error,
    _get_visible_ascend_device_count,
    _maybe_auto_select_idle_ascend_device,
    _parse_npu_smi_hbm_stats,
    _parse_npu_smi_logical_map,
    _select_best_idle_ascend_device,
)


class TestDetermineAvailableMemoryMultiInstance(TestBase):
    """Tests for determine_available_memory() focusing on the multi-instance
    OOM regression (PR #7427)."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_worker(
        self,
        requested_memory: int,
        init_free_memory: int,
        init_total_memory: int,
        model_memory_usage: int | None = None,
    ):
        """Return a minimally-configured NPUWorker mock with memory state set."""
        from vllm_ascend.worker.worker import NPUWorker

        if model_memory_usage is None:
            model_memory_usage = int(0.5 * GiB_bytes)  # Qwen3-0.6B ~0.5 GiB

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = model_memory_usage

        mock_cache_config = MagicMock()
        mock_cache_config.kv_cache_memory_bytes = None
        worker.cache_config = mock_cache_config

        mock_snapshot = MagicMock()
        mock_snapshot.free_memory = init_free_memory
        mock_snapshot.total_memory = init_total_memory
        worker.init_snapshot = mock_snapshot

        worker.requested_memory = requested_memory
        worker.device = "npu:0"
        return worker

    @staticmethod
    def _make_profile_result(free_memory_after: int, non_kv_cache_memory: int):
        """Return a mock profile_result compatible with memory_profiling output.

        The worker code recomputes non_kv_cache_memory as:
            non_torch_increase + torch_peak_increase + weights_memory
        We set non_torch_increase=0, before_profile.torch_peak=0 (so
        torch_peak_increase = peak - 0 = 0 since memory_stats is mocked to
        return peak=0), and weights_memory=non_kv_cache_memory, ensuring the
        recomputed value equals the requested non_kv_cache_memory.
        """
        profile_result = MagicMock()
        profile_result.after_profile.free_memory = free_memory_after
        profile_result.non_kv_cache_memory = non_kv_cache_memory
        profile_result.non_torch_increase = 0
        profile_result.before_profile.torch_peak = 0
        profile_result.weights_memory = non_kv_cache_memory
        return profile_result

    @staticmethod
    def _patch_memory_profiling(profile_result):
        """Return a context manager mocking `memory_profiling` and `torch.npu.memory_stats`."""
        from contextlib import contextmanager

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=profile_result)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_profiling = MagicMock(return_value=mock_ctx)

        @contextmanager
        def combined():
            with (
                patch("vllm_ascend.worker.worker.memory_profiling", mock_profiling),
                patch(
                    "torch.npu.memory_stats",
                    return_value={"allocated_bytes.all.peak": 0},
                ),
            ):
                yield

        return combined()

    # ------------------------------------------------------------------ #
    # Tests
    # ------------------------------------------------------------------ #

    @patch("vllm_ascend.worker.worker.logger")
    def test_single_instance_positive_kv_cache(self, mock_logger):
        """Baseline: single instance on an empty card yields positive KV cache."""
        total = int(64 * GiB_bytes)
        gpu_util = 0.9
        requested_memory = int(total * gpu_util)  # 57.6 GiB
        init_free = int(62 * GiB_bytes)  # almost all free
        non_kv_cache = int(0.5 * GiB_bytes)  # Qwen3-0.6B weights

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        expected = requested_memory - non_kv_cache
        self.assertEqual(result, expected)
        self.assertGreater(result, 0)

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_on_same_card_positive_kv_cache(self, mock_logger):
        """
        Regression test for PR #7427.

        Scenario (64 GiB Ascend 910B card, two Qwen3-0.6B instances,
        gpu_memory_utilization=0.4):

          ┌───────────────────────────────────────────────────────────────┐
          │ Card total:  64 GiB                                           │
          │ Instance 1:  requested_memory = 64 * 0.4 = 25.6 GiB (in use) │
          │ Instance 2 start: init_snapshot.free_memory ≈ 38.4 GiB       │
          │ Instance 2: requested_memory = 25.6 GiB                      │
          │ Profiling (fixed):  non_kv_cache_memory = 0.5 GiB (weights)  │
          │ available = 25.6 - 0.5 = 25.1 GiB  → must be > 0  ✓         │
          └───────────────────────────────────────────────────────────────┘

        Before the fix, non_kv_cache_memory was inflated to include first
        instance memory (~25.6 GiB), yielding available ≈ -1.32 GiB (OOM).
        """
        total = int(64 * GiB_bytes)
        gpu_util = 0.4
        requested_memory = int(total * gpu_util)  # 25.6 GiB

        # First instance already occupies its full requested_memory slice
        first_instance_used = requested_memory  # 25.6 GiB
        init_free = total - first_instance_used  # ~38.4 GiB

        # After the fix: profiling correctly reports only the second
        # instance's own model weights, not the first instance's memory.
        non_kv_cache = int(0.5 * GiB_bytes)  # Qwen3-0.6B weights

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        self.assertGreater(
            result,
            0,
            "Second instance must have positive KV cache memory. "
            "A non-positive value means the multi-instance OOM bug "
            "(PR #7427) has regressed.",
        )
        expected = requested_memory - non_kv_cache
        self.assertEqual(result, expected)
        # Verify model_runner.profile_run() was called during profiling
        worker.model_runner.profile_run.assert_called_once()

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_buggy_non_kv_cache_gives_negative(self, mock_logger):
        """
        Documents the *pre-fix* buggy behaviour that PR #7427 addresses.

        When non_kv_cache_memory is erroneously inflated to include memory
        already held by the first instance (~25.6 GiB extra), the formula
            available = requested_memory - non_kv_cache_memory
        yields a negative value, confirming why the fix was necessary.

        This test is intentionally asserting the *negative* outcome to
        document the regressed state; it is NOT testing the fix itself.
        """
        total = int(64 * GiB_bytes)
        gpu_util = 0.4
        requested_memory = int(total * gpu_util)  # 25.6 GiB

        first_instance_used = requested_memory  # 25.6 GiB
        init_free = total - first_instance_used  # ~38.4 GiB

        # Buggy: non_kv_cache_memory = first-instance memory + second-instance weights
        buggy_non_kv_cache = int((25.6 + 0.5) * GiB_bytes)  # ~26.1 GiB

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            # free_memory decreased only by the actual new allocation (weights)
            free_memory_after=init_free - int(0.5 * GiB_bytes),
            non_kv_cache_memory=buggy_non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        # Pre-fix: 25.6 GiB - 26.1 GiB = -0.5 GiB  (negative → OOM)
        self.assertLess(
            result,
            0,
            "With the pre-fix (buggy) non_kv_cache_memory the result must be "
            "negative; this documents the OOM regression that PR #7427 fixed.",
        )

    @patch("vllm_ascend.worker.worker.logger")
    def test_assert_raises_when_free_memory_increases_after_profile(self, mock_logger):
        """
        determine_available_memory() must raise AssertionError when free memory
        after profiling is greater than before (external process released memory
        during profiling, invalidating the measurement).
        """
        total = int(64 * GiB_bytes)
        requested_memory = int(total * 0.9)
        init_free = int(60 * GiB_bytes)

        worker = self._make_worker(requested_memory, init_free, total)
        # Abnormal: free memory increased after profiling
        profile_result = self._make_profile_result(
            free_memory_after=init_free + int(1 * GiB_bytes),  # went UP
            non_kv_cache_memory=int(0.5 * GiB_bytes),
        )

        with self._patch_memory_profiling(profile_result), self.assertRaises(AssertionError) as ctx:
            worker.determine_available_memory()

        self.assertIn("Error in memory profiling", str(ctx.exception))

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_tight_memory_still_positive(self, mock_logger):
        """
        Edge case: card is almost full when second instance starts.

        Even with very little free memory left, as long as requested_memory >
        non_kv_cache_memory (i.e. there is room for at least some KV blocks),
        the result must be positive.
        """
        total = int(32 * GiB_bytes)  # smaller card (e.g. 910B1)
        gpu_util = 0.3
        requested_memory = int(total * gpu_util)  # 9.6 GiB

        # First instance has consumed most of its requested slice
        first_instance_used = requested_memory  # 9.6 GiB
        init_free = total - first_instance_used  # 22.4 GiB

        non_kv_cache = int(0.5 * GiB_bytes)  # Qwen3-0.6B

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        self.assertGreater(result, 0)
        self.assertEqual(result, requested_memory - non_kv_cache)


def test_format_startup_memory_error_includes_actionable_guidance(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)

    message = _format_startup_memory_error(
        free_memory=int(7.15 * GiB_bytes),
        total_memory=int(60.96 * GiB_bytes),
        gpu_memory_utilization=0.9,
        visible_device_count=8,
    )

    assert "about 0.10" in message
    assert "ASCEND_RT_VISIBLE_DEVICES=<id>" in message
    assert "npu-smi info" in message


def test_parse_npu_smi_hbm_stats_prefers_logical_ids_from_mapping():
    mapping_output = """
        NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
        0                              0                              4                              Ascend 910B1
        1                              0                              7                              Ascend 910B1
    """
    info_output = """
+------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B1               | OK            | 97.4        49                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          58154/ 65536         |
+===========================+===============+====================================================+
| 1     910B1               | OK            | 94.8        49                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3413 / 65536         |
+===========================+===============+====================================================+
    """

    logical_map = _parse_npu_smi_logical_map(mapping_output)
    device_stats = _parse_npu_smi_hbm_stats(info_output, logical_map, visible_device_count=8)

    assert device_stats == [
        (4, (65536 - 58154) << 20, 65536 << 20),
        (7, (65536 - 3413) << 20, 65536 << 20),
    ]


def test_select_best_idle_ascend_device_skips_unprobeable_candidate():
    mapping_output = """
        NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
        0                              0                              4                              Ascend 910B1
        1                              0                              7                              Ascend 910B1
    """
    info_output = """
+------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B1               | OK            | 97.4        49                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          58154/ 65536         |
+===========================+===============+====================================================+
| 1     910B1               | OK            | 94.8        49                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3413 / 65536         |
+===========================+===============+====================================================+
    """

    mapping_result = subprocess.CompletedProcess(["npu-smi", "info", "-m"], 0, stdout=mapping_output)
    info_result = subprocess.CompletedProcess(["npu-smi", "info"], 0, stdout=info_output)

    with patch("vllm_ascend.worker.worker.subprocess.run", side_effect=[mapping_result, info_result]), \
        patch(
            "vllm_ascend.worker.worker._probe_ascend_device_availability",
            side_effect=lambda logical_id: logical_id == 4,
        ) as mock_probe:
        selected_device = _select_best_idle_ascend_device(visible_device_count=8)

    assert selected_device == (4, (65536 - 58154) << 20, 65536 << 20)
    assert [call.args[0] for call in mock_probe.call_args_list] == [7, 4]


def test_auto_select_idle_ascend_device_returns_selected_device(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    parallel_config = SimpleNamespace(world_size=1, local_world_size=1)

    with patch("vllm_ascend.worker.worker.logger") as mock_logger, \
        patch("vllm_ascend.worker.worker._get_visible_ascend_device_count", return_value=8), \
        patch(
            "vllm_ascend.worker.worker._select_best_idle_ascend_device",
            return_value=(6, int(61.5 * GiB_bytes), int(64 * GiB_bytes)),
        ):
        selected_device = _maybe_auto_select_idle_ascend_device(
            local_rank=0, parallel_config=parallel_config
        )

    assert selected_device == 6
    assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ
    mock_logger.info.assert_called_once()


def test_auto_select_idle_ascend_device_skips_multi_worker(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    parallel_config = SimpleNamespace(world_size=2, local_world_size=2)

    with patch("vllm_ascend.worker.worker._get_visible_ascend_device_count", return_value=8), \
        patch("vllm_ascend.worker.worker._select_best_idle_ascend_device") as mock_selector:
        _maybe_auto_select_idle_ascend_device(local_rank=0, parallel_config=parallel_config)

    assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ
    mock_selector.assert_not_called()


def test_get_visible_ascend_device_count_prefers_env_without_torch_init(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")

    with patch("vllm_ascend.worker.worker.subprocess.run") as mock_run:
        assert _get_visible_ascend_device_count() == 1

    mock_run.assert_not_called()


def test_auto_select_idle_ascend_device_does_not_touch_torch_device_count(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    parallel_config = SimpleNamespace(world_size=1, local_world_size=1)

    with patch("vllm_ascend.worker.worker._get_visible_ascend_device_count", return_value=8), \
        patch("vllm_ascend.worker.worker._select_best_idle_ascend_device", return_value=(6, int(61.5 * GiB_bytes), int(64 * GiB_bytes))), \
        patch("torch.npu.device_count", side_effect=AssertionError("torch.npu.device_count should not run during auto-selection")):
        selected_device = _maybe_auto_select_idle_ascend_device(local_rank=0, parallel_config=parallel_config)

    assert selected_device == 6


def test_init_device_falls_back_to_local_rank_when_auto_selected_device_fails():
    with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
        worker = NPUWorker()

    worker.local_rank = 0
    worker.parallel_config = SimpleNamespace(
        world_size=1,
        local_world_size=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        distributed_executor_backend="mp",
    )
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_backend="mp",
            nnodes_within_dp=1,
        )
    )
    worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.5)
    worker.model_config = SimpleNamespace(seed=1)

    mock_snapshot = MagicMock()
    mock_snapshot.total_memory = 8 * GiB_bytes
    mock_snapshot.free_memory = 8 * GiB_bytes

    with patch("vllm_ascend.worker.worker._maybe_auto_select_idle_ascend_device", return_value=7), \
        patch("vllm_ascend.worker.worker.torch.device", side_effect=lambda value: value), \
        patch("vllm_ascend.worker.worker.torch.npu.set_device", side_effect=[RuntimeError("auto-selected init failed"), None]) as mock_set_device, \
        patch("vllm_ascend.worker.worker.MemorySnapshot", return_value=mock_snapshot), \
        patch("vllm_ascend.worker.worker.gc.collect"), \
        patch("vllm_ascend.worker.worker.torch.npu.empty_cache"), \
        patch("vllm_ascend.worker.worker.init_device_properties_triton"), \
        patch("vllm_ascend.worker.worker.set_random_seed"), \
        patch.object(NPUWorker, "_init_worker_distributed_environment"), \
        patch("vllm_ascend.worker.worker.get_ascend_config", return_value=SimpleNamespace(enable_cpu_binding=False)), \
        patch("vllm_ascend.worker.worker.logger") as mock_logger, \
        patch("vllm_ascend.worker.worker.check_ascend_device_type") as mock_check_ascend_device_type, \
        patch("vllm_ascend.worker.worker.torch.npu.is_available", return_value=True), \
        patch("vllm_ascend.worker.worker.torch.npu.device_count", return_value=1), \
        patch("vllm.triton_utils.HAS_TRITON", False):
        device = worker._init_device()

    assert device == "npu:0"
    assert mock_set_device.call_args_list == [(("npu:7",),), (("npu:0",),)]
    mock_logger.warning.assert_called_once()
    mock_check_ascend_device_type.assert_called_once()
