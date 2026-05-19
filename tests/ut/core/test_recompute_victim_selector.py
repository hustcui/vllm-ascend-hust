# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


class TestRecomputeVictimSelector:
    def test_pick_preempt_victim_priority(self):
        scheduler = object.__new__(RecomputeScheduler)
        scheduler.victim_selector = MagicMock()
        scheduler.policy = SchedulingPolicy.PRIORITY
        scheduler.running = [SimpleNamespace(request_id="r1"), SimpleNamespace(request_id="r2")]

        scheduler.victim_selector.pick_victim.return_value = scheduler.running[0]
        victim = scheduler._pick_preempt_victim()

        assert victim.request_id == "r1"
        scheduler.victim_selector.pick_victim.assert_called_once_with(
            scheduler.running,
            SchedulingPolicy.PRIORITY,
            kv_utilization=None,
            now_s=None,
        )

    def test_pick_preempt_victim_non_priority(self):
        scheduler = object.__new__(RecomputeScheduler)
        scheduler.victim_selector = MagicMock()
        scheduler.policy = SchedulingPolicy.FCFS
        scheduler.running = [SimpleNamespace(request_id="r1"), SimpleNamespace(request_id="r2")]

        scheduler.victim_selector.pick_victim.return_value = scheduler.running[1]
        victim = scheduler._pick_preempt_victim()

        assert victim.request_id == "r2"
        scheduler.victim_selector.pick_victim.assert_called_once_with(
            scheduler.running,
            SchedulingPolicy.FCFS,
            kv_utilization=None,
            now_s=None,
        )

    def test_is_kv_consumer_recompute_path_true(self):
        scheduler = object.__new__(RecomputeScheduler)
        scheduler.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=False))

        assert scheduler._is_kv_consumer_recompute_path() is True

    def test_is_kv_consumer_recompute_path_false_when_none(self):
        scheduler = object.__new__(RecomputeScheduler)
        scheduler.vllm_config = SimpleNamespace(kv_transfer_config=None)

        assert scheduler._is_kv_consumer_recompute_path() is False

    def test_is_kv_consumer_recompute_path_false_when_producer(self):
        scheduler = object.__new__(RecomputeScheduler)
        scheduler.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True))

        assert scheduler._is_kv_consumer_recompute_path() is False