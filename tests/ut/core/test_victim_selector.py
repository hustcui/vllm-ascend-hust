# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.victim_selector import UnifiedVictimSelector


def _make_request(
    request_id: str,
    *,
    priority: int = 0,
    arrival_time: float = 0.0,
    num_computed_tokens: int = 0,
    output_tokens: int = 0,
    max_tokens: int | None = 128,
    num_preemptions: int = 0,
):
    output_token_ids = list(range(output_tokens))
    return SimpleNamespace(
        request_id=request_id,
        priority=priority,
        arrival_time=arrival_time,
        num_computed_tokens=num_computed_tokens,
        output_token_ids=output_token_ids,
        max_tokens=max_tokens,
        num_preemptions=num_preemptions,
    )


class TestUnifiedVictimSelector:
    def test_default_non_priority_returns_tail(self):
        selector = UnifiedVictimSelector.from_vllm_config(SimpleNamespace(additional_config={}))
        running = [_make_request("r1"), _make_request("r2"), _make_request("r3")]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r3"

    def test_default_priority_returns_highest_priority(self):
        selector = UnifiedVictimSelector.from_vllm_config(SimpleNamespace(additional_config={}))
        running = [
            _make_request("r1", priority=1, arrival_time=1.0),
            _make_request("r2", priority=3, arrival_time=2.0),
            _make_request("r3", priority=2, arrival_time=3.0),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.PRIORITY)
        assert victim.request_id == "r2"

    def test_utility_mode_prefers_higher_u(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_completion_weight": 0.5,
                    "utility_preempt_weight": 0.3,
                }
            )
        )

        running = [
            _make_request("r1", num_computed_tokens=220, output_tokens=12, max_tokens=128, num_preemptions=0),
            _make_request("r2", num_computed_tokens=260, output_tokens=120, max_tokens=128, num_preemptions=3),
            _make_request("r3", num_computed_tokens=180, output_tokens=60, max_tokens=128, num_preemptions=1),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r1"

    def test_utility_mode_handles_missing_max_tokens(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(additional_config={"enable_utility_victim_selection": True})
        )

        running = [
            _make_request("r1", num_computed_tokens=50, output_tokens=10, max_tokens=None, num_preemptions=0),
            _make_request("r2", num_computed_tokens=70, output_tokens=20, max_tokens=0, num_preemptions=0),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id in {"r1", "r2"}

    def test_kill_switch_falls_back_to_default(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kill_switch": True,
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2")]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r2"

    def test_kv_gate_blocks_utility_when_usage_low(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.95,
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2")]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.5)
        assert victim.request_id == "r2"

    def test_kv_gate_allows_utility_when_usage_high(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.8,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=200, output_tokens=10, num_preemptions=0),
            _make_request("r2", num_computed_tokens=120, output_tokens=100, num_preemptions=2),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.9)
        assert victim.request_id == "r1"

    def test_cooldown_falls_back_to_default_within_window(self):
        selector = UnifiedVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_cooldown_s": 10.0,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=200, output_tokens=10, num_preemptions=0),
            _make_request("r2", num_computed_tokens=120, output_tokens=100, num_preemptions=2),
        ]

        first = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=100.0)
        second = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=105.0)

        assert first.request_id == "r1"
        assert second.request_id == "r2"