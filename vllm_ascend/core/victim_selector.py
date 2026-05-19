from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request

_DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class UtilityVictimSelectorConfig:
    enable_utility_victim_selection: bool = False
    utility_kill_switch: bool = False
    utility_completion_weight: float = 0.5
    utility_preempt_weight: float = 0.3
    utility_kv_gate: float = 0.0
    utility_cooldown_s: float = 0.0
    utility_epsilon: float = 1e-6
    utility_default_max_tokens: int = _DEFAULT_MAX_TOKENS

    @classmethod
    def from_vllm_config(cls, vllm_config) -> UtilityVictimSelectorConfig:
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        fallback_config = cls(
            enable_utility_victim_selection=bool(
                additional_config.get("enable_utility_victim_selection", False)
            ),
            utility_kill_switch=bool(additional_config.get("utility_kill_switch", False)),
            utility_completion_weight=float(additional_config.get("utility_completion_weight", 0.5)),
            utility_preempt_weight=float(additional_config.get("utility_preempt_weight", 0.3)),
            utility_kv_gate=float(additional_config.get("utility_kv_gate", 0.0)),
            utility_cooldown_s=float(additional_config.get("utility_cooldown_s", 0.0)),
        )

        try:
            from vllm_ascend.ascend_config import get_ascend_config

            ascend_config = get_ascend_config()
        except RuntimeError:
            return fallback_config

        return cls(
            enable_utility_victim_selection=bool(
                getattr(
                    ascend_config,
                    "enable_utility_victim_selection",
                    fallback_config.enable_utility_victim_selection,
                )
            ),
            utility_kill_switch=bool(
                getattr(ascend_config, "utility_kill_switch", fallback_config.utility_kill_switch)
            ),
            utility_completion_weight=float(
                getattr(
                    ascend_config,
                    "utility_completion_weight",
                    fallback_config.utility_completion_weight,
                )
            ),
            utility_preempt_weight=float(
                getattr(ascend_config, "utility_preempt_weight", fallback_config.utility_preempt_weight)
            ),
            utility_kv_gate=float(
                getattr(ascend_config, "utility_kv_gate", fallback_config.utility_kv_gate)
            ),
            utility_cooldown_s=float(
                getattr(ascend_config, "utility_cooldown_s", fallback_config.utility_cooldown_s)
            ),
            utility_epsilon=float(getattr(ascend_config, "utility_epsilon", fallback_config.utility_epsilon)),
            utility_default_max_tokens=int(
                getattr(
                    ascend_config,
                    "utility_default_max_tokens",
                    fallback_config.utility_default_max_tokens,
                )
            ),
        )


class UnifiedVictimSelector:
    def __init__(self, config: UtilityVictimSelectorConfig) -> None:
        self.config = config
        self._last_utility_pick_ts = -math.inf

    @classmethod
    def from_vllm_config(cls, vllm_config) -> UnifiedVictimSelector:
        return cls(UtilityVictimSelectorConfig.from_vllm_config(vllm_config))

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        if not running:
            raise ValueError("running is empty, cannot pick victim")

        if not self._utility_enabled:
            return self._pick_default_victim(running, policy)

        if not self._can_use_utility(kv_utilization=kv_utilization, now_s=now_s):
            return self._pick_default_victim(running, policy)

        victim = min(running, key=self._utility_rank_key)
        self._last_utility_pick_ts = self._resolve_now(now_s)
        return victim

    @property
    def _utility_enabled(self) -> bool:
        return self.config.enable_utility_victim_selection and not self.config.utility_kill_switch

    @staticmethod
    def _pick_default_victim(running: Sequence[Request], policy: SchedulingPolicy) -> Request:
        if policy == SchedulingPolicy.PRIORITY:
            return max(
                running,
                key=lambda request: (request.priority, request.arrival_time),
            )
        return running[-1]

    def _can_use_utility(self, *, kv_utilization: float | None, now_s: float | None) -> bool:
        if self.config.utility_kv_gate > 0:
            if kv_utilization is None or kv_utilization < self.config.utility_kv_gate:
                return False

        if self.config.utility_cooldown_s > 0 and self._last_utility_pick_ts > -math.inf:
            now = self._resolve_now(now_s)
            if now - self._last_utility_pick_ts < self.config.utility_cooldown_s:
                return False

        return True

    @staticmethod
    def _resolve_now(now_s: float | None) -> float:
        if now_s is not None:
            return float(now_s)
        return time.monotonic()

    def _utility_rank_key(self, request: Request) -> tuple[float, float, str]:
        utility = self._compute_utility(request)
        arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
        request_id = str(getattr(request, "request_id", ""))
        # Higher utility should be preempted first; tie-break by arrival/request id.
        return (-utility, arrival_time, request_id)

    def _compute_utility(self, request: Request) -> float:
        reward = max(float(getattr(request, "num_computed_tokens", 0) or 0), 0.0)
        completion = self._compute_completion(request)
        num_preemptions = max(float(getattr(request, "num_preemptions", 0) or 0), 0.0)

        delta = (
            1.0
            + self.config.utility_completion_weight * completion
            + self.config.utility_preempt_weight * num_preemptions
        )
        return reward / max(delta + self.config.utility_epsilon, self.config.utility_epsilon)

    def _compute_completion(self, request: Request) -> float:
        output_tokens = self._output_tokens(request)
        max_tokens = getattr(request, "max_tokens", None)
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            max_tokens = self.config.utility_default_max_tokens

        completion = float(output_tokens) / float(max_tokens)
        return min(max(completion, 0.0), 1.0)

    @staticmethod
    def _output_tokens(request: Request) -> int:
        output_token_ids = getattr(request, "output_token_ids", None)
        if output_token_ids is not None:
            try:
                return len(output_token_ids)
            except TypeError:
                pass
        return int(getattr(request, "num_output_tokens", 0) or 0)


def infer_kv_utilization_from_scheduler(scheduler) -> float | None:
    try:
        block_pool = scheduler.kv_cache_manager.block_pool
        total_blocks = float(getattr(block_pool, "num_gpu_blocks", 0) or 0)
        if total_blocks <= 0:
            return None

        free_block_queue = getattr(block_pool, "free_block_queue", None)
        free_blocks = float(getattr(free_block_queue, "num_free_blocks", 0) or 0)
        used_ratio = (total_blocks - free_blocks) / total_blocks
        return min(max(used_ratio, 0.0), 1.0)
    except AttributeError:
        return None