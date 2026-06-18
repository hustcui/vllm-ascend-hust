#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""SimLLM configuration dataclass — populated from VLLM_ASCEND_SIMLLM_* env vars."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend.envs import env_variables


@dataclass
class SimLLMConfig:
    """Centralized Sim-LLM configuration.

    All fields are populated from VLLM_ASCEND_SIMLLM_* environment variables
    via the lazy-access lambda dict in vllm_ascend/envs.py.

    Typical usage::

        config = SimLLMConfig.from_env()
        if config.enabled:
            kv_mgr = KVManager(max_cache_size=config.kv_cache_size)
    """

    # -- Feature gate -------------------------------------------------------
    enabled: bool = False

    # -- Similarity ---------------------------------------------------------
    cosine_threshold: float = 0.8
    lsh_num_bits: int = 64
    lsh_batch_threshold: int = 32

    # -- KV cache -----------------------------------------------------------
    kv_cache_size: int = 1024

    # -- Sandwich config ----------------------------------------------------
    sandwich_bottom: int = 3
    sandwich_top: int = 3

    # -- Embedding ----------------------------------------------------------
    embedding_pooling: str = "mean"

    # -- Deferral -----------------------------------------------------------
    deferral_ratio: float = 0.5
    max_deferrals: int = 3

    @classmethod
    def from_env(cls) -> SimLLMConfig:
        """Populate a SimLLMConfig from VLLM_ASCEND_SIMLLM_* env vars.

        Returns a frozen snapshot — repeated calls may reflect env-var changes
        (useful for testing), but production code should call once at init.
        """
        return cls(
            enabled=env_variables["VLLM_ASCEND_SIMLLM_ENABLED"](),
            cosine_threshold=env_variables["VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD"](),
            lsh_num_bits=env_variables["VLLM_ASCEND_SIMLLM_LSH_NUM_BITS"](),
            lsh_batch_threshold=env_variables["VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD"](),
            kv_cache_size=env_variables["VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE"](),
            sandwich_bottom=env_variables["VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM"](),
            sandwich_top=env_variables["VLLM_ASCEND_SIMLLM_SANDWICH_TOP"](),
            embedding_pooling=env_variables["VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING"](),
            deferral_ratio=env_variables["VLLM_ASCEND_SIMLLM_DEFERRAL_RATIO"](),
            max_deferrals=env_variables["VLLM_ASCEND_SIMLLM_MAX_DEFERRALS"](),
        )
