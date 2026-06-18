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

"""Sim-LLM: Optimizing LLM Inference at the Edge through Inter-Task KV Reuse.

This sub-package implements single-node KV cache reuse for similar inference
tasks on Ascend 910B NPUs within the vLLM-HUST serving stack.

Public API
----------
- SimLLMConfig: Centralized configuration dataclass, populated from
  VLLM_ASCEND_SIMLLM_* environment variables.
- KVManager: LRU-evicted store for task embeddings, LSH hashes, and top-layer KV.
- SimHashHasher: Random-projection SimHash for fast approximate cosine similarity.
- SimilarityIdentifier: Adaptive similarity matching (exhaustive cosine or LSH bucket).
- SandwichConfig: Layer-selective KV retention (bottom-N + top-N).
"""

from vllm_ascend.simllm.config import SimLLMConfig
from vllm_ascend.simllm.embedding import extract_embedding
from vllm_ascend.simllm.kv_manager import CachedTask, KVManager
from vllm_ascend.simllm.kv_reuse import KVReuseEngine
from vllm_ascend.simllm.lsh import SimHashHasher
from vllm_ascend.simllm.sandwich import SandwichConfig
from vllm_ascend.simllm.similarity import MatchResult, SimilarityIdentifier

__all__ = [
    "SimLLMConfig",
    "CachedTask",
    "KVManager",
    "KVReuseEngine",
    "SimHashHasher",
    "extract_embedding",
    "MatchResult",
    "SimilarityIdentifier",
    "SandwichConfig",
]
