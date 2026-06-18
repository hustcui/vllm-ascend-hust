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

"""SimLLM identify hook — thin wrapper around SimilarityIdentifier.identify()."""

from __future__ import annotations

import torch

from vllm_ascend.simllm.kv_manager import KVManager
from vllm_ascend.simllm.similarity import MatchResult, SimilarityIdentifier


def identify_batch(
    batch_embeddings: torch.Tensor,
    batch_hashes: torch.Tensor,
    kv_manager: KVManager,
    similarity_identifier: SimilarityIdentifier,
) -> dict[int, MatchResult]:
    """Match batch embeddings against cached tasks.

    Parameters
    ----------
    batch_embeddings:
        ``[B, D]`` L2-normalized per-request embeddings.
    batch_hashes:
        ``[B]`` int64 SimHash values.
    kv_manager:
        The KVManager instance to query for cached tasks.
    similarity_identifier:
        Configured SimilarityIdentifier.

    Returns
    -------
    ``dict[int, MatchResult]`` mapping batch index → match outcome.
    """
    if batch_embeddings.shape[0] == 0:
        return {}
    return similarity_identifier.identify(
        batch_embeddings, batch_hashes, kv_manager
    )
