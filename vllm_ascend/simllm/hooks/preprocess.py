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

"""SimLLMPreprocessor — lightweight task embedding extraction for LSH matching.

Computes per-request embeddings BEFORE the full transformer forward by using
the model's token-embedding layer (``get_input_embeddings()``).  This is fast
(no transformer computation) and architecture-independent, at the cost of
lower semantic fidelity than full hidden states.

The real hidden-state embeddings are computed in ``SimLLMPostprocessor`` and
stored for FUTURE matching — so after the first forward pass, subsequent
matching uses semantically rich embeddings.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_ascend.simllm.embedding import extract_embedding
from vllm_ascend.simllm.utils import cumsum_to_ranges


class SimLLMPreprocessor:
    """Extract per-request embeddings from token embeddings.

    Parameters
    ----------
    pooling:
        Pooling mode for ``extract_embedding`` (``"mean"``, ``"last"``, or ``"cls"``).
    """

    def __init__(self, pooling: str = "mean") -> None:
        self._pooling = pooling

    @torch.no_grad()
    def extract_embeddings(
        self,
        model: Any,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-request L2-normalized embeddings.

        Parameters
        ----------
        model:
            The vLLM-wrapped model (must support ``get_input_embeddings()``).
        input_ids:
            Flattened token IDs ``[num_tokens]`` for the current batch.
        query_start_loc:
            Cumulative token counts ``[num_reqs + 1]`` delimiting requests.

        Returns
        -------
        ``[num_reqs, D]`` L2-normalized embeddings, one row per request.
        """
        embed_layer = model.get_input_embeddings()
        token_embs = embed_layer(input_ids)  # [num_tokens, D]

        # Convert flat [num_tokens, D] to list of per-request [S_i, D]
        ranges = cumsum_to_ranges(query_start_loc)
        num_reqs = len(ranges)
        req_embs: list[torch.Tensor] = []
        for start, end in ranges:
            if end <= start:
                # Empty request — produce zero embedding (should not happen in practice).
                req_embs.append(
                    torch.zeros(1, token_embs.shape[-1], device=token_embs.device)
                )
            else:
                req_embs.append(token_embs[start:end])  # [S_i, D]

        # Pad to same length for batched extract_embedding.
        max_len = max(r.shape[0] for r in req_embs) if req_embs else 0
        if max_len == 0:
            return torch.zeros(num_reqs, token_embs.shape[-1], device=token_embs.device)

        padded = torch.zeros(
            num_reqs,
            max_len,
            token_embs.shape[-1],
            dtype=token_embs.dtype,
            device=token_embs.device,
        )
        for i, r in enumerate(req_embs):
            padded[i, : r.shape[0], :] = r

        embeddings = extract_embedding(padded, pooling=self._pooling)  # [num_reqs, D]
        return embeddings
