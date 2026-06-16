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

"""Embedding extraction utility for Sim-LLM task similarity.

Extracts a fixed-dimensional task embedding from transformer hidden states
using configurable pooling strategies.
"""

from __future__ import annotations

import torch


def extract_embedding(
    hidden_states: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Extract a pooled task embedding from hidden states.

    Parameters
    ----------
    hidden_states:
        ``[B, S, D]`` or ``[S, D]`` tensor from the final transformer layer.
        *B* = batch, *S* = sequence length, *D* = hidden dimension.
    pooling:
        Pooling strategy — one of ``"mean"``, ``"last"``, ``"cls"``.

        * ``"mean"`` — average over all token positions (excluding padding).
        * ``"last"`` — take the last token's hidden state.
        * ``"cls"`` — take the first token (BOS / CLS token).

    Returns
    -------
    ``[B, D]`` (or ``[1, D]`` if input was ``[S, D]``) L2-normalized embedding.
    """
    if pooling not in ("mean", "last", "cls"):
        raise ValueError(
            f"Unknown pooling mode '{pooling}'. "
            f"Expected one of: 'mean', 'last', 'cls'."
        )

    # Normalise to [B, S, D]
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)  # [1, S, D]
    elif hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be 2D [S, D] or 3D [B, S, D], "
            f"got shape {hidden_states.shape}"
        )

    if pooling == "mean":
        emb = hidden_states.mean(dim=1)  # [B, D]
    elif pooling == "last":
        emb = hidden_states[:, -1, :]  # [B, D]
    elif pooling == "cls":
        emb = hidden_states[:, 0, :]  # [B, D]
    else:
        raise RuntimeError(f"Unreachable pooling mode: {pooling}")

    # L2-normalize — assume input is on the correct device already.
    return torch.nn.functional.normalize(emb, p=2, dim=-1)
