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

"""SimHash LSH for fast approximate cosine-similarity search.

Uses random projection + sign to produce compact binary hashes.
Cosine-similar neighbours are likely to share the same hash.
"""

from __future__ import annotations

import torch


class SimHashHasher:
    """Random-projection SimHash for L2-normalized embeddings.

    Parameters
    ----------
    dim:
        Dimensionality of the input embeddings.
    num_bits:
        Number of hash bits (default 64 — fits in a single int64).
    seed:
        Random seed for the projection matrix (default 42, for reproducibility).
    """

    def __init__(self, dim: int, num_bits: int = 64, seed: int = 42) -> None:
        self.dim = dim
        self.num_bits = num_bits
        gen = torch.Generator().manual_seed(seed)
        self.projections: torch.Tensor = torch.randn(dim, num_bits, generator=gen)

    @torch.no_grad()
    def hash(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute packed int64 SimHash for each embedding.

        Parameters
        ----------
        embeddings:
            ``[B, D]`` tensor of L2-normalized embeddings.

        Returns
        -------
        ``[B]`` int64 tensor of packed hash values.
        """
        # Move projection matrix to the same device as input (NPU-safe).
        if self.projections.device != embeddings.device:
            self.projections = self.projections.to(embeddings.device)

        proj = embeddings @ self.projections  # [B, num_bits]
        bits = (proj > 0).to(torch.int64)  # [B, num_bits]

        # Pack bits: bit i → value 2^i
        powers = 2 ** torch.arange(self.num_bits, device=embeddings.device)
        return (bits * powers).sum(dim=1)  # [B] int64


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between two L2-normalized tensors.

    Parameters
    ----------
    a: ``[D]`` or ``[1, D]``
    b: ``[B, D]``

    Returns
    -------
    ``[B]`` cosine-similarity scores in [-1, 1].
    """
    a = a.view(1, -1)  # [1, D]
    b = b.view(b.shape[0], -1)  # [B, D]
    return (a @ b.T).squeeze(0)  # [B]
