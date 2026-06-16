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

"""KVReuseEngine — Phase A BlockTable pre-population for KV reuse.

Writes cached top-layer KV into vLLM's BlockTable for all transformer layers
of a matched request.  All layers receive the same top-layer KV (wasteful but
simple — validated by the paper's ablation; Phase B will use an external-KV
attention wrapper instead).
"""

from __future__ import annotations

import torch


class KVReuseEngine:
    """Manage cached-KV injection into the vLLM KV cache.

    Phase A strategy: write the SAME top-layer K and V to every transformer
    layer's block table.  This works with the existing attention backend
    unchanged — all layers read from the BlockTable as usual, and the
    top-layer KV from a similar cached task substitutes for every layer's KV.

    Parameters
    ----------
    block_size:
        Number of token slots per KV cache block (default 128 for Ascend 910B).
    num_kv_heads:
        Number of KV attention heads.
    head_size:
        Dimensionality of each attention head.
    """

    def __init__(
        self,
        block_size: int = 128,
        num_kv_heads: int = 8,
        head_size: int = 128,
    ) -> None:
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_injection(
        self,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        target_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align cached KV tensors to *target_seq_len*.

        Returns ``(k, v)`` each of shape ``[1, num_kv_heads, target_seq_len, head_size]``.

        - If cached KV is longer than *target_seq_len*, it is truncated.
        - If cached KV is shorter, it is zero-padded at the end.
        - If already matching length, returned unchanged.
        """
        k = self._align_length(cached_k, target_seq_len)
        v = self._align_length(cached_v, target_seq_len)
        return k, v

    @staticmethod
    def num_blocks_needed(seq_len: int, block_size: int = 128) -> int:
        """Return the minimum number of KV-cache blocks needed for *seq_len* tokens."""
        return (seq_len + block_size - 1) // block_size

    def write_to_cache(
        self,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        block_ids: list[int],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write *k* and *v* into pre-allocated KV-cache blocks.

        Parameters
        ----------
        kv_cache_k:
            Per-layer K-cache tensor ``[num_blocks, block_size, num_kv_heads, head_size]``
            (the leading dimension-2 from ``[2, num_blocks, block_size, ...]``
            is already sliced away — pass only the K-slice).
        kv_cache_v:
            Same shape as *kv_cache_k*, for V.
        block_ids:
            List of physical block IDs to write into.  Length must equal
            ``num_blocks_needed(seq_len)``.
        k:
            Key tensor ``[1, num_kv_heads, L_kv, head_size]``.
        v:
            Value tensor ``[1, num_kv_heads, L_kv, head_size]``.
        """
        seq_len = k.shape[2]
        _ = len(block_ids)  # num_blocks — must match write range
        block_size = kv_cache_k.shape[1]

        # Flatten the leading batch dim: [1, H, L, D] → [L, H, D]
        k_flat = k.squeeze(0).permute(1, 0, 2)  # [L, H, D]
        v_flat = v.squeeze(0).permute(1, 0, 2)  # [L, H, D]

        for block_idx, block_id in enumerate(block_ids):
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            kv_cache_k[block_id, :length, :, :] = k_flat[start:end, :, :]
            kv_cache_v[block_id, :length, :, :] = v_flat[start:end, :, :]

    @staticmethod
    def gather_from_cache(
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather contiguous KV from paged cache blocks.

        Parameters
        ----------
        kv_cache:
            Per-layer K or V cache tensor
            ``[num_blocks, block_size, num_kv_heads, head_size]``.
        block_ids:
            Ordered physical block IDs that hold the sequence.
        seq_len:
            Number of valid tokens to read.
        block_size:
            Number of token slots per block.

        Returns
        -------
        Tensor of shape ``[1, num_kv_heads, seq_len, head_size]`` — the
        contiguous KV sequence assembled from paged blocks.
        """
        if len(block_ids) == 0 or seq_len == 0:
            return kv_cache.new_zeros(
                1, kv_cache.shape[2], 0, kv_cache.shape[3]
            )
        # Read blocks and concatenate along the token dimension.
        blocks = [kv_cache[bid] for bid in block_ids]
        flat = torch.cat(blocks, dim=0)[:seq_len]  # [seq_len, H, D]
        return flat.permute(1, 0, 2).unsqueeze(0)  # [1, H, L, D]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_length(
        self, t: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Truncate or zero-pad *t* along the sequence-length dimension (dim=2)."""
        # t: [1, num_kv_heads, L, head_size]
        cur_len = t.shape[2]
        if cur_len == target_len:
            return t
        if cur_len > target_len:
            return t[:, :, :target_len, :]
        # Pad with zeros at the end.
        pad = torch.zeros(
            1, t.shape[1], target_len - cur_len, t.shape[3],
            dtype=t.dtype, device=t.device,
        )
        return torch.cat([t, pad], dim=2)
