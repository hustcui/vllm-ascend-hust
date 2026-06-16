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

"""SimLLMPostprocessor — store task embeddings + KV in KVManager after forward.

Also handles batch deferral: if the match ratio in the current batch exceeds
``deferral_ratio``, unmatched tasks are flagged for re-queue to the scheduler.
"""

from __future__ import annotations

import logging
import time

import torch

from vllm_ascend.simllm.embedding import extract_embedding
from vllm_ascend.simllm.kv_manager import CachedTask, KVManager
from vllm_ascend.simllm.similarity import MatchResult

logger = logging.getLogger(__name__)


class SimLLMPostprocessor:
    """Store task snapshots in KVManager and decide batch deferral.

    Parameters
    ----------
    kv_manager:
        The KVManager instance shared across batches within the worker.
    pooling:
        Pooling mode for extracting embeddings from hidden states
        (``"mean"``, ``"last"``, or ``"cls"``).
    deferral_ratio:
        If the fraction of matched tasks in a batch exceeds this value,
        unmatched tasks are flagged for deferral.  Default 0.5.
    max_deferrals:
        Maximum deferral count before a task is force-processed.  Default 3.
    """

    def __init__(
        self,
        kv_manager: KVManager,
        pooling: str = "mean",
        deferral_ratio: float = 0.5,
        max_deferrals: int = 3,
    ) -> None:
        self._kv_manager = kv_manager
        self._pooling = pooling
        self._deferral_ratio = deferral_ratio
        self._max_deferrals = max_deferrals

    @torch.no_grad()
    def store_batch(
        self,
        req_ids: list[str],
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        batch_hashes: torch.Tensor,
        top_k: torch.Tensor,
        top_v: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> None:
        """Store one CachedTask entry per request in the batch.

        Parameters
        ----------
        req_ids:
            Request IDs, length B.
        hidden_states:
            Top-layer hidden states ``[num_tokens, D]``.
        query_start_loc:
            Cumulative token counts ``[B + 1]``.
        batch_hashes:
            ``[B]`` int64 SimHash values computed during preprocessing.
        top_k:
            Top-layer Key ``[num_tokens, num_kv_heads, head_size]``.
        top_v:
            Top-layer Value ``[num_tokens, num_kv_heads, head_size]``.
        seq_lens:
            Per-request token counts ``[B]``; defaults to differences of
            *query_start_loc* if not provided.
        """
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return

        embeddings = self._per_request_embeddings(hidden_states, query_start_loc)

        if seq_lens is None:
            seq_len_list = (
                (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
            )
        else:
            seq_len_list = seq_lens.cpu().tolist()

        now = time.monotonic()
        for i in range(num_reqs):
            if embeddings is not None:
                emb = embeddings[i : i + 1]  # [1, D]
            else:
                emb = torch.zeros(1, hidden_states.shape[-1], device=hidden_states.device)
            hsh = int(batch_hashes[i].item())
            s_len = int(seq_len_list[i])
            # Extract per-request KV slices.
            # top_k: [num_tokens, num_kv_heads, head_size]
            # → permute to [num_kv_heads, L_i, head_size] → add batch → [1, H, L_i, D]
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            task_k = top_k[start:end].permute(1, 0, 2).unsqueeze(0)  # [1, H, L_i, D]
            task_v = top_v[start:end].permute(1, 0, 2).unsqueeze(0)

            task = CachedTask(
                task_id=req_ids[i],
                embedding=emb,
                lsh_hash=hsh,
                top_k=task_k,
                top_v=task_v,
                last_access_time=now,
                seq_len=s_len,
            )
            self._kv_manager.store(task)

        logger.debug(
            "SimLLM postprocess: stored %d tasks (cache size=%d).",
            num_reqs,
            self._kv_manager.size(),
        )

    def compute_deferrals(
        self,
        match_results: dict[int, MatchResult],
        batch_size: int,
        deferral_counts: dict[int, int] | None = None,
    ) -> set[int]:
        """Compute which batch indices should be deferred.

        Parameters
        ----------
        match_results:
            Per-index MatchResult from SimilarityIdentifier.
        batch_size:
            Total number of requests in the batch.
        deferral_counts:
            Current deferral count per batch index (defaults to 0).

        Returns
        -------
        Set of batch indices to defer (unmatched in a high-match-ratio batch).
        """
        if batch_size == 0:
            return set()

        matched_count = sum(1 for m in match_results.values() if m.matched)
        match_ratio = matched_count / batch_size

        if match_ratio <= self._deferral_ratio:
            # Low match ratio → process all, no deferral.
            return set()

        deferrals = set()
        for i in range(batch_size):
            if i not in match_results or not match_results[i].matched:
                cnt = (deferral_counts or {}).get(i, 0)
                if cnt < self._max_deferrals:
                    deferrals.add(i)

        logger.debug(
            "SimLLM deferral: %d/%d matched (ratio=%.2f), deferring %d tasks.",
            matched_count, batch_size, match_ratio, len(deferrals),
        )
        return deferrals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_request_embeddings(
        self, hidden_states: torch.Tensor, query_start_loc: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute per-request L2-normalized embeddings from flat hidden states."""
        num_reqs = query_start_loc.shape[0] - 1
        if num_reqs == 0:
            return None
        max_len = 0
        slices: list[torch.Tensor] = []
        for i in range(num_reqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end > start:
                sl = hidden_states[start:end]
                slices.append(sl)
                max_len = max(max_len, sl.shape[0])
        if not slices:
            return None
        # Pad to same length for batched extract_embedding.
        D = slices[0].shape[-1]
        padded = torch.zeros(len(slices), max_len, D, dtype=hidden_states.dtype, device=hidden_states.device)
        for i, s in enumerate(slices):
            padded[i, : s.shape[0], :] = s
        return extract_embedding(padded, pooling=self._pooling)
