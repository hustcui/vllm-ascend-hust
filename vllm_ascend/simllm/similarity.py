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

"""SimilarityIdentifier — adaptive inter-task similarity matching.

Two strategies:

* **Small batch** (``batch_size < lsh_batch_threshold``): exhaustive cosine
  similarity against all candidates in the LSH bucket.
* **Large batch** (``batch_size >= lsh_batch_threshold``): LSH bucket membership
  with KV merging (average K and V across all tasks in the same bucket).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.simllm.kv_manager import CachedTask, KVManager
from vllm_ascend.simllm.lsh import SimHashHasher, cosine_similarity
from vllm_ascend.simllm.utils import tensor_to_float_list, tensor_to_int_list


@dataclass
class MatchResult:
    """Result of similarity matching for one batch element.

    Attributes
    ----------
    matched:
        True if a similar cached task was found.
    source_task_id:
        Task id of the matched cached task, or None.
    cached_k:
        Top-layer K from the matched task (``[1, num_kv_heads, L_kv, head_dim]``).
    cached_v:
        Top-layer V from the matched task.
    similarity_score:
        Cosine similarity score, or None if LSH-merge was used.
    """

    matched: bool
    source_task_id: str | None = None
    cached_k: torch.Tensor | None = None
    cached_v: torch.Tensor | None = None
    similarity_score: float | None = None


class SimilarityIdentifier:
    """Adaptive similarity matching engine.

    Parameters
    ----------
    cosine_threshold:
        Cosine-similarity threshold θ.  Embeddings with cosine >= θ are
        considered a match.  Paper default 0.8.
    lsh_batch_threshold:
        Batch size at which the strategy switches from exhaustive cosine to
        LSH bucket + KV merge.  Default 32.
    lsh_num_bits:
        Number of SimHash bits.  Default 64.
    embedding_dim:
        Dimensionality of the task embeddings.  Default 4096 (Qwen2.5-7B).
    """

    def __init__(
        self,
        cosine_threshold: float = 0.8,
        lsh_batch_threshold: int = 32,
        lsh_num_bits: int = 64,
        embedding_dim: int = 4096,
    ) -> None:
        self.threshold = cosine_threshold
        self.lsh_batch_threshold = lsh_batch_threshold
        self.lsh_hasher = SimHashHasher(dim=embedding_dim, num_bits=lsh_num_bits)

    @torch.no_grad()
    def identify(
        self,
        batch_embeddings: torch.Tensor,
        batch_hashes: torch.Tensor,
        kv_manager: KVManager,
    ) -> dict[int, MatchResult]:
        """Identify similarity matches for a batch.

        Parameters
        ----------
        batch_embeddings:
            ``[B, D]`` L2-normalized task embeddings.
        batch_hashes:
            ``[B]`` int64 SimHash values.
        kv_manager:
            The KVManager instance to query for cached tasks.

        Returns
        -------
        ``dict[int, MatchResult]`` mapping batch index → MatchResult.
        """
        batch_size = batch_embeddings.shape[0]
        results: dict[int, MatchResult] = {}
        hash_values = tensor_to_int_list(batch_hashes)

        if batch_size < self.lsh_batch_threshold:
            # -- Small batch: exhaustive cosine per candidate --------------
            for i in range(batch_size):
                emb = batch_embeddings[i : i + 1]  # [1, D]
                hsh = hash_values[i]
                candidates = kv_manager.lookup_by_hash(hsh)

                if not candidates:
                    results[i] = MatchResult(matched=False)
                    continue

                candidate_embs = torch.cat(
                    [c.embedding for c in candidates], dim=0
                )  # [N, D]
                scores = cosine_similarity(emb, candidate_embs)  # [N]

                score_values = tensor_to_float_list(scores)
                best_idx = max(range(len(score_values)), key=score_values.__getitem__)
                best_score = score_values[best_idx]

                if best_score >= self.threshold:
                    best_candidate = candidates[best_idx]
                    results[i] = MatchResult(
                        matched=True,
                        source_task_id=best_candidate.task_id,
                        cached_k=best_candidate.top_k,
                        cached_v=best_candidate.top_v,
                        similarity_score=best_score,
                    )
                else:
                    results[i] = MatchResult(matched=False)
        else:
            # -- Large batch: LSH bucket + KV merge ------------------------
            # Group batch indices by hash.
            hash_to_indices: dict[int, list[int]] = {}
            for i in range(batch_size):
                hsh = hash_values[i]
                hash_to_indices.setdefault(hsh, []).append(i)

            for hsh, indices in hash_to_indices.items():
                candidates = kv_manager.lookup_by_hash(hsh)
                if not candidates:
                    for idx in indices:
                        results[idx] = MatchResult(matched=False)
                    continue

                merged_k, merged_v = self._merge_kv(candidates)
                for idx in indices:
                    results[idx] = MatchResult(
                        matched=True,
                        source_task_id=None,  # merged — no single source
                        cached_k=merged_k,
                        cached_v=merged_v,
                        similarity_score=None,
                    )

        return results

    @staticmethod
    def _merge_kv(tasks: list[CachedTask]) -> tuple[torch.Tensor, torch.Tensor]:
        """Average K and V across all tasks in the same LSH bucket.

        Parameters
        ----------
        tasks:
            Non-empty list of CachedTask in the same bucket.

        Returns
        -------
        ``(merged_k, merged_v)`` — element-wise mean of top-layer K and V.
        """
        if not tasks:
            raise ValueError("_merge_kv requires at least one task")

        ks = torch.stack([t.top_k for t in tasks])  # [N, ...]
        vs = torch.stack([t.top_v for t in tasks])
        return ks.mean(dim=0), vs.mean(dim=0)
