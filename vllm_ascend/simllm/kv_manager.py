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

"""KVManager — LRU-evicted store for task embeddings, LSH hashes, and top-layer KV.

Operates at the *task/semantic* level, distinct from vLLM's token/block-level
BlockManager.  Uses ``collections.OrderedDict`` for O(1) LRU eviction and a
separate ``dict[int, list[str]]`` for LSH bucket indexing.
"""

from __future__ import annotations

import contextlib
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CachedTask:
    """A single task snapshot stored in KVManager.

    Attributes
    ----------
    task_id:
        Unique request/task identifier.
    embedding:
        Pooled task embedding ``[1, D]`` (L2-normalized).
    lsh_hash:
        Packed int64 SimHash of the embedding.
    top_k:
        Top-layer Key tensor ``[1, num_kv_heads, L_kv, head_dim]``.
    top_v:
        Top-layer Value tensor ``[1, num_kv_heads, L_kv, head_dim]``.
    last_access_time:
        Monotonic timestamp for LRU (seconds, from ``time.monotonic()``).
    seq_len:
        Original sequence length — used for shape compatibility checks.
    """

    task_id: str
    embedding: torch.Tensor
    lsh_hash: int
    top_k: torch.Tensor
    top_v: torch.Tensor
    last_access_time: float
    seq_len: int


class KVManager:
    """LRU-evicted store with LSH bucket indexing.

    Parameters
    ----------
    max_cache_size:
        Maximum number of CachedTask entries.  When exceeded the
        least-recently-accessed entry is evicted automatically.
    """

    def __init__(self, max_cache_size: int = 1024) -> None:
        self._max_cache_size = max_cache_size
        # OrderedDict for O(1) LRU: most-recently-used at the right end.
        self._cache: OrderedDict[str, CachedTask] = OrderedDict()
        # LSH bucket index: hash → [task_id, ...]
        self._buckets: dict[int, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, task: CachedTask) -> None:
        """Insert (or overwrite) a task and maintain LRU + bucket invariants.

        If the task_id already exists the old entry is replaced (including
        bucket membership).  If the cache is over capacity after insertion,
        ``evict_lru()`` is called.
        """
        # Remove old bucket entry if overwriting.
        if task.task_id in self._cache:
            old = self._cache[task.task_id]
            self._remove_from_bucket(old.lsh_hash, task.task_id)

        self._cache[task.task_id] = task
        self._cache.move_to_end(task.task_id)

        self._buckets.setdefault(task.lsh_hash, []).append(task.task_id)

        if len(self._cache) > self._max_cache_size:
            self.evict_lru()

    def lookup_by_hash(self, lsh_hash: int) -> list[CachedTask]:
        """Return all cached tasks whose LSH hash matches *exactly*.

        Updates ``last_access_time`` on every returned task (LRU refresh).
        """
        task_ids = self._buckets.get(lsh_hash, [])
        results: list[CachedTask] = []
        now = time.monotonic()
        for tid in task_ids:
            if tid in self._cache:
                task = self._cache[tid]
                task.last_access_time = now
                # Move to right end (most-recently-used).
                self._cache.move_to_end(tid)
                results.append(task)
        return results

    def get_kv(self, task_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return ``(top_k, top_v)`` for *task_id*, or None if not found.

        Access refreshes the LRU position.
        """
        task = self._cache.get(task_id)
        if task is None:
            return None
        task.last_access_time = time.monotonic()
        self._cache.move_to_end(task_id)
        return task.top_k, task.top_v

    def evict_lru(self) -> None:
        """Evict the least-recently-accessed entry (O(1))."""
        if not self._cache:
            return
        task_id, task = self._cache.popitem(last=False)
        self._remove_from_bucket(task.lsh_hash, task_id)

    def size(self) -> int:
        """Return the current number of cached tasks."""
        return len(self._cache)

    def clear(self) -> None:
        """Reset all state (useful for model reload / testing)."""
        self._cache.clear()
        self._buckets.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_from_bucket(self, lsh_hash: int, task_id: str) -> None:
        """Remove *task_id* from the bucket for *lsh_hash*.

        Cleans up the bucket list and the bucket key if it becomes empty.
        """
        bucket = self._buckets.get(lsh_hash)
        if bucket is None:
            return
        with contextlib.suppress(ValueError):
            bucket.remove(task_id)
        if not bucket:
            del self._buckets[lsh_hash]
