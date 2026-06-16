#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for KVManager — LRU eviction, bucket indexing, CRUD."""

from __future__ import annotations

import time

import pytest
import torch

from vllm_ascend.simllm.kv_manager import CachedTask, KVManager


def _make_task(task_id: str, lsh_hash: int = 0) -> CachedTask:
    """Create a minimal CachedTask for testing."""
    return CachedTask(
        task_id=task_id,
        embedding=torch.randn(1, 16),
        lsh_hash=lsh_hash,
        top_k=torch.randn(1, 4, 128, 64),
        top_v=torch.randn(1, 4, 128, 64),
        last_access_time=time.monotonic(),
        seq_len=128,
    )


class TestKVManagerBasics:
    """Store, lookup, get_kv, size, and clear."""

    def test_store_and_retrieve_one_task(self):
        mgr = KVManager(max_cache_size=10)
        task = _make_task("req-1", lsh_hash=42)
        mgr.store(task)
        assert mgr.size() == 1

        results = mgr.lookup_by_hash(42)
        assert len(results) == 1
        assert results[0].task_id == "req-1"

        k, v = mgr.get_kv("req-1")
        assert k is not None and v is not None
        assert torch.equal(k, task.top_k)
        assert torch.equal(v, task.top_v)

    def test_lookup_empty_bucket(self):
        mgr = KVManager()
        results = mgr.lookup_by_hash(99)
        assert results == []

    def test_lookup_updates_access_time(self):
        mgr = KVManager()
        t1 = _make_task("a", lsh_hash=1)
        old_ts = t1.last_access_time
        mgr.store(t1)
        time.sleep(0.001)  # ensure monotonic tick
        mgr.lookup_by_hash(1)
        assert mgr._cache["a"].last_access_time > old_ts

    def test_get_kv_nonexistent(self):
        mgr = KVManager()
        assert mgr.get_kv("nope") is None

    def test_clear_resets_all_state(self):
        mgr = KVManager()
        mgr.store(_make_task("a", lsh_hash=0))
        mgr.store(_make_task("b", lsh_hash=1))
        assert mgr.size() == 2
        mgr.clear()
        assert mgr.size() == 0
        assert mgr._buckets == {}


class TestLRUEviction:
    """LRU eviction when cache exceeds max_cache_size."""

    def test_eviction_when_full(self):
        mgr = KVManager(max_cache_size=3)
        mgr.store(_make_task("a", lsh_hash=0))
        mgr.store(_make_task("b", lsh_hash=0))
        mgr.store(_make_task("c", lsh_hash=0))
        # a was least recently accessed (stored first, never touched).
        mgr.store(_make_task("d", lsh_hash=0))  # triggers evict of 'a'
        assert mgr.size() == 3
        assert mgr.get_kv("a") is None
        assert mgr.get_kv("b") is not None
        assert mgr.get_kv("c") is not None
        assert mgr.get_kv("d") is not None

    def test_lru_access_refreshes_position(self):
        mgr = KVManager(max_cache_size=3)
        mgr.store(_make_task("a", lsh_hash=0))
        mgr.store(_make_task("b", lsh_hash=1))  # different bucket
        mgr.store(_make_task("c", lsh_hash=0))
        # Access bucket 0: refreshes 'a' and 'c'. Both move to MRU end.
        # Order before: a→b→c. After move_to_end(a): b→c→a. After move_to_end(c): b→a→c.
        # 'b' (hash 1, untouched) is now LRU.
        mgr.lookup_by_hash(0)
        mgr.store(_make_task("d", lsh_hash=2))  # triggers evict of 'b' (now LRU)
        assert mgr.get_kv("a") is not None  # survived
        assert mgr.get_kv("b") is None  # evicted
        assert mgr.get_kv("c") is not None
        assert mgr.get_kv("d") is not None

    def test_no_eviction_under_limit(self):
        mgr = KVManager(max_cache_size=10)
        for i in range(10):
            mgr.store(_make_task(str(i)))
        assert mgr.size() == 10
        for i in range(10):
            assert mgr.get_kv(str(i)) is not None


class TestBucketIndexing:
    """LSH bucket index correctness."""

    def test_same_hash_same_bucket(self):
        mgr = KVManager()
        mgr.store(_make_task("a", lsh_hash=5))
        mgr.store(_make_task("b", lsh_hash=5))
        results = mgr.lookup_by_hash(5)
        assert len(results) == 2
        ids = {r.task_id for r in results}
        assert ids == {"a", "b"}

    def test_different_hashes_different_buckets(self):
        mgr = KVManager()
        mgr.store(_make_task("a", lsh_hash=1))
        mgr.store(_make_task("b", lsh_hash=2))
        mgr.store(_make_task("c", lsh_hash=1))
        assert len(mgr.lookup_by_hash(1)) == 2
        assert len(mgr.lookup_by_hash(2)) == 1
        assert len(mgr.lookup_by_hash(99)) == 0

    def test_bucket_cleanup_on_eviction(self):
        mgr = KVManager(max_cache_size=2)
        mgr.store(_make_task("a", lsh_hash=7))
        mgr.store(_make_task("b", lsh_hash=7))
        mgr.store(_make_task("c", lsh_hash=7))  # evicts 'a'
        results = mgr.lookup_by_hash(7)
        assert len(results) == 2
        ids = {r.task_id for r in results}
        assert ids == {"b", "c"}

    def test_bucket_key_removed_when_empty(self):
        mgr = KVManager(max_cache_size=1)
        mgr.store(_make_task("only", lsh_hash=77))
        assert 77 in mgr._buckets
        mgr.store(_make_task("other", lsh_hash=88))  # evicts 'only'
        assert 77 not in mgr._buckets


class TestConcurrentSafety:
    """Single-process safety — no race conditions within one process."""

    def test_overwrite_updates_bucket(self):
        mgr = KVManager()
        mgr.store(_make_task("a", lsh_hash=10))
        mgr.store(_make_task("a", lsh_hash=20))  # overwrite with new hash
        assert mgr.size() == 1
        assert len(mgr.lookup_by_hash(10)) == 0
        assert len(mgr.lookup_by_hash(20)) == 1

    def test_store_many_sequential(self):
        """No errors when storing many tasks sequentially."""
        mgr = KVManager(max_cache_size=128)
        for i in range(256):
            mgr.store(_make_task(f"t{i}", lsh_hash=i % 16))
        assert mgr.size() == 128  # at capacity
