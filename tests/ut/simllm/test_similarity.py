#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for SimilarityIdentifier — adaptive matching strategies."""

from __future__ import annotations

import time

import pytest
import torch

from vllm_ascend.simllm.kv_manager import CachedTask, KVManager
from vllm_ascend.simllm.similarity import SimilarityIdentifier


def _make_task(
    task_id: str, embedding: torch.Tensor, embedding_dim: int = 128
) -> CachedTask:
    """Create a CachedTask with given embedding and auto-generated KV."""
    from vllm_ascend.simllm.lsh import SimHashHasher

    emb = embedding / embedding.norm(dim=-1, keepdim=True)  # L2-normalize
    hasher = SimHashHasher(dim=embedding_dim, num_bits=64)
    h = hasher.hash(emb.unsqueeze(0)).item()
    return CachedTask(
        task_id=task_id,
        embedding=emb.unsqueeze(0),
        lsh_hash=int(h),
        top_k=torch.randn(1, 4, 64, 64),
        top_v=torch.randn(1, 4, 64, 64),
        last_access_time=time.monotonic(),
        seq_len=64,
    )


class TestSmallBatch:
    """Exhaustive cosine strategy (batch_size < lsh_batch_threshold)."""

    @pytest.fixture
    def identifier(self):
        return SimilarityIdentifier(
            cosine_threshold=0.8,
            lsh_batch_threshold=16,  # small batches below 16
            lsh_num_bits=64,
            embedding_dim=128,
        )

    def test_correct_match_above_threshold(self, identifier):
        mgr = KVManager(max_cache_size=10)
        # Store a task with embedding close to origin.
        base_emb = torch.randn(128)
        base_emb = base_emb / base_emb.norm()
        mgr.store(_make_task("base", base_emb))

        # Query with identical embedding (guaranteed same LSH hash).
        # Perturbation tolerance is tested separately in the LSH test suite.
        query_emb = base_emb.clone()
        query_hash = identifier.lsh_hasher.hash(query_emb.unsqueeze(0))

        results = identifier.identify(
            query_emb.unsqueeze(0),  # [1, D]
            query_hash,  # [1]
            mgr,
        )
        assert results[0].matched
        assert results[0].source_task_id == "base"
        assert results[0].similarity_score is not None
        assert results[0].similarity_score >= 0.8

    def test_no_match_below_threshold(self, identifier):
        mgr = KVManager(max_cache_size=10)
        base_emb = torch.randn(128)
        base_emb = base_emb / base_emb.norm()
        mgr.store(_make_task("base", base_emb))

        # Query with nearly opposite embedding.
        query_emb = -base_emb + 0.01 * torch.randn(128)
        query_emb = query_emb / query_emb.norm()
        query_hash = identifier.lsh_hasher.hash(query_emb.unsqueeze(0))

        results = identifier.identify(
            query_emb.unsqueeze(0),
            query_hash,
            mgr,
        )
        assert not results[0].matched

    def test_empty_bucket_no_match(self, identifier):
        mgr = KVManager()
        emb = torch.randn(128)
        emb = emb / emb.norm()
        h = identifier.lsh_hasher.hash(emb.unsqueeze(0))
        results = identifier.identify(emb.unsqueeze(0), h, mgr)
        assert not results[0].matched

    def test_exact_threshold_boundary_matched(self, identifier):
        """Similarity exactly at threshold should be matched."""
        # Override threshold to 0.8.
        identifier.threshold = 0.8
        mgr = KVManager()
        # Two identical embeddings.
        emb = torch.randn(128)
        emb = emb / emb.norm()
        mgr.store(_make_task("a", emb))
        h = identifier.lsh_hasher.hash(emb.unsqueeze(0))
        results = identifier.identify(emb.unsqueeze(0), h, mgr)
        assert results[0].matched  # score ≈ 1.0 >= 0.8


class TestLargeBatch:
    """LSH bucket + KV merge strategy (batch_size >= lsh_batch_threshold)."""

    @pytest.fixture
    def identifier(self):
        return SimilarityIdentifier(
            cosine_threshold=0.8,
            lsh_batch_threshold=4,  # large batches at 4+
            lsh_num_bits=64,
            embedding_dim=128,
        )

    def test_same_bucket_produces_merge(self, identifier):
        mgr = KVManager(max_cache_size=32)
        # Pre-populate: 3 tasks with different embeddings but same LSH hash.
        # (SimHash is coarse; we force same hash by using the same embedding.)
        emb = torch.randn(128)
        emb = emb / emb.norm()
        mgr.store(_make_task("t0", emb))
        mgr.store(_make_task("t1", emb))
        mgr.store(_make_task("t2", emb))

        # Batch of 4 with same embedding.
        batch_emb = emb.unsqueeze(0).expand(4, -1)
        batch_hash = identifier.lsh_hasher.hash(batch_emb)

        results = identifier.identify(batch_emb, batch_hash, mgr)
        for i in range(4):
            assert results[i].matched
            # Merged => no single source_task_id.
            assert results[i].source_task_id is None
            assert results[i].cached_k is not None
            assert results[i].cached_v is not None
            assert results[i].similarity_score is None

    def test_empty_bucket_no_match_large(self, identifier):
        mgr = KVManager()
        embs = torch.randn(5, 128)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        hashes = identifier.lsh_hasher.hash(embs)
        results = identifier.identify(embs, hashes, mgr)
        for i in range(5):
            assert not results[i].matched


class TestMergeKV:
    """KV merge helper correctness."""

    def test_merge_averages_correctly(self):
        from vllm_ascend.simllm.similarity import SimilarityIdentifier

        tasks = [
            _make_task(f"t{i}", torch.randn(128))
            for i in range(3)
        ]
        # Override top_k/top_v with known values.
        for i, t in enumerate(tasks):
            t.top_k = torch.ones(1, 4, 8, 8) * float(i + 1)
            t.top_v = torch.ones(1, 4, 8, 8) * float((i + 1) * 10)

        mk, mv = SimilarityIdentifier._merge_kv(tasks)
        # k: mean of [1, 2, 3] = 2
        assert torch.allclose(mk, torch.full_like(mk, 2.0))
        # v: mean of [10, 20, 30] = 20
        assert torch.allclose(mv, torch.full_like(mv, 20.0))

    def test_single_element_merge_is_identity(self):
        from vllm_ascend.simllm.similarity import SimilarityIdentifier

        task = _make_task("only", torch.randn(128))
        mk, mv = SimilarityIdentifier._merge_kv([task])
        assert torch.equal(mk, task.top_k)
        assert torch.equal(mv, task.top_v)

    def test_empty_raises(self):
        from vllm_ascend.simllm.similarity import SimilarityIdentifier

        with pytest.raises(ValueError):
            SimilarityIdentifier._merge_kv([])
