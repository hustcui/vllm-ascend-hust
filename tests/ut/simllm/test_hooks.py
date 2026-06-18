#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for Sim-LLM hooks — preprocess, identify, postprocess."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.simllm.hooks.identify import identify_batch
from vllm_ascend.simllm.hooks.postprocess import SimLLMPostprocessor
from vllm_ascend.simllm.hooks.preprocess import SimLLMPreprocessor
from vllm_ascend.simllm.kv_manager import KVManager
from vllm_ascend.simllm.lsh import SimHashHasher
from vllm_ascend.simllm.similarity import MatchResult, SimilarityIdentifier

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _MockEmbedding(torch.nn.Module):
    """Mock embedding layer: returns identity-like embeddings."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1000, dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids % self.weight.shape[0]]


def _make_mock_model(dim: int = 128):
    """Return a mock model with get_input_embeddings()."""
    model = MagicMock()
    model.get_input_embeddings.return_value = _MockEmbedding(dim)
    return model


class _MockBackbone:
    def __init__(self, dim: int = 128):
        self.embed_tokens = _MockEmbedding(dim)


class _MockVLLMModel:
    def __init__(self, dim: int = 128):
        self.model = _MockBackbone(dim)


# ---------------------------------------------------------------------------
# SimLLMPreprocessor
# ---------------------------------------------------------------------------


class TestSimLLMPreprocessor:
    """Token-embedding-based preprocessing."""

    def test_basic_extraction(self):
        preprocessor = SimLLMPreprocessor(pooling="mean")
        model = _make_mock_model(128)
        # 3 requests: 4, 5, 3 tokens → 12 total
        input_ids = torch.arange(12)
        query_start_loc = torch.tensor([0, 4, 9, 12])
        embs = preprocessor.extract_embeddings(model, input_ids, query_start_loc)
        assert embs.shape == (3, 128)
        # L2-normalized.
        norms = embs.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_vllm_wrapped_model_without_embedding_getter(self):
        preprocessor = SimLLMPreprocessor(pooling="mean")
        model = _MockVLLMModel(96)
        input_ids = torch.arange(8)
        query_start_loc = torch.tensor([0, 3, 8])
        embs = preprocessor.extract_embeddings(model, input_ids, query_start_loc)
        assert embs.shape == (2, 96)
        norms = embs.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_empty_batch(self):
        preprocessor = SimLLMPreprocessor()
        model = _make_mock_model(64)
        input_ids = torch.tensor([], dtype=torch.long)
        query_start_loc = torch.tensor([0])
        embs = preprocessor.extract_embeddings(model, input_ids, query_start_loc)
        assert embs.shape == (0, 64)

    def test_single_request(self):
        preprocessor = SimLLMPreprocessor(pooling="last")
        model = _make_mock_model(256)
        input_ids = torch.arange(7)
        query_start_loc = torch.tensor([0, 7])
        embs = preprocessor.extract_embeddings(model, input_ids, query_start_loc)
        assert embs.shape == (1, 256)

    def test_different_pooling_modes(self):
        model = _make_mock_model(64)
        input_ids = torch.arange(5)
        query_start_loc = torch.tensor([0, 5])
        for mode in ("mean", "last", "cls"):
            preprocessor = SimLLMPreprocessor(pooling=mode)
            embs = preprocessor.extract_embeddings(model, input_ids, query_start_loc)
            assert embs.shape == (1, 64)
            norms = embs.norm(p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_reproducibility(self):
        """Same input → same embedding every time."""
        preprocessor = SimLLMPreprocessor()
        model = _make_mock_model(128)
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        qsl = torch.tensor([0, 5])
        e1 = preprocessor.extract_embeddings(model, input_ids, qsl)
        e2 = preprocessor.extract_embeddings(model, input_ids, qsl)
        assert torch.equal(e1, e2)

    def test_different_inputs_different_embeddings(self):
        preprocessor = SimLLMPreprocessor()
        model = _make_mock_model(64)
        ids_a = torch.tensor([1, 2, 3])
        ids_b = torch.tensor([100, 200, 300])
        qsl = torch.tensor([0, 3])
        ea = preprocessor.extract_embeddings(model, ids_a, qsl)
        eb = preprocessor.extract_embeddings(model, ids_b, qsl)
        # Different token IDs → different embeddings (with high probability).
        sim = (ea * eb).sum()
        assert sim.item() < 0.99, f"Expected different embeddings, cos={sim.item():.4f}"


# ---------------------------------------------------------------------------
# identify_batch
# ---------------------------------------------------------------------------


class TestIdentifyBatch:
    """Thin wrapper around SimilarityIdentifier.identify()."""

    def test_empty_batch(self):
        kv_mgr = KVManager(max_cache_size=10)
        identifier = SimilarityIdentifier(embedding_dim=64)
        result = identify_batch(
            torch.zeros(0, 64),
            torch.zeros(0, dtype=torch.int64),
            kv_mgr,
            identifier,
        )
        assert result == {}

    def test_no_cached_tasks(self):
        kv_mgr = KVManager(max_cache_size=10)
        identifier = SimilarityIdentifier(embedding_dim=64, cosine_threshold=0.8)
        hasher = SimHashHasher(dim=64, num_bits=32)
        # Single request.
        emb = torch.randn(1, 64)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        hashes = hasher.hash(emb)
        result = identify_batch(emb, hashes, kv_mgr, identifier)
        # Nothing in cache → no match.
        assert result[0].matched is False

    def test_exact_match(self):
        """Store a task, then match an identical embedding → matched."""
        from vllm_ascend.simllm.kv_manager import CachedTask

        kv_mgr = KVManager(max_cache_size=10)
        identifier = SimilarityIdentifier(embedding_dim=64, cosine_threshold=0.8)
        hasher = SimHashHasher(dim=64, num_bits=32)

        emb = torch.randn(1, 64)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        hsh = int(hasher.hash(emb).item())

        # Pre-populate cache.
        task = CachedTask(
            task_id="task_a",
            embedding=emb.clone(),
            lsh_hash=hsh,
            top_k=torch.randn(1, 4, 10, 64),
            top_v=torch.randn(1, 4, 10, 64),
            last_access_time=time.monotonic(),
            seq_len=10,
        )
        kv_mgr.store(task)

        # Same embedding → should match.
        result = identify_batch(emb.clone(), torch.tensor([hsh]), kv_mgr, identifier)
        assert result[0].matched is True
        assert result[0].source_task_id == "task_a"


# ---------------------------------------------------------------------------
# SimLLMPostprocessor
# ---------------------------------------------------------------------------


class TestSimLLMPostprocessor:
    """KV storage + deferral logic."""

    @pytest.fixture
    def kv_mgr(self) -> KVManager:
        return KVManager(max_cache_size=10)

    @pytest.fixture
    def postprocessor(self, kv_mgr) -> SimLLMPostprocessor:
        return SimLLMPostprocessor(kv_mgr, pooling="mean")

    def test_store_batch(self, kv_mgr, postprocessor):
        hasher = SimHashHasher(dim=128, num_bits=32)
        # Simulate a batch of 2 requests.
        hs = torch.randn(10, 128)  # 10 tokens total
        qsl = torch.tensor([0, 6, 10])
        top_k = torch.randn(10, 8, 16)
        top_v = torch.randn(10, 8, 16)

        # Compute hashes from hidden states (simulating preprocess).
        from vllm_ascend.simllm.hooks.preprocess import SimLLMPreprocessor
        preprocessor = SimLLMPreprocessor(pooling="mean")
        model = _make_mock_model(128)
        input_ids = torch.arange(10)
        embs = preprocessor.extract_embeddings(model, input_ids, qsl)
        hashes = hasher.hash(embs)

        postprocessor.store_batch(
            req_ids=["req_0", "req_1"],
            hidden_states=hs,
            query_start_loc=qsl,
            batch_hashes=hashes,
            top_k=top_k,
            top_v=top_v,
        )
        assert kv_mgr.size() == 2

        # Verify both entries are retrievable.
        for rid in ("req_0", "req_1"):
            result = kv_mgr.get_kv(rid)
            assert result is not None
            k, v = result
            assert k.shape[0] == 1  # batch dim

    def test_store_batch_empty(self, kv_mgr, postprocessor):
        postprocessor.store_batch(
            req_ids=[],
            hidden_states=torch.randn(0, 64),
            query_start_loc=torch.tensor([0]),
            batch_hashes=torch.tensor([], dtype=torch.int64),
            top_k=torch.randn(0, 4, 64),
            top_v=torch.randn(0, 4, 64),
        )
        assert kv_mgr.size() == 0

    def test_lru_eviction_on_full_cache(self, kv_mgr):
        """Storing beyond max_cache_size evicts LRU entries."""
        postprocessor = SimLLMPostprocessor(kv_mgr, pooling="mean")
        hasher = SimHashHasher(dim=64, num_bits=32)
        # Fill cache with 10 entries.
        for i in range(10):
            hs = torch.randn(5, 64)
            qsl = torch.tensor([0, 5])
            top_k = torch.randn(5, 4, 16)
            top_v = torch.randn(5, 4, 16)
            embs = torch.randn(1, 64)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            hashes = hasher.hash(embs)
            postprocessor.store_batch(
                req_ids=[f"req_{i}"],
                hidden_states=hs,
                query_start_loc=qsl,
                batch_hashes=hashes,
                top_k=top_k,
                top_v=top_v,
            )
        assert kv_mgr.size() == 10

        # One more → evicts the oldest.
        hs = torch.randn(5, 64)
        qsl = torch.tensor([0, 5])
        top_k = torch.randn(5, 4, 16)
        top_v = torch.randn(5, 4, 16)
        embs = torch.randn(1, 64)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        hashes = hasher.hash(embs)
        postprocessor.store_batch(
            req_ids=["req_new"],
            hidden_states=hs,
            query_start_loc=qsl,
            batch_hashes=hashes,
            top_k=top_k,
            top_v=top_v,
        )
        # Still 10 (one evicted).
        assert kv_mgr.size() == 10
        # Oldest ("req_0") should be gone.
        assert kv_mgr.get_kv("req_0") is None

    # -- Deferral logic ------------------------------------------------------

    def test_no_deferral_when_low_match(self, postprocessor):
        match_results = {
            0: MatchResult(matched=True),
            1: MatchResult(matched=False),
            2: MatchResult(matched=False),
            3: MatchResult(matched=False),
        }
        deferrals = postprocessor.compute_deferrals(match_results, 4)
        # 1/4 = 25% matched, below 50% → no deferral.
        assert len(deferrals) == 0

    def test_deferral_when_high_match(self, postprocessor):
        match_results = {
            0: MatchResult(matched=True),
            1: MatchResult(matched=True),
            2: MatchResult(matched=True),
            3: MatchResult(matched=False),
        }
        deferrals = postprocessor.compute_deferrals(match_results, 4)
        # 3/4 = 75% matched, above 50% → defer index 3.
        assert deferrals == {3}

    def test_no_deferral_at_exact_threshold(self, postprocessor):
        match_results = {
            0: MatchResult(matched=True),
            1: MatchResult(matched=False),
            2: MatchResult(matched=True),
            3: MatchResult(matched=False),
        }
        deferrals = postprocessor.compute_deferrals(match_results, 4)
        # 2/4 = 50% → NOT above threshold (strict > ), no deferral.
        assert len(deferrals) == 0

    def test_deferral_respects_max_count(self, postprocessor):
        # max_deferrals = 3 (default).
        match_results = {
            0: MatchResult(matched=True),
            1: MatchResult(matched=False),
            2: MatchResult(matched=True),
        }
        # Index 1 already deferred 3 times → cannot defer again.
        deferrals = postprocessor.compute_deferrals(match_results, 3, {1: 3})
        assert deferrals == set()  # index 1 at max, so no deferrals

    def test_deferral_empty_batch(self, postprocessor):
        deferrals = postprocessor.compute_deferrals({}, 0)
        assert deferrals == set()

    def test_deferral_all_matched(self, postprocessor):
        match_results = {
            0: MatchResult(matched=True),
            1: MatchResult(matched=True),
        }
        deferrals = postprocessor.compute_deferrals(match_results, 2)
        # All matched → no unmatched tasks to defer.
        assert len(deferrals) == 0
