#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for SimHash LSH — determinism, collisions, cosine similarity."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.simllm.lsh import SimHashHasher, cosine_similarity


class TestSimHashHasher:
    """SimHash determinism, collision rate, hash size."""

    @pytest.fixture(autouse=True)
    def seed_everything(self):
        torch.manual_seed(42)

    def test_same_input_same_hash(self):
        hasher = SimHashHasher(dim=64, num_bits=64)
        emb = torch.randn(4, 64)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        h1 = hasher.hash(emb)
        h2 = hasher.hash(emb)
        assert torch.equal(h1, h2)

    def test_different_inputs_different_hash(self):
        hasher = SimHashHasher(dim=64, num_bits=64)
        emb_a = torch.randn(1, 64)
        emb_a = torch.nn.functional.normalize(emb_a, dim=-1)
        emb_b = -emb_a  # maximally dissimilar
        h_a = hasher.hash(emb_a)
        h_b = hasher.hash(emb_b)
        assert h_a.item() != h_b.item()

    def test_collision_rate_below_5_percent(self):
        """Verify random embeddings rarely collide (< 5%)."""
        hasher = SimHashHasher(dim=128, num_bits=64)
        embs = torch.randn(1000, 128)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        hashes = hasher.hash(embs)
        unique_frac = len(set(hashes.tolist())) / len(hashes)
        assert unique_frac > 0.95  # < 5% collisions

    def test_output_shape(self):
        hasher = SimHashHasher(dim=32, num_bits=32)
        emb = torch.randn(7, 32)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        h = hasher.hash(emb)
        assert h.shape == (7,)
        assert h.dtype == torch.int64

    def test_reproducible_with_same_seed(self):
        emb = torch.randn(3, 32)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        h1 = SimHashHasher(dim=32, num_bits=64, seed=123).hash(emb)
        h2 = SimHashHasher(dim=32, num_bits=64, seed=123).hash(emb)
        assert torch.equal(h1, h2)

    def test_different_seed_different_projection(self):
        emb = torch.randn(3, 32)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        h1 = SimHashHasher(dim=32, num_bits=64, seed=1).hash(emb)
        h2 = SimHashHasher(dim=32, num_bits=64, seed=2).hash(emb)
        # Should differ with high probability.
        assert not torch.equal(h1, h2)

    def test_num_bits_16_and_128(self):
        """Smoke test that non-default bit counts work."""
        emb = torch.randn(1, 64)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        h16 = SimHashHasher(dim=64, num_bits=16).hash(emb)
        h128 = SimHashHasher(dim=64, num_bits=128).hash(emb)
        assert h16.shape == (1,)
        assert h128.shape == (1,)
        assert h16.dtype == torch.int64
        assert h128.dtype == torch.int64
        assert h16.item() != h128.item()  # Different projection sizes


class TestCosineSimilarity:
    """Cosine similarity helper correctness."""

    def test_identical_vectors(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        a = torch.nn.functional.normalize(a, dim=-1)
        score = cosine_similarity(a, a.unsqueeze(0))
        assert torch.allclose(score, torch.tensor([1.0]), atol=1e-6)

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.tensor([0.0, 1.0])
        b = torch.nn.functional.normalize(b, dim=-1)
        score = cosine_similarity(a, b.unsqueeze(0))
        assert torch.allclose(score, torch.tensor([0.0]), atol=1e-6)

    def test_opposite_vectors(self):
        a = torch.tensor([1.0, 0.0])
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.tensor([-1.0, 0.0])
        b = torch.nn.functional.normalize(b, dim=-1)
        score = cosine_similarity(a, b.unsqueeze(0))
        assert torch.allclose(score, torch.tensor([-1.0]), atol=1e-6)

    def test_matches_torch_cosine(self):
        a = torch.randn(256)
        b = torch.randn(8, 256)
        a_norm = torch.nn.functional.normalize(a, dim=-1)
        b_norm = torch.nn.functional.normalize(b, dim=-1)
        our_score = cosine_similarity(a_norm, b_norm)
        ref_score = torch.nn.functional.cosine_similarity(
            a_norm.unsqueeze(0), b_norm
        )
        assert torch.allclose(our_score, ref_score, atol=1e-6)
