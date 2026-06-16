#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for KVReuseEngine — KV alignment, truncation, padding, block writes."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.simllm.kv_reuse import KVReuseEngine


class TestKVReuseEngine:
    """Core KV alignment and injection logic."""

    @pytest.fixture
    def engine(self) -> KVReuseEngine:
        return KVReuseEngine(block_size=16, num_kv_heads=4, head_size=64)

    @pytest.fixture
    def sample_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached KV: seq_len=20, 4 heads, head_dim=64."""
        k = torch.randn(1, 4, 20, 64)
        v = torch.randn(1, 4, 20, 64)
        return k, v

    # -- prepare_injection (truncation / padding) ----------------------------

    def test_exact_length_no_change(self, engine, sample_kv):
        k, v = sample_kv  # L=20
        k2, v2 = engine.prepare_injection(k, v, 20)
        assert torch.equal(k2, k)
        assert torch.equal(v2, v)

    def test_truncation(self, engine, sample_kv):
        k, v = sample_kv
        k2, v2 = engine.prepare_injection(k, v, 10)
        assert k2.shape == (1, 4, 10, 64)
        assert v2.shape == (1, 4, 10, 64)
        assert torch.equal(k2, k[:, :, :10, :])
        assert torch.equal(v2, v[:, :, :10, :])

    def test_padding(self, engine, sample_kv):
        k, v = sample_kv  # L=20
        k2, v2 = engine.prepare_injection(k, v, 30)
        assert k2.shape == (1, 4, 30, 64)
        assert v2.shape == (1, 4, 30, 64)
        # First 20 positions unchanged.
        assert torch.equal(k2[:, :, :20, :], k)
        # Last 10 positions are zeros.
        assert (k2[:, :, 20:, :] == 0).all()

    def test_single_token(self, engine):
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        k2, v2 = engine.prepare_injection(k, v, 5)
        assert k2.shape == (1, 4, 5, 64)
        assert (k2[:, :, 1:, :] == 0).all()

    # -- num_blocks_needed ---------------------------------------------------

    def test_num_blocks_exact_fit(self, engine):
        assert KVReuseEngine.num_blocks_needed(128, 128) == 1
        assert KVReuseEngine.num_blocks_needed(256, 128) == 2
        assert KVReuseEngine.num_blocks_needed(127, 128) == 1

    def test_num_blocks_partial(self, engine):
        assert KVReuseEngine.num_blocks_needed(129, 128) == 2
        assert KVReuseEngine.num_blocks_needed(1, 128) == 1

    def test_num_blocks_custom_size(self, engine):
        assert KVReuseEngine.num_blocks_needed(100, 64) == 2
        assert KVReuseEngine.num_blocks_needed(64, 64) == 1

    # -- write_to_cache ------------------------------------------------------

    def test_write_single_block(self, engine):
        block_size = 16
        k_cache = torch.zeros(4, block_size, 4, 64)  # 4 blocks
        v_cache = torch.zeros(4, block_size, 4, 64)
        k = torch.ones(1, 4, 8, 64)  # 8 tokens
        v = torch.ones(1, 4, 8, 64) * 2
        engine.write_to_cache(k_cache, v_cache, [2], k, v)
        # Block 2, first 8 slots → should be filled.
        assert (k_cache[2, :8, :, :] == 1).all()
        assert (v_cache[2, :8, :, :] == 2).all()
        # Block 2, remaining 8 slots → still zero.
        assert (k_cache[2, 8:, :, :] == 0).all()
        # Other blocks untouched.
        assert (k_cache[0] == 0).all()
        assert (k_cache[1] == 0).all()

    def test_write_multi_block(self, engine):
        block_size = 16
        k_cache = torch.zeros(5, block_size, 4, 64)
        v_cache = torch.zeros(5, block_size, 4, 64)
        k = torch.randn(1, 4, 40, 64)  # 40 tokens → 3 blocks
        v = torch.randn(1, 4, 40, 64)
        engine.write_to_cache(k_cache, v_cache, [0, 1, 3], k, v)
        # Reconstruct full sequence from blocks.
        k_recon = torch.cat([
            k_cache[0, :, :, :],
            k_cache[1, :, :, :],
            k_cache[3, :8, :, :],  # 40 = 16+16+8
        ])
        k_flat = k.squeeze(0).permute(1, 0, 2)  # [L, H, D]
        assert torch.allclose(k_recon, k_flat)

    def test_write_partial_last_block(self, engine):
        block_size = 16
        k_cache = torch.zeros(3, block_size, 4, 64)
        v_cache = torch.zeros(3, block_size, 4, 64)
        k = torch.randn(1, 4, 33, 64)  # 33 tokens → 2 full + 1 partial
        v = torch.randn(1, 4, 33, 64)
        engine.write_to_cache(k_cache, v_cache, [0, 1, 2], k, v)
        # Block 2 should have only 1 filled slot.
        assert not (k_cache[2, 0, :, :] == 0).all()  # slot 0 written
        assert (k_cache[2, 1:, :, :] == 0).all()  # slots 1-15 empty

    # -- _align_length edge cases --------------------------------------------

    def test_align_zero_target(self, engine):
        k = torch.randn(1, 4, 10, 64)
        result = engine.prepare_injection(k, k.clone(), 0)
        assert result[0].shape == (1, 4, 0, 64)

    def test_align_different_dtype(self, engine):
        k = torch.randn(1, 4, 5, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 5, 64, dtype=torch.float16)
        k2, v2 = engine.prepare_injection(k, v, 10)
        assert k2.dtype == torch.float16
        assert k2[:, :, 5:, :].sum() == 0  # padded zeros in fp16

    # -- gather_from_cache ---------------------------------------------------

    def test_gather_single_block(self, engine):
        """Gather KV from a single cache block."""
        num_blocks, block_size, nh, hs = 4, 16, 4, 64
        k_cache = torch.randn(num_blocks, block_size, nh, hs)
        # Write known data to block 2.
        k_data = torch.randn(1, nh, 8, hs)  # 8 tokens
        k_flat = k_data.squeeze(0).permute(1, 0, 2)  # [L, H, D]
        k_cache[2, :8, :, :] = k_flat

        gathered = KVReuseEngine.gather_from_cache(k_cache, [2], 8, block_size)
        assert gathered.shape == (1, nh, 8, hs)
        assert torch.allclose(gathered, k_data, atol=1e-6)

    def test_gather_multi_block(self, engine):
        """Gather KV spanning multiple cache blocks."""
        num_blocks, block_size, nh, hs = 5, 16, 4, 64
        k_cache = torch.randn(num_blocks, block_size, nh, hs)
        k_data = torch.randn(1, nh, 40, hs)  # 40 tokens → 3 blocks (16+16+8)
        k_flat = k_data.squeeze(0).permute(1, 0, 2)
        k_cache[0, :, :, :] = k_flat[:16]
        k_cache[1, :, :, :] = k_flat[16:32]
        k_cache[3, :8, :, :] = k_flat[32:]

        gathered = KVReuseEngine.gather_from_cache(k_cache, [0, 1, 3], 40, block_size)
        assert gathered.shape == (1, nh, 40, hs)
        assert torch.allclose(gathered, k_data, atol=1e-6)

    def test_gather_partial_last_block(self, engine):
        """Gather where the last block is only partially filled."""
        num_blocks, block_size, nh, hs = 3, 16, 4, 64
        k_cache = torch.randn(num_blocks, block_size, nh, hs)
        k_data = torch.randn(1, nh, 33, hs)  # 33 tokens → 2 blocks + 1 partial
        k_flat = k_data.squeeze(0).permute(1, 0, 2)
        k_cache[0] = k_flat[:16]
        k_cache[1] = k_flat[16:32]
        k_cache[2, :1, :, :] = k_flat[32:33]
        k_cache[2, 1:, :, :] = 0  # rest is zero

        gathered = KVReuseEngine.gather_from_cache(k_cache, [0, 1, 2], 33, block_size)
        assert gathered.shape == (1, nh, 33, hs)
        assert torch.allclose(gathered, k_data, atol=1e-6)

    def test_inject_and_gather_roundtrip(self, engine):
        """Write KV to cache via write_to_cache, read back via gather_from_cache."""
        num_blocks, block_size, nh, hs = 6, 16, 4, 64
        k_cache = torch.zeros(num_blocks, block_size, nh, hs)
        v_cache = torch.zeros(num_blocks, block_size, nh, hs)

        k_orig = torch.randn(1, nh, 42, hs)
        v_orig = torch.randn(1, nh, 42, hs)

        block_ids = [1, 3, 4]
        engine.write_to_cache(k_cache, v_cache, block_ids, k_orig, v_orig)

        # Read back from the same blocks.
        k_read = KVReuseEngine.gather_from_cache(k_cache, block_ids, 42, block_size)
        v_read = KVReuseEngine.gather_from_cache(v_cache, block_ids, 42, block_size)

        assert torch.allclose(k_read, k_orig, atol=1e-6)
        assert torch.allclose(v_read, v_orig, atol=1e-6)

    def test_gather_empty_blocks(self, engine):
        """Gather with empty block_ids returns zero-shaped tensor."""
        k_cache = torch.randn(4, 16, 4, 64)
        gathered = KVReuseEngine.gather_from_cache(k_cache, [], 10, 16)
        assert gathered.shape == (1, 4, 0, 64)

    def test_gather_zero_seq_len(self, engine):
        """Gather with seq_len=0 returns empty sequence dimension."""
        k_cache = torch.randn(4, 16, 4, 64)
        gathered = KVReuseEngine.gather_from_cache(k_cache, [0, 1], 0, 16)
        assert gathered.shape == (1, 4, 0, 64)
