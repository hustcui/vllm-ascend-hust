#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for SandwichConfig — layer-selective KV retention."""

from __future__ import annotations

import pytest

from vllm_ascend.simllm.sandwich import SandwichConfig


class TestSandwichConfig:
    """Layer keep set, should_cache, retention fraction."""

    def test_bottom_layers_in_keep_set(self):
        sc = SandwichConfig(bottom_layers=3, top_layers=0, num_layers=32)
        for layer in (0, 1, 2):
            assert layer in sc.keep_layers

    def test_top_layers_in_keep_set(self):
        sc = SandwichConfig(bottom_layers=0, top_layers=3, num_layers=32)
        for layer in (29, 30, 31):
            assert layer in sc.keep_layers

    def test_middle_layers_not_in_keep_set(self):
        sc = SandwichConfig(bottom_layers=3, top_layers=3, num_layers=32)
        for layer in range(3, 29):
            assert layer not in sc.keep_layers

    def test_default_config(self):
        sc = SandwichConfig()
        expected = set(range(3)) | set(range(29, 32))
        assert sc.keep_layers == expected

    def test_should_cache(self):
        sc = SandwichConfig(bottom_layers=2, top_layers=2, num_layers=12)
        assert sc.should_cache(0) is True
        assert sc.should_cache(1) is True
        assert sc.should_cache(2) is False
        assert sc.should_cache(9) is False
        assert sc.should_cache(10) is True
        assert sc.should_cache(11) is True

    def test_retention_fraction_default(self):
        sc = SandwichConfig(bottom_layers=3, top_layers=3, num_layers=32)
        assert abs(sc.retention_fraction - (6 / 32)) < 1e-6

    def test_retention_fraction_all_layers(self):
        sc = SandwichConfig(bottom_layers=32, top_layers=0, num_layers=32)
        assert abs(sc.retention_fraction - 1.0) < 1e-6

    def test_retention_fraction_zero(self):
        sc = SandwichConfig(bottom_layers=0, top_layers=0, num_layers=32)
        assert sc.retention_fraction == 0.0

    def test_retention_fraction_all_layers_overlapping(self):
        """bottom + top > num_layers → all layers covered (set union)."""
        sc = SandwichConfig(bottom_layers=4, top_layers=4, num_layers=6)
        # keep_layers = {0,1,2,3} ∪ {2,3,4,5} = {0,1,2,3,4,5}
        assert sc.keep_layers == set(range(6))
        assert abs(sc.retention_fraction - 1.0) < 1e-6

    def test_overlapping_sets_union(self):
        """When bottom+top > num_layers, overlapping layers appear once via set union."""
        sc = SandwichConfig(bottom_layers=20, top_layers=20, num_layers=32)
        # Overlap region: layers 12-19 are in both bottom and top
        # Bottom: {0..19}, Top: {12..31} → union is {0..31} = all layers
        assert len(sc.keep_layers) == 32  # all layers covered

    def test_validation_negative_raises(self):
        with pytest.raises(ValueError):
            SandwichConfig(bottom_layers=-1, top_layers=0, num_layers=32)

    def test_validation_negative_num_layers_raises(self):
        with pytest.raises(ValueError):
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=-1)

    def test_partial_coverage(self):
        sc = SandwichConfig(bottom_layers=4, top_layers=4, num_layers=32)
        assert sc.keep_layers == set(range(4)) | set(range(28, 32))
        assert len(sc.keep_layers) == 8
