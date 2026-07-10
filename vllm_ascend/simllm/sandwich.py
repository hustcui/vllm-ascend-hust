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

"""SandwichConfig — layer-selective KV retention for unmatched tasks.

Retain KV for bottom-N (early) and top-N (late) transformer layers only;
discard intermediate-layer KV to reduce memory consumption.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SandwichConfig:
    """Layer-selective KV retention configuration.

    For a 32-layer model with bottom=top=3, only layers {0,1,2,29,30,31}
    retain KV in the BlockTable — 6/32 = 18.75% of normal usage.

    Parameters
    ----------
    bottom_layers:
        Number of early layers whose KV is retained (default 3).
    top_layers:
        Number of late layers whose KV is retained (default 3).
    num_layers:
        Total number of transformer layers in the model.
        Must be >= bottom_layers + top_layers.
    """

    bottom_layers: int = 3
    top_layers: int = 3
    num_layers: int = 32

    def __post_init__(self) -> None:
        if self.bottom_layers < 0 or self.top_layers < 0:
            raise ValueError(
                f"bottom_layers ({self.bottom_layers}) and top_layers "
                f"({self.top_layers}) must be non-negative"
            )
        if self.num_layers < 0:
            raise ValueError(
                f"num_layers ({self.num_layers}) must be non-negative"
            )
        # Note: bottom_layers + top_layers may exceed num_layers — the
        # set union in keep_layers naturally handles overlap.  This is
        # useful for caching more/all layers when memory permits.

    @property
    def keep_layers(self) -> set[int]:
        """Return the set of layer indices whose KV should be retained."""
        return set(range(self.bottom_layers)) | set(
            range(self.num_layers - self.top_layers, self.num_layers)
        )

    def should_cache(self, layer_idx: int) -> bool:
        """Return True if this layer's KV should be written to BlockTable."""
        return layer_idx in self.keep_layers

    @property
    def retention_fraction(self) -> float:
        """Fraction of layers whose KV is retained."""
        if self.num_layers == 0:
            return 0.0
        return len(self.keep_layers) / self.num_layers
