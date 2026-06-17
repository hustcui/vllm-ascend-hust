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

"""Internal helpers for Sim-LLM control-plane tensor materialization."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def tensor_to_int_list(values: torch.Tensor | Sequence[int]) -> list[int]:
    """Materialize an integer tensor or sequence as a Python ``list[int]``.

    Callers use this at control-plane boundaries to avoid many repeated
    ``tensor.item()`` synchronizations on NPU tensors.
    """
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return []
        return [int(v) for v in values.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in values]


def tensor_to_float_list(values: torch.Tensor | Sequence[float]) -> list[float]:
    """Materialize a float tensor or sequence as a Python ``list[float]``."""
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return []
        return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
    return [float(v) for v in values]


def tensor_to_int_matrix(values: torch.Tensor) -> list[list[int]]:
    """Materialize a rank-2 integer tensor as ``list[list[int]]``."""
    if values.ndim != 2:
        raise ValueError("tensor_to_int_matrix expects a rank-2 tensor")
    return [[int(v) for v in row] for row in values.detach().cpu().tolist()]


def cumsum_to_ranges(query_start_loc: torch.Tensor | Sequence[int]) -> list[tuple[int, int]]:
    """Convert cumulative token offsets into ``(start, end)`` ranges."""
    offsets = tensor_to_int_list(query_start_loc)
    return list(zip(offsets[:-1], offsets[1:]))
