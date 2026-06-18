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
from typing import Any

import torch


_INPUT_EMBEDDING_PATHS = (
    "model.embed_tokens",
    "model.model.embed_tokens",
    "language_model.model.embed_tokens",
    "transformer.wte",
)


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


def _looks_like_embedding_layer(layer: Any) -> bool:
    if layer is None or not callable(layer):
        return False
    weight = getattr(layer, "weight", None)
    return isinstance(weight, torch.Tensor)


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        try:
            cur = getattr(cur, part)
        except AttributeError:
            return None
        if cur is None:
            return None
    return cur


def resolve_input_embedding_layer(model: Any) -> Any:
    """Resolve the token embedding layer from HF-style or vLLM-wrapped models.

    vLLM model wrappers do not always expose ``get_input_embeddings()`` even
    when the underlying model has a standard token embedding module.  Sim-LLM
    uses this only at the control-plane preprocessing boundary, so resolving a
    small set of common wrapper paths keeps the runtime path model-agnostic.
    """
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            layer = getter()
        except (AttributeError, NotImplementedError):
            layer = None
        if _looks_like_embedding_layer(layer):
            return layer

    for path in _INPUT_EMBEDDING_PATHS:
        layer = _resolve_attr_path(model, path)
        if _looks_like_embedding_layer(layer):
            return layer

    tried = ", ".join(("get_input_embeddings()", *_INPUT_EMBEDDING_PATHS))
    raise AttributeError(
        "SimLLM could not resolve a token embedding layer from model "
        f"{type(model).__name__}; tried: {tried}"
    )
