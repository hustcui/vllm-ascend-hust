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

_INPUT_EMBEDDING_CALLABLE_PATHS = (
    "embed_input_ids",
    "model.embed_input_ids",
    "language_model.embed_input_ids",
)

_HIDDEN_SIZE_PATHS = (
    "config.hidden_size",
    "model.config.hidden_size",
    "model.model.config.hidden_size",
    "language_model.config.hidden_size",
    "language_model.model.config.hidden_size",
)


class _EmbeddingCallableAdapter:
    """Module-like adapter for vLLM embedding callables."""

    def __init__(self, fn: Any, weight: torch.Tensor | None = None) -> None:
        self._fn = fn
        self.weight = weight

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._fn(input_ids)


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


def _resolve_embedding_weight(model: Any) -> torch.Tensor | None:
    for path in _INPUT_EMBEDDING_PATHS:
        layer = _resolve_attr_path(model, path)
        weight = getattr(layer, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
    return None


def resolve_input_embedding_dim(
    model: Any,
    embedding_layer: Any | None = None,
) -> int:
    """Resolve token embedding width from weight or model config."""
    if embedding_layer is not None:
        weight = getattr(embedding_layer, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
            return int(weight.shape[1])

    weight = _resolve_embedding_weight(model)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        return int(weight.shape[1])

    for path in _HIDDEN_SIZE_PATHS:
        value = _resolve_attr_path(model, path)
        if isinstance(value, int):
            return value

    tried = ", ".join((*_INPUT_EMBEDDING_PATHS, *_HIDDEN_SIZE_PATHS))
    raise AttributeError(
        "SimLLM could not resolve token embedding dimension from model "
        f"{type(model).__name__}; tried: {tried}"
    )


def resolve_input_embedding_layer(model: Any) -> Any:
    """Resolve the token embedding layer from HF-style or vLLM-wrapped models.

    vLLM model wrappers do not always expose ``get_input_embeddings()`` even
    when the underlying model has a standard token embedding module.  Some
    vLLM models expose ``embed_input_ids(input_ids)`` instead of the module
    itself.  Sim-LLM uses this only at the control-plane preprocessing boundary,
    so resolving a small set of common wrapper paths keeps the runtime path
    model-agnostic.
    """
    getter = getattr(model, "get_input_embeddings", None)
    input_id_getter = None
    if callable(getter):
        try:
            layer = getter()
        except TypeError:
            layer = None
            input_id_getter = getter
        except (AttributeError, NotImplementedError):
            layer = None
        if _looks_like_embedding_layer(layer):
            return layer

    for path in _INPUT_EMBEDDING_PATHS:
        layer = _resolve_attr_path(model, path)
        if _looks_like_embedding_layer(layer):
            return layer

    weight = _resolve_embedding_weight(model)
    if input_id_getter is not None:
        return _EmbeddingCallableAdapter(input_id_getter, weight=weight)

    for path in _INPUT_EMBEDDING_CALLABLE_PATHS:
        fn = _resolve_attr_path(model, path)
        if callable(fn):
            return _EmbeddingCallableAdapter(fn, weight=weight)

    tried = ", ".join(
        (
            "get_input_embeddings()",
            *_INPUT_EMBEDDING_PATHS,
            *_INPUT_EMBEDDING_CALLABLE_PATHS,
        )
    )
    raise AttributeError(
        "SimLLM could not resolve a token embedding layer from model "
        f"{type(model).__name__}; tried: {tried}"
    )
