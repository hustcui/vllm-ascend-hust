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
# This file is a part of the vllm-ascend project.
#

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import logger
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.kvcomp_attn.attention_utils import (
    get_kvcomp_decode_params,
    is_enable_hamming_sparse,
    reshape_and_cache_kvcomp,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.ops.triton.kivi_cache import kivi_pack_key_cache, kivi_pack_value_cache
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.kvcomp_utils import KVCompMetaData

# default max value of sliding window size
SWA_INT_MAX = 2147483647


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        for src_cache, dst_cache in zip(src_kv_cache, dst_kv_cache):
            dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            for component_cache in kv_cache:
                component_cache[dst_indices] = component_cache[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # Stable request ids aligned with batch rows.
    req_ids: list[str] | None = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    kvcomp_metadata: KVCompMetaData | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            req_ids=(
                list(common_attn_metadata.req_ids[:num_reqs])
                if getattr(common_attn_metadata, "req_ids", None) is not None
                else None
            ),
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        
        # ★ 新增: INT8 静态 per-channel 量化
        self.enable_int8 = (kv_cache_dtype == "int8")
        self._int8_ready = False
        self._k_inv_scale: torch.Tensor | None = None  # [1, num_kv_heads, head_size]
        self._v_inv_scale: torch.Tensor | None = None
        self._k_offset: torch.Tensor | None = None
        self._v_offset: torch.Tensor | None = None
        # 反量化参数 (BNSD 格式, 传给 NPU)
        self._k_aq_scale = None
        self._k_aq_offset = None
        self._v_aq_scale = None
        self._v_aq_offset = None
        
        self.enable_kivi = (kv_cache_dtype == "kivi_int4")
        cache_config = getattr(self.vllm_config, "cache_config", None)
        kivi_group_size = getattr(cache_config, "kivi_group_size", 128)
        kivi_residual_length = getattr(cache_config, "kivi_residual_length", 128)
        self.kivi_group_size = kivi_group_size if isinstance(kivi_group_size, int) else 128
        self.kivi_residual_length = (
            kivi_residual_length if isinstance(kivi_residual_length, int) else 128
        )
        if self.enable_kivi and self.kivi_residual_length % self.kivi_group_size != 0:
            raise ValueError(
                "KIVI INT4 requires kivi_residual_length "
                f"({self.kivi_residual_length}) to be divisible by "
                f"kivi_group_size ({self.kivi_group_size})."
            )
        self.kivi_bits = 4
        self.k_quant_cache = None
        self.k_scale_cache = None
        self.k_mn_cache = None
        self.v_quant_cache = None
        self.v_scale_cache = None
        self.v_mn_cache = None
        # KIVI keeps the most recent residual window in full precision on NPU,
        # while older tokens are packed into the paged int4 history cache.
        self.kivi_residual_key_cache: torch.Tensor | None = None
        self.kivi_residual_value_cache: torch.Tensor | None = None
        self.kivi_residual_key_slot_ids: torch.Tensor | None = None
        self.kivi_residual_value_slot_ids: torch.Tensor | None = None
        
        # 只在首次计算
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self.sinks = sinks
        self.layerIndex = 0
        self.enable_hamming_sparse = is_enable_hamming_sparse()

        # KIVI diagnostic counters
        self._kivi_step = 0
        self._kivi_decode_fast_count = 0
        self._kivi_dense_attn_count = 0
        self._kivi_skip_count = 0

    def _bind_kivi_cache(self, kv_cache):
        if kv_cache is None:
            return
        if not isinstance(kv_cache, (list, tuple)):
            raise RuntimeError(
                f"KIVI INT4 kv_cache must be a 6-tuple, got {type(kv_cache)}."
            )
        if len(kv_cache) == 0:
            return
        if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) != 6:
            raise RuntimeError(
                "KIVI INT4 kv_cache must be a 6-tuple: "
                "(k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)."
            )
        (
            self.k_quant_cache,
            self.k_scale_cache,
            self.k_mn_cache,
            self.v_quant_cache,
            self.v_scale_cache,
            self.v_mn_cache,
        ) = kv_cache
        self._ensure_kivi_residual_buffers()
    
    def _bind_dense_kv_cache(self, kv_cache) -> None:
        if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
            raise RuntimeError(f"Dense kv_cache must have at least 2 tensors, got {type(kv_cache)}")
        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

    def _bind_kv_cache(self, kv_cache) -> None:
        if self.enable_kivi:
            self._bind_kivi_cache(kv_cache)
        else:
            self._bind_dense_kv_cache(kv_cache)
    
    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                    ) = param

                    sparse_mode = 3
                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "key_antiquant_offset": c8_k_aq_offset,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "value_antiquant_offset": c8_v_aq_offset,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        **extra_args,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 3 if attn_metadata.causal else 0
        extra_args = {}
        
        # ★ INT8 反量化
        if self.enable_int8 and self._int8_ready:
            extra_args = {
                "key_antiquant_scale": self._k_aq_scale,
                "key_antiquant_offset": self._k_aq_offset,
                "value_antiquant_scale": self._v_aq_scale,
                "value_antiquant_offset": self._v_aq_offset,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
            }
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0  
              
        if self.enable_c8_quant:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale,
                "key_antiquant_offset": layer._c8_k_aq_offset,
                "value_antiquant_scale": layer._c8_v_aq_scale,
                "value_antiquant_offset": layer._c8_v_aq_offset,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
            }
            # TODO: Convert kvcache to NZ, and change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                scale=self.scale,
                **extra_args,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
        )
        if self.enable_c8_quant:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale),
                weak_ref_tensors(layer._c8_k_aq_offset),
                weak_ref_tensors(layer._c8_v_aq_scale),
                weak_ref_tensors(layer._c8_v_aq_offset),
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        graph_params.attn_params[num_tokens].append(attn_params)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if kv_cache is not None:
                if self.enable_kivi:
                    if self.k_quant_cache is None:
                        self._bind_kivi_cache(kv_cache)
                else:
                    if self.key_cache is None:
                        self._bind_dense_kv_cache(kv_cache)

            if self.enable_kivi:
                if self.k_quant_cache is None:
                    raise RuntimeError(
                        f"KIVI cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                    )
            else:
                if self.key_cache is None:
                    raise RuntimeError(
                        f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                    )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output[:batch_size]
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
            and self.sinks is None
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        
            
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout= "TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask ,
                sparse_mode= sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,   
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.enable_kivi:
            self._bind_kivi_cache(kv_cache)
            self._write_kivi_cache(key, value, slot_mapping)
            return

        if self.key_cache is None:
            self._bind_dense_kv_cache(kv_cache)
            
        # ★ 量化：key/value fp16 → int8
        if self.enable_int8:
            if not self._int8_ready:
                self._calc_int8_scales(key, value)
            key = self._quantize_kv_to_int8(key)        # ← 你的量化算法
            value = self._quantize_kv_to_int8(value)  # ← 你的量化算法
                
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.enable_kivi:
                self._bind_kivi_cache(kv_cache)
                self._write_kivi_cache(
                    key[: attn_metadata.num_actual_tokens],
                    value[: attn_metadata.num_actual_tokens],
                    attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens],
                )
                return query, key, value, output
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self._bind_dense_kv_cache(kv_cache)
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)
        
         # KIVI INT4 uses a 6-tensor cache layout, not ordinary (k_cache, v_cache).
        # It must bypass the dense KV cache binding and reshape_and_cache path.
        if self.enable_kivi:
            if self.sliding_window is not None:
                raise NotImplementedError(
                    "KIVI INT4 residual cache does not support sliding window yet."
                )
            self._bind_kivi_cache(kv_cache)
            return self._forward_kivi_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )
        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        
        output_padded = None
        # ── 保存 fp16 副本 + 量化写入 cache ──
        float_key, float_value = None, None
        if key is not None and value is not None:
            output_padded = output
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value

            if self.enable_int8:
                if not self._int8_ready:
                    self._calc_int8_scales(key, value)
                key = self._quantize_kv_to_int8(
                    key, self._k_inv_scale, self._k_offset
                )
                value = self._quantize_kv_to_int8(
                    value, self._v_inv_scale, self._v_offset
                )

            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
         # ── INT8 dispatch ──
        if self.enable_int8 and self._int8_ready:
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                output = self._forward_int8_decode(query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                output = self._forward_int8_chunked_prefill(
                    query, float_key, float_value, attn_metadata, output
                )
            else:
                output = self._forward_int8_prefill(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata, output,
                )
            return output


        # ── 非 INT8 ──
        #output_padded = output if (key is not None and value is not None) else None
        if output_padded is not None:
            attn_output = self.forward_impl(
                query, key, value, kv_cache, attn_metadata, output_padded
            )
        else:
            attn_output = self.forward_impl(
                query, key, value, kv_cache, attn_metadata, output
            )
        output[:num_tokens] = attn_output[:num_tokens]
        return output
    
     # ★ 以下是新增方法
      # ═══════════════════════════════════════════════════════════════
    # ★ KIVI INT4 量化: 按原始 KIVI 的“量化历史区 + 全精度 residual 区”计算 attention
    def _forward_kivi_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ) -> torch.Tensor:
        self._bind_kivi_cache(kv_cache)

        self._kivi_step += 1
        state_names = {
            AscendAttentionState.PrefillNoCache: "PrefillNoCache",
            AscendAttentionState.PrefillCacheHit: "PrefillCacheHit",
            AscendAttentionState.DecodeOnly: "DecodeOnly",
            AscendAttentionState.ChunkedPrefill: "ChunkedPrefill",
            AscendAttentionState.SpecDecoding: "SpecDecoding",
        }
        if self._kivi_step % 10 == 1:
            logger.info(
                "[KIVI] step=%d state=%s reqs=%d "
                "(sum: decode_fast=%d dense_attn=%d)",
                self._kivi_step,
                state_names.get(attn_metadata.attn_state, "UNKNOWN"),
                len(attn_metadata.seq_lens_list or []),
                self._kivi_decode_fast_count,
                self._kivi_dense_attn_count,
            )

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            actual_seq_qlen = list(range(1, len(attn_metadata.seq_lens_list) + 1))
        num_tokens = int(actual_seq_qlen[-1])
        query = query[:num_tokens]

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            if key is None or value is None:
                raise RuntimeError("KIVI PrefillNoCache requires dense key/value.")
            return self._forward_kivi_prefill_fia(
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                output=output,
            )
        else:
            if key is not None and value is not None:
                num_actual_tokens = min(
                    getattr(attn_metadata, "num_actual_tokens", key.shape[0]),
                    key.shape[0],
                )
                self._write_kivi_cache(
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    attn_metadata.slot_mapping[:num_actual_tokens],
                )
            seq_lens = attn_metadata.seq_lens_list
            self._sync_kivi_residual_windows(
                attn_metadata.block_tables,
                seq_lens,
            )
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                return self._forward_kivi_decode_fast(
                    query,
                    attn_metadata.block_tables,
                    seq_lens,
                    output,
                )
            if attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_kivi_chunked_prefill(
                    query=query,
                    key=key,
                    value=value,
                    attn_metadata=attn_metadata,
                    output=output,
                )
            dense_key, dense_value = self._gather_dequant_kivi_paged_cache(
                attn_metadata.block_tables,
                seq_lens,
                query.dtype,
            )

        return self._forward_kivi_dense_attention(
            query=query,
            key=dense_key,
            value=dense_value,
            seq_lens=seq_lens,
            actual_seq_qlen=actual_seq_qlen,
            causal=attn_metadata.causal,
            output=output,
        )

    def _repeat_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_kv == 1:
            return tensor
        return tensor.repeat_interleave(self.num_queries_per_kv, dim=1)

    def _build_kivi_causal_mask(
        self,
        q_len: int,
        kv_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        q_pos = torch.arange(
            kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=device
        )
        kv_pos = torch.arange(kv_seq_len, dtype=torch.long, device=device)
        mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        neg_inf = torch.finfo(dtype).min
        return torch.where(
            mask,
            torch.full((), neg_inf, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        ).unsqueeze(0)
    
    def _unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """Vectorized unpack of int32 into 8 int4 lanes (last dim).

        Uses pure tensor ops (no Python list comprehension) for
        performance on both CUDA and Ascend NPU.
        """
        packed_i32 = packed.to(torch.int32)
        shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
        unpacked = (packed_i32.unsqueeze(-1) >> shifts) & 0xF
        return unpacked.to(torch.float32)

    def _check_kivi_cache_bound(self) -> None:
        if (
            self.k_quant_cache is None
            or self.k_scale_cache is None
            or self.k_mn_cache is None
            or self.v_quant_cache is None
            or self.v_scale_cache is None
            or self.v_mn_cache is None
        ):
            raise RuntimeError("KIVI cache tensors are not bound.")

    def _ensure_kivi_residual_buffers(self) -> None:
        if self.kivi_residual_key_cache is not None:
            return

        self._check_kivi_cache_bound()
        num_blocks = self.k_quant_cache.shape[0]
        residual_slots = self.kivi_residual_length
        device = self.k_quant_cache.device

        self.kivi_residual_key_cache = torch.empty(
            (num_blocks, residual_slots, self.num_kv_heads, self.head_size),
            dtype=self.k_scale_cache.dtype,
            device=device,
        )
        self.kivi_residual_value_cache = torch.empty(
            (num_blocks, residual_slots, self.num_kv_heads, self.head_size),
            dtype=self.v_scale_cache.dtype,
            device=device,
        )
        self.kivi_residual_key_slot_ids = torch.full(
            (num_blocks, residual_slots),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.kivi_residual_value_slot_ids = torch.full(
            (num_blocks, residual_slots),
            -1,
            dtype=torch.long,
            device=device,
        )

    def _get_kivi_residual_buffers(
        self,
        *,
        is_key: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_kivi_residual_buffers()
        slot_ids = self.kivi_residual_key_slot_ids if is_key else self.kivi_residual_value_slot_ids
        cache = self.kivi_residual_key_cache if is_key else self.kivi_residual_value_cache
        assert slot_ids is not None and cache is not None
        return slot_ids, cache

    def _get_kivi_block_size(self) -> int:
        self._check_kivi_cache_bound()
        return self.k_quant_cache.shape[-1] * 8

    def _store_kivi_residual_entries(
        self,
        values: torch.Tensor,
        slots: torch.Tensor,
        *,
        is_key: bool,
    ) -> None:
        if values.numel() == 0 or slots.numel() == 0:
            return

        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        block_size = self._get_kivi_block_size()
        values = values.to(cache.dtype)
        block_indices = torch.div(slots, block_size, rounding_mode="floor")

        for idx in range(int(slots.shape[0])):
            slot = int(slots[idx].item())
            block_idx = int(block_indices[idx].item())
            slot_row = slot_ids[block_idx]
            match = (slot_row == slot).nonzero(as_tuple=False)
            if match.numel() > 0:
                lane = int(match[0].item())
            else:
                free = (slot_row < 0).nonzero(as_tuple=False)
                if free.numel() == 0:
                    cache_name = "key" if is_key else "value"
                    raise RuntimeError(
                        "KIVI residual cache row is full before flush: "
                        f"cache={cache_name}, block_idx={block_idx}, "
                        f"residual_slots={self.kivi_residual_length}, slot={slot}."
                    )
                lane = int(free[0].item())
                slot_ids[block_idx, lane] = slot
            cache[block_idx, lane].copy_(values[idx])

    def _lookup_kivi_residual_tensor(
        self,
        slot: int,
        *,
        is_key: bool,
        target_dtype: torch.dtype,
        target_device: torch.device | None = None,
    ) -> torch.Tensor | None:
        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        block_idx = slot // self._get_kivi_block_size()
        slot_row = slot_ids[block_idx]
        match = (slot_row == slot).nonzero(as_tuple=False)
        if match.numel() == 0:
            return None

        lane = int(match[0].item())
        tensor = cache[block_idx, lane]
        if target_device is None:
            target_device = tensor.device
        return tensor.to(device=target_device, dtype=target_dtype)

    def _gather_kivi_residual_tensors(
        self,
        slots: list[int],
        *,
        is_key: bool,
    ) -> torch.Tensor:
        if not slots:
            raise RuntimeError("KIVI residual gather expects a non-empty slot list.")

        target_dtype = self.k_scale_cache.dtype if is_key else self.v_scale_cache.dtype
        tensors = []
        cache_name = "key" if is_key else "value"
        for slot in slots:
            tensor = self._lookup_kivi_residual_tensor(
                int(slot),
                is_key=is_key,
                target_dtype=target_dtype,
            )
            if tensor is None:
                raise RuntimeError(
                    f"KIVI {cache_name} residual tensor for slot {slot} is missing before flush."
                )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _has_kivi_residual_entry(
        self,
        slot: int,
        *,
        is_key: bool,
    ) -> bool:
        slot_ids, _ = self._get_kivi_residual_buffers(is_key=is_key)
        block_idx = slot // self._get_kivi_block_size()
        return bool((slot_ids[block_idx] == slot).any().item())

    def _clear_kivi_residual_entries(
        self,
        slots: list[int],
        *,
        is_key: bool,
    ) -> None:
        if not slots:
            return

        slot_ids, _ = self._get_kivi_residual_buffers(is_key=is_key)
        block_size = self._get_kivi_block_size()
        for slot in slots:
            block_idx = slot // block_size
            slot_row = slot_ids[block_idx]
            match = (slot_row == slot).nonzero(as_tuple=False)
            if match.numel() == 0:
                continue
            lane = int(match[0].item())
            slot_ids[block_idx, lane] = -1

    def _clear_stale_kivi_residual_entries(
        self,
        live_slots: set[int],
        *,
        is_key: bool,
    ) -> None:
        slot_ids, _ = self._get_kivi_residual_buffers(is_key=is_key)
        active_slots = slot_ids[slot_ids >= 0]
        if active_slots.numel() == 0:
            return
        stale_slots = [int(slot) for slot in active_slots.tolist() if int(slot) not in live_slots]
        self._clear_kivi_residual_entries(stale_slots, is_key=is_key)

    def _write_kivi_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key is None or value is None or slot_mapping is None:
            return

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return

        key = key[valid]
        value = value[valid]
        slots = slot_mapping[valid].to(torch.long)
        self._store_kivi_residual_entries(
            key.detach(),
            slots,
            is_key=True,
        )
        self._store_kivi_residual_entries(
            value.detach(),
            slots,
            is_key=False,
        )

    def _get_kivi_ordered_slots(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
    ) -> list[list[int]]:
        self._check_kivi_cache_bound()
        block_size = self.k_quant_cache.shape[-1] * 8
        block_table = block_table.to(torch.long)
        ordered_slots: list[list[int]] = []

        for req_idx, seq_len in enumerate(seq_lens):
            req_slots: list[int] = []
            for pos in range(int(seq_len)):
                block_pos = pos // block_size
                block_offset = pos % block_size
                block_id = int(block_table[req_idx, block_pos].item())
                if block_id >= 0:
                    req_slots.append(block_id * block_size + block_offset)
            ordered_slots.append(req_slots)
        return ordered_slots

    def _is_aligned_kivi_key_window(self, window_slots: list[int]) -> bool:
        if len(window_slots) != self.kivi_residual_length:
            return False

        block_size = self.k_quant_cache.shape[-1] * 8
        group_size = self.kivi_group_size

        for start in range(0, len(window_slots), group_size):
            group_slots = window_slots[start:start + group_size]
            if len(group_slots) != group_size:
                return False

            first_slot = int(group_slots[0])
            last_slot = int(group_slots[-1])
            block_idx = first_slot // block_size
            block_offset = first_slot % block_size

            if block_offset % group_size != 0:
                return False
            if last_slot // block_size != block_idx:
                return False

            expected = list(range(first_slot, first_slot + group_size))
            if group_slots != expected:
                return False

        return True

    def _flush_kivi_key_slots(self, slots: list[int]) -> None:
        if not slots:
            return

        keys = self._gather_kivi_residual_tensors(slots, is_key=True)
        slot_tensor = torch.tensor(
            slots, dtype=torch.long, device=keys.device
        )
        self._write_kivi_key_quant_cache(keys, slot_tensor)
        self._clear_kivi_residual_entries(slots, is_key=True)

    def _flush_kivi_value_slots(self, slots: list[int]) -> None:
        if not slots:
            return

        values = self._gather_kivi_residual_tensors(slots, is_key=False)
        slot_tensor = torch.tensor(
            slots, dtype=torch.long, device=values.device
        )
        self._write_kivi_value_quant_cache(values, slot_tensor)
        self._clear_kivi_residual_entries(slots, is_key=False)

    def _flush_kivi_key_batches(self, window_slots: list[int]) -> None:
        while len(window_slots) >= self.kivi_residual_length:
            flush_slots = window_slots[:self.kivi_residual_length]
            if not self._is_aligned_kivi_key_window(flush_slots):
                break
            self._flush_kivi_key_slots(flush_slots)
            del window_slots[:self.kivi_residual_length]

    def _flush_kivi_value_batches(self, window_slots: list[int]) -> None:
        if len(window_slots) <= self.kivi_residual_length:
            return

        flush_slots = window_slots[:-self.kivi_residual_length]
        self._flush_kivi_value_slots(flush_slots)
        del window_slots[:-self.kivi_residual_length]

    def _sync_kivi_residual_windows(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
    ) -> None:
        ordered_slots = self._get_kivi_ordered_slots(block_table, seq_lens)

        live_slots = {slot for req_slots in ordered_slots for slot in req_slots}
        self._clear_stale_kivi_residual_entries(live_slots, is_key=True)
        self._clear_stale_kivi_residual_entries(live_slots, is_key=False)

        for req_slots in ordered_slots:
            key_window = [
                slot for slot in req_slots
                if self._has_kivi_residual_entry(slot, is_key=True)
            ]
            value_window = [
                slot for slot in req_slots
                if self._has_kivi_residual_entry(slot, is_key=False)
            ]

            self._flush_kivi_key_batches(key_window)
            self._flush_kivi_value_batches(value_window)

    def _write_kivi_key_quant_cache(
        self,
        key: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._check_kivi_cache_bound()
        if key is None or slot_mapping is None:
            return

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return

        key = key[valid].to(self.k_scale_cache.dtype).contiguous()
        slots = slot_mapping[valid].to(torch.long).contiguous()

        if key.shape[-1] != self.head_size:
            raise RuntimeError(
                "KIVI INT4 key head_size must match attention head_size "
                f"({self.head_size}), got {key.shape[-1]}."
            )
        if key.shape[0] % self.kivi_group_size != 0:
            raise RuntimeError(
                "KIVI key flush must contain whole token groups, got "
                f"{key.shape[0]} tokens for group_size={self.kivi_group_size}."
            )

        if self.head_size % 8 != 0:
            raise RuntimeError(
                f"KIVI INT4 int32 packing requires head_size ({self.head_size}) "
                "to be divisible by 8."
            )
        if self.kivi_group_size % 8 != 0:
            raise RuntimeError(
                "KIVI INT4 key packing requires kivi_group_size "
                f"({self.kivi_group_size}) to be divisible by 8."
            )

        block_size = self.k_quant_cache.shape[-1] * 8
        if block_size % self.kivi_group_size != 0:
            raise RuntimeError(
                f"KIVI INT4 key cache requires block_size ({block_size}) to be "
                f"divisible by kivi_group_size ({self.kivi_group_size})."
            )
        if not self._is_aligned_kivi_key_window(slots.tolist()):
            raise RuntimeError("KIVI key flush requires contiguous aligned token groups.")

        kivi_pack_key_cache(
            key,
            slots,
            self.k_quant_cache,
            self.k_scale_cache,
            self.k_mn_cache,
            self.kivi_group_size,
        )

    def _write_kivi_value_quant_cache(
        self,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._check_kivi_cache_bound()
        if value is None or slot_mapping is None:
            return

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return

        value = value[valid].to(self.v_scale_cache.dtype).contiguous()
        slots = slot_mapping[valid].to(torch.long).contiguous()
        head_size = value.shape[-1]

        if head_size != self.head_size:
            raise RuntimeError(
                "KIVI INT4 value head_size must match attention head_size "
                f"({self.head_size}), got {head_size}."
            )

        if head_size % self.kivi_group_size != 0:
            raise RuntimeError(
                f"KIVI INT4 value head_size ({head_size}) must be divisible by "
                f"kivi_group_size ({self.kivi_group_size})."
            )

        if head_size % 8 != 0:
            raise RuntimeError(
                f"KIVI INT4 int32 packing requires head_size ({head_size}) "
                "to be divisible by 8."
            )
        if self.kivi_group_size % 8 != 0:
            raise RuntimeError(
                "KIVI INT4 value packing requires kivi_group_size "
                f"({self.kivi_group_size}) to be divisible by 8."
            )

        kivi_pack_value_cache(
            value,
            slots,
            self.v_quant_cache,
            self.v_scale_cache,
            self.v_mn_cache,
            self.kivi_group_size,
        )


    def _dequant_kivi_key_blocks(
        self,
        k_quant: torch.Tensor,
        k_scale: torch.Tensor,
        k_mn: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        # k_quant: [B, blocks, kv_heads, head_size, block_size / 8]
        q = self._unpack_int4(k_quant).flatten(-2)
        block_size = q.shape[-1]
        # -> [B, blocks, kv_heads, head_size, block_size]

        scale = k_scale.repeat_interleave(self.kivi_group_size, dim=-1)[..., :block_size]
        mn = k_mn.repeat_interleave(self.kivi_group_size, dim=-1)[..., :block_size]

        deq = q.to(scale.dtype) * scale + mn
        # -> [B, blocks, block_size, kv_heads, head_size]
        return deq.permute(0, 1, 4, 2, 3).contiguous().to(target_dtype)


    def _dequant_kivi_value_blocks(
        self,
        v_quant: torch.Tensor,
        v_scale: torch.Tensor,
        v_mn: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        # v_quant: [B, blocks, block_size, kv_heads, head_size / 8]
        q = self._unpack_int4(v_quant).flatten(-2)
        head_size = q.shape[-1]

        scale = v_scale.repeat_interleave(self.kivi_group_size, dim=-1)[..., :head_size]
        mn = v_mn.repeat_interleave(self.kivi_group_size, dim=-1)[..., :head_size]

        deq = q.to(scale.dtype) * scale + mn
        # -> [B, blocks, block_size, kv_heads, head_size]
        return deq.contiguous().to(target_dtype)

    def _normalize_kivi_block_layout(
        self,
        blocks: torch.Tensor,
        batch_size: int,
        max_blocks: int,
        cache_block_size: int,
        name: str,
    ) -> torch.Tensor:
        if blocks.ndim != 5:
            raise RuntimeError(
                f"KIVI {name} cache must be 5D after dequant, got shape={tuple(blocks.shape)}."
            )

        if blocks.shape[0] != batch_size:
            raise RuntimeError(
                f"KIVI {name} batch mismatch after dequant: "
                f"expected {batch_size}, got {blocks.shape[0]}."
            )

        if blocks.shape[1] == max_blocks and blocks.shape[2] == cache_block_size:
            return blocks

        if blocks.shape[1] == cache_block_size and blocks.shape[2] == max_blocks:
            return blocks.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"KIVI {name} cache layout mismatch: shape={tuple(blocks.shape)}, "
            f"expected (*, {max_blocks}, {cache_block_size}, ...)."
        )


    def _gather_dequant_kivi_paged_cache(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_table = block_table.to(torch.long)
        batch_size = len(seq_lens)
        block_table = block_table[:batch_size]
        _, max_blocks = block_table.shape
        flat_ids = block_table.reshape(-1).clamp_min(0)

        k_quant = self.k_quant_cache[flat_ids].view(
            batch_size, max_blocks, *self.k_quant_cache.shape[1:]
        )
        k_scale = self.k_scale_cache[flat_ids].view(
            batch_size, max_blocks, *self.k_scale_cache.shape[1:]
        )
        k_mn = self.k_mn_cache[flat_ids].view(
            batch_size, max_blocks, *self.k_mn_cache.shape[1:]
        )

        v_quant = self.v_quant_cache[flat_ids].view(
            batch_size, max_blocks, *self.v_quant_cache.shape[1:]
        )
        v_scale = self.v_scale_cache[flat_ids].view(
            batch_size, max_blocks, *self.v_scale_cache.shape[1:]
        )
        v_mn = self.v_mn_cache[flat_ids].view(
            batch_size, max_blocks, *self.v_mn_cache.shape[1:]
        )

        cache_block_size = self.k_quant_cache.shape[-1] * 8
        k_blocks = self._dequant_kivi_key_blocks(k_quant, k_scale, k_mn, target_dtype)
        v_blocks = self._dequant_kivi_value_blocks(v_quant, v_scale, v_mn, target_dtype)
        k_blocks = self._normalize_kivi_block_layout(
            k_blocks, batch_size, max_blocks, cache_block_size, "key"
        )
        v_blocks = self._normalize_kivi_block_layout(
            v_blocks, batch_size, max_blocks, cache_block_size, "value"
        )

        ordered_slots = self._get_kivi_ordered_slots(block_table, seq_lens)

        dense_k_parts = []
        dense_v_parts = []
        for req_idx, req_slots in enumerate(ordered_slots):
            req_block_lookup = {
                int(block_id): block_pos
                for block_pos, block_id in enumerate(block_table[req_idx].tolist())
                if int(block_id) >= 0
            }

            req_dense_k = []
            req_dense_v = []
            for slot in req_slots:
                key = None
                value = None
                key = self._lookup_kivi_residual_tensor(
                    slot,
                    is_key=True,
                    target_dtype=target_dtype,
                    target_device=k_blocks.device,
                )
                value = self._lookup_kivi_residual_tensor(
                    slot,
                    is_key=False,
                    target_dtype=target_dtype,
                    target_device=v_blocks.device,
                )
                if key is None:
                    block_id = slot // cache_block_size
                    block_offset = slot % cache_block_size
                    block_pos = req_block_lookup.get(block_id)
                    if block_pos is None:
                        raise RuntimeError(
                            f"KIVI key cache missing block_id={block_id} for req_idx={req_idx}."
                        )
                    key = k_blocks[req_idx, block_pos, block_offset]
                if value is None:
                    block_id = slot // cache_block_size
                    block_offset = slot % cache_block_size
                    block_pos = req_block_lookup.get(block_id)
                    if block_pos is None:
                        raise RuntimeError(
                            f"KIVI value cache missing block_id={block_id} for req_idx={req_idx}."
                        )
                    value = v_blocks[req_idx, block_pos, block_offset]

                req_dense_k.append(key.to(target_dtype))
                req_dense_v.append(value.to(target_dtype))

            if req_dense_k:
                dense_k_parts.append(torch.stack(req_dense_k, dim=0))
                dense_v_parts.append(torch.stack(req_dense_v, dim=0))

        if not dense_k_parts:
            empty = k_blocks.new_empty((0, self.num_kv_heads, self.head_size))
            return empty, empty

        dense_k = torch.cat(dense_k_parts, dim=0)
        dense_v = torch.cat(dense_v_parts, dim=0)
        return dense_k.contiguous(), dense_v.contiguous()    

    def _gather_dequant_kivi_paged_cache_vectorized(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_device = self.k_quant_cache.device
        block_table = block_table[:len(seq_lens)].to(
            device=cache_device, dtype=torch.long
        )
        _, max_blocks = block_table.shape
        cache_block_size = self.k_quant_cache.shape[-1] * 8
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=cache_device)
        block_counts = torch.div(
            seq_lens_t + cache_block_size - 1,
            cache_block_size,
            rounding_mode="floor",
        )
        block_positions = torch.arange(
            max_blocks, dtype=torch.long, device=cache_device
        )
        valid_block_mask = block_positions.unsqueeze(0) < block_counts.unsqueeze(1)
        valid_req_idx, valid_block_pos = valid_block_mask.nonzero(as_tuple=True)
        valid_block_ids = block_table[valid_block_mask]
        if bool((valid_block_ids < 0).any()):
            raise RuntimeError("KIVI block_table has invalid block ids for live tokens.")

        if valid_block_ids.numel() == 0:
            empty = self.k_quant_cache.new_empty(
                (0, self.num_kv_heads, self.head_size),
                dtype=target_dtype,
            )
            return empty, empty

        num_valid_blocks = int(valid_block_ids.numel())
        k_quant = self.k_quant_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.k_quant_cache.shape[1:]
        )
        k_scale = self.k_scale_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.k_scale_cache.shape[1:]
        )
        k_mn = self.k_mn_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.k_mn_cache.shape[1:]
        )
        v_quant = self.v_quant_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.v_quant_cache.shape[1:]
        )
        v_scale = self.v_scale_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.v_scale_cache.shape[1:]
        )
        v_mn = self.v_mn_cache[valid_block_ids].view(
            num_valid_blocks, 1, *self.v_mn_cache.shape[1:]
        )

        k_blocks = self._dequant_kivi_key_blocks(k_quant, k_scale, k_mn, target_dtype)
        v_blocks = self._dequant_kivi_value_blocks(v_quant, v_scale, v_mn, target_dtype)
        k_blocks = self._normalize_kivi_block_layout(
            k_blocks, num_valid_blocks, 1, cache_block_size, "key"
        )
        v_blocks = self._normalize_kivi_block_layout(
            v_blocks, num_valid_blocks, 1, cache_block_size, "value"
        )
        k_blocks = k_blocks[:, 0]
        v_blocks = v_blocks[:, 0]

        valid_tokens_in_block = (
            seq_lens_t[valid_req_idx] - valid_block_pos * cache_block_size
        ).clamp(max=cache_block_size)
        token_positions = torch.arange(
            cache_block_size, dtype=torch.long, device=cache_device
        )
        valid_token_mask = (
            token_positions.unsqueeze(0) < valid_tokens_in_block.unsqueeze(1)
        )

        dense_k = k_blocks[valid_token_mask]
        dense_v = v_blocks[valid_token_mask]

        dense_k_parts: list[torch.Tensor] = []
        dense_v_parts: list[torch.Tensor] = []
        ordered_slots = self._get_kivi_ordered_slots(block_table, seq_lens)
        kv_start = 0
        for req_idx, seq_len in enumerate(seq_lens):
            req_len = int(seq_len)
            req_k = dense_k[kv_start:kv_start + req_len]
            req_v = dense_v[kv_start:kv_start + req_len]
            kv_start += req_len

            req_slots = ordered_slots[req_idx]
            residual_key_slots = [
                slot for slot in req_slots
                if self._has_kivi_residual_entry(slot, is_key=True)
            ]
            residual_value_slots = [
                slot for slot in req_slots
                if self._has_kivi_residual_entry(slot, is_key=False)
            ]
            if residual_key_slots or residual_value_slots:
                pos_by_slot = {slot: pos for pos, slot in enumerate(req_slots)}
                if residual_key_slots:
                    req_k = self._overlay_kivi_residual_tensors(
                        req_k,
                        residual_key_slots,
                        pos_by_slot,
                        target_dtype,
                        is_key=True,
                    )
                if residual_value_slots:
                    req_v = self._overlay_kivi_residual_tensors(
                        req_v,
                        residual_value_slots,
                        pos_by_slot,
                        target_dtype,
                        is_key=False,
                    )

            dense_k_parts.append(req_k)
            dense_v_parts.append(req_v)

        if not dense_k_parts:
            empty = k_blocks.new_empty((0, self.num_kv_heads, self.head_size))
            return empty, empty

        return (
            torch.cat(dense_k_parts, dim=0).contiguous(),
            torch.cat(dense_v_parts, dim=0).contiguous(),
        )

    def _overlay_kivi_residual_tensors(
        self,
        dense: torch.Tensor,
        residual_slots,
        pos_by_slot: dict[int, int],
        target_dtype: torch.dtype,
        *,
        is_key: bool,
    ) -> torch.Tensor:
        positions: list[int] = []
        tensors: list[torch.Tensor] = []
        dense_len = dense.shape[0]
        for slot in residual_slots:
            pos = pos_by_slot.get(slot)
            tensor = self._lookup_kivi_residual_tensor(
                int(slot),
                is_key=is_key,
                target_dtype=target_dtype,
                target_device=dense.device,
            )
            if pos is None or tensor is None or pos >= dense_len:
                continue
            positions.append(pos)
            tensors.append(tensor)

        if not positions:
            return dense

        pos_tensor = torch.tensor(
            positions,
            dtype=torch.long,
            device=dense.device,
        )
        value_tensor = torch.stack(
            tensors,
            dim=0,
        )
        dense = dense.clone()
        dense.index_copy_(0, pos_tensor, value_tensor)
        return dense

    def _forward_kivi_decode_fast(
        self,
        query: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list[int],
        output: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(seq_lens)
        self._kivi_decode_fast_count += 1
        if self._kivi_step % 10 == 1:
            logger.info(
                "[KIVI] step=%d DecodeOnly→FIA_TND reqs=%d "
                "(decode_fast=%d dense_attn=%d skip=%d)",
                self._kivi_step,
                batch_size,
                self._kivi_decode_fast_count,
                self._kivi_dense_attn_count,
                self._kivi_skip_count,
            )
        dense_key, dense_value = self._gather_dequant_kivi_paged_cache_vectorized(
            block_table,
            seq_lens,
            query.dtype,
        )
        actual_seq_lengths_q = list(range(1, batch_size + 1))
        actual_seq_lengths_kv = torch.tensor(
            seq_lens,
            dtype=torch.int32,
            device="cpu",
        ).cumsum(dim=0).tolist()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:batch_size],
            key=dense_key,
            value=dense_value,
            block_table=None,
            input_layout="TND",
            sparse_mode=0,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
        )
        output[:batch_size] = attn_output.view(
            batch_size, self.num_heads, self.head_size
        )
        return output

    def _forward_kivi_prefill_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
        self._kivi_skip_count += 1
        if self._kivi_step % 10 == 1:
            logger.info(
                "[KIVI] step=%d PrefillNoCache→FIA_TND tokens=%d "
                "(decode_fast=%d dense_attn=%d skip=%d)",
                self._kivi_step,
                num_tokens,
                self._kivi_decode_fast_count,
                self._kivi_dense_attn_count,
                self._kivi_skip_count,
            )

        query = query[:num_tokens]
        key = key[:num_tokens]
        value = value[:num_tokens]
        sparse_mode = 3 if attn_metadata.causal else 0

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask if attn_metadata.causal else None,
            block_table=None,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
        )
        output[:num_tokens] = attn_output.view(
            num_tokens, self.num_heads, self.head_size
        )
        return output

    def _forward_kivi_chunked_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_decode = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        block_size = self.k_quant_cache.shape[-1] * 8

        if self._kivi_step % 10 == 1:
            logger.info(
                "[KIVI] step=%d ChunkedPrefill→FIA split "
                "(decode_tokens=%d prefills=%d decodes=%d)",
                self._kivi_step,
                num_decode,
                attn_metadata.num_prefills,
                num_decodes,
            )

        if num_decode > 0:
            self._forward_kivi_decode_fast(
                query[:num_decode],
                attn_metadata.block_tables[:num_decodes],
                attn_metadata.seq_lens_list[:num_decodes],
                output,
            )

        if attn_metadata.num_prefills <= 0:
            return output

        prefill_q = query[num_decode:num_tokens]
        prefill_seq_qlen = [
            actual_seq_qlen[i] - num_decode
            for i in range(num_decodes, len(actual_seq_qlen))
        ]
        prefill_seq_lens = attn_metadata.seq_lens_list[num_decodes:]

        all_new_prefill = True
        for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
            q_start = actual_seq_qlen[i - 1] if i > 0 else 0
            qlen_i = actual_seq_qlen[i] - q_start
            if attn_metadata.seq_lens_list[i] > qlen_i:
                all_new_prefill = False
                break

        if all_new_prefill and key is not None and value is not None:
            prefill_k = key[num_decode:num_tokens]
            prefill_v = value[num_decode:num_tokens]
            prefill_seq_kvlen = prefill_seq_qlen
        else:
            prefill_k, prefill_v = self._gather_dequant_kivi_paged_cache_vectorized(
                attn_metadata.block_tables[num_decodes:],
                prefill_seq_lens,
                query.dtype,
            )
            prefill_seq_kvlen = torch.tensor(
                prefill_seq_lens,
                dtype=torch.int32,
                device="cpu",
            ).cumsum(dim=0).tolist()

        sparse_mode = 3 if attn_metadata.causal else 0
        attn_out, _ = torch_npu.npu_fused_infer_attention_score(
            query=prefill_q,
            key=prefill_k,
            value=prefill_v,
            atten_mask=attn_metadata.attn_mask if attn_metadata.causal else None,
            block_table=None,
            input_layout="TND",
            sparse_mode=sparse_mode,
            block_size=block_size,
            actual_seq_lengths=prefill_seq_qlen,
            actual_seq_lengths_kv=prefill_seq_kvlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
        )
        n_prefill = num_tokens - num_decode
        output[num_decode:num_tokens] = attn_out.view(
            n_prefill, self.num_heads, self.head_size
        )[:n_prefill]
        return output
    
    def _forward_kivi_dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens: list[int],
        actual_seq_qlen,
        causal: bool,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._kivi_dense_attn_count += 1
        prev_q_end = 0
        kv_start = 0
        outputs = []

        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()

        for req_idx, kv_len in enumerate(seq_lens):
            q_end = int(actual_seq_qlen[req_idx])
            q_seq = query[prev_q_end:q_end]
            k_seq = key[kv_start:kv_start + int(kv_len)]
            v_seq = value[kv_start:kv_start + int(kv_len)]

            q_len = q_seq.shape[0]
            if q_len > 0:
                k_seq = self._repeat_kv(k_seq)
                v_seq = self._repeat_kv(v_seq)

                q_states = q_seq.transpose(0, 1)
                attn = torch.matmul(q_states, k_seq.permute(1, 2, 0)) * self.scale

                if causal:
                    attn = attn + self._build_kivi_causal_mask(
                        q_len=q_len,
                        kv_seq_len=int(kv_len),
                        dtype=attn.dtype,
                        device=attn.device,
                    )

                attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
                out = torch.matmul(attn, v_seq.transpose(0, 1))
                outputs.append(out.transpose(0, 1).contiguous())

            prev_q_end = q_end
            kv_start += int(kv_len)

        if outputs:
            attn_output = torch.cat(outputs, dim=0)
            output[:attn_output.shape[0]] = attn_output

        return output
   

    @staticmethod
    def _cu_seqlens_to_seq_lens(cu_seqlens) -> list[int]:
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.tolist()
        prev = 0
        seq_lens = []
        for end in cu_seqlens:
            seq_lens.append(int(end) - prev)
            prev = int(end)
        return seq_lens


    # ★ INT8 独立方法 (镜像 C8)
    # ═══════════════════════════════════════════════════════════════
    def _forward_int8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 Decode: BNSD + int8 paged KV cache + antiquant."""

        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key, value,
            key_antiquant_scale=self._k_aq_scale,
            key_antiquant_offset=self._k_aq_offset,
            value_antiquant_scale=self._v_aq_scale,
            value_antiquant_offset=self._v_aq_offset,
            key_antiquant_mode=0, value_antiquant_mode=0,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD", sparse_mode=0,
            scale=self.scale, block_size=block_size,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_int8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 ChunkedPrefill: decode→BNSD+int8, prefill→TND+fp16."""
      
        num_decode = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])

        # ── ① Decode: BNSD + int8 cache ──
        if num_decode > 0:
            num_block, block_size, _, _ = self.key_cache.shape
            kv_k = self.key_cache.view(num_block, block_size, -1)
            kv_v = self.value_cache.view(num_block, block_size, -1)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode].unsqueeze(2), kv_k, kv_v,
                key_antiquant_scale=self._k_aq_scale,
                key_antiquant_offset=self._k_aq_offset,
                value_antiquant_scale=self._v_aq_scale,
                value_antiquant_offset=self._v_aq_offset,
                key_antiquant_mode=0, value_antiquant_mode=0,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD", sparse_mode=0,
                scale=self.scale, block_size=block_size,
            )
            output[:num_decode] = attn_out.squeeze(2)

        # ── ② Prefill: TND + fp16 (fresh K/V 或 dequant from int8 cache) ──
        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode
                for i in range(num_decodes, len(actual_seq_qlen))
            ]

            # 判断是否所有 prefill 请求都是全新（无历史 KV cache）
            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                # 全新 prefill: 直接用 fp16 fresh K/V
             
                prefill_k = float_key[num_decode:num_tokens]
                prefill_v = float_value[num_decode:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                # 承接已有 cache: gather paged int8 KV → dequant → dense fp16
             
                num_block, blk_size, _, _ = self.key_cache.shape
                paged_k = self.key_cache.view(num_block, blk_size, -1)
                paged_v = self.value_cache.view(num_block, blk_size, -1)
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            cache_block_size = self.key_cache.shape[1]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q, key=prefill_k, value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND", sparse_mode=3,
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads, scale=self.scale,
            )
            n_prefill = num_tokens - num_decode
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_int8_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 Prefill: TND + fp16 (fresh K/V or dequant if from int8 cache)."""
      
        key, value, block_size, block_table, actual_seq_lengths_kv = \
            self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
           
            key = key[:num_tokens]
            value = value[:num_tokens]

        # PrefillCacheHit: key 从 cache 读是 int8 → 手动 dequant
        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv
                    if isinstance(actual_seq_lengths_kv, list)
                    else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(
                    key, value, block_table, seq_lens, query.dtype
                )
                block_table = None
                block_size = self.key_cache.shape[1]
                actual_seq_lengths_kv = torch.tensor(
                    seq_lens, dtype=torch.int32
                ).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - self._k_offset) * (
                    1.0 / self._k_inv_scale
                )
                value = (value.to(query.dtype) - self._v_offset) * (
                    1.0 / self._v_inv_scale
                )

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query, key=key, value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND", sparse_mode=3,
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads, scale=self.scale,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _dequant_paged_kv_to_dense(
        self, key, value, block_table, seq_lens, target_dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged int8 KV blocks and dequantize to dense fp16."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]
        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        dense_k = (dense_k.to(target_dtype) - self._k_offset) * (
            1.0 / self._k_inv_scale
        )
        dense_v = (dense_v.to(target_dtype) - self._v_offset) * (
            1.0 / self._v_inv_scale
        )
        return dense_k, dense_v
    
    def _calc_int8_scales(self, key, value):
        k_max = key.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        v_max = value.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        self._k_inv_scale = 127.0 / k_max
        self._k_offset = torch.zeros_like(self._k_inv_scale)
        self._v_inv_scale = 127.0 / v_max
        self._v_offset = torch.zeros_like(self._v_inv_scale)
        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        self._k_aq_scale = (1.0 / self._k_inv_scale).view(bnsd).contiguous()
        self._k_aq_offset = self._k_offset.view(bnsd).contiguous()
        self._v_aq_scale = (1.0 / self._v_inv_scale).view(bnsd).contiguous()
        self._v_aq_offset = self._v_offset.view(bnsd).contiguous()
        self._int8_ready = True
    
    @staticmethod
    def _quantize_kv_to_int8(x, inv_scale, offset):
            """ Quantize K/V from float to INT8 using static per-channel  scales."""
            return torch.clamp(torch.round(x * inv_scale + offset), -128, 127).to(torch.int8)

class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.view(bnsd).contiguous()

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        dense_k = (dense_k.to(target_dtype) - layer._c8_k_offset) * layer._c8_k_scale
        dense_v = (dense_v.to(target_dtype) - layer._c8_v_offset) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
