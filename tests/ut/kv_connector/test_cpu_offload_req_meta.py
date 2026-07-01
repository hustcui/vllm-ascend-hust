# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from dataclasses import dataclass
from pathlib import Path


def _load_req_meta_class():
    source_path = (
        Path(__file__).parents[3]
        / "vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py"
    )
    module = ast.parse(source_path.read_text(), filename=str(source_path))
    req_meta = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ReqMeta"
    )
    namespace = {"dataclass": dataclass}
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[req_meta], type_ignores=[])),
        filename=str(source_path),
        mode="exec",
    )
    exec(code, namespace)
    return namespace["ReqMeta"]


def test_req_meta_update_does_not_advance_cpu_saved_prefix():
    req_meta = _load_req_meta_class()
    req = req_meta(
        gpu_block_ids=[0, 1],
        cpu_block_ids=[100, 101],
        num_scheduled_tokens=256,
        num_computed_tokens=0,
        num_gpu_computed_tokens=0,
        num_cpu_computed_tokens=0,
    )
    cached_step = req_meta(
        gpu_block_ids=[2],
        cpu_block_ids=[102],
        num_scheduled_tokens=1,
        num_computed_tokens=256,
        num_gpu_computed_tokens=256,
        num_cpu_computed_tokens=256,
    )

    req.update(cached_step)

    assert req.gpu_block_ids == [0, 1, 2]
    assert req.cpu_block_ids == [100, 101, 102]
    assert req.num_cpu_computed_tokens == 0


def test_req_meta_update_preserves_cpu_hit_prefix():
    req_meta = _load_req_meta_class()
    req = req_meta(
        gpu_block_ids=[0, 1],
        cpu_block_ids=[100, 101],
        num_scheduled_tokens=1,
        num_computed_tokens=256,
        num_gpu_computed_tokens=0,
        num_cpu_computed_tokens=256,
    )
    cached_step = req_meta(
        gpu_block_ids=[2],
        cpu_block_ids=[102],
        num_scheduled_tokens=128,
        num_computed_tokens=384,
        num_gpu_computed_tokens=256,
        num_cpu_computed_tokens=384,
    )

    req.update(cached_step)

    assert req.num_cpu_computed_tokens == 256
