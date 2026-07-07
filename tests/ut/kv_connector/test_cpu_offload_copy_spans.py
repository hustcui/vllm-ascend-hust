# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from collections.abc import Sequence
from pathlib import Path


def _load_span_helper():
    helper = _load_source_function("_coalesce_block_copy_spans")
    namespace = {"Sequence": Sequence, "BlockCopySpan": tuple}
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
        filename=str(_source_path()),
        mode="exec",
    )
    exec(code, namespace)
    return namespace["_coalesce_block_copy_spans"]


def _load_directional_span_helper():
    helper = _load_source_function("_coalesce_directional_block_copy_spans")
    namespace = {"Sequence": Sequence, "DirectionalBlockCopySpan": tuple}
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
        filename=str(_source_path()),
        mode="exec",
    )
    exec(code, namespace)
    return namespace["_coalesce_directional_block_copy_spans"]


def _source_path():
    return (
        Path(__file__).parents[3]
        / "vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py"
    )


def _load_source_function(name):
    source_path = _source_path()
    module = ast.parse(source_path.read_text(), filename=str(source_path))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_worker_method(name):
    source_path = (
        Path(__file__).parents[3]
        / "vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py"
    )
    module = ast.parse(source_path.read_text(), filename=str(source_path))
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "CPUOffloadingConnectorWorker"
    )
    return next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_coalesce_block_copy_spans_empty():
    coalesce = _load_span_helper()

    assert coalesce([]) == []


def test_coalesce_block_copy_spans_merges_dual_contiguous_ids():
    coalesce = _load_span_helper()

    assert coalesce([(10, 20), (11, 21), (12, 22)]) == [(10, 20, 3)]


def test_coalesce_block_copy_spans_splits_on_either_side_gap():
    coalesce = _load_span_helper()

    assert coalesce([(0, 5), (1, 6), (4, 8), (5, 9), (6, 12)]) == [
        (0, 5, 2),
        (4, 8, 2),
        (6, 12, 1),
    ]


def test_coalesce_block_copy_spans_preserves_mapping_order():
    coalesce = _load_span_helper()

    assert coalesce([(3, 8), (4, 9), (2, 7), (3, 8)]) == [
        (3, 8, 2),
        (2, 7, 2),
    ]


def test_save_mapping_can_reuse_cpu_gpu_span_helper():
    coalesce = _load_span_helper()
    save_block_mapping = [(20, 10), (21, 11), (31, 14), (32, 15)]
    rank_save_block_mapping = [
        (cpu_block_id, gpu_block_id)
        for gpu_block_id, cpu_block_id in save_block_mapping
    ]

    assert coalesce(rank_save_block_mapping) == [(10, 20, 2), (14, 31, 2)]


def test_directional_spans_merge_reverse_gpu_ids():
    coalesce = _load_directional_span_helper()

    assert coalesce([(10, 20), (11, 19), (12, 18), (15, 7), (16, 8)]) == [
        (10, 20, 3, -1),
        (15, 7, 2, 1),
    ]


def test_save_req_uses_span_helper_and_reports_reduction_fields():
    save_req = _load_worker_method("_save_req")
    calls = [
        node.func.id
        for node in ast.walk(save_req)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    source = _source_path().read_text()

    assert "_coalesce_block_copy_spans" in calls
    assert "rank_save_blocks=%d spans=%d cpu_adjacent_pairs=%d gpu_adjacent_pairs=%d" in source
    assert "contiguous_blocks=%d reverse_blocks=%d fallback_blocks=%d" in source
    assert "original_copy_ops=%d copy_ops_saved=%d" in source


def test_get_finished_runs_save_in_worker_thread():
    get_finished = _load_worker_method("get_finished")
    calls = [
        node.func.attr
        for node in ast.walk(get_finished)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    source = _source_path().read_text()

    assert "_save_req" in calls
    assert "_save_listener" not in source
    assert "save_thread" not in source
