#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests for Sim-LLM worker patch timing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


PATCH_SIMLLM_PATH = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "patch"
    / "worker"
    / "patch_simllm.py"
)


def _load_patch_simllm_module():
    spec = importlib.util.spec_from_file_location(
        "simllm_patch_timing_under_test",
        PATCH_SIMLLM_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simllm_patch_defers_until_model_runner_class_exists(monkeypatch):
    calls = []
    module = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.model_runner_v1", module)
    patch_simllm = _load_patch_simllm_module()
    monkeypatch.setattr(
        patch_simllm,
        "apply_simllm_patch",
        lambda model_runner_cls: calls.append(model_runner_cls),
    )

    patch_simllm.try_apply_simllm_patch()

    assert calls == []


def test_simllm_patch_applies_after_model_runner_class_exists(monkeypatch):
    calls = []

    class DummyNPUModelRunner:
        pass

    module = SimpleNamespace(NPUModelRunner=DummyNPUModelRunner)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.model_runner_v1", module)
    patch_simllm = _load_patch_simllm_module()
    monkeypatch.setattr(
        patch_simllm,
        "apply_simllm_patch",
        lambda model_runner_cls: calls.append(model_runner_cls),
    )

    patch_simllm.try_apply_simllm_patch()

    assert calls == [DummyNPUModelRunner]
