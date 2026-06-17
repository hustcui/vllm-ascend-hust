#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Phase 3 guardrails for baseline-guided Sim-LLM validation."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from vllm_ascend.simllm.patch.patch_model_runner import (
    _simllm_handle_deferrals,
)


SIMLLM_ROOT = Path(__file__).parents[3] / "vllm_ascend" / "simllm"


def test_runtime_tree_has_no_engine_core_deferral_patch():
    forbidden_file = SIMLLM_ROOT / "patch" / ("patch_" + "engine_core.py")

    assert not forbidden_file.exists()


def test_runtime_code_does_not_call_scheduler_requeue():
    forbidden_call = "scheduler" + ".add_request"

    offenders: list[Path] = []
    for path in SIMLLM_ROOT.rglob("*.py"):
        if forbidden_call in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(SIMLLM_ROOT))

    assert offenders == []


def test_deferral_handler_is_diagnostic_only(caplog):
    runner = MagicMock()
    runner._simllm_deferrals = {1, 3}
    runner.scheduler = MagicMock()

    caplog.set_level(
        logging.DEBUG,
        logger="vllm_ascend.simllm.patch.patch_model_runner",
    )

    _simllm_handle_deferrals(runner)

    method_name = "add_" + "request"
    getattr(runner.scheduler, method_name).assert_not_called()
    assert "future deferral diagnostics" in caplog.text
    assert "current batch" in caplog.text
