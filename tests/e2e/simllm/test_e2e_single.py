#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Single-request Sim-LLM E2E smoke tests.

These tests are opt-in because they require an Ascend NPU and a downloadable or
locally available model.  Enable with ``RUN_SIMLLM_E2E=1``.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _require_simllm_e2e() -> None:
    if os.getenv("RUN_SIMLLM_E2E") != "1":
        pytest.skip("Set RUN_SIMLLM_E2E=1 to run Sim-LLM NPU E2E tests.")
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is not installed.")

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch_npu is not usable: {exc}")

    try:
        device_count = torch.npu.device_count()
    except Exception as exc:
        pytest.skip(f"Unable to query Ascend NPU devices: {exc}")

    if not hasattr(torch, "npu") or device_count <= 0:
        pytest.skip("No Ascend NPU device is available.")


@pytest.fixture(scope="module")
def simllm_llm():
    _require_simllm_e2e()
    os.environ["VLLM_ASCEND_SIMLLM_ENABLED"] = "1"

    from vllm import LLM, SamplingParams

    model = os.getenv("SIMLLM_E2E_MODEL", DEFAULT_MODEL)
    llm = LLM(
        model=model,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=8, temperature=0.0)
    return llm, sampling_params


def _assert_generated_tokens(outputs) -> None:
    assert outputs
    for request_output in outputs:
        assert request_output.outputs
        completion = request_output.outputs[0]
        assert getattr(completion, "token_ids", None)


def test_repeated_prompt_warm_cache_smoke(simllm_llm):
    llm, sampling_params = simllm_llm
    prompt = "Explain what KV cache reuse means in one short sentence."

    first_outputs = llm.generate([prompt], sampling_params)
    second_outputs = llm.generate([prompt], sampling_params)

    _assert_generated_tokens(first_outputs)
    _assert_generated_tokens(second_outputs)
