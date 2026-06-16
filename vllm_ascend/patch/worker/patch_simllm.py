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

"""Sim-LLM worker patch entry point.

Auto-loaded by ``vllm_ascend/patch/worker/__init__.py`` at worker init time.
When ``VLLM_ASCEND_SIMLLM_ENABLED=1``, wraps ``NPUModelRunner.execute_model()``
with Sim-LLM preprocessing / KV reuse / postprocessing hooks.
When disabled the patch is a silent no-op.
"""

from vllm_ascend.simllm.patch.patch_model_runner import apply_simllm_patch

apply_simllm_patch()
