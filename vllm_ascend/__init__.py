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

try:
    from ._version import (  # noqa: F401
        __commit_id__,
        __upstream_commit__,
        __upstream_version__,
        __version__,
        __version_tuple__,
    )
except Exception:
    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
    __upstream_version__ = None
    __upstream_commit__ = None
    __commit_id__ = None


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()
