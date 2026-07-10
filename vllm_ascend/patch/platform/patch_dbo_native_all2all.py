#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import sys

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_ASCEND_DBO_NATIVE_ALL2ALL_BACKENDS = {
    "allgather_reducescatter",
    "flashinfer_all2allv",
    "flashinfer_nvlink_two_sided",
}

_ORIGINAL_PROPERTY_ATTR = "_vllm_ascend_original_use_ubatching_property"
_current_use_ubatching = ParallelConfig.use_ubatching
_original_parallel_config_use_ubatching = getattr(
    getattr(_current_use_ubatching, "fget", None),
    _ORIGINAL_PROPERTY_ATTR,
    _current_use_ubatching,
)
_warned_backends: set[str] = set()


def _is_ascend_native_dbo_parallel_config(parallel_config: ParallelConfig) -> bool:
    return (
        getattr(parallel_config, "enable_dbo", False)
        and getattr(parallel_config, "all2all_backend", None)
        in _ASCEND_DBO_NATIVE_ALL2ALL_BACKENDS
    )


def _inside_vllm_config_post_init() -> bool:
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_name == "__post_init__" and frame.f_globals.get("__name__") == "vllm.config.vllm":
            return True
        frame = frame.f_back
    return False


def _patched_use_ubatching(parallel_config: ParallelConfig) -> bool:
    if _is_ascend_native_dbo_parallel_config(parallel_config) and _inside_vllm_config_post_init():
        backend = getattr(parallel_config, "all2all_backend", None)
        if backend not in _warned_backends:
            logger.warning(
                "[DBO_EXPERIMENTAL] Bypassing vLLM's upstream DeepEP-only "
                "microbatch validation for Ascend native all2all backend. "
                "backend=%s. Runtime DBO remains enabled after config validation.",
                backend,
            )
            _warned_backends.add(backend)
        return False

    return _original_parallel_config_use_ubatching.fget(parallel_config)  # type: ignore[union-attr]


setattr(
    _patched_use_ubatching,
    _ORIGINAL_PROPERTY_ATTR,
    _original_parallel_config_use_ubatching,
)

if ParallelConfig.use_ubatching.fget is not _patched_use_ubatching:
    ParallelConfig.use_ubatching = property(_patched_use_ubatching)  # type: ignore[assignment]
