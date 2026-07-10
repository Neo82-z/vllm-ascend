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

from functools import wraps

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_ASCEND_DBO_NATIVE_ALL2ALL_BACKENDS = {
    "allgather_reducescatter",
    "flashinfer_all2allv",
    "flashinfer_nvlink_two_sided",
}

_original_vllm_config_post_init = VllmConfig.__post_init__
_original_parallel_config_use_ubatching = ParallelConfig.use_ubatching


def _is_ascend_native_dbo_config(config: VllmConfig) -> bool:
    parallel_config = getattr(config, "parallel_config", None)
    if parallel_config is None:
        return False
    return (
        getattr(parallel_config, "enable_dbo", False)
        and getattr(parallel_config, "all2all_backend", None)
        in _ASCEND_DBO_NATIVE_ALL2ALL_BACKENDS
    )


@wraps(_original_vllm_config_post_init)
def _patched_vllm_config_post_init(self: VllmConfig):
    if not _is_ascend_native_dbo_config(self):
        return _original_vllm_config_post_init(self)

    parallel_config = self.parallel_config

    # Upstream vLLM 0.23 validates DBO microbatching before platform-specific
    # Ascend MoE communication setup can run, and currently accepts only
    # DeepEP all2all backends. vLLM-Ascend has its own HCCL/FlashComm/All2All
    # communication paths, so keep enable_dbo and the native backend intact.
    # Only suppress the upstream use_ubatching property for this single config
    # while VllmConfig runs the DeepEP-only assert.
    def _patched_use_ubatching(config: ParallelConfig) -> bool:
        if config is parallel_config and _is_ascend_native_dbo_config(self):
            return False
        return _original_parallel_config_use_ubatching.fget(config)  # type: ignore[union-attr]

    ParallelConfig.use_ubatching = property(_patched_use_ubatching)  # type: ignore[assignment]
    try:
        result = _original_vllm_config_post_init(self)
    finally:
        ParallelConfig.use_ubatching = _original_parallel_config_use_ubatching  # type: ignore[assignment]

    if not getattr(parallel_config, "enable_dbo", False):
        return result

    model_config = getattr(self, "model_config", None)
    if model_config is not None and not getattr(model_config, "disable_cascade_attn", False):
        model_config.disable_cascade_attn = True
        logger.warning_once("Disabling cascade attention when Ascend DBO is enabled.")

    logger.warning_once(
        "[DBO_EXPERIMENTAL] Allowing Ascend native all2all backend for DBO. "
        "backend=%s use_ubatching=%s num_ubatches=%s. This bypasses vLLM's "
        "upstream DeepEP-only microbatch validation and must be treated as an "
        "Ascend-specific PoC until multi-rank MoE communication overlap is "
        "verified.",
        parallel_config.all2all_backend,
        parallel_config.use_ubatching,
        parallel_config.num_ubatches,
    )
    return result


if getattr(VllmConfig.__post_init__, "__name__", "") != _patched_vllm_config_post_init.__name__:
    VllmConfig.__post_init__ = _patched_vllm_config_post_init
