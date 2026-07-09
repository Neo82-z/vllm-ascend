#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from typing import Any

import torch
from vllm.utils.math_utils import cdiv

from .base import QuantType
from .registry import register_scheme
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod


def _get_weight_block_size(quant_config: dict[str, Any] | None) -> tuple[int, int] | None:
    if quant_config is None:
        return None
    block_size = quant_config.get("weight_block_size")
    if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
        return None
    return int(block_size[0]), int(block_size[1])


def _dequant_scale_from_inv(scale_inv: torch.Tensor) -> torch.Tensor:
    # Qwen3/MiniMax block-FP8 checkpoints store the dequant multiplier
    # under the historical `weight_scale_inv` name.
    return scale_inv.to(torch.float32).contiguous()


@register_scheme("W8A8FP8_DYNAMIC", "linear")
class AscendW8A8FP8DynamicLinearMethod(AscendW8A8DynamicLinearMethod):
    """Linear method for Ascend W8A8FP8_DYNAMIC.

    This scheme uses FP8 dynamic per-token quantization for activations
    and FP8 per-channel quantization for weights.
    """

    act_quant_type: torch.dtype = torch.float8_e4m3fn

    def __init__(self, quant_config: dict[str, Any] | None = None):
        self.weight_block_size = _get_weight_block_size(quant_config)
        self.has_block_scale_inv = self.weight_block_size is not None

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        if self.has_block_scale_inv:
            return params_dict
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        if self.weight_block_size is None:
            return {}
        block_n, block_k = self.weight_block_size
        return {
            "weight_scale_inv": torch.empty(
                cdiv(output_size, block_n),
                cdiv(input_size, block_k),
                dtype=params_dtype,
            ),
            "_packed_dim": 0,
            "_packed_factor": block_n,
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        output = super().apply(layer, x, bias, tp_rank)
        # TODO: there is a bug in npu_quant_matmul for fp8 with bias
        # after the bug is fixed, the whole apply method can be removed.
        if bias is not None:
            output = (output + bias).to(x.dtype)
        return output

    def process_weights_after_loading(self, layer):
        if not self.has_block_scale_inv:
            return super().process_weights_after_loading(layer)

        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale = _dequant_scale_from_inv(layer.weight_scale_inv.data)
        layer.weight_scale_fp32 = layer.weight_scale


@register_scheme("W8A8FP8_DYNAMIC", "moe")
class AscendW8A8FP8DynamicFusedMoEMethod(AscendW8A8DynamicFusedMoEMethod):
    """FusedMoE method for Ascend W8A8FP8_DYNAMIC."""

    quant_type: QuantType = QuantType.W8A8FP8

    def __init__(self, quant_config: dict[str, Any] | None = None):
        super().__init__()
        self.weight_block_size = _get_weight_block_size(quant_config)
        self.has_block_scale_inv = self.weight_block_size is not None

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        if self.weight_block_size is not None:
            block_n, block_k = self.weight_block_size
            param_dict["w13_weight_scale_inv"] = torch.empty(
                num_experts,
                cdiv(2 * intermediate_size_per_partition, block_n),
                cdiv(hidden_sizes, block_k),
                dtype=params_dtype,
            )
            param_dict["w2_weight_scale_inv"] = torch.empty(
                num_experts,
                cdiv(hidden_sizes, block_n),
                cdiv(intermediate_size_per_partition, block_k),
                dtype=params_dtype,
            )
            return param_dict
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict

    def process_weights_after_loading(self, layer):
        if not self.has_block_scale_inv:
            return super().process_weights_after_loading(layer)

        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        # Checkpoint scale_inv follows the original (N, K) block layout.
        # The weights are transposed above for Ascend kernels, so transpose
        # block-scale axes in the same way.
        layer.w13_weight_scale = _dequant_scale_from_inv(layer.w13_weight_scale_inv.data).transpose(1, 2).contiguous()
        layer.w2_weight_scale = _dequant_scale_from_inv(layer.w2_weight_scale_inv.data).transpose(1, 2).contiguous()
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale

        if self.dynamic_eplb:
            layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
            layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
            layer.w13_weight_scale_fp32_list = [
                weight.clone() for weight in layer.w13_weight_scale_fp32.data.unbind(dim=0)
            ]
            layer.w2_weight_scale_list = [weight.clone() for weight in layer.w2_weight_scale.data.unbind(dim=0)]
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w13_weight_scale_fp32
            del layer.w2_weight_scale
            torch.npu.empty_cache()
