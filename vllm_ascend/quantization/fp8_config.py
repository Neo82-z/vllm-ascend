from typing import Any, Optional, cast

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm_ascend.utils import FP8_METHOD, vllm_version_is

if vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import FusedMoE
else:
    try:
        from vllm.model_executor.layers.fused_moe import MoERunner
    except ImportError:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from .methods import get_scheme_class


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    if vllm_version_is("0.23.0"):
        return isinstance(layer, FusedMoE)
    else:
        return isinstance(layer, MoERunner)


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


def remove_quantization_method():
    if FP8_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(FP8_METHOD)
    if "deepseek_v4_fp8" in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove("deepseek_v4_fp8")


remove_quantization_method()


def create_scheme_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
    quant_type: str = "FP8",
):
    """Create a quantization scheme instance for a layer.

    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        layer_type: The type of layer ("linear", "moe", "attention").
        packed_modules_mapping: Mapping for packed/fused modules.
        quant_type: The registered Ascend quantization type.

    Returns:
        An instance of the appropriate quantization scheme class.
    """
    logger.info_once("Using the vLLM Ascend fp8 Quantization now!")

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        if quant_type == "W8A8_MXFP8":
            return scheme_cls()
        return scheme_cls(quant_description)

    raise NotImplementedError(f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}.")


def _uses_deepseek_fp8_layout() -> bool:
    from vllm.config import get_current_vllm_config

    hf_config = get_current_vllm_config().model_config.hf_config
    return all(hasattr(hf_config, attr) for attr in ("o_groups", "o_lora_rank"))


def _uses_block_fp8_layout(quant_description: dict[str, Any]) -> bool:
    return quant_description.get("weight_block_size") is not None


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(QuantizationConfig):
    def __init__(
        self,
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        self.quant_description = config if config is not None else {}

    def __repr__(self) -> str:
        return "Fp8Config:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return FP8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float8_e4m3fn, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendFp8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = FP8_METHOD

            if _uses_deepseek_fp8_layout():
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "ds_linear",
                    self.packed_modules_mapping,
                )
            else:
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "linear",
                    self.packed_modules_mapping,
                    quant_type="W8A8FP8_DYNAMIC"
                    if _uses_block_fp8_layout(self.quant_description)
                    else "W8A8_MXFP8",
                )
            quant_method = AscendLinearMethod(scheme)
            return quant_method
        if _is_fused_moe_layer(layer):
            layer.ascend_quant_method = FP8_METHOD
            if _uses_deepseek_fp8_layout():
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "w4a8_moe",
                    self.packed_modules_mapping,
                )
            else:
                scheme = create_scheme_for_layer(
                    self.quant_description,
                    prefix,
                    "moe",
                    self.packed_modules_mapping,
                    quant_type="W8A8FP8_DYNAMIC"
                    if _uses_block_fp8_layout(self.quant_description)
                    else "W8A8_MXFP8",
                )
            quant_method = AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


# deepseek_v4_fp8 is handled identically to fp8 on Ascend — reuse the same config.
register_quantization_config("deepseek_v4_fp8")(AscendFp8Config)
