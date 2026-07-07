from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend.ascend_forward_context import set_ascend_forward_context


def _mock_vllm_config():
    vllm_config = MagicMock()
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.is_moe_model = False
    vllm_config.compilation_config.fast_moe_cold_start = False
    vllm_config.compilation_config.static_all_moe_layers = None
    vllm_config.compilation_config.static_forward_context = {}
    return vllm_config


def test_set_ascend_forward_context_forwards_ubatch_state():
    vllm_config = _mock_vllm_config()
    ubatch_slices = [
        UBatchSlice(request_slice=slice(0, 2), token_slice=slice(0, 4)),
        UBatchSlice(request_slice=slice(2, 4), token_slice=slice(4, 8)),
    ]
    slot_mapping = [{"layer.0": torch.tensor([0, 1])}, {"layer.0": torch.tensor([2, 3])}]

    mock_dp_group = MagicMock()
    mock_dp_group.world_size = 1

    with (
        patch("vllm.forward_context.current_platform.set_additional_forward_context", return_value={}),
        patch("vllm_ascend.ascend_forward_context.select_moe_comm_method", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=None),
        patch("vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group),
        patch("vllm_ascend.ascend_forward_context.enable_sp", return_value=False),
        patch("vllm_ascend.ascend_forward_context.flashcomm2_enable", return_value=False),
        patch("vllm_ascend.ascend_forward_context.has_layer_idx", return_value=False),
    ):
        with set_ascend_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            num_tokens=8,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            ubatch_slices=ubatch_slices,
            slot_mapping=slot_mapping,
        ):
            forward_context = get_forward_context()
            assert forward_context.ubatch_slices is ubatch_slices
            assert forward_context.slot_mapping is slot_mapping
            assert forward_context.num_tokens == 8
