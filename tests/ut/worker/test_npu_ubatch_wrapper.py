from types import SimpleNamespace

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker.npu_ubatch_wrapper import (
    NPUUBatchWrapper,
    _copy_ascend_extra_forward_context,
)


def test_slice_model_inputs_slices_token_dimension():
    wrapper = object.__new__(NPUUBatchWrapper)
    input_ids = torch.arange(6)
    positions = torch.arange(6)
    inputs_embeds = torch.arange(18).view(6, 3)
    intermediate_tensors = IntermediateTensors({"hidden": torch.arange(24).view(6, 4)})

    sliced = wrapper._slice_model_inputs(
        slice(2, 5),
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    )

    (
        sliced_input_ids,
        sliced_positions,
        sliced_inputs_embeds,
        sliced_intermediate_tensors,
    ) = sliced
    assert torch.equal(sliced_input_ids, input_ids[2:5])
    assert torch.equal(sliced_positions, positions[2:5])
    assert torch.equal(sliced_inputs_embeds, inputs_embeds[2:5])
    assert torch.equal(
        sliced_intermediate_tensors["hidden"],
        intermediate_tensors["hidden"][2:5],
    )


def test_slice_model_inputs_slices_mrope_positions():
    wrapper = object.__new__(NPUUBatchWrapper)
    positions = torch.arange(18).view(3, 6)

    _, sliced_positions, _, _ = wrapper._slice_model_inputs(
        slice(1, 4),
        input_ids=None,
        positions=positions,
        inputs_embeds=None,
        intermediate_tensors=None,
    )

    assert torch.equal(sliced_positions, positions[:, 1:4])


def test_copy_ascend_extra_forward_context_for_ubatch():
    moe_comm_method = object()
    parent_context = SimpleNamespace(
        additional_kwargs={},
        moe_comm_type=MoECommType.ALLTOALL,
        moe_comm_method=moe_comm_method,
        flash_comm_v1_enabled=False,
        flashcomm_v2_enabled=False,
        mc2_mask=torch.ones(8, dtype=torch.bool),
    )
    child_context = SimpleNamespace(additional_kwargs={})
    ubatch_slice = UBatchSlice(request_slice=slice(0, 1), token_slice=slice(0, 3))
    num_tokens_across_dp = torch.tensor([3, 3], dtype=torch.int32)

    _copy_ascend_extra_forward_context(
        parent_context=parent_context,
        child_context=child_context,
        ubatch_slice=ubatch_slice,
        ubatch_num_tokens_across_dp=num_tokens_across_dp,
        tensor_parallel_size=2,
    )

    assert child_context.moe_comm_type == MoECommType.ALLTOALL
    assert child_context.moe_comm_method is moe_comm_method
    assert child_context.additional_kwargs["moe_comm_method"] is moe_comm_method
    assert child_context.num_tokens == 3
    assert child_context.max_tokens_across_dp == 3
    assert child_context.padded_num_tokens == 4
    assert child_context.pad_size == 0
    assert torch.equal(
        child_context.num_tokens_across_dp,
        num_tokens_across_dp,
    )
    assert torch.equal(
        child_context.mc2_mask,
        torch.tensor([True, True, True, False]),
    )
