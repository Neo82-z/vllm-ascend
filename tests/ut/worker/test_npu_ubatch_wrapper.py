import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.npu_ubatch_wrapper import NPUUBatchWrapper


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

    sliced_input_ids, sliced_positions, sliced_inputs_embeds, sliced_intermediate_tensors = sliced
    assert torch.equal(sliced_input_ids, input_ids[2:5])
    assert torch.equal(sliced_positions, positions[2:5])
    assert torch.equal(sliced_inputs_embeds, inputs_embeds[2:5])
    assert torch.equal(sliced_intermediate_tensors["hidden"], intermediate_tensors["hidden"][2:5])


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
