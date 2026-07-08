# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import ubatching as ubatching_state
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


def _cat_ubatch_outputs(sorted_results: list[Any]) -> Any:
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


def _npu_current_stream() -> Any:
    return torch.npu.current_stream()


def _patch_upstream_ubatching_for_npu() -> None:
    # vLLM's DBO helpers resolve current_stream from module globals. Redirect
    # that lookup to NPU so the same yield helpers can be used on Ascend.
    ubatching_state.current_stream = _npu_current_stream

    def npu_dbo_get_previous_event(func, *args, **kwargs):
        if len(ubatching_state._THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = ubatching_state._THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = ubatching_state._CURRENT_CONTEXTS[ctx_idx]
            with torch.npu.stream(ctx.compute_stream):
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    ubatching_state.dbo_get_previous_event = npu_dbo_get_previous_event


@dataclass
class NPUUbatchMetadata:
    context: "NPUUBatchContext"
    input_ids: torch.Tensor | None
    positions: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


class NPUUBatchContext:
    def __init__(
        self,
        id: int,
        comm_stream: Any,
        compute_stream: Any,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        npu_comm_done_event: Any,
        npu_compute_done_event: Any,
    ) -> None:
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = npu_comm_done_event
        self.gpu_compute_done_event = npu_compute_done_event
        self.recv_hook = None

    def __enter__(self) -> "NPUUBatchContext":
        ubatching_state._THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        ubatching_state._CURRENT_CONTEXTS[self.id] = self
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        ubatching_state._CURRENT_CONTEXTS[self.id] = None
        del ubatching_state._THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def update_stream(self, stream: Any) -> None:
        self.current_stream = stream
        if _npu_current_stream() != self.current_stream:
            torch.npu.set_stream(self.current_stream)

    def _restore_context(self) -> None:
        import vllm.forward_context as forward_context

        forward_context._forward_context = self.forward_context

    def _signal_comm_done(self) -> None:
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self) -> None:
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self) -> None:
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self) -> None:
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self) -> None:
        import vllm.forward_context as forward_context

        assert forward_context._forward_context == self.forward_context
        assert _npu_current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def yield_(self) -> None:
        self.current_stream = _npu_current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self) -> None:
        assert _npu_current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self) -> None:
        assert _npu_current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def switch_to_comm(self) -> None:
        self.update_stream(self.comm_stream)

    def switch_to_compute(self) -> None:
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self) -> None:
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self) -> None:
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self) -> None:
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None


def make_npu_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: Any,
    comm_stream: Any,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
) -> list[NPUUBatchContext]:
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"
    _patch_upstream_ubatching_for_npu()

    ubatching_state._NUM_UBATCHES = num_micro_batches
    if len(ubatching_state._CURRENT_CONTEXTS) < num_micro_batches:
        ubatching_state._CURRENT_CONTEXTS.extend(
            [None] * (num_micro_batches - len(ubatching_state._CURRENT_CONTEXTS))
        )

    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    npu_comm_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]
    npu_compute_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]

    return [
        NPUUBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            npu_comm_done_event=npu_comm_done_events[i],
            npu_compute_done_event=npu_compute_done_events[i],
        )
        for i in range(num_micro_batches)
    ]


class NPUUBatchWrapper:
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.device,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )
        self.device = device
        if runtime_mode != CUDAGraphMode.NONE:
            logger.warning(
                "[DBO_EXPERIMENTAL] NPU ubatching currently runs eagerly; "
                "ACLGraph capture/replay is skipped for ubatched execution."
            )

    @property
    def graph_pool(self) -> None:
        return None

    @property
    def cudagraph_wrapper(self) -> None:
        return None

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not found on NPUUBatchWrapper")

    def unwrap(self) -> Callable:
        return self.runnable

    @staticmethod
    def _slice_model_inputs(
        token_slice: slice,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        IntermediateTensors | None,
    ]:
        sliced_input_ids = input_ids[token_slice] if input_ids is not None else None
        if positions is None:
            sliced_positions = None
        elif positions.ndim == 2:
            sliced_positions = positions[:, token_slice]
        else:
            sliced_positions = positions[token_slice]
        sliced_inputs_embeds = (
            inputs_embeds[token_slice] if inputs_embeds is not None else None
        )
        sliced_intermediate_tensors = None
        if intermediate_tensors is not None:
            sliced_intermediate_tensors = IntermediateTensors(
                {k: v[token_slice] for k, v in intermediate_tensors.items()}
            )
        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    @staticmethod
    def _slice_slot_mapping(
        slot_mapping: Any,
        ubatch_slice: UBatchSlice,
        ubatch_index: int,
    ) -> Any:
        if isinstance(slot_mapping, list):
            return slot_mapping[ubatch_index]
        if isinstance(slot_mapping, dict):
            return {
                layer_name: value[ubatch_slice.token_slice]
                for layer_name, value in slot_mapping.items()
            }
        return slot_mapping

    def _make_forward_contexts(
        self,
        forward_context: ForwardContext,
        ubatch_slices: UBatchSlices,
    ) -> list[ForwardContext]:
        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            ubatch_attn_metadata = (
                attn_metadata[i] if isinstance(attn_metadata, list) else attn_metadata
            )
            ubatch_slot_mapping = self._slice_slot_mapping(slot_mapping, ubatch_slice, i)
            dp_metadata = None
            if self.vllm_config.parallel_config.data_parallel_size > 1:
                ubatch_num_tokens_across_dp = torch.tensor(
                    [ubatch_slice.num_tokens]
                    * self.vllm_config.parallel_config.data_parallel_size,
                    device="cpu",
                    dtype=torch.int32,
                )
                dp_metadata = DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            contexts.append(
                create_forward_context(
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    batch_descriptor=BatchDescriptor(ubatch_slice.num_tokens),
                    ubatch_slices=None,
                    slot_mapping=ubatch_slot_mapping,
                    additional_kwargs=forward_context.additional_kwargs,
                    skip_compiled=forward_context.skip_compiled,
                )
            )
        return contexts

    def _make_ubatch_metadata(
        self,
        ubatch_slices: UBatchSlices,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> list[NPUUbatchMetadata]:
        forward_context = get_forward_context()
        forward_contexts = self._make_forward_contexts(forward_context, ubatch_slices)
        ubatch_ctxs = make_npu_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            compute_stream=_npu_current_stream(),
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                NPUUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                )
            )
        return ubatch_metadata

    def _run_ubatches(self, ubatch_metadata: list[NPUUbatchMetadata], kwargs) -> Any:
        results: list[tuple[int, Any]] = []
        errors: list[BaseException] = []

        def _ubatch_thread(metadata: NPUUbatchMetadata) -> None:
            try:
                thread_kwargs = dict(kwargs)
                thread_kwargs.update(
                    input_ids=metadata.input_ids,
                    positions=metadata.positions,
                    inputs_embeds=metadata.inputs_embeds,
                    intermediate_tensors=metadata.intermediate_tensors,
                )
                with metadata.context, override_forward_context(
                    metadata.context.forward_context
                ):
                    results.append((metadata.context.id, self.runnable(**thread_kwargs)))
            except BaseException as exc:
                errors.append(exc)
                metadata.context.cpu_signal_event.set()

        ubatch_threads = [
            threading.Thread(target=_ubatch_thread, args=(metadata,))
            for metadata in ubatch_metadata
        ]
        for thread in ubatch_threads:
            thread.start()

        self.ready_barrier.wait()
        ubatch_metadata[0].context.cpu_wait_event.set()
        for thread in ubatch_threads:
            thread.join()

        if errors:
            raise errors[0]

        sorted_results = [result for _, result in sorted(results, key=lambda x: x[0])]
        return _cat_ubatch_outputs(sorted_results)

    def __call__(self, *args, **kwargs) -> Any:
        if args:
            return self.runnable(*args, **kwargs)
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        if ubatch_slices is None:
            return self.runnable(**kwargs)
        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=ubatch_slices,
            input_ids=kwargs.get("input_ids"),
            positions=kwargs.get("positions"),
            inputs_embeds=kwargs.get("inputs_embeds"),
            intermediate_tensors=kwargs.get("intermediate_tensors"),
        )
        return self._run_ubatches(ubatch_metadata, kwargs)
