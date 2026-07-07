# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    DPMetadata,
    ForwardContext,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker import ubatching as ubatching_state

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper


def _cat_ubatch_outputs(sorted_results: list[Any]) -> Any:
    """Concatenate per-ubatch outputs along the token dimension."""
    if sorted_results and isinstance(sorted_results[0], tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)


def _npu_current_stream() -> Any:
    return torch.npu.current_stream()


def _patch_upstream_ubatching_for_npu() -> None:
    # vLLM's ubatching module is CUDA-oriented. Its dbo_* helpers look up
    # current_stream from the module globals, so redirect that lookup to NPU.
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

    def _restore_context(self) -> None:
        import vllm.forward_context as forward_context

        forward_context._forward_context = self.forward_context

    def update_stream(self, stream: Any) -> None:
        self.current_stream = stream
        if _npu_current_stream() != self.current_stream:
            torch.npu.set_stream(self.current_stream)

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
    compute_stream,
    comm_stream,
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
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )
        self.aclgraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.aclgraph_wrapper = ACLGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=runtime_mode,
            )
        self.device = device

    @property
    def graph_pool(self) -> Any | None:
        if self.aclgraph_wrapper is not None:
            return self.aclgraph_wrapper.graph_pool
        return None

    @property
    def cudagraph_wrapper(self) -> ACLGraphWrapper | None:
        return self.aclgraph_wrapper

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not found on NPUUBatchWrapper")

    def unwrap(self) -> Callable:
        return self.runnable

    def clear_graphs(self) -> None:
        if self.aclgraph_wrapper is not None:
            self.aclgraph_wrapper.clear_graphs()

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        if positions is None:
            sliced_positions = None
        elif positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )
        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
    ) -> list[NPUUbatchMetadata]:
        forward_contexts = []
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                )
            )

        ubatch_ctxs = make_npu_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[NPUUbatchMetadata] = []
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

    def _run_ubatches(self, ubatch_metadata, args, kwargs) -> Any:
        results: list[tuple[int, Any]] = []
        errors: list[BaseException] = []

        @torch.inference_mode()
        def _ubatch_thread(metadata: NPUUbatchMetadata) -> None:
            try:
                with metadata.context:
                    model_kwargs = dict(kwargs)
                    model_kwargs.update(
                        input_ids=metadata.input_ids,
                        positions=metadata.positions,
                        intermediate_tensors=metadata.intermediate_tensors,
                        inputs_embeds=metadata.inputs_embeds,
                    )
                    model_output = self.runnable(*args, **model_kwargs)
                results.append((metadata.context.id, model_output))
            except BaseException as exc:
                errors.append(exc)
                metadata.context.cpu_signal_event.set()

        with override_forward_context(None):
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
        sorted_results = [value for position, value in sorted(results)]
        return _cat_ubatch_outputs(sorted_results)

    def __call__(self, *args, **kwargs) -> Any:
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        if ubatch_slices is None:
            if (
                forward_context.cudagraph_runtime_mode is not CUDAGraphMode.NONE
                and self.aclgraph_wrapper is not None
            ):
                return self.aclgraph_wrapper(*args, **kwargs)
            return self.runnable(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        input_ids = kwargs.get("input_ids")
        positions = kwargs.get("positions")
        intermediate_tensors = kwargs.get("intermediate_tensors")
        inputs_embeds = kwargs.get("inputs_embeds")
        compute_stream = torch.npu.current_stream()

        dp_metadata = forward_context.dp_metadata
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if forward_context.cudagraph_runtime_mode is not CUDAGraphMode.NONE:
            logger.debug_once(
                "[DBO_EXPERIMENTAL] NPU ubatch execution currently runs "
                "microbatches eagerly; ACLGraph capture/replay for ubatches "
                "is left disabled."
            )

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=ubatch_slices,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            compute_stream=compute_stream,
            dp_metadata=ubatch_dp_metadata,
            batch_descriptor=forward_context.batch_descriptor,
        )
        return self._run_ubatches(ubatch_metadata, args, kwargs)
