# Ascend DBO Implementation

This document describes the current Dual Batch Overlap (DBO) implementation
path on Ascend. The implementation follows upstream vLLM DBO semantics where
possible, and keeps Ascend-specific behavior explicit where NPU streams,
HCCL/MC2 communication, custom operators, or ACLGraph differ from CUDA.

DBO is not a simple CLI switch. The useful path is a coordinated execution
mode across scheduler, DP ranks, attention metadata, model runner, and MoE
communication. This implementation therefore treats DBO as a complete
execution-chain feature, while keeping unsupported combinations guarded until
they have hardware validation.

## Implementation Summary

- Preserves upstream DBO parameters on Ascend: `--enable-dbo`,
  `--dbo-decode-token-threshold`, and `--dbo-prefill-token-threshold`.
- Keeps manual `--ubatch-size` reset on Ascend. Manual ubatching is separate
  from the DBO path and is not enabled here.
- Uses upstream threshold helpers to decide whether the current decode or
  prefill batch should enter ubatching.
- Coordinates the DBO decision across data-parallel ranks before creating
  ubatch slices, so all ranks keep collective ordering consistent.
- Carries ubatch state through Ascend forward context, including request/token
  slice information needed by model runner and attention metadata.
- Adds a minimal eager NPU ubatch wrapper with NPU compute/communication stream
  handoff. ACLGraph capture is intentionally not used for ubatched execution.
- Adds DBO handoff around the first MoE communication boundary implemented in
  this branch, so later MC2/Fused MC2 work can reuse the same scheduling
  shape.
- Adds fallback guards for custom-op availability during MoE startup, allowing
  missing custom operators to be diagnosed instead of being confused with DBO
  scheduling failures.

## Execution Model

The Ascend path keeps the same high-level DBO sequence as upstream vLLM:

1. The model runner receives scheduled token counts for the current step.
2. Decode and prefill thresholds decide whether DBO is eligible for this batch.
3. DP coordination makes the decision rank-consistent before any collective
   operation can be entered.
4. `maybe_create_ubatch_slices` creates two ubatch views of the original batch.
5. Ascend forward context carries the ubatch state to attention, MoE, and model
   runner code.
6. The NPU ubatch wrapper slices model inputs and alternates compute/comm stream
   ownership.
7. MoE dispatch/combine boundaries use the ubatch context to preserve stream
   ordering.

The important invariant is that all ranks either execute the normal batch path
or the same ubatch path. A local threshold decision is not allowed to skip a
collective independently of other ranks.

## Code Walkthrough

The implementation is intentionally split by responsibility:

- `vllm_ascend/platform.py`: keeps `enable_dbo` and upstream DBO thresholds on
  Ascend when the requested configuration is within the current validation
  boundary. It still resets manual `ubatch_size` because that is a separate
  manual microbatching feature, not the automatic DBO path. It also guards
  unverified combinations such as PCP/DCP/context parallelism and sequence
  parallelism. Normal MoE runs keep the Ascend default `flashinfer_all2allv`
  backend. DBO experiments may either preserve an explicit DeepEP backend to
  expose the upstream DeepEP boundary, or use an Ascend-native all2all backend
  through the platform patch described below.
- `vllm_ascend/patch/platform/patch_dbo_native_all2all.py`: bypasses vLLM
  0.23's upstream DeepEP-only microbatch assertion for Ascend-native all2all
  backends. It keeps `enable_dbo` and the real backend intact, temporarily
  suppresses only the `use_ubatching` property during `VllmConfig` post-init,
  and restores normal DBO semantics immediately afterward.
- `vllm_ascend/worker/worker.py`: allocates two workspace slots when DBO is
  enabled. This is deliberately small: workspace allocation follows the config
  decision and does not imply that every batch will be ubatched.
- `vllm_ascend/worker/model_runner_v1.py`: owns the runtime DBO decision. It
  first calls upstream `check_ubatch_thresholds` to decide local eligibility
  from decode/prefill thresholds, then calls `coordinate_batch_across_dp` so all
  DP ranks agree before any collective can run. If the synchronized decision is
  true, it uses `maybe_create_ubatch_slices` and forwards those slices through
  Ascend forward context.
- `vllm_ascend/ascend_forward_context.py`: stores `ubatch_slices` and
  `slot_mapping` in the forward context so attention metadata, model input
  slicing, and MoE communication observe the same batch split.
- `vllm_ascend/worker/npu_ubatch_wrapper.py`: runs the eager NPU ubatch path. It
  slices `input_ids`, `positions`, embeddings, intermediate tensors, attention
  metadata, and slot mapping per ubatch, then executes the wrapped model with
  NPU streams and DBO event handoff. ACLGraph capture is intentionally excluded
  from this first correctness path.
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`: adds the first MoE
  communication handoff boundary. Supported methods yield from compute to
  communication before dispatch/combine and switch back to compute afterward.
  Unsupported communication methods keep eager ordering rather than pretending
  to overlap safely.
- `tests/ut/...`: covers parameter preservation, threshold routing, DP
  coordination, forward-context ubatch state, worker workspace allocation, NPU
  input slicing, and MoE stream handoff hooks.

This file-level split is the main difference from the historical all-in-one
DBO attempt: each layer can become a small upstream PR with its own tests.

## Supported and Guarded Paths

Supported in the current branch:

- DBO configuration preservation on Ascend.
- Decode and prefill threshold routing.
- DP coordination before ubatch slice creation.
- Forward-context ubatch metadata propagation.
- Eager NPU ubatch wrapper.
- Initial MoE communication handoff instrumentation.
- Unit-test coverage for config, threshold routing, DP coordination, forward
  context, NPU input slicing, and MoE handoff calls.

Guarded until separate validation:

- manual `ubatch_size`;
- prefill context parallelism, decode context parallelism, and generic context
  parallelism;
- sequence parallelism;
- ACLGraph/NPUGraph capture and replay;
- full MC2, Fused MC2, and AllToAll overlap claims;
- multi-node performance claims.

These guards are intentional. The rejected historical large DBO attempt mixed
model templates, attention, custom ops, communication, profiling, and metadata
changes in one patch. This branch instead keeps the DBO chain decomposable into
small reviewable changes.

## Custom Operator Boundary

Qwen3-MoE and other large MoE models rely on vLLM-Ascend custom operators for
expert routing and grouped matmul paths. The DBO implementation must therefore
distinguish two failure classes:

- DBO scheduling failures: threshold, ubatch slicing, stream handoff, or DP
  coordination bugs.
- Custom-op availability failures: missing `_C_ascend` registrations,
  incomplete CANN custom-op packages, or mismatched vLLM/vLLM-Ascend versions.

The current branch adds fallback and diagnostics around selected MoE custom-op
entry points, but the preferred production path is still to build and load the
custom operators successfully. A successful custom-op environment should pass:

```bash
find /data/vllm-ascend -name 'vllm_ascend_C*.so' -o -name 'libcust_opapi.so'

python3 - <<'PY'
from vllm_ascend.utils import enable_custom_op
import torch

print("enable_custom_op =", enable_custom_op())
ops = sorted(x for x in torch._C._dispatch_get_all_op_names()
             if x.startswith("_C_ascend::"))
print("custom op count =", len(ops))
print([x for x in ops if "moe" in x.lower()][:50])
PY
```

## Validation Matrix

Validated during this work:

- CANN 9.0.0 dynamic libraries can be loaded when the environment is sourced
  consistently.
- `torch_npu` 2.10.0 can create and copy NPU tensors on 910B.
- Single-node two-card HCCL `all_reduce` and `all_to_all_single` pass.
- vLLM-Ascend plugin loads as `NPUPlatform`.
- Qwen3-MoE W8A8 configuration is recognized as compressed-tensors INT8
  weight/activation quantization.
- vLLM-Ascend custom ops build and register successfully:
  `enable_custom_op=True`, 65 `_C_ascend::` ops are visible, including the MoE
  gating, routing, and grouped matmul operators.
- Qwen3-MoE W8A8 TP=2 + EP server startup passes: both workers load weights,
  KV cache is created, EngineCore warmup completes, and the API server starts.
- Qwen3-MoE W8A8 with DBO enabled and `deepep_high_throughput` preserved also
  reaches API server startup in a TP=2, DP=1 configuration. This validates
  configuration propagation and model wrapping, but not DP=2 coordination.
- A DP=2 + EP + DBO run with `deepep_high_throughput` reaches the DP
  coordinator, two API servers, HCCL worker initialization, DP rank assignment,
  EP rank assignment, and expert placement. It then stops in vLLM's DeepEP
  high-throughput MoE setup because `DeepEPHTPrepareAndFinalize` is referenced
  while the import is guarded by `current_platform.is_cuda_alike()`.
- After this DeepEP boundary was identified, the branch adds an Ascend-native
  DBO gate bypass so the same DP=2 + EP experiment can proceed with the
  platform default `flashinfer_all2allv` backend instead of requiring the
  unavailable upstream `deep_ep` package.
- With the Ascend-native `flashinfer_all2allv` backend, Qwen3-MoE W8A8 DP=2 +
  EP + DBO reaches API server startup. The run starts the DP coordinator and
  two API servers, initializes HCCL with `world_size=2`, assigns DP ranks 0/1
  and EP ranks 0/1, maps 64 local experts per EP rank, loads the 29.07 GiB
  checkpoint on both workers, attaches `NPUUBatchWrapper`, creates KV cache,
  completes EngineCore warmup, and reports `Application startup complete` on
  both API server processes.
- Qwen3-MoE W8A8 DP=2 + EP + DBO prefill microbatching completes a real
  completion request when the experimental MoE handoff is isolated with
  `VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF=1`. With
  `dbo_decode_token_threshold=65536`, `dbo_prefill_token_threshold=128`,
  `max_model_len=384`, `max_num_batched_tokens=384`, and two 160-token
  prompts, both DP ranks agree on `should_ubatch=True`, split the prefill into
  two ubatches, run `NPUUBatchWrapper`, and return HTTP 200. The profiled
  request reports `prompt_tokens=321`, `completion_tokens=2`, and
  `system_fingerprint=vllm-0.23.0-dp2-ep-37aedc47`.
- Torch-NPU profiler collection is available for the same run. The worker
  profile output is written under the configured `torch_profiler_dir` as
  per-rank `*_ascend_pt` directories plus API-server `*.pt.trace.json.gz`
  files. These raw directories require offline `torch_npu.profiler.analyse()`
  before derived CSV/JSON timeline files such as `kernel_details.csv` or
  `trace_view.json` are emitted.
- A decode-only DBO threshold run with
  `dbo_decode_token_threshold=1` and `dbo_prefill_token_threshold=65536`
  completes successfully and emits profiler data. The observed decode steps do
  not split into ubatches because each DP rank owns only one decode token, which
  is below the configured `num_ubatches=2` safety boundary. This is retained as
  evidence that the decode threshold path is safe on Ascend, while decode
  overlap still requires a larger concurrent decode batch or a separate
  profiler run.

Still required before a performance claim:

- multi-card MoE dispatch/combine ordering checks;
- decode/prefill threshold performance sweep;
- MC2/Fused MC2 overlap measurements.
- profiler timeline inspection that shows actual compute/communication overlap;
- DeepEP or equivalent Ascend all2all backend support for true upstream DBO
  stream handoff. The current branch verifies the microbatch execution path and
  provides a switch to isolate the handoff layer, but does not claim final
  overlap performance.

Multi-node DBO communication overlap is not claimed here. The available
resources were sufficient for single-node two-card HCCL and Qwen3-MoE TP=2 +
EP startup. The DBO CLI path also reaches vLLM's upstream DeepEP backend gate:
`enable_dbo=True` produces `use_ubatching=True`, but vLLM currently rejects
non-DeepEP all2all backends for microbatching. The branch therefore presents
the implementation and single-node evidence honestly, while marking DeepEP /
multi-node overlap as follow-up hardware work.

An additional TP=2, DP=1 startup with `deepep_high_throughput` preserved shows
that the platform can carry the DeepEP backend selection through to startup.
A later DP=2 + EP run on the same two cards shows that the system reaches rank
coordination, HCCL initialization, and expert placement. The remaining runtime
boundary has now shifted again: the Ascend-native all2all path reaches server
startup under DBO, so the next evidence to collect is first-token generation
and DBO on/off behavior rather than startup viability.

## Community Submission Strategy

The implementation is complete enough to be presented as an end-to-end DBO
feature branch, but it should not be submitted as one large ready-to-merge PR.
This follows the lesson from earlier large community DBO attempts such as
[`vllm-ascend#4894`](https://github.com/vllm-project/vllm-ascend/pull/4894):
large benchmark-oriented branches are useful as research evidence, but
mergeable upstream work should be split by capability and backed by narrow
tests.
The mergeable route is:

1. DBO config and threshold preservation.
2. DP coordination and threshold tests.
3. Ubatch metadata propagation.
4. Eager NPU ubatch wrapper.
5. MoE communication handoff.
6. Custom-op availability diagnostics and fallbacks.
7. Hardware validation documentation.

This keeps each patch reviewable and avoids repeating the earlier community
failure mode where a very large DBO PR accumulated unrelated conflicts and
unverified communication-order risks.
