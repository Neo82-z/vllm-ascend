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
  parallelism.
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

Still required before a performance claim:

- first-token generation smoke;
- DBO on/off correctness comparison;
- DeepEP or equivalent Ascend all2all backend support for true upstream DBO
  microbatch execution;
- multi-card MoE dispatch/combine ordering checks;
- decode/prefill threshold performance sweep;
- MC2/Fused MC2 overlap measurements.

Multi-node DBO communication overlap is not claimed here. The available
resources were sufficient for single-node two-card HCCL and Qwen3-MoE TP=2 +
EP startup. The DBO CLI path also reaches vLLM's upstream DeepEP backend gate:
`enable_dbo=True` produces `use_ubatching=True`, but vLLM currently rejects
non-DeepEP all2all backends for microbatching. The branch therefore presents
the implementation and single-node evidence honestly, while marking DeepEP /
multi-node overlap as follow-up hardware work.

## Community Submission Strategy

The implementation is complete enough to be presented as an end-to-end DBO
feature branch, but it should not be submitted as one large ready-to-merge PR.
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
