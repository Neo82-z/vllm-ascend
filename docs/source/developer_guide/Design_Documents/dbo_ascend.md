# Experimental DBO on Ascend

This document records the current implementation boundary for Dual Batch
Overlap (DBO) on Ascend. It is intended for development and review. It is not a
performance claim.

## Scope

The first implementation keeps the change narrow:

- Preserve the upstream DBO CLI parameters on Ascend:
  `--enable-dbo`, `--dbo-decode-token-threshold`, and
  `--dbo-prefill-token-threshold`.
- Continue to reset manual `--ubatch-size` on Ascend. Manual ubatching is not
  part of this path.
- Use upstream threshold checks to decide whether the current batch is eligible
  for ubatching.
- Coordinate the batch decision across data-parallel ranks before creating
  ubatch slices, so one rank does not skip a collective while another rank
  enters it.
- Carry ubatch metadata through Ascend forward context.
- Use an eager NPU ubatch wrapper. ACLGraph capture is intentionally out of
  scope for the initial path.
- Add DBO stream handoff around the AllGather MoE dispatch/combine boundary.

## Guarded Combinations

The initial path intentionally avoids feature combinations that need separate
metadata slicing or stream-order validation:

- prefill context parallelism, decode context parallelism, and context
  parallelism;
- sequence parallelism;
- manual `ubatch_size`;
- ACLGraph/NPUGraph capture;
- MC2, Fused MC2, and AllToAll MoE handoff.

These combinations should be enabled only after they have dedicated tests and
hardware validation.

## Validation Boundary

The current tests cover configuration preservation/reset behavior, DBO
threshold routing, DP coordination calls, forward context state propagation,
NPU input slicing, and the AllGather MoE handoff call sites.

Before claiming performance improvement, the implementation still needs Ascend
hardware validation for:

- multi-card DP/EP MoE execution;
- HCCL collective ordering under DBO;
- dispatch/combine overlap with real NPU streams;
- decode and prefill threshold tuning;
- graph capture interaction.
