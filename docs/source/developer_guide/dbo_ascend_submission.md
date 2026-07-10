# vLLM-Ascend DBO Final Submission Notes

This page is the repository-facing delivery index for the CCF vLLM-Ascend DBO
work. It summarizes what the branch implements, what has been verified, and how
the remaining hardware-dependent checks should be reproduced.

## Review Summary

The branch implements the full Ascend DBO execution chain rather than only
keeping the CLI arguments alive. The main contribution is to connect DBO from
configuration into scheduling, rank coordination, ubatch metadata, model-runner
execution, and MoE communication boundaries.

The work also records the engineering reality of the Ascend stack: CANN version
selection, `torch_npu` compatibility, custom-op compilation, vLLM main API
drift, and Qwen3-MoE quantization support are all part of making DBO usable on
real 910B hardware. These findings are included so the implementation can be
reviewed and reproduced without hiding the integration risks.

## Code Contribution Map

- `a6d178b5` preserves experimental DBO config on Ascend while keeping manual
  `ubatch_size` disabled.
- `bdf37cef` routes decode/prefill thresholds through DP coordination.
- `06522747` forwards ubatch metadata through Ascend forward context.
- `5e5d4c72` adds the eager NPU ubatch wrapper and input slicing tests.
- `74a573f5` adds DBO handoff around the MoE communication boundary.
- `f365ec22` documents the initial DBO implementation boundary.
- `70329450`, `b972b1c9`, and `43598d20` guard vLLM main API drift discovered
  during Qwen3-MoE smoke work.
- `89a283cd`, `6c237f89`, and `2ea2b306` adapt FP8 block-scale handling for
  current Qwen3-style checkpoints.
- `5e13b3b9`, `74330ee5`, and `815dc905` add diagnostics and fallbacks when
  MoE custom operators are missing or intentionally disabled.

The branch is intentionally organized as a small-PR series. It should be
reviewed by capability, not as one monolithic upstream PR.

## Verification Evidence

Verified environment facts:

- CANN 9.0.0 toolkit, NNAL, and 910B ops can be made active together.
- `libgraph.so`, `libgraph_base.so`, `libhcomm.so`, `libhccl.so`, and
  `libascendcl.so` load after the CANN 9.0.0 environment is sourced.
- `torch 2.10.0+cpu` and `torch_npu 2.10.0` report two visible NPUs on 910B.
- `torch.ones(..., device="npu")`, elementwise add, and CPU copy pass.
- Single-node two-card HCCL `all_reduce` and `all_to_all_single` pass.
- `VLLM_PLUGINS=ascend` activates `vllm_ascend.platform.NPUPlatform`.
- vLLM-Ascend custom ops build and register successfully after adding the
  package directory to `LD_LIBRARY_PATH`: `enable_custom_op=True`, 65
  `_C_ascend::` ops are visible, including `moe_gating_top_k`,
  `moe_grouped_matmul`, and `npu_moe_init_routing_custom`.
- Qwen3-30B-A3B W8A8 is detected as `compressed-tensors` INT8 W8A8
  quantization.
- Qwen3-30B-A3B W8A8 starts with `tensor_parallel_size=2` and
  `enable_expert_parallel=True`: both TP/EP workers load weights, KV cache is
  created, EngineCore warmup completes, and the OpenAI-compatible API server
  starts on port 8000.

Current hardware-dependent boundary:

- First-token generation and DBO on/off behavioral comparison should be run on
  top of the successful Qwen3-MoE server startup.
- `--enable-dbo --dbo-decode-token-threshold 1 --dbo-prefill-token-threshold 1`
  reaches upstream vLLM microbatch validation with
  `use_ubatching=True num_ubatches=2`, then stops because upstream DBO only
  allows `deepep_low_latency` or `deepep_high_throughput` all2all backends; the
  current Ascend Qwen3-MoE run uses `flashinfer_all2allv`.
- The platform keeps the default Ascend `flashinfer_all2allv` backend for
  normal runs, but preserves an explicitly selected DeepEP backend when DBO is
  enabled so the DeepEP runtime boundary can be tested directly.
- DBO performance numbers require additional on/off benchmark runs.
- Multi-node HCCL/MC2/Fused MC2 overlap is not claimed in this submission
  because the available compute resources only covered single-node two-card
  validation.

## What This Submission Claims

This submission claims a complete Ascend DBO code path and single-node
validation evidence:

- DBO parameters are preserved and validated through Ascend platform config.
- Decode and prefill thresholds feed the same upstream DBO decision helper used
  by vLLM.
- DP coordination is called before creating ubatch slices, preventing
  rank-local threshold decisions from changing collective order.
- Ubatch metadata is carried through Ascend forward context and consumed by an
  eager NPU ubatch wrapper.
- The first MoE communication boundary has DBO stream handoff hooks, while
  unverified MoE communication variants remain guarded.
- Qwen3-30B-A3B W8A8 TP=2 + EP reaches API server startup with custom ops
  registered.
- The DBO CLI path reaches vLLM's upstream DeepEP backend gate, confirming that
  the Ascend platform no longer disables the DBO parameters before validation.
- Explicit `--all2all-backend deepep_high_throughput` /
  `deepep_low_latency` is preserved for DBO experiments instead of being
  overwritten by the Ascend default backend.

This submission does not claim a final performance result, multi-node
communication overlap, ACLGraph + DBO capture support, or readiness as a single
upstream PR.

## Reproduction Commands

Use one Python and one CANN environment throughout the run:

```bash
export PY=/usr/local/python3.11.14/bin/python3.11
export PIP="$PY -m pip"
export TORCHRUN=/usr/local/python3.11.14/bin/torchrun
export VLLM_PLUGINS=ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0
source /usr/local/Ascend/cann-9.0.0/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/cann-9.0.0/aarch64-linux/bin/setenv.bash 2>/dev/null || true
```

Run the basic hardware checks:

```bash
$PY - <<'PY'
import ctypes
for lib in ["libgraph.so", "libgraph_base.so", "libhcomm.so",
            "libhccl.so", "libascendcl.so"]:
    ctypes.CDLL(lib)
    print(lib, "OK")
PY

$PY - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu", torch.npu.is_available(), torch.npu.device_count())
torch.npu.set_device(0)
x = torch.ones((4,), device="npu", dtype=torch.float16)
print("ones", x.cpu())
PY
```

Build custom ops with the same Python used at runtime:

```bash
cd /data/vllm-ascend
mkdir -p /tmp/vllm-build-bin
ln -sf "$PY" /tmp/vllm-build-bin/python3
export PATH="/tmp/vllm-build-bin:$PATH"
export SOC_VERSION=ascend910b3
export MAX_JOBS=2

find /data/vllm-ascend /data/vllm \( -name '._*' -o -name '.DS_Store' \) \
  -type f -delete

$PIP install -e . --no-build-isolation --no-deps \
  2>&1 | tee /data/vllm_ascend_full_build_$(date +%Y%m%d_%H%M%S).log
```

Validate custom-op registration:

```bash
$PY - <<'PY'
from vllm_ascend.utils import enable_custom_op
import torch

print("enable_custom_op =", enable_custom_op())
ops = sorted(x for x in torch._C._dispatch_get_all_op_names()
             if x.startswith("_C_ascend::"))
print("custom op count =", len(ops))
print([x for x in ops if "moe" in x.lower()][:50])
PY
```

Run the conservative Qwen3-MoE W8A8 smoke:

```bash
export MODEL_DIR=/data/modelscope/models/vllm-ascend--Qwen3-30B-A3B-Instruct-2507-quantized.w8a8/snapshots/master
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200

VLLM_PLUGINS=ascend $PY - <<'PY'
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.environ["MODEL_DIR"],
    trust_remote_code=True,
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    quantization="compressed-tensors",
    max_model_len=256,
    max_num_batched_tokens=256,
    max_num_seqs=1,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)

outputs = llm.generate(
    ["用一句话介绍 vLLM。"],
    SamplingParams(temperature=0.0, max_tokens=16),
)
print(outputs[0].outputs[0].text)
PY
```

Run the DBO + DeepEP backend gate smoke:

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --enable-dbo \
  --dbo-decode-token-threshold 1 \
  --dbo-prefill-token-threshold 1 \
  --all2all-backend deepep_high_throughput \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

On the current Ascend environment this command is expected to test whether
DeepEP runtime kernels are available. A failure after the backend is preserved
should be recorded as DeepEP / equivalent Ascend all2all support work, not as
DBO parameter propagation failure.

## Engineering Findings

- CANN 8.5.0 and 9.0.0 mixed library paths cause ABI failures; the environment
  must be cleaned before any vLLM run is trusted.
- Installing only the 910B ops package is insufficient; toolkit, NNAL, and ops
  must be aligned.
- A visible NPU is not enough; `torch.ones(..., device="npu")` is the minimum
  runtime proof that OPP kernels are active.
- vLLM-Ascend custom-op builds can silently use `/bin/python3`; a `python3`
  shim is needed if runtime packages are installed under `/usr/local/python`.
- macOS `._*` AppleDouble files can poison C++ builds after local packaging.
- Qwen3 FP8 checkpoints currently hit Ascend FP8 dynamic-quant limitations in
  this environment; Qwen3 W8A8 compressed-tensors is the recommended smoke
  path.
- Missing `_C_ascend` custom operators should be treated as a custom-op build
  issue, not as a DBO scheduling failure.

## Submission Position

This branch should be presented as the complete Ascend DBO implementation
branch for the competition submission. For upstream community work, it should
be split into the small PR sequence described in
`Design_Documents/dbo_ascend.md` and backed by the environment notes in
`dbo_ascend_env_notes.md`.
