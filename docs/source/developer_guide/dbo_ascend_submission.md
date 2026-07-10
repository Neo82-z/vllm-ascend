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
- With `--enable-dbo`, threshold values set to `1`, and
  `--all2all-backend deepep_high_throughput`, the same Qwen3 W8A8 TP=2 + EP
  server also starts successfully. The model is wrapped with
  `NPUUBatchWrapper`, cascade attention is disabled by vLLM because DBO is
  enabled, and the API server reaches application startup.
- A stricter Qwen3 W8A8 `data_parallel_size=2` + EP + DBO run with
  `deepep_high_throughput` reaches DP coordinator startup, launches two API
  servers, initializes HCCL with `world_size=2`, assigns DP ranks 0/1 and EP
  ranks 0/1, and maps the 128 experts across the two EP ranks. It then stops
  during MoE layer construction because upstream vLLM references
  `DeepEPHTPrepareAndFinalize` on the DeepEP high-throughput path even though
  that symbol is only imported under the CUDA-like platform guard.
- After switching back to Ascend-native `flashinfer_all2allv` and applying the
  narrow microbatch validation bypass, Qwen3 W8A8 DP=2 + EP + DBO reaches API
  server startup. The run starts the DP coordinator and two API servers,
  initializes HCCL `world_size=2`, assigns DP/EP ranks 0/1, maps
  experts across both EP ranks, loads the 29.07 GiB checkpoint on both workers,
  wraps both workers with `NPUUBatchWrapper`, creates KV cache, completes engine
  warmup, receives all DP coordinator subscriptions, and reports
  `Application startup complete` on both API servers.
- A prefill microbatch request has been validated on the same DP=2 + EP
  Qwen3-MoE W8A8 setup with `VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF=1` to isolate
  the remaining stream-handoff layer. With decode threshold disabled
  (`65536`) and prefill threshold set to `128`, two 160-token prompts trigger
  `should_ubatch=True` on both DP ranks. The logs show the batch split into
  `slice(0, 80)` and `slice(80, 160/161)` ubatches and
  `NPUUBatchWrapper running 2 ubatches` on both workers. The HTTP completion
  request returns successfully with `prompt_tokens=321`, `completion_tokens=2`,
  and `system_fingerprint=vllm-0.23.0-dp2-ep-37aedc47`.
- Torch-NPU profiler raw data is produced under
  `/data/vllm_profile_dbo_no_handoff_20260710_090442`, including
  per-worker `*_ascend_pt` directories and API-server `*.pt.trace.json.gz`
  traces. Derived timeline files require offline
  `torch_npu.profiler.profiler.analyse()` and should be interpreted as
  profiler evidence, not as a stable latency benchmark.
- A decode-only DBO configuration was also profiled with
  `--dbo-decode-token-threshold 1` and
  `--dbo-prefill-token-threshold 65536`. The short two-prompt request returns
  HTTP 200 and produces profiler output, but it does not split into ubatches:
  each DP rank has only one decode token in the step, and the Ascend DBO path
  intentionally keeps the upstream safety guard `num_tokens >= num_ubatches`.
  This run is therefore recorded as decode-threshold-path and safety-boundary
  evidence, not as a decode-overlap result.

Current hardware-dependent boundary:

- DBO on/off behavioral comparison and decode-only threshold testing should be
  run on top of the successful DP=2 + EP native all2all server.
- `--enable-dbo --dbo-decode-token-threshold 1 --dbo-prefill-token-threshold 1`
  reaches upstream vLLM microbatch validation with
  `use_ubatching=True num_ubatches=2`, then stops because upstream DBO only
  allows `deepep_low_latency` or `deepep_high_throughput` all2all backends; the
  current Ascend Qwen3-MoE run uses `flashinfer_all2allv`.
- The platform keeps the default Ascend `flashinfer_all2allv` backend for
  normal runs, but preserves an explicitly selected DeepEP backend when DBO is
  enabled so the DeepEP runtime boundary can be tested directly.
- The successful DeepEP high-throughput startup was run with `tensor_parallel_size=2`
  and `data_parallel_size=1`. It validates backend preservation, DBO config
  activation, model wrapping, and startup, but not yet DP=2 rank coordination or
  true DeepEP all-to-all traffic.
- The `data_parallel_size=2` run validates that rank coordination can reach
  distributed worker initialization on the available two-card 910B node. The
  next blocker is an upstream DeepEP high-throughput prepare/finalize import
  boundary for non-CUDA-like platforms, not an HCCL startup failure or a DBO
  threshold/config failure.
- The follow-up implementation therefore adds an Ascend-native all2all DBO
  gate bypass: it lets `flashinfer_all2allv` / `allgather_reducescatter` pass
  the upstream vLLM 0.23 microbatch assertion without changing the real MoE
  communication backend to DeepEP.
- The DP=2 native all2all startup run verifies that this bypass is limited to
  vLLM config validation: runtime DBO is still active, shown by
  `NPUUBatchWrapper` being attached to both DP workers.
- The no-handoff prefill run verifies the scheduling, DP coordination, ubatch
  slicing, metadata propagation, and NPU ubatch wrapper layers. With handoff
  enabled, the system reaches runtime but can hang in worker RPC, so the final
  overlap claim still depends on fixing/validating MoE stream handoff.
- DBO performance numbers require additional on/off benchmark runs and profiler
  timeline analysis.
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
- DeepEP high-throughput startup with DBO enabled reaches API server startup in
  the TP=2, DP=1 configuration.
- DP=2 + EP + DBO reaches HCCL worker initialization and expert placement, then
  exposes the upstream DeepEP HT prepare/finalize import boundary on Ascend.
- The branch contains a narrow `ParallelConfig.use_ubatching` property patch
  that keeps Ascend-native all2all backends available during vLLM 0.23 config
  validation while restoring runtime DBO behavior afterward.
- DP=2 + EP + DBO with Ascend-native `flashinfer_all2allv` now reaches API
  server startup with model weights loaded, KV cache created, DP coordinator
  subscriptions complete, and `NPUUBatchWrapper` attached on both workers.
- DP=2 + EP + DBO prefill microbatching is functionally validated in
  no-handoff mode: both ranks trigger `should_ubatch=True`, create two ubatch
  slices, execute `NPUUBatchWrapper`, and return a successful OpenAI-compatible
  completion response.

This submission does not claim a final performance result, multi-node
communication overlap, ACLGraph + DBO capture support, or readiness as a single
upstream PR.

## Report Narrative

The final competition report should follow the same engineering style as the
historical community DBO PR
[`vllm-ascend#4894`](https://github.com/vllm-project/vllm-ascend/pull/4894),
but with claims scaled to the hardware that was actually available.

Recommended report structure:

1. **Motivation and scope**: Explain that DBO is valuable for MoE serving
   because it overlaps microbatch compute and communication, but that Ascend
   needs platform-specific work in scheduling, metadata, NPU streams, MoE
   communication, custom ops, and version compatibility.
2. **Major changes**: Present the implementation by layer: platform config,
   decode/prefill thresholds, DP coordination, ubatch metadata, eager NPU
   wrapper, MoE handoff, and custom-op diagnostics.
3. **Validation matrix**: Report each verified layer separately instead of
   claiming one opaque end-to-end number. Include CANN/torch_npu, HCCL
   all-reduce/all-to-all, custom-op registration, Qwen3 W8A8 TP=2 + EP startup,
   DBO + DeepEP high-throughput startup, the DP=2 + EP DeepEP boundary run, the
   final DP=2 + EP + DBO native `flashinfer_all2allv` startup, and the DP2/EP
   prefill microbatch completion request.
4. **Resource boundary**: State that the available hardware is single-node
   two-card 910B, not the TP=8 / multi-node environment used by larger
   community experiments. Therefore this submission validates the code path and
   startup behavior, while leaving DP=2/DP>2 communication overlap and
   throughput curves as follow-up work.
5. **Evidence over claims**: Use concrete log facts: `enable_custom_op=True`,
   65 `_C_ascend::` operators registered, `NPUUBatchWrapper` enabled, Qwen3
   W8A8 API server startup complete, DBO thresholds preserved, and
   `should_ubatch=True` / `NPUUBatchWrapper running 2 ubatches` for the
   prefill microbatch request.
6. **Upstream plan**: End with the small-PR sequence from the design document
   instead of asking reviewers to accept a large monolithic patch.

The key difference from a TP=8 report is the conclusion: this work should be
presented as a complete DBO implementation and single-node validation package,
not as a final multi-node performance benchmark.

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
the DeepEP backend name can be preserved through vLLM-Ascend platform config.
In the TP=2, DP=1 configuration used during this work, the command reaches API
server startup.

Run the stricter DP=2 boundary smoke:

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --enable-dbo \
  --dbo-decode-token-threshold 1 \
  --dbo-prefill-token-threshold 1 \
  --all2all-backend deepep_high_throughput \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager
```

This run reaches HCCL `world_size=2`, DP ranks 0/1, EP ranks 0/1, and expert
placement across the two ranks. It stops before weight loading because vLLM
0.23 imports `DeepEPHTPrepareAndFinalize` only under a CUDA-like platform
guard, while the DeepEP high-throughput MoE path still references it on Ascend.
That is the current DeepEP backend boundary to solve before claiming true DBO
all-to-all overlap.

Run the Ascend-native DBO smoke after applying the gate bypass:

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --enable-dbo \
  --dbo-decode-token-threshold 1 \
  --dbo-prefill-token-threshold 1 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager
```

This intentionally omits `--all2all-backend deepep_high_throughput` so Ascend
uses its native backend. The relevant logs from the successful run are:

```text
Defaulting api_server_count to data_parallel_size (2)
world_size=2 rank=0 ... backend=hccl
world_size=2 rank=1 ... backend=hccl
rank 0 ... DP rank 0 ... TP rank 0, EP rank 0
rank 1 ... DP rank 1 ... TP rank 0, EP rank 1
[EP Rank 0/2] Expert parallelism is enabled ... Local/global number of experts: 64/128
[EP Rank 1/2] Expert parallelism is enabled ... Local/global number of experts: 64/128
[DBO_EXPERIMENTAL] Wrapped model with NPUUBatchWrapper.
[DBO_EXPERIMENTAL] Bypassing vLLM's upstream DeepEP-only microbatch validation for Ascend native all2all backend. backend=flashinfer_all2allv.
All engine subscriptions received by DP coordinator
Application startup complete.
```

This is the closest route to the historical vLLM-Ascend DBO approach: keep
upstream DBO scheduling semantics, but execute communication through Ascend's
own MoE communication stack rather than CUDA DeepEP.

The `use_ubatching=False` value printed near the platform config log in this
run is expected during `VllmConfig.__post_init__`. The patch temporarily
suppresses `ParallelConfig.use_ubatching` only while bypassing vLLM 0.23's
DeepEP-only assertion. Runtime DBO remains enabled afterward, which is why both
workers attach `NPUUBatchWrapper`.

Run the verified DBO prefill microbatch request with profiler enabled:

```bash
export VLLM_ASCEND_DBO_TRACE=1
export VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF=1
export PROF_DIR=/data/vllm_profile_dbo_prefill_$(date +%Y%m%d_%H%M%S)

vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --enable-dbo \
  --dbo-decode-token-threshold 65536 \
  --dbo-prefill-token-threshold 128 \
  --max-model-len 384 \
  --max-num-batched-tokens 384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.78 \
  --enforce-eager \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false}" \
  2>&1 | tee /data/qwen3_dp2_ep_dbo_prefill_profile_serve_$(date +%Y%m%d_%H%M%S).log
```

Then send the profiled request:

```bash
curl -i -X POST http://127.0.0.1:8000/start_profile

OUT=/data/qwen3_dp2_ep_dbo_prefill_profile_$(date +%Y%m%d_%H%M%S).json
curl -sS -w "http_code=%{http_code} wall=%{time_total} sec\n" -o "$OUT" \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/data/two_long_prompts_160_payload.json

cat "$OUT"
curl -i -X POST http://127.0.0.1:8000/stop_profile
sleep 10

grep -nEi "should_ubatch=True|split batch|NPUUBatchWrapper running|HTTP/1.1 200" \
  /data/qwen3_dp2_ep_dbo_prefill_profile_serve_*.log | tail -80
```

Run the matched no-DBO baseline by removing only the DBO flags:

```bash
unset VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF
export PROF_DIR=/data/vllm_profile_baseline_prefill_$(date +%Y%m%d_%H%M%S)

vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --max-model-len 384 \
  --max-num-batched-tokens 384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.78 \
  --enforce-eager \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false}" \
  2>&1 | tee /data/qwen3_dp2_ep_baseline_prefill_profile_serve_$(date +%Y%m%d_%H%M%S).log
```

Use the same request payload and `/start_profile` / `/stop_profile` sequence as
the DBO run. Keep the comparison conservative: profiler wall time is useful as
a sanity check, but the important baseline evidence is that the same DP2 + EP
W8A8 model runs without `should_ubatch=True`.

Run the decode-only DBO check by disabling prefill ubatching and setting the
decode threshold to one token. Upstream vLLM uses `num_tokens >= threshold`, so
`--dbo-decode-token-threshold 1` is the right way to route one-token decode
steps into the DBO decision path. In the current DP=2 test, however, each rank
receives only one decode token per step, while Ascend DBO keeps the safety
guard `num_tokens >= num_ubatches`. With `num_ubatches=2`, the expected trace is
`uniform_decode=True` and `should_ubatch=False`; this is a useful negative
control rather than a decode microbatch split:

```bash
export VLLM_ASCEND_DBO_TRACE=1
export VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF=1
export PROF_DIR=/data/vllm_profile_dbo_decode_$(date +%Y%m%d_%H%M%S)

vllm serve "$MODEL_DIR" \
  --served-model-name qwen3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --enable-dbo \
  --dbo-decode-token-threshold 1 \
  --dbo-prefill-token-threshold 65536 \
  --max-model-len 384 \
  --max-num-batched-tokens 384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.78 \
  --enforce-eager \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false}" \
  2>&1 | tee /data/qwen3_dp2_ep_dbo_decode_profile_serve_$(date +%Y%m%d_%H%M%S).log
```

Use a short prompt pair with a larger decode length:

```bash
cat > /data/two_short_decode_payload.json <<'JSON'
{
  "model": "qwen3",
  "prompt": ["用一句话介绍 vLLM。", "用一句话介绍 MoE。"],
  "max_tokens": 16,
  "temperature": 0
}
JSON

curl -i -X POST http://127.0.0.1:8000/start_profile

OUT=/data/qwen3_dp2_ep_dbo_decode_profile_$(date +%Y%m%d_%H%M%S).json
curl -sS -w "http_code=%{http_code} wall=%{time_total} sec\n" -o "$OUT" \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @/data/two_short_decode_payload.json

cat "$OUT"
curl -i -X POST http://127.0.0.1:8000/stop_profile
sleep 10

grep -nEi "uniform_decode=True|num_tokens_unpadded=1|num_ubatches=2|should_ubatch|split batch|NPUUBatchWrapper running|HTTP/1.1 200" \
  /data/qwen3_dp2_ep_dbo_decode_profile_serve_*.log | tail -120
```

Parse profiler raw output offline:

```bash
$PY - <<'PY'
from pathlib import Path
from torch_npu.profiler.profiler import analyse

for root in ["/data"]:
    for path in sorted(Path(root).glob("vllm_profile_*")):
        if path.is_dir():
            print("analyse", path)
            analyse(str(path))
PY

find /data/vllm_profile_* -type f \( \
  -name "trace_view.json" \
  -o -name "kernel_details.csv" \
  -o -name "operator_details.csv" \
  -o -name "op_statistic.csv" \
  -o -name "step_trace_time.csv" \
  -o -name "*.pt.trace.json.gz" \
\) | sort
```

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
