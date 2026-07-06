# DBO Ascend Environment and Submission Notes

## 2026-07-06: CCF Submission Scope

This branch is a CCF Open Source Innovation Contest submission branch for the
vLLM-Ascend DBO parameters:

- `--enable-dbo`
- `--dbo-decode-token-threshold`

The code in this branch should be treated as an experimental PoC / RFC rather
than a merge-ready production change. It records an initial Ascend DBO bring-up
path and the engineering issues discovered while aligning vLLM-Ascend with the
current vLLM main branch.

### Branch and Upstream State

- Submission branch: `ccf/dbo-submission-20260706`
- Original DBO PoC branch: `feature/dbo-ascend`
- DBO PoC commit: `b79df9afe57b2849bb31a72669578ca2583beaea`
- DBO PoC commit title: `Enable initial DBO path on Ascend`
- Upstream vLLM-Ascend main observed on 2026-07-06:
  `56a1f19afe3c4d10b2e636477e7e45554f09c8e6`

The DBO PoC commit was intentionally not rebased onto the latest upstream main
for this contest archive. Upstream moved substantially during the bring-up
period, and rebasing the PoC directly mixes DBO scheduling work with unrelated
main-branch compatibility fixes. Those compatibility issues should be split
into separate follow-up work before any formal upstream merge request.

### Implemented PoC Behavior

The PoC focuses on the first scheduling and metadata path needed by DBO:

- Preserve `parallel_config.enable_dbo` on Ascend instead of forcing it off.
- Allocate two ubatch workspaces when `enable_dbo` is set.
- Create ubatch slices through vLLM's `maybe_create_ubatch_slices` helper.
- Route DP metadata through the DBO-aware coordination path when DP and
  ubatching are both active.
- Add focused debug logging around DBO decision points and ubatch slices.
- Add unit coverage for config preservation and the DP coordination call path.

This is not a complete Ascend DBO implementation. In particular, it does not
claim proven communication-compute overlap for MoE workloads.

### Verified Locally

Single-card Ascend checks reached the following points:

- `torch_npu` reports `torch.npu.is_available() == True`.
- `torch.npu.device_count()` reports one visible NPU.
- A small `torch.float16` tensor can be allocated on NPU and copied back to CPU.
- Qwen3-0.6B startup progressed through vLLM config creation, Ascend platform
  plugin loading, model architecture resolution, and EngineCore startup.

The available hardware was not sufficient to validate DBO's real target path:
multi-card / multi-node DP+EP MoE inference with HCCL or MC2 communication.

### Not Yet Verified

The following items remain required before the code can be considered for a
merge-ready upstream PR:

- Real DP/EP behavior with at least two visible Ascend NPUs.
- HCCL `all_to_all_single`, MC2, or Fused MC2 collective ordering under DBO.
- MoE dispatch/combine overlap with two ubatches.
- Decode threshold performance behavior for `--dbo-decode-token-threshold`.
- Prefill/decode mixed-batch behavior under chunked prefill.
- ACLGraph / CUDAGraph-mode interaction with DBO on Ascend.
- Accuracy and performance comparison against DBO disabled.

### Main-Branch Compatibility Observations

While testing official `upstream/main` against the current local vLLM main, the
following compatibility gaps were observed:

- `patch_glm47_tool_call_parser.py` assumes the vLLM private method
  `Glm47MoeModelToolParser._extract_tool_call_regions` still exists, but the
  current vLLM main no longer exposes it.
- `NPUModelRunner` in official `upstream/main` uses `self.pin_memory` before
  initializing it, while the current vLLM GPU runner initializes this field with
  `is_pin_memory_available()`.
- Current vLLM MoE code has moved toward a factory / runner structure, so older
  Ascend fused-MoE extension points that assume `FusedMoE` is a class require
  separate compatibility work.

Temporary local smoke-unblock patches for these issues are intentionally not
included in the DBO PoC commit. They should be reviewed and submitted as
separate main-compatibility changes if needed.

This note records the initial single-card Ascend 910B environment used for
DBO bring-up on the `feature/dbo-ascend` branch. It intentionally omits SSH
endpoints and instance credentials because the fork may be public.

## 2026-06-25: Gitee AI Ascend 910B Single-Card Instance

### Purpose

Use the first available Ascend 910B card to validate the vLLM-Ascend runtime
environment before multi-node resources become available.

This environment is useful for:

- CANN / driver / `torch_npu` sanity checks.
- Single-card vLLM-Ascend setup.
- Reproducing current DBO parameter compatibility behavior.
- Preparing DBO parameter-path and metadata-splitting work.

This environment is not sufficient for:

- Proving cross-node HCCL collective ordering.
- Validating MoE all-to-all behavior across physical nodes.
- Demonstrating DBO communication-compute overlap under multi-node latency.

### Resource Summary

- Platform: Gitee AI compute marketplace.
- Chip vendor filter: Ascend 910B.
- Observed marketplace node family: `shangtang-ascend910b-node-*`.
- Visible card count inside the container: 1.
- Visible NPU logical id: 8.
- NPU model: `910B2C`.
- HBM capacity: 64 GB.
- `npu-smi` version: `26.0.rc1`.

### `npu-smi info`

```text
npu-smi 26.0.rc1                            Version: 26.0.rc1

NPU   Name    Health  Power(W)  Temp(C)  Hugepages-Usage(page)
8     910B2C  OK      85.6      34       0 / 0

Chip          Bus-Id       AICore(%)  Memory-Usage(MB)  HBM-Usage(MB)
0             0000:65:00.0 0          0 / 0             3396 / 65536

No running processes found in NPU 8
```

### Ascend Installation Layout

```text
/usr/local/Ascend
├── ascend-toolkit
├── driver
└── nnal
```

The command below produced no visible output in this container:

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.info 2>/dev/null || true
```

### Python / PyTorch / torch_npu

```text
Python: 3.11.13
Compiler: GCC 12.3.1 (openEuler 12.3.1-98.oe2403sp2)
torch: 2.7.1+cpu
torch_npu: 2.7.1
torch.npu.is_available(): True
torch.npu.device_count(): 1
```

The `torch` package reports `2.7.1+cpu`, but `torch_npu` is installed and
`torch.npu.is_available()` returns `True`, so the NPU extension is active.

### Missing / Minimal Container Tools

The container image is minimal. For example:

```text
hostname: command not found
```

Future environment probes should prefer fallback commands such as:

```bash
cat /etc/hostname 2>/dev/null || true
cat /etc/os-release 2>/dev/null || true
ip -br addr 2>/dev/null || true
python3 - <<'PY'
import socket
print(socket.gethostname())
PY
```

### Next Single-Card Checks

1. Confirm exact CANN toolkit and driver versions.
2. Confirm `torch_npu` can allocate tensors on `npu:0`.
3. Run a minimal NPU tensor sanity test.
4. Clone or mount the `feature/dbo-ascend` branch.
5. Reproduce the current `--enable-dbo` reset behavior on vLLM-Ascend.
6. Prepare DBO parameter-path changes without claiming multi-node support.

## 2026-07-01: Updated CANN 9.0.0 Single-Card Smoke

An updated Ascend 910B single-card instance was used to validate that the
software stack could move beyond Python import and platform registration.

### Runtime Summary

- NPU model: `910B2C`.
- Visible NPU count: 1.
- CANN: 9.0.0.
- `npu-smi` version: `26.0.rc1`.
- Python: 3.11.15.
- `torch`: 2.10.0+cpu.
- `torch_npu`: 2.10.0.
- vLLM source tree: `/data/vllm`.
- vLLM version observed in logs: `0.23.1rc1.dev444+gd490b9816`.
- vLLM-Ascend source tree: `/data/vllm-ascend`.
- vLLM-Ascend package version observed in logs:
  `0.1.dev3779+gb79df9afe`.
- Test model: Qwen3-0.6B from ModelScope.

### NPU Tensor Smoke

```text
npu available: True
npu count: 1
tensor([1., 1., 1., 1.], dtype=torch.float16)
```

### vLLM-Ascend Startup Smoke

With `VLLM_PLUGINS=ascend`, the platform plugin loaded successfully and vLLM
resolved the Qwen3-0.6B architecture:

```text
Platform plugin ascend is activated
Resolved architecture: Qwen3ForCausalLM
Using max model len 512
device_config=npu
Initializing a V1 LLM engine
```

The run did not complete generation in this environment. It exposed the
main-branch compatibility issues listed above before a full model execution
smoke could be completed. This is still useful for the DBO submission because it
shows that the failure point moved past argument parsing, plugin registration,
model config resolution, and basic NPU availability.

### Future Two-Node Checks

If another Ascend 910B instance becomes available on a different physical node,
run these checks before any vLLM benchmark:

1. TCP reachability between containers.
2. `torch.distributed` initialization with `backend="hccl"`.
3. HCCL `all_reduce` smoke test.
4. HCCL `all_to_all_single` smoke test.
5. MoE dispatch/combine sanity test.

Only after these pass should the environment be treated as useful for DBO
cross-node validation.
