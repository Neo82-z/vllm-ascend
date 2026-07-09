# vLLM-Ascend DBO 环境与主线漂移记录

本文记录 CCF vLLM-Ascend DBO 交付验证期间在 Ascend 910B 环境、CANN/NNAL/ATB、vLLM main 与 vLLM-Ascend main 适配中遇到的问题。本文作为最终交付的工程复现附录，用于保留已验证事实、失败分类和后续拆分 PR 前必须处理的工程风险。

## 当前目标

当前目标是验证 Ascend 后端上 DBO 相关链路的最低可行路径：

- `--enable-dbo`
- `--dbo-decode-token-threshold`
- `--dbo-prefill-token-threshold`
- DP coordination
- ubatch slicing / metadata 传递
- MoE 场景下 dispatch/combine handoff 的最小路径

当前验证重点不是性能收益，而是先确认：

1. CANN runtime 可用；
2. torch_npu 基础算子可用；
3. 单节点双卡 HCCL collective 可用；
4. vLLM-Ascend 插件可加载；
5. MoE 模型能进入 NPU worker / model runner；
6. DBO 参数不会在 Ascend 平台层被强制禁用。

2026-07-09 后续 smoke 目标收敛为 Qwen3 及以上 MoE 模型。Qwen1.5-MoE
与 DeepSeek-V2-Lite 的记录仅作为环境和主线 API 漂移样本保留，不再作为当前
DBO PoC 的主要验证对象。

## 已确认可用项

### CANN 9.0.0 动态库

重装 CANN 9.0.0 toolkit、NNAL 和 910B ops 后，以下动态库可被加载：

```text
libgraph.so OK
libgraph_base.so OK
libhcomm.so OK
libhccl.so OK
libascendcl.so OK
```

### torch_npu 基础 smoke

环境：

```text
torch 2.10.0+cpu
torch_npu 2.10.0
npu available True
npu count 2
```

基础 NPU tensor 创建、CPU copy 和 elementwise add 已通过：

```text
ones tensor([1., 1., 1., 1.], dtype=torch.float16)
add tensor([...], dtype=torch.float16)
```

### 单节点双卡 HCCL

`torch.distributed` HCCL smoke 已通过：

```text
all_reduce:
rank=0 all_reduce=[3.0, 3.0, 3.0, 3.0]
rank=1 all_reduce=[3.0, 3.0, 3.0, 3.0]

all_to_all_single:
rank=0 input=[0.0, 1.0] output=[0.0, 10.0]
rank=1 input=[10.0, 11.0] output=[1.0, 11.0]
```

这说明当前机器至少具备单节点双卡通信能力，可以作为 DBO / MoE 通信路径的基础 smoke 环境。

### vLLM-Ascend plugin

插件可被 vLLM 加载，并识别为 Ascend platform：

```text
Platform plugin ascend is activated
platform <vllm_ascend.platform.NPUPlatform object ...>
```

日志中出现：

```text
Breakable cudagraph is force disabled on Ascend because DeepSeek V4 PIECEWISE cudagraph is not supported yet.
```

这不是当前 DBO PoC 的失败原因。它表示 Ascend 平台上部分 cudagraph/breakable cudagraph 路径被平台层主动关闭。

## CANN / NNAL / ATB 问题记录

### 多版本 CANN 串库

机器上同时存在 CANN 8.5.0 和 CANN 9.0.0 时，`LD_LIBRARY_PATH` 如果混入 CANN 8.5.0，会导致 ABI 符号不匹配：

```text
libgraph.so FAIL
/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/libgraph_base.so:
undefined symbol: _ZN2ge16GetSanitizedNameERKSs
```

处理方式：

- 清理 `LD_LIBRARY_PATH`；
- 确认 `/usr/local/Ascend/cann -> /usr/local/Ascend/cann-9.0.0`；
- 只 source CANN 9.0.0 的环境；
- 不要把 `/usr/local/Ascend` 下所有库目录无脑加入 `LD_LIBRARY_PATH`。

### CANN 9.0.0 安装不完整

仅安装 `Ascend-cann-910b-ops_9.0.0` 不够。曾出现：

```text
libgraph.so: cannot open shared object file
libgraph_base.so: cannot open shared object file
libhccl.so FAIL libhcomm.so: cannot open shared object file
```

说明 runtime / graph / communication 依赖并未完整安装或路径未生效。需要安装或升级：

```text
Ascend-cann-toolkit_9.0.0_linux-aarch64.run
Ascend-cann-nnal_9.0.0_linux-aarch64.run
Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
```

### 910B ops 未生效

当 torch_npu 能看到设备，但基础算子失败时，表现为：

```text
aclnnInplaceOne failed, error code is 561103
Parse dynamic kernel config fail
OnesLike ADD_TO_LAUNCHER_LIST_AICORE failed
```

这不是 vLLM 或 DBO 问题。它表示 CANN OPP / 910B ops 包没有正确安装或没有被当前环境变量选中。最低验收线应是：

```python
torch.ones((4,), device="npu", dtype=torch.float16)
```

只有该测试通过后，才应继续 HCCL、vLLM-Ascend 或 DBO 验证。

### ATB / NNAL 路径

DeepSeek-V2-Lite 进入 NPU worker 初始化后，出现：

```text
OSError: libatb.so: cannot open shared object file: No such file or directory
OSError: Could not load this library:
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/libop_plugin_atb.so
```

这说明 torch_npu ATB extension 需要 NNAL ATB 动态库路径。需要将实际 `libatb.so` 所在目录加入 `LD_LIBRARY_PATH`，例如：

```bash
find /usr/local/Ascend -name 'libatb.so*' -print
find /usr/local/Ascend -name 'libatb_speed.so*' -print
```

然后将对应目录加入：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.0.0/atb_speed/lib:$LD_LIBRARY_PATH
```

实际路径应以 `find` 结果为准。

## custom ops 编译问题记录

### CANN 8.5.0 不支持部分 main 分支 op 配置

CANN 8.5.0 编译 custom ops 时出现：

```text
Invalid socVersion ascend950 of op LightningIndexer
Invalid socVersion ascend950 of op SparseFlashAttention
```

这说明当前 vLLM-Ascend main 源码中包含面向 A5 / Ascend 950 的 op 配置，而 CANN 8.5.0 工具链无法识别相关 `socVersion`。

### CANN 9.0.0 头文件缺失

切换到 CANN 9.0.0 后，编译 custom ops 曾出现：

```text
fatal error: register/op_def_registry.h: No such file or directory
```

同时 CANN 9.0.0 目录中缺少旧版 op def 相关头文件：

```text
op_def_registry.h
op_def.h
op_def_factory.h
```

本次 smoke 曾临时从 CANN 8.5.0 复制 legacy register headers 到 CANN 9.0.0 include 目录以继续验证。这是本地环境 workaround，不应作为正式 PR 的方案。

### macOS AppleDouble 文件污染源码包

从 macOS 打包上传源码时，出现 `._*` AppleDouble 文件参与 C++ 编译：

```text
Mac OS X ... com.apple.provenance
error: missing terminating " character
error: 'Mac' does not name a type
```

处理方式：

```bash
find /data/vllm-ascend /data/vllm \( -name '._*' -o -name '.DS_Store' \) -type f -delete
```

后续打包应排除 macOS 元数据文件。

### 基础系统工具缺失

编译 third-party protobuf 时曾出现：

```text
/bin/sh: line 1: patch: command not found
```

需要补装 GNU `patch`。

### Python 解释器混用

CMake 配置阶段曾使用系统 `/bin/python3.11`，而运行 vLLM 使用 `/usr/local/python3.11.14/bin/python3.11`。这会导致构建时缺少 Python 包，例如：

```text
ModuleNotFoundError: No module named 'regex'
```

需要确保构建、运行、pip 安装使用同一个 Python：

```bash
export PY=/usr/local/python3.11.14/bin/python3.11
export PIP="$PY -m pip"
```

custom ops 编译阶段还会通过 `HI_PYTHON=python3` 间接调用 Python。如果 `python3`
解析到系统 Python，而运行依赖安装在 `/usr/local/python3.11.14`，会在 op 编译
脚本中继续出现缺包问题。当前 workaround 是在 `PATH` 前置一个 shim：

```bash
mkdir -p /tmp/vllm-build-bin
ln -sf "$PY" /tmp/vllm-build-bin/python3
export PATH="/tmp/vllm-build-bin:$PATH"
```

随后确认：

```bash
which python3
python3 - <<'PY'
import sys, numpy
print(sys.executable)
print(numpy.__version__, numpy.__file__)
PY
```

### custom ops 未注册的运行时表现

在 custom ops 未完整编译或未安装到 `_cann_ops_custom` 时，模型可能已经完成
plugin 加载和部分权重加载，但 MoE 路径会在路由或 grouped matmul 附近失败。
典型检查结果为：

```text
vllm_ascend._cann_ops_custom only contains .gitkeep
vllm_ascend.vllm_ascend_C missing
enable_custom_op = False
_C_ascend registered ops count = 0
```

这说明问题属于 custom-op package / torch binding 注册失败，不应归因于 DBO
threshold 或 ubatch metadata。最低验收命令：

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

如果 `enable_custom_op=False` 或 `_C_ascend::moe_gating_top_k` 等 MoE op 不存在，
Qwen3-MoE W8A8 的启动失败应记录为 custom ops 问题。

### custom ops 编译耗时

CANN custom ops 编译会进入 `opc.py` 和 `bisheng` 阶段。该阶段使用 CPU 编译
AICore kernel，不使用 NPU 执行。大算子例如 `compressor` 会包含多个 tiling
key 变体，一个 `.done` 文件只有在整组脚本结束后才会出现，因此 `.done` 数量
长时间不变不一定表示卡死。

监控命令：

```bash
date
find /data/vllm-ascend/csrc/build/binary/ascend910b/gen -name "*.done" | wc -l
find /data/vllm-ascend/csrc/build -type f -mmin -2 2>/dev/null | wc -l
ps -eo pid,etime,pcpu,cmd | grep -E "opc.py|bisheng|cmake --build|ninja" | grep -v grep
```

只要 `bisheng` 仍有接近满核 CPU 或最近两分钟有文件更新，就应继续等待。

## vLLM main 与 vLLM-Ascend main API 漂移

当前验证过程中，vLLM 与 vLLM-Ascend 均来自 main 附近版本，但仍出现多处私有 API 漂移。以下问题均不属于 DBO 逻辑本身，而是主线适配风险。

### speculative decode 私有函数变化

pytest / worker patch 加载时出现：

```text
ImportError: cannot import name '_compute_global_logsumexp'
from 'vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils'
```

这是 `patch_v2.patch_triton` 对 vLLM 内部 speculative decode 函数的依赖漂移。

### DP device id helper 变化

vLLM engine arg config 阶段出现：

```text
AttributeError:
module 'vllm.v1.engine.utils' has no attribute
'get_physical_gpu_ids_for_local_dp_rank'
```

这是 `patch_dp_device_ids` 依赖的 vLLM DP helper 移动或删除。

### MoE 类型导入路径变化

quantization import 阶段出现：

```text
ImportError: cannot import name 'MoERunner'
from 'vllm.model_executor.layers.fused_moe'
```

临时处理为从更深路径导入：

```text
vllm.model_executor.layers.fused_moe.runner.moe_runner
```

同时 `RoutedExperts` 也需要与 `MoERunner` 分开导入，避免同一行 import 因一个符号失败导致另一个符号也不可用。

### FusedMoE factory 与 class 接口变化

DeepSeek-V2-Lite smoke 已进入模型构建和 MoE layer 初始化阶段后出现：

```text
TypeError:
FusedMoE.__init__() got an unexpected keyword argument 'runner_cls'
```

原因是当前 `/data/vllm` 中 `vllm.model_executor.layers.fused_moe.layer.FusedMoE`
仍是旧式 class，其 `__init__` 不接受 `runner_cls` / `runner_args`。而
`vllm_ascend.patch.platform.patch_fused_moe` 按新版 vLLM factory 语义向
`FusedMoE` 注入 Ascend MoE runner。

该问题说明当前 DeepSeek-V2-Lite smoke 使用的 vLLM 版本与 vLLM-Ascend
MoE patch 预期不一致。由于主线正在转向新版 `FusedMoE` factory 语义，本
PoC 不再为旧式 class 接口增加兼容层，避免把验证工作转向过时组合。

### KVCacheSpecRegistry 删除

NPUModelRunner import 阶段出现：

```text
ModuleNotFoundError:
No module named 'vllm.v1.kv_cache_spec_registry'
```

当前 vLLM 已改为：

```text
vllm.v1.core.single_type_kv_cache_manager.spec_manager_map
```

本次 smoke 临时添加了兼容 shim：

```text
vllm/v1/kv_cache_spec_registry.py
```

用于将旧的 `KVCacheSpecRegistry.register/get_manager_class` 代理到 `spec_manager_map`。该 shim 仅用于 smoke，不应作为正式方案直接合入。

### 无关模型 patch 提前 import

Qwen2-MoE / DeepSeek smoke 过程中，多次被无关模型 patch 阻塞：

```text
AttributeError:
type object 'DFlashQwen3ForCausalLM' has no attribute '_read_mask_embedding'
```

这是 `patch_qwen3_dflash` 在 worker patch 初始化时被全量 import，而当前模型并不是 Qwen3 DFlash。临时处理为对无关 patch 增加 guard，或使用 minimal worker patch set。

另一个例子：

```text
AttributeError:
module 'vllm.v1.worker.gpu.spec_decode' has no attribute 'speculator'
```

同样属于 worker patch 全量 import 与 vLLM main API 漂移叠加。

### FlashLB / numba 依赖

即使 `Dynamic EPLB is False`，EPLB policy factory 仍会 import FlashLB：

```text
ModuleNotFoundError: No module named 'numba'
```

vLLM 自身 speculative decode 也依赖 `numba`：

```text
from numba import get_num_threads, jit, njit, prange, set_num_threads
```

已通过安装：

```text
numba 0.66.0
llvmlite 0.48.0
```

解决 vLLM 自身 import 问题。对于 FlashLB，临时给 `policy_flashlb` import 增加 guard，避免未启用 Dynamic EPLB 时被阻塞。

### Bailing MoE Linear Attention 路径变化

`register_ascend_customop` 阶段出现：

```text
ModuleNotFoundError:
No module named 'vllm.model_executor.layers.mamba.linear'
```

触发路径：

```text
vllm_ascend.ops.bailing_moe_linear_attn
from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import ...
```

这说明 vLLM main 中 Mamba / MiniMax linear attention 目录结构已经发生变化，而 vLLM-Ascend 的 custom op 注册仍按旧路径引用。该问题不是 DeepSeek-V2-Lite 或 DBO 本身导致，而是 vLLM-Ascend custom op 注册阶段过早 import 了模型特定模块。

正式修复方向应是：

- 将模型特定 custom op registration 延迟到实际模型需要时；
- 或在 `register_ascend_customop` 中对可选模型路径做 import guard；
- 或按当前 vLLM main 新路径更新 `bailing_moe_linear_attn` import。

## 模型验证状态

### Qwen1.5-MoE-A2.7B-Chat（停止追踪）

ModelScope config 读取成功：

```text
model_type: qwen2_moe
architectures: ['Qwen2MoeForCausalLM']
num_experts: 60
num_experts_per_tok: 4
moe_intermediate_size: 1408
```

但 engine 启动过程中被多个无关 patch / main API drift 阻塞，未能进入稳定推理。
该模型属于较旧 Qwen2-MoE 架构，不再作为当前 smoke 主线。

### DeepSeek-V2-Lite（停止追踪）

ModelScope config 读取成功：

```text
model_type: deepseek_v2
architectures: ['DeepseekV2ForCausalLM']
n_routed_experts: 64
num_experts_per_tok: 6
moe_intermediate_size: 1408
kv_lora_rank: 512
```

该模型更贴近 vLLM DBO 文档中的 DeepSeek / MoE 场景。但当前 smoke 过程中仍被以下问题阻塞：

- ATB `libatb.so` 路径；
- `bailing_moe_linear_attn` 依赖的 vLLM mamba linear 路径漂移；
- vLLM-Ascend worker/custom op 全量 import 导致的无关模型模块提前失败。

由于该路径继续推进会变成旧模型和旧 vLLM MoE 接口兼容工作，后续不再以该模型
作为 DBO PoC 的优先目标。

### Qwen3+ MoE（后续目标）

后续 smoke 优先选择 Qwen3 及以上的 MoE 模型，目标是贴近当前 vLLM /
vLLM-Ascend 主线 MoE factory 语义，避免为旧式 `FusedMoE` class 接口补
兼容层。建议先完成 dense Qwen3 基础链路，再切到 Qwen3 MoE：

1. Qwen3 dense 小模型：验证 vLLM-Ascend worker、NPU device、权重加载与基础推理；
2. Qwen3 MoE 小/中模型：验证 MoE layer 初始化和 custom op 注册；
3. 双卡 MoE：验证 HCCL / all_to_all 与 EP/DP 配置；
4. DBO 参数：验证 `--enable-dbo` 与 prefill/decode threshold 在 MoE 场景下进入调度路径。

### Qwen3-30B-A3B FP8

Qwen3-30B-A3B-Instruct-2507-FP8 config 读取成功：

```text
model_type: qwen3_moe
architectures: ['Qwen3MoeForCausalLM']
num_hidden_layers: 48
hidden_size: 2048
num_experts: 128
num_experts_per_tok: 8
moe_intermediate_size: 768
quant_method: fp8
activation_scheme: dynamic
weight_block_size: [128, 128]
```

该模型不推荐作为当前 Ascend smoke 主线。运行时会进入 FP8 dynamic quant
相关路径，并触发当前环境不支持的 `float8_e4m3fn` 动态量化能力。该失败属于
torch_npu/CANN FP8 支持边界，不是 DBO 路径问题。

### Qwen3-30B-A3B W8A8

当前推荐 smoke 模型是 vLLM-Ascend 提供的 Qwen3-30B-A3B W8A8 checkpoint：

```text
quant_method: compressed-tensors
format: int-quantized
input_activations: int8 token dynamic symmetric
weights: int8 channel static symmetric
```

运行时应使用：

```text
quantization="compressed-tensors"
tensor_parallel_size=2
enable_expert_parallel=True
max_model_len=256
max_num_batched_tokens=256
max_num_seqs=1
enforce_eager=True
```

该模型已经推进到 Qwen3-MoE / W8A8 / custom-op 相关路径。若后续仍失败，优先
检查 `_C_ascend` custom ops 是否完成注册，而不是回到旧模型兼容问题上。

## 分层验证建议

后续继续验证时，不应直接从 vLLM serve 开始，而应按以下顺序：

1. CANN 动态库：

```python
import ctypes
for lib in ["libgraph.so", "libgraph_base.so", "libhcomm.so", "libhccl.so", "libascendcl.so"]:
    ctypes.CDLL(lib)
```

2. torch_npu 基础算子：

```python
import torch, torch_npu
torch.npu.set_device(0)
x = torch.ones((4,), device="npu", dtype=torch.float16)
print(x.cpu())
```

3. HCCL collective：

```bash
torchrun --nproc_per_node=2 /data/hccl_2card_smoke.py
torchrun --nproc_per_node=2 /data/hccl_alltoall_smoke.py
```

4. vLLM-Ascend plugin：

```python
import vllm
import vllm_ascend
from vllm.platforms import current_platform
print(current_platform)
```

5. vLLM-Ascend worker import：

```python
from vllm_ascend.worker.worker import NPUWorker
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
```

6. 单卡 dense smoke；
7. 单卡 MoE smoke；
8. 双卡 HCCL + TP/EP smoke；
9. DBO 参数 smoke；
10. MoE + DBO + communication overlap 验证。

## 对 CCF DBO PoC 的结论

当前代码与验证记录说明：

- DBO 参数、threshold、DP coordination、ubatch metadata、NPU ubatch wrapper
  和 MoE communication handoff 已形成完整代码链路；
- 单节点双卡 HCCL 已验证；
- CANN 9.0.0 + torch_npu 2.10.0 在正确安装和环境变量配置后可运行基础 NPU 算子；
- Qwen3-MoE W8A8 是当前最合适的双卡 MoE smoke 路径；
- 端到端生成和性能 benchmark 的最后关键依赖是 custom ops 全量编译与注册；
- vLLM main 与 vLLM-Ascend main 存在多处私有 API 漂移，需要拆成单独 compatibility PR。

建议后续正式拆分：

1. 先固定版本矩阵：vLLM commit、vLLM-Ascend commit、CANN、torch_npu、NNAL、ATB；
2. 单独提交 main compatibility patch，解决无关 patch 全量 import 和私有 API 漂移；
3. 再提交 DBO config / threshold / DP coordination；
4. 再提交 ubatch metadata；
5. 最后提交 MoE communication handoff；
6. 性能声明必须等多卡 MoE + HCCL/MC2/Fused MC2 实测后再写。
