# vLLM-Ascend DBO 适配任务 SKILL

## 任务目标

本任务使用 AI 编程助手辅助完成 vLLM-Ascend 上 DBO（Dual Batch Overlap）
功能链路的适配、验证和文档整理。目标不是一次性提交不可审阅的大型补丁，而是把
DBO 在 Ascend 上需要经过的关键链路拆解清楚，并形成可复现、可继续推进的代码与
验证材料。

本次适配重点：

- 保留 Ascend 平台上的 `--enable-dbo` 配置。
- 对齐 vLLM 上游的 decode/prefill threshold 语义。
- 在 DP ranks 之间协调是否进入 ubatch，避免 collective 顺序不一致。
- 构造并传递 Ascend ubatch metadata。
- 使用 NPU ubatch wrapper 执行 microbatch。
- 推进 Qwen3-MoE W8A8、DP=2、EP 场景的功能验证。
- 记录 custom ops、CANN、torch-npu、HCCL 和 profiler 相关工程问题。

## AI 辅助开发方式

AI 工具主要用于以下工作：

1. 代码阅读和上游语义对齐
   - 阅读 vLLM DBO、MoE、DP coordination、ubatch slicing 的现有实现。
   - 对比 vLLM-Ascend 中 platform、worker、MoE communication、quantization
     模块的现状。
   - 识别上游 API 漂移，例如 `MoERunner` import path、DP device helper、
     spec decode 和模型 patch 入口变化。

2. 小步实现和风险隔离
   - 按“参数保留、threshold、DP coordination、metadata、NPU wrapper、MoE
     handoff”的顺序推进。
   - 对暂时无法稳定验证的 MoE stream handoff 增加环境开关
     `VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF`，用于隔离调度/切片链路和通信 overlap
     链路。
   - 对 DBO 决策路径增加 `VLLM_ASCEND_DBO_TRACE` trace 日志，便于复现和审查。

3. 服务器验证和问题定位
   - 辅助整理 CANN 9.0.0、torch-npu 2.10.0、HCCL、custom ops 编译和注册检查。
   - 定位 Qwen3-MoE W8A8 启动中的量化、custom op 和 vLLM API 兼容问题。
   - 生成 baseline、DBO prefill、decode boundary 的请求 payload、启动命令、
     profiler 导出命令和证据归档脚本。

4. 文档与交付整理
   - 形成设计说明、提交说明、部署验证手册、工程问题记录和 profiler 观测图。
   - 明确哪些内容已经验证，哪些内容仍属于后续边界，避免夸大性能结论。

## 已验证路径

最终验证环境：

- Ascend 910B3，两张本地卡。
- CANN 9.0.0。
- torch-npu 2.10.0。
- vLLM 0.23.0。
- Qwen3-30B-A3B W8A8。
- `data_parallel_size=2`。
- `enable_expert_parallel=True`。
- `quantization=compressed-tensors`。

已验证内容：

- Ascend platform plugin 可加载。
- NPU tensor smoke 通过。
- HCCL `all_reduce` 和 `all_to_all_single` 通过。
- vLLM-Ascend custom ops 编译并注册成功，`enable_custom_op=True`。
- Qwen3-MoE W8A8 DP=2 + EP 服务可启动并完成请求。
- DBO prefill 场景触发 `should_ubatch=True`。
- 两个 DP rank 均切成两个 ubatch。
- 日志出现 `NPUUBatchWrapper running 2 ubatches`。
- OpenAI-compatible completion 请求返回 HTTP 200。

## 验证边界

当前提交不声明最终端到端性能收益，也不声明已经证明通信/计算 overlap。

原因：

- 本次最终可用资源是本地 2 卡环境，不是稳定多节点 DP/EP MoE 集群。
- MoE stream handoff 涉及 collective 顺序和死锁风险，需要更长时间和更多 profiler
  资源验证。
- 当前 prefill DBO 主验证使用 `VLLM_ASCEND_DISABLE_DBO_MOE_HANDOFF=1` 隔离尚需继续
  稳定的通信 handoff 层。
- decode threshold path 已完成安全边界验证，但当前 vLLM 调度每个 DP rank 每步只有
  1 个 decode token，因此不会满足 `num_tokens >= num_ubatches` 的切分条件。

后续如果验收方向被认可，将继续补齐：

- 多卡/多节点 DP+EP MoE 通信验证。
- MoE dispatch/combine stream handoff 稳定性。
- HCCL/MC2/Fused MC2 顺序与 timeline overlap。
- baseline、prefill DBO、decode DBO 的多轮性能对比。
- 面向社区的小 PR 拆分与正式合入。

## 复现入口

交付材料中包含：

- `README.md`：最终交付包说明。
- `DBO_TECHNICAL_EXPLANATION.md`：DBO 技术理解说明。
- `DEPLOYMENT_AND_VALIDATION.md`：部署与验证手册。
- `GITLINK_AND_GITHUB_LINKS.md`：GitLink/GitHub 分支与提交链接。
- `track1_2026vLLMAscend_DBO_Evidence_Profiler.tar.gz`：服务器验证证据和 raw profiler。

