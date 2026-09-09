# TileLang-TPU 双编程模型测试报告

本文只依据当前干净基线认定支持范围。早期实验仅用于解释设计选择，不计入当前结果。

## 1. 测试口径

当前基线为：

- Git 提交：`e5774525e3a6e11d0d6010e979203c55181a8872`；
- 源码快照标识：`c49cf3594f2ce5d215e54d331a10ca0bbbf92bd6667fba3e172b2b3621f7f5c4`；
- 本地证据目录：`research/artifacts/2026-09-09/final-e5774525/`（已被 Git 忽略，不随仓库提交）。

每个用例都在独立进程中重新编译、加载并执行一次，以精确值或 PyTorch 为参考结果。core、FP8
和 demo 同时验收数值与原始记录（raw trace）；PCIe 还可要求解码后的指令时间（decoded timing）。
完整 TPU-Kernel 矩阵只扩大数值覆盖，不开启 recorder。本机没有能够从 CModel raw 中提取
duration 的解码器，因此 `timed_instruction_count=0` 是明确的证据边界。

可执行组合为：

| 运行环境 | 编程目标 |
| --- | --- |
| CModel | BM1690/TPU-Kernel、SG2260E/TPU-Kernel、SG2260E/RV |
| PCIe | SG2260E/TPU-Kernel、SG2260E/RV |

## 2. 当前结果

正式验收集合由 8 份 CModel 汇总文件（summary）、2 份完整 PCIe summary、14 份 TPU-Kernel PCIe
分片和 15 份 demo PCIe 分片组成。39 份 summary 均记录同一 Git 提交和源码快照标识，且
`implementation_worktree_dirty=false`、`complete=true`，失败和取消为 0。

| 环境 | core | FP8 | TPU-Kernel 全量 | demo | 实际执行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM1690 CModel | 28/28 | 42/42 | 152/152 | 36/36 | 258/258 |
| SG2260E CModel | 56/56 | 42/42 | 146/146 | 51/51 | 295/295 |
| SG2260E PCIe | 56/56 | 42/42 | 146/146 | 51/51 | 295/295 |
| **合计** |  |  |  |  | **848/848** |

848 是矩阵运行器（runner）的执行次数，不是去重后的能力数。SG2260E core 在 TPU-Kernel 和 RV
上各执行 28 项；demo 在 TPU-Kernel 上执行 36 项，在 RV 上执行 FP16/BF16/FP32 的逐元素四则
运算和 matmul，共 15 项。
RMSNorm、Split-K RMSNorm、RoPE、SwiGLU 和 FlashAttention 当前只适用 TPU-Kernel，不能由
TPU-Kernel 的结果外推为 RV 支持。

FP8 矩阵仅覆盖 TPU-Kernel，不得据此外推 RV FP8 能力。

SG2260E PCIe 的 core 56、FP8 42、demo 51 共 149 个严格 profiling 用例，均有非空 raw，
并由 `bigTpuProfile 0.3.5` 解码 3568 条合法 ns 事件。其中 RV 的 core 28 项产生 412 条，
demo 15 项产生 168 条，合计 580 条。一次 recorder 发射只用于核对映射和定位问题，不用于
TPU-Kernel/RV 性能排名。

## 3. 顺序晋级与故障边界

CModel 必须使用同一份干净源码，依次执行 BM1690 和 SG2260E；正式 SG2260E 结果只取满足此顺序
的四份 `*-retry1`。PCIe 只调度两份 CModel summary 共同覆盖、且 selector（唯一标识一项能力的
精确参数组合）完全相同的用例。失败或取消的 summary、
通过前缀和恢复 canary 均不能用于晋级。

PCIe runner 对编译、加载、总截止时间、数值、raw、解码和板卡健康实行首错停止。每个子进程
处于带父进程退出约束的私有进程组。进程未正常回收时，runner 会在限定时间内依次发送 TERM
和 KILL，并完成进程回收与管道清空。设备锁覆盖运行前检查、任务发射和运行后检查；每次发射后，
必须连续两次间隔采样均为 `Active/0%`，才能继续使用板卡。

本轮完整 TPU-Kernel 整批、其后的两个分片及 demo 整批，曾分别在一次运行后检查中读到主要管理
遥测字段同时为 `F` 的 `Fault`。四次任务都立即停止并保存原始探测数据；即使当前用例已
通过数值校验，也不计为板端通过。确认受控进程退出、板卡连续恢复为 `Active/0%`，并通过精确
canary（小规模恢复验证）后，正式回归改用更小的严格串行分片。TPU-Kernel 的 14 个完整分片恰好
覆盖 146 项，demo 的 15 个完整分片恰好覆盖 51 项，没有重复或遗漏。

主要管理遥测字段同时为 `F`，说明当时相关遥测不可用，但不足以确定故障发生在驱动、固件还是
`tpu-smi`。分片通过也不能证明长期连续负载稳定，因此首个 `Fault` 即停止的策略仍然保留。正式
集合的 295 次运行后检查全部通过；最高瞬时利用率为 10%，最长稳定等待为 1.904881 秒。验证结束
时设备为 `Active/0%`，没有受控子进程、运行会话或持久隔离标记残留。

## 4. 保留的历史结论

早期 PCIe matmul 曾暴露真实依赖错误：`T.Pipelined(num_stages=1)` 把 A/B DMA 与直接消费它们的
GEMM 放入同一并行区，CModel 数值通过而板端最大绝对误差达到 30.15625。改用依赖有序的
`T.serial` 后，CModel 与 PCIe 重新通过。这说明 CModel 是必要前置条件，但不能替代涉及真实
并发时序的板端验证；在建立 TPU token/buffer-version/hazard contract 前，software pipeline
不属于已支持能力。

2026-09-04/07 的 core、FP8、TPU-Kernel 和 profiling 工件仍保存在已被 Git 忽略的本地证据目录
中，用于复盘上述实现演进。它们早于当前 region ABI、循环 liveness 和设备监管修改，只能作为
历史证据，不能作为 `e5774525` 板端执行的前置验证依据。

## 5. 复现原则

1. 先以同一干净实现完成 BM1690 CModel，再完成 SG2260E CModel；任一失败均禁止上板。
2. PCIe 必须显式给出设备 ID、加载确认和精确用例子集或全量确认；profiling 另需独立确认。
3. 要求逐指令时间时，首次向硬件发射任务前，先完成无需访问硬件的 decoder 运行前检查。
4. 数值/raw 验收要求 `complete=true`、所有 case `passed`，且 recorder 文件规范命名并非空；严格
   timing 还要求 `parser_status=ready`、至少一条 timing、全部 begin/end/duration 为有限数值、
   `unit="ns"`、`duration>=0`、`end>=begin`。
5. 首个异常即停止剩余 case；仍存活的受控进程按有界流程清理。只有进程组清理或板卡健康无法
   证明时才持久隔离；恢复后先做只读健康检查和精确 canary，再使用新目录。
6. 工件写入已被 Git 忽略的 `research/artifacts/<date>/<run>/`，不得覆盖失败记录或在 `/tmp` 留下
   中间文件。

本报告证明的是清单内 shape、dtype、variant 和 target 的功能正确性，不证明任意 tail、动态
shape、广义 broadcast/reshape、异步并发、跨核调度或性能最优。每个 selector 的授权范围见
[机器能力契约](../tpu-op-contract/README.md)，完整设计见
[TPU 后端报告](../tpu-backend-design/README.md)。
