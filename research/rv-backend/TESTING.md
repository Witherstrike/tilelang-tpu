# TileLang-TPU 双后端测试报告

> 本文是 2026-09-04 核心五算子 profiling 竖切的历史记录。当时的 driver 只含四则与 matmul；当前 driver 默认还包含 FP16/FP32 的 local-roundtrip 与 global-to-global copy，每个合法 target 共 9 项。当前全量数值结论、FP8 边界和结构化 decoder 契约以 [TPU 算子验证报告](../tpu-backend-design/test-report.md) 和 [profiling 设计](../ppl-profiling/README.md) 为准。本文的历史 timing 不作为当前 decoder conformance 证据。

## 1. 范围与判定

本报告的数值测试以 PyTorch 结果为 oracle。每个 case 在独立进程中 fresh-compile、加载、发射一次并回传比较；同时要求成功产生 profiling raw trace。CModel 不伪造指令时间，故 `timed_instruction_count=0` 是预期结果；本轮 PCIe 工件保存了当时 decoder 产生的设备命令 duration。

本历史矩阵由 `testing/python/jit/tpu_core_ops_matrix.py` 的早期五项 registry 驱动，涵盖 `elementwise-{add,sub,mul,div}` 与 `matmul`：

| 运行时 | 组合 |
| --- | --- |
| CModel | SG2260E/TPU-Kernel、SG2260E/RV、BM1690/TPU-Kernel |
| PCIe | SG2260E/TPU-Kernel、SG2260E/RV |

PCIe 矩阵是 fail-stop：一个 case 超时、加载失败、数值不符或没有 raw trace，立即记录部分结果并停止后续测试。它需要显式 `--allow-pcie --allow-pcie-profile --device-id <n>`；不可因设备风险跳过这些门禁。当前 runner 默认把 decoder 作为最佳努力；要复现本文历史 timing 验收，必须再加 `--require-decoded-timing`，此时解析缺失、失败或非法 interval 也会立即停止。

## 2. 已完成的 CModel 结果

实验时间：2026-09-04。最终摘要保存在被 Git 忽略的
`research/artifacts/2026-09-04/cmodel-final/`。

| 组合 | 算子数 | 数值结果 | raw 命令总数/单 case | trace 文件/单 case |
| --- | ---: | --- | --- | --- |
| SG2260E / TPU-Kernel | 5 | 全部通过 | elementwise 58；matmul 78 | 24 |
| SG2260E / RV | 5 | 全部通过 | elementwise 58；matmul 78 | 24 |
| BM1690 / TPU-Kernel | 5 | 全部通过 | elementwise 94；matmul 122 | 48 |

合计 15/15 通过。SG2260E 每个 case 的 24 个 trace 文件和 BM1690 的 48 个 trace 文件反映了各自 4/8 核 CModel 拓扑及多类 engine dump；它们不是四/八核并行计算的证明。当前 host `core_num=1`，故这些测试验证的是单核工作负载在正确芯片拓扑、指令 ABI 和数值路径下可用。

当前 `596a736` 的 [canonical core summary](../artifacts/2026-09-07/core-cmodel-596a736/summary.json) 已按现行九项 registry 重跑三个合法 target，共 27/27；其中 SG2260E/RV 新增四条 copy case，local-roundtrip 覆盖 G2L→L2L→L2S，global-to-global 覆盖 S2S，FP16/FP32 均按 `(4,32)` quarter-integer 输入逐元素精确相等。该当前结果不与本节历史 15 项重复累计。

2026-09-04 同期的静态/单元回归为 `101 passed, 4 skipped`，覆盖当时的 target capability、PPL layout、模型隔离、raw ABI 混用拒绝、copy/cast/layout 的失败闭合、生成源码隔离、PCIe compile/link（不加载硬件）、profiling 超时清理以及 AddressAssign 的 GEMM read/write effect；该数字仅是历史里程碑，不是当前总数。4 个 skip 均需要独立外部工具或显式硬件授权。

## 3. 历史 PCIe 最终矩阵

以下所有 case 曾在 device 0 上 fresh-compile、单次 launch、回传并完成数值比较；当时 decoder 状态均为 `ready`。它们早于当前 region ABI 与安全监管修改，只作为 `historical_passed` 范围，不能授权当前提交加载板卡。逐条 `begin/end/duration/opcode` 位于 ignored 的
`research/artifacts/2026-09-04/pcie-final/*.json`。

| 组合 | add | sub | mul | div | matmul | 数值结论 | 指令 timing |
| --- | --- | --- | --- | --- | --- | --- |
| SG2260E / TPU-Kernel | 通过/16 | 通过/16 | 通过/16 | 通过/16 | 通过/36 | 5/5 通过 | 全部解码 |
| SG2260E / RV | 通过/16 | 通过/16 | 通过/16 | 通过/16 | 通过/36 | 5/5 通过 | 全部解码 |

括号内是 decoded instruction 数。逐元素每例均为 8 次 `tensorLd`、4 次算术、4 次 `tensorSt`；matmul 每例为 16 次 `tensorLd`、8 次 `MM2_NN`、4 次 accumulator zero/copy、4 次 `data_convert`、4 次 `tensorSt`。

单次记录的算术 duration（每条）如下；只用于证明 profiling 和指令选择生效，不作统计性能结论：

| opcode | TPU-Kernel | RV |
| --- | ---: | ---: |
| add / sub / mul | 14 ns | 12 ns |
| div | 74 ns | 72 ns |
| MM2_NN | 51 ns | 51 ns |
| data_convert | 12 ns | 12 ns |

### 3.1 PCIe 暴露的软件流水冒险

第一次 TPU-Kernel matmul 板端运行正常返回，但数值最大绝对误差为 `30.15625`，矩阵按首错停止，未继续 RV。生成源码显示 `T.Pipelined(num_stages=1)` 把 A/B DMA load 与立即消费它们的 GEMM 放进同一 `tpu_parallel_start/end` 区；CModel 编译路径会剥掉该标记，所以此前 CModel 结果无法发现这类真实并发冒险。

核心正确性 kernel 随后改为 `T.serial`，三组 CModel matmul 重新通过；受监护 PCIe TPU-Kernel matmul 也数值通过并解码 36 条指令，之后 RV 五项全部通过。该过程说明 CModel 是必要前置条件，但不能替代涉及真实调度/并行语义的 PCIe 验证。software pipeline 在建立后端 dependency/hazard contract 前不计入已支持能力。

## 4. 复现原则

1. 先运行全部 CModel 基线；任何 CModel 失败均禁止进入 PCIe。
2. PCIe 每个 case 必须在新的、受 supervisor 管理的进程中执行；为外层命令设置 process-group timeout。
3. 只在板卡空闲、指定 device id 后启动；首个异常即终止父进程组并跳过剩余 case。
4. 数值/raw 验收检查 `summary.json` 的 `complete=true`、每 case `status=passed`、数值 worker 成功，并要求 `cdm_profile_data_dev*` 下至少存在一个命名符合 `global.profile` 或 `cdmlibN_N.profile` 的文件，且目录内所有此类文件非空；逐指令 timing 验收额外使用 `--require-decoded-timing`，再要求 `parser_status=ready`、`timed_instruction_count>0`，且每条 begin/end/duration 都是非 bool 的有限数值、unit 恰为 `ns`、`duration>=0`、`end>=begin`。
5. 将实验输出保留在 `research/artifacts/<date>/<runtime>-matrix/`；该目录已被 `.gitignore` 排除，只提交本报告和可复现的测试代码。

## 5. 解释边界

本报告证明的是所列形状、dtype 和前端契约的功能正确性。它不证明：任意尺寸的 tail 行为、广义 broadcast/reshape、异步并发、跨核调度、吞吐/延迟最优性，或未列出的 RV 指令。性能研究应关闭 recorder，使用重复测量、统计量和单独的多核调度设计。
