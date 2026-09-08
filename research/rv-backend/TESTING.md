# TileLang-TPU 双后端测试报告

> 当前结论以 2026-09-08 的 clean 实现基线 `6b6772be62803338ce52cc0e57b82c47752c738d` 为准。本文后半仍保留 2026-09-04/07 的历史竖切和故障定位过程，但这些旧工件不再授权当前能力。全量数值结论、FP8 边界和 decoder 契约以 [TPU 后端设计与验证报告](../tpu-backend-design/README.md)、[公开 demo 验收报告](../tpu-demo-ops/README.md) 和 [profiling 设计](../ppl-profiling/README.md) 为准。

## 1. 范围与判定

本报告的数值测试以 PyTorch 结果为 oracle。每个 case 在独立进程中 fresh-compile、加载、发射一次并回传比较；同时要求成功产生 profiling raw trace。CModel 不伪造指令时间，故 `timed_instruction_count=0` 是预期结果；当前 PCIe 严格验收还要求 decoder 产生合法的 ns 区间。

核心矩阵由 `testing/python/jit/tpu_core_ops_matrix.py` 驱动，涵盖 copy、elementwise dense/W-broadcast/max 和 matmul；公开二层 demo 由 `testing/python/jit/tpu_demo_ops_matrix.py` 驱动：

| 运行时 | 组合 |
| --- | --- |
| CModel | SG2260E/TPU-Kernel、SG2260E/RV、BM1690/TPU-Kernel |
| PCIe | SG2260E/TPU-Kernel、SG2260E/RV |

PCIe 矩阵是 fail-stop：一个 case 超时、加载失败、数值不符、没有 raw trace、decode 不合格或板卡未在 deadline 内静止，立即记录部分结果、隔离设备并停止后续测试。它需要显式 `--allow-pcie --allow-pcie-profile --device-id <n>`；不可因设备风险跳过这些门禁。严格 timing 验收再加 `--require-decoded-timing`，此时会在任何硬件派发前完成无硬件 decoder preflight。decoder 的专用 Python/PYTHONPATH 不进入编译或数值 worker。

## 2. 当前 `6b6772be` canonical 结果

当前 12 份 summary 均记录同一实现 revision、`implementation_worktree_dirty=false`、`complete=true`，失败和取消为 0。三种可执行环境共完成 848/848 次实际 launch：

| 环境 | core | FP8 | TPU-Kernel 全量 | demo | 实际 launch |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM1690 CModel | 28/28 | 42/42 | 152/152 | 36/36 | 258/258 |
| SG2260E CModel | 56/56 | 42/42 | 146/146 | 51/51 | 295/295 |
| SG2260E PCIe | 56/56 | 42/42 | 146/146 | 51/51 | 295/295 |

不同矩阵会有意重复验证 selector，848 是 runner 的实际执行数，不是去重能力数。SG 核心矩阵在 TPU-Kernel/RV 各执行 28 项；公开 demo 在 TPU-Kernel 执行 36 项，在 RV 执行 elementwise 四则与 matmul 的 FP16/BF16/FP32 共 15 项。RMSNorm、Split-K RMSNorm、RoPE、SwiGLU 和三种输入变体的 FlashAttention 当前只适用 TPU-Kernel，并以三种 dtype 完成 BM CModel、SG CModel、SG PCIe 闭环；不能外推为 RV 支持。

SG PCIe 的 core 56、FP8 42、demo 51 共 149 个严格 profiling case 全部保留 raw capture，并由 `bigTpuProfile==0.3.5` 解码 3568 条有效 ns 事件。其中 RV 是 core 28 项 412 条事件、demo 15 项 168 条事件，共 43 项 580 条事件。TPU-Kernel 全量 146 项是独立的非 recorder 数值扩面。单次 event 用于检查映射和定位热点，不用于 TPU-Kernel/RV 性能排名。

每个 case 返回后，runner 仍持有整张设备锁，并要求同一物理设备连续两次间隔采样为 `Active/0%` 才可复用。首项实测为 `10% → 10% → 0% → 0%`；这证明“host 命令返回”和“设备已静止”不是同一时刻。最终板卡为 `Active/0%`，无受控 worker/supervisor、session 或 quarantine marker 残留。

## 3. 历史 CModel 结果

实验时间：2026-09-04。最终摘要保存在被 Git 忽略的
`research/artifacts/2026-09-04/cmodel-final/`。

| 组合 | 算子数 | 数值结果 | raw 命令总数/单 case | trace 文件/单 case |
| --- | ---: | --- | --- | --- |
| SG2260E / TPU-Kernel | 5 | 全部通过 | elementwise 58；matmul 78 | 24 |
| SG2260E / RV | 5 | 全部通过 | elementwise 58；matmul 78 | 24 |
| BM1690 / TPU-Kernel | 5 | 全部通过 | elementwise 94；matmul 122 | 48 |

合计 15/15 通过。SG2260E 每个 case 的 24 个 trace 文件和 BM1690 的 48 个 trace 文件反映了各自 4/8 核 CModel 拓扑及多类 engine dump；它们不是四/八核并行计算的证明。当前 host `core_num=1`，故这些测试验证的是单核工作负载在正确芯片拓扑、指令 ABI 和数值路径下可用。

`e5e3087` 的 [历史 core summary](../artifacts/2026-09-07/core-cmodel-e5e3087/summary.json) 曾按当时九项 registry 重跑三个合法 target，共 27/27；其中 SG2260E/RV 包含四条 copy case。它与同轮 TPU-Kernel 288/288、FP8 76/76 只保留为实现演进记录，当前授权已经由 `6b6772be` 的 12 份 canonical summary 取代。

2026-09-04 同期的静态/单元回归为 `101 passed, 4 skipped`，覆盖当时的 target capability、PPL layout、模型隔离、raw ABI 混用拒绝、copy/cast/layout 的失败闭合、生成源码隔离、PCIe compile/link（不加载硬件）、profiling 超时清理以及 AddressAssign 的 GEMM read/write effect；该数字仅是历史里程碑，不是当前总数。4 个 skip 均需要独立外部工具或显式硬件授权。

## 4. 历史 PCIe 最终矩阵

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

### 4.1 PCIe 暴露的软件流水冒险

第一次 TPU-Kernel matmul 板端运行正常返回，但数值最大绝对误差为 `30.15625`，矩阵按首错停止，未继续 RV。生成源码显示 `T.Pipelined(num_stages=1)` 把 A/B DMA load 与立即消费它们的 GEMM 放进同一 `tpu_parallel_start/end` 区；CModel 编译路径会剥掉该标记，所以此前 CModel 结果无法发现这类真实并发冒险。

核心正确性 kernel 随后改为 `T.serial`，三组 CModel matmul 重新通过；受监护 PCIe TPU-Kernel matmul 也数值通过并解码 36 条指令，之后 RV 五项全部通过。该过程说明 CModel 是必要前置条件，但不能替代涉及真实调度/并行语义的 PCIe 验证。software pipeline 在建立后端 dependency/hazard contract 前不计入已支持能力。

## 5. 历史 `e5e3087` PCIe 重验证

2026-09-07 的 [SG2260E/RV 核心 summary](../artifacts/2026-09-07/pcie-rv-core-e5e3087/summary.json) 记录 `complete=true`、`implementation_worktree_dirty=false`。九项 case 全部通过数值、规范命名且非空的 raw profile 和 decoded timing 严格门禁，共得到 108 条 ns timing：

| case | 结果 | decoded timing |
| --- | ---: | ---: |
| FP16/FP32 global-to-global copy | 2/2 | 各 1 条，共 2 条 |
| FP16/FP32 local-roundtrip copy | 2/2 | 各 3 条，共 6 条 |
| add/sub/mul/div | 4/4 | 各 16 条，共 64 条 |
| matmul | 1/1 | 36 条 |
| 合计 | 9/9 | 108 条 |

local-roundtrip 覆盖 G2L→L2L→L2S，global-to-global 覆盖 S2S；两种 dtype 都按 `(4,32)` 输入逐元素精确相等。每个 case 的 `parser_status=ready`、`decoded_timing_accepted=true`，所有 interval 均以 ns 计量。RV matmul 36 条事件 duration 合计 6320 ns、跨度 9896 ns；四则各 16 条（4 op、8 load、4 store），add/sub/mul/div 合计分别为 3682/3198/3324/3858 ns；FP16 local/S2S copy 分别为 3 条合计 498 ns、跨度 1796 ns，以及 1 条 265 ns；FP32 对应为 3 条合计 496 ns、跨度 2685 ns，以及 1 条 303 ns。该证据证明当前九个固定 selector 的数值与 profiling 链路可用，这些单次 recorder 时间不是吞吐或延迟统计。

同一基线还完成 [TPU-Kernel matmul profiling](../artifacts/2026-09-07/pcie-tpukernel-matmul-e5e3087/summary.json) 1/1、36 条 ns timing，以及独立的 TPU-Kernel numeric 矩阵 [core 54/54](../artifacts/2026-09-07/pcie-tpukernel-core-e5e3087/summary.json)、[extended 15/15](../artifacts/2026-09-07/pcie-tpukernel-extended-e5e3087/summary.json)、[reductions 72/72](../artifacts/2026-09-07/pcie-tpukernel-reductions-e5e3087/summary.json)，合计 141/141。TPU-Kernel matmul 的 36 条事件为 8 `MM2_NN`、4 `copy`、4 `data_convert`、16 `tensorLd`、4 `tensorSt`，duration 合计 6844 ns、跨度 45060 ns。这组对照说明两种编程模型共用的 PCIe host、supervisor 与 decoder 路径均能闭环；numeric 矩阵没有逐指令 timing 门禁，不应把 141 项写成 profiling 结果。

FP8 对照矩阵在 [canonical summary](../artifacts/2026-09-07/pcie-fp8-e5e3087/summary.json) 中为 38/38：E4M3、E5M2 各 19 项，38 个 case 各产生一个非空 raw 目录，共解码 134 条 timing（BDC 38 条/840 ns，GDMA 96 条/22258 ns）。它证明 SG2260E/TPU-Kernel 上固定 FP8 selector 的板端路径，不扩大 RV selector 范围。

三份严格 timing summary 都在硬件派发前完成无硬件 decoder preflight，并记录 `bigTpuProfile 0.3.5` 与实际 `bigTpuProfile.bmprofile_perfAI.ProfileParser.parse` 身份。decoder-only Python/PYTHONPATH 不进入编译/数值 worker。

板端运行前后均观测为 `Active`、利用率 0%，结束后未发现受控 profiler、runner、decoder 或 runtime 后代残留。较早一次 `Fault`/100% preflight 只记录当时的 fail-stop 决策。`44a6fc2` 与 `e5e3087` 现均为历史层；当前结论只采用 clean `6b6772be` 工件。本机没有 BM1690 板卡，不能给出 BM1690 PCIe 结论。

## 6. 复现原则

1. 先运行全部 CModel 基线；任何 CModel 失败均禁止进入 PCIe。
2. PCIe 每个 case 必须在新的、受 supervisor 管理的进程中执行；为外层命令设置 process-group timeout；FP8 还必须显式且唯一指定 chip。
3. 只在板卡空闲、指定 device id 后启动；每次返回后在设备锁内等待连续两个间隔采样为 `Active/0%`；首个异常即终止受控进程组、持久隔离设备并跳过剩余 case。
4. 数值/raw 验收检查 `summary.json` 的 `complete=true`、每 case `status=passed`、数值 worker 成功，并要求 `cdm_profile_data_dev*` 下至少存在一个命名符合 `global.profile` 或 `cdmlibN_N.profile` 的文件，且目录内所有此类文件非空；逐指令 timing 验收额外使用 `--require-decoded-timing`，再要求 `parser_status=ready`、`timed_instruction_count>0`，且每条 begin/end/duration 都是非 bool 的有限数值、unit 恰为 `ns`、`duration>=0`、`end>=begin`。
5. 将实验输出保留在 `research/artifacts/<date>/<runtime>-matrix/`；该目录已被 `.gitignore` 排除，只提交本报告和可复现的测试代码。

## 7. 解释边界

本报告证明的是所列形状、dtype 和前端契约的功能正确性。它不证明：任意尺寸的 tail 行为、广义 broadcast/reshape、异步并发、跨核调度、吞吐/延迟最优性，或未列出的 RV 指令。性能研究应关闭 recorder，使用重复测量、统计量和单独的多核调度设计。
