# TileLang TPU 算子验证报告

## 1. 证据口径

本报告以 `e5774525e3a6e11d0d6010e979203c55181a8872` 为实现基线，正式工件位于
[本机验收目录](../artifacts/2026-09-09/final-e5774525/)。选定的 39 份 summary 均记录同一
revision、`implementation_worktree_dirty=false`、`complete=true`、`status=passed`，且
失败数为 0。它们包括 8 份 CModel、2 份完整 PCIe 矩阵、14 份 TPU-Kernel PCIe 分片和
15 份 demo PCIe 分片。

BM1690 CModel 使用四份原始矩阵，SG2260E CModel 只使用四份 `*-cmodel-retry1` 矩阵。
PCIe 使用完整 core/FP8 矩阵和第 2 节列明的通过分片。原始并发 CModel 运行、失败矩阵及
分片的通过前缀、所有 recovery canary 均仅作诊断，不计入正式结果。

每个数值 case 在独立 worker 中编译、加载、执行，并与精确或 PyTorch oracle 比较。
PCIe summary 绑定同一实现的两芯片 CModel 前置证据和实际工具链/runtime 身份。矩阵首错
停止；恢复后的重跑保留独立目录，不覆盖失败记录。

`research/artifacts/**` 被 Git 忽略，本文链接用于本机复核，不是随仓库发布的原始工件。

收尾阶段又独立执行了当前分支相对 SG2260E 起始分支新增或修改的全部测试文件，结果为
682 passed、12 skipped；原生目标增量构建通过。机器可读契约的普通校验与本地工件严格校验
均通过，覆盖 4 个 target 条目、16 类 op、74 项 capability 和 81 条 evidence。

## 2. 正式验收结果

| 矩阵 | BM1690 CModel | SG2260E CModel | SG2260E PCIe |
| --- | ---: | ---: | ---: |
| 双后端核心映射 | [28/28](../artifacts/2026-09-09/final-e5774525/core-bm1690-cmodel/summary.json) | [56/56](../artifacts/2026-09-09/final-e5774525/core-sg2260e-cmodel-retry1/summary.json) | [56/56](../artifacts/2026-09-09/final-e5774525/core-sg2260e-pcie/summary.json) |
| FP8 TPU-Kernel | [42/42](../artifacts/2026-09-09/final-e5774525/fp8-bm1690-cmodel/summary.json) | [42/42](../artifacts/2026-09-09/final-e5774525/fp8-sg2260e-cmodel-retry1/summary.json) | [42/42](../artifacts/2026-09-09/final-e5774525/fp8-sg2260e-pcie/summary.json) |
| 完整 TPU-Kernel op | [152/152](../artifacts/2026-09-09/final-e5774525/tpukernel-bm1690-cmodel/summary.json) | [146/146](../artifacts/2026-09-09/final-e5774525/tpukernel-sg2260e-cmodel-retry1/summary.json) | [146/146](../artifacts/2026-09-09/final-e5774525/tpukernel-sg2260e-pcie-shards/) |
| 高层 demo | [36/36](../artifacts/2026-09-09/final-e5774525/demo-bm1690-cmodel/summary.json) | [51/51](../artifacts/2026-09-09/final-e5774525/demo-sg2260e-cmodel-retry1/summary.json) | [51/51](../artifacts/2026-09-09/final-e5774525/demo-sg2260e-pcie-shards/) |
| **分阶段合计** | **258/258** | **295/295** | **295/295** |

正式集合的 `scheduled_case_count` 与 `passed_case_count` 求和均为 848，其中 CModel
553/553、PCIe 295/295。这是选定验收集合的执行次数，不是独立能力数，也不是整个实验过程
的总发射次数。各矩阵有意重复验证部分指令；失败尝试和恢复实验另行保留。

PCIe 分片按完整 case 标识核对覆盖范围，同一矩阵内无重复、无遗漏：

| 分片集合 | 选定目录 | 分片数 | case 数 |
| --- | --- | ---: | ---: |
| TPU-Kernel 基础 | `copy`、`fill-gemm`、`pointwise` | 3 | 59 |
| TPU-Kernel 扩展 | `extended-exp`、`extended-gather`、`extended-rope`、`extended-rsqrt`、`extended-sigmoid` | 5 | 15 |
| TPU-Kernel reduction | sum/max × FP16/BF16/FP32；BF16 max 只取 `reduce-max-bfloat16-retry1` | 6 | 72 |
| TPU-Kernel demo | 四则、matmul、rmsnorm、rmsnorm-splitk、rope、swiglu、flashattn | 10 | 36 |
| RV demo | 四则、matmul | 5 | 15 |

TPU-Kernel 的 `extended` 与 `reduce-max-bfloat16` 失败分片不参与合并。demo 的每个分片
覆盖三种 dtype，FlashAttention 另覆盖三种输入分布，因此该分片为 9 项。

## 3. 低层覆盖

### 3.1 双后端核心矩阵

每个编程模型包含 28 项：4 项 FP16/FP32 copy、4 项基础四则、12 项三种基础浮点的 W
broadcast 四则、7 项 dense/broadcast/负无穷 max，以及 1 项 matmul。BM1690 只调度
TPU-Kernel 28 项；SG2260E 对 TPU-Kernel 与 RV Tensor 各调度同一组 28 项，因此为 56 项。

三阶段分别为 BM CModel 28/28、SG CModel 56/56、SG PCIe 56/56。这证明相同前端语义在
SG2260E 上能由两个后端独立生成和执行，不表示运行时自动回退。

### 3.2 完整 TPU-Kernel 矩阵

SG2260E 的 146 项与 BM1690 的共同部分如下：

| 能力族 | case 数 | 已验证范围 |
| --- | ---: | --- |
| copy/cast | 23 | 三种基础浮点、六种整数、S2S、FP32 rank-3 与本地 cast |
| fill | 3 | FP16/BF16/FP32 非零填充 |
| GEMM | 6 | FP16/BF16，NN overwrite/accumulate 与 NT overwrite |
| tensor add/sub/mul/div | 16 | 三种基础浮点 dense，并含 FP32 W broadcast |
| tensor max | 5 | 三种 dense、FP32 broadcast、FP32 negative-infinity |
| scalar add/mul | 6 | 两个 op × 三种基础浮点 |
| exp/sigmoid | 6 | 两个 op × 三种基础浮点 |
| reduce-sum/reduce-max | 72 | 两个 op × 三种基础浮点 × 12 个 width |
| rsqrt | 3 | 三种基础浮点 |
| rope/gather | 6 | 两个 op × 三种基础浮点 |
| **共同部分** | **146** |  |

BM1690 另有 FP32/INT32/UINT32 × 升序/降序的 6 项 topk，因此为 152。SG2260E 的头文件
虽暴露对应 HAU 原语，但 CModel 实验已证明该芯片不适用；生产能力表明确不调度，而不是在
运行时冒险调用。

BM1690 CModel 152/152、SG2260E CModel 146/146、SG2260E PCIe 146/146。三阶段正式结果的最大
绝对误差都来自 `reduce-sum.bfloat16.w65`，为 0.0625，并在既定容差内。

远程提交前的 pass 审查另发现并修复了循环回边 liveness：旧逻辑会把“循环外初始化、每轮
前段读取”的 `weight` 与每轮后段写入的 `scratch` 分到同一地址。最小复现旧地址为
`weight/scratch/out=[0,0,128]`；修复后这三个 buffer 的地址互不重叠。31 个新增参数化用例
覆盖三个合法 target、符号/嵌套 `For`、`While`、首次只在
循环内使用、静态 0/1 次循环与 loop-local scratch；AddressAssign 全部 42 项通过。该变更会
保守增加循环外 buffer 的占用时间；本报告的 e5774525 正式矩阵已在包含该修复的实现上重跑。

### 3.3 FP8 矩阵

E4M3 与 E5M2 每种格式各 21 项：copy、S2S、zero fill、双向 cast、add/sub/mul、三种
broadcast、max 与 max-broadcast、add/mul scalar、gather、RoPE，以及 NN/NT 的 overwrite
与 accumulate GEMM。两格式合计 42 项。

BM1690 CModel、SG2260E CModel 和 SG2260E PCIe 均为 42/42。该结论只覆盖 summary 中的
固定 shape、受控输入域和 selector；不外推到非零 FP8 fill、FP16/BF16 与 FP8 间的所有组合、
FP8 C、NaN/Inf 或更大 shape。

## 4. 高层算子覆盖

TPU-Kernel 的 36 项构成为：elementwise 12、matmul 3、RMSNorm 3、RMSNorm split-k 3、
RoPE 3、SwiGLU 3、FlashAttention 9。RV Tensor 的 15 项构成为 elementwise 12 与
matmul 3。每组都完整覆盖 FP16、BF16、FP32。

| 阶段 | TPU-Kernel | RV Tensor | 合计 |
| --- | ---: | ---: | ---: |
| BM1690 CModel | 36 | 不适用 | 36/36 |
| SG2260E CModel | 36 | 15 | 51/51 |
| SG2260E PCIe | 36 | 15 | 51/51 |

三个阶段的最大绝对误差均为 0.015625，最大平均绝对误差均约为 0.00260836，后者来自
BF16 elementwise-div。上述数值由本轮正式 summary 重新汇总，不沿用旧基线的误差统计。
FP32 matmul 和 FlashAttention 表示 FP32 逻辑 I/O；矩阵计算显式使用 BF16 输入和 FP32
累加，不能据此声称原生 FP32 GEMM 已验证。完整算法、oracle 与容差见
[高层算子报告](../tpu-demo-ops/README.md)。

## 5. Profiling 证据

CModel 的 core、FP8、demo 每项都有非空 raw 命令记录，但本环境没有提供可接受的 CModel
duration decoder，故 timed 均为 0。raw 命令数如下；它们证明 trace 被实际收集，不表示零耗时。

| CModel profile 矩阵 | BM1690 raw 命令 | SG2260E raw 命令 |
| --- | ---: | ---: |
| core | 2604 | 3176 |
| FP8 | 3434 | 1914 |
| demo | 5330 | 4736 |

SG2260E PCIe 的 profiling case 每项各有一个非空 raw trace 目录，并由
`bigTpuProfile 0.3.5` 的 `bigTpuProfile.bmprofile_perfAI.ProfileParser.parse` 解码：

| PCIe profile 矩阵 | case | decoded ns 事件 | BDC / GDMA |
| --- | ---: | ---: | ---: |
| core | 56 | 824 | 218 / 606 |
| FP8 | 42 | 150 | 42 / 108 |
| demo | 51 | 2594 | 2168 / 426 |
| **合计** | **149** | **3568** | **2428 / 1140** |

完整 TPU-Kernel 146 项是数值矩阵，不宣称逐指令 timing。上述 3568 个事件来自 149 次相互
独立的单次诊断发射。它们适合核对 op→instruction 映射和定位异常，不具备 warm-up、repeat、
置信区间或 recorder 开销校正，不能当作稳态 benchmark，也不能把事件时长简单相加后比较
不同后端性能。

## 6. PCIe 空闲判定与故障复盘

板端矩阵在完整 device session lock 内执行空闲检查。命令返回后，`tpu-smi` 仍可能短暂
报告非零利用率。runner 在 10 秒单调时钟 deadline 内轮询，要求两个间隔至少 0.25 秒的
连续 `Active/0%` 样本；再次出现非零利用率会重置计数。单次 `Fault`、拓扑错误、无效
JSON 或 probe 错误立即失败。postflight 无法证明空闲时，runner 保留 session/quarantine
状态并停止剩余 case，不自动清除隔离或继续运行。

当前实现保留失败采样的原始 payload、时刻和错误信息。正式集合的 295 次 PCIe 执行均通过
postflight；这是各次受控执行的通过记录，不是连续 295 次无中断运行的证明。

本轮完整 TPU-Kernel、完整 demo 和部分 TPU-Kernel 分片曾在数值检查之后出现健康检查
失败。已保留的 `Fault` 样本中，温度、时钟、利用率、电压均为 `F`。这些字段不可用于推断
过热、降频或电压异常，也不足以确定故障来自指令、固件、运行时还是遥测链路。根因尚未证实。
失败完整矩阵、失败分片及 recovery canary 都只用于诊断；即使其中某次数值和解码已通过，
也不提升为正式通过 case。

恢复后的完整通过分片按第 2 节重新组成验收集合。它们证明已列 selector 能在受控条件下
执行，不证明故障已消失或可以放宽单次 Fault 停止策略。后续复测仍须先确认设备健康、按既有
恢复流程处理隔离，并保留首次异常证据；长期稳定性需要独立的重复运行实验。

## 7. 结论边界与后续工作

当前证据支持：

- BM1690 TPU-Kernel 的 CModel 低层 op、FP8 与高层 demo；
- SG2260E TPU-Kernel 的 CModel/PCIe 低层 op、FP8 与高层 demo；
- SG2260E RV Tensor 的 CModel/PCIe 核心映射，以及 elementwise/matmul 高层表达。

仍需按优先级补齐：

1. **BM1690 PCIe**：本机没有 BM1690 板卡，SG2260E 结果不得外推。
2. **RV 复合算子**：补齐 reduction 与 math 语义后，依次验证 RMSNorm、split-k、RoPE、
   SwiGLU、FlashAttention；在此之前保持 fail-closed。
3. **边界输入和 shape**：覆盖 tail、动态 shape、NaN/Inf、除零、更多 broadcast/alias 和更大
   GEMM/reduction；现有固定 selector 不能代表整个 dtype 空间。
4. **FP8 扩围**：验证非零 fill、更多跨 dtype cast、FP8 C、异常值和 shape 边界。
5. **四核执行**：增加 SG2260E 四核分片、同步与依赖安全测试。当前矩阵证明数值正确性，
   不证明多核扩展效率。
6. **性能测试**：另建无 profiling recorder 的 warm-up/repeat 基准和统计阈值，避免把本报告的
   单次诊断时间误作性能结论。

新增证据应先进入 case registry 和机器可读 op contract，再按 BM1690 CModel、SG2260E
CModel、SG2260E PCIe 的顺序晋级；不得从相邻 dtype、另一芯片或复合 workload 反推支持。
