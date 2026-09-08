# TileLang TPU 算子验证报告

## 1. 证据口径

本报告只以 `research/artifacts/2026-09-08/final-6b6772be/` 下 12 份
`summary.json` 作为当前 canonical 结果。它们全部记录：

- revision：`6b6772be62803338ce52cc0e57b82c47752c738d`；
- source state：`2c22969b21ce5aa66d44cdea6585bfbd6dc897f3c947248c809dc21bc2231170`；
- `implementation_worktree_dirty=false`；
- `complete=true`、`status=passed`、failed=0。

source identity 覆盖除 `research/**` 外的 tracked 与 untracked 文件。PCIe summary 还绑定
两份同 revision CModel 晋级证据、只读 git snapshot、PPL 1.7 公共/逐芯片输入、TileLang/TVM
动态库、交叉工具链、firmware、TPUDNN、安装的 runtime 与 `tpu-smi` 身份。

每个数值 case 在 fresh worker 中编译、加载并单次执行，与精确或 PyTorch oracle 比较。
矩阵首错停止；CModel 与 PCIe 是独立证据层。`research/artifacts/**` 被 Git 忽略，以下链接
用于本机复核，不是仓库内发布物。

## 2. Canonical 结果

| 矩阵 | BM1690 CModel | SG2260E CModel | SG2260E PCIe |
| --- | ---: | ---: | ---: |
| 双后端核心映射 | [28/28](../artifacts/2026-09-08/final-6b6772be/core-bm1690-cmodel/summary.json) | [56/56](../artifacts/2026-09-08/final-6b6772be/core-sg2260e-cmodel/summary.json) | [56/56](../artifacts/2026-09-08/final-6b6772be/core-sg2260e-pcie/summary.json) |
| FP8 TPU-Kernel | [42/42](../artifacts/2026-09-08/final-6b6772be/fp8-bm1690-cmodel/summary.json) | [42/42](../artifacts/2026-09-08/final-6b6772be/fp8-sg2260e-cmodel/summary.json) | [42/42](../artifacts/2026-09-08/final-6b6772be/fp8-sg2260e-pcie/summary.json) |
| 完整 TPU-Kernel op | [152/152](../artifacts/2026-09-08/final-6b6772be/tpukernel-bm1690-cmodel/summary.json) | [146/146](../artifacts/2026-09-08/final-6b6772be/tpukernel-sg2260e-cmodel/summary.json) | [146/146](../artifacts/2026-09-08/final-6b6772be/tpukernel-sg2260e-pcie/summary.json) |
| 高层 demo | [36/36](../artifacts/2026-09-08/final-6b6772be/demo-bm1690-cmodel/summary.json) | [51/51](../artifacts/2026-09-08/final-6b6772be/demo-sg2260e-cmodel/summary.json) | [51/51](../artifacts/2026-09-08/final-6b6772be/demo-sg2260e-pcie/summary.json) |
| **分阶段合计** | **258/258** | **295/295** | **295/295** |

12 份 summary 的实际 `scheduled_case_count` 与 `passed_case_count` 求和均为 **848**，所以本轮
保存的执行结果为 **848/848**。该数是矩阵执行次数，不是互不重叠的能力点数量：核心映射、
完整 TPU-Kernel、FP8 和高层 demo 有意从不同层级重复验证部分指令。

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

BM1690 CModel 152/152、SG2260E CModel 146/146、SG2260E PCIe 146/146。三份结果的最大
绝对误差都来自 `reduce-sum.bfloat16.w65`，为 0.0625，并在既定容差内。

远程提交前的 pass 审查另发现并修复了循环回边 liveness：旧逻辑会把“循环外初始化、每轮
前段读取”的 `weight` 与每轮后段写入的 `scratch` 分到同一地址。最小复现旧地址为
`weight/scratch/out=[0,0,128]`；按 allocation depth 延展可重复循环的 live range 后为
`[0,128,256]`。31 个新增参数化用例覆盖三个合法 target、符号/嵌套 `For`、`While`、首次只在
循环内使用、静态 0/1 次循环与 loop-local scratch；AddressAssign 全部 42 项通过。该变更会
保守增加循环外 buffer 的占用时间，因此下面的最终数值矩阵必须绑定修复后的新实现重新运行。

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

BM CModel 的最大绝对误差为
`tpukernel/flashattn.bfloat16.weighted-keys` 的 0.015625；SG CModel 与 PCIe 均为
`rv/elementwise-div.bfloat16` 的 0.015625。三阶段最大平均绝对误差均为
`tpukernel/flashattn.float16.weighted-keys` 的 0.012373005971312523。完整算法、oracle 与
容差见 [高层算子报告](../tpu-demo-ops/README.md)。

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

板端矩阵在完整 device session lock 内执行空闲观察。一次板卡发射结束后，`tpu-smi` 常会在
短时间内仍报告最近利用率；因此当前策略在一个 monotonic 总 deadline 内轮询，并要求连续
两个、间隔采样的 `Active/0%`。单个 `0%` 不足以晋级，`0 -> 非零` 会重置计数；Fault、拓扑
错误、无效 JSON 或 probe 错误立即失败。postflight 无法在 deadline 内稳定空闲时，会保留当前
session 的 fail-closed 状态并写 quarantine，且停止后续 case。

旧 revision `137c85d5bbf9036063904ba7c7e48c16db0302ce` 的首个 SG2260E PCIe
TPU-Kernel FP32 add 已通过数值 oracle，采集 5 个非空 raw 文件并解码 16 个 ns 事件；旧门禁
因紧接执行的一次 `Active/9%` 样本而把矩阵停在第 1 项。其 summary 因旧异常路径没有保留
完整通过 payload，只能作为诊断证据，不能计入最终通过结果：

`research/artifacts/2026-09-08/final-137c85d5/core-sg2260e-pcie/`

`6b6772be` 将策略改为持锁总 deadline、连续两个零样本，并在 postflight 失败时保留已完成的
numeric/profile 结果和失败阶段。当前 core 的真实首例观测为
`10% -> 10% -> 0% -> 0%`，正常等待后通过。四份 canonical PCIe summary 共包含 295 个
postflight：全部通过连续零判定，样本数为 4 或 5，观测到的最高瞬时利用率为 10%，最长 settle
为 1.923667 s，最终样本均为 0%。

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
