# TileLang TPU 高层算子设计与验收报告

## 1. 目的与范围

本报告记录 `tpu_demo` 高层算子的统一实现、数值判定和分阶段验收方法。范围包括
elementwise add/sub/mul/div、matmul、RMSNorm、RMSNorm split-k、RoPE、SwiGLU 和
FlashAttention，公开 dtype 为 FP16、BF16、FP32。

报告只把可复现证据写成结论。当前实现与 source-only 检查仍在本轮修改中；BM1690
CModel、SG2260E CModel 和 SG2260E PCIe 的最终矩阵结果将在同一干净实现基线上依次生成，
下文结果栏在此之前明确保持“待最终矩阵填入”。既有低层 op 工件不自动升级为这些高层
算子的通过证据。

## 2. 统一架构

高层 demo 分为四层：

```text
cases.py（能力与 case 注册）
  → run.py（唯一语义分派入口）
    → <operator>/<operator>.py（TileLang builder + PyTorch oracle）
      → common.py（target/runtime 门禁、compile/launch、比较与结果协议）

testing/python/jit/tpu_demo_ops_matrix.py
  → TPUInstructionProfiler（隔离进程、raw/timing、deadline）
    → tpu_demo_ops_worker.py（一个 case、一次 compile、一次 launch）
      → run_case(...)
```

算子模块可安全导入，不在 module scope 编译或加载 runtime。用户侧表达继续使用
`T.ppl_*` 语义；`chip + programming_model` 只在 target 选择和 codegen 层决定映射。
BM1690 只允许 TPU-Kernel，SG2260E 可选择 TPU-Kernel 或 RV Tensor。当前只有
elementwise 与 matmul 的依赖全部属于双后端可移植语义；RMSNorm、RoPE、SwiGLU 和
FlashAttention 仍依赖 TPU-Kernel 专属 math/reduction/composite op，因此注册表对 RV
fail-closed。

36 个 case 的构成为：

| 算子族 | 变体 | dtype | case 数 | RV 可用 |
| --- | --- | --- | ---: | --- |
| elementwise | add/sub/mul/div | FP16/BF16/FP32 | 12 | 是 |
| matmul | tiled accumulate | FP16/BF16/FP32 | 3 | 是 |
| RMSNorm | ordinary | FP16/BF16/FP32 | 3 | 否 |
| RMSNorm | split-k | FP16/BF16/FP32 | 3 | 否 |
| RoPE | interleaved even/odd | FP16/BF16/FP32 | 3 | 否 |
| SwiGLU | sigmoid composite | FP16/BF16/FP32 | 3 | 否 |
| FlashAttention | balanced / descending-max / weighted-keys | FP16/BF16/FP32 | 9 | 否 |

所有 builder 只接受静态正整数维度，并要求公开 extent 可被 tile 整除。当前没有 tail
谓词或 masked DMA，因此非整 tile shape 必须拒绝，不能用向上取整后越界访问代替。

## 3. 算法与映射

### 3.1 Elementwise 与 matmul

elementwise 的公开张量先搬入 local tile，再调用统一的 add/sub/mul/div 语义并写回。
matmul 以 FP32 local accumulator 跨 K tile 累加；FP16/BF16 直接作为矩阵乘数，公开 FP32
路径则将 A/B 转为 BF16 后执行矩阵指令，输出仍为 FP32。这样既保留用户可见的 FP32
接口，也不虚构硬件矩阵引擎的 FP32 乘数能力。

### 3.2 RMSNorm、RoPE 与 SwiGLU

RMSNorm 按行计算 `x * rsqrt(mean(x²) + epsilon)`。普通版一次覆盖完整 reduction width；
split-k 先逐 K tile 累加平方和，再反向读取各 tile 完成归一化，避免为方便验证而复制一套
不同语义。低精度输入统一提升到 FP32 完成平方、归约和 rsqrt，最后转回公开 dtype。

RoPE 使用偶/奇 lane 的交错旋转语义组合 `x*cos` 与 `x*sin`。SwiGLU 计算
`silu(gate) * up`，即 `gate * sigmoid(gate) * up`；低精度输入同样使用 FP32 中间量。

### 3.3 FlashAttention

当前 FlashAttention 使用 BSHD 布局，并以 online softmax 跨两个 K/V tile 合并：每轮先取
当前 tile 行最大值，再用逐元素 `max(previous_max, current_max)` 得到全局历史最大值，重标定
旧的 `row_sum/accumulator` 后累加当前 `exp(score - row_max)`。该状态转移是跨 tile 数值稳定性
的核心，不能把 `row_max` 简化为当前 tile 最大值。

三个验证变体承担不同职责：

- `balanced` 使用普通受控随机输入，覆盖常规数值路径；
- `descending-max` 刻意使后一 K tile 的 logits 最大值低于前一 tile，专门防止历史最大值被
  覆盖而导致指数溢出或错误重标定；
- `weighted-keys` 使用单调 key logits 与 key-dependent value，使均匀权重、只保留 argmax
  或丢失 tile 内权重差异的实现产生可观测误差，补足常量 value 无法识别的权重路径缺陷。

FP32 公开路径的 Q/K/V 先量化到 BF16 后参与两次 GEMM，softmax 状态保持 FP32；PyTorch
oracle 必须复现相同的 Q/K/V 输入量化边界，但仍以数学上的理想 FP32 softmax 为语义基准。
kernel 在 probability/value GEMM 前会把 FP32 指数权重 tile 转成矩阵引擎 dtype；oracle
刻意不复刻该 probability cast 的逐 tile 舍入，专属容差衡量它相对理想语义的误差，而不是
把实现舍入写入参考答案。`is_causal` 必须是严格 bool：非 bool 值报类型错误，`True` 明确
报尚未实现；只缩短 K tile 循环不能正确表达对角 tile 内 mask，完整 mask 落地前仅允许
`False`。

## 4. Oracle 与容差

每个结果先做以下不可放宽的结构检查：

1. actual 与 expected 的 shape 完全一致；
2. dtype 完全一致；
3. 两侧所有元素均为有限值；
4. 再按同一算子族的固定 atol/rtol 执行逐元素比较。

当前验收实际使用的容差如下。它们是验收上限，不是测得误差；最终报告还应从 summary 填入每个
case 的实际最大/平均绝对误差。

| 算子族 | FP16 atol/rtol | BF16 atol/rtol | FP32 atol/rtol | 说明 |
| --- | ---: | ---: | ---: | --- |
| elementwise add/sub/mul | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 | 同一基础逐元素规则 |
| elementwise div | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-5 / 1e-5 | 输入生成器使用严格正、远离零的分母 |
| matmul | 1e-2 / 1e-2 | 2e-2 / 2e-2 | 1e-2 / 1e-2 | FP32 路径含 BF16 乘数量化 |
| RMSNorm / SwiGLU | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-2 / 1e-2 | FP32 中间计算 |
| RoPE | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 | 交错旋转 |
| FlashAttention | 2e-2 / 2e-2 | 2e-2 / 2e-2 | 2e-2 / 2e-2 | 两次低精度矩阵乘及已刻画的近似 exp 误差 |

若某项超过容差，应先依据误差模式、raw 指令与具体 tile 状态修正实现或契约；不得仅为让矩阵
通过而提高整族容差。NaN/Inf 即使在 `isclose` 下偶然匹配也判失败。

## 5. Profiling 与证据分层

高层矩阵复用 `TPUInstructionProfiler`，但把三类结果严格分开：

- `numeric`：一次 launch 的输出与 oracle；
- `raw_instruction_count`：CModel dump 或 PCIe recorder 中实际收集的命令；
- `timed_instruction_count`：经受控 decoder 得到、具有有限 begin/end/duration 和 `ns` 单位的
  指令事件。

CModel raw trace 通常只能确认指令选择，不包含真实 duration；因此 `timed=0` 不能写成“零
耗时”。PCIe 的单次 decoded timing 用于核对 op→instruction 映射和定位异常，也不能当作稳定
性能均值。若最终验收目标包括逐指令时间，PCIe invocation 必须使用
`--require-decoded-timing`，且 decoder preflight 必须在任何板卡 launch 前通过。

## 6. PCIe 安全与晋级

执行顺序固定为 BM1690 CModel → SG2260E CModel → SG2260E PCIe。PCIe runner 要求
`--allow-pcie` 与 `--allow-pcie-profile` 双确认、显式且只能为 0 的 device id，以及显式 case/op 子集或
`--all-pcie-cases`。两份 CModel promotion summary 必须来自当前同一 clean commit/source
state，完整通过并逐 case 覆盖即将上板的 selector。它们记录的 TileLang/TVM 动态库、host
编译器、PPL 公共与逐芯片文件、backend、emulator 和 CModel runtime 内容身份还必须与本轮
PCIe 工具链的共同部分完全一致；交叉工具链、firmware、TPUDNN、安装的 board runtime 和
实际 `tpu-smi` 则作为 PCIe 专属身份写入 summary。

每个 case 在 fresh worker 中 compile 一次、launch 一次，不做 benchmark 重复。worker 位于
parent-death 约束的独立进程组；首个编译、加载、deadline、数值、raw、strict timing 或板卡
健康错误会触发 TERM→KILL→reap，并跳过剩余用例。完整 invocation 持有同一设备锁；实际
worker 从 promoted commit 和对应 TVM gitlink 构造的私有只读快照编译，每次发射前仍重新哈希
共享源码与 manifest 所覆盖的编译/runtime 输入。该 manifest 不递归覆盖 host GCC 的内部程序、
系统头文件/链接器/libc，也不覆盖 Python/PyTorch 或可选 decoder 文件内容，因此属于晋级门而非
hermetic build 证明。若进程组不能完全回收则生成持久 quarantine marker；若父进程异常死亡
则会留下 session marker。下一轮会默认拒绝使用该设备，直到操作者检查 PID/PGID、确认无残留
并恢复板卡后人工解除。PCIe preflight 和每个通过 case 的 postflight 还要求恰好一张可见卡、
一个 chip、唯一 `sg-host-drv` 的 `1f1c:1690` 设备，状态为 `Active` 且利用率为 0%。发生失败
后不得在同一 runtime 实例中盲目重试。

## 7. 最终验收记录

以下三行必须按顺序由同一干净实现生成。结果、revision、case manifest 和误差统计均应直接
取自对应 `summary.json`，在矩阵完成前不得填入推测值。

| 阶段 | 预期调度范围 | 结果 | canonical evidence |
| --- | --- | --- | --- |
| BM1690 CModel | TPU-Kernel 36 case | **待最终矩阵填入** | `research/artifacts/<date>/<bm-cmodel-run>/summary.json` |
| SG2260E CModel | TPU-Kernel 36 + RV 15，共 51 case | **待最终矩阵填入** | `research/artifacts/<date>/<sg-cmodel-run>/summary.json` |
| SG2260E PCIe | 前两阶段覆盖的同一 selector，最多 51 case | **待最终矩阵填入** | `research/artifacts/<date>/<sg-pcie-run>/summary.json` |

最终填报时至少记录：完整/通过/失败 case 数，implementation revision 与 clean 状态，各 target
分布，最大误差及对应 case，raw/timed 指令总数，decoder identity，PCIe 前后板卡状态，以及
是否存在受控进程残留。任何阶段未完整通过，后续阶段保持未执行，不得把阶段性开发工件标为
canonical。

## 8. 后续扩展原则

新增 dtype、shape、tail、causal mask 或 RV 复合算子时，先扩展依赖无关的 case registry 和
精确 oracle，再增加 target capability；随后按 source selection、CModel、PCIe 三层升级证据。
若底层指令的 dtype/shape 范围只能靠实验确认，应为最小 selector 建独立低层用例，并更新
`research/tpu-op-contract/contract.json`，不得从相邻 dtype、另一芯片或复合 workload 反推
支持。
