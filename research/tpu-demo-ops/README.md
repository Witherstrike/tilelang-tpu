# TileLang TPU 高层算子设计与验收报告

## 1. 范围与结论

本报告覆盖 `tpu_demo` 中面向用户的 elementwise、matmul、RMSNorm、RMSNorm
split-k、RoPE、SwiGLU 和 FlashAttention。公开数据类型为 FP16、BF16、FP32。

当前 canonical 基线为：

- revision：`6b6772be62803338ce52cc0e57b82c47752c738d`；
- source state：`2c22969b21ce5aa66d44cdea6585bfbd6dc897f3c947248c809dc21bc2231170`；
- evidence root：`research/artifacts/2026-09-08/final-6b6772be/`；
- 3 份高层 summary 均记录 `implementation_worktree_dirty=false`、
  `complete=true`、`status=passed`。

结果为 BM1690 CModel 36/36、SG2260E CModel 51/51、SG2260E PCIe
51/51。这里的 51 项由 SG2260E TPU-Kernel 36 项和 RV Tensor 15 项组成；
三种运行环境都覆盖了 FP16、BF16、FP32。

## 2. 分层设计

```text
tpu_demo/cases.py                         能力和 case 注册
  -> tpu_demo/run.py                      唯一语义分派入口
    -> <operator>/<operator>.py           TileLang builder + PyTorch oracle
      -> tpu_demo/common.py               target/runtime 门禁、编译、执行、比较

testing/python/jit/tpu_demo_ops_matrix.py 矩阵、晋级门和首错停止
  -> TPUInstructionProfiler               隔离进程、deadline、raw/timing
    -> tpu_demo_ops_worker.py              每个 case 独立编译并执行一次
```

前端保持同一组 `T.ppl_*` 语义，不要求用户改写算法来选择芯片或编程模型。
`chip + programming_model` 在 target、lowering 和 codegen 边界完成选择：BM1690 只接受
TPU-Kernel；SG2260E 接受 TPU-Kernel 与 RV Tensor。未实现的映射在注册/编译阶段明确拒绝，
不回退到另一后端，也不通过名字猜测能力。

算子模块导入时不编译、不加载 runtime。用于验证单条 op 的程序放在 `testing/`，
`tpu_demo/` 只保留可复用的用户级表达、输入生成器和 oracle。

## 3. 能力矩阵

每个表格单元都是一个独立的 dtype case；FP16、BF16、FP32 各占一项。

| 算子族 | 变体 | TPU-Kernel | RV Tensor | 说明 |
| --- | --- | ---: | ---: | --- |
| elementwise | add/sub/mul/div | 12 | 12 | 两后端共用前端语义 |
| matmul | tiled accumulate | 3 | 3 | 两后端共用前端语义 |
| RMSNorm | ordinary | 3 | 0 | 依赖 TPU-Kernel math/reduction |
| RMSNorm | split-k | 3 | 0 | 跨 K tile 累加平方和 |
| RoPE | interleaved even/odd | 3 | 0 | 当前为 TPU-Kernel 专属组合 |
| SwiGLU | sigmoid composite | 3 | 0 | 当前为 TPU-Kernel 专属组合 |
| FlashAttention | balanced/descending-max/weighted-keys | 9 | 0 | 当前为 TPU-Kernel 专属组合 |
| **合计** |  | **36** | **15** | SG2260E 合计 51 |

因此：

- BM1690 CModel 调度 TPU-Kernel 36 项；
- SG2260E CModel 和 PCIe 均调度 TPU-Kernel 36 项与 RV Tensor 15 项；
- “RV 仅 15 项”是显式能力边界，不表示复合算子被静默改走 TPU-Kernel。

## 4. 算法与硬件映射

### 4.1 Elementwise 与 matmul

elementwise 将公开张量分块搬入 local memory，执行 add/sub/mul/div 后写回。除法的测试输入
使用严格正且远离零的分母，当前证据不覆盖除零和异常值传播。

matmul 使用 FP32 local accumulator 跨 K tile 累加。FP16/BF16 直接作为矩阵乘数；公开
FP32 路径先把 A/B 转为 BF16 再进入矩阵引擎，结果仍写为 FP32。oracle 在同一边界量化
A/B，既维持 FP32 用户接口，也不把硬件不具备的 FP32 乘数能力写进契约。

### 4.2 RMSNorm、RoPE 与 SwiGLU

RMSNorm 计算 `x * rsqrt(mean(x^2) + epsilon)`。普通版一次覆盖 reduction width；split-k
逐 K tile 累加平方和，再读取各 tile 完成归一化。低精度输入在 FP32 中完成平方、归约和
rsqrt，最后转回公开 dtype。

RoPE 对偶/奇 lane 执行交错旋转。SwiGLU 计算
`gate * sigmoid(gate) * up`，低精度路径同样使用 FP32 中间量。

### 4.3 FlashAttention

FlashAttention 使用 BSHD 布局和跨 K/V tile 的 online softmax。每轮保留历史行最大值，
在加入当前 tile 前重标定旧的 `row_sum` 与 accumulator，避免把历史最大值错误覆盖。

三个变体分别检查常规数值、后一 tile 最大值下降时的历史最大值保持，以及 key-dependent
value 对非均匀权重的敏感性。公开 FP32 路径先将 Q/K/V 量化为 BF16，softmax 状态保持
FP32；oracle 复现 Q/K/V 的量化边界，但不把 probability tile 的实现舍入写成数学定义。
当前 `is_causal` 只接受严格布尔值，并仅实现 `False`；对角 tile 内 mask 完成前，`True`
必须 fail-closed。

所有 builder 当前只接受静态正整数 shape，并要求公开 extent 可被 tile 整除。实现尚无 tail
predicate 或 masked DMA，因而非整 tile shape 不能靠向上取整后越界访问来“兼容”。

## 5. 数值判定

每个 case 依次检查 shape、dtype、有限值和逐元素误差。任何 NaN/Inf 都直接失败。容差是
验收上限，不是实测误差：

| 算子族 | FP16 atol/rtol | BF16 atol/rtol | FP32 atol/rtol |
| --- | ---: | ---: | ---: |
| elementwise add/sub/mul | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 |
| elementwise div | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-5 / 1e-5 |
| matmul | 1e-2 / 1e-2 | 2e-2 / 2e-2 | 1e-2 / 1e-2 |
| RMSNorm / SwiGLU | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-2 / 1e-2 |
| RoPE | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 |
| FlashAttention | 2e-2 / 2e-2 | 2e-2 / 2e-2 | 2e-2 / 2e-2 |

本轮最大绝对误差如下：

- BM1690 CModel：`flashattn.bfloat16.weighted-keys`，0.015625；
- SG2260E CModel 与 PCIe：`rv/elementwise-div.bfloat16`，0.015625；
- 最大平均绝对误差均出现在 `tpukernel/flashattn.float16.weighted-keys`，
  为 0.012373005971312523。

所有值均在固定容差内；没有通过放宽容差处理失败。

## 6. 分阶段验收结果

| 阶段 | 调度范围 | 结果 | canonical evidence |
| --- | --- | ---: | --- |
| BM1690 CModel | TPU-Kernel 36 | 36/36 | [summary](../artifacts/2026-09-08/final-6b6772be/demo-bm1690-cmodel/summary.json) |
| SG2260E CModel | TPU-Kernel 36 + RV 15 | 51/51 | [summary](../artifacts/2026-09-08/final-6b6772be/demo-sg2260e-cmodel/summary.json) |
| SG2260E PCIe | 与 SG CModel 相同的 51 项 | 51/51 | [summary](../artifacts/2026-09-08/final-6b6772be/demo-sg2260e-pcie/summary.json) |

三份 summary 的 scheduled、completed、passed 分别相等，failed 与 cancelled 均为 0。
PCIe 每项保留一个非空 raw trace 目录，51 项共解码 2594 个合法 ns 事件，其中 BDC 2168、
GDMA 426；decoder 为 `bigTpuProfile 0.3.5` 的结构化 `ProfileParser.parse` 接口。
CModel 保存了非空 raw 命令，但没有可用 duration，因此 `timed_instruction_count=0` 只表示
“没有时长证据”，不能解释成“指令耗时为零”。

这些 profiling 数据来自每个 case 的一次诊断发射，用于核对映射和定位问题。它没有
warm-up、重复采样、稳态统计或 recorder 开销校正，不能作为后端性能排名或 benchmark。

## 7. PCIe 安全闭环

PCIe 只在 BM1690 与 SG2260E 的同 revision CModel summary 完整通过后晋级。每个 case 在
fresh worker 中编译一次、执行一次；任一编译、加载、deadline、数值、raw、decoder 或板卡
健康错误都会终止受控进程组并停止剩余矩阵。

板卡空闲判断持有完整 device session lock，并在一个 monotonic 总 deadline 内轮询。只有
`Active` 且连续两个、间隔采样的 `0%` 才视为稳定空闲；`0 -> 非零` 会重置连续计数，
Fault、拓扑不符、无效 JSON 和 probe 失败立即拒绝。发射后的 settle 若超时或观察不完整，
当前 session 会保持 fail-closed 并写入 quarantine；已完成的数值与 profiling 证据仍保存在
失败 case 中，失败阶段标为 board postflight。

旧 revision `137c85d5bbf9036063904ba7c7e48c16db0302ce` 的首次 SG2260E PCIe
TPU-Kernel FP32 add 已通过数值，产生 5 个非空 raw 文件并解码出 16 个 ns 事件；旧逻辑却在
紧接发射的一次采样看到 `9%` 后停止。它位于
`research/artifacts/2026-09-08/final-137c85d5/core-sg2260e-pcie/`，仅作为问题定位证据，
不计入 canonical 通过数。

`6b6772be` 修复后的 canonical core 首例真实观察到 `10% -> 10% -> 0% -> 0%`，在同一锁和
总 deadline 内正确等待到连续两个零样本后通过。高层 PCIe 的 51 个 postflight 也全部完成
稳定空闲判定。

## 8. 当前边界与下一步

1. 为 RV Tensor 补齐 reduction、rsqrt/exp/sigmoid 和复合算子所需语义，再逐项开放
   RMSNorm、RoPE、SwiGLU、FlashAttention；在此之前继续 fail-closed。
2. 实现 tail predicate/masked DMA、动态 shape 和 causal mask，并为每项增加独立 oracle。
3. 在可用 BM1690 板卡上执行同 revision PCIe 矩阵；SG2260E 结果不能外推。
4. 如需性能结论，另建无诊断 recorder 的 warm-up/repeat benchmark，报告统计分布；不得把
   本报告的单次指令 timing 改写为稳态性能。
5. 增加四核分片和跨核依赖验证。当前正确性矩阵不证明 SG2260E 四核并行效率。
