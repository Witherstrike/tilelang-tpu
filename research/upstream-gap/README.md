# TileLang-TPU 现状与上游能力差距

## 1. 结论

截至 2026-09-04，本分支已经完成 SG2260E 的最小双后端数值闭环：同一组
`T.ppl_*` 前端表达先降为 `tl.tpu.*`，再选择 TPU-Kernel 或 RV Tensor（RVT）
codegen。实现覆盖 `copy`、零 `fill`、FP16/BF16 GEMM 以及 FP16/BF16/FP32 的
add/sub/mul/div；本轮数值实验实际验证的是 FP16 GEMM 和 FP32
add/sub/mul/div，它们已通过 SG2260E CModel 与 PCIe，BM1690 的 TPU-Kernel
CModel 回归也通过。

这仍不是“TileLang 二层 op 已完整支持 TPU”。当前完成的是安全、可验证的核心竖切；
主要缺口依次是：标准二层 API 收敛、通用 shape/dtype/tail 规划、reduction 与组合算子、
依赖正确的软件流水、显式多核计划，以及迁移到上游注册式 backend 架构。

## 2. 当前边界

### 2.1 三条独立选择轴

| 选择轴 | 规范取值 | 作用 |
| --- | --- | --- |
| 芯片 | `bm1690`、`sg2260e` | PPL arch、编译宏、物理核数和能力集。 |
| 编程模型 | `tpukernel`、`rv` | 指令 ABI、descriptor 与生命周期。BM1690 只允许前者。 |
| 运行时 | `cmodel`、`pcie` | 模拟器或真实板卡；不改变 op 语义。 |

`TPUChipSpec` 是唯一能力表：BM1690 为 `tpub_7_1`/8 核，SG2260E 为
`tpub_7_1_e`/4 核；二者共用本项目所需的 TPUv7 LMEM 几何。物理核数只描述拓扑，
当前 host 模板仍为 `core_num=1`，不能把 CModel 初始化 4/8 核写成多核工作负载。

旧 `device_mode="atomic"` 只是 TPU-Kernel 的历史误称。入口暂作弃用兼容，内部 target、
缓存身份、codegen 和诊断统一使用 `tpukernel`/`rv`。

### 2.2 当前编译竖切

```text
T.ppl_* compatibility frontend
        │
        ▼
tl.tpu.{copy,fill,gemm,add,sub,mul,div}
        │
        ├─ common parsing / shape+dtype contract / AddressAssign
        ├─ codegen_tpukernel.cc ─► tpu_kernel.h
        └─ codegen_rv.cc        ─► rvt_api.h
                                      │
                                      ▼
                         PPL 1.7 ─► CModel / PCIe
```

共同层不再以 PPL 作为编程模型名称；源码已拆为 `codegen_tpu_common`、
`codegen_tpukernel` 和 `codegen_rv`。规范 FFI 是 `target.build.tilelang_tpu`，旧
`target.build.tilelang_ppl` 仅为兼容别名。`ppl_layout.py` 中的 PPL 指厂商 SDK，且只
接受 PPL 1.7 `deps/` 布局，不再区分或探测旧工具链。

### 2.3 已证明的能力

| 能力 | SG2260E TPU-Kernel | SG2260E RV | BM1690 TPU-Kernel |
| --- | --- | --- | --- |
| add/sub/mul/div，FP32 32×32 tile | CModel + PCIe 数值通过 | CModel + PCIe 数值通过 | CModel 数值通过 |
| 64×64 FP16 matmul，32×32 tile，FP32 累加 | CModel + PCIe 数值通过 | CModel + PCIe 数值通过 | CModel 数值通过 |
| PCIe 指令 profiling | 每个逐元素 16 条、matmul 36 条 timing | 同左 | 未上板 |
| 四/八核工作划分 | 未实现（当前单核） | 未实现（当前单核） | 未实现（当前单核） |

完整矩阵、容差和初次 PCIe software-pipeline 冒险见
`research/rv-backend/TESTING.md`。源码可编译、CModel 可运行和 PCIe 数值通过是三种
不同强度的证据，本文不从一个层级外推另一个层级。

## 3. 当前核心 op 契约

| 中立 op | 已实现语义 | 明确拒绝的范围 |
| --- | --- | --- |
| `tl.tpu.copy` | 等形静态 region；S2L/L2S/S2S/L2L；RV 用 GR/TR + FREE stride。 | 越界、两端 extent 不同、global 跨 dtype；TPU-Kernel FP16↔BF16 直接 cast；RV 非浮点 f2f。 |
| `tl.tpu.fill` | TPU-Kernel 浮点常量；RV typed CR + `rvt_cp` 的零填充。 | RV 非零常量及未建模 dtype。 |
| `tl.tpu.gemm` | local N=H=1；FP16/BF16 A/B；NN/NT；显式 overwrite/accumulate；M/N/K ≤ 65535。 | `transpose_A`、TPU-Kernel NT accumulate、batched/4-D 冒充矩阵、未注册 dtype。 |
| `tl.tpu.add/sub/mul/div` | local N=H=1；同 dtype FP16/BF16/FP32；等形或 rhs W broadcast。 | global operand、一般 broadcast、混合 dtype；未定义的 NaN/Inf/零除完整语义。 |

RV descriptor 由编译器拥有：CR=R0–R7、TR=R8–R31、GR=R32–R39，精度使用
`PRECISION/FP8TYPE`，生命周期为 `rvt_kernel_start → lanemask/body → rvt_sync_i`。
raw `rvt_*` 仍是专家 escape hatch；它与 compiler-owned `tl.tpu.*` descriptor/lifecycle
不能混用。TPU-Kernel 则独立使用 `tpu_initialize → body → tpu_poll`。

## 4. 二层 op 与功能差距

优先级含义：P0 是扩大功能前必须解决的正确性/架构边界；P1 建立常用模型的基础算子；
P2 面向复杂模型与性能；P3 是规模化和上游化。

| 层级/能力 | 当前状态 | Why | 具体措施与验收 |
| --- | --- | --- | --- |
| 标准 `T.copy/T.fill/T.gemm` | 核心语义仍由 `T.ppl_*` 兼容入口暴露。 | 公共 API 带厂商名会阻止同一程序跨后端，也让 op contract 与实现耦合。 | **P0**：令标准 TileOp 降到同一 `tl.tpu.*`；`ppl_*` 变薄兼容别名。两种入口的 TIR 与数值 golden 必须一致。 |
| capability/op spec | shape、dtype、effect 检查仍分布在 frontend、AddressAssign、codegen。 | 新增 op 容易出现地址分析认为只写、codegen 实际读写等漂移。 | **P0**：建立 `TpuOpSpec`，集中声明 operand role、scope、effect、dtype/layout、workspace、backend selector；所有 pass 消费同一注册。 |
| tail 与动态 shape | 当前核心用例要求静态整 tile。 | 隐式越界 DMA 是板端高风险问题，不能靠 CModel 偶然通过。 | **P0**：引入 tile validity/mask 或显式 pad/crop planner；先覆盖非整除 M/N/K，再允许动态符号。每个 tail 有正例与越界负例。 |
| cast/量化/混合 dtype | 只覆盖 matmul 输出 FP32→FP16 和受限 copy cast。 | 推理模型需要 BF16、INT8/INT4、scale/zero-point 和确定舍入。 | **P1**：按两套 ISA 建 capability matrix，分别映射 f2f/i2i/i2f/f2i、round/saturate/quant；未列组合编译期失败。 |
| 一般 elementwise | 仅二元同 dtype与受限 W broadcast。 | bias、mask、门控和残差需要 scalar、row/column、compare/select。 | **P1**：统一 broadcast axis 映射；加入常量 add/mul、min/max、compare、select、clamp；定义 in-place、NaN/Inf 与除零行为。 |
| reduction | 历史 TPU-Kernel 有受限 `ppl.reduce_sum/max` handler，未进入中立契约；RV 无 lowering。 | softmax、norm、attention 与 loss 都依赖可靠 reduction。 | **P1**：先连续轴 sum/max/min，明确 init/clear、accumulator dtype、tail；再多轴与跨 tile。CModel/PCIe 分阶段验证。 |
| 激活与数学函数 | 历史 TPU-Kernel 有 exp/sigmoid/rsqrt 等专用 handler；RV 未统一。 | 这些实现常依赖 table/workspace/近似精度，不能仅按名字认为等价。 | **P1**：为 exp/exp2/rsqrt/sigmoid/GELU/SiLU 建误差契约和 workspace planner；逐后端注册实现或组合 lowering。 |
| normalization/softmax | 未形成标准二层实现。 | 是 attention 和现代网络的高频组合，能检验 reduction+elementwise+数值稳定性。 | **P1**：先 RMSNorm/LayerNorm，再稳定 softmax；明确 FP32 累加、epsilon、最大值归约和尾块。 |
| reshape/view/transpose | view 类前端存在，但 TPU descriptor/layout 语义未系统化。 | 逻辑 view 与真实 DMA transpose 混淆会产生错误 stride。 | **P1**：区分零成本 view 与物化 layout transform；验证 global/local stride、别名与地址区间。 |
| gather/topk/sort | 仅历史 TPU-Kernel 特殊 handler，未做双后端契约。 | 索引边界、稳定排序、workspace 与 index dtype 都影响正确性。 | **P2**：先 gather，再 top-k/sort；定义越界策略、stable 语义、K 限制、workspace，并增加随机/重复值测试。 |
| convolution/im2col | 标准 `c2d_im2col` 存在，TPU 端未闭环。 | 需要 layout、padding、dilation、DMA 和 GEMM 协同，不是单条 intrinsic。 | **P2**：先显式 im2col+GEMM 参考实现，再按 capability 融合；验证 NCHW/NHWC 与非对齐边界。 |
| batch GEMM/attention | 当前 GEMM 限二维 local tile。 | FlashAttention 等需要 batch/head、online reduction、mask 与流水。 | **P2**：先 batched GEMM descriptor/partition，再组合 softmax；复杂 attention 只在基础 op 稳定后接入。 |
| software pipeline/async | 核心正确性示例使用 `T.serial`。初次 PCIe 证明 `num_stages=1` 可让 DMA 与直接消费者 GEMM 竞态。 | CModel 会剥掉 TPU parallel marker，不能验证真实依赖；错误流水比无优化更危险。 | **P0/P2**：先建立 DMA/compute token、读写 hazard 和 buffer versioning；做 serial-vs-pipeline 等价测试后才开放 overlap。 |
| 多核/Persistent | 物理 4/8 核已建模，launch 仍单核。 | 直接把 `core_num` 改为 4/8 会重复整个 grid 并竞写输出。 | **P2/P3**：引入 `LaunchPlan`：per-core range、offset、output ownership、同步/归约、错误回收；先 elementwise 无冲突分片，再 GEMM/reduction。 |
| profiling/benchmark/autotune | CModel raw 与 PCIe 逐指令 timing 已打通。 | recorder 的 host wall time带开销，不能直接供 autotune 或性能回归。 | **P1/P2**：artifact manifest 记录 target/SDK/input；profiling 用于定位，另建 warmup/repeat/统计 benchmark；最后才让 autotuner消费。 |
| 持久缓存/导出 | TPU 预编译 artifact 仍故意 fail-closed。 | 私有 `libkernel.so`、SDK/runtime identity 与 device ABI 未打包时，cache hit 可能绕过加载门禁。 | **P3**：建立包含 resolved target、pass、SDK/toolchain/runtime、二进制依赖的 manifest；加载前逐项验证并在新进程 rehydrate。 |

## 5. 上游 TileLang 所需重构

上游 TileLang 当前的
[Backend Layout](https://github.com/tile-ai/tilelang/blob/main/tilelang/backend/README.md)
已明确：通用 `tilelang/backend` 只保留注册与共享设施，target 特定的 pipeline、host/device
codegen、op 与 intrinsic 应由 `tilelang/<backend>` 和 `src/<backend>` 拥有。TPU 应按这一
边界演进，而不是继续在 `engine/lower.py` 中累积条件分支。

建议目标结构：

```text
tilelang/tpu/
  language.py       # common language + 明确的 TPU 专属扩展
  target.py         # chip/model/runtime normalizer 与 capability
  pipeline.py       # 完整且可审阅的 TPU pass 顺序
  codegen.py        # host/device codegen 注册
  op/               # TpuOpSpec 与标准 TileOp lowering
  intrinsics/       # 专属 TPU-Kernel/RV escape hatch
  toolchain.py      # PPL 1.7 resolver/build hook

src/tpu/
  codegen/          # common / tpukernel / rv
  op/               # native op verification/lowering
  memory/           # TPUv7 LMEM profile 与 AddressAssign
```

迁移顺序：

1. **先注册、不改语义**：把 target normalization、pass list、`target.build.tilelang_tpu`
   和 execution adapter 接到上游 registry；保持现有数值矩阵不变。
2. **再统一 op contract**：标准 `T.copy/fill/gemm/reduce` 与兼容 `T.ppl_*` 都进入
   `TpuOpSpec`；删除 AddressAssign/codegen 的重复字符串知识。
3. **最后拆 native 目录**：将本轮的三份 codegen 和 TPUv7 memory planner 移入
   `src/tpu`，让通用 engine 不再知道 PPL、RVT 或 SG2260E。
4. **可上游与私有内容分层**：target/backend hook、op contract 和无 SDK 单测适合上游；
   PPL 路径、专有头文件、固件/runtime 与板端安全策略留在可选 TPU adapter。

每一步都以“同一前端在 SG 双后端与 BM 基线上结果不变”为验收，避免一次大迁移同时改变
目录、pass 顺序和硬件语义。

## 6. TileLang-Ascend 的参考价值

[tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) 当前示例覆盖 GEMM/Batch GEMM、
elementwise、attention、softmax、normalization、activation、reduce、sort、convolution、
loss 和 dispatch/combine；其
[路线图](https://github.com/tile-ai/tilelang-ascend/issues/3) 还将自动同步插入、
Cube/Vector 分离、`T.Pipelined`、`T.Parallel`、tail 和 persistent 多核分别跟踪。

应借鉴的是工程分层和验收顺序：

| Ascend 经验 | TPU 对应措施 | 不应照搬 |
| --- | --- | --- |
| 计算单元自动分工 | 用 capability/op spec 选择 TPU-Kernel 或 RV，并记录 fallback。 | Cube/Vector 名称与指令语义。 |
| 自动同步与 pipeline | 建立 TPU DMA/BDC/RV 的依赖 token 与 hazard verifier。 | Ascend flag/barrier 编号和可见性规则。 |
| `T.Parallel`/Persistent | 用显式 `LaunchPlan` 做 SG 4 核、BM 8 核分片。 | Ascend 核数、任务队列和默认调度。 |
| 以复杂 op 检验基础层 | 用 RMSNorm/softmax/attention 检验 reduction、broadcast、tail。 | 直接复制其 kernel 或 layout。 |
| examples + batch regression | 每个支持声明必须进入 chip×backend×runtime 矩阵。 | 只以示例可编译替代数值/板端证据。 |

Ascend 自身仍把透明单元分离、自动 tail、persistent、多算子覆盖和性能回归列为持续工作；
因此它是后端分层参照，不是“另一 NPU 已完整解决”的证据。

## 7. 分阶段任务清单

### P0：守住正确性边界

1. 建立 `TpuOpSpec` 单点语义源，消除 effect/shape/dtype 的重复分发。
2. 让标准 `T.copy/fill/gemm` 与 `T.ppl_*` 生成同一中立 IR。
3. 实现静态 tail/pad 方案；在此前继续拒绝可能越界的 region。
4. 对 software pipeline 加 hazard verifier；不能验证依赖时编译期拒绝或明确串行化。
5. 保持 PCIe fresh process、双授权、device id、父死亡/进程组和 bounded kill/drain。

### P1：形成模型基础算子层

1. reduction sum/max/min 与 FP32 accumulator。
2. scalar/general broadcast、compare/select、cast/quant capability matrix。
3. exp/rsqrt/activation，再组合 RMSNorm、LayerNorm、softmax。
4. 为每个 op 建 SG TPU-Kernel/RV CModel 数值测试；只把通过项逐个推进 PCIe。

### P2：性能与复杂算子

1. dependency-correct DMA/compute pipeline 与 buffer versioning。
2. gather/topk/sort、transpose/im2col、Batch GEMM/conv。
3. 从无写冲突 elementwise 开始实现 SG 4 核/BM 8 核 `LaunchPlan`。
4. 在稳定基础 op 上实现 attention/dispatch-combine 等组合 kernel。

### P3：上游化与生产工程

1. 迁入注册式 `tilelang/tpu`、`src/tpu` 后端竖切。
2. 建 artifact manifest、持久缓存与新进程 rehydrate 测试。
3. 分离 instruction profiling、低扰动 benchmark 和 autotune 数据模型。
4. 建 chip×programming-model×runtime×op×dtype/tail 的长期 CI/板端矩阵。

## 8. 判定原则

- “支持”必须同时给出前端契约、生成指令、数值 oracle 和注明的 runtime/chip 范围。
- CModel 是 PCIe 的前置条件，不是并行/时序语义的替代品。
- 不支持的 shape、dtype、layout 或生命周期必须在编译期失败，不能静默 fallback。
- SG2260E 的 4 核是硬件事实；在 `LaunchPlan` 完成前，系统只宣称单核工作负载正确。
- 单次 profiling duration 用于定位指令，不用于宣称性能优劣。
