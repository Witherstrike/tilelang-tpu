# TileLang-TPU 现状、上游差距与演进路线

## 1. 当前结论

当前分支已经建立可审阅的 TPU 后端竖切：

- 编译选择拆为 `chip × programming_model`，运行选择独立为 `runtime_mode`；
- BM1690 使用 TPU-Kernel；SG2260E 可选择 TPU-Kernel 或 RV Tensor；
- 核心前端先降为 `tl.tpu.*`，再选择 `codegen_tpukernel` 或 `codegen_rv`；
- TPU-Kernel 专属能力使用 `tl.tpukernel.*`，不会在 RV target 上静默回退；
- target pass、残余 IR verifier、AddressAssign、codegen 与 PPL 1.7 toolchain 均按同一完整 target 工作；
- 未建模的 vector、同步、dtype、shape、ABI 或芯片能力在编译期失败。

数值证据已经从“核心示例可运行”扩展到完整的非 FP8 TPU-Kernel 基线、独立 FP8 矩阵和 RV 核心竖切。当前 canonical CModel 基线统一绑定实现提交 `44a6fc2ab6e8ca60569853fd781e21bfa0b78335`：[TPU-Kernel 288/288](../artifacts/2026-09-07/tpukernel-cmodel-44a6fc2/summary.json)、[FP8 76/76](../artifacts/2026-09-07/fp8-cmodel-44a6fc2/summary.json) 和 [三组核心矩阵 27/27](../artifacts/2026-09-07/core-cmodel-44a6fc2/summary.json)。最后一项对三个合法 target 各跑 9 项，其中 SG2260E/RV 包含 FP32 四则、FP16 GEMM 以及 FP16/FP32 的 local-roundtrip、global-to-global copy，共 9/9。三个 runner 都自行记录该 revision，且确认实现工作区干净。

| 组合 | CModel | PCIe |
| --- | --- | --- |
| BM1690 / TPU-Kernel | `44a6fc2` 基础矩阵 147/147；FP8 38/38；核心 9/9 | 未验证 |
| SG2260E / TPU-Kernel | `44a6fc2` 基础矩阵 141/141；FP8 38/38；核心 9/9 | 历史基础矩阵 140/140；当前提交未验证 |
| SG2260E / RV | `44a6fc2` 核心 9/9，含四则、GEMM，以及 FP16/FP32 × {local-roundtrip、S2S} 四条 copy case；local 覆盖 G2L/L2L/L2S | 历史核心五项；当前提交未验证 |

SG2260E/TPU-Kernel 的历史板端 final 按 core 53、extended 15、reductions 72 三批执行，均首错停止且没有 timeout、retry 或 device fault。profiling 的单次 dispatch 与 recorder/raw 采集子阶段已验证；会话内因缺 decoder 以 `complete=false` 结束，随后仅对既有 trace 在隔离的 `bigTpuProfile==0.3.5` 环境解码出 36 条有效事件。最近一次有界 `tpu-smi --noloop --json_format` 前置检查返回 `status=Fault`、`tpu_util=100%`，因此没有启动当前实现的板端算子。契约把旧板端范围标为不具授权作用的 `historical_passed`，当前没有任何 PCIe stage 为 `passed`；生产环境也不会自动安装 decoder。

这些结果仍不等于“TileLang 二层算子已完整支持 TPU”。当前主要差距已经从基础 TPU-Kernel 指令连通和 SG 板端数值验证，转向统一 op spec、标准二层 API、tail/dynamic shape、FP8 板端验证、RV 能力扩展、多核与依赖安全流水，以及可持续的上游后端边界。

### 1.1 上游观测基线

本轮于 2026-09-07 直接核对官方仓库；这里的 commit 只用于说明调研时点，不参与本地 PPL/手册一致性校验：

| 项目 | 调研快照 | 与本后端直接相关的事实 |
| --- | --- | --- |
| [TileLang](https://github.com/tile-ai/tilelang) | `main@11ec2397` | v0.1.13 已引入 multi-backend language dialect、backend CodeGen registry、source-aware diagnostics 与 TIRX 演进；TPU 继续把分支堆进通用 `engine` 会扩大后续 rebase 成本 |
| [TileLang v0.1.13 说明](https://github.com/tile-ai/tilelang/discussions/2848) | discussion #2848 | 上游同时修复 buffer offset、effect/liveness、跨 target call 等问题，说明 descriptor/effect/pass 顺序必须作为可验证契约，而非 emitter 局部约定 |
| [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend) | default `ascendc_pto@9da16d6` | 已有 GEMM/Batch GEMM、elementwise、softmax/norm/activation/reduce/sort/conv/attention/dispatch-combine，并把 auto vectorization、Cube/Vector scope、同步、pipeline、buffer reuse、跨核通信分别建模 |
| [TileOPs](https://github.com/tile-ai/TileOPs) | `main@001ef39` | 明确区分 L2 调用契约与 L1 kernel，并以 manifest 驱动 reference、signature、workload、source、test、benchmark 和 validator，适合作为 TPU op contract 的上游化方向 |
| [SOPHGO PPL](https://github.com/sophgo/PPL/releases/tag/v1.7.122) | latest release `v1.7.122` | 本机使用同版本号的 PPL 1.7 发布布局；公开 README 尚未把 SG2260E 列为支持芯片，因此 SG 能力仍以本机 `tpub_7_1_e` 头文件、真实编译和分层实测为准 |

由此得到的工程判断是：短期保持当前 fail-closed 竖切，避免在功能尚未闭合时追随上游大迁移；中期把 target resolver、pass hook、semantic op spec 和 artifact manifest 接入上游 registry/dialect 边界；算子层按 TileOPs 的 L2/L1 契约组织，而不是把 Ascend 的 Cube/Vector 名称或 GPU 的 warp/barrier 语义直接复制到 TPU。

## 2. 现有能力边界

### 2.1 选择模型

| 维度 | 取值 | 当前约束 |
| --- | --- | --- |
| 芯片 | `bm1690`、`sg2260e` | BM 为 `tpub_7_1`/8 核；SG 为 `tpub_7_1_e`/4 核 |
| 编程模型 | `tpukernel`、`rv` | BM 只允许 TPU-Kernel；SG 允许两者 |
| 运行时 | `cmodel`、`pcie` | 不参与指令选择；PCIe 额外受加载门禁和进程监管 |

BM1690 与 SG2260E 在当前 allocator 所需的 TPUv7 LMEM 几何上共用 profile；PPL arch、编译宏、核数与能力集仍分别声明。SG2260E topk 是重要反例：头文件存在 HAU symbol，但 `tpub_7_1_e` runtime 不实现该原语，因此 SG codegen 直接拒绝，而 BM CModel 已验证。

### 2.2 ABI 与 pass 边界

```text
T.ppl_* frontend
  ├─ tl.tpu.{copy,fill,gemm,add,sub,mul,div}
  │    ├─ TPU-Kernel emitter
  │    └─ RV emitter
  └─ tl.tpukernel.{scalar,exp,sigmoid,reduce,rsqrt,gather,topk,rope}
       └─ TPU-Kernel emitter only
```

TPU pipeline 当前采用保守语义：不运行 vectorization、software-pipeline injection 和 `StorageRewrite`。这不是性能目标，而是正确性边界；在 residual vector IR、DMA/compute 依赖和结构化 buffer 生命周期尚未建模时，这些 pass 可能生成语法成立但语义错误的程序。

AddressAssign 为封闭 semantic op 集建模 read/write/read-write effect。GEMM 的 `accumulate` 是显式 ABI 位，同时决定 C 是 write 还是 read-write。contract verifier 在 target pass 前后执行，防止前端或 pass 引入另一编程模型、未知 extern、GPU barrier 或 vector residual IR。

每个 typed semantic buffer 参数都以 `tl.region(BufferLoad, access_mask, logical_extents)` 进入 native codegen。非 copy op 要求 whole-buffer region；copy 才允许显式子区间。裸 `tir.tvm_access_ptr` 兼容入口已删除，因为它不足以保留 view/reshape 的 logical descriptor。descriptor 等价的 presentation alias 可作为唯一表示；rank/shape/dtype/scope 改变、重复 allocation owner、越界 region 或错误 mask 均 fail-closed。

### 2.3 当前 op 现状

| 族 | 已验证范围 | 主要未覆盖 |
| --- | --- | --- |
| copy/cast | TPU-Kernel：FP16/BF16/FP32 与六种整数同 dtype local roundtrip/S2S、`(2,3,17)` FP32 rank-3、FP32 相关本地 cast、FP8 local roundtrip/S2S；RV：FP16/FP32 的 G2L/L2L/L2S/S2S | FP16↔BF16、更多混合 dtype、rank-3 PCIe、RV 其他 dtype |
| fill | FP16/BF16/FP32 非零常量；FP32 零值；FP8 零值 | FP16/BF16 独立零 case、FP8 非零 |
| GEMM | TPU-Kernel：FP16/BF16 NN overwrite/accumulate、NT overwrite；FP8 NN/NT overwrite 与 accumulate。RV：FP16 NN accumulate 已有数值证据，基础浮点 NT accumulate 仅有源码选择证据 | batch、transpose-A、基础浮点 FP32 overwrite（NN/NT）数值验证、RV BF16/NT 数值验证、TPU-Kernel 基础浮点 NT accumulate、FP8 C、tail |
| add/sub/mul/div | 三种浮点等形；FP32 W broadcast；FP8 等形及 W-broadcast add/sub/mul | FP16/BF16 broadcast、FP8 div、一般 broadcast |
| scalar | FP16/BF16/FP32/E4M3/E5M2 add/mul；FP8 默认非饱和 | sub/div、动态 scalar、可选 saturation |
| exp/sigmoid | 三种浮点 | RV、更多函数、误差域扩展 |
| reduction | FP16/BF16/FP32 row sum/max，12 个 EU 边界 width（含 63/64/65） | RV、min/arg、跨 tile/多轴 |
| rsqrt | FP16/BF16/FP32 通用原语 | FP8、RV |
| gather/rope | FP16/BF16/FP32/E4M3/E5M2；FP8 gather 精确保持 encoded bytes，RoPE 为偶宽 interleaved add | RV、整数 gather payload、一般 layout |
| topk | BM 的 FP32/INT32/UINT32 双向、K-sized 输出及稳定重复键；SG 编译期不适用 | K=1/K=length、长度上限、BM PCIe；RV |

FP8 的“手册/头文件声明、公开 frontend、TileLang codegen、CModel、PCIe”分别记录。最终矩阵 76/76，即两芯片 × 两格式 × 19 个公开 case；它只提升所列 selector。早先 scalar 两芯片 exit 139 已定位为 direct mixed-precision API 的非法 dtype tuple；合法 PPL/TileLang 路径均使用 cast 后的通用 add_C/mul_C，因此该崩溃不构成硬件负向证据。

RV 的 cross-dtype copy 也必须按方向管理：当前 FP16→BF16 只有 `rvt_cvt_f2f` 生成源码证据，数值层仍为 `unverified`；BF16→FP16 尚无精确 codegen 证据。两者不能因共用一个 emitter 类型分支而合并提升。

SG2260E/TPU-Kernel 的非 FP8 列表曾在真实芯片按同一 case registry 通过 140/140，其中包括基础浮点 scalar、exp/sigmoid/rsqrt、gather/rope、FP32 W broadcast 及十二个 reduction 边界 width。由于当前实现此后修改了 region ABI、copy lowering、runner/profiling 监管和证据契约，这批结果仅用于确定回归范围，不能关闭 `44a6fc2` 的板端门禁；BM 板端、FP8 板端和更宽 RV selector 同样未验证。

## 3. 差距排序原则

后续任务按四项标准排序：

1. **正确性外溢范围**：一个缺口是否会使多个 op 产生错误地址、依赖或静默 fallback；
2. **模型复用度**：能力是否被 normalization、softmax、attention 等大量上层 op 复用；
3. **验证成本与板端风险**：能否先在 source/CModel 证明，再以小步 PCIe 验证；
4. **上游可维护性**：新实现是否减少 target 特判和重复契约，而不是继续增加分叉。

因此先解决 op spec、tail 和依赖模型，再扩复杂算子；先建立通用组合能力，再写单个模型专用 kernel。

## 4. P0：编译器正确性与架构收敛

### 4.1 注册式 `TpuOpSpec`

**现状**：operand role、dtype/shape、effect 和失败策略分布在 frontend、residual verifier、AddressAssign 与两个 emitter 中。

**Why**：新增 op 时，任一层漏改都可能产生“前端接受、地址分析乐观、codegen 读取额外 buffer”的不一致；这是板端错误的系统性来源。

**措施**：

1. 定义 `TpuOpSpec`：semantic name、frontend alias、operand role/scope、effect、dtype/layout constraint、workspace、applicable programming model 与 emitter key；
2. 由该表生成/驱动 frontend guard、verifier allowlist、AddressAssign effect 和 codegen dispatch；
3. chip-specific capability 作为 spec 的 predicate，不在 emitter 内散落字符串判断；
4. 合法性按完整 `chip × programming model × dtype × layout × attributes` selector 表达，保留 TPU-Kernel FP8 NT accumulation、TPU-Kernel 基础浮点 NT rejection 与 RV 基础浮点 NT source-only support 这样的精确分支；
5. 为每个 spec 自动生成 positive/negative source test 与 contract selector 骨架。

**验收**：删除任一 op 的 emitter 注册后，编译在统一诊断处失败；effect 与 emitter operand 数自动一致；机器契约可从 spec 检查引用闭包。

### 4.2 Python/native target 能力单源化

**现状**：Python `TPU_CHIP_SPECS` 是主能力源，native build 入口镜像一份合法 chip/model 组合。

**Why**：两份手写表可能在加入芯片或编程模型时漂移，使 Python 接受而 native 拒绝，或相反。

**措施**：采用一份可编译的数据描述生成 Python 与 C++ 常量；生成内容包含 PPL arch、宏、核数、编程模型和 chip feature flags。

**验收**：构建期比较生成表；所有合法/非法 target 组合在 Python 与 native 层得到一致结果。

### 4.3 tail、动态 shape 与 alias

**现状**：已验证 case 均为静态、规则 tile；copy/gemm/reduction 在有限静态范围 fail-closed。当前所有 TPUv7/PPL `dim4` 单维必须是编译期整数且位于 `[1,65535]`，exp/sigmoid 另有 `H*W<=65535`，reduction 还校验 EU 对齐后的派生宽度。typed semantic ABI 已保留 logical region，并区分 descriptor 等价 presentation alias 与真正的 shape/layout/ownership 变化；运行时跨参数重叠仍只按各 op 的显式 alias 契约判断。

**Why**：隐式越界 DMA 或错误 stride 在 PCIe 上可能卡住设备；没有统一 tail 语义就无法安全扩大 shape。

**措施**：

1. 先实现静态非整除 tile 的 validity + pad/crop planner；
2. 每种 DMA/compute op 明确 mask 能力，没有硬件 mask 时分配受控 padding；
3. 用 analyzer 证明 region bounds，并建立 alias/overlap 规则；
4. 静态 tail 稳定后再支持符号 extent 和 runtime guard。

**验收**：M/N/K 与 W 在 EU/tile 边界前后都有数值正例；越界、重叠与无法证明的 region 均为编译期负例。

### 4.4 依赖安全的 pass 模型

**现状**：TPU pipeline 禁用 vectorization 和 software pipeline；PCIe 已证明未经建模的 DMA/GEMM overlap 会产生数值错误。

**Why**：直接复用 GPU pass 的 barrier、lane 或 async 语义会把“优化”变成错误程序。

**措施**：

1. 定义 DMA/BDC/RV command token、buffer version、producer-consumer 与 barrier 可见域；
2. 加 hazard verifier，证明 RAW/WAR/WAW 安全后才允许 overlap；
3. 为 residual vector IR 定义 lane、Ramp、load/store 和 reduction 规则，再逐项开放 vector pass；
4. serial 与 pipeline 版本必须做 CModel/PCIe 等价测试。

**验收**：任何缺 token 或冲突 buffer 的 pipeline 在编译期失败；通过的 pipeline 有可解释 trace 和数值等价证据。

### 4.5 标准二层 API

**现状**：中立 IR 已形成，但用户入口仍以 `T.ppl_*` 为主。

**Why**：厂商命名进入公共 TileLang 层会阻碍同一程序在 CUDA/HIP/TPU/Ascend 后端间复用，也使上游难以接受。

**措施**：让标准 `T.copy/T.fill/T.gemm` 与二层 elementwise/reduce op 降到同一 semantic registry；TPU 特有参数通过 target capability 或明确的 extension 表达，不复制一套 API。

**验收**：标准入口与当前入口生成等价 semantic IR，并在三种有效 target 上保持相同数值结果；完成迁移后只保留一个公共语义入口。

## 5. P1：模型基础算子

### 5.1 cast、量化与 dtype contract

**Why**：推理需要 BF16、FP8、INT8/INT4、scale/zero-point、舍入与饱和；仅看指令名无法保证两条 ISA 语义一致。

**措施**：按 `src dtype × dst dtype × round × saturate × scope × backend` 建矩阵；实现 f2f/i2i/i2f/f2i 与 quant/dequant；未列组合编译期拒绝。FP8 E4M3/E5M2 永远分开记录。

### 5.2 一般 elementwise 与 broadcast

**Why**：bias、mask、门控和残差需要 scalar、row/column/batch broadcast、compare/select/clamp。

**措施**：先验证 FP16/BF16 W broadcast，再抽象 broadcast-axis/zero-stride planner；加入 min/max、compare、select、clamp 与明确的 in-place/NaN/Inf/零除策略。

### 5.3 reduction 的中立化与扩展

**Why**：sum/max 已在 TPU-Kernel CModel 稳定，但仍是 `tl.tpukernel.*`；RV、min、arg、跨 tile 与 FP32 accumulator 是 softmax/norm 的前置。

**措施**：定义 `tl.tpu.reduce` 的 axis、init、accumulator dtype、workspace 和 tail 语义；TPU-Kernel复用现有 composite，RV 按 ISA 实现；先连续轴，再多轴/跨 tile。

### 5.4 数学函数与 activation

**Why**：exp/sigmoid/rsqrt 的 workspace、输入域和近似误差都属于算子契约；不同后端不能只按同名函数视为等价。

**措施**：建立 exp/exp2/log/rsqrt/sigmoid/GELU/SiLU 的误差 envelope 和 workspace planner；允许“原语实现”或“已注册组合 lowering”，但都必须按 backend 独立验证。

### 5.5 normalization 与 softmax

**Why**：RMSNorm、LayerNorm 和稳定 softmax 是现代模型的基础组合，也是检验 reduction、broadcast、tail 与数值稳定性的最佳中层 workload。

**措施**：先 RMSNorm（sum-square + rsqrt + scale），再 LayerNorm 和 max-subtracted softmax；统一 FP32 accumulator、epsilon、mask 与尾块策略。

### 5.6 layout/view/transpose

**Why**：逻辑 view 与真实 DMA transform 混淆会生成错误 stride；GEMM、attention 和 convolution 都依赖可靠 layout。

**措施**：区分零成本 view 与物化 transform，建立 global/local stride、contiguity、alignment 与 alias verifier；先 2D transpose，再 blocked layout。

## 6. P2：复杂算子与性能

### 6.1 显式多核 `LaunchPlan`

**Why**：SG 的 4 核和 BM 的 8 核已经建模，但直接把 `core_num` 改大只会复制整个 grid 并竞写输出。

**措施**：`LaunchPlan` 明确 per-core range、地址偏移、output ownership、跨核同步/归约和错误回收；先无写冲突 elementwise，再 GEMM，最后 reduction。

### 6.2 async pipeline

**Why**：保守串行保证正确性但无法发挥 DMA/compute overlap；它依赖 P0 的 token/hazard 模型。

**措施**：双 buffer versioning、显式 wait/signal、capacity check 和 schedule legality；profiling 只用于定位 overlap，性能结论使用无 recorder 的 warmup/repeat benchmark。

### 6.3 Batch GEMM 与 attention

**Why**：当前 GEMM 只描述 local rank-2 tile；attention 需要 batch/head、mask、online reduction 与流水。

**措施**：先 batched descriptor 与 partition，再验证 online softmax，最后组合 FlashAttention；基础 op 的 contract 未闭合前不引入专用快捷路径。

### 6.4 gather/topk/sort

**Why**：当前 gather/topk 是 TPU-Kernel 专属，且 topk 有芯片 runtime 差异。索引越界、稳定排序、重复值、workspace 和 K 上限都影响语义。

**措施**：将 gather 纳入中立 op；定义 index OOB policy。BM topk 已按真实 K-element 写入 extent 收紧输出并验证稳定重复键，下一步覆盖 K=1/K=length、长度上限和 PCIe；SG 保持拒绝，直到厂商提供有效实现；RV 只在 ISA 与算法路径明确后注册。

### 6.5 convolution 与 layout lowering

**Why**：convolution 需要 padding/dilation/layout/DMA/GEMM 协同，不是简单转发一条 intrinsic。

**措施**：先 im2col+GEMM 参考实现，覆盖 NCHW/NHWC 和非对齐边界；再以 capability 驱动融合或专用指令。

### 6.6 FP8 扩展

**Why**：当前 FP8 已形成 76/76 的公开 CModel 矩阵，包含 copy/cast/fill、dense/W-broadcast/scalar arithmetic、gather、RoPE 与 NN/NT GEMM，但 PCIe、异常值域、可选 saturation 与 RV mapping 仍未形成生产闭环。

**措施**：以板卡健康为硬前置，先在当前提交完成 TPU-Kernel matmul canary、SG 两编程模型核心矩阵和 TPU-Kernel 非 FP8 分批重验，再进入 FP8 copy/arithmetic/scalar/gather/RoPE/GEMM；之后为异常值与 saturation 建独立契约。当前 scalar 只承诺非饱和 E4M3-NaN/E5M2-infinity overflow，不能把被 PPL 丢弃的 saturation flag 暴露给用户。RV 按 descriptor、round/saturate 与 accumulator 逐项实现，不能因 ISA 文档列出 FP8 就整体开放。

## 7. P3：上游化与生产工程

### 7.1 后端目录与注册

目标结构：

```text
tilelang/tpu/
  target.py       # chip/model/runtime resolver 与 capability
  pipeline.py     # TPU pass 顺序
  language.py     # 标准 op lowering + TPU extension
  op/             # TpuOpSpec
  toolchain.py    # PPL 1.7 resolver/build

src/tpu/
  codegen/common/
  codegen/tpukernel/
  codegen/rv/
  memory/
```

**Why**：通用 engine 不应持续累积 TPU/PPL/RV 条件分支。

**措施**：先把现有行为接到 backend registry，不改语义；再以 `TpuOpSpec` 消除重复字符串；最后移动 native 目录。每一步都重跑 BM/SG CModel 基线。

### 7.2 artifact manifest 与缓存

**Why**：TPU artifact 依赖 resolved target、pass pipeline、PPL SDK、runtime 和私有 `libkernel.so`；缺少 manifest 的 cache hit 可能加载错误 ABI。

**措施**：manifest 记录编译身份、运行身份、toolchain/runtime 依赖和输入签名；新进程 rehydrate 前逐项校验，身份不同即 miss。

### 7.3 长期验证矩阵

**Why**：芯片、编程模型、runtime、dtype 与 op 组合迅速增长，手工报告无法防止状态漂移。

**措施**：

- 每次提交运行 source-only + 三种合法 target 组合的 CModel；
- PCIe 按风险分 smoke/core/extended 三层，首错停止；
- profiling 与普通数值矩阵都经 parent-death supervisor 启动 worker；外层 runner 被强杀的无孤儿进程性质由单测锁定；
- contract 与 case registry 做双向一致性检查；
- instruction profiling 与性能 benchmark 分开保存；
- tracked 报告只汇总结论，raw artifact 继续 ignored。

### 7.4 profiling 依赖与性能判定

**现状**：SG2260E 硬件 dispatch 与 recorder raw 采集成功；受监管会话内没有可用 decoder，因而正确报告 unavailable。随后以临时隔离安装的 vendor package 对同一 raw 文件离线解码，得到 36 条具有有效 ns 区间的 BDC/GDMA 事件，且未再次下发板卡。

**Why**：若框架静默安装或把 decoder 缺失当成数值失败，会破坏离线构建、依赖可审计性和板端安全；单次 recorder 时长也不能代替稳定性能 benchmark。

**措施**：将 raw capture、decode 和 benchmark 保持为三个独立阶段；生产端只探测显式配置的兼容 decoder，记录版本与解析状态，不自动安装；性能比较另用无 recorder 的 warmup/repeat 流程，并保存环境与统计量。

## 8. 对 TileLang 与 TileLang-Ascend 的借鉴

上游 TileLang 需要提供的是后端注册边界、标准二层 op 和 target-specific pipeline hook；TPU 不应要求通用 GPU pass 理解 PPL/RV。适合上游的内容包括 target resolver 接口、`TpuOpSpec` 形态、无 SDK 的 verifier/source tests 与通用 artifact manifest；PPL 路径、专有头文件、固件/runtime 和 PCIe 安全门禁留在可选 TPU adapter。

TileLang-Ascend 的价值在于验证工程分层：计算单元选择、自动同步、pipeline、tail、persistent 多核和复杂 op 应是可分别审阅的层，而不是一个“大而全” lowering。可借鉴注册方式与验收顺序，不能复制其 Cube/Vector 名称、barrier 规则、核数或 layout。

建议用以下组合 workload 逐层验收：

1. RMSNorm 检验 reduction + rsqrt + broadcast；
2. stable softmax 检验 max/sum + exp + tail；
3. Batch GEMM 检验 descriptor 与多核 partition；
4. attention 检验前述能力与依赖安全 pipeline；
5. MoE dispatch/combine 检验 gather/scatter、sort 与多核 ownership。

## 9. 可执行路线图

| 阶段 | 任务 | 退出条件 |
| --- | --- | --- |
| P0-A | `TpuOpSpec` + target 表单源化 | verifier/effect/emitter 由同一 spec 驱动；所有负例诊断一致 |
| P0-B | static tail + hazard verifier | 非整除核心 op CModel 通过；错误 pipeline 编译期拒绝 |
| P1-A | cast/quant + broadcast + 中立 reduction | 三种有效 target 的精确 capability 矩阵建立 |
| P1-B | math + RMSNorm/softmax | 误差、workspace、tail 契约在所有适用 target 的 CModel 通过 |
| P2-A | `LaunchPlan` + dependency-safe async | SG 4 核/BM 8 核无竞写，serial/pipeline 数值等价 |
| P2-B | Batch GEMM、gather/sort、conv、attention | 基础 op 与组合 workload 均有分层证据 |
| P3 | 上游目录、manifest、CI/benchmark | 通用 engine 无 TPU 细节；cache/板端回归可复现 |

## 10. 支持判定原则

- “支持”必须绑定精确 chip、programming model、dtype、shape/layout、variant 与验证 stage；
- 头文件或 ISA 声明只能证明底层候选能力；
- CModel 是 PCIe 的必要前置，不证明板端并发、驱动和稳定性；
- unsupported 与 unverified 必须分开：前者是当前实现明确拒绝，后者是尚无足够证据；
- 首次板端异常立即终止受控进程组并跳过剩余测试；
- 单次 instruction timing 用于映射审查，不用于性能排名；
- 机器事实以 `research/tpu-op-contract/contract.json` 为准，叙述文档不得扩大其 scope。
