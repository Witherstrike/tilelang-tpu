# TileLang TPU 双芯片、双编程模型设计

## 1. 设计结论

当前实现采用三条正交选择轴：

| 轴 | 规范取值 | 决定内容 |
| --- | --- | --- |
| 芯片 | `bm1690`、`sg2260e` | PPL arch、编译宏、物理核数与芯片能力 |
| 编程模型 | `tpukernel`、`rv` | device ABI、descriptor、指令和 kernel 生命周期 |
| 运行时 | `cmodel`、`pcie` | host runtime、链接依赖、加载与安全策略 |

芯片和编程模型组成编译身份，必须完整写入 TVM Target；运行时只决定如何承载已选定的 device program，不进入 target：

```python
kernel = tilelang.compile(
    program,
    target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel",
    runtime_mode="cmodel",
)
```

有效组合为 BM1690+TPU-Kernel、SG2260E+TPU-Kernel 和 SG2260E+RV。BM1690+RV 在 target 解析阶段失败。BM1690 与 SG2260E 共享当前 allocator 使用的 TPUv7 LMEM 几何；物理核数分别为 8 和 4。物理核数是拓扑事实，不代表当前单核 host launch 已实现多核分片。

## 2. 单一选择源

`tilelang/engine/tpu_config.py` 的只读 `TPU_CHIP_SPECS` 是 Python 侧的能力源：

- BM1690：`tpub_7_1`、8 核、只允许 `tpukernel`；
- SG2260E：`tpub_7_1_e`、4 核、允许 `tpukernel/rv`。

`TPUTargetSpec(chip, programming_model)` 只从完整 target 构造；`TPURuntimeConfig(runtime_mode)` 独立构造，缺省为安全的 CModel。PPL resolver、JIT adapter 和 profiling 都消费这两个对象，不再各自推断芯片或编程模型。native build 入口仍做同样的防御性校验，防止 Python 边界被绕过。

设计不提供模糊 selector：裸 `target="tpu"`、缺少 `-mcpu`、缺少 `-tpu-programming-model`、未知取值以及 BM1690+RV 都在工具链调用前失败。非 TPU target 也不得携带 TPU semantic extern。

## 3. 编译流程与 pass 映射

```text
TileLang frontend
  │
  ├─ T.ppl_{copy,fill,gemm,add,subtract,mul,div,max}
  │      └─ tl.tpu.*              后端中立 ABI
  │
  └─ T.ppl_{scalar,exp,...,rope}
         └─ tl.tpukernel.*        TPU-Kernel 专属 ABI
                    │
                    ▼
          target/module contract verifier
                    │
          BindTarget + frontend legalize
                    │
          TPU conservative pass pipeline
                    │
          second contract verification
                    │
          AddressAssign (TPUv7 LMEM)
                    │
        target.build.tilelang_tpu
                    │
          codegen_tpu.{h,cc}
          ├─ codegen_tpukernel.cc ─► tpu_kernel.h
          └─ codegen_rv.cc        ─► rvt_api.h
                    │
             PPL 1.7 build/link
          ├─ CModel runtime
          └─ installed PCIe runtime
```

源码命名遵循 [TVM 的 target source codegen](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst) 与 [TileLang CUDA codegen](https://github.com/tile-ai/tilelang/blob/main/src/cuda/codegen/codegen_cuda.h) 惯例：`codegen_tpu.{h,cc}` 是唯一的 TPU 目标源码生成器，`codegen_tpukernel.cc`、`codegen_rv.cc` 是同一生成器按编程模型拆分的指令选择实现。当前没有独立的 emitter 类，也不处在 IR-to-IR lowering 阶段，因此不以 `emitter`、`lowering` 或 `common` 命名这些文件。

TPU pass pipeline 只启用已有明确语义的变换：绑定、前端合法化、简化、if binding、buffer allocation location 规划、if 合并、opaque block lowering、窄化、unroll 与最终简化。以下通用 GPU 优化不进入 TPU pipeline：

- vector legalization/vectorization：TPU 源码生成器尚未定义残余 vector lane、Ramp 和 vector load/store 的完整语义；
- software-pipeline planning/injection：DMA/compute token、buffer version 和 hazard 尚未建模；
- `StorageRewrite`：会破坏 TPU codegen 依赖的结构化 `DeclBuffer/Allocate` 配对。

contract verifier 在 target pass 前后各运行一次。它拒绝 vector residual IR、GPU barrier/synchronization、pure/未知 extern、跨编程模型调用和 ABI 混用。随后才执行 `AddressAssign`；该 pass 对非 TPU target 显式失败，不能成为无声 no-op。

残余 IR 的允许集同样是封闭的：循环只允许 serial/unrolled；直接 `BufferLoad/BufferStore`、条件 allocation、`AllocateConst`、未消费 `AttrStmt`、producer/prefetch 节点和 verbatim customized code 均失败。唯一的结构化 load/Ramp 例外是 `tl.tpu.copy` 自带的连续 region marker，且 Ramp 必须 unit-stride、lane 数等于显式 extent。raw `rvt_*` 只接受用户管理的寄存器编码，若参数引用任一 TileLang buffer/data Var，也会在 lowering 边界拒绝。

这套顺序的核心不变量是：任何可能改变指令、地址 effect 或同步语义的选择，都必须在地址分配和 codegen 前确定；不能用 codegen fallback 修补一个含义不完整的 IR。`AddressAssign` 还逐个 `PrimFunc` 核对其已绑定 target 与调用方 target 完全一致，LMEM 地址元数据绑定 data Var 身份而非可碰撞的显示名。

descriptor 合法性也在 TPU 代码生成器中单源化。TPU 本地 scope 只有 `shared/shared.dyn/local/local.fragment`；所有 `dim4` extent 必须是编译期整数且位于 `[1,65535]`。copy 两侧原始 rank 可分别为 1..4，按统一 N/C/H/W 规则左补 1 后比较，不要求原始 rank 相等。TPU-Kernel 特殊约束继续在编程模型层验证：exp/sigmoid 要求完整 shape4 一致且 `H*W<=65535`，reduction 的 EU 对齐后 padded width 也必须可由 `dim4` 表示。

所有 typed TPU semantic op 的 buffer 参数统一以 `tl.region(BufferLoad, access_mask, extents)` 穿过 TIR/native 边界。非 copy 算子只接受从零开始、覆盖完整逻辑 Buffer 的 region；copy 保留显式子区间，并独立证明连续性与边界。native 层再以 data Var 身份核对 compiler-owned descriptor 的 dtype、原始 rank、归一化 shape、scope 与 ownership。裸 `tir.tvm_access_ptr` 不再是兼容入口，因为它会丢失 `T.view/T.reshape` 的逻辑形状。只改变展示名称且 descriptor 完全等价的 alias 可以作为唯一表示；改变 rank、shape、dtype、scope，或同时制造第二个 allocation owner 的 alias 会 fail-closed。

## 4. ABI 分层

| 层 | 名称空间 | 责任 |
| --- | --- | --- |
| 用户前端 | `T.ppl_*` | 提供当前用户可调用的 TileLang 表达和静态 shape/dtype 检查 |
| 中立语义 | `tl.tpu.*` | copy、fill、gemm、add/sub/mul/div/max；允许 target 在 TPU-Kernel 与 RV 间选择 |
| TPU-Kernel 语义 | `tl.tpukernel.*` | scalar、exp、sigmoid、rsqrt、reduction、gather、topk、rope |
| RV 专家 ABI | 隔离的 `rvt_*` | 用户自管 descriptor/lifecycle 的低层路径；不能与 `tl.tpu.*` 混用 |

内部 `ppl.*` 不是编程模型，raw `tpu_*` 也不是支持的 TIR ABI。共同层只持有 tensor region、dtype、shape、effect 和目标选择；TPU-Kernel/RV 文件持有各自的函数签名、descriptor 及指令配置。这样增加一个新编程模型时，不需要在前端复制一组核心 op。

RV 的 compiler-owned 路径配置 CR/TR/GR descriptor、precision/FP8 subtype、FREE stride 与 lane mask，并负责 `rvt_kernel_start → body → rvt_sync_i`。TPU-Kernel 使用独立的 `tpu_initialize → body → tpu_poll` 生命周期；两个协议不会在同一 kernel 中拼接。

## 5. 算子映射

### 5.1 后端中立核心

| 中立 op | TPU-Kernel | RV | 主要契约 |
| --- | --- | --- | --- |
| `tl.tpu.copy` | GDMA S2L/L2S/S2S、BDC L2L/cast | RV DMA load/store/copy 与受限 f2f | 静态等 extent；跨 dtype 只允许 local；不支持组合编译期失败 |
| `tl.tpu.fill` | `tpu_bdc_set_C` | typed CR + copy | RV 目前只开放零值 |
| `tl.tpu.gemm` | `fp_mm`、`fp_mm_R_trans`、FP8 MM 族 | `fmm2[a]_{nn,nt}` | rank-2 local；无 transpose-A；overwrite/accumulate 显式；TPU-Kernel 的基础浮点 NT accumulate 拒绝，RV 对应源码映射已通过但尚无数值证据；仅 TPU-Kernel 开放 FP8，RV FP8 尚未映射 |
| `tl.tpu.add/sub/mul/div/max` | `tpu_bdc_fp_*`/`tpu_bdc_max` | `rvt_f*` | local、同 dtype、等形或 rhs W broadcast；FP16/BF16/FP32 双后端已验证，RV FP8 尚未映射 |

GEMM 的 `accumulate` 同时决定数值语义和内存 effect：overwrite 时 C 为 write，accumulate 时 C 为 read-write。非 FP8 overwrite 的 C 可与 A/B 同 dtype，也可为 FP32；accumulate 必须使用 FP32 C。现有数值矩阵只覆盖“同 dtype overwrite”和“FP32 accumulate”，因此 FP32 overwrite 在契约中保持 `unverified`。FP8 GEMM 固定为同型 FP8 A/B 和 FP32 C。

copy 的搬运方向分开记账。当前 SG2260E/RV 已在同一 `(4,32)` 精确矩阵验证 FP16/FP32 的 G2L、L2L、L2S 与 S2S；local-roundtrip 用一条 case 串起前三种方向，global-to-global 单独验证 S2S。跨 dtype 能力仍按转换方向记录：TPU-Kernel 明确拒绝 FP16↔BF16；RV 的 FP16→BF16 已有精确生成源码证据但没有数值证据，BF16→FP16 连精确 codegen 证据也尚未建立。因此后续 Agent 不能用同 dtype DMA 结果外推转换，也不能把 RV 的两个转换方向合并。

### 5.2 TPU-Kernel 专属算子

| 算子 | 映射/实现 | 当前 dtype |
| --- | --- | --- |
| scalar add/mul | FP32 常量 cast 后调用 `tpu_bdc_fp_add_C/fp_mul_C` | FP16/BF16/FP32/E4M3/E5M2；FP8 为默认非饱和语义 |
| exp | load exp coeff + `tpu_bdc_fp_exp` | FP16/BF16/FP32 |
| sigmoid | negation、exp、reciprocal 与加法组合；五 buffer | FP16/BF16/FP32 |
| reduce sum/max | padding + 两阶段 pooling composite | FP16/BF16/FP32，dim=1 |
| rsqrt | 通用 `tpu_bdc_fp_rsqrt` | FP16/BF16/FP32 |
| gather | `tpu_gdma_h_gather_S2S` | FP16/BF16/FP32/E4M3/E5M2 payload + UINT32 index |
| topk | `tpu_hau_sort_natural_index` | BM1690 的 FP32/INT32/UINT32、升序/降序、精确 K-sized 输出；SG2260E 编译期拒绝 |
| rope | 偶/奇 lane 的两次 fp add composite | FP16/BF16/FP32/E4M3/E5M2 |

特殊算子使用 `tl.tpukernel.*`，因此 RV target 在 lowering/codegen 边界明确拒绝，而不是偷偷回退到 TPU-Kernel。

TopK 的 shape 与排序语义已经由实验收紧：输入为 `length`，两个输出均为 `K`；BM1690 只写前 K 项，重复键按自然索引递增保持稳定。SG2260E 即使头文件有 HAU symbol，仍在 codegen 阶段拒绝。

### 5.3 前端复合算子边界

`tpu_demo` 复用同一组 TileLang 前端表达，不按硬件复制实现。elementwise add/sub/mul/div 与
matmul 只使用后端中立 ABI，因此在 SG2260E 上可由 target 选择 TPU-Kernel 或 RV，在 BM1690
上选择 TPU-Kernel；三种路径均覆盖 FP16、BF16、FP32。RMSNorm、Split-K RMSNorm、RoPE、
SwiGLU 和 FlashAttention 仍依赖 reduction、rsqrt、exp 或 TPU-Kernel RoPE 等专属语义，当前只
允许 TPU-Kernel。SG2260E/RV 会在编译边界拒绝这些 composite，而不是混入 TPU-Kernel 指令
或因底层已有若干 primitive 就推断整个 composite 可用。

## 6. FP8 能力边界

两颗芯片的 TPU-Kernel CModel 已验证 E4M3/E5M2 的：

- 同格式 local roundtrip/S2S copy、零 fill、FP32 双向 cast；
- 等形及 rhs W-broadcast add/sub/mul/max；
- FP8 A/B、FP32 C 的 NN overwrite/accumulate 和 NT overwrite GEMM；
- 公开 `T.ppl_gemm` 的 NT accumulate，两颗芯片 × 两种格式 4/4 通过；
- scalar add/mul 的公开路径，两颗芯片 × 两种格式 × 两种运算 8/8 通过；
- UINT32 索引的 global gather 按 selected encoded bytes 精确通过；
- 偶宽 rank-2 interleaved RoPE add composite 通过。

NT accumulation 采用编程模型与 dtype 双重选择：TPU-Kernel 只为 FP8 A/B + FP32 C 放行 `_R_trans(..., result_add=true)`，基础 FP16/BF16 因原始右转置 API 没有 accumulation 参数而在 codegen 拒绝；SG2260E/RV 的 FP16/BF16 + FP32 C 已有 `rvt_fmm2a_nt` 精确源码选择回归，但 CModel 与 PCIe 数值均未验证。

FP8 div 明确不支持；非零 fill、其他 cast、FP8 output GEMM 以及 exp/sigmoid/rsqrt/reduction/topk 不从已测子集外推。RV ISA 虽声明 FP8 arithmetic/GEMM，当前 TileLang RV codegen 尚未实现这些 descriptor/指令组合，故 fail-closed。PPL 1.7 可读手册进一步确认 pointwise DataType 包含 FP8、矩阵输入包含 E4M3/E5M2，并记录 BM1690/tpub_7_1_e 的 saturation 差异；这些只作为 declaration evidence，支持提升仍依赖精确 codegen/CModel/PCIe 结果。

FP8 scalar 采用 PPL 1.7 的 canonical same-format 路径：FP32 常量先 `tpu_cast(..., RM_HALF_TO_EVEN)` 到目标格式，再调用通用 `tpu_bdc_fp_add_C/fp_mul_C`。moderate 输入与公开生产路径均通过。历史 direct `tpu_bdc_fp8_*_C` exit 139 是 FP8 dst/src + FP32 `C_dtype` 的非法参数探针，不是硬件负向证据。边界实验证实当前为非饱和语义：E4M3 overflow 产生 NaN，E5M2 产生 infinity；可选 saturation 仍不得暴露。

机器可读的逐 selector 事实位于 `research/tpu-op-contract/contract.json`。当前验收实现基线
`e5774525e3a6e11d0d6010e979203c55181a8872` 的八份 CModel summary 均位于
[正式工件目录](../artifacts/2026-09-09/final-e5774525/)，合计 553/553 次 launch。SG2260E 只取四份
`*-cmodel-retry1`，BM1690 取四份原始矩阵：核心矩阵为 BM1690
28/28、SG2260E 56/56，FP8 为 42/42 + 42/42，demo 为 36/36 + 51/51，完整
TPU-Kernel 为 152/152 + 146/146。各 summary 均自记同一 revision 并确认
`implementation_worktree_dirty=false`。这些矩阵之间存在有意的 selector 重叠，553 是回归
launch 数，不是 553 项独立能力；能力提升仍须落到 contract 的精确 chip/model/dtype/layout
selector。

## 7. 地址、effect 与失败策略

`AddressAssign` 不按函数名猜测一般副作用，而是为封闭 semantic op 集声明 operand role：

- copy：src read、dst write；
- fill：dst write；
- GEMM：A/B read，C 依 `accumulate` 为 write 或 read-write；
- 二元/标量/rsqrt/rope：输出 write、输入 read；
- reduction：输入与 tmp 含 padding/工作区更新，按 read-write 建模；
- exp/sigmoid：复合序列中的 workspace 采用保守冲突关系；
- gather/topk：索引和值的输入/输出角色显式记录。

生命周期分析同时建模循环回边。对一个可能重复的 `For`（静态 extent 大于 1 或符号
extent）或 `While`，凡在循环外分配、循环内使用的 local buffer，其 live range 保守覆盖整个
循环；嵌套循环按 allocation depth 逐层闭包。这样，循环前初始化并在每轮前段读取的常量 tile
不会与本轮后段 scratch 复用地址。静态 0/1 次循环没有回边，循环内分配的临时量也不会被无谓
提升到外层生命周期，因而仍保留可证明安全的顺序复用。

未知 semantic op 不允许悄悄采用乐观 effect。shape、dtype、scope、layout、目标能力或生命周期不完整时，编译应在首次可判定的阶段终止。

别名同样属于公开契约，而不是留给底层碰运气：descriptor-equivalent presentation alias 与运行时存储重叠是两个概念。前者只在不改变 descriptor 且不产生第二个 allocation owner 时允许；后者按 op 约束。exp 的四个 buffer、sigmoid 的五个 buffer、gather 的 output/param/index、topk 的两个输出与输入都要求存储两两不同；RoPE 只要求输出与四个输入分别不同，输入之间的 read/read alias 允许。其他算子的 in-place 或重叠组合没有被数值矩阵覆盖时，不得从等形结果外推。

## 8. PPL 1.7、运行时与 profiling

工具链只识别 PPL 1.7 `deps/` 发布布局。基础 device 编译、CModel runtime、PCIe runtime 与 profiling 依赖分层校验：普通 CModel 不需要 PCIe cross compiler/TPUDNN profiling 库；PCIe 不得链接 SDK 内的 emulator runtime，而应使用已安装的板端 `libtpuv7_rt.so`。

本地 `tpu_kernel_manual.md` 是官方 v23.03.01 网页手册的可搜索转录，用于解释 TPU-Kernel 编程模型、内存、同步与通用 API 语义；其正文面向较早芯片，不能单独证明 BM1690/SG2260E 或 FP8 selector。当前芯片的声明范围以 PPL 1.7 手册、对应 `tpub_7_1/tpub_7_1_e` 头文件为准，最终支持仍以 source/codegen/CModel/PCIe 分层证据为准。

profiling 复用 PPL 的运行记录协议，而非把 `ppl_compile.py --profiling` 原样套到 TileLang 生成的 C：

- CModel：`FILE_DUMP_CMD` 收集 raw 命令；有 PerfAI 时再解码；
- PCIe：TPUDNN recorder 包围一次 launch 并保留 raw 文件；仅当调用方显式提供兼容的 `bigTpuProfile/PerfAI` 时，才在独立解释器和 `PYTHONPATH` 中离线投影为稳定 JSON。该 decoder 环境不注入编译或数值 worker。

生产代码不会联网或自动安装 decoder。严格 timing 矩阵在任何板端派发前先做无硬件 preflight，确认结构化 parser API 可用，并把实际包版本与 API 身份写入 summary。当前隔离环境验证的身份为 `bigTpuProfile 0.3.5` 和 `bigTpuProfile.bmprofile_perfAI.ProfileParser.parse`；这仍是显式外部依赖，不是基础 runtime 保证。

数值验证与 profiling 分开判定。当前 [SG2260E PCIe core
56/56](../artifacts/2026-09-09/final-e5774525/core-sg2260e-pcie/summary.json)、[FP8
42/42](../artifacts/2026-09-09/final-e5774525/fp8-sg2260e-pcie/summary.json) 和 [demo
51/51](../artifacts/2026-09-09/final-e5774525/demo-sg2260e-pcie-shards/) 共 149 个
case，均通过 numeric、非空 raw recorder 与 decoded timing 三重门禁；对应 149 组 recorder
目录和 3568 条合法 ns 事件。每组目录恰含 `global.profile` 与四个 SG2260E 核的
`cdmlib0_*.profile`，因此 summary 的 `raw_trace_file_count=1` 指一组目录，不是一个物理文件。
decoder 身份为 `bigTpuProfile 0.3.5` 和
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse`。

[完整 TPU-Kernel PCIe 矩阵](../artifacts/2026-09-09/final-e5774525/tpukernel-sg2260e-pcie-shards/)
由 14 个完整通过分片覆盖 146/146 次数值执行，不启用 profiling；demo 的 51/51 项则由
15 个完整通过分片覆盖。两类分片分别按 case 去重汇总，失败矩阵和 canary 不计入验收。
数值矩阵与 149 个 profiling case 的验收条件不同。demo 的 51 项由 TPU-Kernel 36 项和
RV 15 项组成；RV 只包含三种 dtype 的四则与
matmul；RMSNorm、Split-K、RoPE、SwiGLU、FlashAttention 的 RV composite 仍未实现。

PCIe preflight/postflight 在同一 device session lock 内执行有界静默检查：仅
`Active + 合法非零利用率` 可以继续轮询，拓扑、状态、JSON、probe 或进程组异常立即失败；
10 秒总 deadline 内必须取得两次间隔 0.25 秒的连续 `Active/0%`。dispatch 后若仍不能证明
静默，session/quarantine marker 持久保留，后续 launch 停止。单次 `Fault` 也立即触发隔离，
不因数值或 decoder 已通过而放行。

`e5774525` 的正式结果为 CModel 553/553、SG2260E PCIe 295/295，合计 848/848。该集合
覆盖已修复循环回边 liveness 的实现，但不代表一次连续运行完成全部 PCIe 测试。完整矩阵和
部分分片曾因健康检查失败而中止；失败样本的温度、时钟、利用率和电压均显示 `F`，根因尚未
证实。后续完整通过分片只证明其各自的数值与门禁结果，不能消除历史故障或证明长期稳定性。
正式集合、分片选择及故障边界见 [测试报告](test-report.md)。

PCIe timing 来自每 case 一次带 recorder 的设备指令事件，只用于审查映射和定位问题，不是
去除 recorder 开销、重复采样后的端到端 benchmark。本机无 BM1690 板卡，因而 BM1690 PCIe
仍为 `unverified`；CModel raw 也没有兼容的本机 PerfAI，不能生成真实 duration。

## 9. 扩展规则

新增 chip、编程模型或 op 时遵循以下顺序：

1. 在能力注册中声明合法 target 组合，不在 codegen 中猜测；
2. 先定义 semantic ABI、operand effect、shape/dtype/layout 与 failure policy；
3. 让 pass/verifier/AddressAssign 消费同一契约；
4. 编程模型各自实现指令选择，未实现项 fail-closed；
5. 先 source-only，再 CModel 数值，最后受控 PCIe；
6. 只按精确 selector 更新机器契约，不能从 SDK 声明、另一芯片或另一运行时复制状态。

当前实现仍有两项重要架构债务：Python 与 native 目标能力表尚需生成式单源化；operation spec 仍分散在 frontend、verifier、AddressAssign 和 codegen。FP8 NT accumulate 的修复说明合法性必须按完整 dtype/layout/attribute selector 判断。下一阶段应引入注册式 `TpuOpSpec`，由同一 selector 规则生成 frontend 校验、verifier/effect 与 backend capability 查询，再迁移到上游式 `tilelang/tpu`、`src/tpu` 后端边界。
