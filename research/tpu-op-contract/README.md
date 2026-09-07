# TileLang TPU 算子能力契约

本目录给出一份可由人和自动化工具共同消费的能力事实：`schema.json` 约束结构，`contract.json` 记录算子语义、目标组合、失败策略和逐阶段证据。它刻意区分“源码能够表达”“生成代码通过”“CModel 数值通过”和“PCIe 数值通过”，避免从头文件声明或单次模拟结果推导出更宽的硬件支持。

## 1. 目标与选择维度

| target id | 芯片 | 编程模型 | PPL arch | 物理核数 | 是否有效 |
| --- | --- | --- | --- | ---: | --- |
| `bm1690.tpukernel` | BM1690 | TPU-Kernel | `tpub_7_1` | 8 | 是 |
| `bm1690.rv` | BM1690 | RV Tensor | — | — | 否，target 解析阶段拒绝 |
| `sg2260e.tpukernel` | SG2260E | TPU-Kernel | `tpub_7_1_e` | 4 | 是 |
| `sg2260e.rv` | SG2260E | RV Tensor | `tpub_7_1_e` | 4 | 是 |

运行时 `cmodel/pcie` 不属于 target，也不改变算子或指令选择。契约中的 target result 只引用上述三个有效组合。

源码根采用可移植语义：仓库为 `.`，ignored 实验为 `research/artifacts`，本地资料为 `.local_content`，外部 PPL 1.7 SDK 为 `${PPL_PROJECT_ROOT}`。Schema 同时禁止 source root 和 evidence path 使用机器绝对路径。

资料也有证据层级：本地 `tpu_kernel_manual.md`（官方 v23.03.01 网页手册转录）用于解释通用编程模型和 API 语义，不用于推导 TPUv7/FP8 能力；实际 selector 声明以 PPL 1.7 手册及 `tpub_7_1/tpub_7_1_e` 头文件为准，最终状态仍由 TileLang source、codegen、CModel 与 PCIe 逐层决定。

## 2. 状态模型

每个 capability 由 `operation_id + variant + dtype_bindings + constraints` 唯一限定，再按 target 记录以下阶段：

1. `declared`：当前公开 TileLang 前端可表达该 selector；这一阶段不做 target-specific 指令选择，因而同一 capability 在各有效 target 上的 `declared.status` 必须一致。若只有内部 ABI 能通过，则该阶段仍失败，并标为 `known_gap`；
2. `codegen_passed`：精确 selector 已通过目标相关 lowering/source emission；是否进一步完成 device 编译、加载或执行，只能由该 stage 的 `scope/reason/evidence_ids` 和后续数值 stage 判断；
3. `cmodel_numeric_passed`：CModel 输出通过指定 oracle；
4. `pcie_numeric_passed`：真实芯片输出通过指定 oracle；
5. `sdk_declared`：可选，只说明手册或头文件存在底层声明。

阶段状态为 `passed/historical_passed/failed/unverified/not_applicable`。其中
`historical_passed` 只允许用于 PCIe：它保留旧实现上真实通过的硬件事实，但不是当前实现的
上板授权；对应 `scope` 必须显式以 `Historical` 开头，当前实现没有同基线重跑时必须使用该状态，而不能保留 `passed`。总状态的判定规则是：

- `supported`：精确 selector 至少有 codegen 证据；数值与 PCIe 仍按各自阶段判断，因此不能单独用该字段判定可上板；
- `unverified`：存在声明或候选路径，但没有精确 codegen 证据；
- `unsupported`：当前 TileLang 在必要阶段显式拒绝该 selector；
- `experimental`：路径存在但语义或安全契约尚未闭合，默认不得调度。

`sdk_declared=passed` 不能提升任何 TileLang 阶段。某个 capability 含多个 binding 时，只有所有 binding 都被同一证据覆盖才可整体提升；否则必须拆行。本轮因此将 BF16 与整数 copy、FP32 与 FP16/BF16 broadcast 分别记录。

### 2.1 全局 semantic buffer ABI

Schema 1.1 起，`invariants` 记录跨 op 的不可绕过规则。当前 `semantic-buffer.region-abi` 覆盖全部 15 个 operation：每个 typed semantic buffer 参数必须是 `tl.region(BufferLoad, access_mask, logical_extents)`，不保留裸 `tir.tvm_access_ptr` 兼容入口。非 copy op 要求零起点 whole-buffer region；copy 才能携带显式子区间，并额外校验连续 Ramp、bounds 和 local C-axis 起点。Schema 1.2 又把 ignored runtime artifacts 从 tracked repo evidence 中分离，并加入不具当前上板授权意义的 `historical_passed` 状态。

这条 ABI 的目的不是改变用户侧 `T.ppl_*` 表达，而是把 logical Buffer 的 dtype、原始 rank、shape、scope 与 storage identity 一直保留到 native codegen。native 层把这些字段与 compiler-owned descriptor 逐项核对；因此改变 rank/shape/dtype/scope 的 `T.view`、`T.reshape` 或直接 `T.Tensor(..., data=...)` 会被拒绝，不会静默使用 owner descriptor 执行。descriptor 完全等价且作为唯一表示的 presentation alias 仍合法。region 的 mask、返回 dtype、rank、extent 或 marker 结构即使绕开公开 frontend 人工构造，也会在 residual verifier 或 native parser fail-closed。

## 3. 已验证边界

### 3.1 TPU-Kernel CModel

2026-09-07 对实现基线 `44a6fc2ab6e8ca60569853fd781e21bfa0b78335` 的 [canonical CModel 重跑](../artifacts/2026-09-07/tpukernel-cmodel-44a6fc2/summary.json) 为 fail-stop、逐 case 独立进程测试，总计 288/288：

| 芯片 | 结果 | 覆盖 |
| --- | ---: | --- |
| SG2260E | 141/141 | copy/cast/fill、FP16/BF16 GEMM、浮点四则、scalar add/mul、exp、五 buffer sigmoid、sum/max reduction、三种基础浮点 rsqrt、gather、rope；topk 不适用 |
| BM1690 | 147/147 | SG 的 141 个共同 case，加 FP32/INT32/UINT32 各自升序和降序 topk |

较早的 SG/BM 独立 `*-final` summary 与该精确基线重跑的共有 selector 一致，但不与 288 项重复累计。主要数值范围如下：

- copy：`(4,32)` 上覆盖 FP16/BF16/FP32 以及 INT8/UINT8/INT16/UINT16/INT32/UINT32 的 S2L→L2L→L2S 与 S2S，并覆盖本地 FP16/BF16→FP32、FP32→FP16/BF16 cast；另有 `(2,3,17)` FP32 rank-3 本地 roundtrip，在两芯片各 102 个元素精确通过。源/目标原始 rank 可不同，但都必须为 1..4，并按 r1=`[1,1,1,W]`、r2=`[1,C,1,W]`、r3=`[N,C,1,W]`、r4 原样归一化后 extent 相等；
- fill：`(4,32)`，三种浮点 dtype，常量 `1.25`；
- GEMM：`16×16×16`，FP16/BF16 的 NN overwrite、NN accumulate 和 NT overwrite；overwrite 的 C 与输入同 dtype，accumulate 的 C 为 FP32；
- add/sub/mul/div：`(4,32)` 等形三种浮点 dtype，以及 FP32 rhs `(4,1)` W broadcast；
- scalar/exp/sigmoid：`(4,32)` 三种浮点 dtype；sigmoid 使用 `(out, inp, work0, work1, coeff)`，`coeff=(64,32)`；
- reduction：65 行，FP16/BF16/FP32 的 sum/max，width 为 `15,16,17,31,32,33,47,48,49,63,64,65`，覆盖 EU 边界两侧；
- gather：`param=(17,32)`、`index=(7,1)` UINT32、`output=(7,32)`，三种浮点 payload 精确比较；
- rope：`(4,32)` 三种浮点 dtype；rsqrt 通过通用 `tpu_bdc_fp_rsqrt` 开放 FP16/BF16/FP32；
- topk：BM1690 的 `length=257, K=11`，FP32/INT32/UINT32 均覆盖升序和降序；公开输出 buffer 精确为 K 个元素。[sentinel 实验](../artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json) 证实 HAU 只写 `[0,K)`，[重复键实验](../artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json) 证实相同值按自然索引递增稳定排序。SG2260E 在 codegen 阶段拒绝，当前不会进入 CModel 或 PCIe。

所有进入 TPUv7/PPL `dim4` 的 N/C/H/W extent 都必须是编译期整数并落在闭区间 `[1,65535]`；TPU 本地 scope 的闭集为 `shared/shared.dyn/local/local.fragment`。exp/sigmoid 还要求完整 shape4 一致且 `H*W<=65535`；reduction 的 EU 对齐后 padded width 也不得超过 65535。相应正负例集中在 `testing/python/jit/test_tpu_codegen_descriptor_contract.py`。

同一干净实现基线的 [核心 profiling/CModel 重跑](../artifacts/2026-09-07/core-cmodel-44a6fc2/summary.json) 为 27/27：BM1690/TPU-Kernel、SG2260E/TPU-Kernel、SG2260E/RV 各 9 项。每组包含 FP32 add/sub/mul/div、FP16 A/B 与 FP32 累加的 NN GEMM，以及 FP16/FP32 的 local-roundtrip、global-to-global copy。SG2260E/RV 的 local case 直接覆盖 G2L→L2L→L2S，global case 覆盖 S2S，输入为 `(4,32)` 可精确表示的 quarter-integer，按逐元素完全相等判定；对应 raw 命令数分别为 45 和 43。四个逐元素 case 各保留 58 条 raw 命令，GEMM 保留 78 条。所有 case 均生成了非空 CModel raw trace，但本机没有兼容 decoder，本轮也未启用后处理，故 `timed_instruction_count=0`。这些记录只能证明指令选择和数值结果，不能证明逐指令耗时。

### 3.2 FP8 TPU-Kernel

[最终 FP8 CModel summary](../artifacts/2026-09-07/fp8-cmodel-44a6fc2/summary.json) 同样绑定实现基线 `44a6fc2ab6e8ca60569853fd781e21bfa0b78335`，结果为 76/76：`2 chips × 2 formats × 19 public cases`，每例均完成数值判定并保留非空 CModel raw 命令 trace。此前 40 项基础矩阵及若干增量实验仍是定位实现演进的辅助证据，能力判定以这份完整重跑为准。

| 算子族 | 当前 TileLang TPU-Kernel 范围 | RV 状态 |
| --- | --- | --- |
| copy | 同格式 local roundtrip 与 S2S；FP32 与同一 FP8 格式双向本地 cast | fail-closed |
| fill | 只开放零值 | fail-closed |
| add/sub/mul | 等形 `(1,64)` 及 rhs `(1,1)` W broadcast | ISA 有声明，TileLang 尚未映射 |
| div | 不支持；已审阅接口的 operand 范围不含 FP8 | 不支持 |
| GEMM | 同型 FP8 A/B、FP32 C；NN/NT overwrite 与 accumulate 均已通过 | ISA 有声明，TileLang 尚未映射 |
| scalar add/mul | 同型 E4M3/E5M2，FP32 常量先 round-to-even cast；默认非饱和语义 | 不适用 |
| gather | `param=(17,32)`、UINT32 `index=(7,1)`；两格式均按 selected encoded bytes 精确验证 | 不适用 |
| rope | `(4,32)` 偶/奇 lane FP8 add composite；两格式均已验证 | 不适用 |
| exp/sigmoid/rsqrt/reduce/topk | 当前生产路径不开放 FP8 | 不适用 |

PPL 1.7 高层 DSL 和当前 TileLang 使用同一 scalar 序列：FP32 常量以 round-to-even cast 成目标 FP8，再调用通用 `tpu_bdc_fp_add_C/tpu_bdc_fp_mul_C`。[PPL probe](../artifacts/2026-09-05/fp8-scalar-ppl-probe/summary.json) 在两芯片的 moderate 与 boundary 输入上完成；当前公共路径随后 8/8 通过。历史 E4M3 direct `tpu_bdc_fp8_add_C` 探针在 [SG2260E](../artifacts/2026-09-05/fp8-scalar-cmodel/summary.json) 与 [BM1690](../artifacts/2026-09-05/fp8-scalar-bm1690-cmodel/summary.json) 的 exit 139 已定位为非法参数：FP8 dst/src 搭配 FP32 `C_dtype` 违反 `sizeof(C_dtype) <= sizeof(dst_dtype)`。它不是硬件不支持证据。当前只承诺默认非饱和行为；E4M3 overflow 为 NaN、E5M2 为 infinity，不暴露可选 saturation。

### 3.3 当前 PCIe 证据与门禁

板卡恢复健康后，2026-09-07 在干净实现基线 `44a6fc2ab6e8ca60569853fd781e21bfa0b78335` 上按 fail-stop 顺序完成 151 次 SG2260E device 0 launch。每份 summary 都由 runner 自记相同 revision、`implementation_worktree_dirty=false` 与 `source_identity_scope="tracked files excluding research/**"`；每个 contract runtime expectation 又锁定完整 required case manifest，不能用同数量的其他 selector 替换。

| 批次 | 结果 | 覆盖 |
| --- | ---: | --- |
| [TPU-Kernel matmul canary](../artifacts/2026-09-07/pcie-tpukernel-matmul-44a6fc2/summary.json) | 1/1 | `64×64×64` FP16 GEMM、FP32 accumulator zero、FP32→FP16 输出转换；数值、raw recorder 与 36 条 ns timing 均通过 |
| [RV core](../artifacts/2026-09-07/pcie-rv-core-44a6fc2/summary.json) | 9/9 | FP32 四则、FP16 GEMM，以及 FP16/FP32 各自的 local-roundtrip 与 S2S；数值、raw recorder 与 108 条 ns timing 均通过 |
| [TPU-Kernel core](../artifacts/2026-09-07/pcie-tpukernel-core-44a6fc2/summary.json) | 54/54 | 23 个浮点/整数 copy/cast（含 `(2,3,17)` FP32 rank-3）、三种非零 fill、六种 FP16/BF16 GEMM、16 种 tensor arithmetic、六种 scalar arithmetic |
| [TPU-Kernel extended](../artifacts/2026-09-07/pcie-tpukernel-extended-44a6fc2/summary.json) | 15/15 | FP16/BF16/FP32 exp、sigmoid、rsqrt、gather、rope |
| [TPU-Kernel reductions](../artifacts/2026-09-07/pcie-tpukernel-reductions-44a6fc2/summary.json) | 72/72 | 三种基础浮点 sum/max，覆盖 width `15,16,17,31,32,33,47,48,49,63,64,65` |

TPU-Kernel 三个数值批次合计现行 141/141；独立 matmul canary 用于验证 accumulator zero 与 PCIe profiling，不与 141 项解释为不同 selector。RV 的四条独立 copy case 直接闭合 G2L→L2L→L2S 与 S2S，因此 `copy.same-fp32-fp16.same-space` 可以在该精确 `(4,32)` 范围内从 `unverified` 提升为当前 `passed`；GEMM 内的 FP32→FP16 cast 和 zero fill 仍只按复合 workload 记载，不外推为独立 selector。

两份 profiling 矩阵都显式要求 decoded timing，所有 case 的 parser 均为 `ready`，并同时保留非空 raw trace。单次事件用于证明指令选择和定位问题，不是稳定性能结论；生产路径仍不会自动安装 decoder。此前 2026-09-04/05 的 RV 5/5、TPU-Kernel 140/140 与离线 decode 文件只保留为演进记录，不再授权或重复累计当前 capability。

契约现在有 43 个当前 `pcie_numeric_passed=passed` stage：42 个是对旧历史精确范围的同 revision 重跑，另一个是 RV FP16/FP32 L2L/S2S 的四条直接 copy case。BM1690 PCIe、FP8 PCIe及更宽的 SG2260E/RV selector 没有对应板端证据，继续保持 `unverified`；SG2260E topk 仍为 `not_applicable`。

### 3.4 远程提交前审查

实现基线 `44a6fc2ab6e8ca60569853fd781e21bfa0b78335` 的 source-only 最终回归为 335 passed、4 skipped，其中 descriptor contract 32/32、frontend contract 41/41、core matrix 35/35。四个 skip 均为需要显式启用的真实 profiling worker，默认回归不会访问 CModel 或板卡。

本轮审查闭合了以下提交风险：profiling 以保存的 PGID 检查 supervisor 正常退出后的残留后代，并按 TERM→KILL 执行有界清理；CModel、PCIe、offline decoder 和 PerfAI parser 共用这一语义。契约校验器对不支持的 schema 结构、未闭合引用及不匹配当前 revision/target/capability 的 runtime evidence 均 fail-closed。TPU target 若选择 DLPack，会在 lowering 前拒绝；生成的 host 参数使用位置化 `arg_<index>` 标识，不再信任可能重复或不符合 C++ 标识符规则的 TIR name hint。

Python 兼容性按项目声明的 3.8 下界收紧。包含 `T.prim_func` 的 TVM Script 源文件不能启用 `from __future__ import annotations`，否则 `T.Tensor` 等注解会变成字符串，TVM 无法取得实时类型对象；专门的 AST 回归对此设门禁。FP8 elementwise selector 也改为只在确有 `-broadcast` 后缀时剥离后缀，裸 `add/sub/mul` 名称保持不变，六种带/不带后缀的 case 均由 source-only 回归覆盖。

## 4. 特殊失败的解释

- SG2260E topk：`tpub_7_1_e` 头文件存在 HAU symbol，但运行库明确拒绝该原语。当前实现据此在 codegen 失败；历史 pre-guard CModel assertion 仅作为 guard 的依据，当前 CModel/PCIe 阶段均为 `not_applicable`。
- NT accumulate：公开 frontend 保留显式语义，最终由编程模型选择。TPU-Kernel `_R_trans(..., result_add=true)` 只为 FP8 开放，并由内部 ABI 4/4、公开 `T.ppl_gemm` 两芯片两格式 4/4 验证；其基础 FP16/BF16 右转置 API 没有 `result_add`，所以在 target codegen 拒绝。SG2260E/RV 的 FP16/BF16 + FP32 C 已通过 `rvt_fmm2a_nt` 源码选择回归，但 CModel 与 PCIe 数值仍为 `unverified`。
- FP8 scalar：公开 add/mul 已由完整矩阵验证。历史 exit 139 来自 direct mixed-precision API 的非法 dtype tuple，只保留为错误调用样本，不得标成芯片不支持。
- `unverified` 不是弱支持。自动生成器不得绕过 guard，也不得把同类 dtype、另一芯片或另一运行时的结果复制过来。

## 5. 自动化消费与升级

自动化工具默认只消费 `support_status="supported"`，并根据用途额外要求 CModel 或 PCIe stage 为 `passed`。升级一项能力时：

1. 将 selector 缩到证据实际覆盖的 dtype、shape、transpose、accumulate 和 broadcast；
2. 新增带 `source_root/path/locator/observed_at` 的 evidence；
3. 先记录 declared/codegen，再记录带 oracle 的 CModel，最后才允许 PCIe；
4. 每个 PCIe case 使用独立进程组和超时，首错终止并跳过剩余测试；
5. 更新 stage 的 `scope/reason/evidence_ids`，不得顺带提升相邻 selector。

Agent 或生成器使用时还必须遵守以下规则：

- `target_results` 中没有目标 target，表示该 operation 的 `backend_applicability` 不包含该编程模型，不得视为默认支持；
- 要在 CModel 调度，同时要求 `support_status="supported"` 和 `cmodel_numeric_passed.status="passed"`；要上板，还必须要求 `pcie_numeric_passed.status="passed"`。`historical_passed` 只能用于回归范围规划，绝不能作为当前上板门禁；
- `unverified` 和 `not_applicable` 都不允许生成运行时调度；`unsupported` 必须保持现有拒绝路径；
- 匹配必须覆盖完整 `dtype_bindings + constraints + variant`，不能只按 operation id 或 dtype 族做模糊匹配；
- runtime report 的 `runtime_expectation.claim_targets` 与 `capability_ids` 共同限定可授权的 target/capability 笛卡尔积；两组 ID 必须属于契约闭集且组合适用。stage 只有同时匹配 runtime、当前 revision、target 和 capability 才可取得当前数值证据；自然语言 `scope` 还应写明 shape、dtype、方向与容差，便于人工复核；
- `sdk_declared` 只能用于排定后续实现候选，不能作为调度门禁。

`research/artifacts/**` 被 Git 忽略，适合保留本机完整结果；因此克隆仓库后工件可能不存在。可移植结论由本目录契约和 `research/tpu-backend-design/test-report.md` 汇总，工件路径只是证据定位符。

当前五份 PCIe canonical evidence 都列出了完整 `required_case_ids`；其数量分别与 target case count 的 1、9、54、15、72 完全相等，严格校验因此能识别 selector 缺失或等量替换。仍刻意保留的边界是 TPU-Kernel CModel 288 项与 FP8 CModel 76 项 evidence 目前只锁定总数、target 分布和全量通过，尚未把 capability 与直接或复合覆盖它的具体 case id 建成机器可检验的明确映射。本轮已人工比较 CModel 新旧 case 集合一致；后续若允许 Agent 自动提升这些 CModel capability，必须先增加稳定 case manifest 或 `claims_by_target` 映射，不能仅凭数量相同自动升级。

## 6. 校验

```bash
python3 -m json.tool research/tpu-op-contract/schema.json >/dev/null
python3 -m json.tool research/tpu-op-contract/contract.json >/dev/null
python3 research/tpu-op-contract/validate.py
# 在保存了 ignored runtime evidence 的验收机上再执行：
python3 research/tpu-op-contract/validate.py --require-local-artifacts
```

`validate.py` 只依赖 Python 标准库，并直接执行仓库所用的 JSON Schema 2020-12 子集；它检查顶层版本/时间、ID/引用唯一性、无孤立 evidence、vocabulary 与 dtype 闭集、operation/target/evidence/constraint 引用闭包、target 与 `TPU_CHIP_SPECS` 的 arch/核数/编程模型一致性、capability 的完整 target 覆盖、前端 `declared` 的 target 无关性及 `src.frontend` 证据、operand effect/direction 一致性、stage 名称/顺序/证据、可移植路径、源码 locator，以及 contract internal symbols 与 frontend、`lower.py` 闭集、AddressAssign、codegen 集合的一致性。runner-recorded evidence 还必须带匹配当前基线的 revision、已知 source-identity scope，以及 target/capability 均闭合且机器可检验的 runtime expectation；本地 artifact 存在时会核对 `complete`、runtime、case 总数、全量 pass、target 分布、必需 case 与重复 case id，严格模式还要求工件实际存在。它同时锁定 SG topk、FP8 scalar、RV FP16/BF16 单向 cast、编程模型相关的基础浮点 NT accumulate，以及“frontend 已接受但仍缺精确验证”的 FP32 overwrite selector。可再选用完整 JSON Schema validator 做交叉检查，但它不再是结构校验生效的前提。
