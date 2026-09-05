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

源码根采用可移植语义：仓库为 `.`，本地资料为 `.local_content`，外部 PPL 1.7 SDK 为 `${PPL_PROJECT_ROOT}`。Schema 同时禁止 source root 和 evidence path 使用机器绝对路径。

资料也有证据层级：本地 `tpu_kernel_manual.md`（官方 v23.03.01 网页手册转录）用于解释通用编程模型和 API 语义，不用于推导 TPUv7/FP8 能力；实际 selector 声明以 PPL 1.7 手册及 `tpub_7_1/tpub_7_1_e` 头文件为准，最终状态仍由 TileLang source、codegen、CModel 与 PCIe 逐层决定。

## 2. 状态模型

每个 capability 由 `operation_id + variant + dtype_bindings + constraints` 唯一限定，再按 target 记录以下阶段：

1. `declared`：当前公开 TileLang 前端可表达该 selector；若只有内部 ABI 能通过，则该阶段仍失败，并标为 `known_gap`；
2. `codegen_passed`：实际 case 已完成源码生成、编译和加载；
3. `cmodel_numeric_passed`：CModel 输出通过指定 oracle；
4. `pcie_numeric_passed`：真实芯片输出通过指定 oracle；
5. `sdk_declared`：可选，只说明手册或头文件存在底层声明。

阶段状态为 `passed/failed/unverified/not_applicable`。总状态的判定规则是：

- `supported`：精确 selector 至少有 codegen 证据；数值与 PCIe 仍按各自阶段判断；
- `unverified`：存在声明或候选路径，但没有精确 codegen 证据；
- `unsupported`：当前 TileLang 在必要阶段显式拒绝该 selector；
- `experimental`：路径存在但语义或安全契约尚未闭合，默认不得调度。

`sdk_declared=passed` 不能提升任何 TileLang 阶段。某个 capability 含多个 binding 时，只有所有 binding 都被同一证据覆盖才可整体提升；否则必须拆行。本轮因此将 BF16 与整数 copy、FP32 与 FP16/BF16 broadcast 分别记录。

## 3. 已验证边界

### 3.1 TPU-Kernel CModel

2026-09-05 两份 canonical final 矩阵均为 fail-stop、逐 case 独立进程测试：

| 芯片 | 结果 | 覆盖 |
| --- | ---: | --- |
| SG2260E | 140/140 | copy/cast/fill、FP16/BF16 GEMM、浮点四则、scalar add/mul、exp、五 buffer sigmoid、sum/max reduction、三种基础浮点 rsqrt、gather、rope；topk 不适用 |
| BM1690 | 146/146 | SG 的 140 个共同 case，加 FP32/INT32/UINT32 各自升序和降序 topk |

证据分别为 [SG2260E final summary](../artifacts/2026-09-05/tpukernel-sg2260e-cmodel-final/summary.json) 和 [BM1690 final summary](../artifacts/2026-09-05/tpukernel-bm1690-cmodel-final/summary.json)。主要数值范围如下：

- copy：`(4,32)`，FP16/BF16/FP32 以及 INT8/UINT8/INT16/UINT16/INT32/UINT32 的 S2L→L2L→L2S 与 S2S；本地 FP16/BF16→FP32、FP32→FP16/BF16 cast；源/目标原始 rank 可不同，但都必须为 1..4，并按 r1=`[1,1,1,W]`、r2=`[1,C,1,W]`、r3=`[N,C,1,W]`、r4 原样归一化后 extent 相等；
- fill：`(4,32)`，三种浮点 dtype，常量 `1.25`；
- GEMM：`16×16×16`，FP16/BF16 的 NN overwrite、NN accumulate 和 NT overwrite；overwrite 的 C 与输入同 dtype，accumulate 的 C 为 FP32；
- add/sub/mul/div：`(4,32)` 等形三种浮点 dtype，以及 FP32 rhs `(4,1)` W broadcast；
- scalar/exp/sigmoid：`(4,32)` 三种浮点 dtype；sigmoid 使用 `(out, inp, work0, work1, coeff)`，`coeff=(64,32)`；
- reduction：65 行，FP16/BF16/FP32 的 sum/max，width 为 `15,16,17,31,32,33,47,48,49,63,64,65`，覆盖 EU 边界两侧；
- gather：`param=(17,32)`、`index=(7,1)` UINT32、`output=(7,32)`，三种浮点 payload 精确比较；
- rope：`(4,32)` 三种浮点 dtype；rsqrt 通过通用 `tpu_bdc_fp_rsqrt` 开放 FP16/BF16/FP32；
- topk：BM1690 的 `length=257, K=11`，FP32/INT32/UINT32 均覆盖升序和降序；公开输出 buffer 精确为 K 个元素。[sentinel 实验](../artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json) 证实 HAU 只写 `[0,K)`，[重复键实验](../artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json) 证实相同值按自然索引递增稳定排序。SG2260E 在 codegen 阶段拒绝，当前不会进入 CModel 或 PCIe。

所有进入 TPUv7/PPL `dim4` 的 N/C/H/W extent 都必须是编译期整数并落在闭区间 `[1,65535]`；TPU 本地 scope 的闭集为 `shared/shared.dyn/local/local.fragment`。exp/sigmoid 还要求完整 shape4 一致且 `H*W<=65535`；reduction 的 EU 对齐后 padded width 也不得超过 65535。相应正负例集中在 `testing/python/jit/test_tpu_codegen_descriptor_contract.py`。

### 3.2 FP8 TPU-Kernel

[最终 FP8 CModel summary](../artifacts/2026-09-05/fp8-cmodel-final/summary.json) 为 76/76：`2 chips × 2 formats × 19 public cases`，每例均完成数值判定并保留 raw profiling trace。此前 40 项基础矩阵及若干增量实验仍是定位实现演进的辅助证据，能力判定以这份完整重跑为准。

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

### 3.3 PCIe

2026-09-05 的 SG2260E/TPU-Kernel 板端 final 数值矩阵为 140/140：

| 批次 | 结果 | 覆盖 |
| --- | ---: | --- |
| [core](../artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/core/summary.json) | 53/53 | 浮点/整数 copy、支持的本地 cast、非零 fill、FP16/BF16 GEMM、基础浮点等形四则、FP32 W broadcast、基础浮点 scalar add/mul |
| [extended](../artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/extended/summary.json) | 15/15 | FP16/BF16/FP32 exp、sigmoid、rsqrt、gather、rope |
| [reductions](../artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/reductions/summary.json) | 72/72 | 三种基础浮点 sum/max，覆盖十二个 EU 边界 width |

三批均逐 case 使用独立受控进程组和超时，未发生 timeout、retry 或 device fault。2026-09-04 的 [RV summary](../artifacts/2026-09-04/pcie-final/rv-summary.json) 仍只证明 SG2260E/RV 的 FP32 四则与 FP16 GEMM 五项数值结果；不能据此扩展 RV selector。

profiling 证据分两层记录：受监管的 [板端 profiling summary](../artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/summary.json) 表明 matmul 硬件 dispatch 和五份 raw recorder 文件采集成功，但该受监管会话内没有可用的 vendor decoder，故以 `parser_status=unavailable` 结束。随后仅在临时隔离环境安装 `bigTpuProfile==0.3.5`，对同一份 raw trace 离线解码，未再次下发板卡；[规范化 JSON](../artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/sg2260e-tpukernel-matmul-m_z_ztmr/decoded_0/tilelang_pcie_profile.json) 含 36 条有效 ns 区间：BDC 16、GDMA 20，按 opcode 为 copy 4、MM2_NN 8、data_convert 4、tensorLd 16、tensorSt 4。生产路径不会自动安装 decoder，因此 raw 采集已经闭环，而日常逐指令时间解码仍以显式提供兼容的外部 decoder 为前提。

BM1690 PCIe、FP8 PCIe 及更宽的 SG2260E/RV 能力没有对应板端证据，契约保持 `unverified`。

## 4. 特殊失败的解释

- SG2260E topk：`tpub_7_1_e` 头文件存在 HAU symbol，但运行库明确拒绝该原语。当前实现据此在 codegen 失败；历史 pre-guard CModel assertion 仅作为 guard 的依据，当前 CModel/PCIe 阶段均为 `not_applicable`。
- FP8 NT accumulate：TPU-Kernel `_R_trans(..., result_add=true)` 先由内部 ABI 4/4 验证，随后公开 `T.ppl_gemm` 路径也在两芯片、两格式上 4/4 通过。frontend 现采用 dtype-specific guard；FP16/BF16 的原始右转置 API 没有 `result_add`，仍不支持 NT accumulate。
- FP8 scalar：公开 add/mul 已由完整矩阵验证。历史 exit 139 来自 direct mixed-precision API 的非法 dtype tuple，只保留为错误调用样本，不得标成芯片不支持。
- `unverified` 不是弱支持。自动生成器不得绕过 guard，也不得把同类 dtype、另一芯片或另一运行时的结果复制过来。

## 5. 自动化消费与升级

自动化工具默认只消费 `support_status="supported"`，并根据用途额外要求 CModel 或 PCIe stage 为 `passed`。升级一项能力时：

1. 将 selector 缩到证据实际覆盖的 dtype、shape、transpose、accumulate 和 broadcast；
2. 新增带 `source_root/path/locator/observed_at` 的 evidence；
3. 先记录 declared/codegen，再记录带 oracle 的 CModel，最后才允许 PCIe；
4. 每个 PCIe case 使用独立进程组和超时，首错终止并跳过剩余测试；
5. 更新 stage 的 `scope/reason/evidence_ids`，不得顺带提升相邻 selector。

`research/artifacts/**` 被 Git 忽略，适合保留本机完整结果；可移植结论由本目录契约和 `research/tpu-backend-design/test-report.md` 汇总。

## 6. 校验

```bash
python3 -m json.tool research/tpu-op-contract/schema.json >/dev/null
python3 -m json.tool research/tpu-op-contract/contract.json >/dev/null
python3 research/tpu-op-contract/validate.py
```

`validate.py` 只依赖 Python 标准库，检查 ID 唯一性、operation/target/evidence/constraint 引用闭包、无效 target 未被 capability 引用、`passed/failed` stage 证据、可移植路径、源码 locator，以及 contract internal symbols 与 frontend、`lower.py` 闭集、AddressAssign、codegen 集合的一致性。环境装有 JSON Schema validator 时，可再将 `schema.json` 作为结构层附加校验。
