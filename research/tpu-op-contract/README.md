# TileLang TPU 算子能力契约

本目录提供一份供人和自动化工具共同使用的当前能力事实：

- `schema.json` 定义数据结构；
- `contract.json` 记录算子语义、目标组合、失败策略与逐阶段证据；
- `validate.py` 执行结构、语义闭包和本地运行工件校验。

契约严格区分“前端可表达”“目标代码生成通过”“CModel 数值通过”“PCIe 数值通过”和“取得逐指令 timing”。低层 SDK 有声明、源码能生成或 CModel 通过，都不能单独推出真实板卡支持。

## 1. 目标模型

| target id | 芯片 | 编程模型 | PPL arch | 物理核数 | 是否有效 |
| --- | --- | --- | --- | ---: | --- |
| `bm1690.tpukernel` | BM1690 | TPU-Kernel | `tpub_7_1` | 8 | 是 |
| `bm1690.rv` | BM1690 | RV Tensor | — | — | 否，target 解析阶段拒绝 |
| `sg2260e.tpukernel` | SG2260E | TPU-Kernel | `tpub_7_1_e` | 4 | 是 |
| `sg2260e.rv` | SG2260E | RV Tensor | `tpub_7_1_e` | 4 | 是 |

`cmodel/pcie` 是执行模式，不是 target，也不参与算子或指令选择。契约只为三个有效 target 建立结果。

运行资料按可信度分层：PPL 1.7 手册及 `tpub_7_1/tpub_7_1_e` 头文件说明 selector；TileLang 源码与 codegen 证明接入；CModel/PCIe 数值工件才证明精确 case。`.local_content/tpu_kernel_manual.md` 是通用 TPU-Kernel API 参考，不能据此推导 TPUv7、FP8 或某个 target 的可执行性。

## 2. 状态与匹配规则

capability 由 `operation_id + variant + dtype_bindings + constraints` 唯一限定，并按 target 记录：

1. `declared`：公开 TileLang 前端能否表达；
2. `codegen_passed`：精确 selector 是否通过目标相关 lowering/source emission；
3. `cmodel_numeric_passed`：CModel 输出是否通过指定 oracle；
4. `pcie_numeric_passed`：真实芯片输出是否通过指定 oracle；
5. `sdk_declared`：可选，只说明底层手册或头文件有声明。

阶段状态包括 `passed/historical_passed/failed/unverified/not_applicable`。`historical_passed` 仅保留旧基线硬件事实，不能授权当前实现上板。`support_status="supported"` 也不等价于数值或 PCIe 已通过；调度器必须额外检查所需执行阶段为 `passed`。

匹配必须覆盖完整的 dtype、shape、transpose、accumulate、broadcast、variant 与 target。不得只按 operation id 或 dtype 族模糊匹配，也不得从相邻 dtype、芯片、后端或执行模式外推。

### 2.1 Semantic buffer ABI

全局 invariant `semantic-buffer.region-abi` 覆盖 16 个 operation。每个 typed semantic buffer 参数必须以 `tl.region(BufferLoad, access_mask, logical_extents)` 跨越 TPU 语义边界；不保留裸 `tir.tvm_access_ptr` 兼容入口。非 copy op 要求零起点 whole-buffer region；copy 可以携带显式子区间，但必须满足连续 Ramp、bounds 与 local C-axis 起点约束。

native 层把 dtype、原始 rank、shape、scope 和 storage identity 与 compiler-owned descriptor 逐项核对。改变这些属性的 view/reshape/重新包装会 fail-closed；只有 descriptor 完全等价、且不引入第二种表示的 presentation alias 合法。

## 3. 当前 canonical 回归

当前契约绑定完整 revision `e5774525e3a6e11d0d6010e979203c55181a8872` 和 source state `c49cf3594f2ce5d215e54d331a10ca0bbbf92bd6667fba3e172b2b3621f7f5c4`。canonical 根目录为 [`research/artifacts/2026-09-09/final-e5774525`](../artifacts/2026-09-09/final-e5774525/)。正式集合共 39 份 summary；它们均满足 `complete=true`、全部 case `passed`、`implementation_worktree_dirty=false`，且源码身份范围为 tracked 与 untracked 文件（排除 `research/**`）。

### 3.1 CModel：553/553

| summary | target 分布 | 结果 |
| --- | --- | ---: |
| [core BM1690](../artifacts/2026-09-09/final-e5774525/core-bm1690-cmodel/summary.json) | TPU-Kernel 28 | 28/28 |
| [core SG2260E](../artifacts/2026-09-09/final-e5774525/core-sg2260e-cmodel-retry1/summary.json) | TPU-Kernel 28 + RV 28 | 56/56 |
| [FP8 BM1690](../artifacts/2026-09-09/final-e5774525/fp8-bm1690-cmodel/summary.json) | TPU-Kernel 42 | 42/42 |
| [FP8 SG2260E](../artifacts/2026-09-09/final-e5774525/fp8-sg2260e-cmodel-retry1/summary.json) | TPU-Kernel 42 | 42/42 |
| [TPU-Kernel BM1690](../artifacts/2026-09-09/final-e5774525/tpukernel-bm1690-cmodel/summary.json) | TPU-Kernel 152 | 152/152 |
| [TPU-Kernel SG2260E](../artifacts/2026-09-09/final-e5774525/tpukernel-sg2260e-cmodel-retry1/summary.json) | TPU-Kernel 146 | 146/146 |
| [demo BM1690](../artifacts/2026-09-09/final-e5774525/demo-bm1690-cmodel/summary.json) | TPU-Kernel 36 | 36/36 |
| [demo SG2260E](../artifacts/2026-09-09/final-e5774525/demo-sg2260e-cmodel-retry1/summary.json) | TPU-Kernel 36 + RV 15 | 51/51 |

core 矩阵直接覆盖 copy、FP16/BF16/FP32 四则与 broadcast、max 以及 GEMM；matmul 内的 `T.ppl_fill(C_acc, 0)` 同时覆盖零 fill。FP8 矩阵在两芯片各覆盖 E4M3/E5M2 的 21 项，共 42 项；包括同格式 copy、零 fill、FP32 双向 cast、dense/W-broadcast add/sub/mul/max、scalar add/mul、gather、rope，以及 NN/NT overwrite/accumulate GEMM。

BM1690 与 SG2260E 的 TPU-Kernel 基础差异是 topk：BM 的 152 项包含 FP32/INT32/UINT32 升降序 topk，SG 的运行库明确拒绝该原语，因此在 codegen fail-closed，不进入 CModel/PCIe。

### 3.2 PCIe：295/295

| summary | target 分布 | 结果 |
| --- | --- | ---: |
| [core SG2260E](../artifacts/2026-09-09/final-e5774525/core-sg2260e-pcie/summary.json) | TPU-Kernel 28 + RV 28 | 56/56 |
| [FP8 SG2260E](../artifacts/2026-09-09/final-e5774525/fp8-sg2260e-pcie/summary.json) | TPU-Kernel 42 | 42/42 |
| [TPU-Kernel SG2260E：14 个分片](../artifacts/2026-09-09/final-e5774525/tpukernel-sg2260e-pcie-shards/) | TPU-Kernel 146 | 146/146 |
| [demo SG2260E：15 个分片](../artifacts/2026-09-09/final-e5774525/demo-sg2260e-pcie-shards/) | TPU-Kernel 36 + RV 15 | 51/51 |

39 份 canonical summary 的实际 launch 总数是 `553 + 295 = 848`。两个分片集合均经过精确并集校验，没有重复、遗漏或额外 case。848 是验收执行次数，不是互斥 capability 数；core、TPU-Kernel 专项和 demo 之间存在 selector 重叠，不能重复解释为新增能力。

早先并发启动的 SG2260E CModel 结果不满足顺序晋级要求；PCIe 的失败整批、失败分片和恢复 canary 也不构成完整闭集。这些工件保留用于诊断，但不参与 848 的计数，也不被 `evidence[]` 中带 `runtime_expectation` 的条目引用。

FP8 的 42 个 SG2260E PCIe case 已全部实证，包含两种格式的 dense 与 W-broadcast max。该结论只覆盖 summary 中的有限、可精确判定输入与固定 shape；不得外推到 NaN、infinity、signed-zero tie、动态 shape、BM1690 PCIe 或 RV FP8。

### 3.3 demo 与后端边界

demo summary 是 mixed-backend 工件：运行时由 summary 顶层字段和 `numeric.runtime_mode` 共同给出；每条 `results[]` 直接记录 `chip/programming_model/status/key`。TPU-Kernel 36 项覆盖 elementwise、matmul、rmsnorm、rmsnorm-splitk、rope、swiglu、flashattn；RV 15 项只覆盖 elementwise 与 matmul 的直接 selector。

这些 demo 只能给已验证的 selector/stage 提供证据。TPU-Kernel composite 通过不能提升 RV 的 rmsnorm、split-k、rope、swiglu 或 flashattn；demo 中 FP32 用户输出若内部使用 BF16 计算，也不能据此提升“FP32 A/B 直接 GEMM”能力。

`validate.py` 对 mixed-backend summary 逐条重建 canonical key `runtime/chip/backend/case_id`，并 fail-closed 检查：

- 必需字段缺失、未知 backend 或 top-level/record backend 冲突；
- runtime 不一致、非 `passed` 状态、重复 key 或重复 canonical case；
- case 总数、target 分布、精确 required case 集与契约 expectation 不一致。

其余使用 `cases{}` 或 `results[]` collection 的 summary 也按同样的闭集规则校验；`cases{}` 本身不表示单后端，例如 SG core 同时包含 TPU-Kernel 与 RV。

### 3.4 profiling 的证据含义

profiling 与数值通过是两个正交维度：

- CModel core/FP8/demo 保留 raw trace，但当前 decoder 不可用，`timed_instruction_count=0`；它们只能证明指令选择和数值结果。
- SG2260E PCIe core、FP8、demo 分别得到 824、150、2594 个有效 ns interval；共 149 个 profiling case、3568 个 interval，parser 均为 `ready`，decoder 为 `bigTpuProfile 0.3.5`。
- TPU-Kernel 146 项的 14 份 PCIe 分片是数值回归，没有 decoded timing；不能因为同一 target 的其他矩阵启用了 profiling 而补写 timing。

单次 timing 用于核对实际指令与定位退化，不是稳定 benchmark。PCIe runner 对每个 case 使用独立进程组、超时与 TERM→KILL 有界清理；每次执行后要求板卡连续两次报告 0% utilization，首错即停止并跳过余项。

## 4. canonical 与历史特征证据

canonical 回归负责授权当前 revision 的数值阶段；较早工件只在它提供不可替代的语义或失败边界时保留，不与 canonical launch 数累计，也不覆盖当前授权：

- [topk tail sentinel](../artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json)：证明只写 `[0,K)`；
- [topk stable ties](../artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json)：证明相同值按自然索引稳定排序；
- [SG topk rejection](../artifacts/2026-09-05/tpukernel-sg2260e-rope-topk-cmodel/summary.json)：解释当前 fail-closed guard；
- [FP8 scalar PPL probe](../artifacts/2026-09-05/fp8-scalar-ppl-probe/summary.json)：刻画合法 scalar lowering 与 boundary 行为；历史 direct `tpu_bdc_fp8_*_C` exit 139 只是非法 dtype tuple 的反例，不代表芯片不支持。

被 canonical clean-run 覆盖、且不再提供独立语义的 dirty max 探针不属于当前契约证据。

## 5. 自动化消费与升级

自动化工具应遵守以下规则：

- CModel 调度要求 `support_status="supported"` 且 `cmodel_numeric_passed.status="passed"`；上板还要求 `pcie_numeric_passed.status="passed"`。
- `unverified/not_applicable/unsupported/experimental/historical_passed` 均不得作为当前上板授权。
- `target_results` 没有某 target，表示该 operation 不适用该组合，不是默认支持。
- runtime evidence 的 `claim_targets` 与 `capability_ids` 只定义候选闭集；stage 还必须匹配 runtime、revision、target、capability 和精确 case 语义。
- `sdk_declared=passed` 只用于排定实现候选，不能提升任何 TileLang 阶段。

升级能力时，应先收窄 selector，再依次补充 frontend/codegen、CModel、PCIe 证据。每份 runner-recorded 工件必须带完整 Git revision、干净实现身份、target case counts 和 required case manifest。`research/artifacts/**` 被 Git 忽略；在没有本地工件的 clone 中，普通校验检查静态闭包，验收机用 strict-local 模式校验工件内容。

## 6. 校验

```bash
python3 -m json.tool research/tpu-op-contract/schema.json >/dev/null
python3 -m json.tool research/tpu-op-contract/contract.json >/dev/null
python3 research/tpu-op-contract/validate.py
python3 research/tpu-op-contract/validate.py --require-local-artifacts
```

`validate.py` 只依赖 Python 标准库，并执行项目使用的 JSON Schema 2020-12 子集。除 JSON 结构外，它还检查 ID/引用闭包、target 与 `TPU_CHIP_SPECS` 一致性、capability 完整 target 覆盖、frontend 声明的 target 无关性、operand effect/direction、阶段顺序、portable path、源码 locator、内部 symbol 与 frontend/lowering/AddressAssign/codegen registry 一致性，以及上述 runtime evidence 闭集。
