# TileLang TPU 算子验证报告

## 1. 范围与判定

本报告汇总 2026-09-04 至 2026-09-07 的已保存实验与 source-only 回归。每个数值 case 都在独立进程中 fresh-compile、加载、单次执行并与 PyTorch/精确 oracle 比较；矩阵采用首错停止。CModel 和 PCIe 是独立证据层，未执行的层级保持 `unverified`。当前 canonical 实现基线为 `e5e308797640fa2c3e789cdddd9ebed9246ddd47`；下文列出的当前 CModel 与 SG2260E PCIe summary 均记录该 revision，且 tracked implementation worktree 为 clean。

`research/artifacts/**` 已由 Git 忽略。下文只使用仓库相对路径引用实验结果；机器可读状态以 `research/tpu-op-contract/contract.json` 为准。

## 2. 结果总览

| 日期 | runtime | target | 结果 | 结论 |
| --- | --- | --- | ---: | --- |
| 2026-09-07 | CModel | `e5e3087`：SG2260E/TPU-Kernel | 141/141 | region ABI 下全部适用 case 通过；`(2,3,17)` FP32 rank-3 copy 的 102 个元素精确；topk 按芯片能力未调度 |
| 2026-09-07 | CModel | `e5e3087`：BM1690/TPU-Kernel | 147/147 | 与 SG 共同的 141 项及 6 个 K-sized、稳定重复键 topk 全部通过 |
| 2026-09-07 | CModel | `e5e3087`：两芯片/TPU-Kernel FP8 | 76/76 | 两芯片 × 两格式 × 19 个公开 case 全部通过，每例保留非空 CModel raw 命令 trace |
| 2026-09-07 | CModel profiling | `e5e3087`：三组合法 target 的核心矩阵 | 27/27 | 每组 9 项：四则、GEMM、FP16/FP32 local-roundtrip 与 S2S；每例 raw trace 非空，但本机无兼容 decoder，因而 `timed=0` |
| 2026-09-07 | 阶段性 CModel profiling | `50d8c77`：SG2260E/TPU-Kernel matmul | 1/1 | 收集 24 个 raw trace 文件、78 条 raw 命令；CModel trace 没有 duration，`timed=0` |
| 2026-09-05 | CModel | 两芯片/通用 rsqrt | 6/6 | FP16/BF16/FP32 均通过 |
| 2026-09-05 | CModel | 历史 FP8 direct `_C` 非法探针 | 0/2 | 两次 exit 139 已定位为参数违反 dtype-size predicate，不是硬件负向结果 |
| 2026-09-07 | source-only | TPU descriptor + frontend contract | 32/32 + 41/41 | descriptor 边界与 frontend alias/region 约束分别验证，非等价 alias fail-closed |
| 2026-09-07 | source-only | 远程提交前 TPU/PPL 非硬件回归 | 366 passed，4 skipped | 较 `e5e3087` 的 348/4 新增 18 个多段 case-id/portable path 契约正反例；skipped 均需显式 opt-in，默认不访问板卡 |
| 2026-09-07 | compile/link/header-only | PPL 1.7 私有 CModel/PCIe artifact | 8/8 | 7 项真实 compile/link + 1 项 SDK header/flag 检查；PCIe 只链接，不加载板卡 |
| 2026-09-07 | PCIe profiling | `e5e3087`：SG2260E/TPU-Kernel matmul | 1/1，36 条 timing | 数值、非空 raw profile 和 decoded timing 严格验收均通过；36 条区间均以 ns 计量 |
| 2026-09-07 | PCIe profiling | `e5e3087`：SG2260E/RV 核心矩阵 | 9/9，108 条 timing | 四则、GEMM 和四个 FP16/FP32 copy case 均通过数值、非空 raw profile 与 decoded timing 严格验收 |
| 2026-09-07 | PCIe profiling | `e5e3087`：SG2260E/TPU-Kernel FP8 | 38/38，134 条 timing | 两格式各 19 项均通过数值、38 个非空 raw 目录和 decoded timing 严格验收 |
| 2026-09-07 | PCIe numeric | `e5e3087`：SG2260E/TPU-Kernel | 141/141 | core 54/54、extended 15/15、reductions 72/72；三批均 `complete=true`、failed=0 |
| 2026-09-05 | 历史 PCIe | SG2260E/TPU-Kernel | 140/140 | core 53/53、extended 15/15、reductions 72/72；仅为旧实现基线的回归范围，不授权当前提交上板 |
| 2026-09-05 | 历史 PCIe profiling | SG2260E/TPU-Kernel matmul | 1 次 launch | 数值 dispatch 成功，并采集一个含五个 profile 文件的 raw 目录；既有 trace 后续在隔离 decoder 环境离线得到 36 条有效事件。会话内缺 decoder，故原始矩阵 summary 为 `complete=false` |
| 2026-09-04 | 历史 PCIe | SG2260E/RV | 5/5 | FP32 四则与 FP16 GEMM 数值通过；仅为旧实现核心竖切，不授权当前提交上板 |
| 2026-09-07 | PCIe 运行边界 | BM1690 | 未验证 | 本机没有 BM1690 板卡；SG2260E 的结果不得外推为 BM1690 PCIe 结果 |

当前 canonical 证据：

- `research/artifacts/2026-09-07/tpukernel-cmodel-e5e3087/summary.json`
- `research/artifacts/2026-09-07/fp8-cmodel-e5e3087/summary.json`
- `research/artifacts/2026-09-07/core-cmodel-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-tpukernel-matmul-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-rv-core-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-fp8-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-tpukernel-core-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-tpukernel-extended-e5e3087/summary.json`
- `research/artifacts/2026-09-07/pcie-tpukernel-reductions-e5e3087/summary.json`

过程定位与历史证据：

- `research/artifacts/2026-09-05/tpukernel-integer-copy-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-reduce-w63-w65-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json`
- `research/artifacts/2026-09-07/tpukernel-cmodel-profile-50d8c77/summary.json`
- `research/artifacts/2026-09-07/pcie-preflight-region-abi-50d8c77/summary.json`
- `research/artifacts/2026-09-05/fp8-gemm-nt-accumulate-abi-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-ppl-probe/summary.json`
- `research/artifacts/2026-09-05/tpukernel-rsqrt-generic-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-bm1690-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/core/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/extended/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/reductions/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/offline-decode-summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/sg2260e-tpukernel-matmul-m_z_ztmr/decoded_0/tilelang_pcie_profile.json`
- `research/artifacts/2026-09-04/pcie-final/rv-summary.json`

## 3. 最终 TPU-Kernel CModel 矩阵

### 3.1 case 分布

| 算子族 | 每芯片 case 数 | 选择范围 |
| --- | ---: | --- |
| copy/cast | 23 | 三种浮点同 dtype 的 local roundtrip + S2S；一个 FP32 rank-3 local roundtrip；四个 FP32 相关本地 cast；六种整数各自 local roundtrip + S2S |
| fill | 3 | FP16/BF16/FP32，值 1.25 |
| GEMM | 6 | FP16/BF16 × NN overwrite、NN accumulate、NT overwrite |
| tensor arithmetic | 16 | add/sub/mul/div × 三种浮点 dense；每个 op 增加 FP32 W broadcast |
| scalar arithmetic | 6 | add-scalar/mul-scalar × 三种浮点 |
| exp/sigmoid | 6 | 两个函数 × 三种浮点 |
| reduction | 72 | sum/max × 三种浮点 × 十二个 width |
| rsqrt | 3 | FP16/BF16/FP32 通用 `tpu_bdc_fp_rsqrt` |
| rope/gather | 6 | 两个算子 × 三种浮点 |
| topk | BM 6、SG 0 | FP32/INT32/UINT32 × 升序/降序，输出 extent=K |

共同部分合计 141 个 case；BM1690 加 6 个 topk，因此为 147。

### 3.2 精确数值范围

| 能力 | shape/输入 | oracle 与容差 |
| --- | --- | --- |
| copy/cast/fill | 通常为 `(4,32)`；另有 FP32 copy `(2,3,17)` | 浮点与六种整数 copy、cast、fill 均按目标 dtype 精确比较；rank-3 copy 的 102 个元素在两芯片均零误差 |
| tensor add/sub/mul | `lhs/out=(4,32)`；dense rhs 同形，broadcast rhs=`(4,1)` | FP32 `1e-5`；FP16 `5e-3`；BF16 `3e-2` |
| tensor div | 同上，分母为严格正值 | FP32 `1e-5`；FP16 `1e-2`；BF16 `6e-2` |
| scalar add/mul | `(4,32)`；常量 -0.25/0.75 | 对应基础 dtype 容差 |
| GEMM | `M=N=K=16`；累加初值 0.5 | FP16 atol 0.03/rtol 0.02；BF16 atol 0.15/rtol 0.03 |
| exp | `(4,32)`，输入截断至 `[-2,2]` | FP32 0.01；FP16 0.02；BF16 0.08 |
| sigmoid | `(4,32)`，输入截断至 `[-8,8]` | 同 exp；五 buffer + `coeff=(64,32)` |
| reduction | `rows=65`，width=`15,16,17,31,32,33,47,48,49,63,64,65` | max 精确；sum FP32 2e-4、FP16 3e-2、BF16 atol 0.2/rtol 0.05 |
| rsqrt | `(4,32)` 严格正输入 | FP32 atol/rtol 0.01；FP16 0.02；BF16 0.08 |
| gather | `param=(17,32)`、`index=(7,1)`、`out=(7,32)` | 224 个 payload 元素/每 dtype 精确 |
| rope | `(4,32)` | 对应基础 dtype 容差 |
| BM topk | `src=(257,)`、`dst_data=dst_idx=(11,)`，输入含重复排序键 | 六种 dtype/direction 组合的 K 个 value/index 精确且相同值保持自然索引递增；独立 sentinel 实验证实 `[K,length)` 不写 |

精确基线 summary 的 `complete=true`、`status=passed`，`scheduled_case_count=completed_case_count=passed_case_count=288`，`failed_case_count=0`。其中 SG2260E 141 项、BM1690 147 项。它证明的是所列 selector；不证明动态 shape、未列 broadcast、运行时存储重叠、NaN/Inf、零除或多核扩展。

### 3.3 编译期 descriptor 边界

`testing/python/jit/test_tpu_codegen_descriptor_contract.py` 的 32 项 source-only 正负例全部通过，确认：四种规范 local scope 都进入 copy 与 whole-buffer semantic descriptor 路径，其他 scope fail-closed；`dim4` 单维接受 65535、拒绝 0 和 65536；copy 两侧 rank 独立归一化后比较；exp/sigmoid 与 reduction 的派生 descriptor 边界被检查；全局描述符按 Var 身份而非可重复的 Buffer 名称关联。另有 `test_tpu_frontend_contract.py` 41/41，覆盖 region 的 handle dtype、BufferLoad marker、访问掩码、rank、extent、bounds、local C-axis 起点和 alias 规则。两组共同证明：`T.view/T.reshape/T.Tensor` 的 descriptor 等价 presentation alias 可作为同一表示，改变 rank/shape/dtype/scope、输出重叠或形成第二 allocation owner 的 alias 会 fail-closed。它们是编译门禁，不替代 CModel/PCIe 数值证据。

### 3.4 当前工作树的非硬件回归

2026-09-07 在本机 PPL 1.7 环境执行：

```bash
pytest -q -rs testing/python/jit/test_ppl_layout.py \
  testing/python/jit/test_tpu_*.py \
  testing/python/transform/test_tilelang_transform_address_assign.py
```

最终远程提交前结果为 `366 passed, 4 skipped`；相比 clean `e5e3087` 实现基线已执行的 348/4，新增 18 个多段 runtime case-id 与 portable path 正反契约。四个 skip 均要求显式开启真实 profiling worker，普通单测不会静默访问 CModel 或板卡。回归覆盖 PCIe decoder 依赖隔离、无硬件 preflight、身份记录，以及 FP8 matrix 的 runtime/授权/显式目标门禁。定向 SDK 测试共 `8 passed`，其中 7 项真实 compile/link、1 项检查 SG2260E RV header/flags；PCIe 用例只生成并链接私有 artifact，没有加载设备。NT accumulate 测试确认同一公开 `T.ppl_gemm(..., transpose_B=True, accumulate=True)` 在 TPU-Kernel target 的 codegen 失败，而在 SG2260E/RV target 生成 `rvt_fmm2a_nt`；direct semantic call 不能绕过 FP32 C 约束。

实现提交 `e5e308797640fa2c3e789cdddd9ebed9246ddd47` 的正式 runner 完成 TPU-Kernel CModel 288/288（SG2260E 141/141、BM1690 147/147）、FP8 CModel 76/76 和三组核心 CModel profiling 27/27。对应 summary 均记录该 revision，且 `implementation_worktree_dirty=false`。CModel profiling case 的 raw trace 均非空；由于本机没有兼容 CModel decoder，验收未启用后处理，全部 `timed_instruction_count=0`。因此 27/27 只证明 CModel 数值与 raw 收集，不表示已获得逐指令耗时。SG2260E PCIe 的当前证据单独列于第 6 节。

### 3.5 远程提交前审查结论

远程提交前审查发现的安全、契约与兼容性问题已修复，并纳入上述 source-only 回归：

- profiling 在启动时保存 PGID，不以 supervisor leader 的 `poll()` 结果代替进程组存活检查。超时、异常和“worker 成功退出但普通后代仍存活”都执行 TERM→KILL 有界回收；后一种情况会判该 case 失败。CModel、PCIe、offline decoder 和 PerfAI parser 共用这套语义。
- PCIe decoder 可使用独立的 `pcie_decoder_python` 与 `pcie_decoder_pythonpath`。后者替换而不是追加 decoder 子进程的 `PYTHONPATH`，不会进入编译/数值 worker；decoder 子进程同时删除板卡授权和 recorder 环境。严格 timing 矩阵会在任何硬件派发前执行无硬件 preflight，验证 `bigTpuProfile` 的结构化 `parse` API，并把包版本和实际 parser API 写入 summary。
- contract validator 会先检查自身支持的 schema 子集，然后核对 revision、target/capability 闭包和 runtime evidence。未知关键字、未解析引用、越界证据或不完整 artifact 均 fail-closed。
- TPU target 选择 DLPack 时会在 lowering 之前明确拒绝，避免之后以 `AssertionError` 失败。TPU host wrapper 改用位置化 `arg_<index>` C++ 标识符，因此重复或含标点的 TIR name hint 不会破坏 host ABI。
- Python 路径按项目声明的 3.8 下界回收了超出版本的联合类型和字符串 API。TVM Script 中的 `T.Tensor` 等注解必须保持为可求值对象，因此包含 `T.prim_func` 的源文件不能启用 `from __future__ import annotations`；AST 回归已锁定这条限制。
- FP8 elementwise selector 只在 case 名实际以 `-broadcast` 结尾时去掉该后缀，不再损坏裸 `add/sub/mul` 名称；六种可选后缀组合的选择回归已通过。
- FP8 runner 以 CModel 为默认模式；PCIe 必须同时给出 load/profile 两个确认、唯一显式 chip 和合法 device id。每例只在 fresh worker 中发射一次，任何编译、数值、raw 或严格 timing 错误都会保存部分 summary 并停止后续 case。

## 4. FP8 结果

最终 76 个通过 case 的笛卡尔积为：

```text
chip        = {bm1690, sg2260e}
format      = {e4m3, e5m2}
operation   = {
  copy, copy-global-to-global, fill-zero,
  cast-to-fp8, cast-from-fp8,
  add, sub, mul, add-broadcast, sub-broadcast, mul-broadcast,
  add-scalar, mul-scalar, gather, rope,
  gemm-nn-overwrite, gemm-nn-accumulate,
  gemm-nt-overwrite, gemm-nt-accumulate
}
```

FP8 copy/fill/arithmetic 使用 `(1,64)`；W broadcast 的 rhs 为 `(1,1)`。Gather 使用 `param=(17,32)`、UINT32 `index=(7,1)` 与 `output=(7,32)`，按选中 payload 的 encoded bytes 精确比较。RoPE 使用 `(4,32)`，按交错偶/奇 lane 的量化加法 oracle 比较。GEMM 使用 A=`(16,64)`、B=`(64,16)` 或 stored-B=`(16,64)`、C=`(16,16)` FP32；accumulate 从 C=1 开始。

[canonical CModel summary](../artifacts/2026-09-07/fp8-cmodel-e5e3087/summary.json) 记录实现基线 `e5e3087` 的 76/76，并为每例保留非空 CModel raw 命令 trace。较早的 40 项基础矩阵、S2S/broadcast、scalar/public NT、gather 和 RoPE 分组实验是同一公开集合的前置证据，不与最终结果相加；direct canonical `tl.tpu.gemm` NT accumulate 4/4 也仅用于早期 ABI 定位。

NT accumulate 的公开 helper 保留为 target-independent 语义，再由编程模型分流：TPU-Kernel 的 FP8 A/B + FP32 C 生成 `_R_trans(..., result_add=true)` 并匹配 `C + A @ B.T`；TPU-Kernel 的 FP16/BF16 因底层右转置 API 没有 `result_add` 而在 codegen 拒绝；SG2260E/RV 的 FP16/BF16 + FP32 C 已通过 `rvt_fmm2a_nt` 源码选择回归，但尚无 CModel/PCIe 数值结果。公开 helper 还允许基础浮点 overwrite 写入 FP32 C（NN/NT），该 selector 也尚未做精确源码与数值验证。

SG2260E PCIe 已在同一 clean revision 上重跑两种 FP8 格式各 19 项，共 38/38；每项数值通过并各自产生一个非空 raw 目录，严格解码共得到 134 条合法 ns timing。该结果只把上述精确 shape、输入域和 selector 从 CModel 提升为 SG2260E 板端证据。仍未验证的 FP8 范围包括 BM1690 PCIe、非零 fill、FP16/BF16 与 FP8 间 cast、FP8 C、异常值与更大 shape。

单次 recorder 的两种格式呈现相同 case→opcode 结构：copy 为 `tensorLd+tensorSt`，S2S 为 `tensorLd`，zero fill 为 BDC `copy+tensorSt`，cast 为 `data_convert+tensorLd+tensorSt`，binary 为对应算术 `+2 tensorLd+tensorSt`，scalar 为算术 `+tensorLd+tensorSt`，RoPE 为 `2 add+4 tensorLd+tensorSt`，gather 为 `DMA_gather`，GEMM 为 `MM2_NN/MM2_NT+2 tensorLd+tensorSt`，accumulate 再增加一次 BDC `copy`。两格式合计 opcode 数如下；时间仅用于核对映射，不是性能基准。

| opcode | 事件数 | duration 总和 | 单条范围 |
| --- | ---: | ---: | ---: |
| `MM2_NN` | 4 | 184 ns | 46 ns |
| `MM2_NT` | 4 | 304 ns | 67–85 ns |
| `add` | 10 | 118 ns | 11–12 ns |
| `copy` | 6 | 60 ns | 10 ns |
| `data_convert` | 4 | 56 ns | 14 ns |
| `mul` | 6 | 70 ns | 11–12 ns |
| `sub` | 4 | 48 ns | 12 ns |
| `DMA_gather` | 2 | 2674 ns | 1141–1533 ns |
| `tensorLd` | 60 | 15134 ns | 227–353 ns |
| `tensorSt` | 34 | 4450 ns | 114–141 ns |

### 4.1 FP8 scalar 调用契约与历史无效探针

PPL 1.7 高层 DSL 证明同型 FP8 scalar 不走显式 `tpu_bdc_fp8_*_C`，而是先把 FP32 常量 round-to-even cast 到 E4M3/E5M2，再调用通用 `tpu_bdc_fp_add_C/fp_mul_C`。两芯片 moderate 实验全部精确；生产公开路径的 E4M3/E5M2 add/mul 8/8 也通过。

边界实验给出当前精确 overflow 语义：E4M3 产生 NaN，E5M2 产生 infinity；测试中的 saturation true/false 输出相同，因此只支持默认非饱和行为，不暴露可选 saturation。

历史两次 direct E4M3 `tpu_bdc_fp8_add_C` exit 139 的根因也已闭合：探针传入 FP8 dst/src 与 FP32 `C_dtype`，违反 `sizeof(C_dtype) <= sizeof(dst_dtype)`。这两项只能记为非法参数探针，既不否定显式 mixed-precision API，也不否定芯片 FP8 scalar 能力。

## 5. topk 芯片差异

BM1690 CModel 的 FP32/INT32/UINT32 升序与降序均通过，公开 API 与 final worker 使用精确 K-sized 输出。最终矩阵的 6 个 case 均验证重复键按自然索引递增；额外的 length-sized sentinel 实验覆盖相同组合，并确认 value/index 尾部修改数均为零。因此 HAU 的已验证语义是 K-element 写入和稳定 tie-break。SG2260E 的 `tpub_7_1_e` 头文件虽然包含 `tpu_hau_sort_natural_index`，历史 pre-guard CModel 在该原语处明确 assertion。证据位于：

`research/artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json`

`research/artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json`

`research/artifacts/2026-09-05/tpukernel-sg2260e-rope-topk-cmodel/summary.json`

生产实现已经把该事实转为芯片能力 guard：SG2260E topk 在 codegen 失败，当前 CModel 与 PCIe 阶段为 `not_applicable`。不得为了“接口一致”在 SG2260E 上运行该原语。

## 6. SG2260E PCIe 当前闭环

实现基线 `e5e308797640fa2c3e789cdddd9ebed9246ddd47` 已在 SG2260E device 0 上完成当前 TPU-Kernel、RV 核心与 FP8 验收。运行前后的板卡状态均为 `Active`、利用率 0%；结束后未发现受控 profiler、runner、decoder 或 runtime 后代残留。该观测只覆盖本轮有界会话，不等价于长期稳定性证明。

TPU-Kernel 数值矩阵按三批运行：

| 批次 | 结果 | 数值范围 |
| --- | ---: | --- |
| core | 54/54 | 23 个 copy/cast、3 个非零 fill、6 个 FP16/BF16 GEMM、16 个 tensor arithmetic、6 个 scalar arithmetic |
| extended | 15/15 | 三种基础浮点各自的 exp、sigmoid、rsqrt、gather、rope |
| reductions | 72/72 | sum/max × 三种基础浮点 × 12 个 width，含 63/64/65 |

三批合计 141/141；summary 均为 `complete=true`、`status=passed`、failed=0。它们是当前 region ABI 与安全监管实现上的 PCIe 数值证据，但不包含逐指令 timing 门禁，也不外推到 BM1690、RV 扩展项、FP8 矩阵之外的 dtype/shape 或未列 scope。

profiling 与上述数值矩阵独立。当前 TPU-Kernel matmul canary 为 1/1，严格要求数值正确、规范命名且非空的 raw profile、`parser_status=ready` 和合法 decoded timing；结果解码 36 条 ns 区间，其中 BDC 16、GDMA 20。36 条事件 duration 合计 6844 ns、时间线跨度 45060 ns，opcode 为 8 `MM2_NN`、4 `copy`、4 `data_convert`、16 `tensorLd`、4 `tensorSt`。

当前 RV 核心矩阵为 9/9，共 108 条 ns timing：两个 global-to-global copy 各 1 条、两个 local-roundtrip copy 各 3 条、四个 elementwise case 各 16 条、matmul 36 条。RV matmul 的 opcode 分布与上述 TPU-Kernel matmul 相同，duration 合计 6320 ns、跨度 9896 ns；add/sub/mul/div 各由 4 条算术、8 条 load、4 条 store 组成，duration 总和依次为 3682/3198/3324/3858 ns。FP16 local-roundtrip 为 3 条、合计 498 ns、跨度 1796 ns，S2S 为 1 条、265 ns；FP32 对应为 3 条、496 ns、跨度 2685 ns，以及 1 条、303 ns。九项均满足同一严格门禁。

FP8 profiling 矩阵覆盖 E4M3、E5M2 各 19 项，共 38/38；38 个 case 各有一个规范命名且非空的 raw 目录，解码得到 134 条合法 ns timing，其中 BDC 38 条、duration 合计 840 ns，GDMA 96 条、合计 22258 ns。两种格式的 case→opcode 结构一致，完整映射和 opcode 聚合见第 4 节。以上所有 duration 均来自每例一次带 recorder 的诊断发射，不能用于后端间性能排序或回归阈值。

对应证据分别为 `research/artifacts/2026-09-07/pcie-tpukernel-matmul-e5e3087/summary.json`、`research/artifacts/2026-09-07/pcie-rv-core-e5e3087/summary.json`、`research/artifacts/2026-09-07/pcie-fp8-e5e3087/summary.json`，以及同日带 `e5e3087` 后缀的三份 TPU-Kernel numeric summary。单次 recorder 事件用于审查指令映射和定位瓶颈，不构成无 recorder 开销、warmup/repeat 条件下的性能结论。

同日较早的 preflight 曾报告 `status=Fault`、`tpu_util=100%`，当时按照 fail-stop 规则没有发射算子。后续只有在状态恢复为 `Active`、利用率 0% 后才启动上述矩阵；旧 preflight artifact 保留为过程记录，不能覆盖稍后的成功 summary。

2026-09-05 的 TPU-Kernel 140/140、2026-09-04 的 RV 5/5，以及 `44a6fc2` 的前一轮重验证仍保留为历史证据；本节当前结论只采用同 revision、clean tracked implementation 的 `e5e3087` 工件。本机没有 BM1690 板卡，故 BM1690 PCIe 保持 `unverified`。

早期 PCIe GEMM 曾暴露 software-pipeline 数据依赖冒险；改为串行 K 循环后两种编程模型均通过。这也是当前 TPU pass pipeline 禁用 software-pipeline injection 的实验证据。

## 7. 安全流程

板端测试必须遵守：

1. 精确 selector 的 CModel 已通过；
2. 板卡空闲且显式指定 device id；
3. PCIe load/profiling 分别显式授权；
4. 每个 case 经 parent-death supervisor 在新的受控进程组内运行；
5. supervisor 使用统一 deadline，超时执行 TERM→KILL→reap；外层 runner 被 SIGKILL 时也由 `PR_SET_PDEATHSIG` 清理 worker 与普通后代；
6. 任一加载错误、runtime 错误、超时、数值不符，或缺少内含规范命名且全部非空 profile 文件的 raw trace 后，终止父进程组并跳过剩余 case；只有显式要求 decoded timing 时，解析缺失/失败才属于该次验收失败；
7. 不在同一 runtime 实例盲目重试。

## 8. 尚待验证

| 优先级 | 项目 | 原因 |
| --- | --- | --- |
| P1 | BM1690 PCIe | 当前 BM1690 只有 147/147 CModel，本机没有 BM1690 板卡；取得设备后仍需独立验证 board runtime、驱动与数值路径 |
| P1 | RV BF16、NT overwrite/accumulate 与基础浮点 FP32 overwrite | source path 存在或 frontend 已接受，但精确 RV CModel/PCIe 证据不足；其中 FP32 overwrite 连精确 source-emission 结果也未记录 |
| P2 | 可复现的 decoder 发布与工件身份 | decoder-only Python/PYTHONPATH、无硬件 preflight 和包/API 身份已落地，但框架不会自动安装 vendor 包；后续应固定受支持版本集合，并把 runtime/SDK/输入身份纳入正式 manifest |
| P2 | FP8 边界扩展 | SG 板端当前只证明两格式各 19 个固定 case；还需覆盖非零 fill、FP16/BF16↔FP8 cast、FP8 C、异常值和更广 shape |
| P2 | tail、异常值、alias 与更广 shape | 当前矩阵是静态、规则形状和受控输入域；TopK 写入 tail 已单独闭合，其他 op 仍需覆盖 |
| P2 | 多核分片与依赖安全 pipeline | 当前只证明单核串行语义 |

任何新增实验都应先更新 case registry，再把精确 scope 写入 `research/tpu-op-contract/contract.json`；不得只在叙述报告中扩大支持范围。
