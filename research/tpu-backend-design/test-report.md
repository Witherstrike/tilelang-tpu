# TileLang TPU 算子验证报告

## 1. 范围与判定

本报告汇总 2026-09-04 至 2026-09-05 的已保存实验。每个数值 case 都在独立进程中 fresh-compile、加载、单次执行并与 PyTorch/精确 oracle 比较；矩阵采用首错停止。CModel 和 PCIe 是独立证据层，未执行的层级保持 `unverified`。

`research/artifacts/**` 已由 Git 忽略。下文只使用仓库相对路径引用实验结果；机器可读状态以 `research/tpu-op-contract/contract.json` 为准。

## 2. 结果总览

| 日期 | runtime | target | 结果 | 结论 |
| --- | --- | --- | ---: | --- |
| 2026-09-05 | CModel | SG2260E/TPU-Kernel | 140/140 | final 全部适用 case 通过；topk 按芯片能力未调度 |
| 2026-09-05 | CModel | BM1690/TPU-Kernel | 146/146 | final 共同矩阵与 6 个 K-sized、稳定重复键 topk case 全部通过 |
| 2026-09-05 | CModel | 两芯片/TPU-Kernel FP8 | 76/76 | 两芯片 × 两格式 × 19 个公开 case 全部通过，并保留 raw profiling trace |
| 2026-09-05 | CModel | 两芯片/通用 rsqrt | 6/6 | FP16/BF16/FP32 均通过 |
| 2026-09-05 | CModel | 历史 FP8 direct `_C` 非法探针 | 0/2 | 两次 exit 139 已定位为参数违反 dtype-size predicate，不是硬件负向结果 |
| 2026-09-05 | source-only | TPU descriptor contract | 14/14 | 四种 local scope；dim4 的 0/65535/65536；copy 归一化；exp/sigmoid H*W；reduction padded width |
| 2026-09-05 | PCIe | SG2260E/TPU-Kernel | 140/140 | core 53/53、extended 15/15、reductions 72/72；无 timeout、retry 或 device fault |
| 2026-09-05 | PCIe profiling | SG2260E/TPU-Kernel matmul | 1 次 launch | 数值 dispatch 与五份 raw trace 采集成功；既有 trace 后续在隔离 decoder 环境离线得到 36 条有效事件 |
| 2026-09-04 | PCIe | SG2260E/RV | 5/5 | FP32 四则与 FP16 GEMM 数值通过；只证明该核心竖切 |

证据摘要：

- `research/artifacts/2026-09-05/tpukernel-sg2260e-cmodel-final/summary.json`
- `research/artifacts/2026-09-05/tpukernel-bm1690-cmodel-final/summary.json`
- `research/artifacts/2026-09-05/tpukernel-integer-copy-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-reduce-w63-w65-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-topk-tail-semantics-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-topk-stable-ties-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-cmodel-final/summary.json`
- `research/artifacts/2026-09-05/fp8-gemm-nt-accumulate-abi-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-ppl-probe/summary.json`
- `research/artifacts/2026-09-05/tpukernel-rsqrt-generic-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-cmodel/summary.json`
- `research/artifacts/2026-09-05/fp8-scalar-bm1690-cmodel/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/core/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/extended/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-final/reductions/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/summary.json`
- `research/artifacts/2026-09-05/tpukernel-sg2260e-pcie-profiling-final/sg2260e-tpukernel-matmul-m_z_ztmr/decoded_0/tilelang_pcie_profile.json`
- `research/artifacts/2026-09-04/pcie-final/rv-summary.json`

## 3. 最终 TPU-Kernel CModel 矩阵

### 3.1 case 分布

| 算子族 | 每芯片 case 数 | 选择范围 |
| --- | ---: | --- |
| copy/cast | 22 | 三种浮点同 dtype 的 local roundtrip + S2S；四个 FP32 相关本地 cast；六种整数各自 local roundtrip + S2S |
| fill | 3 | FP16/BF16/FP32，值 1.25 |
| GEMM | 6 | FP16/BF16 × NN overwrite、NN accumulate、NT overwrite |
| tensor arithmetic | 16 | add/sub/mul/div × 三种浮点 dense；每个 op 增加 FP32 W broadcast |
| scalar arithmetic | 6 | add-scalar/mul-scalar × 三种浮点 |
| exp/sigmoid | 6 | 两个函数 × 三种浮点 |
| reduction | 72 | sum/max × 三种浮点 × 十二个 width |
| rsqrt | 3 | FP16/BF16/FP32 通用 `tpu_bdc_fp_rsqrt` |
| rope/gather | 6 | 两个算子 × 三种浮点 |
| topk | BM 6、SG 0 | FP32/INT32/UINT32 × 升序/降序，输出 extent=K |

共同部分合计 140 个 case；BM1690 加 6 个 topk，因此为 146。

### 3.2 精确数值范围

| 能力 | shape/输入 | oracle 与容差 |
| --- | --- | --- |
| copy/cast/fill | `(4,32)` | 浮点与六种整数 copy、cast、fill 均按目标 dtype 精确比较 |
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

两份 final summary 的 `complete=true`、`status=passed`、completed 与 scheduled 数量一致，failed 为 0。它们证明的是所列 selector；不证明动态 shape、未列 broadcast、alias、NaN/Inf、零除或多核扩展。

### 3.3 编译期 descriptor 边界

`testing/python/jit/test_tpu_codegen_descriptor_contract.py` 的 14 项 source-only 正负例全部通过，确认：四种规范 local scope 都进入同一 LMEM descriptor 路径，其他 scope fail-closed；`dim4` 单维接受 65535、拒绝 0 和 65536；copy 的两侧 rank 独立归一化后比较；exp/sigmoid 拒绝 `H*W=65536`；reduction 拒绝 EU 对齐后宽度变为 65536 的派生 descriptor。该组测试证明编译门禁，不替代任何 CModel/PCIe 数值证据。

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

[canonical summary](../artifacts/2026-09-05/fp8-cmodel-final/summary.json) 记录 76/76，并为每例保留 raw trace。较早的 40 项基础矩阵、S2S/broadcast、scalar/public NT、gather 和 RoPE 分组实验是同一公开集合的前置证据，不与最终结果相加；direct canonical `tl.tpu.gemm` NT accumulate 4/4 也仅用于早期 ABI 定位。

NT accumulate 的公开 helper 现按 dtype 分流：FP8 A/B + FP32 C 生成 `_R_trans(..., result_add=true)` 并匹配 `C + A @ B.T`；FP16/BF16 因底层右转置 API 没有 `result_add` 而继续保持 unsupported。

未验证范围包括 PCIe、非零 fill、FP16/BF16 与 FP8 间 cast、FP8 C、异常值与更大 shape。

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

## 6. SG2260E PCIe 核心闭环

2026-09-05 的 TPU-Kernel final 使用与 CModel 相同的公开 case registry，在 SG2260E device 0 完成 140/140：

| 批次 | 结果 | 数值范围 |
| --- | ---: | --- |
| core | 53/53 | 22 个 copy/cast、3 个非零 fill、6 个 FP16/BF16 GEMM、16 个 tensor arithmetic、6 个 scalar arithmetic |
| extended | 15/15 | 三种基础浮点各自的 exp、sigmoid、rsqrt、gather、rope |
| reductions | 72/72 | sum/max × 三种基础浮点 × 12 个 width，含 63/64/65 |

三个 summary 均为 `complete=true`、`status=passed`、failed=0；监管器未记录 timeout、retry 或 device fault。这里的 140 项只提升契约中逐项列出的 SG2260E/TPU-Kernel `pcie_numeric_passed`，不提升 FP8、BM1690 或 RV 的相邻 selector。

profiling 与数值矩阵独立。受监管的 matmul profiling 硬件 dispatch 数值执行成功并生成五份 `cdm_profile_data_dev0-0` raw 文件，但该会话内没有可用的 `bigTpuProfile/PerfAI`，所以 summary 按严格解析准则记录 `parser_status=unavailable`，不能伪装成完整 profiling pass。随后仅在临时隔离环境安装 `bigTpuProfile==0.3.5`，对既有 raw trace 离线解码，未再次下发板卡；规范化结果包含 36 个有效 ns 区间：BDC 16、GDMA 20，opcode 分布为 copy 4、MM2_NN 8、data_convert 4、tensorLd 16、tensorSt 4。生产框架有意不自动安装依赖，因此 recorder/raw 采集已验证，常规逐指令解码仍要求用户显式配置兼容 decoder。

单次 recorder 事件用于审查指令映射和定位瓶颈，不构成无 recorder 开销、warmup/repeat 条件下的性能结论。

早期 PCIe GEMM 曾暴露 software-pipeline 数据依赖冒险；改为串行 K 循环后两种编程模型均通过。这也是当前 TPU pass pipeline 禁用 software-pipeline injection 的实验证据。

## 7. 安全流程

板端测试必须遵守：

1. 精确 selector 的 CModel 已通过；
2. 板卡空闲且显式指定 device id；
3. PCIe load/profiling 分别显式授权；
4. 每个 case 在新的受控进程组内运行；
5. supervisor 使用统一 deadline，超时执行 TERM→KILL→reap；
6. 任一加载错误、runtime 错误、超时、数值不符或 trace 解析失败后，终止父进程组并跳过剩余 case；
7. 不在同一 runtime 实例盲目重试。

## 8. 尚待验证

| 优先级 | 项目 | 原因 |
| --- | --- | --- |
| P0 | FP8 SG PCIe | 76/76 CModel 不能替代真实板端 FP8 执行；应按 copy/cast、arithmetic、GEMM、gather/rope 分批首错停止 |
| P1 | BM1690 PCIe | 当前 BM1690 只有 146/146 CModel；板端 runtime、驱动和稳定性仍无证据 |
| P1 | RV FP16/BF16 与 NT overwrite | source path 存在但精确 RV CModel/PCIe 证据不足 |
| P2 | 可部署的 profiling decoder | raw 采集已闭环，离线 36-event 解码只在隔离环境验证；若要求常规 timing，需要显式供应并锁定兼容 vendor decoder |
| P2 | tail、异常值、alias 与更广 shape | 当前矩阵是静态、规则形状和受控输入域；TopK 写入 tail 已单独闭合，其他 op 仍需覆盖 |
| P2 | 多核分片与依赖安全 pipeline | 当前只证明单核串行语义 |

任何新增实验都应先更新 case registry，再把精确 scope 写入 `research/tpu-op-contract/contract.json`；不得只在叙述报告中扩大支持范围。
