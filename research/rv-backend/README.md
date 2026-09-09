# TileLang-TPU 双后端设计报告

## 1. 目标与结论

本实现把 TPU 支持拆为互不混淆的三条选择轴：物理芯片、设备编程模型和 host 运行时。用户仍以 `T.ppl_*` 编写 copy、fill、GEMM 和逐元素算子；这些入口现在先生成中立的 `tl.tpu.*` TIR，再由后端选择传统 TPU-Kernel 或 SG2260E RV Tensor（RVT）指令。

当前已覆盖的中立算子是 `copy`、`fill`、`gemm`、`add`、`sub`、`mul`、`div` 和
`max`。面向用户的统一入口与 36 个高层 case 位于 `tpu_demo/run.py`、
`tpu_demo/cases.py`；逐元素四则与 matmul 的 FP16/BF16/FP32 共 15 个 case 可选择 RV，
完整使用与晋级流程见 `tpu_demo/README.md`。

这里的 “PPL” 仅指 PPL 1.7 SDK/ABI；它不是 TileLang 的编程模型名称。源码按职责拆为 `codegen_tpu_common.cc`、`codegen_tpukernel.cc` 和 `codegen_rv.cc`。

## 2. 选择模型与硬件边界

| 轴 | 取值 | 作用 |
| --- | --- | --- |
| 芯片 | `bm1690`、`sg2260e` | 选择 PPL arch、编译宏、物理核数与能力集。 |
| 编程模型 | `tpukernel`、`rv` | 选择指令 ABI 与 codegen。BM1690 仅允许前者；SG2260E 允许两者。 |
| 运行时 | `cmodel`、`pcie` | 选择模拟器或板端 runtime；不改变指令语义。 |

Python 编译编排以 `TPUChipSpec` 为能力注册表：BM1690 映射到 `tpub_7_1`、8 个物理核；SG2260E 映射到 `tpub_7_1_e`、4 个物理核。两者在本项目当前使用的 TPUv7 LMEM 几何相同，故共享内存档案；PPL arch、宏、物理核数和 RV 能力不能由目录名或默认 BM1690 假设推断。native codegen 边界目前仍镜像校验受支持的 chip/model 组合，后续应由同一份声明生成，消除人工同步点。

物理核数不等于当前 kernel 的并行发射核数。CModel 会按 4/8 核拓扑初始化，但现有 host 模板的 `core_num=1`，示例也以串行 grid 工作负载验证语义。因此本报告只声称单核工作负载在正确拓扑下运行；多核切分、跨核同步和性能扩展仍是后续工作。

公共 API 只保留一份编译身份：完整 TVM Target 解析为
`TPUTargetSpec(chip, programming_model)`。host 承载方式单独解析为
`TPURuntimeConfig(runtime_mode)`，不参与指令选择。裸 `target="tpu"`、缺少 `-mcpu` 或
`-tpu-programming-model` 的 target，以及 BM1690+RV 组合都会在 lowering 前失败。

```python
kernel = tilelang.compile(
    program,
    target=("tpu -mcpu=sg2260e "
            "-tpu-programming-model=rv"),
    runtime_mode="cmodel",
)
```

## 3. 编译与代码生成架构

```text
T.ppl_* 前端
    │  生成 tl.tpu.{copy,fill,gemm,add,sub,mul,div,max}
    ▼
TileLang TPU lowering + AddressAssign
    │  target: tpu -mcpu=<chip> -tpu-programming-model=<model>
    ├── tpukernel ──► codegen_tpukernel.cc ──► tpu_kernel.h ABI
    └── rv        ──► codegen_rv.cc        ──► rvt_api.h ABI
                         │
                         ▼
                 PPL 1.7 编译/链接 ──► CModel 或 PCIe host
```

`target.build.tilelang_tpu` 是唯一 FFI 入口。TPU codegen 限制一个模块只含一个
`PrimFunc`，并拒绝将保留的 host 入口名当作 device kernel，避免 ABI 歧义。

共同层负责解析 TIR 参数、region、dtype、LMEM 地址和读写 effect；专属层只做指令选择。此分层使同一 `tl.tpu.*` 语义能有两种指令实现，并避免把 RV descriptor 细节泄漏到前端。历史 `ppl.*` 和 raw `tpu_*` TIR extern 已从支持 ABI 中移除，任何 TPU target 都会拒绝；原始 `rvt_*` extern 只作为专家级 ABI escape hatch。raw RVT 与 `tl.tpu.*`、`tl.tpukernel.*` 或 TPU-Kernel 生命周期的混用会在 lowering/codegen 边界失败，而不是生成含义不明的命令流。

### 3.1 已实现的算子契约

| 前端 | 统一 TIR | TPU-Kernel | RVT | 当前约束 |
| --- | --- | --- | --- | --- |
| `ppl_copy` | `tl.tpu.copy` | S2L/L2S/S2S/L2L GDMA/BDC；本地 cast | `rvt_dma_ld/st/cp`，本地浮点转换用 `rvt_cvt_f2f` | 两端静态 region 的规范化 N/C/H/W extent 必须相同。SG2260E/RV 的 FP16/FP32 G2L→L2L→L2S 与独立 S2S 已在 `(4,32)` 精确 CModel/PCIe 验证。跨 dtype 仅本地；TPU-Kernel 拒绝 FP16↔BF16。RV 转换类型闭集包含 FP16/BF16/FP32，但当前只有 FP16→BF16 精确 codegen 证据，两向均无数值证据。 |
| `ppl_fill` | `tl.tpu.fill` | `tpu_bdc_set_C` | typed CR + `rvt_cp` | RV 当前只开放零填充，用于累加器初始化。 |
| `ppl_gemm` | `tl.tpu.gemm` | `tpu_bdc_fp_mm` / right-transpose 变体；另有已验证的同格式 FP8 A/B + FP32 C 路径 | `rvt_fmm2[a]_{nn,nt}` | RV 当前只映射 FP16/BF16；local N=H=1；无 `transpose_A`；M/N/K 在 `[1,65535]`。`accumulate` 显式表达读写 C；NT accumulate 的 `rvt_fmm2a_nt` 已有精确源码选择回归但尚无数值证据。TPU-Kernel 的 FP8 NT accumulate 是独立 selector，不能外推给 RV。 |
| `ppl_add/subtract/mul/div/max` | `tl.tpu.{add,sub,mul,div,max}` | 四则为 `tpu_bdc_fp_*`，max 为 `tpu_bdc_max`；add/sub/mul/max 另有已验证候选 FP8 路径 | `rvt_fadd/fsub/fmul/fdiv/fmax` | RV 当前限同 dtype FP16/BF16/FP32；local N=H=1；右操作数可作 W 维广播。RV broadcast 用逻辑 `(M,W)`、W stride=0 的 FREE_LAYOUT descriptor，避免把 `(M,1)` 错当成由 peer 自动扩展。`div` 配置 RV rsqrt 迭代并以容差验证。TPU-Kernel FP8 div 与全部 RV FP8 映射仍未开放。 |

算子不支持的 dtype、形状、布局、尾块和别名组合必须在编译期报错；不以静默 fallback 或错误指令换取“可编译”。AddressAssign 以写/读写 effect 区分 GEMM 的覆盖与累加，保证 C 的 bank/生存期分析与指令语义一致。

### 3.2 RV Tensor 指令映射

RVT codegen 直接遵循 SG2260E PPL 1.7 的 `rvt_api.h`：CR 为 R0–R7、TR 为 R8–R31、GR 为 R32–R39；descriptor 的精度字段统一由 `PRECISION(DT_*)` 构造，浮点 subtype 使用 `FP8TYPE(DT_*)`，整数符号字段使用 `SIGN(DT_*)`，不硬编码内部编码。global/local DMA subview 使用显式 FREE layout/stride；矩阵与逐元素本地 tile 使用其要求的 HW-aligned descriptor。

RV kernel 生命周期为 `rvt_kernel_start()`，先配置 GDMA lane mask，执行 descriptor/DMA/计算，最后 `rvt_sync_i(0xdeadbeef, 0)`。它不混入 TPU-Kernel 的初始化或轮询协议。GEMM 的累加位直接选 `rvt_fmm2a_nn`/`rvt_fmm2a_nt`，覆盖语义选 `rvt_fmm2_nn`/`rvt_fmm2_nt`；除法在发射 `rvt_fdiv` 前设置 rsqrt 迭代次数。

更详细的 SDK 生成代码审阅和 descriptor 例子见 ignored 的本地实验记录
`research/artifacts/2026-09-04/isa-reference/README.md`；原始规范来源为
`.local_content/rv_tensor_extension.md`。

## 4. 工具链与运行时隔离

`ppl_layout.py` 只解析 PPL 1.7 的 `deps/` 发布布局。它从 `TPUChipSpec` 取得 arch 和核数，
检查 kernel header、helper、CModel runtime、固件/模拟器与 RV header；当 SG2260E target
选择 RV 而 SDK 未提供 `rvt_api.h` 时，构建会在调用工具链前失败。

CModel 使用 SDK runtime；PCIe 使用安装在板端环境中的 `libtpuv7_rt.so`，禁止误把
CModel runtime 当作板端 runtime。一次 JIT 加载会绑定 `TPUTargetSpec`、
`TPURuntimeConfig`、device id 与 SDK/runtime 路径；切换其中任一身份都会被拒绝，防止同一
进程混用 CModel/PCIe、BM/SG 或两个 ABI。

## 5. Profiling 与板端安全

PPL 的 `--profiling` 会改变 PPL 自身的 host/codegen 流程；TileLang 已自行生成 device source 与 host wrapper，不能简单转发该参数。实现复用的是 PPL/TPUDNN 的记录协议：`TPUInstructionProfiler` 在独立 worker 中 fresh-compile、加载、单次 dispatch、收集 raw trace，并将可用的 PerfAI/`bigTpuProfile` 输出投影为稳定 JSON。CModel 记录命令文本；PCIe recorder 可提供逐条命令 duration。

PCIe 默认拒绝加载。profiling 矩阵必须显式确认 `--allow-pcie` 与
`--allow-pcie-profile`；纯数值 TPU-Kernel 矩阵使用独立的 `--allow-pcie-load` 确认。两类路径都
要求唯一 device id，并由父死亡信号、私有进程组与统一 deadline 监管。SIGTERM、SIGKILL、最终
reap 和 pipe drain 均有硬时限；任一 worker 超时、失败，或成功退出后仍遗留同进程组后代，矩阵
立即清理并停止，不继续向状态未知的板卡发射命令。profiling 数值/raw 验收还要求
`cdm_profile_data_dev*` 中存在规范命名且非空的 `global.profile` 或 `cdmlibN_N.profile`；增加
`--require-decoded-timing` 后，decoder ready、非空 timing 与全部区间合法也成为硬门禁。每次
矩阵只删除自己创建的 JIT scratch，并保留 raw/decoded 报告。带 recorder 的 host 墙钟时间不能
当作无扰动 kernel 性能。

## 6. 已知限制与后续路线

1. 当前 portable contract 是核心闭环，不是完整 TileLang 二级算子集。仍缺 reduction、广播的一般形式、常量逐元素、激活/归一化、异步 token/barrier、tail planner 与更丰富 dtype。
2. RV fill 暂限零，GEMM 暂限二维 FP16/BF16 输入；更广的 RV ISA 应逐条写入 descriptor/layout/synchronization 契约并配数值回归，不能将 raw ABI 调用算作算子支持。
3. 本轮正确性示例故意使用串行 K 循环。现有通用 software-pipeline pass 会把 `num_stages=1` 的生产者 DMA 与立即消费它的 GEMM 放入 TPU 并行区，PCIe 已证明这会产生数据冒险；后续须建立 TPU dependency/hazard 模型后才可重新开放重叠。
4. 当前工作负载单核发射。后续需要以 tile partitioner 分配 4/8 核、明确 core-local 地址与跨核同步，再建立按核心的数值和缩放测试。
5. 上游化应先抽取无私有 SDK 依赖的 target resolver、backend capability/op contract 和测试接口；PPL 工具链与 RV ABI 保留在本仓库可选后端中。这样能与 CUDA/HIP 等按 target 分流的结构一致，而不在通用 engine 堆积 TPU 特判。

## 7. 相关文件

- `tilelang/engine/tpu_config.py`：`TPUChipSpec`、`TPUTargetSpec`、`TPURuntimeConfig` 与严格解析。
- `tilelang/jit/adapter/ppl_layout.py`、`libgen.py`：PPL 1.7 解析、CModel/PCIe 构建与链接。
- `src/target/codegen_tpu_common.{h,cc}`：中立 TIR 解析、ABI guard、公共元数据。
- `src/target/codegen_tpukernel.cc`、`src/target/codegen_rv.cc`：两条指令选择路径。
- `testing/python/jit/tpu_core_ops_matrix.py`、`tpu_profile_worker.py`：新进程数值/trace 矩阵。
- `research/ppl-profiling/README.md`：profiling 协议与使用说明。
