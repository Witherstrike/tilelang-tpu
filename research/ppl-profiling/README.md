# TileLang-TPU 的 PPL 指令 Profiling 设计与实现

本文说明为什么需要独立 profiling 模块、PPL 1.7 的真实工作机制、TileLang-TPU 的
CModel/PCIe 实现、使用方式、安全边界，以及 2026-09-04 至 2026-09-07 的分层证据。

这里严格区分三类时间：host wall time、原始命令记录、解码后的设备指令时间。只有最后一类
可以回答“某条 TIU/GDMA 指令用了多久”。

## 1. 当前结论

| 路径 | 已实现 | 本机证据 | 仍有限制 |
| --- | --- | --- | --- |
| SG2260E + TPU-Kernel + CModel | `FILE_DUMP_CMD`、四核拓扑设置、raw/.txt 解析、可选 PerfAI | `596a736` 非 FP8 141/141、FP8 38/38；核心 profiling 9/9，含四条 copy case（FP16/FP32 × local-roundtrip/S2S）、四则与 GEMM | 本机 SDK 未附 CModel PerfAI，因此没有 CModel 真实逐指令时间 |
| SG2260E + RV + CModel | 同一隔离收集框架 | `596a736` 核心 9/9；FP16/FP32 的 G2L→L2L→L2S 与 S2S、FP32 四则、FP16 GEMM 均有独立数值 oracle 和 raw | CModel raw 只有命令文本，不提供真实 duration |
| SG2260E + TPU-Kernel + PCIe | TPUDNN recorder、单次 dispatch、raw 收集、结构化 `bigTpuProfile` JSON 投影 | 历史非 FP8 数值矩阵 140/140；2026-09-05 matmul 收集一个 raw 目录并离线解码 36 条 ns 事件 | 受监管运行时未安装 decoder；`50d8c77` 一次性前置探针报告 `Fault`，本轮未发射算子 |
| SG2260E + RV + PCIe | 与 TPU-Kernel 共用 TPUDNN host 路径 | 历史固定 shape/dtype 核心五项数值通过；2026-09-04 工件保留了当时的解码行 | 当前结构化 decoder 契约尚未对 RV trace 重跑；`50d8c77` 本轮未发射算子 |

因此，CModel 的 raw 命令收集与 PCIe 的 recorder 路径均已发挥作用。当前结构化逐指令解码的
canonical 证据是历史 SG2260E/TPU-Kernel matmul 的 36 条事件；RV 板端已有历史数值与 raw 收集证据，但解码器契约需单独重跑。单次 recorder 结果只用于诊断，不用于宣称统计性能。

## 2. 为什么不能直接给 JIT 加 `--profiling`

PPL 1.7 自身的 `ppl_compile.py --profiling` 是其已标记 deprecated 的 CLI 别名，实际进入与 `--autotune` 相同的分支。这是对外部 PPL 脚本的事实说明，TileLang 不保留该别名。该分支做的不只是“编译时打开一个宏”，而是完整的 PPL 前端流程：生成 autotune host glue、编译、运行、收集和后处理。

TileLang 已直接生成 PPL `kernel.c`、host wrapper 与 `main.so`，输入不是 PPL `.pl`。照搬该
开关会重复或错配 codegen，也无法自动把 TPUDNN profile session 接到 TileLang 自己的
`tpuRtKernelLaunch` 上。

正确的迁移边界是复用 PPL 的运行时协议：

```text
CModel:
  fresh worker + FILE_DUMP_CMD
    → BD/GDMA/SDMA raw 与文本 sidecar
      → 可选 PerfAI AutoRunner
        → profile_data.js → 逐命令时间

PCIe / TPUv7:
  fresh worker + 三重授权
    → tpudnnHandleFromStream(现有 device/stream/module)
      → tpudnnEnableProfile
        → 恰好一次 TileLang run/launch/sync
      → tpudnnDisableProfile
    → cdm_profile_data_dev*
      → 已安装的 bigTpuProfile 离线解码
        → TileLang 稳定 JSON → 逐命令时间
```

这也是独立 `TPUInstructionProfiler` 的原因。现有通用 `tilelang.profiler.Profiler` 依赖
CUDA Event 和 `torch.cuda.synchronize()`；把它用于 TPU 会混淆 runtime、同步和计时含义。

## 3. 模块职责与实现原理

| 组件 | 职责 |
| --- | --- |
| `tilelang/jit/adapter/tpu_profiling.py` | 配置校验、隔离 worker、deadline、raw 工件发现、CModel/PCIe 解码结果归一化 |
| `tilelang/jit/_tpu_profile_supervisor.py` | Linux parent-death 与整进程组清理，防止 vendor worker 成为孤儿 |
| `tilelang/jit/_tpu_pcie_profile_decoder.py` | 离线调用已安装的 `bigTpuProfile`；绝不自动安装包；输出稳定 JSON |
| `src/tl_templates/tpu/main_template.cpp` | PCIe profile 会话内建立 TPUDNN handle，执行 enable/sync/disable；普通运行不启用 |
| `tilelang/jit/adapter/ppl_layout.py` | 解析 PPL 1.7 SDK，并明确分开 CModel runtime 与安装在 `/opt` 的 PCIe board runtime |
| `tilelang/jit/adapter/libgen.py` | PCIe host 链接 board `libtpuv7_rt.so` 与 PPL chip backend 的 `libtpudnn.so` |
| `testing/python/jit/tpu_profile_worker.py` | 在子进程中 fresh-compile、加载、数值验证和 dispatch，不复用外部 `main.so` |
| `testing/python/jit/tpukernel_ops_matrix.py` | 非 profiling 数值矩阵同样经 parent-death supervisor 启动每个 worker，保留逐 case deadline、首错停止与部分 summary |

### 3.1 编译身份与运行身份分离

常规 JIT 的编译身份只来自完整 TVM Target，并解析为
`TPUTargetSpec(chip, programming_model)`；运行身份解析为独立的
`TPURuntimeConfig(runtime_mode)`。前者决定 PPL arch、指令 ABI 和 codegen，后者只决定使用
CModel 还是 PCIe host runtime。两者不会通过目录名或隐式芯片默认值相互推导。

`TPUProfilingConfig` 还要保存输出目录、超时和后处理策略，因此保留会话级的 `chip`、
`programming_model` 和 `runtime_mode` 字段；其 `target_spec` 与 `runtime_config` 属性分别返回
上述两个规范对象。核数统一来自 `TPUChipSpec`：BM1690 为 8 核且只支持 TPU-Kernel，
SG2260E 为 4 核且支持 TPU-Kernel/RV。两者共享当前编译器建模的 TPUv7 LMEM 几何，
profiling 不维护第二份能力表。

### 3.2 CModel 收集

CModel worker 在自己的 cwd 中运行，使用安全的相对 `FILE_DUMP_CMD=<label>`。profiler 从
`TPUChipSpec` 设置 `TPU_RT_CORE_NUM`（SG2260E 为 4，BM1690 为 8），同时设置
`TILELANG_TPU_BENCHMARK_RUNS=0`，并删除父环境中可能遗留的全部 PCIe 授权。raw `.txt`
只提供 engine/core/command-id/opcode；它没有可靠 begin/end，模块不会伪造 duration。

如果显式配置 `perfai_root` 或 `PPL_PERFAI_ROOT`，则运行
`AutoRunner.sh -d <session-dir> -e <vendor-chip>`。解析器保留完整 `timeline_events`，并只把
`bd/bdc/tiu/gdma/sdma/vsdma/cdma/dma` 通道放入 `instruction_timings`，避免把 CPU、layer
或 subnet 行冒充硬件指令。

### 3.3 PCIe 收集

PCIe 必须同时满足：

1. `TILELANG_TPU_ALLOW_PCIE_LOAD=1`；
2. `TILELANG_TPU_ALLOW_PCIE_PROFILE=1`；
3. `TILELANG_TPU_DEVICE_ID=<非负整数>`。

`run_pcie()` 再注入 `BMLIB_ENABLE_ALL_PROFILE=1`、`PROFILE_RECORD_SIZE`、
`PROFILE_BOOK_KEEPING` 和内部的 `TILELANG_TPU_PROFILE_SESSION=1`。只设置前三个 vendor
环境变量并不能 profiling；真正的 recorder 生命周期在生成的 host 中：

```text
H2D 完成
  → tpudnnHandleFromStream(device_id, stream, tpu_module)
  → tpudnnEnableProfile(record_size, book_keeping)
  → kernel call（内部 launch + sync）
  → tpudnnSync
  → tpudnnDisableProfile
  → D2H / cleanup
```

profile artifact 在同一 `main.so` 中只能消费一次。第一次 recorder 建立失败也不允许在同一
runtime 实例中重试；要重新 fresh-compile 并启动新 worker。这把“恰好一次”从测试约定提升为
host ABI 的 fail-closed 约束。

### 3.4 为什么 PCIe runtime 必须与 SDK CModel runtime 分开

PPL 1.7 `deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so` 依赖
`libcdm_daemon_emulator.so`，是 SDK 的模拟器链；PCIe 执行使用
`/opt/tpuv7/tpuv7-current/lib/libtpuv7_rt.so`。头文件和 `libtpudnn.so` 仍来自统一的 PPL
1.7 `deps/` 布局。

首次板端验证暴露了旧 TileLang PCIe link 仍固定 SDK runtime、并强制链接
`libcdm_daemon_emulator`：日志进入 AP/CModel 内存初始化并以 255 退出。修正后：

- PCIe 编译延迟检查安装的 board runtime；可用
  `TILELANG_TPU_PCIE_RUNTIME_PATH` 指向非标准安装的 `lib` 目录；
- 明确拒绝把该变量指回 SDK CModel runtime；
- PCIe `main.so` 的 RPATH 是 board runtime + PPL chip backend；
- 不再链接 `libcdm_daemon_emulator.so`；
- 编译时解析出的 runtime identity 随私有 artifact 固化，加载时不因环境变化而漂移。

这一修复也是 PCIe profiling 能真实工作的必要条件，不只是链接整理。

### 3.5 PCIe 离线解码契约

PPL 1.7 的 `autotune.py` 会调用 `to_txt()`，并在缺包时自动执行 pip；这两种行为都不进入
TileLang 的生产路径。TileLang 自带的 PCIe decoder 只接受 `parse()` 返回的结构化
`ProfileResult`，读取 `bd_events/gdma_events/sdma_events/cdma_events` 后写
`tilelang_pcie_profile.json`。本轮以隔离安装的 `bigTpuProfile 0.3.5` 验证了这条 canonical
路径。

框架不会自动安装或升级包。PCIe 路径只接受内置 decoder 生成的 canonical JSON，不再扫描
PerfAI Web UI 的 `profile_data.js`；decoder 成功退出却没有生成 JSON 时会明确返回
`missing-report`，避免把未受控的 UI 格式误当成稳定接口。稳定 JSON 保存 engine、begin/end、
`ns` 单位、core、command id、opcode，以及 vendor 的原始 info/detail/metadata；内置 decoder
不满足结构化契约时会显式报错，raw trace 保持不变。

## 4. 使用方法

### 4.1 CModel

```python
from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig

config = TPUProfilingConfig(
    chip="sg2260e",
    programming_model="rv",       # 或 tpukernel
    runtime_mode="cmodel",
    output_dir="./profiles",
    label="rv-control",
    timeout_s=60,
    perfai_root="/path/to/PerfAI",  # 可选
)
report = TPUInstructionProfiler(config).run_cmodel(
    ["/path/to/python", "/path/to/fresh_worker.py"],
    environment={"PPL_PROJECT_ROOT": "/path/to/ppl-1.7"},
)
```

### 4.2 PCIe

```python
config = TPUProfilingConfig(
    chip="sg2260e",
    programming_model="tpukernel",  # 或 rv
    runtime_mode="pcie",
    output_dir="./profiles",
    label="matmul-on-board",
    timeout_s=60,
)
report = TPUInstructionProfiler(config).run_pcie(
    ["/path/to/python", "/path/to/fresh_worker.py"],
    environment={
        "PPL_PROJECT_ROOT": "/path/to/ppl-1.7",
        "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
        "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
        "TILELANG_TPU_DEVICE_ID": "0",
    },
)
```

`command` 必须是可信 worker，并在该子进程中编译/加载自己的私有 TileLang artifact。profiler
无法阻止恶意命令主动 `setsid()` 或 daemonize；这类逃离受控进程组的行为不受支持。

核心矩阵的 PCIe CLI 把验收层级显式分开：原有
`--runtime-mode pcie --allow-pcie --allow-pcie-profile --device-id <n>` 要求 worker 数值通过，且至少生成一个 `cdm_profile_data_dev*` 目录；目录中必须存在命名为 `global.profile` 或 `cdmlib<core>_<group>.profile` 的 recorder 文件，并且所有此类文件都非空。decoder 只作最佳努力。若本次目标确实是逐指令时间，再增加 `--require-decoded-timing`；此时 `parser_status=ready`、至少一条 timing，以及每条记录的 begin/end/duration 均为非 bool 的有限数值、`duration>=0`、`end>=begin`、`unit="ns"`，共同构成硬门禁。这样 decoder 缺失不会污染数值+raw 结论，严格 timing 任务也不会把 raw-only 或畸形 interval 误报为完成。

worker 内的编译调用仍使用规范公共接口，例如：

```python
kernel = tilelang.compile(
    program,
    target=("tpu -mcpu=sg2260e "
            "-tpu-programming-model=rv"),
    runtime_mode="cmodel",
)
```

裸 TPU target 会在 lowering 前失败；PCIe worker 只把上例的 `runtime_mode` 改为 `pcie`，
不会改写编译身份。

未指定 `output_dir` 时，结果保存在当前工作目录的 `tilelang-tpu-profiles/<label>-*`，不再默认
写入 `/tmp`。每次会话创建新子目录，不删除历史报告。

## 5. 报告字段与状态

| 字段 | 含义 |
| --- | --- |
| `raw_trace_files` | CModel raw 文件；PCIe 只收录内含规范命名且非空 `.profile` 文件的 `cdm_profile_data_dev*` 目录 |
| `raw_instructions` | CModel 文本 sidecar 中可识别的命令；没有时间值 |
| `decoded_report_paths` | PCIe `bigTpuProfile` 的稳定 JSON 投影 |
| `perfai_report_paths` | CModel PerfAI 的 `profile_data.js`；PCIe 路径恒为空 |
| `timeline_events` | 可用时间线的完整事件；PCIe 稳定 JSON 当前只含设备命令 |
| `instruction_timings` | 设备命令的 engine/core/id/opcode/begin/end/duration/unit |

常见 `parser_status`：

- `ready`：真实 decoder 报告存在且至少有一条设备命令；
- `not-requested`：`postprocess=False`；
- `unavailable`：外部 decoder 不存在；raw 仍保留；
- `no-raw-trace`：worker 成功但 recorder 没有产物；
- `no-device-command-events`：decoder 成功，但 case 只有被过滤的控制/系统命令；
- `busy` / `timed-out` / `deadline-exhausted`：并发锁或总 deadline 阻止后处理；
- `failed` / `missing-report` / `invalid-report`：外部工具或 schema 错误。

## 6. 进程安全、并发和清理

```text
外层 timeout
  └─ pytest / profiler P
       └─ supervisor S（PR_SET_PDEATHSIG=SIGTERM，private process group）
            └─ fresh JIT worker 或离线 decoder
                 └─ compiler / runtime / parser 子孙
```

内部 `timeout_s` 是 worker、CModel PerfAI 锁等待、AutoRunner 或 PCIe decoder 共用的绝对
wall-clock deadline，不为每阶段重新计时。超时、KeyboardInterrupt 或父进程死亡时，
supervisor 终止自己的整个进程组；SIGTERM、SIGKILL、最终 reap 和日志 pipe drain 各有硬
上限，即使驱动调用处于不可中断状态也不会再落入无期限 `communicate()`。profiling worker
与 TPU-Kernel 数值矩阵 worker 都走该 supervisor；即使外层 runner 被 SIGKILL，Linux
`PR_SET_PDEATHSIG` 仍会触发 supervisor 清理 worker 及其普通后代。opt-in 硬件矩阵在第一项失败后停止，不继续下一项。

CModel PerfAI 的 `auto_build` 是共享可变目录。TileLang 用按 PerfAI root 命名的 Linux 抽象
Unix socket 串行化自身会话；socket 随进程退出自动释放，不产生旧版 `/tmp/*.lock` 残留。

矩阵在 `research/artifacts` 下创建唯一、自有的 scratch，并通过 `try/finally` 只删除该目录；
raw/decoded 报告不受影响。验证构建、pytest 基目录和临时安装的 decoder 依赖不纳入 Git，
完成报告后清理；仓库只保留实现、测试和本 README。

## 7. 历史验证与当前证据的边界

7.1–7.3 保留 2026-09-04 核心 profiling 竖切的原始计数，用于说明实现是如何被验证的。它们不替代 [当前完整数值报告](../tpu-backend-design/test-report.md)，也不替代 2026-09-05 对结构化 `ProfileResult` 解码契约的离线验证。

### 7.1 2026-09-04 CModel

- SG2260E TPU-Kernel 与 RV：add/sub/mul/div、64×64 FP16 matmul 共 10 项全部
  `allclose` 通过；每个逐元素解析 58 条 raw 命令，matmul 78 条。
- BM1690 TPU-Kernel：同样五项全部通过；逐元素 94、matmul 122 条 raw 命令。
- 三种组合合计 15/15；24/48 个 trace 文件反映 4/8 核模拟器拓扑，不代表工作负载多核执行。

### 7.2 2026-09-04 PCIe device 0

- TPU-Kernel 与 RV 的 add/sub/mul/div、matmul 均完成加载、单次发射、回传、数值比对、
  recorder 和当时的 JSON 投影，合计 10/10。
- 每个逐元素 case 为 16 条：`tensorLd` 8、对应算术 4、`tensorSt` 4。单条
  TPU-Kernel/RV add/sub/mul 分别为 14/12 ns，div 为 74/72 ns。
- 每个 matmul 为 36 条：`tensorLd` 16、`MM2_NN` 8、copy 4、`data_convert` 4、
  `tensorSt` 4；两后端 `MM2_NN` 和 convert 单条均为 51 ns 与 12 ns。
- 初次 TPU-Kernel matmul 因 `T.Pipelined(num_stages=1)` 将 load 与直接消费者 GEMM 放入
  并行区而数值失败；矩阵按首错停止。改为依赖有序的 `T.serial` 后，CModel 三组合和
  PCIe 均通过。这证明 CModel 无法替代真实并行时序验证。
- 这些历史工件中的 timing 可用于回顾当时的指令选择，但当时 decoder 路径不是当前内置 decoder 的结构化 `ProfileResult` → canonical JSON 契约；因此机器能力契约只将这些 summary 用作 RV/TPU-Kernel 核心数值证据，不用作当前 decoder 的 conformance 证据。

host 打印的单次时间约 8 ms，包含 runtime 调用和 profiling 开销，不能与上述设备命令 ns
直接求和或当成无扰动性能。性能结论需要关闭 profiling 后的重复测量与统计设计。

### 7.3 2026-09-04 工程回归

- TPU 相关静态/单元测试：`101 passed, 4 skipped`；覆盖两种编程模型、三个合法 chip/model target 组合、工具链、codegen、
  AddressAssign、运行时门禁、超时进程组与 bounded kill/drain。
- CModel 数值矩阵：15/15；PCIe 数值+timing 矩阵：10/10。
- 初次 PCIe matmul 数值失败实际触发 fail-stop，RV 未继续运行；修复、重过 CModel 和单项
  canary 后才启动 RV。全部完成后无 profiler、runner、vendor worker 或 JIT scratch 残留。

### 7.4 2026-09-07 region ABI 重验证

- `50d8c77` 上 SG2260E/TPU-Kernel matmul profiling 冒烟通过，保留 24 个 raw trace 文件和 78 条 raw 命令；`timed=0` 如实反映 CModel trace 没有 duration。
- 编译器基线不变，在验收拆分后的 `55c1c6d` 上 [SG2260E/RV add/sub/mul/div/matmul](../artifacts/2026-09-07/rv-core-cmodel-acceptance-55c1c6d/summary.json) 再次为 5/5；逐元素各 58 条、matmul 78 条 raw 命令，证明 profiling 收集没有因严格 `tl.region` ABI 或分层验收失效。
- 最终实现基线 `596a736` 的 [三组核心矩阵](../artifacts/2026-09-07/core-cmodel-596a736/summary.json) 为 27/27：BM1690/TPU-Kernel、SG2260E/TPU-Kernel、SG2260E/RV 各 9/9。每组新增 FP16/FP32 local-roundtrip 与 S2S；SG2260E/RV local case 分别收集 45 条 raw 命令，S2S 分别为 43 条，逐元素各 58 条，GEMM 78 条，全部 `timed=0`。
- 当前 source-only 总回归为 318 passed、4 skipped；新增门禁覆盖 raw 目录语义、严格 timing 的数值/有限性/单位/区间、外层 runner 死亡后的进程树清理，以及 runtime evidence 的 target/capability 绑定与闭集。
- 当前环境没有兼容 `bigTpuProfile`。RV PCIe runner 已把 numeric/raw 与 decoded-timing 解耦：默认可以独立验收数值与 recorder raw；显式 `--require-decoded-timing` 才会在 decoder 缺失时失败。两种模式均把 `acceptance`、`decoded_timing_required` 和每 case 的实际解析状态写入 summary。
- 板端前置检查识别到 PCIe 设备、驱动与 device node。首次无参数 `tpu-smi` 因默认 `--loop` 持续运行，完整进程组已被 TERM 且无残留；改用带 10 秒硬上限的一次性 `--noloop --json_format` 后正常退出，但报告 `status=Fault`、`tpu_util=100%`。依照首错停止策略，随后跳过所有 `50d8c77` PCIe case。

## 8. 尚未完成的工作

1. 获得与本机 SG2260E raw schema 匹配的 CModel PerfAI，验证真实 CModel
   `profile_data.js`，而不只依赖 fixture。
2. 设备恢复健康后，先在当前提交依次运行 SG2260E/TPU-Kernel matmul canary、SG 两编程模型各 9 项核心矩阵，再分批重验 TPU-Kernel 非 FP8；全部通过后才进入 FP8 PCIe 的 copy/cast、arithmetic、GEMM、gather/RoPE。更宽 RV selector 仍须先分别建立 CModel 前置证据。
3. 为 profile artifact 增加正式 manifest（resolved target、SDK、board runtime、kernel、输入
   shape/dtype）；在此之前不要用不同构建之间的 timing 做自动回归判定。
4. 将 profiling duration 与普通 benchmark 分层：profiling 用于定位指令，benchmark 用于低
   扰动统计；不能让 autotune 直接消费一次有 recorder 开销的 host wall time。
5. 为 dependency-correct TPU software pipeline 增加 hazard/buffer-version 模型；在此之前核心
   correctness worker 保持串行。
