# TileLang-TPU 的 PPL 指令 Profiling 设计与实现

本文说明为什么需要独立 profiling 模块、PPL 1.7 的真实工作机制、TileLang-TPU 的
CModel/PCIe 实现、使用方式、安全边界，以及 2026-09-04 至 2026-09-08 的分层证据。

这里严格区分三类时间：host wall time、原始命令记录、解码后的设备指令时间。只有最后一类
可以回答“某条 TIU/GDMA 指令用了多久”。

## 1. 当前结论

| 路径 | `6b6772be` canonical 结果 | Profiling 结论 | 边界 |
| --- | --- | --- | --- |
| BM1690 + TPU-Kernel + CModel | core 28/28、FP8 42/42、demo 36/36、完整 TPU-Kernel 152/152 | 前三组逐例保留非空 raw；本机没有兼容的 CModel PerfAI，故不生成真实逐指令时间 | 本机没有 BM1690 板卡，PCIe 未验证 |
| SG2260E + TPU-Kernel/RV + CModel | core 56/56、FP8 42/42、demo 51/51、完整 TPU-Kernel 146/146 | 前三组逐例保留非空 raw；CModel `timed_instruction_count=0` 是显式能力边界 | RV 只覆盖公共 core 与 demo 的 elementwise/matmul，不外推到复合算子 |
| SG2260E + TPU-Kernel/RV + PCIe core | 56/56 | 数值、recorder raw 和 decoded timing 三重门禁全部通过；824 条 ns 事件 | 固定测试 shape/selector；每例只发射一次 |
| SG2260E + TPU-Kernel FP8 + PCIe | 42/42 | 同一三重门禁通过；150 条 ns 事件 | 不外推到 RV FP8 或未列 selector |
| SG2260E + demo + PCIe | 51/51（TPU-Kernel 36、RV 15） | 同一三重门禁通过；2594 条 ns 事件 | RV 15 项仅为三种 dtype 的四则与 matmul |
| SG2260E + 完整 TPU-Kernel + PCIe | 146/146 | 独立 numeric 矩阵，不启用 profiling | 用于完整 op 正确性，不与 149 个 profiling case 重复计数 |

以上结果来自同一干净实现基线 `6b6772be62803338ce52cc0e57b82c47752c738d`。CModel 的
八份矩阵合计 553/553 次 launch；其中 core/FP8/demo 共 255 个 case 保留 raw，完整
TPU-Kernel 的 298 个 case 只做数值验证。本机 PPL 1.7 与 `/opt` 均没有兼容的 CModel
`AutoRunner.sh`，因此 CModel summary 明确记录 `parser_status=unavailable`，不会用 host
时间伪造设备 duration。

SG2260E PCIe 的 core 56、FP8 42 和 demo 51 共 149 个 profiling case 全部通过
`numeric-raw-and-decoded-timing`，合计 149 组 recorder 和 3568 条合法 ns 事件。三份
summary 都记录 `bigTpuProfile 0.3.5` 以及
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse`。这些事件来自每个 case 的单次
recorder 发射，适合核对指令映射和定位异常；它们不是重复采样后的端到端 benchmark，不能
作为稳定吞吐或延迟结论。

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
| `tilelang/jit/adapter/tpu_profiling.py` | 配置校验、隔离 worker、decoder-only Python/PYTHONPATH、无硬件 preflight、deadline、raw 工件发现及结果归一化 |
| `tilelang/jit/_tpu_profile_supervisor.py` | Linux parent-death 与整进程组清理，防止 vendor worker 成为孤儿 |
| `tilelang/jit/_tpu_pcie_profile_decoder.py` | 离线调用已安装的 `bigTpuProfile`；绝不自动安装包；输出稳定 JSON |
| `src/tl_templates/tpu/main_template.cpp` | PCIe profile 会话内建立 TPUDNN handle，执行 enable/sync/disable；普通运行不启用 |
| `tilelang/jit/adapter/ppl_layout.py` | 解析 PPL 1.7 SDK，并明确分开 CModel runtime 与安装在 `/opt` 的 PCIe board runtime |
| `tilelang/jit/adapter/libgen.py` | PCIe host 链接 board `libtpuv7_rt.so` 与 PPL chip backend 的 `libtpudnn.so` |
| `testing/python/jit/tpu_profile_worker.py` | 在子进程中 fresh-compile、加载、数值验证和 dispatch，不复用外部 `main.so` |
| `testing/python/jit/tpukernel_ops_matrix.py` | 非 profiling 数值矩阵同样经 parent-death supervisor 启动每个 worker，保留逐 case deadline、首错停止与部分 summary |
| `testing/python/jit/tpu_fp8_ops_matrix.py` | CModel 默认；PCIe 要求双确认、唯一显式 chip/device，严格 timing 时先离线 preflight；任一首错停止 |

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

框架不会自动安装或升级包。`TPUProfilingConfig.pcie_decoder_python` 可指定专用解释器，
`pcie_decoder_pythonpath` 可指定一个或多个 decoder-only import root（目录或 Python 可导入的
zip/wheel archive）；这些路径替换而不是追加
decoder 子进程的 `PYTHONPATH`，不会进入 fresh-compile/数值 worker。decoder 子进程还会删除
load/profile 确认、device id、recorder 和 TileLang profile-session 环境，因而离线解析没有再次
触碰板卡的授权条件。未显式配置时仍沿用当前解释器与既有环境，以保持原调用兼容。

`pcie_decoder_python` 保留调用方给出的虚拟环境 launcher 路径，不解析成它所指向的系统
Python；否则会丢失该虚拟环境的依赖发现。`preflight_pcie_decoder()` 不创建 runtime、不加载板卡，只在有界 supervisor 下导入
`bigTpuProfile`，验证
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse` 可调用，并返回 `package`、
`package_version`、`parser_api`。核心与 FP8 矩阵在 `--require-decoded-timing` 时先执行该检查；
preflight 失败会在第一个硬件 worker 之前写入失败 summary 并停止。正式 decoder JSON 和
`TPUProfileReport.decoder_identity` 也保留相同身份，避免“能 import”与“实际解析者”漂移。

PCIe 路径只接受内置 decoder 生成的 canonical JSON，不再扫描
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
    pcie_decoder_python="/path/to/decoder-python",       # 可选，仅 decoder
    pcie_decoder_pythonpath=("/path/to/decoder-packages",),  # 可选，仅 decoder
)
profiler = TPUInstructionProfiler(config)
decoder_identity = profiler.preflight_pcie_decoder(  # 无硬件；严格 timing 前调用
    environment={"PPL_PROJECT_ROOT": "/path/to/ppl-1.7"},
)
report = profiler.run_pcie(
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

核心与 FP8 矩阵还接受 `--pcie-decoder-python <python>` 和可重复的
`--pcie-decoder-pythonpath <dir>`；两项均只传给 decoder。FP8 PCIe 另外要求恰好一个显式
`--chip`，不能使用 CModel 默认的双芯片展开。其完整门禁是 load/profile 两个确认、合法
`--device-id`、唯一 chip、fresh worker 单次 launch 和首错停止。

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
| `raw_trace_files` | CModel raw 文件；PCIe 只收录内含规范命名且非空 `.profile` 文件的 `cdm_profile_data_dev*` recorder 目录 |
| `raw_instructions` | CModel 文本 sidecar 中可识别的命令；没有时间值 |
| `decoded_report_paths` | PCIe `bigTpuProfile` 的稳定 JSON 投影 |
| `decoder_identity` | 正式 PCIe decoder 实际使用的包名、版本与 parser API |
| `perfai_report_paths` | CModel PerfAI 的 `profile_data.js`；PCIe 路径恒为空 |
| `timeline_events` | 可用时间线的完整事件；PCIe 稳定 JSON 当前只含设备命令 |
| `instruction_timings` | 设备命令的 engine/core/id/opcode/begin/end/duration/unit |

PCIe summary 中的 `raw_trace_file_count=1` 表示“一组 recorder 目录”，不是只有一个物理
trace 文件。本轮 149 个目录的结构完全一致：每个目录恰含一个 `global.profile` 和四个
SG2260E 核对应的 `cdmlib0_0.profile` 至 `cdmlib0_3.profile`，五个文件均非空。用目录数作为
case 级计数，可避免把芯片核数误计为独立测试数。

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

内部 `timeout_s` 是 worker、CModel PerfAI 锁等待、AutoRunner 或 PCIe decoder 共用的绝对 wall-clock deadline，不为每阶段重新计时。profiler 在启动 supervisor 时保存 PGID，后续判断进程组是否清空，不依赖 leader 的 `poll()` 状态。worker 即使以 0 退出，只要同 PGID 下还有普通后代，profiler 就会回收整个进程组并判该 case 失败。

超时、KeyboardInterrupt 或父进程死亡时，supervisor 终止自己的整个进程组。TERM→KILL、最终 reap 和日志 pipe drain 都有硬上限，即使驱动调用处于不可中断状态，也不会落入无期限 `communicate()`。CModel worker、PCIe worker、offline decoder 和 PerfAI parser 共用这套语义。profiling worker 与 TPU-Kernel 数值矩阵 worker 都经过该 supervisor；即使外层 runner 被 SIGKILL，Linux `PR_SET_PDEATHSIG` 仍会触发 supervisor 清理 worker 及其普通后代。opt-in 硬件矩阵在第一项失败后停止，不继续下一项。

板卡“健康”和“已静默”是两项不同条件。每次 PCIe preflight/postflight 都在同一个 device
session lock 内调用受监管的 `tpu-smi`，先立即拒绝拓扑错误、非 `Active` 状态、非法 JSON、
probe 超时或残留后代；只有格式合法的 `Active + 非零利用率` 可以在一个 10 秒 monotonic
总 deadline 内继续等待。采样间隔为 0.25 秒，必须取得两次间隔开的连续 `0%` 才能释放锁；
`0% → 非零 → 0%` 会清空连续计数，不能凭一次瞬时零值放行。summary 保存每次采样的时间、
probe 耗时、利用率、最大利用率和连续零计数。

如果 dispatch 后未能证明静默，当前 session marker 不会在异常退出时被删除，同时写入持久
quarantine marker；后续 launch 在人工核查、恢复板卡并清除 marker 之前 fail-closed。
preflight 尚未发射 kernel，因此失败只停止本轮，不自动污染板卡状态。该边界使进程组清理与
硬件静默验证相互补充：前者证明受控进程已退出，后者证明设备队列也已回到稳定空闲。

CModel PerfAI 的 `auto_build` 是共享可变目录。TileLang 用按 PerfAI root 命名的 Linux 抽象
Unix socket 串行化自身会话；socket 随进程退出自动释放，不产生旧版 `/tmp/*.lock` 残留。

矩阵在 `research/artifacts` 下创建唯一、自有的 scratch，并通过 `try/finally` 只删除该目录；
raw/decoded 报告不受影响。验证构建、pytest 基目录和临时安装的 decoder 依赖不纳入 Git，
完成报告后清理；仓库只保留实现、测试和本 README。

## 7. 历史验证与当前证据的边界

7.1–7.4 保留早期 profiling 竖切与上一轮 canonical 计数，用于说明实现如何演进。它们不替代
[当前算子验收报告](../tpu-demo-ops/README.md)，也不替代 7.5 的 `6b6772be` 最终证据。

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

### 7.4 2026-09-07 `e5e3087` canonical 重验证

- clean 实现基线 `e5e308797640fa2c3e789cdddd9ebed9246ddd47` 的 [TPU-Kernel 数值矩阵](../artifacts/2026-09-07/tpukernel-cmodel-e5e3087/summary.json) 为 288/288（SG2260E 141/141、BM1690 147/147），[FP8 profiling 矩阵](../artifacts/2026-09-07/fp8-cmodel-e5e3087/summary.json) 为 76/76；后者每例同时保留非空 CModel raw trace。
- 同一基线的 [三组核心 profiling 矩阵](../artifacts/2026-09-07/core-cmodel-e5e3087/summary.json) 为 27/27：BM1690/TPU-Kernel、SG2260E/TPU-Kernel、SG2260E/RV 各 9/9。每例 raw trace 非空；CModel 后处理未启用，全部 `timed=0`。
- `e5e3087` 实现基线的 source-only 回归为 348 passed、4 skipped；补入 18 个多段 runtime case-id 与 portable path 契约正反例后，远程提交前工作树为 366 passed、4 skipped。覆盖 decoder-only 环境、离线 preflight/API 身份、FP8 PCIe 参数门禁和首错停止；四个 skip 均需显式开启真实 profiling worker。
- PCIe numeric/raw 与 decoded timing 保持解耦；严格矩阵在派发前先验证 decoder。当前 [TPU-Kernel matmul](../artifacts/2026-09-07/pcie-tpukernel-matmul-e5e3087/summary.json) 为 1/1、36 条 timing；[RV 核心矩阵](../artifacts/2026-09-07/pcie-rv-core-e5e3087/summary.json) 为 9/9、108 条 timing；[FP8 矩阵](../artifacts/2026-09-07/pcie-fp8-e5e3087/summary.json) 为 38/38、38 个 raw 目录、134 条 timing。三者都记录 `bigTpuProfile 0.3.5` 及 `bigTpuProfile.bmprofile_perfAI.ProfileParser.parse`。
- TPU-Kernel 非 profiling 数值矩阵为 [core 54/54](../artifacts/2026-09-07/pcie-tpukernel-core-e5e3087/summary.json)、[extended 15/15](../artifacts/2026-09-07/pcie-tpukernel-extended-e5e3087/summary.json)、[reductions 72/72](../artifacts/2026-09-07/pcie-tpukernel-reductions-e5e3087/summary.json)，合计 141/141。这些 numeric summary 不含逐指令 timing 门禁，不能与 profiling case 数相加。
- 运行前后板卡均为 `Active`、利用率 0%，结束后没有受控 profiler、runner、decoder 或 runtime 后代残留。本机没有 BM1690 板卡，BM1690 PCIe 保持 `unverified`。`44a6fc2` 及更早工件只保留为历史过程，不再作为当前基线。

本轮非 FP8 单次诊断中，TPU-Kernel matmul 的 36 条事件 duration 合计 6844 ns、时间线跨度 45060 ns；RV matmul 同为 36 条且 opcode 分布相同，合计 6320 ns、跨度 9896 ns。两者均为 8 `MM2_NN`、4 `copy`、4 `data_convert`、16 `tensorLd`、4 `tensorSt`。RV add/sub/mul/div 各 16 条（4 op、8 load、4 store），duration 总和依次为 3682/3198/3324/3858 ns；FP16 local/S2S copy 分别为 3 条合计 498 ns、跨度 1796 ns，以及 1 条 265 ns；FP32 对应为 3 条合计 496 ns、跨度 2685 ns，以及 1 条 303 ns。

FP8 的两种格式呈现一致的 case→opcode 映射：copy=`tensorLd+tensorSt`，S2S=`tensorLd`，fill=`copy+tensorSt`，cast=`data_convert+tensorLd+tensorSt`，binary=`op+2 tensorLd+tensorSt`，scalar=`op+tensorLd+tensorSt`，RoPE=`2 add+4 tensorLd+tensorSt`，gather=`DMA_gather`，GEMM=`MM2_NN/MM2_NT+2 tensorLd+tensorSt`，accumulate 另有一次 `copy`。两格式合计 134 条：`MM2_NN` 4 条/184 ns、`MM2_NT` 4/304 ns、`add` 10/118 ns、`copy` 6/60 ns、`data_convert` 4/56 ns、`mul` 6/70 ns、`sub` 4/48 ns、`DMA_gather` 2/2674 ns、`tensorLd` 60/15134 ns、`tensorSt` 34/4450 ns；按 engine 聚合为 BDC 38 条/840 ns、GDMA 96 条/22258 ns。所有时间都来自每例一次带 recorder 的诊断发射，不是 benchmark。

### 7.5 2026-09-08 `6b6772be` 完整重验证

- 同一干净 revision `6b6772be62803338ce52cc0e57b82c47752c738d` 的八份 CModel summary
  均位于 [canonical artifact 目录](../artifacts/2026-09-08/final-6b6772be/)，合计 553/553：core 为 BM1690
  28/28、SG2260E 56/56，FP8 为 42/42 + 42/42，demo 为 36/36 + 51/51，完整
  TPU-Kernel 为 152/152 + 146/146。所有 summary 均记录
  `implementation_worktree_dirty=false`。
- SG2260E PCIe 的 [core 56/56](../artifacts/2026-09-08/final-6b6772be/core-sg2260e-pcie/summary.json)、
  [FP8 42/42](../artifacts/2026-09-08/final-6b6772be/fp8-sg2260e-pcie/summary.json) 和
  [demo 51/51](../artifacts/2026-09-08/final-6b6772be/demo-sg2260e-pcie/summary.json) 共
  149/149；每个 case 的 numeric、非空 raw recorder 组和 decoded timing 均通过严格门禁。
  三组分别解码 824、150、2594 条事件，共 3568 条。
- demo 中每个 elementwise case 为 4 条事件；matmul 的 FP16/BF16 为 36 条、FP32 为
  48 条；RMSNorm 的 FP16/BF16 为 22 条、FP32 为 18 条；Split-K RMSNorm 对应为
  96/72 条；RoPE 均为 20 条；SwiGLU 为 48/42 条；FlashAttention 每种 dtype 有三个
  输入变体，FP16/BF16 每例 190 条、FP32 每例 198 条。这些是当前固定 workload 的实际
  summary 计数，不是算子的静态指令数契约。
- 独立的 [TPU-Kernel 完整数值矩阵
  146/146](../artifacts/2026-09-08/final-6b6772be/tpukernel-sg2260e-pcie/summary.json)
  不开启 profiling，因而只证明正确性与受控板端执行；不能与上述 149 个 case 合并成 295
  个 profiling case。
- 在 `137c85d5` 的第一次 SG2260E PCIe 诊断中，TPU-Kernel FP32 add 已通过数值 oracle，
  五个 raw 文件和 16 条 decoded ns 事件也都有效；旧 postflight 却在同步发射后的即时
  `Active/9%` 单次采样处停止。稍后的人工 `0%` 观察不属于该失败 summary，因此该工件只作
  缺陷定位，不能晋升为通过证据。加入有界 settle 后，`6b6772be` 的 149 个 profiling
  postflight 均先观察到 8% 至 10% 的非零利用率，再以两次连续 0% 完成；这直接验证了新门禁既不会误拒绝
  正常回落，也不会用单个瞬时零值掩盖持续占用。

## 8. 尚未完成的工作

1. 获得与本机 SG2260E raw schema 匹配的 CModel PerfAI，验证真实 CModel
   `profile_data.js`，而不只依赖 fixture。
2. 扩展当前 FP8 边界：非零 fill、FP16/BF16↔FP8 cast、FP8 C、异常值与更广 shape；更宽 RV selector 仍须先分别建立 CModel 前置证据。
3. 为 profile artifact 增加正式 manifest（resolved target、SDK、board runtime、decoder identity、kernel、输入
   shape/dtype）；在此之前不要用不同构建之间的 timing 做自动回归判定。
4. 将 profiling duration 与普通 benchmark 分层：profiling 用于定位指令，benchmark 用于低
   扰动统计；不能让 autotune 直接消费一次有 recorder 开销的 host wall time。
5. 为 dependency-correct TPU software pipeline 增加 hazard/buffer-version 模型；在此之前核心
   correctness worker 保持串行。
6. 取得 BM1690 板卡后，独立验证其 PCIe runtime、数值与 profiling；当前环境不能提供这层证据。
