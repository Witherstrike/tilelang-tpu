# TileLang-TPU 指令 Profiling 设计与验证

本文说明 profiling 模块的用途、与 PPL 1.7 的关系、CModel/PCIe 实现、安全约束和当前实测结果。
文中的 host wall time、原始命令记录和解码后的设备指令时间是三种不同数据；只有最后一种可用于
分析单条 TIU/GDMA 指令。

## 1. 结论

当前证据绑定：

- revision：`e5774525e3a6e11d0d6010e979203c55181a8872`；
- source state：`c49cf3594f2ce5d215e54d331a10ca0bbbf92bd6667fba3e172b2b3621f7f5c4`；
- evidence root：[`research/artifacts/2026-09-09/final-e5774525`](../artifacts/2026-09-09/final-e5774525/)；
- 所有正式 summary 均为干净实现、`complete=true`、失败和取消为 0。

| 路径 | 数值结果 | Profiling 结果 | 适用边界 |
| --- | ---: | --- | --- |
| BM1690 CModel | 258/258 | core、FP8、demo 的 106 个 case 均有 raw；无兼容 decoder，不报告 duration | 未验证 BM1690 PCIe |
| SG2260E CModel | 295/295 | core、FP8、demo 的 149 个 case 均有 raw；无兼容 decoder，不报告 duration | RV 仅覆盖已登记 selector |
| SG2260E PCIe core | 56/56 | 56 组 raw，824 条合法 ns 事件 | TPU-Kernel/RV 各 28 项 |
| SG2260E PCIe FP8 | 42/42 | 42 组 raw，150 条合法 ns 事件 | 仅 TPU-Kernel |
| SG2260E PCIe demo | 51/51 | 51 组 raw，2594 条合法 ns 事件 | TPU-Kernel 36，RV 15 |
| SG2260E PCIe 完整 TPU-Kernel | 146/146 | 纯数值矩阵，不开启 recorder | 14 个正式分片的精确并集 |

CModel 共完成 553/553 次 launch；其中 255 个 profiling case 保存 8664 个 raw 文件、21194 条
命令记录，`timed_instruction_count=0`。本机 PPL 1.7 和 `/opt` 环境没有与这些 CModel raw
兼容的 PerfAI AutoRunner，因此该值表示“没有时长证据”，不是“耗时为零”。

SG2260E PCIe 的 149 个 profiling case 产生 149 组 recorder，由 `bigTpuProfile 0.3.5` 的
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse` 解码出 3568 条合法 ns 事件：BDC 2428，
GDMA 1140。每个 case 只进行一次带 recorder 的诊断发射；这些数据适合检查指令映射和定位异常，
不构成稳态延迟、吞吐或后端性能排名。

## 2. 为什么需要独立模块

PPL 1.7 的 `ppl_compile.py --profiling` 是已弃用的 CLI 别名，实际进入 `--autotune` 流程。该流程
面向 PPL `.pl` 输入，会生成专用 host glue，并负责编译、执行、采集和后处理。TileLang 已自行生成
`kernel.c`、host wrapper 和 `main.so`，若直接转发该参数，会重复 codegen，也无法把 profile 会话
正确接入 TileLang 的 `tpuRtKernelLaunch`。

TileLang 因此只复用 PPL/TPUDNN 的记录协议，并由 `TPUInstructionProfiler` 管理自己的编译与运行：

```text
CModel worker
  -> FILE_DUMP_CMD
  -> BD/GDMA/SDMA raw 与文本 sidecar
  -> 可选 PerfAI AutoRunner
  -> profile_data.js

PCIe worker
  -> tpudnnHandleFromStream
  -> tpudnnEnableProfile
  -> 单次 launch + sync
  -> tpudnnDisableProfile
  -> cdm_profile_data_dev*
  -> bigTpuProfile 离线解码
  -> TileLang 稳定 JSON
```

通用 `tilelang.profiler.Profiler` 依赖 CUDA Event 和 `torch.cuda.synchronize()`，不适用于 TPU 的
runtime、同步与设备时间语义。

## 3. 组件与实现原理

| 组件 | 职责 |
| --- | --- |
| `tilelang/jit/adapter/tpu_profiling.py` | 配置校验、隔离 worker、总 deadline、raw 发现、报告归一化和无硬件 decoder preflight |
| `tilelang/jit/_tpu_profile_supervisor.py` | parent-death signal、私有进程组和有界 TERM→KILL 清理 |
| `tilelang/jit/_tpu_pcie_profile_decoder.py` | 调用已安装的 `bigTpuProfile`，输出稳定 JSON；不自动安装依赖 |
| `src/tl_templates/tpu/main_template.cpp` | 在 PCIe host 中建立、同步并关闭 TPUDNN profile 会话 |
| `tilelang/jit/adapter/ppl_layout.py` | 只解析 PPL 1.7 `deps/` 布局，并区分 CModel 与板端 runtime |
| `tilelang/jit/adapter/libgen.py` | 为 CModel/PCIe 选择正确 runtime、RPATH 和 TPUDNN 依赖 |
| `testing/python/jit/tpu_*_ops_matrix.py` | 执行分层矩阵、晋级校验、首错停止和证据持久化 |

### 3.1 编译目标与运行模式

编译身份由完整 TVM Target 解析为 `TPUTargetSpec(chip, programming_model)`；运行身份由
`TPURuntimeConfig(runtime_mode)` 表达。前者决定 PPL arch、指令 ABI、pass 和 codegen，后者只选择
CModel 或 PCIe host runtime。目录名、环境变量和运行模式均不能反向猜测芯片或编程模型。

`TPUChipSpec` 是核数和有效组合的唯一来源：BM1690 为 8 核且只接受 TPU-Kernel；SG2260E 为
4 核并接受 TPU-Kernel/RV。工具链只支持 PPL 1.7 `deps/`，没有旧版 PPL 目录回退。

### 3.2 CModel

CModel worker 在独立工作目录中运行，设置安全的相对 `FILE_DUMP_CMD`、正确的
`TPU_RT_CORE_NUM` 和 `TILELANG_TPU_BENCHMARK_RUNS=0`，同时删除继承的 PCIe 授权变量。raw
文本包含 engine、core、command id 和 opcode，但没有可靠的 begin/end，框架不会用 host 时间
补写 duration。

只有显式配置 `perfai_root` 或 `PPL_PERFAI_ROOT` 时，才运行 `AutoRunner.sh -d <session> -e
<chip>`。后处理保留完整时间线，只把 `bd/bdc/tiu/gdma/sdma/vsdma/cdma/dma` 通道认定为设备
指令。

### 3.3 PCIe

PCIe profiling 必须同时提供：

1. `TILELANG_TPU_ALLOW_PCIE_LOAD=1`；
2. `TILELANG_TPU_ALLOW_PCIE_PROFILE=1`；
3. 唯一且非负的 `TILELANG_TPU_DEVICE_ID`。

profiler 额外设置 recorder 环境，并在生成的 host 中执行以下生命周期：H2D 完成后建立 TPUDNN
handle，开启 recorder，恰好执行一次 kernel 并同步，随后关闭 recorder，再进行 D2H 和清理。
profile artifact 在同一 `main.so` 中只允许消费一次；建立失败后必须 fresh-compile 并启动新 worker，
不能复用可能残留状态的 runtime。

PPL SDK 内的 `deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so` 属于 CModel 链，依赖
`libcdm_daemon_emulator.so`。PCIe 必须使用 `/opt/tpuv7/tpuv7-current/lib/libtpuv7_rt.so` 或由
`TILELANG_TPU_PCIE_RUNTIME_PATH` 显式指定的板端 runtime，并拒绝该变量指回 SDK CModel 目录。
PPL 公共头文件与逐芯片 `libtpudnn.so` 仍来自同一个 PPL 1.7 `deps/` 根。

### 3.4 解码与稳定报告

TileLang 不使用 PPL autotune 的自动 pip 安装，也不把 PerfAI Web UI 文件当作稳定接口。内置
decoder 只接受 `ProfileParser.parse()` 返回的结构化结果，读取 BDC/GDMA/SDMA/CDMA 事件，写入
`tilelang_pcie_profile.json`。稳定 JSON 保存 engine、begin、end、unit、core、command id 和 opcode；
adapter 读入后计算 `duration=end-begin`，形成 `TPUInstructionTiming`。严格模式要求三个时间值有限、
`duration >= 0`、`end >= begin` 且 `unit="ns"`。

`pcie_decoder_python` 和 `pcie_decoder_pythonpath` 只进入离线 decoder 子进程，不进入编译或数值
worker。`preflight_pcie_decoder()` 在首次硬件派发前以无板卡权限的子进程验证包版本和 parser API；
失败时不会加载设备。

## 4. 使用方法

### 4.1 Python API

```python
from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig

config = TPUProfilingConfig(
    chip="sg2260e",
    programming_model="rv",       # 或 tpukernel
    runtime_mode="cmodel",
    output_dir="./profiles",
    label="rv-matmul",
    timeout_s=120,
)
report = TPUInstructionProfiler(config).run_cmodel(
    ["/path/to/python", "/path/to/fresh_worker.py"],
    environment={"PPL_PROJECT_ROOT": "/path/to/ppl-1.7"},
)
```

低层 PCIe API 使用 `run_pcie()`，并在 `environment` 中显式加入三项授权；需要逐指令时间时，
调用方还应先运行 `preflight_pcie_decoder()`。该 API 负责单个 profile 会话的授权、进程监管和
device lock，不验证 BM/SG CModel 晋级证据，也不执行 `tpu-smi` preflight/postflight 或持久隔离。
正式板端验收必须使用下一节的矩阵 runner。

### 4.2 矩阵 CLI

以下命令展示接口形态；板端运行还必须提供同 revision 的 BM1690 与 SG2260E CModel summary：

```bash
python3 testing/python/jit/tpu_core_ops_matrix.py \
  --runtime-mode pcie \
  --output-dir research/artifacts/<date>/<run> \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --require-decoded-timing \
  --bm-cmodel-summary <bm-summary.json> \
  --sg-cmodel-summary <sg-summary.json>
```

不指定 `output_dir` 的 Python API 会写入当前目录的 `tilelang-tpu-profiles/<label>-*`，不会默认写
`/tmp`。矩阵输出统一放在已被 Git 忽略的 `research/artifacts/`；每次运行使用独立目录，不覆盖
旧证据。

## 5. 验收与安全边界

本节的晋级、板卡健康检查和 quarantine 语义属于 `tpu_*_ops_matrix.py`，不是低层
`TPUInstructionProfiler.run_pcie()` 的隐式行为。

PCIe 数值/raw 验收要求 worker 数值通过，且 `cdm_profile_data_dev*` 中至少存在一个规范命名、
非空的 `global.profile` 或 `cdmlibN_N.profile`。增加 `--require-decoded-timing` 后，
`parser_status=ready`、至少一条 timing 和全部区间合法也成为硬门禁。SG2260E 当前每组 recorder
包含一个 `global.profile` 和四个核文件；summary 的 `raw_trace_file_count=1` 表示一组 recorder，
不是一个物理文件。

每个 worker、decoder 和外部 parser 都在私有进程组中运行，并受同一个单调总 deadline 约束。
父进程死亡、超时或异常时，supervisor 有界执行 TERM→KILL、reap 和 pipe drain。成功退出后若仍有
同进程组后代，也判为失败。硬件矩阵首错停止，不继续向状态未知的设备发射任务。

矩阵 runner 的外层设备锁覆盖 preflight、所有 case 和 postflight。每次发射后，runner 要求同一设备连续两个间隔
采样均为 `Active/0%`；非零利用率只可在有界 settle 中等待，`Fault`、拓扑错误、非法 JSON、probe
超时或未稳定空闲均立即失败并写 quarantine。当前正式 295 个 PCIe postflight 全部通过；最高
瞬时利用率 10%，最长稳定等待 1.904881 秒。

本轮完整 TPU-Kernel 整批、两个后续分片和 demo 整批曾各遇到一次主要管理遥测字段同时为 `F` 的 `Fault`。runner
均在首次异常处停止，并保存温度、时钟、利用率、电压等原始 payload；数值已通过的 case 也未晋升
为板端通过。受控进程退出、连续健康探测和精确 canary 通过后，正式验证改用更小的严格串行分片。
最终 TPU-Kernel 14 个分片精确覆盖 146 项，demo 15 个分片精确覆盖 51 项。失败任务和 canary
只作诊断，不计入 848。主要管理遥测字段同时为 `F` 表明当时相关遥测不可用，但不足以确定驱动、固件或 `tpu-smi`
中的具体根因；分片通过也不能证明长时间连续负载稳定性。

所有矩阵仅清理自己创建的 scratch。raw、decoded 报告和失败 payload 保留在
`research/artifacts/`，仓库不保留 `/tmp` 中间文件。

## 6. 报告字段

| 字段 | 含义 |
| --- | --- |
| `raw_trace_files` | CModel raw 文件，或 PCIe recorder 目录 |
| `raw_instructions` | 从 CModel sidecar 解析的命令；不含可靠时间 |
| `decoded_report_paths` | PCIe decoder 生成的稳定 JSON |
| `decoder_identity` | 实际包名、版本和 parser API |
| `timeline_events` | decoder 返回的完整有效时间线 |
| `instruction_timings` | 设备命令的 engine/core/id/opcode/begin/end/duration/unit |
| `parser_status` | `ready`、`unavailable`、`no-raw-trace`、`invalid-report` 等明确状态 |

## 7. 尚未覆盖

1. 本机缺少与当前 CModel raw 匹配的 PerfAI，尚无 CModel 单指令 duration。
2. BM1690 PCIe 尚未验证，SG2260E 结果不能外推。
3. 当前 timing 是单次诊断数据；性能回归需关闭 recorder，增加 warm-up、重复采样和统计阈值。
4. 工件还缺少统一的性能 manifest；在 resolved target、SDK、runtime、decoder、shape 和 dtype 全部
   固化前，不应自动比较不同构建的时间。
5. software pipeline 仍缺少 TPU dependency/hazard 模型；正确性矩阵继续使用依赖有序的串行路径。
