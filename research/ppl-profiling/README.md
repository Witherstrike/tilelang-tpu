# TileLang-TPU 的 PPL 指令 Profiling 设计与实现

> 本文记录 PPL 1.7 profiling 机制、TileLang-TPU 的接入边界，以及验收标准。
> 这里的“指令耗时”专指 PerfAI 解码出的 TPU 命令引擎时间线；CModel 原始 dump
> 本身不包含可靠的 begin/end 时间，不能把它伪装成耗时数据。

## 1. 结论

TileLang 不应把 PPL 的 `ppl_compile.py --profiling` 或 `--autotune` 原样塞进 JIT。
TileLang 已经直接生成 PPL `kernel.c`，不是 PPL 前端的 `.pl` 输入；PPL 的前端开关会
调用原生 `ppl-compile --autotune`，生成 PPL 专属的 autotune host glue。直接复用该开关会
重复/错配代码生成，而不会给现有 TileLang host ABI 正确添加 profiling。

可复用的是 PPL 的运行时协议：

```text
fresh CModel test worker
  └─ FILE_DUMP_CMD=<relative label>
       └─ CModel raw BD/GDMA/SDMA dump + .txt sidecar
            └─ explicit external PerfAI AutoRunner
                 └─ PerfWeb/profile_data.js
                      ├─ all timeline events
                      └─ recognized TPU command lanes → per-command duration
```

首版实现提供独立的 `TPUInstructionProfiler`，而不是复用 CUDA-only 的
`tilelang.profiler.Profiler`。后者依赖 CUDA Event 与 `torch.cuda.synchronize()`，不适合
TPU runtime。

## 2. 对 PPL `--profiling` 的审阅

本机 PPL 1.7 的 `python/tool/ppl_compile.py` 有以下行为：

| PPL 行为 | 实际含义 | TileLang 处理 |
| --- | --- | --- |
| `--profiling` | 已弃用的 `--autotune` 入口。 | 不新增同名 JIT 编译选项。 |
| `--profiling` / `--autotune` | 先运行 native `ppl-compile --autotune`，再 build 与 profile。 | 不移植前端 codegen；TileLang 自己已经生成 kernel/host 源。 |
| CModel autotuner | 一次 call + sync 前设置 `FILE_DUMP_CMD`，之后清除。 | 在一个新的 CModel 子进程中设置相同环境契约。 |
| CModel 后处理 | 在 PerfAI 中运行 `AutoRunner.sh -d <raw-dir> -e <chip>`。 | 仅在用户明确提供可执行 PerfAI 时运行；绝不自动安装 Python 包或下载工具。 |
| PCIe autotuner | TPUv7 用 `tpudnnEnableProfile`/sync/`tpudnnDisableProfile`。 | 当前不实现、更不执行 PCIe profile；见第 6 节。 |

PPL 的 `profiling_parser.py` 只打印 `summary_data` 聚合项。逐命令数据实际在 PerfAI
的 `PerfWeb/profile_data.js` 的 `time_data` 中；因此本实现解析该时间线，而不把总时间
误当成单条指令耗时。

## 3. 接口与数据模型

```python
from tilelang.jit import (
    TPUInstructionProfiler,
    TPUProfilingConfig,
)

config = TPUProfilingConfig(
    chip="sg2260e",
    device_mode="tpukernel",      # 或 rv；与物理 chip 组合会校验
    runtime_mode="cmodel",
    output_dir="/tmp/tilelang-profiles",  # 每次运行创建新子目录
    label="matmul",
    timeout_s=60,
    perfai_root="/opt/PerfAI",    # 可选；不提供时保留 raw trace
)
report = TPUInstructionProfiler(config).run_cmodel(
    command=["/path/to/python", "/path/to/fresh_compile_worker.py"],
    environment={
        "PPL_PROJECT_ROOT": "/path/to/ppl-1.7",
        "PYTHONPATH": "...",
        "LD_LIBRARY_PATH": "...",
    },
)
```

`command` 是可信的测试 worker：它必须在子进程内 fresh-compile 并加载自己的私有
TileLang TPU artifact，不能加载缓存/数据库/其他 JIT 实例遗留的 `main.so`。profiler
会写入 `TILELANG_TPU_PROFILE_CHIP`、`...DEVICE_MODE`、`...RUNTIME_MODE`，供受控 test
worker 显式传给 `tilelang.compile`；它不尝试从任意外部命令反向推断实际编译配置。

`TPUProfileReport` 分三层保存证据：

| 字段 | 可信含义 |
| --- | --- |
| `raw_trace_files` / `raw_instructions` | CModel 原始命令及其 core、command id、opcode；没有制造耗时。 |
| `timeline_events` | PerfAI 的完整时间线，包含 host、subnet、layer、TPU 命令等。 |
| `instruction_timings` | 仅 `bd/bdc/tiu/gdma/sdma/vsdma/cdma/dma` 命令通道；包含 begin、end、duration、unit 和原始字段。 |

不同 PerfAI 版本的附加列不完全相同。解析器保留完整 `fields`，并只在列名或 PPL
历史格式 `func_type="bd_id=…"`/`"gdma_id=…"` 足够明确时投影 `core_id`、`command_id`
和 `opcode`。命令关联是 best-effort；跨 core 或多次 launch 时不能只按 id 做全局唯一键。

## 4. CModel 安全与可复现性

每次 profile 会话都使用独立临时结果目录、独立 cwd 和新进程组。超时会先终止整个
worker process group，再在必要时 SIGKILL；父进程的 cwd 和环境变量不会改变。会话还会：

- 固定 `TILELANG_TPU_BENCHMARK_RUNS=0`，确保一次编译/一次 dispatch，避免 benchmark
  循环污染 dump；
- 设置与 chip 对应的 `TPU_RT_CORE_NUM`（BM1690 为 8，SG2260E 为 4）；
- 清除继承的 `TILELANG_TPU_ALLOW_PCIE_LOAD`、`TILELANG_TPU_ALLOW_PCIE_PROFILE`、
  `TILELANG_TPU_DEVICE_ID` 与 `BMLIB_ENABLE_ALL_PROFILE`，使 CModel worker 不能因父
  环境意外沿用板卡授权；
- 要求 `FILE_DUMP_CMD` 是相对、安全 label。实测 CModel 对绝对路径 label 不产生预期
  raw dump，因此必须借由私有 cwd 定位输出。

PerfAI 不存在或未显式配置时，状态为 `unavailable`，但 raw artifacts 和 worker 日志会被
保留。不要在这种情况下报告“每条指令耗时已经得到”。本机 PPL 1.7 包没有
`third_party/PerfAI/AutoRunner.sh`，所以真实 PerfAI 耗时尚待外部工具到位后验收。

PPL 自身在 PerfAI cwd 中使用 `auto_build`。TileLang 不复制其 `rm -rf auto_build` 行为；
它会按 resolved PerfAI root 建立 host-temporary inter-process lock，串行化同机 TileLang
profile session。直接运行 PPL/其他工具的用户仍必须与该 root 排他，直到 AutoRunner 的
并发/工作目录契约在该版本中被实测确认。

## 5. 当前测试与验收标准

新增的 unit tests 覆盖：

- raw binary + `.txt` sidecar 的解析，包含 BD/GDMA command id/opcode；
- CModel 私有 cwd、4-core 设置、PCIe gate 清理与 benchmark=0；
- 显式 PerfAI fixture 的时间线解析、host 事件过滤、历史 `func_type` ID 格式；
- RV 模式映射到 vendor PerfAI 名 `sg2260erv`（它不是新的 TileLang chip）；
- 非法 BM1690+RV 配置、NaN/Infinity timeout、PCIe device id 越界；
- CModel worker 超时后的进程组终止。

已接入的真实 CModel case 默认不运行，避免普通 pytest 意外启动 vendor emulator。准备好
PPL、TileLang runtime 和 pytest 后，使用外层 watchdog 显式启用：

```bash
setsid --wait timeout --kill-after=5s 60s \
  env TILELANG_TPU_RUN_CMODEL_PROFILE=1 \
      PPL_PROJECT_ROOT=/path/to/ppl-1.7 \
      pytest testing/python/jit/test_tpu_profiling.py \
      -k 'sg2260e and cmodel_profile_worker'
```

这两个 case 调用 `tpu_profile_worker.py`；worker 会读取 profile session 的
`chip/device_mode/runtime_mode`，将其显式传给 `tilelang.compile`，然后在自己的进程内
fresh-compile。它不是从环境猜测 target，也不会使用已有 JIT artifact。

本分支已经通过受控 worker 做了两次真实、隔离的 CModel 验证：

| worker | 已验证证据 | 未覆盖范围 |
| --- | --- | --- |
| `tpukernel-matmul` | fresh-compile SG2260E FP16 64×64 matmul，数值 `allclose=True`；产生 24 个 raw artifact，解析 78 条命令（BD 30、GDMA 34、SDMA 14）。 | 无真实 PerfAI，因此没有 measured duration；不代表其他 dtype/shape、异步、多核或 PCIe。 |
| `rv-control` | fresh-compile `rvt_kernel_start → rvt_sync_all`，worker 成功返回；产生 24 个 raw artifact，解析 48 条控制命令（BD/GDMA/SDMA 各 16）。 | 不是 RV tensor arithmetic 或 descriptor 数值验证。 |

这证明 `FILE_DUMP_CMD` 与现有直接 `tpuRtKernelLaunch` host path 能收集 CModel 原始指令
流；它不证明 PerfAI duration 解码，也不代表 RV tensor 数值或 PCIe profile 已成功。

正式验收应依次满足：

1. 受控 worker fresh-compile SG2260E TPU-Kernel 小算子，数值比对通过，且 `raw_instructions`
   非空；
2. 在已安装、独占的 PerfAI 上，`instruction_timings` 非空且带合理的 unit；
3. RV 先只验 `rvt_kernel_start → rvt_sync_all` 的 CModel 控制路径和 trace；在 CR/TR/GR
   descriptor 的数值链完整前，不把 `rvt_fadd` 当做 tensor 数值测试；
4. PCIe 只做 profile host variant 的静态 compile/link；获得人工一次性授权后才单次板端
   dispatch，并由新进程/超时看门狗控制。

## 6. PCIe：明确未实现

`pcie_profile_environment_overrides()` 目前只做三重授权的预检，并返回 **environment overrides**：
`BMLIB_ENABLE_ALL_PROFILE=1`、`PROFILE_RECORD_SIZE`、`PROFILE_BOOK_KEEPING`。它不返回完整
环境，也不加载库、初始化设备或发射 kernel。

这不是逐指令 PCIe profile 实现。PPL 的 TPUv7 路径还需要 TPUDNN handle 与
`tpudnnEnableProfile(handle, record_size, book_keeping)` / launch / sync /
`tpudnnDisableProfile`，并通过 `bigTpuProfile` + PerfAI 解码 `cdm_profile_data_dev*`。
当前 TileLang host 直接调用 `tpuRtKernelLaunch`，没有这段 TPUDNN session，也没有链接对应
profile host variant。因此必须保持 PCIe `run` API 不存在；任何“只设置环境变量即可看到
PCIe 指令耗时”的说法都是错误的。

未来 PCIe 任务应是独立的、可审计的 host variant：在 `dlopen` 前同时要求
`TILELANG_TPU_ALLOW_PCIE_LOAD=1`、`TILELANG_TPU_ALLOW_PCIE_PROFILE=1`、合法 device id；
新进程中只执行一次 launch，超时杀整个进程组，先静态验证再经人工授权实际运行，且绝不在
TileLang 中自动 `pip install` PPL 的外部解析器。

## 7. 已完成的双后端提升与剩余工作

本功能先在 main 派生分支独立研发，再合并至 SG2260E 双后端分支。提升时已经完成：

1. `TPUProfilingConfig` 直接使用 `TPUCompileConfig` / `TPUChipSpec` 规范化 chip、
   `atomic → tpukernel` 兼容别名、programming model、runtime 和 physical core count；
   profiling 不再维护第二份能力表；
2. 保留双后端 `tilelang.jit.compile()` 的 `chip`、`device_mode`、`runtime_mode` 传递与
   target 规范化，只增加 profiling public exports；
3. 增加 `testing/python/jit/tpu_profile_worker.py`。它只接受 profiler 注入的 CModel 三轴
   选择，清楚拒绝 PCIe 环境，且在子进程内 fresh-compile 受控 case；
4. 增加 opt-in pytest CModel cases：`TILELANG_TPU_RUN_CMODEL_PROFILE=1` 才会运行真实
   worker，常规单测不会意外启动模拟器；
5. 同步更新 `research/upstream-gap/README.md` 的证据等级与缺口。

仍待完成的是外部 PerfAI 的真实 SG2260E 输出兼容验收，以及 PCIe 的独立 TPUDNN profile
host variant。两项都不能由当前 fixture、raw trace 或静态链接替代。
