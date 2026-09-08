# TileLang TPU 算子示例

本目录只保留面向用户的高层算子示例。算子模块负责构造 TileLang 程序和
PyTorch oracle，但导入模块不会编译、加载运行库或访问设备。低层指令与回归探针统一放在
`testing/python/jit`，不再与示例混放。

## 环境

首次构建先初始化子模块并安装 TileLang TPU 扩展：

```bash
git submodule update --init --recursive
./install_tpu.sh
```

运行前令 `PPL_PROJECT_ROOT` 指向统一的 PPL 1.7 SDK，再加载最小运行库环境：

```bash
export PPL_PROJECT_ROOT=<ppl-1.7-sdk>
source tpu_demo/env.sh
```

修改 C++ codegen 后应使用仓库已经配置的 CMake build 目录重新构建；不要在 demo 目录保存
生成的 C 文件或临时产物。

## 目录与入口

```text
tpu_demo/
├── cases.py                 # 语义用例与后端能力注册表
├── common.py                # target/runtime、数值比较与公共门禁
├── run.py                   # 单个 CModel 用例的唯一公开入口
├── elementwise/             # add、sub、mul、div
├── matmul/
├── rmsnorm/                 # 普通版与 split-k
├── rope/
├── swiglu/
└── flashattn/
```

用户代码通过 `tpu_demo.run.run_case` 或等价的 `python3 -m tpu_demo.run` 运行一个
CModel 用例；不要直接执行各算子模块。完整验收只通过
`testing/python/jit/tpu_demo_ops_matrix.py`，由它为每个 case 创建隔离 worker、收集
profiling 证据并执行板端安全门禁。

注册表共有 36 个语义用例：elementwise 四种操作、matmul、RMSNorm、RMSNorm
split-k、RoPE、SwiGLU 各覆盖 FP16/BF16/FP32；FlashAttention 还为每种 dtype
分别注册 `balanced`、`descending-max` 与 `weighted-keys` 三个输入变体。
`descending-max` 刻意让后一个 K tile 的行最大值下降，用来检验跨 tile online-softmax
的历史最大值合并；`weighted-keys` 使用单调变化的 key logits 和与 key 相关的 value，
使均匀权重、argmax 近似或丢失 tile 内权重差异的错误无法被常量 value 掩盖。elementwise
与 matmul 共 15 个用例可选择 TPU-Kernel 或 RV Tensor；其余复合算子当前只选择
TPU-Kernel，注册表不会把缺失的 RV lowering 当成隐式支持。

查看精确 case id 不会加载 TPU runtime：

```bash
python3 testing/python/jit/tpu_demo_ops_matrix.py --list-cases
```

单个 CModel 示例：

```bash
python3 -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

编译身份由完整 target 的 `chip + programming_model` 决定，运行身份由
`runtime_mode` 独立决定。BM1690 是 8 核 TPU-Kernel target；SG2260E 是 4 核，支持
TPU-Kernel 与 RV Tensor。裸 `target="tpu"`、BM1690/RV 或复合算子/RV 组合均会明确
拒绝。

## 数值语义

每个 case 只执行一次 kernel launch，并与同 dtype、同 shape 的 PyTorch oracle 比较。
比较前先严格核对 shape、dtype 和有限值，再应用按算子族与公开 dtype 划分的 atol/rtol。
所有容差都由 `tpu_demo/common.py` 按算子族与 dtype 统一选择；算子实现不得按单次实验
临时覆盖。

FP32 是公开输入输出语义，不代表矩阵引擎使用 FP32 乘数。TPU-Kernel 与 RV Tensor
的矩阵指令都不接受 FP32 A/B，因此 matmul 和 FlashAttention 的 FP32 路径先将乘数
量化为 BF16，再以 FP32 累加；oracle 必须复现该输入量化边界。FlashAttention kernel
还会在 probability/value GEMM 前把 FP32 的指数权重 tile 转成矩阵引擎 dtype，而 oracle
有意保留理想 FP32 softmax，不复制这一步实现舍入；两者的可接受偏差由 FlashAttention
专属容差限定。RMSNorm 与 SwiGLU 则在低精度公开输入上使用 FP32 中间计算，最后转换回
公开 dtype。

当前示例要求静态正整数维度和整 tile 划分，不提供静默越界的 tail 路径。不满足整除
条件的 shape 会在编译或派发前失败。FlashAttention 的 `is_causal` 是严格 bool 参数：
非 bool 值直接报类型错误，`True` 则明确报尚未实现。当前只覆盖 `False`；在对角 tile
mask 正确建模前，不暴露只缩短 K 循环的不完整 causal 快捷路径。

## 验收顺序

正式证据必须使用同一份干净源码，分三次 invocation 严格按以下顺序晋级：

1. BM1690 CModel：36 个 TPU-Kernel case；
2. SG2260E CModel：36 个 TPU-Kernel case，加 15 个可适用的 RV case，共 51 个；
3. SG2260E PCIe：只调度前两阶段均有通过证据的同一组 selector，最多 51 个。

CModel 示例命令如下；`research/artifacts` 已被 Git 忽略，输出目录必须尚不存在，runner 会
原子占用该路径并拒绝混入或覆盖任何已有内容（包括已有空目录）。

CModel runner 默认尝试 PPL `--profiling` 对应的可选 PerfAI 后处理；若
`PPL_PERFAI_ROOT` 或 `PPL_THIRD_PARTY_PATH/PerfAI` 下没有兼容 `AutoRunner.sh`，summary 会
保留非空 raw trace 并明确写 `parser_status=unavailable`。这不影响数值验收，但不能解释为已
获得逐指令 duration。

```bash
python3 testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip bm1690 \
  --programming-model tpukernel \
  --output-dir research/artifacts/<date>/tpu-demo-bm-cmodel

python3 testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip sg2260e \
  --output-dir research/artifacts/<date>/tpu-demo-sg-cmodel
```

PCIe 不是普通 demo 模式，`tpu_demo.run` 不接受它。矩阵在任何硬件 launch 前同时要求：

- `--allow-pcie` 与 `--allow-pcie-profile` 两个独立确认；
- 明确且只能为 `0` 的 `--device-id`；当前机器还必须恰好只看见一张卡、一个 chip，并在
  `sg-host-drv` 下解析到唯一的 `1f1c:1690` PCIe 设备，才可证明逻辑 id 与物理板卡一致；
- `--case`/`--op` 的显式子集，或额外的 `--all-pcie-cases` 全量确认；
- `--bm-cmodel-summary` 与 `--sg-cmodel-summary`，两份 summary 必须完整通过、来自同一
  clean commit 和 source state，并覆盖即将上板的每个 selector；CModel 与 PCIe 当前实际
  使用的 TileLang/TVM 动态库、host 编译器、PPL 公共与逐芯片头文件、backend、emulator 和
  CModel runtime 也必须具有相同的内容摘要；
- 若本轮要求逐指令耗时，则增加 `--require-decoded-timing` 及显式 decoder 配置；decoder
  会在首次硬件派发前完成无硬件 preflight。

全量 PCIe 命令形态为：

```bash
python3 testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode pcie \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --bm-cmodel-summary research/artifacts/<date>/tpu-demo-bm-cmodel/summary.json \
  --sg-cmodel-summary research/artifacts/<date>/tpu-demo-sg-cmodel/summary.json \
  --require-decoded-timing \
  --pcie-decoder-python <decoder-python> \
  --pcie-decoder-pythonpath <decoder-directory-or-wheel> \
  --output-dir research/artifacts/<date>/tpu-demo-sg-pcie
```

矩阵对每个 case fresh-compile，并令 worker 恰好 launch 一次；不运行 benchmark 重复轮次。
PCIe invocation 从 preflight 到最终检查始终持有同一个 `/run/lock` 设备锁。runner 先对
交叉工具链、firmware、TPUDNN、安装的 board runtime 与实际 `tpu-smi` 做内容寻址，再从
晋级提交生成私有、只读的 Git execution snapshot；每个 case 发射前重新核对共享 checkout
与 manifest 所覆盖的编译/runtime 输入，worker 只从该快照导入源码，因此“检查通过后源码被替换”的窗口不会改变实际
编译输入。

该 manifest 精确覆盖已加载的 TileLang/TVM 动态库、host C/C++ driver 文件、PPL 公共与逐芯片
目录，以及 PCIe 专属的交叉工具链根、firmware、TPUDNN、board runtime 和 `tpu-smi`。它不是
hermetic build 证明：host GCC 的内部程序、系统头文件/链接器/libc、Python/PyTorch 环境和可选
decoder 文件内容仍需由部署或实验报告单独固定。

每个 worker 位于带 parent-death 约束的独立进程组中。编译、加载、超时、数值、raw
profile、严格 timing 或板卡健康检查任一失败，矩阵都会保存部分 summary，终止受控父进程组
并跳过剩余 case，不在同一 runtime 实例中重试。若进程组在有界 TERM→KILL 后仍未完全回收，
设备会写入持久 quarantine marker；若 runner 被强制杀死，持久 session marker 会保留。两者
都会让后续 invocation fail-closed，不能因 `tpu-smi` 看似空闲就继续。操作者必须先检查记录的
PID/PGID、恢复板卡并确认无残留进程，再人工移除对应 marker。PCIe 前置检查与每个 case 的
后置检查都要求设备为 `Active`；runner 在整张设备锁内按单调总 deadline 轮询，只有同一物理
设备连续两次间隔采样为 `0%` 才允许下一次 launch。命令返回后的短暂非零利用率属于待收敛
状态，不会被误判为新任务可立即复用；超时、无效输出或设备身份变化仍会持久 fail-close。

## 结果解释

`summary.json` 的数值结果、raw 指令和 decoded timing 是三个不同证据层：

- 数值通过只证明该精确 dtype、shape、variant 与 target 的 oracle；
- 非空 raw trace 证明本次 launch 产生了可检查的设备命令；
- decoded timing 才能描述单条设备指令的时间，单次采样只用于映射与诊断，不是稳定性能
  基准。

契约状态记录在 `research/tpu-op-contract/contract.json`，本轮高层算子的设计与验收进度记录在
`research/tpu-demo-ops/README.md`。忽略目录中的 artifact 是本机证据，不应直接提交 Git。
