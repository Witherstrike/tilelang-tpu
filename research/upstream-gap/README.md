# TileLang-TPU：现状、能力边界与演进路线

> 范围：本文只说明 BM1690 与 SG2260E 的 TileLang-TPU 后端、PPL 1.7 工具链、
> TPU-Kernel 与 RV Tensor（RVT）两条编程模型，以及与上游 TileLang / TileLang-Ascend
> 的衔接策略。它不是性能宣传，也不把“能生成或链接”写成“板端算子已可用”。

## 结论先行

当前应把 TPU 的选择拆成四个明确维度，而不是用一个含义混杂的 target 或 mode 字符串。
其中 capability 是由物理芯片派生的不可选档案，不是第五个独立开关：

```text
物理芯片（chip） ──► 能力档案（capabilities）
编程模型（programming_model）
运行位置（runtime）
发射计划（launch policy；当前尚未公开）
```

- 物理芯片只有 `bm1690` 与 `sg2260e` 两个明确目标；它们共享 TPUv7 的大部分
  TPU-Kernel / LMEM 基础语义，但 PPL 架构、编译宏、物理核数和 RVT 能力不能靠
  默认值或文件夹名猜测。
- 唯一支持的工具链布局是 PPL 1.7 所采用的 `deps/` 发布布局。resolver 从其 chip map
  解析芯片，并检查核心头文件、运行库、固件、模拟器与 PPL helper；PCIe 交叉编译器在
  实际 build 阶段再检查。不再探测或兼容旧布局。
- 公开的 TVM target 规范写法是 `target="tpu -mcpu=bm1690"` 或
  `target="tpu -mcpu=sg2260e"`。旧 `chip=` 参数和早期的 `-model=` 写法仅在入口
  兼容，并且必须与 `-mcpu` 一致；规范化后的 Target 一定携带 `-mcpu`，但普通的
  workload `Target.model` 元数据会保留，只有值为已知芯片的 legacy `-model=` 才会移除。
  已规范化的内部 Target 还携带非公开的
  `-tpu-programming-model=tpukernel|rv`，由 `device_mode` 一次性写入；Python lower
  和 C++ codegen 都要求它与 capability 一致，不能从 extern 名称或编译宏反推模型。
  形如 `sg2260erv` 的未知、芯片样式 legacy `-model=` 即使同时给出了合法 `-mcpu` 也会
  拒绝，避免把拼写错误静默保留为 workload metadata。
- 原来的 `device_mode="atomic"` 不是“启用原子指令”的意思，而是传统 PPL
  TPU-Kernel 路径。当前公开规范名称是 `device_mode="tpukernel"`；RVT 对应
  `device_mode="rv"`。旧值只保留为有弃用告警的兼容别名。`TPUCompileConfig` 提供
  `programming_model` 只读语义属性，但它不是第二套参数名。
- `runtime_mode="cmodel" | "pcie"` 与编程模型独立。SG2260E 的物理核数为 4，但当前
  已验证的 TPU-Kernel matmul 仍是**单核发射**；“CModel 配置了 4 核”绝不等于
  “四核 kernel 已正确执行”。
- vendor TPU runtime（CModel 与 PCIe）是进程全局状态。首次加载会保留
  `(runtime, chip, core_count, programming_model, device, PPL SDK/runtime path identity)`；
  同一进程请求另一种组合会在 `dlopen` 前失败，必须改用新进程。这既防止
  `TPU_RT_CORE_NUM` 污染模拟器，也防止未验证 deinit 的 CModel/PCIe runtime 或另一套
  PPL runtime 相互复用；它不是多核调度功能。
- TPU `main.so` 也不是可随意 rehydrate 的普通共享库。TileLang JIT 的
  `LibraryGenerator` loader 当前只允许加载本实例刚刚编译出的私有产物，并复用当时捕获的
  PPL layout；任意预编译 TPU 产物（包括 cache/database）在 manifest 能打包并验证
  host/device 库、target 与 SDK/runtime 身份前一律拒绝。手写 `ctypes.CDLL` 不属于这条
  Python loader 合约，仍必须由调用方遵守 vendor runtime 的全部约束。
- SG2260E 的传统 TPU-Kernel CModel 已有隔离的 FP16 64×64 matmul 数值成功证据，且
  使用了规范的 `tpu -mcpu=sg2260e` target；可作为现有 PPL 算子继续 bring-up 的起点。
  RVT 目前只是 raw ABI bridge 与 CModel 控制路径已验证；RVT tensor 算子数值结果和
  任何 RVT PCIe 发射都尚未证明。
- 裸 `target="tpu"` 仍为兼容入口：它解析为 BM1690 的 TPU-Kernel 配置，默认 runtime
  为 PCIe；但 PCIe 动态加载仍由显式环境闸门拦住。新代码应总是指定 `-mcpu`；SG2260E 或
  RV 选择未指定 runtime 时默认 CModel。

后续工作必须先收紧上述边界，再逐步把通用 TileLang 二层算子接入 TPU 后端。不能因为
SG2260E 支持旧 TPU-Kernel，便假定所有现有 `ppl.*` handler 已在 SG 或 PCIe 上可用；
也不能因为 RVT 源码能编译，便把它当作可执行的通用 tensor lowering。

## 1. 证据标签与阅读方式

本文所有状态使用下列标签，避免把不同强度的结论混在一起：

| 标签 | 含义 | 不代表什么 |
| --- | --- | --- |
| **[实测]** | 受控运行已比较数值结果。 | 不代表其他形状、dtype、核数或运行位置也通过。 |
| **[控制路径]** | 已在 CModel 验证初始化、同步等生命周期调用可走通。 | 不代表 tensor 指令已配置正确并产生数值结果。 |
| **[静态]** | 已完成 IR、代码生成、编译或链接检查，未加载、未发射到板端。 | 不代表 ABI、运行时、数值或硬件稳定性正确。 |
| **[未验证]** | 当前没有足以得出结论的证据。 | 不能由相邻芯片、CModel 或静态结果外推。 |
| **[计划]** | 已定义实现方向与验收条件，尚未完成。 | 不是已经支持的 API。 |

## 2. 目标模型：按能力选择，而不是按特例分支

### 2.1 芯片 capability 档案

PPL 1.7 的 chip map 给出下列实际映射。它是工具链选择的输入，不应在各层各自
拼接 arch 名称。

| 逻辑芯片 | PPL 架构 | 物理核数 | 传统 TPU-Kernel | RVT | 当前结论 |
| --- | --- | ---: | --- | --- | --- |
| `bm1690` | `tpub_7_1` | 8 | 支持 | 未作为本项目的 RVT 目标承诺 | TPUv7 基线目标。 |
| `sg2260e` | `tpub_7_1_e` | 4 | 支持 | PPL 1.7 SDK 提供 RVT 头文件时支持选择 | 与 BM1690 共用基础 TPU-Kernel 语义，但必须用自身 arch、宏和核数。 |

SG2260E 除指令集扩展和 4 核拓扑外，当前已知的传统 TPU-Kernel 基础能力可复用
BM1690 的 TPUv7 共同档案。因此应采用“共同档案 + 芯片覆写”的结构，而不是把每个
`if (chip == "sg2260e")` 散落到 codegen、地址分配、JIT 和模板中。

建议的内部不可变对象如下：

```text
TPUChipSpec
  chip                    # bm1690 | sg2260e
  ppl_arch                # tpub_7_1 | tpub_7_1_e
  ppl_compile_definitions # __tpub_7_1__/__sg2260__ 等，唯一宏来源
  physical_core_count     # 8 | 4
  programming_models      # {tpukernel} 或 {tpukernel, rv}

共同 TPUv7 memory profile
  LMEM 对齐、bank 与 shape 规范

TPUCompileConfig
  chip                    # 与 Target -mcpu 一致
  device_mode             # tpukernel | rv
  runtime_mode            # cmodel | pcie

规划中的 LaunchPlan
  launch_policy           # 当前模板固定 single_core；尚不是公开配置
```

`physical_core_count` 是硬件拓扑；`launch_policy` 是本次 kernel 实际会使用的核数。
二者必须分开存储、分开显示、分开测试。SG2260E 的四核信息应当用于能力校验和未来
分片规划，不能自动复制参数并把同一个 kernel 发到四个核上，否则会产生重复写入或
数据竞争。

### 2.2 命名与兼容策略

公开配置应使用下列语义：

```python
target="tpu -mcpu=sg2260e"
device_mode="tpukernel"          # 或 "rv"
runtime_mode="cmodel"            # 或显式 "pcie"
```

当前模板固定单核发射；`launch_policy` 是未来必须显式引入的对象，不能把它伪装成
现有 API 已支持的参数。

`target="auto"` 也不能成为隐藏的第三种选择方式。它仍优先选择 CUDA/HIP；只有用户同时
设置 `TILELANG_TPU_CHIP`（受 `TPUChipSpec` 校验）和可由 `PPLLayout` 验证的
`PPL_PROJECT_ROOT` 时，才会返回规范的 `tpu -mcpu=<chip>`。否则回落到可移植 C backend，
而不是旧的裸 `tpu`/BM1690 PCIe 默认值。这样“自动”只省去已明确配置的 target 拼接，
不承担物理芯片猜测或板端授权。

过渡期可以接受兼容输入，并在入口一次性规范化。当前只有误导性的 `atomic` 会发出
`DeprecationWarning`；其余兼容输入尚未承诺弃用时点：

| 兼容输入 | 规范化结果 | 当前行为与原因 |
| --- | --- | --- |
| `device_mode="atomic"` | `device_mode="tpukernel"` | 有弃用告警；它选择的是 PPL TPU-Kernel 编程模型，不是原子算子开关。 |
| `chip="sg2260e"` | 写入 `Target -mcpu=sg2260e` | 当前兼容；显式 target 才是新调用的规范写法。 |
| `target="tpu -model=sg2260e"` | `-mcpu=sg2260e`，移除该 legacy model | 当前兼容；只有 `model` 值为已知芯片时才这样处理。 |
| `mode="pcie"` | `runtime_mode="pcie"` | 当前兼容、无告警；`mode` 不应同时承载设备代码和运行位置的含义。 |

规范化之后的中间层、缓存键、生成代码守卫和报错信息只使用新名称。这样既不破坏旧
调用，又不会让新 API 继续传播含义不清的 `atomic` 名称。

`Target.model` 不是新的 chip 参数：它仍可存放如 `matmul_smoke` 的 workload 元数据。
不过当它看起来像芯片 SKU（`bm...` / `sg...` 加数字）时，它就是历史设备选择拼写；未知
SKU 必须失败，已知 SKU 必须与 `-mcpu` 一致。这样可同时避免两个相反的错误：把普通
模型标签当芯片，或把 `sg2260erv` 之类错别字悄悄当作普通标签。

每个 kernel 必须只选择一种编程模型。`tpukernel` 的 `ppl.*` 调用和 raw `rvt_*` ABI
调用不能在同一个 kernel 内静默混用；两者的描述符、命令、同步和副作用模型尚未有
已验证的组合契约。若未来确需混合，必须先定义显式桥接 op、资源所有权和 fence，
并以独立测试证明，而不是放宽当前的模型围栏。

### 2.3 PPL 1.7 工具链边界

`PPLLayout` 的职责是把一个已验证的 PPL 1.7 `deps/` 根目录和逻辑芯片转换为 SDK
布局信息：chip map、include、runtime、firmware、emulator 与 PPL helper。resolver 缺失
这些核心工件时立即报出具体路径；PCIe cross compiler 在 `LibraryGenerator` 的实际 build
阶段检查并报路径错误。两层都不回退到另一套目录，也不混合不同布局的头文件和库。

一次成功编译会把 PPL 根目录、TPUv7 runtime 库目录和 chip backend 库目录的规范路径
随私有产物一起保存在 `LibraryGenerator` 中；随后的 `dlopen` 不会重新追随环境变量里的
`PPL_PROJECT_ROOT`。这个 path identity 是同进程隔离手段，不是对磁盘内容的签名或哈希
证明；因此它不能使任意预编译 `main.so` 安全，预编译加载仍须等待 manifest。

本轮实测使用的本地 SDK 是 `ppl_v1.7.122-g05ebfb36-20260528`。工具链不再把该版本目录
名写死；PCIe build 在 `third_party/toolchains_dir/*/bin/` 中要求恰好一个
`riscv64-unknown-linux-gnu-gcc`。这允许 PPL 小版本升级，又会在 SDK 同时放入多套交叉
编译器时明确报歧义，而不是随机选一个。

RVT 是 capability 条件，不是第三种物理芯片：只有在 `chip="sg2260e"` 的 SDK 档案
实际提供 `rvt_api.h` 时，`device_mode="rv"` 才合法。SDK 不满足时应在编译前
失败，绝不能悄悄改走 TPU-Kernel 路径。

本机审阅的 PPL 1.7 `chip_map.json` 还列出 `sg2260erv`，但该 SDK 包没有对应的
`tpub_7_1_e_rv` 完整目录、配置或库；反而 `sg2260e → tpub_7_1_e` 同时具备
`tpu_kernel.h` 与 `rvt_api.h`。因此 public chip 只暴露 `sg2260e`，由
`device_mode="tpukernel" | "rv"` 选择 ISA/编程模型；不能把目录名当作第三颗芯片。

## 3. 当前状态与证据边界

| 能力/断言 | 状态 | 证据 | 本证据没有覆盖的范围 |
| --- | --- | --- | --- |
| BM1690 / SG2260E 的 PPL 1.7 arch 与物理核数选择 | **[静态]** | PPL 1.7 layout resolver 从 chip map 得到 `tpub_7_1`/8 与 `tpub_7_1_e`/4，并检查核心 SDK 工件。 | 不证明交叉编译器、任一 op 的数值正确性或板端可用性。 |
| SG2260E 传统 TPU-Kernel matmul，CModel | **[实测]** | 隔离 matmul 已完成数值比对。 | 仅是该测试范围、单核发射；不代表所有 `ppl.*` op、尾块、异步或 PCIe。 |
| SG2260E 传统 TPU-Kernel，PCIe | **[静态]** | 已对 `ppl.fill/copy` 的最小 kernel 完成交叉编译、链接，未加载 `main.so`、未初始化板端。 | 不代表 PCIe ABI、数值或板端稳定性；尚无硬件成功结论。 |
| 历史 PPL host demos 的 TPUv7 宏 | **[静态]** | 示例 host source 已从误导性的 `__bm1690__` 条件改为 PPL 1.7 的 `__sg2260__` 或 `__sg2260e__` 共同 TPUv7 runtime 条件。 | 没有逐个编译或运行这些历史 demo；不能把宏修正写成每个 demo 的 SG2260E 支持。 |
| TPU runtime/profile 与产物来源隔离 | **[静态]** | `LibraryGenerator.load_lib()` 仅接受本实例编译的私有 `main.so`，并在任何 CModel/PCIe `dlopen` 前预留 `(runtime, chip, physical_core_count, programming_model, device, PPL SDK/runtime paths)`；BM/SG、CModel/PCIe、RVT/TPU-Kernel、PCIe device 或 SDK path 切换会失败。 | 不是签名的持久 manifest；不代表多核 kernel 已发射，也不替代每个 kernel 的数值测试。 |
| RVT raw bridge | **[静态]** | `tilelang/language/rvt.py` 将受限的 `rvt_*` 名称原样发为 C ABI extern；codegen 在 RVT 模型下加入相应头文件与守卫。 | 它不构造 CR/TR/GR 描述符，也不是 tensor op lowering。 |
| RVT CModel 控制路径 | **[控制路径]** | 隔离进程中实际运行 `rvt_kernel_start → rvt_sync_all`，CModel kernel launch 返回成功。 | 已签入测试主要覆盖静态编译/链接；不证明 DMA、算术指令、描述符生命周期或 tensor 数值。 |
| RVT tensor 指令 | **[静态]** | `rvt_fadd` 等 raw 调用已做编译/链接覆盖。 | 该调用没有可执行 tensor 描述符；没有 RVT tensor 数值结果。 |
| RVT PCIe | **[静态]** | 仅完成交叉编译、链接和产物检查；未加载动态库、未初始化板端、未 dispatch。 | 不代表任何 PCIe 可运行性或板端稳定性。 |
| 多核执行 | **[未验证]** | 当前 host kernel 模板实际使用 `core_num = 1`。 | CModel 的四核拓扑设置不能替代四核发射与同步验证。 |

这张表也给出近期策略：先把 SG2260E 的既有 TPU-Kernel 路径用于小范围、单核 CModel
正确性基线；RVT 保持为隔离实验路径，直到最小 DMA–计算–回写链获得 CModel 数值证据。

## 4. 当前实现的结构性差距

### 4.1 target 信息仍有多处来源

当前代码的关键事实分布在 Python JIT、PPL layout、C++ 地址分配、PPL codegen 与 host
模板中。下面不是要求将所有东西塞进一个大类，而是规定每项信息只能有一个权威来源：

| 信息 | 当前可见位置 | 应收敛到的权威对象 | 风险 |
| --- | --- | --- | --- |
| logical chip、PPL arch、编译宏、SDK 路径 | `tilelang/engine/tpu_config.py`、`tilelang/jit/adapter/ppl_layout.py` | `TPUChipSpec` + `PPLLayout` | 编译宏、库与芯片不匹配。 |
| 编程模型与运行位置 | `tilelang/engine/tpu_config.py`、lower/JIT/cache 参数 | 规范化后的 `TPUCompileConfig` | 旧 `device_mode` 容易把 TPU-Kernel、RVT、CModel、PCIe 混为一谈。 |
| PPL SDK/runtime 与可加载产物 | `PPLLayout`、`LibraryGenerator` | 编译时捕获的 `PPLLayout.runtime_identity` + 私有 `main.so` 路径 | 若加载时重读环境或接受预编译库，可能混用 runtime ABI 或绕过运行时闸门。 |
| LMEM shape、对齐、bank | `src/target/tpuv7_lmem.h`、`src/transform/address_assign.cc`、`src/target/codegen_ppl.cc` | 共同 `tpuv7` memory profile，按 capability 引用 | 若未来芯片真实改变 LMEM 几何，必须新增 profile，不能继续隐含复用。 |
| 物理核数与实际发射 | PPL layout 与 `src/tl_templates/tpu/kernel_template.cpp` | `physical_core_count` 与 `launch_policy` 两个字段 | 把 4 误当作“可自动多核执行”。 |
| op 的读写 effect 与 PPL 发射 | `address_assign.cc` 和 `codegen_ppl.cc` 的各自字符串分支 | 后端 op specification / 注册表 | 新增 op 时地址、别名、尾块和 codegen 容易不同步。 |

第一阶段已将 `bm1690_lmem` 的共同规则改为语义中立的 `tpuv7` 档案，并让 C++ PPL
codegen 接收带 `-mcpu` 的 Target。native codegen 现在还强制要求内部
`tpu-programming-model`，并检查 `chip × model`（BM1690 只允许 TPU-Kernel，SG2260E
允许 TPU-Kernel/RVT）；Python 的 preflight 和 C++ final fence 都会把 `ppl.*`/`tpu_*`
归为 TPU-Kernel、把 `rvt_*` 归为 RVT，禁止同一 kernel 混用。真正决定 PPL arch、编译宏、
核数与 SDK RVT 头文件可用性的仍是 Python 的 `TPUChipSpec`、`PPLLayout` 与
`LibraryGenerator`。尚未完成的是把完整 capability 对象传入每个 native pass；后续新增
芯片仍必须在 registry 增加显式档案。

### 4.1.1 已修复的 backend 污染，和仍需拆出的后端垂直切片

此前 `lower`、JIT adapter 与 `LibraryGenerator` 会无条件构造 BM1690/`atomic` 配置，导致
CUDA/HIP/CPU 等非 TPU target 也携带 TPU 状态，甚至走到 PPL codegen。现在只有
`target.kind == "tpu"` 才解析 `TPUCompileConfig`、绑定 `-mcpu`/内部模型属性、创建 PPL
workspace；非 TPU 分支恢复其原有 `device_codegen` 路径。这是与 AMD/NVIDIA target 选择
等价的最小隔离，而不是声称 TPU 已拥有完整上游 backend plugin。

同一条边界还会扫描明确的 `ppl.`、`tpu_`、`rvt_` vendor extern：若 target 已解析为
C/CUDA/HIP（包括无 TPU opt-in 的 `auto`），lower 会立即提示用户显式选择
`tpu -mcpu=<chip>`，而不会生成带无效 `ppl.foo()` 调用的普通 C 源码。历史
`tpu_demo/ppl/` 中仍有裸 `tilelang.lower(func)` 的脚本；它们现在会得到这条明确诊断，
需要逐个迁移为显式 `-mcpu`、`device_mode`、`runtime_mode` 配置，不能再依赖过去的
隐式 BM1690 默认值。

下一步仍应把这些 TPU 文件迁入独立 backend context（见第 7 节）。原因是通用
`tilelang/engine/lower.py` 目前仍要保留一处 TPU 分支来调用历史 PPL codegen；真正的
软工终态应让 target normalizer、pipeline、toolchain、runtime manifest 和 codegen 都由
TPU backend 注册，从通用 engine 移除 TPU 专属知识。

### 4.2 当前 TPU 更接近 PPL extern 后端，而不是通用 TileOp 后端

现有 PPL codegen 与地址分配已对下列专用名字保留 handler：copy、fill、gemm、部分
逐元素、reduce sum/max、RoPE、gather 与 topk。它们是有价值的 bring-up 资产，但多数
仍是 `ppl.*` 字符串分发；通用 `T.copy`、`T.fill`、`T.gemm`、`T.reduce` 并没有完整的
TPU 专属 lowering 链。

同时，通用 TileLang phase 中 layout inference、`LowerTileOp` 和安全访存相关路径并不能
直接为 TPU 打开：现有通用 op lowering 主要继承 GPU 线程、warp、layout 假设。TPU 需要
自己的语义检查、LMEM 地址规划、尾块处理和 PPL/RVT lowering，而不是把 GPU pass 解除
注释。

### 4.3 代码生成所见的张量元数据不应依赖名字

`codegen_ppl.cc` 目前维护 `__ppl_tensor_info`，部分处理还根据变量名或局部 scope 推断
张量位置；`address_assign.cc` 则另外维护 extern 的访问关系。这能支撑早期 demo，但无法
可靠地支持别名、in-place、动态边界、尾块、异步依赖或多核分片。

应引入一个后端内部 `TpuOpSpec`（名称可调整），为每个已支持 op 集中声明：

```text
op kind / operand role / 读写 region / dtype 与 shape 限制
LMEM 或 global placement / 对齐与尾块规则 / 临时 workspace
同步依赖 / 可用编程模型 / 具体 PPL 或 RVT lowering key
```

地址分配、安全访问检查、模型围栏和 codegen 都消费这一个声明。未注册的 op 必须在
编译期给出明确诊断，不能落入猜测式 codegen。

### 4.4 单核正确性先于多核性能

SG2260E 的四核拓扑是能力属性，不是 launch 参数默认值。当前模板固定单核是合理的
安全基线：同一组输入/输出参数没有分片协议时，直接增大核数只会让各核做同一工作。

未来多核必须以显式 `LaunchPlan` 实现，至少包括：每核 tile 范围、global offset、输出
不重叠证明、跨核归约方式、buffer 生命周期、同步机制和失败回收。先用单核 CModel
对每个 op 建数值基线，再做一维分片 copy / elementwise，最后才考虑 GEMM 分块和归约。

### 4.5 持久缓存仍故意关闭，不能把“已有规范 key”误读为可安全复用

当前 cache/database 的 TPU 读写仍 fail-closed，`LibraryGenerator` 自身也拒绝任意未由本
实例刚编译的 TPU `main.so`。虽然未来 key 已会规范化 legacy
`atomic`、`-model=` 与 `-mcpu`，这只解决了同一逻辑选择生成不同 key 的一小部分问题。
在允许持久化前还必须解决：resolved `target="auto"` 的最终 target 记录、PPL
SDK/toolchain ABI 身份、pass 配置、`main.so` 对私有 `libkernel.so` 的绝对路径依赖、
chip/model/runtime/device 的 manifest，以及 database rehydrate 时 TPU 配置的完整传递。
否则缓存命中可能把错误的 host/runtime 组合 `dlopen` 到当前进程，绕过本应存在的安全判定。

因此这不是“性能优化以后再补”的细节，而是 P0 安全契约：只有 manifest 打包、加载前
校验和跨进程测试都完成后，才能开启 TPU 持久缓存。

## 5. 二层算子覆盖与缺口

下表刻意区分标准二层 API 与历史 PPL 兼容扩展。“静态 handler”只表示当前后端存在
专用处理，不等价于通用 API、完整语义或 SG 硬件通过。

| TileLang 二层能力 | TPU-Kernel 当前状态 | RVT 当前状态 | 下一步及优先级 |
| --- | --- | --- | --- |
| 标准 `T.copy` / `T.fill` | **[未验证]**：没有完整 TPU 专属 lowering 链。 | **[未验证]**：没有通用 lowering。 | **P0**：统一 buffer scope、shape/stride、cast、tail 与读写 effect，先做单核 CModel。 |
| PPL 兼容 `T.ppl_copy` / `T.ppl_fill` | 有对应 `ppl.copy` / `ppl.fill` handler **[静态]**；已在 matmul 链中经 CModel 间接覆盖。 | 有 raw DMA ABI 入口 **[静态]**；没有描述符 builder。 | **P0**：将兼容入口和标准 TileOp 收敛到同一语义 IR，再补独立数值矩阵。 |
| 标准 `T.gemm` | **[未验证]**：尚未形成通用 TPU lowering。 | **[未验证]**。 | **P0**：定义 layout/dtype/tail 契约并接入标准 TileOp。 |
| PPL 兼容 `T.ppl_gemm` | `ppl.gemm` handler；SG2260E FP16 64×64 CModel matmul **[实测]**。 | 没有 RVT GEMM lowering。 | **P0**：扩展 matmul 基线；不能把单例成功写成通用 GEMM。 |
| 标量/逐元素、cast、compare、select | add/sub/mul/div、部分标量与数学函数有 codegen 处理 **[静态]**，但没有二层语义契约。 | `fadd/fsub/fmul/fmac` 为 raw 调用 **[静态]**，未有数值证据。 | **P1**：先定义广播、in-place、舍入、NaN/Inf 与 dtype 语义，再按 capability 接入。 |
| 标准 `T.reduce` / reducer 与 PPL `ppl_reduce_*` | 标准 API **[未验证]**；仅见受限 PPL sum/max 专用路径 **[静态]**。 | 无通用 lowering。 | **P1**：先连续轴 sum/max/min，再推广 axis、rank、init、tail；scan 与跨核归约后置。 |
| layout、地址与安全访存 | 手写 LMEM 地址规划可工作于现有路径，但 effect 与 codegen 分散。 | 无描述符/寄存器生命周期建模。 | **P0**：引入 `TpuOpSpec` 和统一地址/边界检查。 |
| `transpose` / `im2col` / `c2d_im2col` | 未形成通用接入。 | 无。 | **P2**：先澄清 DMA / workspace / layout 契约，再做专用实现。 |
| gather / topk | 有特殊 PPL handler **[静态]**。 | 无。 | **P2**：补 index dtype、边界、稳定性、workspace 和 CModel 矩阵。 |
| async copy、pipeline、同步 | 当前没有可验证的 TPU 依赖 IR；PPL 标记不等于调度模型。 | 有 raw control/sync 入口 **[控制路径]**，不是调度器。 | **P2**：以 DMA/compute dependency token 建模，不能复用 CUDA/Ascend barrier 语义。 |
| 多核、原子型归约、scan | 未验证。 | 未验证。 | **P2–P3**：在单核语义与输出分片证明之后再做。 |
| 诊断、profiling、autotune | 缺少统一的后端能力矩阵与失败产物。 | 同左。 | **P1**：保留 IR、生成源、PPL 命令与运行状态；不支持项应编译期失败。 |

## 6. 分阶段路线图

### P0：目标正确性与可重复的单核基线

| 工作项 | Why | 具体措施 | 验收 |
| --- | --- | --- | --- |
| 收敛 capability registry | 已完成第一阶段的 BM1690 / SG2260E 显式档案、Target 绑定与 PPL resolver 交叉校验；否则新增 SG 分支会继续放大漂移。 | 继续让模板/后续 native pass 只消费规范化 Target 或显式 capability，而不自行拼接芯片字符串。 | 非法 chip/model/runtime 组合在编译前失败；为未来持久缓存准备的 canonical key 只存规范值（当前持久缓存读写仍禁用）。 |
| target-kind 双环境 CI | `tpu` target 既要能在干净 TVM 中由 TileLang 注册，也要能兼容早期已注册 `-mcpu` 的安装；只在一类开发机 import 成功不足以证明安装脚本安全。 | 用 fresh-process CI 分别构建/导入干净 TVM 与 legacy-patched TVM，验证 `tpu -mcpu=... -tpu-programming-model=...` 的 parse、bind 与 codegen；拒绝缺少 `-mcpu` 的同名 target。 | 两条 lane 都不修改 TVM 的 `target_kind.cc`，且注册/扩展无重复 key 或晚期 ABI 错误。 |
| 完成 `atomic` 到 `tpukernel` 的语义迁移 | 旧名称误导使用者并妨碍后续文档与日志。 | 公开 API 使用 `device_mode="tpukernel"` 或 `"rv"`；旧 `atomic` 只在入口兼容、告警并立即规范化。 | 新生成源、错误、缓存和文档不再把旧名称当作规范名称。 |
| 固定 PPL 1.7 解析 | 混用旧/新 SDK 的头文件与库会造成难诊断 ABI 问题。 | 只接受 PPL 1.7 `deps/` 布局，检查必需工件和 SDK RVT 头文件。 | 旧布局、未知芯片、缺 RVT 头文件均有确定错误；BM/SG 的 arch 与核数单测通过。 |
| 建立 TPU-Kernel 单核 CModel 基线 | SG 已有一个 matmul 成功点，但没有系统性边界。 | 为 copy、fill、elementwise、gemm、sum/max 建小尺寸、非对齐尾块、多个 dtype 与错误路径测试。 | 每项先有数值结果、IR/codegen golden 和 negative test；未通过者不进入 PCIe。 |
| 收敛 op effect 与 LMEM 规则 | 双重字符串分发会使地址和 codegen 行为不一致。 | 引入 `TpuOpSpec`；将共同 BM/SG LMEM 规则从硬件名中抽离为 `tpuv7` memory profile。 | 新增一个 op 只需一个语义注册点；未注册 op 明确失败。 |
| 设定板端与运行时安全闸门 | 错误 kernel 可能使设备或宿主调用卡住，且 vendor runtime 没有已验证的进程内 deinit 生命周期。 | 库内在 PCIe `dlopen` 前检查 `TILELANG_TPU_ALLOW_PCIE_LOAD=1`、严格的非负 device-id，只接受本实例刚编译的私有产物，并为所有 CModel/PCIe load 预留同一个 `(runtime, chip, cores, model, device, PPL SDK/runtime paths)` 档案；测试 harness 由外部受监控子进程执行，超时/异常时监督器终止父进程及进程组并跳过余下硬件用例。 | CModel 未通过时没有 PCIe job；不同 CModel/PCIe/SDK 档案必须用新进程；外部 harness 失败能保留最小诊断产物并安全停止。 |
| 完成 TPU 持久缓存 manifest | 当前 cache/database 被有意禁用；没有 ABI、私有库和 runtime identity 就复用产物会破坏上述闸门。 | 固化 resolved target、PPL SDK/toolchain ABI、pass config、`libkernel.so` bundle、chip/model/runtime/device manifest，并在加载前逐项校验。 | `auto` target 与显式 target 不碰撞；错误 SDK/runtime/device 的缓存绝不 `dlopen`。 |

### P1：通用语义最小闭环与 RVT 数值最小链

| 工作项 | Why | 具体措施 | 验收 |
| --- | --- | --- | --- |
| `T.copy` / `T.fill` / `T.gemm` 的 TPU lowering | 当前用户仍要借助 `ppl.*`，二层 API 不能跨后端复用。 | 将通用 TileOp 先降为规范化 TPU op，再由 `TpuOpSpec` 选 PPL TPU-Kernel lowering；保留 `ppl_*` 仅作兼容层。 | 标准 `T.*` 与兼容入口共享同一 IR、CModel 数值测试和清晰限制。 |
| 逐元素与 reduction planner | attention、归一化与激活依赖这些基础组合。 | 定义 dtype、broadcast、axis、init、tail 和 in-place 规则；按 PPL 实际能力逐项开放。 | 支持集合有 capability table；任何未实现组合在编译期报错而非错误生成。 |
| RVT 最小 tensor 链 | raw bridge 不能证明 RVT 可以计算。 | 先实现显式 descriptor/register builder，验证 `DMA load → 一条算术 → DMA store` 的单核 CModel 数值链；对资源与同步做生命周期检查。 | 至少一个输入–输出 tensor 测试获得数值证据；raw ABI 仍保留为专家级 escape hatch。 |
| 可诊断性 | PPL、CModel 和 IR 问题必须可定位，尤其是 PCIe 前。 | 保存规范化 target、IR、生成 C、PPL 命令、标准输出/错误与启动策略。 | 失败报告能指出 chip、模型、runtime、op 与阶段，而非只报告链接失败。 |

### P2：复杂二层算子、流水与受控性能路径

| 工作项 | Why | 具体措施 | 验收 |
| --- | --- | --- | --- |
| layout / 索引算子 | transpose、im2col、gather、topk 是上层 attention/conv 常用组成。 | 先做局部 DMA/reshape，再引入专用 kernel；明确 workspace、索引 dtype、边界和稳定性。 | 每个 op 有约束表、CModel 数值矩阵和不支持路径。 |
| DMA–计算流水 | 真实性能依赖重叠，但过早优化会隐藏正确性问题。 | 设计 TPU 原生 dependency/event IR，区分 DMA 与 compute；不复用 CUDA 或 Ascend 的 barrier 含义。 | 单核 pipeline 先证明无读写冲突，再与无 pipeline 基线做等价检查。 |
| RVT lowering 扩展 | RVT 的价值在受控地覆盖适合它的 tensor 指令，而不是把所有 PPL op 重写一遍。 | 为每条候选指令记录 descriptor、数据类型、对齐、同步和 fallback；只选有数值测试的子集接入。 | 每个已宣称 RVT op 都有 CModel tensor 数值证明与模型围栏测试。 |
| SG2260E PCIe 最小冒烟 | CModel 不能覆盖真实 runtime/driver/ABI。 | 审阅静态产物后，一次只跑一个最小、单核、受 watchdog 管理的数值用例。 | 成功需包含加载、发射、回传和数值比对；超时即停止该批而非继续试错。 |

### P3：多核与上游 backend 迁移

| 工作项 | Why | 具体措施 | 验收 |
| --- | --- | --- | --- |
| 显式多核 launch plan | SG2260E 的四核只有在数据分片、同步和归约正确时才有价值。 | 从无写冲突的 copy/elementwise 开始，逐步实现 per-core range、offset、output ownership 和同步；GEMM/归约最后进行。 | 单核/多核结果一致，且有竞争检测、尾块和异常回收测试。 |
| TileLang backend 垂直切片 | 继续在通用 engine 中堆 TPU 特判会使升级和 AMD/NVIDIA 等后端选择机制脱节。 | 建立 `tilelang/tpu/` 和 `src/tpu/`：target normalizer、capability、language extension、pipeline、codegen、toolchain hook；将 PPL SDK 保持为可选依赖。 | 选择 TPU 的逻辑由 backend context 一次解析，通用 engine 不含 TPU 专属分支。 |
| 上游贡献与长期矩阵 | 私有 SDK 代码不宜直接上游，但通用 backend 接口和 op contract 有复用价值。 | 先在本项目稳定 API 与回归矩阵；再提出无 PPL 依赖的 normalizer / backend hook / contract 测试。 | 发布按 chip、模型、runtime、核数、op 的支持矩阵，而非笼统“TPU 支持”。 |

## 7. 与上游 TileLang / TileLang-Ascend 的衔接

本节的上游边界以 TileLang 官方的
[Backend Layout](https://github.com/tile-ai/tilelang/blob/main/tilelang/backend/README.md)
和 [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) 为参照。前者已经将
target normalizer、pass pipeline 与 host/device codegen 做成可注册的后端边界；后者是
NPU 后端工程分层的参考，而不是 TPU 指令接口的模板。

### 7.1 对接 TileLang：采用 backend context，而不是重载一个 target 字符串

当前上游 TileLang 的后端架构将一个目标后端视为完整的垂直切片：语言方言、target
规范化与 context、显式 pass pipeline、host/device codegen；构建、加载、发射则由可复用的
execution backend 负责。TPU 应顺着这个边界接入：

```text
通用 TileLang 二层 API
        │
        ▼
TPU target normalizer ──► TPUChipSpec / TPUCompileConfig
        │                         │
        ▼                         ▼
TPU backend pipeline       model fence + PPL 1.7 validation
        │
        ├── TPU-Kernel lowering ─► PPL 1.7 codegen
        └── RVT lowering        ─► RVT descriptor/codegen
        │
        ▼
共享执行层：CModel 或受控 PCIe launch
```

这和 GPU 在同一 TileLang 前端中通过 capability / backend 选择 CUDA、ROCm 等目标的原则
一致：目标差异属于已解析的 backend context，而非散落在通用 lower、缓存和 codegen 的
字符串判断中。可实施的目录边界为：

```text
tilelang/tpu/
  target.py          # normalizer、capability registry、诊断
  language.py        # TPU 专属扩展；raw RVT 放在明确命名空间
  pipeline.py        # TPU 专属语义检查、lowering、地址/尾块处理顺序
  codegen.py         # PPL TPU-Kernel / RVT 的选择与产物契约
  toolchain.py       # PPL 1.7 resolver 和构建钩子
src/tpu/
  memory_profile.*   # 共同 TPUv7 LMEM 规则
  op_spec.*          # effect / constraint 注册
  codegen_*.*        # 后端专属 native codegen
```

迁移期间可以保留旧的 `ppl_*` 兼容入口，但新功能必须从标准 `T.copy`、`T.fill`、`T.gemm`、
`T.reduce` 等语义 API 进入。PPL 与 RVT 名称应留在后端实现或显式专家扩展中，不能成为
默认的可移植用户 API。

### 7.2 借鉴 Ascend 的工程分层，不复制其硬件接口

TileLang-Ascend 的价值在于展示一个 NPU 后端需要完整处理：目标绑定、layout、二层 op
lowering、安全访问、内存规划、pipeline、同步和诊断。它可作为“编译器必须拥有这些语义
层”的参照，而不是 API 或指令的一对一翻译。

| Ascend 类问题 | TPU 应采用的原则 | 不能直接照搬的部分 |
| --- | --- | --- |
| 局部存储层级与 tile placement | 在 `TpuOpSpec` 中明确 global/LMEM、对齐、生命周期与 workspace。 | L0A/L0B/L0C、UB 的名称、容量和分配规则。 |
| Cube / Vector 算子分工 | 为 TPU-Kernel / RVT 分别声明可用 op 与 dtype，不让前端猜测。 | Ascend 专有 Cube/Vector intrinsic 与指令选择。 |
| async / flag / barrier | 用 PPL DMA、TPU compute 与 RVT 的真实依赖模型构造 token。 | Ascend flag/barrier 的编号、可见性和时序。 |
| cross-core pipeline | 先证明输出 ownership 和同步，再逐步开放多核。 | Ascend 的核间切分策略和默认并行度。 |

因此，上游重构的优先级不是“先移植 Ascend”，而是先完成 P0/P1 的 TPU 语义闭环；随后把
稳定的 target normalizer、backend manifest、op contract 和测试作为可能的上游贡献。PPL
SDK 路径、固件、专有 ABI 与板端安全策略应继续保留在可选的 TPU 后端插件内。

## 8. PCIe 运行安全与验收门槛

PCIe 不是常规单元测试后端。当前 TileLang JIT loader 已实现的是 PCIe `dlopen` 的显式
许可、严格 device-id 闸门、仅加载本实例刚编译的私有产物、`main.so` 内对加载后环境
变更的 device 绑定校验，以及所有 CModel/PCIe 加载共用的含 PPL SDK/runtime path identity
的 process runtime-profile 闸门；“独立监督器、超时后杀父进程组、跳过后续板端用例”仍
必须由外部验收 harness 落实，不能误写成库内已经具备 watchdog。手写 `ctypes.CDLL` 不在
该 Python loader 合约内。CModel smoke 和后续 PCIe bring-up 必须用**不同进程**。每个准备
上板的变更遵循如下闸门：

1. 静态检查：target、PPL layout、编译宏、库、生成源与模型围栏全部通过。
2. CModel：先完成数值测试，至少覆盖一个非对齐尾块和一个失败路径。
3. 审阅：确认实际 `launch_policy`、buffer 大小、copy-back、超时和进程隔离策略。
4. PCIe：在新进程中仅发射一个最小单核用例；监督器独立于被测进程，超时或异常时终止
   该任务的父进程及其进程组，跳过剩余硬件测试。
5. 扩展：只有加载、发射、回传和数值比对全部成功后，才增加 dtype、shape、流水或多核。

任何阶段的失败都应留下规范化 target、IR、生成源码、PPL 命令和日志。静态构建成功或
CModel 控制路径成功都不能绕过下一道门。

## 9. 主要代码与证据阅读路径

### TileLang-TPU

- `tilelang/engine/tpu_config.py`：`TPUChipSpec`、chip、`device_mode` 和
  `runtime_mode` 的单点规范化，以及 `tpu -mcpu` 绑定。
- `tilelang/utils/target.py`：CUDA/HIP/TPU 的 auto selection；TPU 只能由显式 chip 与
  已验证的 PPL SDK opt-in，不再将无 GPU 主机猜成 bare TPU。
- `src/target/tpu_target_kind.cc`：在 TileLang 自身注册 `tpu` Target；不再由安装脚本覆盖
  TVM 的 `target_kind.cc`。它兼容已含且接受 `-mcpu` 的旧注册，并会拒绝不兼容的同名
  target，避免晚些时候才产生含混错误。
- `tilelang/jit/adapter/ppl_layout.py`：唯一 PPL 1.7 目录解析、芯片 arch、核数、RVT
  头文件检查与规范的 SDK/runtime path identity。
- `tilelang/jit/adapter/libgen.py`：PPL CModel / PCIe 编译、链接、私有工作目录、工具链
  参数，以及拒绝未验证预编译 TPU 产物的加载边界。
- `tilelang/jit/adapter/tpu.py`：process-global runtime profile、TPU host ABI 与串行执行
  边界；它防止 CModel/PCIe、chip、模型、device 和 PPL SDK/runtime identity 在一个
  Python 进程中切换。
- `tilelang/language/rvt.py`：raw RVT C ABI bridge；它刻意不承担 tensor descriptor builder。
- `src/target/codegen_ppl.cc`、`src/transform/address_assign.cc`：现有 PPL handler、effect
  推断和需收敛的字符串分发。
- `src/target/tpuv7_lmem.h`：BM1690 / SG2260E 当前共同的 TPUv7 memory profile；若
  新芯片有真实几何差异，应新增 profile 而非改回硬编码芯片名。
- `src/tl_templates/tpu/kernel_template.cpp`：当前单核发射事实的直接来源。
- `src/tl_templates/tpu/main_template.cpp`、`tilelang/jit/adapter/tpu.py`：CModel 核数设置、
  PCIe 许可/device-id 闸门、加载后 device 绑定校验与 host 运行时边界。
- `testing/python/jit/test_tpu_config.py`、`testing/python/jit/test_ppl_layout.py`、
  `testing/python/jit/test_tpu_rvt.py`、`testing/python/jit/test_tpu_adapter.py`：capability、
  PPL layout、模型围栏、PCIe 加载闸门与 RVT 静态/私有编译覆盖的边界。

### 上游 TileLang

- `tilelang/backend/README.md`：target backend 与 execution backend 的职责边界。
- `tilelang/backend/module.py`、`tilelang/backend/target.py`：backend manifest、context 和
  target normalizer 的参考实现。
- `tilelang/backend/pass_pipeline/`：显式后端 pass pipeline 的组织方式。
- `tilelang/language/`：通用二层 API 的语义面；TPU 以此作为兼容目标，而非继续扩张
  `ppl_*` 公开接口。

以上路线将“SG2260E 能跑一个传统 TPU-Kernel matmul”“RVT 能生成 raw ABI 调用”和
“TileLang-TPU 已具备可移植二层算子后端”这三个不同阶段明确隔开。当前处于前两者之间：
先得到可重复、可诊断、单核正确的基础能力，再安全地推进 RVT、PCIe、多核和上游接入。
