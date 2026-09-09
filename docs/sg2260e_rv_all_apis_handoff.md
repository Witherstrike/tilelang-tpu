# SG2260E RV 全 API 开发与设备验证交接

更新：2026-09-09。分支：`feature/sg2260e-rv-backend-cmodel-v2`。

## 当前状态与边界

18 个公开 `T.ppl_*` 入口均有 RV 实现与测试样例。这里的“接通”指
源码生成、构建路径和测试入口，不表示所有 dtype/shape 均支持，也不表示
设备数值已验收。本次环境无 TPU，未执行 cmodel/PCIe 数值测试。
没有修改 TVM 子模块，没有使用此前排除的失败分支，也没有新增 agent 报告。

| 公开 API | 当前实现与限制 |
| --- | --- |
| copy | global↔local、global→global、local→local；跨 dtype 转换先在 local 完成；不支持重叠区域的 memmove 语义 |
| add/subtract/mul/div | local FP16/FP32；lhs 与输出同形；rhs 各维允许 1 或同形 |
| add_C/mul_C | local FP16/FP32；CR scalar；重复调用使用独立 C 临时作用域 |
| fill/clear | local FP16/FP32；clear 为 fill(0) |
| gemm | FP16 输入、FP32 累加；NN/NT/TT；TN 显式拒绝；调用者负责清零 |
| rsqrt | local FP16/FP32，常量迭代次数 1–8 |
| gather | global FP16 参数/输出、UINT32 索引；param_h 匹配表高度；未扩展 local gather |
| exp2/sigmoid | local FP32；exp2 按现有 API 语义计算 exp(x)，不是 2**x；两个不同工作缓冲区，不得与输出重叠 |
| reduce_sum/reduce_max | local HW-aligned 2D，FP16/FP32，dim=1，输出(M,1)，输入输出分离；max 保留 clear=False 语义 |
| rope_add | local HW-aligned 2D FP16/FP32；偶数列宽；输出不得与输入 alias |
| topk | global contiguous FP32/INT32/UINT32；索引 INT32/UINT32；0<K≤length，常量参数，三个缓冲区分离 |

所有 shape 要满足既有 RV descriptor 静态尺寸/寄存器约束。未知操作或越界
配置仍在编译期报错，不回退 `tpu_bdc_*`/`tpu_gdma_*`/`tpu_hau_*`。
不同 Buffer 名称但共享同一物理存储的隐式 alias 由调用者避免。

## 实现要点

- `codegen_ppl_rv.cc` 保持独立 RV emitter。算术使用 rvt 指令；没有通过
  PPL 编译器重新运行存在 blocker 的 reduction/top-k lowering。
- exp 用 exp(x/2)²、范围约简和 7 阶 Horner 多项式，FP32/int32 重解释
  重建指数。输入钳位 [-104,89]，保留 NaN；coeff/table 参数为兼容公开
  宏保留，不读其中内容。该算法的舍入、subnormal、NaN 行为必须设备验证。
- reduction 逐列累加/取最大值；rope 用 FREE-stride 偶/奇切片。
  这是功能基线，不是优化后的向量 reduction 实现。
- top-k 为稳定重复选择，先按值再按原索引排序，NaN 排在有限值和无穷之后，
  升降序均如此；输入不修改。复杂度 O(K²×length)，大规模使用前必须优化。
  AddressAssign 预留 7×64 字节标量 tile；scratch、容量和保守全程 live range
  写入 `tir.tpu.lmem.__rv_topk_scratch.*`。最多另需 7 个 TR、4 个 CR，
  超过寄存器上限则编译错误。需重点检查混合 float compare / integer index
  select、同 bank scratch 和同步行为。
- `rv_legalize.cc` 解析公开宏生成的 access_ptr let，并折叠常量几何 let；
  因此测试不仅覆盖手写 call_extern，也覆盖真实 `T.ppl_*` 宏。
- 缺少下游 TPU target 注册的 CPU-only TVM 由 TileLang 条件注册 target；
  已有注册不覆盖。标准 host/keys 等属性与 TVM target 契约一致。
- SG2260E allocator 沿用 chip description 和不复用不同 buffer 地址的保守
  策略；已有 PPL final MLIR 对比测试继续有效。新复合算子没有等价 PPL
  final MLIR golden，不将其描述为已通过 PPL 地址一一对比。

## 本地静态验证

本次结果：C++ 构建成功；下列回归集 158 passed、3 skipped（opt-in 设备
测试）；29 个串行样例全部完成 cmodel 编译链接，无 CDLL 加载/设备初始化。
PCIe 仅验证配置分发、命令构造、checker/firmware/runtime 选择及缺失编译器
诊断；未执行交叉链接或实机运行。串行/pipeline C 语法检查均已通过。

在仓库根目录执行：

```bash
cmake --build build -j2
export PYTHONPATH="$PWD:$PWD/3rdparty/tvm/python"
export TVM_LIBRARY_PATH="$PWD/build/tvm"
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/tvm:${LD_LIBRARY_PATH:-}"
export PPL_PROJECT_ROOT="$PWD/../ppl_v1.7.122-g05ebfb36-20260528"
export TVM_BACKTRACE_LIMIT=0
.venv-cpu/bin/python -m pytest -q \
  testing/python/target/test_ppl_rv_all_apis.py \
  testing/python/target/test_tilelang_codegen_ppl_rv.py \
  testing/python/transform/test_tilelang_transform_rv_legalize.py \
  testing/python/transform/test_tilelang_transform_address_assign.py \
  testing/python/transform/test_ppl_final_mlir_lmem_compare.py \
  testing/python/jit/test_tpu_config.py \
  testing/python/jit/test_ppl_layout.py
```

`TVM_BACKTRACE_LIMIT=0` 是本环境的测试设置：gdb 将非法调用时的崩溃定位到
TVM `BacktraceSyminfoCallback`，不是 top-k 执行。关闭长回溯后能捕获原始
编译诊断；未修改用户已有的 TVM 子模块。

测试覆盖导出 API 集合、两种 runtime 的完整 engine.lower、29 个串行/
pipeline C 语法样例、负向契约、top-k scratch、PCIe 命令构造与旧 corpus。
C 语法检查需要 PPL SDK，缺失时 skip；不可将 skip 记作通过。

## Runtime 构建与设备端执行

cmodel 使用 PPL 1.7 SG2260E headers、helper、checker、emulator/runtime，
沿用四核配置。PCIe 使用 RISC-V 固件，host wrapper 使用主机 g++。

```bash
# PCIe 部署选择其一配置交叉编译器；显式路径优先。
export PPL_RISCV_CC=/absolute/path/to/riscv64-unknown-linux-gnu-gcc
# 或 CROSS_TOOLCHAINS 下包含 Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/
# export CROSS_TOOLCHAINS=/absolute/path/to/toolchains
export PPL_PCIE_RUNTIME_LIB=/absolute/path/to/hardware/tpuv7/lib
```

未设置 runtime override 时使用 SDK runtime_lib。确保这是真机 runtime，
不是 emulator stub；设备驱动、firmware、host runtime 版本必须匹配。
未找到 compiler/firmware/checker 会给出明确错误；本机未执行 PCIe 交叉链接。

只构建链接、不 CDLL 加载、不初始化设备：

```bash
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py \
  --case all --runtime cmodel --compile-only
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py \
  --case all --runtime pcie --compile-only
```

建议设备验证顺序：串行 copy → add → gemm，然后其余算子，再 pipeline。
每个 case 独立 Python 进程；`--case all` 自动顺序启动并设单例 300 秒超时，
失败最后汇总。不要并发启动：现有 wrapper/build 共用
`src/tl_templates/tpu/` 产物，后一个 case 会覆盖前一个。

```bash
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case copy --runtime cmodel
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case add --runtime cmodel
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case gemm --runtime cmodel
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case all --runtime cmodel
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case all --runtime pcie
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case all --runtime cmodel --pipeline
.venv-cpu/bin/python testing/python/target/test_ppl_rv_all_apis.py --case all --runtime pcie --pipeline
```

这里 pipeline 样例验证结构化 parallel region，不等同于多 stage software
pipeline 调度的全覆盖。已有 `test_tilelang_codegen_ppl_rv.py` 另保留两 K-tile
GEMM 累加和 pipeline cmodel 用例，可按旧文档单独进程启用。

输出缓冲区初始为 42，整数精确比较、浮点 rtol=2e-5/atol=2e-6；这不是
放宽误差的授权。exp 极值用例包含 NaN/±Inf，但绝对误差不验证 subnormal
的相对精度，设备 agent 应另外检查该项。wide reduction 覆盖65行33列，
top-k 覆盖正反序、重复值、NaN、整数极值；还包含 clear=False、重复 scalar/
gather、global copy、local cast。

## 设备 agent 必须记录

1. 分支/提交、SDK、交叉编译器、驱动、runtime、芯片版本及全部环境变量路径。
2. 每个 case 的 runtime、pipeline、退出码、checker/编译/执行日志、误差。
3. 失败后在运行下一 case 前保存 `src/tl_templates/tpu/` 下的 kernel.c、
   kernel.cpp、main.cpp、libkernel.so、main.so；区分编译、链接、初始化、
   descriptor/checker、数值、超时失败。
4. 优先验收新 exp/sigmoid、top-k 的混合类型比较、wide reduction 的地址跨度，
   然后扩展 dtype/shape、并发/多核、真正多 stage pipeline 和性能测试。

全部设备门禁通过之前，不应把本分支标记为 SG2260E RV 数值验收完成。
