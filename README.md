# TileLang-TPU

TileLang-TPU 为 TileLang 增加了 SOPHGO TPU 后端。用户仍然使用 TileLang Python DSL
编写程序，编译器根据目标芯片和编程模型生成 TPU-Kernel 或 RV Tensor 代码，并通过
CModel 或 PCIe 运行。

## 当前支持范围

| 芯片 | 物理核数 | 编程模型 | CModel | PCIe |
| --- | ---: | --- | --- | --- |
| BM1690 | 8 | TPU-Kernel | 已验证 | 本项目尚未验证 |
| SG2260E | 4 | TPU-Kernel | 已验证 | 已验证 |
| SG2260E | 4 | RV Tensor | 已验证 | 已验证 |

这里的“已验证”只覆盖能力契约中列出的算子、数据类型、形状和参数，不能直接推广到任意
输入。精确范围见
[`research/tpu-op-contract/contract.json`](./research/tpu-op-contract/contract.json)，
测试结论见
[`research/tpu-backend-design/test-report.md`](./research/tpu-backend-design/test-report.md)。

## 开始使用

- 安装依赖、配置 PPL 1.7 和构建项目：
  [`docs/get_started/Installation.md`](./docs/get_started/Installation.md)
- 查看高层算子、单例运行和完整测试矩阵：
  [`tpu_demo/README.md`](./tpu_demo/README.md)
- 查看 TPU-Kernel 与 RV Tensor 的设计和指令映射：
  [`research/tpu-backend-design/README.md`](./research/tpu-backend-design/README.md)

安装完成后，可以先运行一个 SG2260E RV Tensor CModel 用例：

```bash
python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

## 目标选择

TPU 编译需要同时指定芯片和编程模型：

```python
import tilelang

kernel = tilelang.compile(
    program,
    out_idx=-1,
    target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel",
    runtime_mode="cmodel",
)
```

三个选择项各有明确职责：

| 选择项 | 取值 | 作用 |
| --- | --- | --- |
| `-mcpu` | `bm1690`、`sg2260e` | 选择芯片架构、编译宏和核数 |
| `-tpu-programming-model` | `tpukernel`、`rv` | 选择设备端指令接口 |
| `runtime_mode` | `cmodel`、`pcie` | 选择模拟器或真实板卡运行时 |

芯片和编程模型共同决定生成什么设备代码；`runtime_mode` 只决定如何运行这些代码。裸
`target="tpu"`、缺少任一编译选项或使用 BM1690 + RV 的组合都会在编译前报错。

源码生成部分按目标后端的常见结构组织：

```text
TileLang 前端
  -> TPU 语义检查与 Pass
  -> codegen_tpu             TPU 目标源码生成器与编程模型分派
       |- codegen_tpukernel  TPU-Kernel 指令选择
       `- codegen_rv         RV Tensor 指令选择
  -> CModel 或 PCIe 运行时
```

## 算子接口

当前前端提供以下 TPU 表达：

- 两种编程模型均可映射的核心操作：`T.ppl_copy`、`T.ppl_fill`、`T.ppl_gemm`、
  `T.ppl_add`、`T.ppl_subtract`、`T.ppl_mul`、`T.ppl_div`、`T.ppl_max`。
- TPU-Kernel 专属操作：标量运算、`exp`、`sigmoid`、`rsqrt`、reduce、gather、
  top-k 和 RoPE 等。
- RV Tensor 低层接口：`T.rvt_*`，用于显式管理 CR/TR/GR 描述符的场景。

高层 `T.ppl_*` 调用会先变成稳定的 TPU 语义，再由目标选择具体指令。低层 `T.rvt_*`
直接对应 PPL 1.7 RV Tensor ABI，不能与高层语义调用混在同一个 kernel 中。

## 项目结构

- [`tilelang/engine/`](./tilelang/engine/)：目标配置、编译流程和 TPU Pass
- [`tilelang/language/`](./tilelang/language/)：TileLang TPU 前端接口
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/)：PPL 1.7 工具链、JIT 和 profiling
- [`src/target/`](./src/target/)：TPU 源码生成与运行时模块
- [`src/transform/`](./src/transform/)：TPU 地址分配等变换
- [`tpu_demo/`](./tpu_demo/)：高层算子示例
- [`testing/python/jit/`](./testing/python/jit/)：编译器、算子和运行时测试
- [`research/`](./research/)：设计、能力契约和实验报告

## 开发检查

修改 C++ 后先重新构建：

```bash
cmake --build build-tpu --parallel 10
```

提交前运行格式化与 TPU 相关测试。下面四个文件要求 CUDA 或 HIP，TPU-only 环境应跳过：

```bash
./format.sh
python -m pytest -q testing/python/jit \
  testing/python/transform/test_tilelang_transform_address_assign.py \
  --ignore=testing/python/jit/test_tilelang_jit_callback.py \
  --ignore=testing/python/jit/test_tilelang_jit_gemm.py \
  --ignore=testing/python/jit/test_tilelang_jit_gemm_ctypes.py \
  --ignore=testing/python/jit/test_tilelang_jit_gemm_cython.py
```

PCIe 测试必须使用 `testing/python/jit/tpu_demo_ops_matrix.py` 串行执行。该工具会检查
CModel 前置结果、独占设备并在首个错误后停止；不要直接设置内部板卡放行环境变量。

## 致谢

本项目基于 [TileLang](https://github.com/tile-ai/tilelang)，并使用
[SOPHGO PPL](https://github.com/sophgo/PPL) 提供的 TPU 编译与运行组件。
