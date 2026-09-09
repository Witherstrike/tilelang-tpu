# 安装 TileLang-TPU

本文说明如何从源码配置 TileLang-TPU。当前 TPU 后端没有单独发布 PyPI wheel，因此推荐
使用项目自带的 TVM 子模块和 CMake 构建原生库，再直接从源码目录导入 Python 包。

## 1. 适用范围

当前安装流程面向 Linux x86_64，已在以下环境验证：

- Ubuntu 22.04；
- Python 3.10；
- CMake 3.26 或更高版本；
- PPL 1.7.122 发布包；
- BM1690 CModel，以及 SG2260E CModel 和 PCIe。

项目元数据允许 Python 3.8 及以上版本，但提交前的完整验证环境是 Python 3.10。使用其他
Python 或 PPL 1.7 小版本时，应重新执行本文的工具链检查和 CModel 测试。

仅运行 CModel 不需要物理板卡。SG2260E PCIe 还需要已安装的 TPUv7 驱动运行库，并且
必须先完成 BM1690 和 SG2260E 的 CModel 测试。

## 2. 安装系统依赖

Ubuntu/Debian 可使用：

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential python3 python3-dev python3-venv \
  libtinfo-dev zlib1g-dev libedit-dev libxml2-dev
```

CMake 由后面的 Python 虚拟环境安装，这样可以稳定满足项目要求的最低版本。

## 3. 获取源码

新建工作目录时：

```bash
git clone https://github.com/xwhzz/tilelang-tpu.git
cd tilelang-tpu
git submodule update --init --recursive 3rdparty/tvm
```

已经有仓库时，只需在仓库根目录执行最后一条命令。TPU 构建依赖项目固定的 TVM 版本；
不要用系统中的其他 TVM 替换它。顶层 CUTLASS 和 Composable Kernel 子模块属于 GPU
后端，TPU-only 构建不需要初始化。

## 4. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt "cmake>=3.26"
```

需要运行测试和格式检查时，再安装开发依赖：

```bash
python -m pip install -r requirements-test.txt
```

不要在 TPU-only 环境中执行 `pip install -e .`。当前 `setup.py` 仍沿用上游 GPU 打包
流程，会在配置阶段检查 CUDA；这与本文的 TPU 源码构建方式不是同一条安装路径。

## 5. 准备 PPL 1.7 SDK

从 SOPHGO 提供的 PPL 1.7 发布包中取得完整开发 SDK，并解压到仓库外。`PPL_PROJECT_ROOT`
必须指向包含 `deps/` 的顶层目录，例如：

```bash
export PPL_PROJECT_ROOT=/absolute/path/to/ppl-1.7-sdk
test -f "${PPL_PROJECT_ROOT}/deps/chip/chip_map.json"
```

项目只支持 PPL 1.7 的 `deps/` 发布布局。它会根据 `chip_map.json` 分别选择：

- BM1690：`tpub_7_1`；
- SG2260E：`tpub_7_1_e`。

不要加载 PPL 自带的全局环境脚本，也不要设置 `PPL_KERNEL_PATH` 或 `TPU_KERNEL_PATH`。
TileLang-TPU 会为每次 JIT 编译创建独立目录，并使用与目标芯片匹配的头文件和库。

## 6. 构建原生库

在仓库根目录运行：

```bash
./build_tpu.sh
```

脚本会在独立的 `build-tpu/` 中配置并构建 TileLang 与 TVM，不会修改 TVM 子模块，
也不会安装 Python 包。独立目录可避免复用 GPU 构建留下的 CMake 配置。TileLang 会
自动查找 `build-tpu/libtilelang_module.so` 和 `build-tpu/tvm/libtvm.so`。

随后配置当前 shell：

```bash
export TILELANG_TPU_SOURCE="$(pwd)"
export PYTHONPATH="${TILELANG_TPU_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export PPL_PROJECT_ROOT=/absolute/path/to/ppl-1.7-sdk
```

标准 `build-tpu/` 布局不需要手工设置 `LD_LIBRARY_PATH`、`TVM_LIBRARY_PATH` 或
`TILELANG_LIBRARY_PATH`。如果自行使用非标准构建目录，需要分别设置后两个变量指向
TileLang 构建目录和其中的 `tvm/` 目录。显式设置的 `TILELANG_LIBRARY_PATH` 会作为唯一的
TileLang 原生库搜索路径；路径错误时会直接报错，不会退回仓库中的其他构建。

## 7. 检查安装

先确认 Python 能加载刚构建的原生库：

```bash
python -c 'import tilelang; print(tilelang.__version__)'
```

再检查 PPL 目录中是否包含两种芯片的 CModel 文件和 SG2260E RV Tensor 头文件：

```bash
python - <<'PY'
import os

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

root = os.environ["PPL_PROJECT_ROOT"]
for chip in ("bm1690", "sg2260e"):
    layout = resolve_ppl_layout(root, chip)
    layout.require_runtime("cmodel")
    if chip == "sg2260e":
        layout.require_rvt_api()
    print(f"{chip}: {layout.arch}, CModel ready")
PY
```

最后查看算子列表并运行两个 SG2260E CModel 用例：

```bash
python testing/python/jit/tpu_demo_ops_matrix.py --list-cases

python -m tpu_demo.run \
  --case elementwise-add.float16 \
  --chip sg2260e \
  --programming-model tpukernel \
  --runtime-mode cmodel

python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

两条命令都输出 `"status": "passed"`，才说明 Python、原生库、PPL SDK 和 CModel
运行链路已经接通。

## 8. 配置 SG2260E PCIe

PCIe 除了上述内容，还需要：

1. PPL SDK 中能够唯一找到一个 `riscv64-unknown-linux-gnu-gcc` 交叉编译器；
2. SDK 包含 SG2260E 对应的 firmware；
3. 需要采集 profiling 时，SDK 还应包含 TPUDNN 头文件和库；
4. 主机已经安装 TPUv7 板卡驱动和运行库；
5. `tpu-smi` 能看到唯一的目标板卡。

默认板端运行库目录是 `/opt/tpuv7/tpuv7-current/lib`。只有驱动安装在其他位置时才设置：

```bash
export TILELANG_TPU_PCIE_RUNTIME_PATH=/absolute/path/to/board-runtime/lib
```

SDK 中的 `deps/runtime/tpuv7-runtime/lib` 是 CModel 运行库，不能用于 PCIe。下面的检查
只核对文件和工具链，不会访问板卡：

```bash
python - <<'PY'
import os

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

layout = resolve_ppl_layout(os.environ["PPL_PROJECT_ROOT"], "sg2260e")
layout.require_profiling("pcie")
print("SG2260E PCIe and profiling toolchain ready")
PY
```

实际板卡测试请严格按照
[`tpu_demo/README.md`](../../tpu_demo/README.md) 的三阶段命令执行。不要手工设置
`TILELANG_TPU_ALLOW_PCIE_LOAD`、设备编号或 profiling 内部变量；测试工具会在持有设备锁时
统一设置，并保证同一时刻只运行一个任务。

## 9. 更新和重新构建

拉取代码后先同步固定的 TVM 子模块，再重新构建：

```bash
git pull --ff-only
git submodule update --init --recursive 3rdparty/tvm
./build_tpu.sh
```

只修改了 C++ 源码时，也可以直接执行：

```bash
cmake --build build-tpu --parallel 10
```

## 10. 常见问题

### 找不到 CMake

确认虚拟环境已经激活，并运行 `cmake --version`。若版本低于 3.26，重新执行第 4 节的
安装命令。

### 找不到 `libtilelang_module.so` 或 `libtvm.so`

确认 `./build_tpu.sh` 已成功结束，并从仓库根目录运行命令。若从其他目录运行，检查
`PYTHONPATH` 是否包含仓库绝对路径。

### 提示 PPL 1.7 布局不完整

确认 `PPL_PROJECT_ROOT` 指向 SDK 顶层，而不是 `deps/` 本身。不要把不同 PPL 版本的头文件
和库拼在同一个目录中。

### TPU-only 环境提示缺少 CUDA

这通常说明执行了 `pip install .` 或 `pip install -e .`。退出该流程，按本文第 4 至第 7 节
使用源码目录和 `build-tpu/` 中的原生库。

### PCIe 误用了 CModel 运行库

不要把 `${PPL_PROJECT_ROOT}/deps/runtime/tpuv7-runtime/lib` 加到 PCIe 进程的
`LD_LIBRARY_PATH`。使用系统安装的板端运行库，或通过
`TILELANG_TPU_PCIE_RUNTIME_PATH` 指向它。
