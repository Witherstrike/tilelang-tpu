# TileLang TPU 算子与测试

本目录保存面向用户的高层算子示例。每个算子模块包含两部分：`build_*` 函数构造
TileLang 程序，`run` 函数生成输入、编译执行，并与 PyTorch 参考结果比较。低层指令测试和
编译器回归测试统一放在 `testing/python/jit/`。

## 算子范围

| 算子 | 默认验证形状 | 用例数 | TPU-Kernel | RV Tensor |
| --- | --- | ---: | --- | --- |
| elementwise add/sub/mul/div | `[4, 32]` | 12 | 支持 | 支持 |
| matmul | `32×32 @ 32×32`，tile 16 | 3 | 支持 | 支持 |
| RMSNorm | `[8, 64]` | 3 | 支持 | 暂不支持 |
| split-K RMSNorm | `[8, 128]`，特征维度按 32 分块 | 3 | 支持 | 暂不支持 |
| RoPE | `[8, 32]`，相邻元素配对 | 3 | 支持 | 暂不支持 |
| SwiGLU | `[8, 32]` | 3 | 支持 | 暂不支持 |
| FlashAttention | BSHD `[1, 32, 1, 16]` | 9 | 支持 | 暂不支持 |

每类算子都覆盖 `float16`、`bfloat16` 和 `float32`。FlashAttention 每种数据类型还包含
`balanced`、`descending-max` 和 `weighted-keys` 三种输入，因此 TPU-Kernel 共 36 个
语义用例。RV Tensor 当前覆盖四种 elementwise 操作和 matmul，共 15 个用例。

表中的“支持”只表示默认用例已接入并通过当前能力契约，不表示任意形状都能运行。

split-K RMSNorm 在特征维度上分块，按顺序执行两遍计算：第一遍合并平方和，第二遍完成
归一化。这里的 split-K 不是 GEMM 中的并行 split-K。

## 使用算子构造函数

各子包公开以下构造函数：

- `tpu_demo.elementwise.build_elementwise`
- `tpu_demo.matmul.build_matmul`
- `tpu_demo.rmsnorm.build_rmsnorm`、`build_rmsnorm_splitk`
- `tpu_demo.rope.build_rope`
- `tpu_demo.swiglu.build_swiglu`
- `tpu_demo.flashattn.build_flashattn`

例如：

```python
from tpu_demo.matmul import build_matmul

program = build_matmul(
    m=32,
    n=32,
    k=32,
    block_m=16,
    block_n=16,
    block_k=16,
    dtype="float16",
)
```

构造函数只建立 TileLang 程序。统一的数值验证入口是 `tpu_demo.run.run_case`，命令行形式
为 `python -m tpu_demo.run`。`tpu_demo.cases` 是轻量注册表，可以在不加载 TPU 运行库的
情况下读取用例列表；编译和运行才会使用 PPL CModel 或访问设备。

## 查看和运行单个用例

以下命令均在已经配置好的仓库根目录运行。查看全部用例：

```bash
python testing/python/jit/tpu_demo_ops_matrix.py --list-cases
```

只查看 RV Tensor 可用的用例：

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --list-cases \
  --programming-model rv
```

运行一个 SG2260E RV Tensor CModel 用例：

```bash
python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

`tpu_demo.run` 只接受 CModel。真实板卡必须使用后面的批量测试工具，避免绕过设备锁、
CModel 前置检查和首错停止规则。

## 数值规则和已知限制

每个用例只执行一次 kernel，并与相同 shape、dtype 的 PyTorch 结果比较。比较顺序是：

1. 检查 shape 和 dtype 完全一致；
2. 检查结果全部为有限值；
3. 使用 `tpu_demo/common.py` 中按算子和 dtype 统一定义的 `atol/rtol`。

当前示例还有以下边界：

- 所有维度必须是编译期正整数，并能被相应 tile 整除；尚未实现 tail 路径。
- matmul 和 FlashAttention 的 FP32 输入会先把矩阵乘法操作数转换为 BF16，再使用 FP32
  累加；参考实现会复现这个输入转换边界。
- RMSNorm 和 SwiGLU 对 FP16/BF16 输入使用 FP32 中间计算，最后转换回输入 dtype。
- FlashAttention 当前只支持 `is_causal=False`。非布尔值或 `True` 会明确报错。
- `descending-max` 和 `weighted-keys` 用来发现 online softmax 跨 tile 合并或权重处理错误，
  不是额外公开的算子模式。

## 单元测试

修改示例、注册表或批量运行工具后，至少运行：

```bash
python -m pytest -q \
  testing/python/jit/test_tpu_demo_contract.py \
  testing/python/jit/test_tpu_demo_ops_matrix.py
```

相关文件的职责如下：

- `test_tpu_demo_contract.py`：注册表、shape、dtype、数值比较和非法组合。
- `test_tpu_demo_ops_matrix.py`：批量测试、PCIe 安全检查和阶段衔接规则。
- `tpu_demo_ops_matrix.py`：CModel/PCIe 端到端测试入口。
- `research/tpu-demo-ops/README.md`：完整实验结果和误差分析。

## 完整验证矩阵

完整验证必须从同一份干净且已经提交的源码开始，并按以下顺序分别执行：

1. BM1690 CModel：36 个 TPU-Kernel 用例；
2. SG2260E CModel：36 个 TPU-Kernel 用例和 15 个 RV Tensor 用例，共 51 个；
3. SG2260E PCIe：只运行前两阶段已有通过结果的同一组用例，最多 51 个。

每次使用新的输出目录：

```bash
RUN_ROOT=research/artifacts/demo-validation-01

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip bm1690 \
  --programming-model tpukernel \
  --output-dir "${RUN_ROOT}/bm1690-cmodel"

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip sg2260e \
  --output-dir "${RUN_ROOT}/sg2260e-cmodel"
```

第二条命令不指定 `--programming-model`，因此会依次运行 SG2260E TPU-Kernel 和 RV
Tensor 中适用的用例。两次 CModel 都完整通过后，才能运行 PCIe：

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode pcie \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --bm-cmodel-summary "${RUN_ROOT}/bm1690-cmodel/summary.json" \
  --sg-cmodel-summary "${RUN_ROOT}/sg2260e-cmodel/summary.json" \
  --output-dir "${RUN_ROOT}/sg2260e-pcie"
```

PCIe 测试独占一张板卡并串行执行，任一编译、运行、数值或板卡状态错误都会停止后续
用例。普通正确性测试会保存原始 profiling 数据，但不要求安装指令解码器。只有确实需要
逐指令耗时时，才额外传入：

```bash
--require-decoded-timing \
--pcie-decoder-python <decoder-python> \
--pcie-decoder-pythonpath <decoder-path>
```

`summary.json` 中的数值结果、原始指令记录和解码耗时是三类不同证据。单次 profiling
适合核对指令映射和定位问题，不能当作稳定性能基准。完整能力边界以
[`research/tpu-op-contract/contract.json`](../research/tpu-op-contract/contract.json) 为准。
