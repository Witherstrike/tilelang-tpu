# TPU Operator Examples

This directory contains user-facing TileLang examples for SOPHGO TPUs. Each
operator module provides a `build_*` function that constructs a TileLang
program and a `run` function that compares its output with a PyTorch reference.
Low-level instruction and compiler regression tests live under
`testing/python/jit/`.

## Available operators

| Operator | Default workload | TPU-Kernel | RV Tensor |
| --- | --- | --- | --- |
| Elementwise | Add, subtract, multiply, and divide on `[4, 32]` | Yes | Yes |
| Matmul | `32 x 32` by `32 x 32`, using `16 x 16 x 16` tiles | Yes | Yes |
| RMSNorm | `[8, 64]` | Yes | Yes |
| Split-K RMSNorm | `[8, 128]`, split into width-32 tiles | Yes | Yes |
| RoPE | `[8, 32]`, with interleaved even and odd elements | Yes | No |
| SwiGLU | `[8, 32]` | Yes | Yes |
| FlashAttention | BSHD `[1, 32, 1, 16]` | Yes | No |

The RV normalization and SwiGLU examples use FP32 intermediates, including
exp/sigmoid. Their FP16/BF16 inputs and outputs are explicitly converted.
See [the RV validation results](../docs/validation/sg2260e-rv-essential-results.json)
for the recorded numerical cases.

Every example supports `float16`, `bfloat16`, and `float32`. The split-K
RMSNorm example splits the feature dimension, accumulates the sum of squares,
and then normalizes each tile. It is unrelated to parallel split-K GEMM.

## Builder functions

The operator packages export these builders:

- `tpu_demo.elementwise.build_elementwise`
- `tpu_demo.matmul.build_matmul`
- `tpu_demo.rmsnorm.build_rmsnorm`
- `tpu_demo.rmsnorm.build_rmsnorm_splitk`
- `tpu_demo.rope.build_rope`
- `tpu_demo.swiglu.build_swiglu`
- `tpu_demo.flashattn.build_flashattn`

For example:

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

Builders only create TileLang programs. `tpu_demo.run.run_case` handles input
generation, compilation, execution, and comparison. The `tpu_demo.cases`
registry can be imported without loading a TPU runtime.

## List and run examples

Run these commands from the repository root after completing the project
installation.

List all cases:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py --list-cases
```

List the cases available for RV Tensor:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --list-cases \
  --programming-model rv
```

Run one SG2260E RV Tensor case with CModel:

```bash
python -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

`python -m tpu_demo.run` runs CModel cases only. Use the matrix runner for PCIe
execution so that all device access remains serial.

## Numerical comparison

Each case runs one kernel and compares it with a reference that uses the same
shape and public data type. The comparison checks:

1. Output shape and data type
2. Finite output values
3. Elementwise error using the operator-specific `atol` and `rtol` values in
   `tpu_demo/common.py`

The examples have the following limits:

- Dimensions must be positive compile-time integers.
- Tiled dimensions must divide evenly; tail handling is not implemented.
- FP32 matmul and FlashAttention convert matrix operands to BF16 and accumulate
  in FP32. Their reference functions apply the same conversion.
- RMSNorm and SwiGLU use FP32 intermediate values for FP16 and BF16 inputs.
- FlashAttention accepts `is_causal=False`. Causal masking is not implemented.
- The FlashAttention input variants `descending-max` and `weighted-keys` test
  online-softmax state updates; they are not separate operator modes.

## Unit tests

Run the registry and runner tests after changing an example or its test
infrastructure:

```bash
python -m pytest -q \
  testing/python/jit/test_tpu_demo_contract.py \
  testing/python/jit/test_tpu_demo_ops_matrix.py
```

The files have distinct roles:

- `test_tpu_demo_contract.py` checks the case registry, shapes, data types,
  comparison rules, and invalid combinations.
- `test_tpu_demo_ops_matrix.py` checks matrix selection, stage transitions,
  and PCIe access rules.
- `tpu_demo_ops_matrix.py` runs end-to-end CModel and PCIe cases.

## CModel and PCIe matrix

Use a new output directory for each run. Execute BM1690 CModel first, then
SG2260E CModel:

```bash
export TILELANG_TPU_DEMO_RUN="research/artifacts/demo-run"

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip bm1690 \
  --programming-model tpukernel \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/bm1690-cmodel"

python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode cmodel \
  --chip sg2260e \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/sg2260e-cmodel"
```

The SG2260E command selects TPU-Kernel and all applicable RV Tensor cases. Run
the PCIe stage only after both CModel commands finish successfully:

```bash
python testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode pcie \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --bm-cmodel-summary "${TILELANG_TPU_DEMO_RUN}/bm1690-cmodel/summary.json" \
  --sg-cmodel-summary "${TILELANG_TPU_DEMO_RUN}/sg2260e-cmodel/summary.json" \
  --output-dir "${TILELANG_TPU_DEMO_RUN}/sg2260e-pcie"
```

The PCIe runner holds an exclusive device lock and stops after the first
compile, runtime, numerical, profiling, or device-health error. It records raw
profiling data without requiring an instruction decoder. Add the following
options when decoded per-instruction timing is required:

```bash
--require-decoded-timing \
--pcie-decoder-python /path/to/decoder/python \
--pcie-decoder-pythonpath /path/to/decoder/package
```

Numerical results, raw instruction records, and decoded timing are separate
fields in `summary.json`. A single profiled execution is useful for inspecting
instruction selection, but it is not a performance benchmark.
