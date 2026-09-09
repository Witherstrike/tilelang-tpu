# High-Level TPU Operator Design

## Scope

The `tpu_demo` package provides reusable TileLang implementations and PyTorch
references for these operators:

- Elementwise add, subtract, multiply, and divide
- Matrix multiplication
- RMSNorm and split-K RMSNorm
- Rotary position embedding (RoPE)
- SwiGLU
- FlashAttention

Each operator accepts FP16, BF16, and FP32 inputs. The same frontend expression
is used for every target. Target resolution selects TPU-Kernel or RV Tensor
during lowering; it never falls back to another programming model at runtime.

## Package structure

```text
tpu_demo/cases.py                 case registry
tpu_demo/run.py                   command-line dispatcher
tpu_demo/<operator>/              TileLang builder and PyTorch reference
tpu_demo/common.py                target selection, execution, and comparison
testing/python/jit/               unit and matrix tests
```

Importing an operator module does not compile code or load a device runtime.
Small programs that test individual intrinsics belong in `testing/`, while
`tpu_demo/` contains complete operator examples.

## Backend coverage

| Operator | Variant | TPU-Kernel cases | RV Tensor cases |
| --- | --- | ---: | ---: |
| Elementwise | Add, subtract, multiply, divide | 12 | 12 |
| Matmul | Tiled K accumulation | 3 | 3 |
| RMSNorm | Standard | 3 | 0 |
| RMSNorm | Split-K | 3 | 0 |
| RoPE | Interleaved even and odd lanes | 3 | 0 |
| SwiGLU | Sigmoid composite | 3 | 0 |
| FlashAttention | Three input distributions | 9 | 0 |
| **Total** | | **36** | **15** |

The case counts include one case for each supported public data type. BM1690
uses TPU-Kernel. SG2260E supports both columns. Composite operators remain
TPU-Kernel-only because their reduction and math primitives have not yet been
implemented for RV Tensor.

## Implementation details

### Elementwise operations

Each program copies a tile to local memory, performs one arithmetic operation,
and copies the result back. Division tests use positive denominators away from
zero. Division by zero and exceptional-value propagation are outside the
current contract.

### Matrix multiplication

Matmul accumulates K tiles in an FP32 local buffer. FP16 and BF16 inputs are
used directly. The FP32 interface converts matrix operands to BF16 before the
matrix engine and keeps FP32 accumulation and output. The PyTorch reference
applies the same conversion boundary. This path must not be described as
native FP32 matrix multiplication.

### RMSNorm and split-K RMSNorm

RMSNorm computes

```text
y = x * rsqrt(mean(x * x) + epsilon)
```

The split-K implementation accumulates the sum of squares across K tiles
before normalizing each tile. Low-precision inputs use FP32 intermediate
values for the square, reduction, and reciprocal square root.

### RoPE and SwiGLU

RoPE rotates interleaved even and odd lanes. SwiGLU computes
`gate * sigmoid(gate) * up`. Low-precision SwiGLU inputs use FP32 intermediate
values.

### FlashAttention

FlashAttention uses BSHD layout and online softmax across K/V tiles. It keeps
the running row maximum and rescales the previous sum and accumulator before
adding a new tile. Three input distributions cover balanced values, a lower
maximum in a later tile, and nonuniform attention weights.

The FP32 interface converts Q, K, and V to BF16 before matrix operations while
keeping softmax state in FP32. Only `is_causal=False` is supported. Causal
masking is rejected until diagonal-tile masking is implemented.

## Shape rules

Builders require positive static dimensions and tile-aligned extents. The
current implementation has no masked DMA or tail predicate, so it rejects
non-divisible shapes instead of rounding them up and risking an out-of-bounds
access.

## Numerical checks

Every case checks output shape, data type, finite values, and elementwise error
against its PyTorch reference.

| Operator | FP16 `atol/rtol` | BF16 `atol/rtol` | FP32 `atol/rtol` |
| --- | ---: | ---: | ---: |
| Elementwise add/subtract/multiply | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 |
| Elementwise divide | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-5 / 1e-5 |
| Matmul | 1e-2 / 1e-2 | 2e-2 / 2e-2 | 1e-2 / 1e-2 |
| RMSNorm and SwiGLU | 1e-2 / 1e-2 | 3e-2 / 3e-2 | 1e-2 / 1e-2 |
| RoPE | 5e-3 / 5e-3 | 2e-2 / 2e-2 | 1e-5 / 1e-5 |
| FlashAttention | 2e-2 / 2e-2 | 2e-2 / 2e-2 | 2e-2 / 2e-2 |

NaN or infinity in an output is always a failure. Tolerances are fixed
acceptance limits and are not adjusted after a failed run.

## Validation

The operator matrix is run in this order:

1. BM1690 CModel with TPU-Kernel
2. SG2260E CModel with TPU-Kernel and RV Tensor
3. SG2260E PCIe with the same SG2260E cases

PCIe execution starts only after both CModel stages pass. Each PCIe case runs
in a separate process, uses a bounded timeout, and stops the remaining matrix
on the first error. See the [test report](../tpu-backend-design/test-report.md)
for the accepted results and profiling evidence.

## Next steps

- Add masked tails and dynamic-shape guards.
- Add causal masking to FlashAttention.
- Implement RV Tensor reductions and math functions before enabling composite
  operators on that backend.
- Define per-core partitions before using all SG2260E cores concurrently.
- Keep diagnostic profiling separate from repeat-based performance benchmarks.
