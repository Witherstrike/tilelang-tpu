# TileLang-TPU Backend Design

This document describes the compiler architecture for BM1690 and SG2260E,
including target selection, pass boundaries, operation semantics, instruction
selection, memory effects, and runtime integration.

## 1. Design overview

The backend separates three independent choices:

| Axis | Values | Controls |
| --- | --- | --- |
| Chip | `bm1690`, `sg2260e` | PPL architecture, compile definitions, physical cores, and chip features |
| Programming model | `tpukernel`, `rv` | Device ABI, descriptors, instructions, and kernel lifecycle |
| Runtime mode | `cmodel`, `pcie` | Host runtime, linking, loading, and device access |

The chip and programming model form the compile target. The runtime mode only
selects how the compiled device program runs.

```python
kernel = tilelang.compile(
    program,
    target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel",
    runtime_mode="cmodel",
)
```

The valid compile targets are:

- BM1690 with TPU-Kernel
- SG2260E with TPU-Kernel
- SG2260E with RV Tensor

BM1690 and SG2260E share the TPUv7 local-memory geometry used by the allocator.
They differ in PPL architecture, compile definitions, physical core count, and
RV availability. BM1690 has eight physical cores; SG2260E has four. Current
kernels launch on one logical core, so the topology model does not yet provide
multi-core partitioning.

## 2. Target configuration

The read-only `TPU_CHIP_SPECS` table in `tilelang/engine/tpu_config.py` is the
Python source of chip capabilities:

- BM1690 maps to `tpub_7_1`, has eight cores, and accepts `tpukernel`.
- SG2260E maps to `tpub_7_1_e`, has four cores, and accepts `tpukernel` and
  `rv`.

`TPUTargetSpec(chip, programming_model)` is created from a complete TVM target.
`TPURuntimeConfig(runtime_mode)` is created separately and defaults to CModel.
The PPL resolver, JIT adapter, and profiler consume these objects instead of
inferring configuration from directories or environment variables.

Compilation rejects a bare `target="tpu"`, a missing target option, an unknown
value, and BM1690 with RV Tensor. The native build entry repeats the target
check so direct FFI callers cannot bypass the Python boundary.

## 3. Compilation pipeline

```text
TileLang frontend
  |
  +-- T.ppl_{copy,fill,gemm,add,subtract,mul,div,max}
  |      `-- tl.tpu.*                    portable TPU semantics
  |
  `-- T.ppl_{scalar,exp,...,rope}
         `-- tl.tpukernel.*              TPU-Kernel-only semantics
                    |
                    v
          target and module checks
                    |
          BindTarget and frontend lowering
                    |
          conservative TPU pass pipeline
                    |
          second module check
                    |
          AddressAssign for TPUv7 LMEM
                    |
          target.build.tilelang_tpu
                    |
          codegen_tpu.{h,cc}
          +-- codegen_tpukernel.cc --> tpu_kernel.h
          `-- codegen_rv.cc        --> rvt_api.h
                    |
              PPL 1.7 build
          +-- CModel runtime
          `-- installed PCIe runtime
```

`target.build.tilelang_tpu` is the only TPU build entry. A TPU module contains
one `PrimFunc`, and reserved host entry names cannot be used as device kernel
names.

The source files follow standard TVM target-codegen naming:

- `codegen_tpu.{h,cc}` implements the shared TPU source generator, semantic
  parsing, and programming-model dispatch.
- `codegen_tpukernel.cc` implements TPU-Kernel instruction selection.
- `codegen_rv.cc` implements RV Tensor instruction selection.

The two instruction files are parts of one source generator, not separate
compiler backends. Shared checks therefore remain in one place without hiding
programming-model-specific ABI details.

### Pass policy

The TPU pipeline uses transformations whose residual IR and memory effects are
defined for the TPU source generator: target binding, frontend lowering,
simplification, conditional binding, allocation placement, conditional merge,
opaque-block lowering, narrowing, unrolling, and final simplification.

The following generic GPU transformations remain disabled:

- Vector legalization and vectorization, because residual vector lanes,
  `Ramp`, and vector loads and stores do not yet have complete TPU semantics
- Software-pipeline planning and injection, because DMA/compute tokens, buffer
  versions, and hazards are not modeled
- `StorageRewrite`, because it removes the structured `DeclBuffer` and
  `Allocate` relationship required by TPU code generation

The module contract is checked before and after the target passes. It rejects
residual vector IR, GPU barriers, unknown extern calls, mixed programming
models, unsupported loop kinds, direct buffer loads and stores, conditional
allocations, unconsumed attributes, prefetch nodes, and raw custom source.

The only structured `BufferLoad` and `Ramp` exception is the contiguous region
marker used by `tl.tpu.copy`. Its ramp must have unit stride and a lane count
equal to the explicit region extent.

## 4. Semantic ABI

| Layer | Namespace | Responsibility |
| --- | --- | --- |
| Public frontend | `T.ppl_*` | User-facing TileLang operations and static shape/type checks |
| Portable TPU semantics | `tl.tpu.*` | Operations that can select TPU-Kernel or RV Tensor |
| TPU-Kernel semantics | `tl.tpukernel.*` | Operations available only through TPU-Kernel |
| Low-level RV ABI | isolated `rvt_*` calls | User-managed descriptors and lifecycle |

The legacy `ppl.*` and raw `tpu_*` TIR extern interfaces are not accepted. Raw
RV calls remain available for expert kernels but cannot be mixed with
compiler-managed TPU operations.

Every typed TPU buffer operand crosses the semantic boundary as
`tl.region(BufferLoad, access_mask, logical_extents)`. Non-copy operations
require a zero-based region covering the complete logical buffer. Copy accepts
an explicit subregion after proving its bounds and continuity.

Native code generation checks the buffer's data variable, type, original rank,
normalized shape, scope, and allocation owner against the compiler-owned
descriptor. A presentation alias is allowed only when it leaves the descriptor
unchanged and does not introduce another allocation owner.

Local TPU scopes are `shared`, `shared.dyn`, `local`, and `local.fragment`.
Every normalized N/C/H/W extent must be a compile-time integer in
`[1, 65535]`. Operation-specific derived dimensions, such as aligned reduction
widths, must fit the same descriptor fields.

## 5. Portable operation mapping

| Semantic operation | TPU-Kernel | RV Tensor | Main contract |
| --- | --- | --- | --- |
| `tl.tpu.copy` | GDMA S2L/L2S/S2S and BDC L2L/cast | RV DMA load/store/copy and limited f2f cast | Equal static extents; cross-type conversion is local only |
| `tl.tpu.fill` | `tpu_bdc_set_C` | Typed CR followed by `rvt_cp` | RV accepts zero only |
| `tl.tpu.gemm` | FP and FP8 matrix families | `rvt_fmm2[a]_{nn,nt}` | Rank-2 local tensors; no transpose-A; explicit overwrite or accumulation |
| `tl.tpu.add/sub/mul/div/max` | `tpu_bdc_fp_*` and `tpu_bdc_max` | `rvt_f*` | Matching local types; equal shape or right-hand W broadcast |

GEMM's `accumulate` attribute controls both numerical behavior and memory
effects. The output is write-only in overwrite mode and read-write in
accumulation mode. Accumulation requires FP32 output. Non-FP8 overwrite accepts
an output matching the inputs or FP32. FP8 GEMM uses matching FP8 inputs and an
FP32 output.

RV GEMM currently accepts FP16 or BF16 inputs. RV elementwise operations accept
FP16, BF16, and FP32. RV W broadcast uses a zero-stride free-layout descriptor,
not a local-memory expansion.

Unsupported types, shapes, layouts, aliases, tails, and attributes produce a
compile-time error. Code generation does not fall back to another programming
model.

## 6. TPU-Kernel-only operations

| Operation | Implementation | Data types |
| --- | --- | --- |
| Scalar add/multiply | FP32 constant cast followed by `tpu_bdc_fp_add_C` or `tpu_bdc_fp_mul_C` | FP16, BF16, FP32, E4M3, E5M2 |
| Exp | Coefficient load and `tpu_bdc_fp_exp` | FP16, BF16, FP32 |
| Sigmoid | Negation, exp, reciprocal, and add composite | FP16, BF16, FP32 |
| Reduce sum/max | Padding and two-stage pooling composite on axis 1 | FP16, BF16, FP32 |
| Rsqrt | `tpu_bdc_fp_rsqrt` | FP16, BF16, FP32 |
| Gather | `tpu_gdma_h_gather_S2S` with UINT32 index | FP16, BF16, FP32, E4M3, E5M2 payload |
| Top-k | `tpu_hau_sort_natural_index` | BM1690 FP32, INT32, and UINT32 |
| RoPE primitive | Two interleaved floating-point add operations | FP16, BF16, FP32, E4M3, E5M2 |

Top-k is available on BM1690 and rejected for SG2260E. It writes K values and
K indices; equal values retain ascending source-index order.

The high-level RMSNorm, split-K RMSNorm, RoPE, SwiGLU, and FlashAttention
examples depend on these TPU-Kernel-only operations. Elementwise arithmetic and
matmul use portable semantics and can select RV Tensor on SG2260E.

## 7. FP8 contract

TPU-Kernel exposes E4M3 and E5M2 only for registered operation and attribute
combinations. The current contract includes:

- Same-format local and S2S copy
- Zero fill and conversion to and from FP32
- Equal-shape and W-broadcast add, subtract, multiply, and max
- Scalar add and multiply with the default non-saturating behavior
- Gather with UINT32 indices
- Interleaved even/odd RoPE add
- NN and NT GEMM variants with FP32 output

FP8 divide, nonzero fill, other conversions, FP8 GEMM output, and FP8
exp/sigmoid/rsqrt/reduction/top-k are not part of the contract. RV FP8 mappings
are also disabled.

FP8 scalar operations first convert the FP32 constant with
`RM_HALF_TO_EVEN`, then call the standard same-format scalar instruction. The
current non-saturating behavior produces NaN on E4M3 overflow and infinity on
E5M2 overflow; the public API does not expose a saturation option.

## 8. Address assignment and effects

`AddressAssign` uses a closed operation registry rather than guessing effects
from function names:

- Copy reads its source and writes its destination.
- Fill writes its destination.
- GEMM reads A and B; C is written or read-written according to `accumulate`.
- Binary, scalar, rsqrt, and RoPE operations write outputs and read inputs.
- Reduction workspaces are modeled conservatively as read-write.
- Exp and sigmoid workspaces retain conservative conflict relationships.
- Gather and top-k assign explicit value and index roles.

Loop liveness includes the back edge. A local buffer allocated outside a
repeating `For` or `While` remains live across the loop when used inside it.
Static zero- or one-trip loops have no back edge, and loop-local allocations
keep their local lifetime.

Unknown semantic operations never receive optimistic effects. Incomplete
shape, type, scope, layout, target, or lifecycle information stops compilation
at the first stage that can identify the error.

## 9. PPL, runtime, and profiling

The toolchain accepts the PPL 1.7 `deps/` release layout. Base compilation,
CModel, PCIe, RV headers, and profiling dependencies are checked separately.
CModel uses the SDK runtime. PCIe uses the installed TPUv7 board runtime and
the SDK's headers, chip libraries, cross-compiler, and firmware.

Profiling reuses PPL and TPUDNN recording formats without running the PPL
autotuning pipeline. CModel captures raw command records and can use a
configured PerfAI decoder. PCIe wraps one launch in a TPUDNN recorder and can
decode it through an explicitly configured `bigTpuProfile` environment.

Numerical checks and profiling are independent. A recorded single launch is
appropriate for instruction inspection, not for throughput or latency
benchmarking. The detailed design is in
[`../ppl-profiling/README.md`](../ppl-profiling/README.md).

## 10. Extension rules

Each new operation or target capability must define:

1. Public semantics and a typed TIR representation
2. Operand roles, memory effects, alias rules, and workspace ownership
3. Supported chip, programming model, type, shape, layout, and attributes
4. Instruction selection and synchronization requirements
5. Positive and negative source tests
6. CModel numerical cases before PCIe cases
7. A machine-readable contract entry with exact evidence references

Raw SDK symbols or successful compilation alone do not establish an operation
contract. See [`test-report.md`](test-report.md) for runtime results and
[`../tpu-op-contract/README.md`](../tpu-op-contract/README.md) for the contract
format.
