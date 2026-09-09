# RV Tensor Backend

This document describes the SG2260E RV Tensor backend and its relationship to
the TPU-Kernel programming model. Both paths use the same TileLang frontend and
the same PPL 1.7 SDK, but select different device instruction interfaces.

## Target model

TPU compilation has three explicit selection axes:

| Axis | Values | Purpose |
| --- | --- | --- |
| Chip | `bm1690`, `sg2260e` | PPL architecture, compile definitions, core count, and chip capabilities |
| Programming model | `tpukernel`, `rv` | Device ABI, descriptors, instructions, and kernel lifecycle |
| Runtime mode | `cmodel`, `pcie` | Simulator or board host runtime |

BM1690 supports TPU-Kernel. SG2260E supports TPU-Kernel and RV Tensor. The
runtime mode does not participate in instruction selection.

```python
kernel = tilelang.compile(
    program,
    target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
    runtime_mode="cmodel",
)
```

`PPL` names the SDK and ABI provider; it is not a TileLang programming model.
`codegen_tpu.{h,cc}` contains the shared TPU source generator, while
`codegen_tpukernel.cc` and `codegen_rv.cc` contain programming-model-specific
instruction selection.

## Lowering path

```text
T.ppl_* frontend operations
        |
        v
tl.tpu.{copy,fill,gemm,add,sub,mul,div,max}
        |
        v
TPU semantic checks, lowering, and AddressAssign
        |
        +---- TPU-Kernel -> codegen_tpukernel.cc -> tpu_kernel.h
        |
        `---- RV Tensor  -> codegen_rv.cc        -> rvt_api.h
        |
        v
PPL 1.7 build and CModel or PCIe runtime
```

The public `T.ppl_*` operations first produce programming-model-independent
`tl.tpu.*` semantics. Instruction selection occurs after target binding. The
frontend therefore does not expose RV descriptors or require separate operator
implementations for each backend.

Raw `rvt_*` calls remain an expert interface for kernels that manage registers,
descriptors, and synchronization directly. A kernel cannot mix raw RV calls
with compiler-managed `tl.tpu.*` or TPU-Kernel operations.

## Operation mapping

| TileLang operation | TPU-Kernel | RV Tensor | Main RV constraints |
| --- | --- | --- | --- |
| `tl.tpu.copy` | GDMA S2L, L2S, S2S; BDC L2L and local cast | `rvt_dma_ld`, `rvt_dma_st`, `rvt_dma_cp`, `rvt_cvt_f2f` | Static equal extents; cross-type conversion is local only |
| `tl.tpu.fill` | `tpu_bdc_set_C` | Typed CR followed by `rvt_cp` | Zero fill only |
| `tl.tpu.gemm` | FP and FP8 matrix instructions | `rvt_fmm2[a]_{nn,nt}` | Rank-2 local FP16 or BF16 inputs; no transpose-A |
| `tl.tpu.add/sub/mul/div/max` | `tpu_bdc_fp_*`, `tpu_bdc_max` | `rvt_fadd/fsub/fmul/fdiv/fmax` | Matching FP16, BF16, or FP32 types; equal shape or W broadcast |

GEMM overwrite mode accepts an output matching the input type or FP32.
Accumulation requires an FP32 output. The `accumulate` flag selects the RV
`fmm2a` form and marks the output as read-write for address analysis.

The current RV elementwise broadcast describes the right operand as a
zero-stride W view. This is a descriptor operation and does not copy or expand
the local tensor.

## RV descriptors and lifecycle

The implementation follows the SG2260E PPL 1.7 `rvt_api.h` register ranges:

- CR: R0-R7
- TR: R8-R31
- GR: R32-R39

Descriptors use `PRECISION(DT_*)` and the matching floating-point subtype from
`FP8TYPE(DT_*)`. Global and local DMA regions use explicit free-layout strides;
local matrix and elementwise tiles use the hardware-aligned layout required by
the instruction.

An RV kernel starts with `rvt_kernel_start()`, configures the GDMA lane mask,
emits descriptors and commands, and finishes with
`rvt_sync_i(0xdeadbeef, 0)`. It does not use TPU-Kernel initialization or
polling.

## Compiler checks

The compiler rejects an RV kernel before code generation when it contains:

- A TPU-Kernel-only operation such as reduction, `exp`, `sigmoid`, `rsqrt`,
  gather, top-k, or RoPE
- FP8 RV arithmetic or GEMM
- Unsupported rank, extent, type, transpose, alias, or region layout
- Residual vector operations or unregistered extern calls
- A mixture of raw RV and compiler-managed TPU operations

`AddressAssign` uses operand effects from the semantic operation. Copy sources
are read-only and destinations are write-only. GEMM outputs are write-only in
overwrite mode and read-write in accumulation mode. Loop-carried local buffers
remain live across the loop back edge.

## Toolchain and runtime

`ppl_layout.py` accepts the PPL 1.7 `deps/` release layout. It obtains the
architecture and core count from `TPUChipSpec`, checks `rvt_api.h` for an RV
target, and selects the required compiler, libraries, emulator, or firmware.

CModel uses the SDK runtime. PCIe uses the installed TPUv7 board runtime while
retaining headers and chip libraries from the same SDK. A loaded JIT module is
bound to one chip, programming model, runtime mode, device ID, and SDK/runtime
identity.

## High-level examples

The SG2260E RV path supports the shared frontend implementations of:

- Elementwise add, subtract, multiply, and divide
- Matmul

These examples support FP16, BF16, and FP32 public inputs and outputs. RMSNorm,
split-K RMSNorm, RoPE, SwiGLU, and FlashAttention remain on TPU-Kernel because
their current implementations require TPU-Kernel-only math or reduction
operations.

## Current limits

- RV fill accepts zero only.
- RV GEMM accepts two-dimensional FP16 or BF16 matrix inputs.
- RV supports one W-axis broadcast form, not general NumPy broadcasting.
- RV reductions, activation functions, normalization, attention, and FP8
  mappings are not implemented.
- Kernels run on one logical core; four-core partitioning and synchronization
  are not implemented.
- The software-pipeline pass remains disabled until TPU DMA/compute hazards and
  command dependencies are modeled.

## Main implementation files

- `tilelang/engine/tpu_config.py`: chip, programming model, and runtime parsing
- `tilelang/jit/adapter/ppl_layout.py`: PPL 1.7 SDK layout
- `tilelang/jit/adapter/libgen.py`: CModel and PCIe compilation
- `src/target/codegen_tpu.{h,cc}`: shared TPU source generation
- `src/target/codegen_tpukernel.cc`: TPU-Kernel instruction selection
- `src/target/codegen_rv.cc`: RV Tensor instruction selection
- `testing/python/jit/tpu_core_ops_matrix.py`: core operation matrix
- `testing/python/jit/tpu_profile_worker.py`: isolated numerical and trace worker
