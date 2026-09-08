# TileLang-TPU

TileLang-TPU is a TPU-oriented extension of TileLang for SOPHGO accelerators. It preserves the TileLang Python DSL while adding TPU lowering, TPU code generation, and JIT runtime integration, enabling TileLang kernels to be compiled and executed on TPU platforms.

The project provides TPU support for BM1690 and SG2260E. A complete TPU target
selects both compile-time axes as
`target="tpu -mcpu=<chip> -tpu-programming-model=<tpukernel|rv>"`; the separate
`runtime_mode="cmodel|pcie"` selects how that program is hosted.

## Highlights

- TileLang frontend with TPU target support
- TPU-specific DSL intrinsics such as `ppl_copy`, `ppl_gemm`, `ppl_reduce_sum`, `ppl_reduce_max`, `ppl_rsqrt`, and `ppl_rope_add`
- End-to-end JIT flow from Python kernel definition to generated TPU host/device artifacts
- BM1690 and SG2260E target selection with CModel and guarded PCIe build paths
- Operator coverage aligned with kernels commonly used by Llama and DeepSeek workloads

## What This Project Does

TileLang-TPU is a TileLang-to-TPU compiler and runtime path for SOPHGO accelerators.

- It reuses the TileLang frontend and extends it with TPU-specific lowering, codegen, and JIT support.
- It uses the underlying TPU compilation tools also used in the PPL repository, but it is not built as a layer on top of the PPL software stack.
- It focuses on turning TileLang programs into runnable TPU kernels, including host/device wrapper generation and execution flow integration.

In short, this repository is about bringing TileLang's programming model to SOPHGO TPU targets rather than repackaging PPL itself.

## Current Status

- The current mainline JIT path supports BM1690 and PPL 1.7 SG2260E.
- `tilelang.compile(..., target="tpu -mcpu=sg2260e -tpu-programming-model=tpukernel")`
  is wired into the repository.
- CModel loading and PCIe static build paths are present; PCIe board dispatch
  remains deliberately opt-in. Only the scoped results recorded under
  `research/` are claimed; this is not blanket PCIe validation.
- TPU demos are available under [`tpu_demo/`](./tpu_demo/).
- This project is under active development, and pull requests are welcome.

## Requirements

- Linux and Python 3
- SOPHGO PPL 1.7 SDK
- Access to supported hardware or a working `cmodel` setup

## Key TPU Paths

- [`tilelang/engine/`](./tilelang/engine/)
- [`tilelang/language/`](./tilelang/language/)
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/)
- [`src/target/`](./src/target/)
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/)
- [`src/transform/`](./src/transform/)
- [`tpu_demo/`](./tpu_demo/)

## Quick Start

### 1. Prepare the repository

```bash
git submodule update --init --recursive
```

### 2. Build and install

#### 2.1 Build via environment variable

```bash
./install_tpu.sh
export PYTHONPATH=.
```

#### 2.2 Build via pip local project

```bash
pip install -e . -v
```

### 3. Run a TPU demo

```bash
# SG2260E RV Tensor CModel numerical example
PPL_PROJECT_ROOT=/path/to/ppl-1.7 python3 -m tpu_demo.run \
  --case matmul.float16 \
  --chip sg2260e \
  --programming-model rv \
  --runtime-mode cmodel
```

The public demo registry contains import-safe elementwise, matmul, RMSNorm,
RMSNorm split-k, RoPE, SwiGLU, and FlashAttention examples. It is not a blanket
hardware-support claim; the exact staged evidence and the only supported PCIe
workflow are documented in [`tpu_demo/README.md`](./tpu_demo/README.md).

## Programming Model

TileLang-TPU keeps the familiar TileLang workflow and switches the backend to TPU:

```python
import tilelang

kernel = tilelang.compile(
    my_kernel,
    out_idx=-1,
    target=("tpu -mcpu=sg2260e "
            "-tpu-programming-model=tpukernel"),
    runtime_mode="cmodel",
)
```

Compilation and execution have separate identities. `TPUTargetSpec` is derived
only from the complete TVM Target and contains `(chip, programming_model)`.
`TPURuntimeConfig` contains only `runtime_mode`; it does not choose instructions
or change device semantics. BM1690 has 8 physical cores and supports TPU-Kernel.
SG2260E has 4 physical cores and supports TPU-Kernel and RV. Both chips share the
TPUv7 LMEM geometry used by the current allocator, but their PPL architecture,
core topology and programming-model capability remain explicit.

A bare `target="tpu"`, a target missing either `-mcpu` or
`-tpu-programming-model`, and an unsupported chip/model pair all fail before
lowering. There is no second public selector for either compile-time axis.

`target="auto"` never selects a TPU or falls back to a bare TPU target. CUDA/HIP
keep priority; otherwise it uses the portable C backend. Environment variables
may locate the PPL SDK and runtime, but they never replace either compile-time
field in the explicit TPU Target, so a CPU-only process cannot silently become a
TPU job.

Portable `T.ppl_copy/fill/gemm/add/subtract/mul/div/max` expressions are lowered
to `tl.tpu.*` first, then selected as TPU-Kernel or RV Tensor instructions from
the complete target. The current public demo registry exercises RV elementwise
add/sub/mul/div and matmul for FP16, BF16, and FP32. Selecting
`-tpu-programming-model=rv` also exposes a direct expert bridge to the PPL 1.7 RV
Tensor ABI. `T.rvt_*` calls require explicit CR/TR/GR descriptor setup. The generic
`T.rvt_call` escape hatch only supports APIs whose C arguments are representable
by TIR; APIs taking C structs by value need dedicated helpers first. Raw
`T.rvt_*` and high-level `T.ppl_*` calls also cannot share one kernel, because
their descriptor and command-stream ownership models are different.

Runtime support remains scoped by exact dtype, shape, variant, target, and
execution mode. See the machine-readable
[`contract.json`](./research/tpu-op-contract/contract.json) and the high-level
[`design and validation report`](./research/tpu-demo-ops/README.md); no result may
be extrapolated to arbitrary shapes, tails, multi-core execution, or raw
`T.rvt_*` programs.

All TPU JIT entry points default to the fail-safe `cmodel` runtime when no
runtime mode is supplied. PCIe
loading is fail-closed because loading the runtime can touch the board even
before a dispatch. Board execution is accepted only through the staged matrix
runner after matching BM1690 and SG2260E CModel summaries have passed on the same
clean source/toolchain identity:

```bash
python3 testing/python/jit/tpu_demo_ops_matrix.py \
  --runtime-mode pcie --chip sg2260e --device-id 0 \
  --allow-pcie --allow-pcie-profile --all-pcie-cases \
  --bm-cmodel-summary <bm-summary.json> \
  --sg-cmodel-summary <sg-summary.json> \
  --output-dir <new-not-yet-existing-artifact-directory>
```

The runner verifies the single-card topology and a content-addressed manifest
of the selected compiler/PPL/runtime inputs, holds one device lock across the whole invocation, compiles from a
read-only snapshot of the promoted Git commit, and rechecks source/toolchain
identity before every launch. Timeout or incomplete process-group cleanup stops
the matrix; a persistent session/quarantine marker keeps later runs fail-closed
until an operator has inspected and recovered the board.

TPU JIT compilation writes each kernel and its host wrapper to a private
temporary workspace; it no longer depends on a process-global `PPL_KERNEL_PATH`
or shared generated files under `src/tl_templates/tpu/`. TPU cache/database
artifact loading is deliberately disabled until it can bundle that private
kernel with a verified target/runtime manifest. Before any TPU `dlopen`,
the runtime also reserves `(runtime, chip, core count, programming model,
device, PPL SDK/runtime identity)` for the process; change any of those in a
fresh process, not in a long-lived Python worker. Through the TileLang JIT
loader, a TPU `main.so` may only be loaded by the `LibraryGenerator` instance
that just compiled it; prebuilt TPU artifacts remain disabled until a bundled,
validated manifest exists.

Inside TPU kernels, the common building blocks are exposed as TileLang DSL intrinsics in [`tilelang/language/customize.py`](./tilelang/language/customize.py), including:

- `T.ppl_copy`
- `T.ppl_fill`
- `T.ppl_gemm`
- `T.ppl_add`, `T.ppl_subtract`, `T.ppl_mul`, `T.ppl_div`, `T.ppl_max`
- `T.ppl_add_C`, `T.ppl_mul_C`
- `T.ppl_exp`, `T.ppl_sigmoid`, `T.ppl_rsqrt`
- `T.ppl_reduce_sum`
- `T.ppl_reduce_max`
- `T.ppl_gather`, `T.ppl_topk`
- `T.ppl_rope_add`

## Examples

The current examples mainly cover operators commonly used in Llama and DeepSeek workloads, and more operators will be added over time.

Representative examples include:

- Elementwise add/subtract/multiply/divide
- Matmul
- RMSNorm and split-k RMSNorm
- RoPE
- SwiGLU
- FlashAttention

## Repository Layout

- [`tilelang/`](./tilelang/): TileLang frontend and TPU-facing Python entry points
- [`src/target/`](./src/target/): TPU codegen and runtime modules
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/): checked-in TPU code templates; JIT artifacts use private temporary workspaces
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/): TPU JIT wrapper and library generation flow
- [`tpu_demo/`](./tpu_demo/): import-safe high-level examples and their case registry
- [`testing/python/jit/`](./testing/python/jit/): low-level probes and staged matrix runners

## Development Notes

- If you modify C++ code, rebuild the native components before rerunning demos:

```bash
cmake --build <configured-build-directory> --parallel 10
```

- Format the repository with:

```bash
./format.sh
```

## Acknowledgements

TileLang-TPU builds on open-source work from [TileLang](https://github.com/tile-ai/tilelang) and uses the TPU compilation ecosystem associated with [SOPHGO PPL](https://github.com/sophgo/PPL/).
