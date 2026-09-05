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
# SG2260E TPU-Kernel CModel numerical baseline
PPL_PROJECT_ROOT=/path/to/ppl-1.7 python tpu_demo/matmul/tpu_test_matmul_fp16.py
```

The other historical demos are development examples, not a blanket SG2260E
support claim: review their complete Target and runtime selection before
running them. Do not run a PCIe demo until its CModel numerical case has passed
and an external watchdog is in place.

For a more detailed setup guide, see [`tpu_demo/instruction.md`](./tpu_demo/instruction.md).

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

Selecting `-tpu-programming-model=rv` exposes a direct, low-level bridge to the
PPL 1.7 RV Tensor ABI. `T.rvt_*` calls require explicit CR/TR/GR descriptor setup.
The generic
`T.rvt_call` escape hatch only supports APIs whose C arguments are representable
by TIR; APIs taking C structs by value need dedicated helpers first. Raw
`T.rvt_*` and high-level `T.ppl_*` calls also cannot share one kernel, because
their descriptor and command-stream ownership models are different.

The recorded SG2260E evidence covers FP32 add/sub/mul/div and a fixed-shape FP16
matmul in both CModel and supervised PCIe runs; see the reports under
`research/`. This scope must not be extrapolated to arbitrary dtype, shape,
tail, multi-core execution, or raw `T.rvt_*` programs.

All TPU JIT entry points default to the fail-safe `cmodel` runtime when no
runtime mode is supplied. PCIe
loading is fail-closed because loading the runtime can touch the board even
before a dispatch. Only after a CModel numerical smoke test should a supervised
PCIe bring-up explicitly set both variables below, select `runtime_mode="pcie"`,
and run one dispatch under an external watchdog:

```bash
export TILELANG_TPU_ALLOW_PCIE_LOAD=1
export TILELANG_TPU_DEVICE_ID=<verified-board-id>
timeout --kill-after=5s 30s python your_single_dispatch_smoke.py
```

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
- `T.ppl_reduce_sum`
- `T.ppl_reduce_max`
- `T.ppl_add`, `T.ppl_subtract`, `T.ppl_mul`, `T.ppl_div`
- `T.ppl_add_C`, `T.ppl_mul_C`
- `T.ppl_rsqrt`
- `T.ppl_rope_add`

## Examples

The current examples mainly cover operators commonly used in Llama and DeepSeek workloads, and more operators will be added over time.

Representative examples include:

- Matmul
- RMSNorm
- RoPE
- Reduce
- SwiGLU
- FlashAttention

## Repository Layout

- [`tilelang/`](./tilelang/): TileLang frontend and TPU-facing Python entry points
- [`src/target/`](./src/target/): TPU codegen and runtime modules
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/): checked-in TPU code templates; JIT artifacts use private temporary workspaces
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/): TPU JIT wrapper and library generation flow
- [`tpu_demo/`](./tpu_demo/): TPU demos and bring-up scripts

## Development Notes

- If you modify C++ code, rebuild the native components before rerunning demos:

```bash
make -j 10
```

- Format the repository with:

```bash
./format.sh
```

## Acknowledgements

TileLang-TPU builds on open-source work from [TileLang](https://github.com/tile-ai/tilelang) and uses the TPU compilation ecosystem associated with [SOPHGO PPL](https://github.com/sophgo/PPL/).
