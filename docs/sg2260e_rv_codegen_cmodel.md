# SG2260E RV code generation and cmodel

## Architecture

SG2260E RV uses the dedicated `target.build.tilelang_ppl_rv` emitter. Python
dispatches to it only for `device_mode="rv"`; the atomic emitter remains
separate. The emitter accepts RV-legalized schema version 1 TIR and rejects an
unknown chip, schema, operation, or incompatible descriptor at compile time.

The generated device C contains the PPL runtime argument structure, a fixed
`main_kernel(const void *)` launcher, `rvt_kernel_start`, final
`rvt_sync_i`, and `TPUKERNEL_FUNC_REGISTER(main_kernel)`. Pipeline regions are
consumed as structured TIR attributes and emitted as paired
`tpu_parallel_start/end`; cmodel does not delete or rewrite those calls as C
text.

## Supported operations

| Operation | Supported RV form |
|---|---|
| copy | global/local DMA, global/global copy, local/local copy or dtype conversion; FREE-layout views |
| add/subtract/mul/div, add_C/mul_C | local FP16/FP32 arithmetic; tensor RHS broadcasting |
| fill | FP16/FP32 destination with scalar CR conversion |
| GEMM | FP16 inputs, FP32 read-write accumulator; NN, NT, and TT `fmm2a` variants |
| reciprocal square root | FP16/FP32, constant iteration count 1–8 (default 3) |
| gather | global FP16 table/output, UINT32 index, validated `param_h` |
| exp2 / sigmoid | local FP32 arithmetic decomposition; exp2 computes natural exp, as in the public API |
| reduce_sum / reduce_max | local 2D FP16/FP32, dim=1; column-wise RV decomposition |
| rope_add | local 2D FP16/FP32, even last dimension, non-aliasing output |
| topk | global contiguous FP32/INT32/UINT32, INT32/UINT32 indices, constant K/length; stable RV selection |

TN GEMM has no SG2260E `fmm2` variant and fails explicitly. The original PPL
reduction/top-k blockers remain, but TileLang now decomposes these operations
into available RV instructions instead of calling the blocked lowering.
Unsupported dtype/shape forms still fail without atomic fallback. The new
decompositions have compilation coverage, not device numerical sign-off.
See [the all-API handoff](sg2260e_rv_all_apis_handoff.md) for boundaries and tests.

## Local memory

`ChipDescription` supplies lane, EU, alignment, bank, capacity, and reuse
policy for BM1690 and SG2260E. `AddressAssign` records address, size, bank,
live range, conflict edges, selected chip, and reuse policy in machine-readable
PrimFunc attributes. The corpus comparator checks these summaries against PPL
final MLIR using structural invariants rather than requiring a unique physical
address solution.

BM1690 retains lifetime-based address reuse. SG2260E conservatively disables
reuse between distinct buffers: raw RV DMA/TIU completion can outlive the TIR
statement range, so static reuse can overwrite an operand still in flight.

## PPL 1.7 cmodel

The cmodel build uses the SG2260E (`tpub_7_1_e`) headers, `ppl_helper.c`, all
PPL checker C sources, `libtpuv7_emulator.so`, TPUv7 runtime, and emulator
daemon. RV supports `chip=sg2260e` with `runtime_mode=cmodel` or `pcie`.
PCIe cross-compiles the kernel, helper and checker, links SG2260E firmware,
and links the host wrapper against the deployment's TPUv7 runtime. PCIe
execution still requires validation on a device host.

Build and run the normal gates:

```bash
cmake --build build -j2
export PYTHONPATH="$PWD:$PWD/3rdparty/tvm/python"
export TVM_LIBRARY_PATH="$PWD/build/tvm"
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/tvm"
.venv-cpu/bin/python -m pytest -q \
  testing/python/transform/test_tilelang_transform_rv_legalize.py \
  testing/python/transform/test_ppl_final_mlir_lmem_compare.py \
  testing/python/target/test_tilelang_codegen_ppl_rv.py \
  testing/python/jit/test_tpu_config.py
```

Numerical emulator tests are opt-in and must be run in separate processes,
because PPL 1.7 emulator reinitialization in one process can exit with status
255:

```bash
export PPL_PROJECT_ROOT=../ppl_v1.7.122-g05ebfb36-20260528
export TILELANG_RUN_TPU_CMODEL_TESTS=1
.venv-cpu/bin/python -m pytest -q \
  testing/python/target/test_tilelang_codegen_ppl_rv.py \
  -k pipeline_copy_add_cmodel_numerics
.venv-cpu/bin/python -m pytest -q \
  testing/python/target/test_tilelang_codegen_ppl_rv.py \
  -k 'gemm_accumulation_cmodel_numerics and not pipeline'
.venv-cpu/bin/python -m pytest -q \
  testing/python/target/test_tilelang_codegen_ppl_rv.py \
  -k 'gemm_accumulation_cmodel_numerics and pipeline'
```

Historical results from the preceding development iteration (not rerun or
extended to the new operators in this continuation): SG2260E cmodel results
were exact for pipeline copy/add and within
`5.96e-8` for serial and pipeline two-K-tile FP32 GEMM accumulation (test
tolerance `1e-6`).
