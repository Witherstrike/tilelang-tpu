# SG2260E RV TIR legalization and register allocation

## Placement and dispatch

`RVLegalizeAndAllocateRegisters` runs in `OptimizeForTarget` immediately after
`AddressAssign`. It is selected only when the target is TPU and the normalized
compile configuration has `chip="sg2260e"` and `device_mode="rv"`. The atomic
and BM1690 paths are unchanged.

This is the final structured-IR boundary before RV device code generation. The
pass deliberately does not inspect or replace generated C strings.

## TIR contract

Every supported `ppl.*` external call is rewritten to `ppl.rv.*`. Its tensor
arguments become `LetStmt`-bound `ppl.rv.tensor_view` descriptors containing:

- buffer name and semantic operand role;
- register class and allocated register number;
- layout and global/local memory scope;
- dtype code, bits, and lanes;
- access kind (read, write, read-write, or conservative);
- original region/access expression;
- normalized four-dimensional shape and stride.

Scalar arguments that occupy a control register become explicit
`ppl.rv.scalar` descriptors. Function attributes record chip, device mode,
schema version, legalization completion, and operation count. Version 1 uses
three independent spaces: CR starts at 1, TR at 8, and GR at 32.

Layout is attached to each operand view. This matters because one buffer may
need different layouts for full and partial accesses. Global full views are
continuous, local full views are HW-aligned, and partial copy/global views are
free-layout. Allocation is deterministic within a function and reuses the same
register for the same `(buffer, layout)` view.

The pass recognizes function buffer maps, block `alloc_buffers`,
`match_buffers`, lowered `Allocate` nodes, TileLang regions, and raw
`tvm_access_ptr` operands. Unknown `ppl.*` schemas and unsupported chips fail
before device C generation rather than silently losing information.

## Covered operation schemas

The initial schema covers copy, fill, GEMM, binary/scalar elementwise,
exp/sigmoid/rsqrt, sum/max reduction, gather, top-k, and rope-add. Structural
tests exercise one representative of every supported operator family, verify
the function attributes and descriptor fields, and verify deterministic
allocation.

## Current limitations and next step

This change implements the structured legalization/register-allocation
contract, not the RV C emitter. The existing atomic `target.build.tilelang_ppl`
emitter must not be used to interpret `ppl.rv.*` calls. The next implementation
step is a dedicated RV code-generation path that consumes these descriptors
and emits `rv_*` APIs. Exact local-bank address/liveness allocation should be
added there or in a follow-up structured pass.

## Verification

```bash
make -C build -j"$(nproc)"
PYTHONPATH=.:3rdparty/tvm/python LD_LIBRARY_PATH=build:build/tvm \
  .venv-cpu/bin/python -m pytest -q \
  testing/python/transform/test_tilelang_transform_rv_legalize.py \
  testing/python/jit/test_tpu_config.py \
  testing/python/jit/test_ppl_layout.py
```

PCIe hardware execution is not claimed; this iteration concerns the common
compile-time TIR contract for cmodel and PCIe.
