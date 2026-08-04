# PPL SG2260E RV golden corpus

This directory captures the PPL 1.7.122 lowering contract used to guide
TileLang's SG2260E RV legalization and register allocation.  The `.pl` files
are deliberately small and have no `__TEST__` body.  Generated files under
`artifacts/` were produced by the packaged compiler, not written by hand.

## Reproduction

```bash
cd tilelang-tpu
export PPL_PROJECT_ROOT=../ppl_v1.7.122-g05ebfb36-20260528
bash testing/ppl_rv_golden/generate.sh /tmp/ppl-rv-golden
```

The exact compiler invocation for every operator is:

```bash
$PPL_PROJECT_ROOT/bin/ppl-compile OP.pl \
  -I "$PPL_PROJECT_ROOT/inc" \
  --chip sg2260e -D__sg2260e__ --rv --O3 --print-ir -o OUTPUT
```

`--print-ir` is required.  In RV mode the default invocation emits host/device
sources but does not retain the intermediate MLIR files.

Compiler identity used for the checked-in corpus:

- PPL release: `1.7.122-g05ebfb36-20260528`
- LLVM reported by `ppl-compile --version`: `17.0.0git`
- `ppl-compile` SHA-256:
  `c64d26e6cf08e18c42db7bcdb4a57c6289ad8110fa0f0949a1805fb3d829ccab`

## Result matrix

| case | frontend | opt | final | RV MLIR | device C | result |
|---|---:|---:|---:|---:|---:|---|
| copy | yes | yes | yes | yes | yes | success |
| fill | yes | yes | yes | yes | yes | success |
| elementwise add/mul | yes | yes | yes | yes | yes | success |
| FP16 GEMM | yes | yes | yes | yes | yes | success |
| sum reduction | yes | yes | yes | yes | yes | success |
| reciprocal square root | yes | yes | yes | yes | yes | success |
| gather-h | yes | yes | yes | yes | yes | success |
| top-k | yes | yes | yes | no | no | expected compiler failure, rc=23 |

Top-k fails in the packaged PPL compiler's RV instruction optimization pass:
`ppl.hau.topK` expects a tensor memref after its operands have already been
legalized to integer register descriptors.  `artifacts/topk/compile.log`
contains the full diagnostic.  The frontend/opt/final files are compiler
outputs emitted before that failure and must not be interpreted as a complete
golden path.

## Register and layout observations

The important boundary is `*_final.mlir` to `*_rv.mlir`:

- `final` still carries tensor semantics and local-memory allocation.  Global
  tensors use memory space `2`; local tensors use memory space `3` and include
  `address`, `size`, `idx`, `live_range`, and `bank_conflict` attributes.
- RV legalization replaces tensor operands with integer register IDs.
  `ppl_rv.reg_wgaddr/reg_wgshape` configure global tensors and
  `ppl_rv.reg_wladdr/reg_wlshape` configure local tensors.
- The generated C maps global tensors to `RVT_CFGGR` using registers 32 and
  above with `CONTINUOUS_LAYOUT`.  Local tensors map to `RVT_CFGTR` using
  registers 8 and above with `HW_ALIGN_LAYOUT`.
- Global stride is logical contiguous NCHW.  For `{1,8,1,16}` FP16 it is
  `{128,16,16,1}`.  Local aligned stride is `{32,32,16,1}` and a 128-element
  tensor occupies 64 bytes in the allocator's banked local representation.
- Local addresses are bank-based (`0`, `16384`, `32768`, ...).  Live ranges
  allow address reuse: elementwise `r` and `result` both use address 0 because
  their relevant lifetimes permit it.
- Scalars are materialized separately with `ppl_rv.scalar` followed by
  `ppl_rv.reg_wscalar`; control registers such as saturation, rounding,
  quantization, rsqrt iteration, and gather index are explicit RV operations.
- Operator lowering is structural: copy uses `dmaload/dmastore`, fill uses
  `tiu_set_c`, elementwise uses `arith`, GEMM uses `fmm2`, reduction uses
  `tiu_reduce`, special function uses `rsqrt`, and gather uses `dma_gather_h`.

Register *numbers* are not stable golden values. Repeated compilations produce
identical frontend/opt/final MLIR but may permute independently allocatable
global/local registers in `*_rv.mlir` and C. TileLang tests should therefore
check register classes, uniqueness/interference, layout and operation dataflow,
not byte-for-byte equality of allocated register numbers.
