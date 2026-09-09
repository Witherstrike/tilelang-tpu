# TPU target configuration

TPU compilation has three independent selections:

- `chip`: the logical chip name understood by the PPL SDK, such as `bm1690`
  or `sg2260e`;
- `device_mode`: device API/code-generation mode, either `atomic` or `rv`;
- `runtime_mode`: execution environment, either `cmodel` or `pcie`.

For example:

```python
kernel = tilelang.compile(
    func,
    target="tpu",
    chip="sg2260e",
    device_mode="atomic",
    runtime_mode="cmodel",
)
```

The configuration is normalized into `TPUCompileConfig` and propagated through
`compile`, the kernel cache, `JITKernel`, `lower`, the compiled artifact, the
Cython adapter, and `LibraryGenerator`. `LibraryGenerator` resolves `chip`
through the installed PPL SDK's chip map instead of using a hard-coded BM1690
path.

The legacy `mode="cmodel"`/`mode="pcie"` argument remains an alias for
`runtime_mode`. Supplying conflicting values raises `ValueError`.

SG2260E RV now has a structured TIR legalization/register-allocation pass. It
runs after address assignment and records explicit GR/TR/CR and operand-view
layout information; see `sg2260e_rv_legalization.md`. The dedicated RV C emitter
supports both runtime selections. See [the all-API handoff](sg2260e_rv_all_apis_handoff.md)
for supported forms, deployment variables and outstanding device verification.
The atomic `tpu_*` emitter remains separate; existing atomic kernels can
continue to select `device_mode="atomic"`.
