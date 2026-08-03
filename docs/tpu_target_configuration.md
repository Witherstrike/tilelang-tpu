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

SG2260E RV lowering and code generation are intentionally not part of this
parameterization change. Selecting `device_mode="rv"` currently raises
`NotImplementedError` at the device build boundary instead of silently
producing atomic `tpu_*` APIs. A working SG2260E baseline can use
`device_mode="atomic"`; the next backend implementation step will consume the
already-propagated RV setting during lowering and code generation.
