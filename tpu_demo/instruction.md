# tilelang + tpu Guides

## Installation
After cloning the repository, initialise its submodules and build TileLang.

```bash
git submodule update --init --recursive
./install_tpu.sh
```

After these commands, we should correctly `import tilelang` in python.

**Note:** rebuild after modifying C++ source. Prefer the configured CMake build
so build options remain consistent:

```bash
cmake --build build --parallel 10 # choose a safe job count for the host
```

## Target selection and bring-up status

New TPU examples must provide the complete compile-time target. Runtime remains
a separate host-side choice:

```python
tilelang.compile(
    kernel,
    target=("tpu -mcpu=sg2260e "
            "-tpu-programming-model=tpukernel"),
    runtime_mode="cmodel",
)
```

The target resolves to `TPUTargetSpec(chip, programming_model)`; the runtime
argument resolves independently to `TPURuntimeConfig(runtime_mode)`. BM1690 is
an 8-core TPU-Kernel target. SG2260E is a 4-core target supporting TPU-Kernel and
RV. Both use the TPUv7 LMEM geometry modeled by the current compiler. A bare
`target="tpu"` or a target missing either compile-time field is rejected.

`tpu_demo/matmul/tpu_test_matmul_fp16.py` is the SG2260E TPU-Kernel CModel
numerical baseline. Historical `tpu_demo/ppl/` scripts are raw SDK development
examples, not blanket SG2260E claims, and require per-op review. A vendor
`ppl.` / `tpu_` / `rvt_` extern fails clearly if it would lower to a non-TPU
target; do not work around that error with `target="auto"`.

PCIe code may be statically built but is not a normal test target. Do not load
or dispatch it until the corresponding CModel numerical case has passed; use a
fresh, externally supervised process with the explicit PCIe safety variables.
