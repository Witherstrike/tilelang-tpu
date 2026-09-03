# tilelang + tpu Guides

## Installation
After cloning the repository, initialise its submodules, apply the current TIR
compatibility patch, and build TileLang. This patch is not used to register the
TPU target kind: TileLang registers that itself and `install_tpu.sh` does not
overwrite vendored TVM target source.

```bash
git submodule update --init --recursive
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
cd ../..
./install_tpu.sh
```

After these commands, we should correctly `import tilelang` in python.

**Note:** rebuild after modifying C++ source. Prefer the configured CMake build
so build options remain consistent:

```bash
cmake --build build --parallel 10 # choose a safe job count for the host
```

## Target selection and bring-up status

New TPU examples must select all three axes explicitly:

```python
tilelang.compile(
    kernel,
    target="tpu -mcpu=sg2260e",
    device_mode="tpukernel",  # or "rv" for direct RV Tensor ABI work
    runtime_mode="cmodel",
)
```

`tpu_demo/matmul/tpu_test_matmul_fp16.py` is the SG2260E TPU-Kernel CModel
numerical baseline. Historical `tpu_demo/ppl/` scripts and bare
`target="tpu"` examples are not blanket SG2260E claims: they retain BM1690
compatibility defaults or require per-op migration. A vendor `ppl.` / `tpu_` /
`rvt_` extern now fails clearly if it would lower to a non-TPU target; do not
work around that error with `target="auto"`.

PCIe code may be statically built but is not a normal test target. Do not load
or dispatch it until the corresponding CModel numerical case has passed; use a
fresh, externally supervised process with the explicit PCIe safety variables.
