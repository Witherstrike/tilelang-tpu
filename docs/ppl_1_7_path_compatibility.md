# PPL 1.7 Path Compatibility

This note records the compatibility work required to run the current BM1690
TileLang-TPU flow with PPL 1.7. It does not add SG2260E RV code generation.

## Reproduced failures

The previous JIT adapter constructed every PPL path from
`runtime/bm1690/...`. PPL 1.7 maps the logical chip name `bm1690` to the
architecture directory `tpub_7_1` in `deps/chip/chip_map.json`, and splits its
files among these roots:

| Purpose | PPL 1.7 location |
| --- | --- |
| TPU kernel headers and backend | `deps/chip/tpub_7_1` |
| Device helper and common kernel code | `deps/common/dev` |
| Host headers | `deps/common/host/include` |
| TPUv7 runtime | `deps/runtime/tpuv7-runtime` |

After fixing those paths, the matmul demo exposed three more version
boundaries:

1. PPL 1.7 host headers require C++17, while TileLang used C++11.
2. `host_test_utils.h` was included by the generated entry point but never
   used. In PPL 1.7 it unnecessarily pulls in the separate tpuDNN headers.
3. PPL 1.7 defines `__ppl_get_dtype` in `ppl_helper.h`, while TileLang's code
   generator emitted a second definition for old PPL releases.

The cmodel entry point also used the PCIe-specific device id 14. Cmodel now
selects device 0 under `USING_CMODEL`; PCIe retains its previous device id.

## Implemented compatibility layer

`tilelang/jit/adapter/ppl_layout.py` detects PPL 1.7 through
`deps/chip/chip_map.json`, resolves the logical chip to its architecture, and
returns all include, source, runtime, backend, emulator, firmware, and
toolchain paths. If the map is absent, it resolves the legacy layout instead.
Both cmodel and PCIe compilation use this one result.

For PPL 1.7 BM1690, compilation defines `__tpub_7_1__` and `__sg2260__`, as
specified by PPL's `config_common.cmake`. Cmodel additionally defines
`USING_CMODEL`. Runtime and emulator directories are embedded as rpaths, and
the JIT sets `PPL_KERNEL_PATH` to its generated `libkernel.so` automatically.

PCIe uses the resolved `libfirmware_core.a` rather than the old
`-lbm1690`. The current PPL release in this workspace does not contain the
Xuantie cross compiler under `third_party/toolchains_dir`, so PCIe compilation
now stops immediately with the exact missing compiler path. Actual PCIe
execution was not attempted because the host has no corresponding hardware.

## Cmodel verification

The repository was built with the environment-variable workflow from the
README. A CPU-only uv environment was used, and the example was switched to
`mode="cmodel"`. The verification command was equivalent to:

```bash
export PPL_PROJECT_ROOT=/path/to/ppl_v1.7.122-g05ebfb36-20260528
export PYTHONPATH=.
export LD_LIBRARY_PATH=build:build/tvm:$PPL_PROJECT_ROOT/deps/runtime/tpuv7-runtime/lib:$PPL_PROJECT_ROOT/deps/chip/tpub_7_1/lib
export TPU_RT_CORE_NUM=8
python tpu_demo/matmul/tpu_test_matmul_fp16.py
```

PPL cmodel uses local TCP and Unix sockets, so it must run in an environment
that permits local socket creation. The 64x64 FP16 matmul completed with
`torch.allclose == True`; the maximum absolute difference was
`0.00048828125`.

## Scope for SG2260E RV work

This change deliberately leaves the logical chip fixed to BM1690. The new
resolver already understands PPL's `sg2260e -> tpub_7_1_e` mapping, but merely
selecting that mapping would still compile TileLang's current atomic `tpu_*`
kernel output. SG2260E RV support must be introduced separately at the target,
lowering/codegen, RV host/runtime, and test layers rather than being presented
as a path-only switch.
