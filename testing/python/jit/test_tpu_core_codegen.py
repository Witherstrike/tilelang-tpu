# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Source-only checks for the portable TPU core-op instruction selectors.

These tests deliberately stop after ``tilelang.lower``.  They neither build a
runtime wrapper nor load or dispatch a kernel, so they are safe to run on a
host that also has a PCIe TPU attached.
"""

import os
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.tpu_config import TPUCompileConfig
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper


@T.prim_func
def _portable_core_ops(
        A: T.Tensor((16, 16), "float16"),
        B: T.Tensor((16, 16), "float16"),
        C: T.Tensor((16, 16), "float32"),
        X: T.Tensor((1, 32), "float32"),
        Y: T.Tensor((1, 32), "float32"),
        Z: T.Tensor((1, 32), "float32")):
    """One frontend program shared by TPU-Kernel and RV Tensor codegen."""
    T.func_attr({"global_symbol": "portable_core_ops", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        a_shared = T.alloc_shared((16, 16), "float16")
        b_shared = T.alloc_shared((16, 16), "float16")
        acc_shared = T.alloc_shared((16, 16), "float32")
        x_shared = T.alloc_shared((1, 32), "float32")
        y_shared = T.alloc_shared((1, 32), "float32")
        z_shared = T.alloc_shared((1, 32), "float32")

        T.ppl_fill(acc_shared, T.float32(0))
        T.ppl_copy(A, a_shared)
        T.ppl_copy(B, b_shared)
        T.ppl_gemm(a_shared, b_shared, acc_shared)
        T.ppl_copy(acc_shared, C)

        T.ppl_copy(X, x_shared)
        T.ppl_copy(Y, y_shared)
        T.ppl_add(z_shared, x_shared, y_shared)
        T.ppl_subtract(z_shared, x_shared, y_shared)
        T.ppl_mul(z_shared, x_shared, y_shared)
        T.ppl_div(z_shared, x_shared, y_shared)
        T.ppl_copy(z_shared, Z)


@T.prim_func
def _portable_and_raw_rv(
        X: T.Tensor((1, 32), "float32"),
        Y: T.Tensor((1, 32), "float32")):
    T.func_attr({"global_symbol": "portable_and_raw_rv", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        x_shared = T.alloc_shared((1, 32), "float32")
        y_shared = T.alloc_shared((1, 32), "float32")
        z_shared = T.alloc_shared((1, 32), "float32")
        T.ppl_copy(X, x_shared)
        T.ppl_copy(Y, y_shared)
        T.ppl_add(z_shared, x_shared, y_shared)
        # Raw RV calls own their register descriptors and lifecycle.  Mixing
        # one into a compiler-managed portable kernel must fail closed.
        T.rvt_fadd(T.uint64(10), T.uint64(8), T.uint64(9))


@T.prim_func
def _ffi_alias_probe(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "ffi_alias_probe", "tir.noalias": T.bool(True)})
    T.evaluate(T.call_extern("handle", "tpu_sync_all_bdc"))


@T.prim_func
def _global_to_global_copy(
        A: T.Tensor((1, 32), "float32"),
        B: T.Tensor((1, 32), "float32")):
    T.func_attr({"global_symbol": "global_to_global_copy", "tir.noalias": T.bool(True)})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        T.ppl_copy(A, B)


@T.prim_func
def _local_fp16_to_bf16_copy():
    T.func_attr({"global_symbol": "local_fp16_to_bf16_copy"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        src = T.alloc_shared((1, 32), "float16")
        dst = T.alloc_shared((1, 32), "bfloat16")
        T.ppl_copy(src, dst)


@T.prim_func
def _local_integer_convert():
    T.func_attr({"global_symbol": "local_integer_convert"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        src = T.alloc_shared((1, 32), "int8")
        dst = T.alloc_shared((1, 32), "int16")
        T.ppl_copy(src, dst)


@T.prim_func
def _rank4_elementwise():
    T.func_attr({"global_symbol": "rank4_elementwise"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        lhs = T.alloc_shared((2, 4, 1, 8), "float32")
        rhs = T.alloc_shared((2, 4, 1, 8), "float32")
        out = T.alloc_shared((2, 4, 1, 8), "float32")
        T.ppl_add(out, lhs, rhs)


@T.prim_func
def _overwrite_fp16_gemm():
    T.func_attr({"global_symbol": "overwrite_fp16_gemm"})
    with T.Kernel(1, 1, is_cpu=True) as (_bx, _by):
        lhs = T.alloc_shared((16, 16), "float16")
        rhs = T.alloc_shared((16, 16), "float16")
        out = T.alloc_shared((16, 16), "float16")
        T.ppl_gemm(lhs, rhs, out, accumulate=False)


def _lower_core_ops(chip, device_mode):
    return tilelang.lower(
        _portable_core_ops,
        target=f"tpu -mcpu={chip}",
        device_mode=device_mode,
        runtime_mode="cmodel",
    ).kernel_source


@pytest.fixture(scope="module")
def sg2260e_sources():
    return {
        "tpukernel": _lower_core_ops("sg2260e", "tpukernel"),
        "rv": _lower_core_ops("sg2260e", "rv"),
    }


def test_public_ppl_helpers_form_one_backend_neutral_tir_contract():
    extern_names = set()

    def collect(node):
        if (isinstance(node, tvm.tir.Call) and
                getattr(node.op, "name", None) == "tir.call_extern"):
            extern_names.add(str(node.args[0].value))

    tvm.tir.stmt_functor.post_order_visit(_portable_core_ops.body, collect)
    assert extern_names == {
        "tl.tpu.copy",
        "tl.tpu.fill",
        "tl.tpu.gemm",
        "tl.tpu.add",
        "tl.tpu.sub",
        "tl.tpu.mul",
        "tl.tpu.div",
    }


def test_sg2260e_tpukernel_selects_only_tpukernel_instructions(sg2260e_sources):
    source = sg2260e_sources["tpukernel"]

    assert "TileLang TPU target: sg2260e, programming model: tpukernel" in source
    assert "TPU-Kernel externs require TPU device_mode=tpukernel" in source
    assert "tpu_initialize()" in source
    assert "tpu_poll()" in source
    for instruction in (
            "tpu_gdma_cpy_S2L(",
            "tpu_gdma_cpy_L2S(",
            "tpu_bdc_set_C(",
            "tpu_bdc_fp_mm(",
            "tpu_bdc_fp_add(",
            "tpu_bdc_fp_sub(",
            "tpu_bdc_fp_mul(",
            "tpu_bdc_fp_div("):
        assert instruction in source
    assert '#include "rvt_api.h"' not in source
    assert "rvt_kernel_start(" not in source
    assert "rvt_sync_i(" not in source
    assert "rvt_fmm" not in source


def test_sg2260e_rv_selects_only_rv_tensor_instructions(sg2260e_sources):
    source = sg2260e_sources["rv"]

    assert "TileLang TPU target: sg2260e, programming model: rv" in source
    assert "RVT externs require TPU device_mode=rv" in source
    assert '#include "atomic_def.h"' in source
    assert '#include "rvt_api.h"' in source
    assert "TILELANG_TPU_OPAQUE_RAW_RVT_ABI" not in source

    # The compiler, rather than the frontend program, owns this lifecycle.
    assert source.count("rvt_kernel_start()") == 1
    assert source.count("rvt_cfg_lanemask(gdma_get_lane_mask())") == 1
    assert source.count("rvt_sync_i(0xdeadbeef, 0)") == 1
    assert source.index("rvt_kernel_start()") > source.index("rvt_cfg_lanemask(")

    for descriptor_fragment in (
            "rvt_gr(32, PRECISION(DT_FP16), FP8TYPE(DT_FP16)",
            "rvt_tr(8, PRECISION(DT_FP16), FP8TYPE(DT_FP16)",
            "PRECISION(DT_FP32), FP8TYPE(DT_FP32)",
            "FREE_LAYOUT",
            "HW_ALIGN_LAYOUT"):
        assert descriptor_fragment in source
    for instruction in (
            "rvt_dma_ld(",
            "rvt_dma_st(",
            "rvt_cp(",
            "rvt_fmm2a_nn(",
            "rvt_fadd(",
            "rvt_fsub(",
            "rvt_fmul(",
            "rvt_cfg_rsqrt_iter(3)",
            "rvt_fdiv("):
        assert instruction in source
    assert "rvt_cr(1, PRECISION(DT_FP32), FP8TYPE(DT_FP32)" in source
    assert "rvt_cfg_quant(0)" in source

    assert "tpu_initialize()" not in source
    assert "tpu_poll()" not in source
    assert "tpu_gdma_cpy_" not in source
    assert "tpu_bdc_fp_" not in source


def test_same_frontend_reaches_distinct_sg2260e_instruction_selectors(sg2260e_sources):
    tpukernel = sg2260e_sources["tpukernel"]
    rv = sg2260e_sources["rv"]

    assert tpukernel != rv
    assert "tpu_bdc_fp_mm(" in tpukernel and "rvt_fmm2a_nn(" not in tpukernel
    assert "rvt_fmm2a_nn(" in rv and "tpu_bdc_fp_mm(" not in rv
    assert "tpu_bdc_fp_div(" in tpukernel and "rvt_fdiv(" not in tpukernel
    assert "rvt_fdiv(" in rv and "tpu_bdc_fp_div(" not in rv


def test_bm1690_tpukernel_source_regression_and_rv_rejection():
    source = _lower_core_ops("bm1690", "tpukernel")
    assert "TileLang TPU target: bm1690, programming model: tpukernel" in source
    assert "tpu_bdc_fp_mm(" in source
    assert "tpu_bdc_fp_add(" in source
    assert '#include "rvt_api.h"' not in source

    with pytest.raises(ValueError, match="bm1690.*does not support device_mode='rv'"):
        _lower_core_ops("bm1690", "rv")


def test_portable_ops_and_raw_rv_abi_cannot_share_a_kernel():
    with pytest.raises(tvm.error.TVMError, match="cannot share a kernel with raw.*rvt"):
        tilelang.lower(
            _portable_and_raw_rv,
            target="tpu -mcpu=sg2260e",
            device_mode="rv",
            runtime_mode="cmodel",
        )


def test_legacy_codegen_registry_alias_matches_the_neutral_name():
    target = tvm.target.Target({
        "kind": "tpu",
        "mcpu": "sg2260e",
        "tpu-programming-model": "tpukernel",
    })
    module = tvm.IRModule({"ffi_alias_probe": _ffi_alias_probe})
    canonical = tvm._ffi.get_global_func("target.build.tilelang_tpu")
    compatibility = tvm._ffi.get_global_func("target.build.tilelang_ppl")

    assert canonical(module, target) == compatibility(module, target)


def test_reserved_runtime_entry_name_is_rejected():
    reserved = _ffi_alias_probe.with_attr("global_symbol", "main_kernel")
    with pytest.raises(tvm.error.TVMError, match="conflicts with a generated runtime entry"):
        tilelang.lower(
            reserved,
            target="tpu -mcpu=sg2260e",
            device_mode="tpukernel",
            runtime_mode="cmodel",
        )


def test_global_to_global_copy_uses_system_memory_instruction():
    tpukernel = tilelang.lower(
        _global_to_global_copy,
        target="tpu -mcpu=sg2260e",
        device_mode="tpukernel",
        runtime_mode="cmodel",
    ).kernel_source
    rv = tilelang.lower(
        _global_to_global_copy,
        target="tpu -mcpu=sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    ).kernel_source

    assert "tpu_gdma_cpy_S2S(" in tpukernel
    assert "tpu_bdc_cpy(" not in tpukernel
    assert "rvt_gr(32" in rv and "rvt_gr(33" in rv
    assert "rvt_dma_cp(33, 32)" in rv


def test_copy_conversion_capabilities_fail_closed():
    with pytest.raises(tvm.error.TVMError, match="does not support direct FP16/BF16"):
        tilelang.lower(
            _local_fp16_to_bf16_copy,
            target="tpu -mcpu=sg2260e",
            device_mode="tpukernel",
            runtime_mode="cmodel",
        )

    rv_source = tilelang.lower(
        _local_fp16_to_bf16_copy,
        target="tpu -mcpu=sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    ).kernel_source
    assert "rvt_cvt_f2f(9, 8)" in rv_source

    with pytest.raises(tvm.error.TVMError, match="rvt_cvt_f2f only accepts"):
        tilelang.lower(
            _local_integer_convert,
            target="tpu -mcpu=sg2260e",
            device_mode="rv",
            runtime_mode="cmodel",
        )


@pytest.mark.parametrize("device_mode", ("tpukernel", "rv"))
def test_portable_elementwise_rejects_non_matrix_local_layout(device_mode):
    with pytest.raises(tvm.error.TVMError, match="requires N=1"):
        tilelang.lower(
            _rank4_elementwise,
            target="tpu -mcpu=sg2260e",
            device_mode=device_mode,
            runtime_mode="cmodel",
        )


def test_overwrite_gemm_accepts_input_dtype_and_selects_non_accumulating_form():
    tpukernel = tilelang.lower(
        _overwrite_fp16_gemm,
        target="tpu -mcpu=sg2260e",
        device_mode="tpukernel",
        runtime_mode="cmodel",
    ).kernel_source
    rv = tilelang.lower(
        _overwrite_fp16_gemm,
        target="tpu -mcpu=sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    ).kernel_source

    assert "DT_FP16, DT_FP16, false)" in tpukernel
    assert "overwrite_fp16_gemm();" in tpukernel
    assert "rvt_fmm2_nn(10, 8, 9, 0, 0, 0)" in rv
    assert "rvt_fmm2a_nn(" not in rv


@pytest.mark.parametrize("device_mode", ("tpukernel", "rv"))
def test_sg2260e_pcie_core_ops_compile_and_link_without_loading(device_mode):
    """Validate the board artifact boundary without dlopen or dispatch."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    config = TPUCompileConfig(
        chip="sg2260e", device_mode=device_mode, runtime_mode="pcie")
    target = tvm.target.Target("tpu -mcpu=sg2260e")
    artifact = tilelang.lower(
        _portable_core_ops,
        target=target,
        chip=config.chip,
        device_mode=config.device_mode,
        runtime_mode=config.runtime_mode,
    )
    generator = LibraryGenerator(target, tpu_config=config)
    try:
        wrapper = TLWrapper(
            target, tpu_workspace_dir=generator.tpu_workspace_dir)
        module = tvm.IRModule({"portable_core_ops": _portable_core_ops})
        wrapper.assign_optimized_module(module)
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        assert (workspace / "libkernel.so").is_file()
        assert (workspace / "main.so").is_file()
        kernel_source = (workspace / "kernel.c").read_text(encoding="utf-8")
        if device_mode == "rv":
            assert "rvt_fmm2a_nn(" in kernel_source
            assert "tpu_initialize()" not in kernel_source
        else:
            assert "tpu_bdc_fp_mm(" in kernel_source
            assert "rvt_fmm2a_nn(" not in kernel_source
    finally:
        generator.remove_lib()
