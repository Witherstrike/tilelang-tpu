import os
from pathlib import Path

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.tpu_config import TPUCompileConfig
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout
from tilelang.jit.adapter.wrapper import TLWrapper


@T.prim_func
def _rvt_smoke_primfunc(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "rvt_smoke", "tir.noalias": T.bool(True)})
    T.rvt_kernel_start()
    # This mirrors the only RVT lifecycle use found in the PPL 1.7 package.
    # It is a control-path smoke test, not a tensor instruction test.
    T.rvt_sync_i(T.uint64(0xDEADBEEF), T.uint64(0))


@T.prim_func
def _rvt_codegen_primfunc(A: T.Tensor((1,), "float32")):
    """Compile/link coverage for a real RVT tensor instruction.

    The raw register IDs are deliberately not executed: a valid fadd needs
    preceding CR/TR/GR configuration and descriptor lifetime management.
    """
    T.func_attr({"global_symbol": "rvt_codegen", "tir.noalias": T.bool(True)})
    T.rvt_kernel_start()
    T.rvt_fadd(T.uint64(10), T.uint64(8), T.uint64(9))
    T.rvt_sync_all()


@T.prim_func
def _plain_multifunc_primfunc(A: T.Tensor((1,), "float32")):
    T.func_attr({"global_symbol": "plain_multifunc", "tir.noalias": T.bool(True)})
    A[0] = A[0]


def test_rvt_call_requires_rvt_symbol_name():
    with pytest.raises(ValueError, match="beginning with 'rvt_'"):
        T.rvt_call("ppl.copy")


def test_rvt_frontend_emits_vendor_extern_calls():
    call = T.rvt_fadd(tvm.tir.const(1, "uint64"), tvm.tir.const(2, "uint64"),
                      tvm.tir.const(3, "uint64"))
    assert call.op.same_as(tvm.ir.Op.get("tir.call_extern"))
    assert call.args[0].value == "rvt_fadd"

    dma = T.rvt_call("rvt_dma_hscatter", tvm.tir.const(1, "uint64"),
                     tvm.tir.const(2, "uint64"), tvm.tir.const(3, "uint64"),
                     tvm.tir.const(4, "uint64"), tvm.tir.const(5, "uint64"))
    assert dma.args[0].value == "rvt_dma_hscatter"


def test_rvt_lowering_keeps_explicit_vendor_calls():
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target="tpu",
        chip="sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    )
    source = artifact.kernel_source
    assert '#ifndef TILELANG_TPU_RV' in source
    assert '#error "RVT externs require TPU device_mode=rv"' in source
    assert '#include "rvt_api.h"' in source
    assert "rvt_kernel_start()" in source
    assert "rvt_fadd((uint64_t)10, (uint64_t)8, (uint64_t)9)" in source
    assert "rvt_sync_all()" in source

    atomic_source = tilelang.lower(
        _rvt_codegen_primfunc,
        target="tpu",
        chip="sg2260e",
        device_mode="atomic",
        runtime_mode="cmodel",
    ).kernel_source
    assert '#error "RVT externs require TPU device_mode=rv"' in atomic_source


def test_rvt_preamble_survives_a_later_plain_primfunc():
    """The RVT include/guard belongs to the whole codegen module, not one func."""
    module = tvm.IRModule({
        "rvt_first": _rvt_smoke_primfunc,
        "plain_last": _plain_multifunc_primfunc,
    })
    source = tilelang.lower(
        module,
        target="tpu",
        chip="sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    ).kernel_source
    assert '#include "rvt_api.h"' in source
    assert '#error "RVT externs require TPU device_mode=rv"' in source
    assert "rvt_kernel_start()" in source


def test_local_sg2260e_ppl_rvt_header_if_sdk_is_configured():
    """A real SDK check is opt-in, so unit tests remain hermetic by default."""
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        pytest.skip("PPL_PROJECT_ROOT is not configured")
    layout = resolve_ppl_layout(ppl_root, "sg2260e")
    assert layout.require_rvt_api().name == "rvt_api.h"

    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig(
            chip="sg2260e", device_mode="rv", runtime_mode="cmodel"),
    )
    try:
        definitions, _ = generator._ppl_compile_flags(layout, ".", "rv")
        assert "-DTILELANG_TPU_RV" in definitions
    finally:
        generator.remove_lib()


def test_rvt_cmodel_compile_is_private_if_sdk_is_configured():
    """Compile/link the real RVT source without loading or dispatching it."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    config = TPUCompileConfig(
        chip="sg2260e", device_mode="rv", runtime_mode="cmodel")
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target="tpu",
        chip=config.chip,
        device_mode=config.device_mode,
        runtime_mode=config.runtime_mode,
    )
    generator = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)
    try:
        wrapper = TLWrapper(
            tvm.target.Target("tpu"), tpu_workspace_dir=generator.tpu_workspace_dir)
        wrapper.assign_optimized_module(tvm.IRModule({"rvt_codegen": _rvt_codegen_primfunc}))
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        kernel_path = workspace / "libkernel.so"
        main_path = workspace / "main.so"
        assert kernel_path.is_file() and main_path.is_file()
        assert b"rvt_fadd" in (workspace / "kernel.c").read_bytes()
        assert str(kernel_path).encode() in main_path.read_bytes()
        assert b'setenv("TPU_RT_CORE_NUM", TILELANG_TPU_CMODEL_CORE_NUM, 1)' in (
            workspace / "main.cpp").read_bytes()
    finally:
        generator.remove_lib()


def test_rvt_pcie_compile_is_private_without_loading_if_sdk_is_configured():
    """Cross-compile/link RVT without dlopen or a board dispatch."""
    if not os.environ.get("PPL_PROJECT_ROOT"):
        pytest.skip("PPL_PROJECT_ROOT is not configured")

    config = TPUCompileConfig(
        chip="sg2260e", device_mode="rv", runtime_mode="pcie")
    artifact = tilelang.lower(
        _rvt_codegen_primfunc,
        target="tpu",
        chip=config.chip,
        device_mode=config.device_mode,
        runtime_mode=config.runtime_mode,
    )
    generator = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)
    try:
        wrapper = TLWrapper(
            tvm.target.Target("tpu"), tpu_workspace_dir=generator.tpu_workspace_dir)
        wrapper.assign_optimized_module(tvm.IRModule({"rvt_codegen": _rvt_codegen_primfunc}))
        wrapper.assign_host_module(artifact.host_mod)
        wrapper.assign_device_module(artifact.device_mod)
        generator.update_lib_code(wrapper.wrap(artifact.kernel_source))
        generator.compile_lib(timeout=60)

        workspace = Path(generator.tpu_workspace_dir)
        kernel_path = workspace / "libkernel.so"
        main_path = workspace / "main.so"
        assert kernel_path.is_file() and main_path.is_file()
        assert b"rvt_fadd" in (workspace / "kernel.c").read_bytes()
        assert str(kernel_path).encode() in main_path.read_bytes()
    finally:
        generator.remove_lib()
