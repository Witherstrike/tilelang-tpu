import ctypes
from pathlib import Path

import pytest
import torch

from tilelang import tvm
import tilelang.language as T
from tilelang.engine.param import KernelParam
from tilelang.engine.tpu_config import TPUCompileConfig
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.tpu import (
    make_tpu_forward,
    reject_unverified_tpu_database_artifact,
)
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper


class _FakeRun:

    def __init__(self, status=0):
        self.status = status
        self.input_values = None

    def __call__(self, argv):
        self.input_values = list((ctypes.c_float * 3).from_address(argv[0]))
        if self.status == 0:
            ctypes.memmove(argv[1], argv[0], 3 * ctypes.sizeof(ctypes.c_float))
        return self.status


class _FakeLibrary:

    def __init__(self, status=0):
        self.tilelang_tpu_run = _FakeRun(status)


def _two_tensor_params():
    return [
        KernelParam(torch.float32, [3]),
        KernelParam(torch.float32, [3]),
    ]


@T.prim_func
def _two_buffer_wrapper_primfunc(
    A: T.Tensor((1,), "float32"),
    B: T.Tensor((1,), "float32"),
):
    T.func_attr({"global_symbol": "opaque_rvt", "tir.noalias": T.bool(True)})
    B[0] = A[0]


def test_tpu_forward_uses_data_ptr_and_allocates_output():
    lib = _FakeLibrary()
    forward = make_tpu_forward(lib, _two_tensor_params(), [1], {})
    base = torch.arange(10, dtype=torch.float32)
    source = base[2:5]  # contiguous, but its storage offset is non-zero

    result = forward(source)

    assert lib.tilelang_tpu_run.input_values == [2.0, 3.0, 4.0]
    assert torch.equal(result, source)


def test_tpu_forward_does_not_copy_outputs_after_failure():
    lib = _FakeLibrary(status=-7)
    forward = make_tpu_forward(lib, _two_tensor_params(), [1], {})
    output = torch.full((3,), -1.0)

    with pytest.raises(RuntimeError, match="status -7"):
        forward(torch.arange(3, dtype=torch.float32), output)

    assert torch.equal(output, torch.full((3,), -1.0))


def test_tpu_forward_copies_explicit_buffers_for_opaque_externs():
    """An opaque extern may mutate an argument without yielding result_idx."""
    lib = _FakeLibrary()
    forward = make_tpu_forward(lib, _two_tensor_params(), [], {})
    source = torch.tensor([4.0, 5.0, 6.0])
    destination = torch.full((3,), -1.0)

    assert forward(source, destination) == 0
    assert torch.equal(destination, source)


def test_tpu_forward_rejects_aliased_storage():
    forward = make_tpu_forward(_FakeLibrary(), _two_tensor_params(), [], {})
    storage = torch.arange(6, dtype=torch.float32)

    with pytest.raises(ValueError, match="aliased Tensor storage"):
        forward(storage[:3], storage[3:])


def test_raw_rvt_wrapper_copies_all_buffers_even_with_result_idx(tmp_path):
    """Raw RVT pointer effects are opaque to TIR's result-index analysis."""
    TLTPUSourceWrapper(
        scheduled_ir_module=tvm.IRModule({"opaque_rvt": _two_buffer_wrapper_primfunc}),
        source="void opaque_rvt(void) { rvt_kernel_start(); }",
        target=tvm.target.Target("tpu"),
        output_indices=[1],
        output_dir=str(tmp_path),
    )
    main_source = (tmp_path / "main.cpp").read_text(encoding="utf-8")
    assert main_source.count("tpuRtMemcpyD2S(") == 2


def test_tpu_workspace_is_private_per_generator():
    config = TPUCompileConfig("sg2260e", "rv", "cmodel")
    first = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)
    second = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)
    try:
        assert first.tpu_workspace_dir != second.tpu_workspace_dir
        assert Path(first.tpu_workspace_dir).is_dir()
        assert Path(second.tpu_workspace_dir).is_dir()
    finally:
        first.remove_lib()
        second.remove_lib()


def test_pcie_library_load_is_fail_closed(monkeypatch):
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "rv", "pcie"),
    )
    try:
        monkeypatch.delenv("TILELANG_TPU_ALLOW_PCIE_LOAD", raising=False)
        with pytest.raises(RuntimeError, match="TILELANG_TPU_ALLOW_PCIE_LOAD=1"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


def test_tpu_database_artifact_loading_is_disabled_without_manifest():
    with pytest.raises(RuntimeError, match="manifest"):
        reject_unverified_tpu_database_artifact(tvm.target.Target("tpu"))
