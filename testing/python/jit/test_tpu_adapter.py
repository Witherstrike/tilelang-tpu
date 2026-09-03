import ctypes
import os
from pathlib import Path

import pytest
import torch
import importlib

tpu_adapter = importlib.import_module("tilelang.jit.adapter.tpu")

from tilelang import tvm
import tilelang.language as T
from tilelang.engine.param import KernelParam
from tilelang.engine.tpu_config import TPUCompileConfig
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.ctypes.adapter import CtypesKernelAdapter
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
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


class _FakePPLLayout:

    def __init__(self, runtime_identity):
        self.runtime_identity = runtime_identity

    def runtime_identity_for(self, runtime_mode):
        return self.runtime_identity


_TEST_SDK_IDENTITY = ("/test/ppl", "/test/ppl/runtime", "/test/ppl/backend")


def _mark_verified_tpu_artifact(generator, lib_path="/tmp/not-loaded.so",
                                sdk_identity=_TEST_SDK_IDENTITY):
    """Model a private artifact emitted by ``LibraryGenerator.compile_lib``.

    Unit tests must not resolve a real PPL SDK just to test the Python-side
    dlopen safety gates.  This intentionally sets both pieces of provenance
    that a successful TPU compilation captures.
    """
    generator._ppl_layout = _FakePPLLayout(sdk_identity)
    generator.libpath = lib_path
    generator._tpu_compiled_libpath = os.path.realpath(lib_path)
    generator._tpu_compiled_runtime_identity = sdk_identity


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
        source=(
            "#define TILELANG_TPU_OPAQUE_RAW_RVT_ABI 1\n"
            "void opaque_rvt(void) { rvt_kernel_start(); }"
        ),
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


def test_tpu_runtime_profile_is_single_process_identity(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    sg = TPUCompileConfig("sg2260e", "tpukernel", "cmodel")
    bm = TPUCompileConfig("bm1690", "tpukernel", "cmodel")

    tpu_adapter.reserve_tpu_runtime_profile(
        sg, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    tpu_adapter.reserve_tpu_runtime_profile(
        sg, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_profile(
            bm, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_profile(
            TPUCompileConfig("sg2260e", "tpukernel", "pcie"), device_id=0,
            sdk_identity=_TEST_SDK_IDENTITY)
    with pytest.raises(RuntimeError, match="fresh process"):
        tpu_adapter.reserve_tpu_runtime_profile(
            sg, device_id=0,
            sdk_identity=("/test/another-ppl", "/test/another/runtime", "/test/another/backend"))


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


@pytest.mark.parametrize("device_id", [None, "", "-1", "not-a-device", str(2**31)])
def test_pcie_library_load_requires_a_valid_device_id_before_dlopen(monkeypatch, device_id):
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "pcie"),
    )
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        if device_id is None:
            monkeypatch.delenv("TILELANG_TPU_DEVICE_ID", raising=False)
        else:
            monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", device_id)
        with pytest.raises(RuntimeError, match="TILELANG_TPU_DEVICE_ID"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


def test_pcie_library_load_validates_device_id_before_calling_cdll(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "pcie"),
    )
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
        _mark_verified_tpu_artifact(generator)
        captured = {}

        class FakeLibrary:
            pass

        fake_library = FakeLibrary()

        def bind_device(device_id):
            captured["device_id"] = device_id
            return 0

        fake_library.tilelang_tpu_bind_device = bind_device

        def fake_cdll(path):
            captured["path"] = path
            return fake_library

        monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
        assert generator.load_lib("/tmp/not-loaded.so") is fake_library
        assert captured["path"] == "/tmp/not-loaded.so"
        assert captured["device_id"] == 0
    finally:
        generator.remove_lib()


def test_cmodel_library_load_binds_device_zero_before_returning(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "rv", "cmodel"),
    )
    try:
        _mark_verified_tpu_artifact(generator)
        captured = {}

        class FakeLibrary:
            pass

        fake_library = FakeLibrary()

        def bind_device(device_id):
            captured["device_id"] = device_id
            return 0

        fake_library.tilelang_tpu_bind_device = bind_device
        monkeypatch.setattr(ctypes, "CDLL", lambda path: fake_library)
        assert generator.load_lib("/tmp/not-loaded.so") is fake_library
        assert captured["device_id"] == 0
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_a_host_library_that_rejects_device_binding(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "cmodel"),
    )
    try:
        _mark_verified_tpu_artifact(generator)
        class FakeLibrary:
            pass

        fake_library = FakeLibrary()
        fake_library.tilelang_tpu_bind_device = lambda _device_id: -3
        monkeypatch.setattr(ctypes, "CDLL", lambda path: fake_library)
        with pytest.raises(RuntimeError, match="rejected the reserved runtime device"):
            generator.load_lib("/tmp/not-loaded.so")
    finally:
        generator.remove_lib()


def test_pcie_load_rejects_a_cmodel_runtime_profile_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    tpu_adapter.reserve_tpu_runtime_profile(
        TPUCompileConfig("sg2260e", "tpukernel", "cmodel"), device_id=0,
        sdk_identity=_TEST_SDK_IDENTITY)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "pcie"),
    )
    try:
        monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
        monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")
        _mark_verified_tpu_artifact(generator)
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_cmodel_load_rejects_a_pcie_runtime_profile_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    tpu_adapter.reserve_tpu_runtime_profile(
        TPUCompileConfig("sg2260e", "tpukernel", "pcie"), device_id=0,
        sdk_identity=_TEST_SDK_IDENTITY)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "cmodel"),
    )
    try:
        _mark_verified_tpu_artifact(generator)
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_sdk_identity_transition_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    config = TPUCompileConfig("sg2260e", "tpukernel", "cmodel")
    tpu_adapter.reserve_tpu_runtime_profile(
        config, device_id=0, sdk_identity=_TEST_SDK_IDENTITY)
    generator = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)
    try:
        _mark_verified_tpu_artifact(
            generator,
            sdk_identity=("/test/other-ppl", "/test/other/runtime", "/test/other/backend"))
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="fresh process"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_unverified_prebuilt_artifact_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "cmodel"),
    )
    try:
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="verified manifest"):
            generator.load_lib("/tmp/not-loaded.so")
        assert loaded == []
    finally:
        generator.remove_lib()


def test_tpu_load_rejects_another_path_from_the_same_generator_before_dlopen(monkeypatch):
    monkeypatch.setattr(tpu_adapter, "_TPU_RUNTIME_PROFILE", None)
    generator = LibraryGenerator(
        tvm.target.Target("tpu"),
        tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "cmodel"),
    )
    try:
        _mark_verified_tpu_artifact(generator, lib_path="/tmp/compiled-main.so")
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))
        with pytest.raises(RuntimeError, match="not produced by this LibraryGenerator"):
            generator.load_lib("/tmp/another-prebuilt-main.so")
        assert loaded == []
    finally:
        generator.remove_lib()


@pytest.mark.parametrize("adapter_cls", [CtypesKernelAdapter, CythonKernelAdapter])
def test_tpu_database_adapters_reject_prebuilt_artifacts_before_dlopen(monkeypatch, adapter_cls):
    raw_prim_func = tvm.tir.PrimFunc(
        [], tvm.tir.Evaluate(tvm.tir.const(0, "int32")),
    ).with_attr("global_symbol", "cached_tpu_artifact")
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path: loaded.append(path))

    with pytest.raises(RuntimeError, match="manifest"):
        adapter_cls.from_database(
            params=[],
            result_idx=[],
            target="tpu -mcpu=sg2260e",
            func_or_mod=raw_prim_func,
            kernel_global_source="",
            kernel_lib_path="/tmp/prebuilt-tpu-main.so",
            tpu_config=TPUCompileConfig("sg2260e", "tpukernel", "cmodel"),
        )
    assert loaded == []


def test_tpu_database_artifact_loading_is_disabled_without_manifest():
    with pytest.raises(RuntimeError, match="manifest"):
        reject_unverified_tpu_database_artifact(tvm.target.Target("tpu"))
