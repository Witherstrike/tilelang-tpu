import pytest
import importlib
import tilelang
from tilelang import tvm

kernel_module = importlib.import_module("tilelang.jit.kernel")

jit_api = importlib.import_module("tilelang.jit")

from tilelang.engine.tpu_config import (
    TPUCompileConfig,
    resolve_tpu_compile_config,
)
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.kernel import JITKernel


def test_compile_forwards_explicit_tpu_configuration(monkeypatch):
    captured = {}

    def fake_cached(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(jit_api, "cached", fake_cached)
    jit_api.compile(
        func=object(),
        target="tpu",
        chip="sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    )

    assert captured["chip"] == "sg2260e"
    assert captured["device_mode"] == "rv"
    assert captured["runtime_mode"] == "cmodel"
    assert captured["mode"] is None


def test_tpu_config_normalizes_chip_and_legacy_mode():
    config = resolve_tpu_compile_config(
        chip="SG2260E", device_mode="rv", mode="cmodel")

    assert config == TPUCompileConfig(
        chip="sg2260e", device_mode="rv", runtime_mode="cmodel")


def test_rv_defaults_to_cmodel_until_pcie_is_explicit():
    assert resolve_tpu_compile_config(
        chip="sg2260e", device_mode="rv") == TPUCompileConfig(
            "sg2260e", "rv", "cmodel")
    assert resolve_tpu_compile_config(
        chip="sg2260e", device_mode="rv", runtime_mode="pcie") == TPUCompileConfig(
            "sg2260e", "rv", "pcie")


def test_tpu_config_rejects_conflicting_runtime_aliases():
    with pytest.raises(ValueError, match="Conflicting TPU runtime modes"):
        resolve_tpu_compile_config(runtime_mode="cmodel", mode="pcie")


@pytest.mark.parametrize("field,value", [
    ("device_mode", "invalid"),
    ("runtime_mode", "soc"),
])
def test_tpu_config_rejects_unknown_modes(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match="Unsupported TPU"):
        resolve_tpu_compile_config(**kwargs)


def test_jit_kernel_receives_tpu_config_without_compiling():
    kernel = JITKernel(
        target="tpu",
        from_database=True,
        chip="sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    )

    assert kernel.tpu_config == TPUCompileConfig("sg2260e", "rv", "cmodel")
    assert kernel.mode == "cmodel"


def test_library_generator_receives_tpu_config():
    config = TPUCompileConfig("sg2260e", "rv", "pcie")
    generator = LibraryGenerator(tvm.target.Target("tpu"), tpu_config=config)

    assert generator.tpu_config is config
    assert generator.mode == "pcie"


@pytest.mark.parametrize("backend,adapter_name", [
    ("ctypes", "CtypesKernelAdapter"),
    ("cython", "CythonKernelAdapter"),
])
def test_database_adapter_forwards_tpu_config(monkeypatch, backend, adapter_name):
    """Cached TPU artifacts must retain the chip/device/runtime tuple."""
    captured = {}

    class FakeAdapter:

        @classmethod
        def from_database(cls, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(kernel_module, adapter_name, FakeAdapter)
    kernel = JITKernel(
        target="tpu",
        execution_backend=backend,
        from_database=True,
        chip="sg2260e",
        device_mode="rv",
        runtime_mode="cmodel",
    )

    adapter = kernel._create_adapter_from_database(
        params=[],
        result_idx=[],
        target="tpu",
        func_or_mod=object(),
        kernel_global_source="",
        kernel_lib_path="/tmp/tilelang-unused.so",
    )

    assert adapter is not None
    assert captured["tpu_config"] == TPUCompileConfig("sg2260e", "rv", "cmodel")
