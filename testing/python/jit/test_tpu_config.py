import pytest
import importlib
from types import SimpleNamespace
import tilelang
from tilelang import tvm

jit_api = importlib.import_module("tilelang.jit")
libgen_module = importlib.import_module("tilelang.jit.adapter.libgen")
TPU_TARGET = SimpleNamespace(kind=SimpleNamespace(name="tpu"))

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


def test_rv_library_generator_dispatches_to_sg2260e_cmodel(monkeypatch, tmp_path):
    config = TPUCompileConfig("sg2260e", "rv", "cmodel")
    generator = LibraryGenerator(TPU_TARGET, tpu_config=config)
    captured = {}

    monkeypatch.setenv("PPL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(libgen_module, "resolve_ppl_layout",
                        lambda root, chip: (root, chip))
    monkeypatch.setattr(
        generator,
        "tpu_compile_cmodel",
        lambda timeout, layout: captured.update(timeout=timeout, layout=layout),
    )

    generator.compile_lib(timeout=17)

    assert captured == {
        "timeout": 17,
        "layout": (str(tmp_path), "sg2260e"),
    }


def test_rv_library_generator_preserves_structured_pipeline_calls(tmp_path):
    config = TPUCompileConfig("sg2260e", "rv", "cmodel")
    generator = LibraryGenerator(TPU_TARGET, tpu_config=config)
    source = tmp_path / "kernel.c"
    original = "void kernel(void) { tpu_parallel_start(); tpu_parallel_end(); }\n"
    source.write_text(original, encoding="utf-8")

    generator._prepare_cmodel_kernel_source(str(source))

    assert source.read_text(encoding="utf-8") == original
