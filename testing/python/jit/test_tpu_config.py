import pytest
import importlib
from types import SimpleNamespace
import tilelang
from tilelang import tvm

kernel_module = importlib.import_module("tilelang.jit.kernel")

jit_api = importlib.import_module("tilelang.jit")
target_utils = importlib.import_module("tilelang.utils.target")
lower_module = importlib.import_module("tilelang.engine.lower")

from tilelang.engine.tpu_config import (
    TPUCompileConfig,
    bind_tpu_target,
    get_tpu_chip_spec,
    get_tpu_target_chip,
    resolve_tpu_compile_config,
)
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.kernel import JITKernel
from tilelang.cache.kernel_cache import KernelCache


def test_auto_target_never_implicitly_selects_bm1690_pcie(monkeypatch):
    monkeypatch.setattr(target_utils, "check_cuda_availability", lambda: False)
    monkeypatch.setattr(target_utils, "check_hip_availability", lambda: False)
    monkeypatch.setattr(target_utils, "_configured_tpu_auto_target", lambda: None)
    assert target_utils.determine_target("auto") == "c"

    monkeypatch.setattr(
        target_utils, "_configured_tpu_auto_target", lambda: "tpu -mcpu=sg2260e")
    assert target_utils.determine_target("auto") == "tpu -mcpu=sg2260e"


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


def test_atomic_is_a_deprecated_alias_for_tpukernel():
    with pytest.warns(DeprecationWarning, match="device_mode='atomic'"):
        config = resolve_tpu_compile_config(
            chip="sg2260e", device_mode="atomic", runtime_mode="cmodel")

    assert config.device_mode == "tpukernel"
    assert config.programming_model == "tpukernel"


def test_tpu_chip_capabilities_are_explicit_and_fail_closed():
    sg = get_tpu_chip_spec("SG2260E")
    assert sg.ppl_arch == "tpub_7_1_e"
    assert sg.ppl_compile_definitions == ("__tpub_7_1_e__", "__sg2260e__")
    assert sg.physical_core_count == 4
    assert sg.programming_models == ("tpukernel", "rv")

    bm = get_tpu_chip_spec("bm1690")
    assert bm.ppl_arch == "tpub_7_1"
    assert bm.ppl_compile_definitions == ("__tpub_7_1__", "__sg2260__")
    assert bm.physical_core_count == 8
    assert bm.programming_models == ("tpukernel",)

    with pytest.raises(ValueError, match="does not support device_mode='rv'"):
        resolve_tpu_compile_config(chip="bm1690", device_mode="rv")
    with pytest.raises(ValueError, match="Unsupported TPU chip"):
        resolve_tpu_compile_config(chip="sg2260erv")
    with pytest.raises(ValueError, match="legacy target model"):
        get_tpu_target_chip(tvm.target.Target("tpu -model=sg2260erv"))


def test_rv_defaults_to_cmodel_until_pcie_is_explicit():
    assert resolve_tpu_compile_config(
        chip="sg2260e", device_mode="rv") == TPUCompileConfig(
            "sg2260e", "rv", "cmodel")
    assert resolve_tpu_compile_config(
        chip="sg2260e", device_mode="rv", runtime_mode="pcie") == TPUCompileConfig(
            "sg2260e", "rv", "pcie")


def test_sg_tpukernel_defaults_to_cmodel_but_bm_keeps_legacy_pcie_default():
    assert resolve_tpu_compile_config(chip="sg2260e") == TPUCompileConfig(
        "sg2260e", "tpukernel", "cmodel")
    assert resolve_tpu_compile_config(chip="bm1690") == TPUCompileConfig(
        "bm1690", "tpukernel", "pcie")
    assert TPUCompileConfig("sg2260e").runtime_mode == "cmodel"


def test_tpu_target_chip_is_canonical_mcpu_and_conflicts_are_rejected():
    sg_target = tvm.target.Target("tpu -mcpu=sg2260e")
    assert get_tpu_target_chip(sg_target) == "sg2260e"
    config = resolve_tpu_compile_config(
        device_mode="tpukernel", target_chip=get_tpu_target_chip(sg_target))
    assert config == TPUCompileConfig("sg2260e", "tpukernel", "cmodel")
    bound_sg_target = bind_tpu_target(sg_target, config)
    assert bound_sg_target.mcpu == "sg2260e"
    assert bound_sg_target.attrs["tpu-programming-model"] == "tpukernel"

    legacy_target = tvm.target.Target("tpu -model=sg2260e")
    canonical = bind_tpu_target(
        legacy_target,
        resolve_tpu_compile_config(target_chip=get_tpu_target_chip(legacy_target)),
    )
    assert canonical.mcpu == "sg2260e"
    assert canonical.model == "unknown"

    with pytest.raises(ValueError, match="Conflicting TPU chip selections"):
        resolve_tpu_compile_config(chip="bm1690", target_chip="sg2260e")
    with pytest.raises(ValueError, match="Conflicting TPU chip target attributes"):
        get_tpu_target_chip(tvm.target.Target({
            "kind": "tpu", "mcpu": "bm1690", "model": "sg2260e"}))
    with pytest.raises(ValueError, match="legacy target model"):
        get_tpu_target_chip(tvm.target.Target({
            "kind": "tpu", "mcpu": "sg2260e", "model": "sg2260erv"}))
    with pytest.raises(ValueError, match="Conflicting TPU programming-model"):
        bind_tpu_target(
            tvm.target.Target({
                "kind": "tpu", "mcpu": "sg2260e", "tpu-programming-model": "rv"}),
            config,
        )

    workload_target = tvm.target.Target({
        "kind": "tpu", "mcpu": "sg2260e", "model": "matmul_smoke"})
    assert get_tpu_target_chip(workload_target) == "sg2260e"
    assert bind_tpu_target(workload_target, config).model == "matmul_smoke"


def test_tpu_config_rejects_conflicting_runtime_aliases():
    with pytest.raises(ValueError, match="Conflicting TPU runtime modes"):
        resolve_tpu_compile_config(runtime_mode="cmodel", mode="pcie")


def test_future_tpu_cache_key_normalizes_legacy_mode_and_target_spelling():
    cache = KernelCache()
    with pytest.warns(DeprecationWarning, match="device_mode='atomic'"):
        legacy_config, legacy_target = cache._canonical_tpu_cache_selection(
            "tpu -model=sg2260e", None, "atomic", None, mode="cmodel")
    canonical_config, canonical_target = cache._canonical_tpu_cache_selection(
        "tpu -mcpu=sg2260e", None, "tpukernel", "cmodel")

    assert legacy_config == canonical_config == TPUCompileConfig(
        "sg2260e", "tpukernel", "cmodel")
    assert legacy_target == canonical_target


def test_tpu_persistent_cache_save_and_load_are_disabled_without_a_manifest():
    cache = KernelCache()
    tpu_kernel = SimpleNamespace(target=tvm.target.Target("tpu -mcpu=sg2260e"))

    with pytest.raises(RuntimeError, match="manifest"):
        cache._save_kernel_to_disk("unsafe-tpu-artifact", tpu_kernel)
    with pytest.raises(RuntimeError, match="manifest"):
        cache._load_kernel_from_disk(
            "unsafe-tpu-artifact", target="tpu -mcpu=sg2260e")


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


def test_jit_kernel_reads_chip_from_tpu_target_without_legacy_chip_argument():
    kernel = JITKernel(
        target="tpu -mcpu=sg2260e",
        from_database=True,
        device_mode="tpukernel",
    )

    assert kernel.tpu_config == TPUCompileConfig("sg2260e", "tpukernel", "cmodel")
    assert kernel.target.mcpu == "sg2260e"


def test_non_tpu_jit_does_not_validate_or_store_tpu_configuration():
    kernel = JITKernel(
        target="llvm",
        execution_backend="ctypes",
        from_database=True,
        chip="not-a-tpu-chip",
        device_mode="rv",
        runtime_mode="cmodel",
    )

    assert kernel.target.kind.name == "llvm"
    assert kernel.tpu_config is None
    assert kernel.mode is None


def test_lower_keeps_non_tpu_targets_on_the_regular_codegen_path():
    """TPU configuration must not turn a normal C target into PPL lowering."""
    A = tvm.tir.decl_buffer((4,), "float32", name="A")
    B = tvm.tir.decl_buffer((4,), "float32", name="B")
    i = tvm.tir.Var("i", "int32")
    body = tvm.tir.For(
        i, 0, 4, tvm.tir.ForKind.SERIAL,
        tvm.tir.BufferStore(B, tvm.tir.BufferLoad(A, [i]), [i]))
    prim_func = tvm.tir.PrimFunc(
        [A.data, B.data], body, buffer_map={A.data: A, B.data: B},
    ).with_attr("global_symbol", "cpu_copy")

    artifact = tilelang.lower(prim_func, target="c")

    assert artifact.tpu_config is None
    assert "cpu_copy" in artifact.kernel_source


@pytest.mark.parametrize("extern_name", ["ppl.fill", "tpu_sync_all_bdc", "rvt_fadd"])
def test_tpu_externs_cannot_silently_lower_for_a_non_tpu_target(extern_name):
    prim_func = tvm.tir.PrimFunc(
        [], tvm.tir.Evaluate(tvm.tir.call_extern("handle", extern_name)),
    ).with_attr("global_symbol", "must_select_tpu")

    with pytest.raises(ValueError, match="explicit TPU target"):
        tilelang.lower(prim_func, target="c")


def test_tpu_externs_cannot_silently_lower_after_auto_selects_c(monkeypatch):
    prim_func = tvm.tir.PrimFunc(
        [], tvm.tir.Evaluate(tvm.tir.call_extern("handle", "ppl.fill")),
    ).with_attr("global_symbol", "auto_must_select_tpu")
    monkeypatch.setattr(
        lower_module, "determine_target", lambda target: tvm.target.Target("c"))

    with pytest.raises(ValueError, match="explicit TPU target"):
        tilelang.lower(prim_func, target="auto")


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
