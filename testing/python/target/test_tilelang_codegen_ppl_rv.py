import ctypes
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tvm
from tvm.script import from_source

import tilelang
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper
from tilelang.engine.tpu_config import TPUCompileConfig


SOURCE = r'''
@T.prim_func
def main_kernel_inner(A: T.Buffer((1, 8, 1, 16), "float16"),
         B: T.Buffer((1, 8, 1, 16), "float16"),
         C: T.Buffer((1, 8, 1, 16), "float16")):
    L = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    R = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    O = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), R.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.add", T.tvm_access_ptr(T.type_annotation("float16"), O.data, 0, 128, 2), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), R.data, 0, 128, 1)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), O.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 128, 2)))
'''


def _module(source=SOURCE):
    return tvm.IRModule({"main_kernel_inner": from_source(source, {"T": tvm.script.tir})})


def _rv_source(source=SOURCE):
    mod = _rv_module(source)
    return tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)


def _rv_module(source=SOURCE):
    mod = tvm.tir.transform.LowerOpaqueBlock()(_module(source))
    func = mod["main_kernel_inner"]
    func = func.with_attr("tir.tpu.chip", "sg2260e")
    func = func.with_attr("tir.tpu.device_mode", "rv")
    mod = tvm.IRModule({"main_kernel_inner": func})
    mod = tilelang.transform.AddressAssign()(mod)
    return tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")(mod)


def _arguments(text):
    return [re.sub(r"/\*.*?\*/", "", item).strip() for item in text.split(",")]


def _constant(text):
    match = re.fullmatch(r"v\d+_(-?\d+)", text)
    return match.group(1) if match else text


def _parse_rv_c(source):
    """Return register-ID-independent configuration and instruction graphs."""
    registers = {}
    for match in re.finditer(r"RVT_CFG(GR|TR)\s*\(([^;]+?)\);", source):
        args = _arguments(match.group(2))
        registers[args[0]] = {
            "class": match.group(1), "dtype": args[2], "layout": args[4]
        }
    for match in re.finditer(r"RVT_CFG(GR_REG|TR)_SHAPE\s*\(([^;]+?)\);", source):
        args = _arguments(match.group(2))
        if args[0] in registers:
            registers[args[0]]["shape"] = tuple(_constant(arg) for arg in args[1:5])

    physical_to_canonical = {}
    class_counts = {"GR": 0, "TR": 0}

    def canonical(reg):
        if reg not in physical_to_canonical:
            register_class = registers[reg]["class"]
            physical_to_canonical[reg] = f"{register_class}{class_counts[register_class]}"
            class_counts[register_class] += 1
        return physical_to_canonical[reg]

    operations = []
    for match in re.finditer(r"\b(rvt_(?:dma_ld|dma_st|fadd))\s*\(([^)]+)\)", source):
        operands = tuple(canonical(arg) for arg in _arguments(match.group(2)))
        operations.append((match.group(1), operands))
    specs = {
        physical_to_canonical[reg]: (
            spec["class"], spec["dtype"], spec["layout"], spec.get("shape"))
        for reg, spec in registers.items() if reg in physical_to_canonical
    }
    return specs, operations


def _golden_source(case):
    path = (Path(__file__).resolve().parents[3] / "testing" / "ppl_rv_golden" /
            "artifacts" / case / "device" / f"{case}.c")
    return path.read_text(encoding="utf-8")


def _instruction_class_signature(source, instruction):
    """Canonicalize tensor operands to GR/TR and all control operands to S."""
    classes = {}
    for match in re.finditer(r"RVT_CFG(GR|TR)\s*\(([^;]+?)\);", source):
        classes[_arguments(match.group(2))[0]] = match.group(1)
    match = re.search(rf"\b{instruction}\s*\(([^;]+?)\);", source)
    assert match, f"missing {instruction}"
    return tuple(classes.get(arg, "S") for arg in _arguments(match.group(1)))


def test_copy_add_matches_golden_instruction_structure():
    source = _rv_source()
    copy_golden = _golden_source("copy")
    add_golden = _golden_source("elementwise")
    copy_specs, copy_ops = _parse_rv_c(copy_golden)
    add_specs, add_ops = _parse_rv_c(add_golden)
    source_specs, source_ops = _parse_rv_c(source)
    expected_names = [copy_ops[0][0], copy_ops[0][0]]
    expected_names += [name for name, _ in add_ops if name == "rvt_fadd"]
    expected_names += [copy_ops[-1][0]]
    expected_ops = [
        (expected_names[0], ("TR0", "GR0")),
        (expected_names[1], ("TR1", "GR1")),
        (expected_names[2], ("TR2", "TR0", "TR1")),
        (expected_names[3], ("GR2", "TR2")),
    ]
    assert source_ops == expected_ops
    golden_specs = set(copy_specs.values()) | set(add_specs.values())
    assert set(source_specs.values()).issubset(golden_specs)
    assert len(re.findall(r"\bRVT_CFGGR\s*\(", source)) == 3
    assert len(re.findall(r"\bRVT_CFGTR\s*\(", source)) == 3
    for layout in ("CONTINUOUS_LAYOUT", "HW_ALIGN_LAYOUT"):
        assert layout in copy_golden
        assert layout in add_golden
        assert source.count(layout) >= 3
    assert len(re.findall(r"RVT_CFG(?:GR_REG|TR)_SHAPE\([^;]*, 1, 8, 1, 16\)", source)) == 6
    assert "rvt_cfg_satu(0, 0)" in source
    assert "rvt_cfg_round_mode(0)" in source

    # Physical register IDs are intentionally not compared.  PPL can permute
    # them between cmodel and PCIe while preserving this operation/dataflow.
    assert "tpu_bdc_" not in source
    assert "tpu_gdma_" not in source

    # The public launch ABI and synchronization sequence must match PPL's C,
    # while the kernel-specific symbol name is intentionally unconstrained.
    for token in ("_entry(const void *", "rvt_kernel_start()",
                  "rvt_sync_i(0xdeadbeef, 0)"):
        assert token in copy_golden
        assert token in add_golden
        assert token in source


def test_rv_source_is_valid_c_with_ppl_17_sg2260e_headers(tmp_path):
    ppl_root = Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528"
    if not ppl_root.is_dir():
        pytest.skip("PPL 1.7 SG2260E SDK is not available")
    layout = resolve_ppl_layout(str(ppl_root), "sg2260e")
    path = tmp_path / "copy_add.c"
    path.write_text(_rv_source(), encoding="utf-8")
    command = [
        "cc", "-fsyntax-only", "-std=c11",
        *(f"-D{definition}" for definition in layout.compile_definitions),
        "-DTILELANG_PPL_HELPER_HAS_GET_DTYPE",
        *(f"-I{include}" for include in layout.include_dirs),
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_structured_pipeline_markers_are_emitted_in_order():
    mod = _rv_module()
    func = mod["main_kernel_inner"]
    marker_node = tvm.tir.IntImm("int32", 0)
    start = tvm.tir.AttrStmt(
        marker_node, "tpu_parallel_start", 0, tvm.tir.Evaluate(0))
    end = tvm.tir.AttrStmt(
        marker_node, "tpu_parallel_end", 0, tvm.tir.Evaluate(0))
    body = tvm.tir.SeqStmt([start, func.body, end])
    generated = tvm.get_global_func("target.build.tilelang_ppl_rv")(
        tvm.IRModule({"main_kernel_inner": func.with_body(body)}))

    assert generated.count("tpu_parallel_start();") == 1
    assert generated.count("tpu_parallel_end();") == 1
    assert generated.index("tpu_parallel_start();") < generated.index("rvt_dma_ld(")
    assert generated.index("rvt_dma_st(") < generated.index("tpu_parallel_end();")


def _run_cmodel(source, inputs, expected, output_initial=None, pipeline=True,
                atol=0):
    """Compile and execute an RV kernel on the SG2260E cmodel."""
    if os.environ.get("TILELANG_RUN_TPU_CMODEL_TESTS") != "1":
        pytest.skip("set TILELANG_RUN_TPU_CMODEL_TESTS=1 to run the TPU emulator")
    ppl_root = Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528"
    if not ppl_root.is_dir():
        pytest.skip("PPL 1.7 SG2260E SDK is not available")

    mod = _rv_module(source)
    func = mod["main_kernel_inner"]
    marker_node = tvm.tir.IntImm("int32", 0)
    start = tvm.tir.AttrStmt(
        marker_node, "tpu_parallel_start", 0, tvm.tir.Evaluate(0))
    end = tvm.tir.AttrStmt(
        marker_node, "tpu_parallel_end", 0, tvm.tir.Evaluate(0))
    if pipeline:
        mod = tvm.IRModule({
            "main_kernel_inner": func.with_body(tvm.tir.SeqStmt([start, func.body, end]))
        })
    generated = tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)
    target = SimpleNamespace(kind=SimpleNamespace(name="tpu"))
    source_mod = _module(source)
    output_index = len(source_mod["main_kernel_inner"].params) - 1
    TLTPUSourceWrapper(source_mod, generated, target, output_indices=[output_index])
    config = TPUCompileConfig("sg2260e", "rv", "cmodel")
    generator = LibraryGenerator(target, tpu_config=config)
    generator.update_lib_code(generated)
    generator.compile_lib(timeout=120)

    library = ctypes.CDLL(generator.libpath)
    library.tilelang_tpu_run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.tilelang_tpu_run.restype = ctypes.c_int
    output = (np.zeros_like(expected) if output_initial is None else
              np.array(output_initial, copy=True))
    arrays = [*inputs, output]
    arguments = (ctypes.c_void_p * len(arrays))(
        *(array.ctypes.data for array in arrays))
    assert library.tilelang_tpu_run(arguments) == 0
    np.testing.assert_allclose(output, expected, rtol=0, atol=atol)


def test_pipeline_copy_add_cmodel_numerics(monkeypatch):
    monkeypatch.setenv(
        "PPL_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528"))
    rng = np.random.default_rng(20260805)
    lhs = rng.normal(size=(1, 8, 1, 16)).astype(np.float16)
    rhs = rng.normal(size=lhs.shape).astype(np.float16)
    _run_cmodel(SOURCE, [lhs, rhs], (lhs + rhs).astype(np.float16))


def test_same_global_buffer_subviews_keep_distinct_offsets():
    source = SOURCE.replace(
        "A.data, 0, 128, 1", "A.data, 16, 112, 1", 1).replace(
            "L.data, 0, 128, 2", "L.data, 16, 112, 2", 1).replace(
                "B.data, 0, 128, 1", "A.data, 0, 128, 1", 1)
    generated = _rv_source(source)
    assert "+ 16 * 2" in generated
    assert "+ 0 * 2" in generated
    global_configs = re.findall(r"RVT_CFGGR\([^;]+", generated)
    assert any("+ 16 * 2" in config for config in global_configs)
    assert any("+ 0 * 2" in config for config in global_configs)


def test_local_free_layout_subviews_reconfigure_one_register_and_stride():
    source = r'''
@T.prim_func
def main_kernel_inner(A: T.Buffer((256,), "float16")):
    L = T.alloc_buffer((256,), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 112, 1), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 112, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 16, 112, 1), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 16, 112, 2)))
'''
    generated = _rv_source(source)
    configs = [_arguments(match.group(1)) for match in re.finditer(
        r"RVT_CFGTR\s*\(([^;]+?)\);", generated)]
    assert len(configs) == 2
    assert len({config[0] for config in configs}) == 1
    assert any("+ 0 * 2" in config[5] for config in configs)
    assert any("+ 16 * 2" in config[5] for config in configs)
    assert generated.count("RVT_TR_STRIDE(") == 2
    assert all(config[4] == "FREE_LAYOUT" for config in configs)


@pytest.mark.parametrize(
    ("argument", "value", "diagnostic"),
    [
        (3, 0, "requires GR or TR register class"),
        (8, 64, "Unsupported SG2260E RV dtype"),
    ],
)
def test_invalid_tensor_view_schema_fails_at_compile_time(argument, value, diagnostic):
    mod = _rv_module()
    func = mod["main_kernel_inner"]
    changed = False

    def rewrite(node):
        nonlocal changed
        if changed or not isinstance(node, tvm.tir.Call):
            return None
        if node.op.name != "tir.call_extern" or not node.args:
            return None
        name = node.args[0]
        if not isinstance(name, tvm.tir.StringImm) or name.value != "ppl.rv.tensor_view":
            return None
        args = list(node.args)
        args[argument] = tvm.tir.IntImm("int32", value)
        changed = True
        return tvm.tir.call_extern(node.dtype, name.value, *args[1:])

    body = tvm.tir.stmt_functor.ir_transform(func.body, None, rewrite, ["tir.Call"])
    assert changed
    mod = tvm.IRModule({"main_kernel_inner": func.with_body(body)})
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)


def test_atomic_codegen_remains_separate_from_rv_codegen():
    empty = r'''
@T.prim_func
def atomic_kernel_inner(A: T.Buffer((1,), "float16")):
    T.evaluate(0)
'''
    mod = tvm.IRModule({
        "atomic_kernel_inner": from_source(empty, {"T": tvm.script.tir})
    })
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    source = tvm.get_global_func("target.build.tilelang_ppl")(mod)
    assert "rvt_" not in source
    assert "RVT_CFG" not in source
    assert "atomic_kernel_inner" in source


def test_topk_fails_at_the_documented_ppl_boundary():
    source = SOURCE.replace('"ppl.add"', '"ppl.topk"')
    mod = tvm.tir.transform.LowerOpaqueBlock()(_module(source))
    mod = tilelang.transform.AddressAssign()(mod)
    mod = tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")(mod)
    with pytest.raises(tvm.error.TVMError, match="PPL 1.7 SG2260E RV lowering"):
        tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)


def test_unknown_rv_operation_fails_without_atomic_fallback():
    source = r'''
@T.prim_func
def main_kernel_inner():
    T.evaluate(T.call_extern("int32", "ppl.rv.unknown"))
'''
    mod = tvm.tir.transform.LowerOpaqueBlock()(_module(source))
    func = mod["main_kernel_inner"].with_attr("tir.tpu.rv.legalized", 1)
    func = func.with_attr("tir.tpu.chip", "sg2260e")
    func = func.with_attr("tir.tpu.device_mode", "rv")
    func = func.with_attr("tir.tpu.rv.schema_version", 1)
    mod = tvm.IRModule({"main_kernel_inner": func})
    with pytest.raises(tvm.error.TVMError,
                       match=re.escape("does not implement ppl.rv.unknown")):
        tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)


GATE2_SOURCES = {
    "fill": r'''
@T.prim_func
def main_kernel_inner(A: T.Buffer((1, 8, 1, 16), "float16")):
    L = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 2), T.float32(1.5)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 2)))
''',
    "gemm": r'''
@T.prim_func
def main_kernel_inner(A0: T.Buffer((1, 16, 1, 16), "float16"), B0: T.Buffer((1, 16, 1, 16), "float16"), A1: T.Buffer((1, 16, 1, 16), "float16"), B1: T.Buffer((1, 16, 1, 16), "float16"), C: T.Buffer((1, 16, 1, 16), "float32")):
    L0 = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    R0 = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    L1 = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    R1 = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    O = T.alloc_buffer((1, 16, 1, 16), "float32", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 2), T.float32(0)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A0.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), L0.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B0.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), R0.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), L0.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), R0.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 3), 0, 0, 16, 16, 16))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A1.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), L1.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B1.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), R1.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), L1.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), R1.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 3), 0, 0, 16, 16, 16))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float32"), C.data, 0, 256, 2)))
''',
    "gemm_nt": r'''
@T.prim_func
def main_kernel_inner(A: T.Buffer((1, 16, 1, 8), "float16"), B: T.Buffer((1, 16, 1, 8), "float16"), C: T.Buffer((1, 16, 1, 16), "float32")):
    L = T.alloc_buffer((1, 16, 1, 8), "float16", scope="shared.dyn")
    R = T.alloc_buffer((1, 16, 1, 8), "float16", scope="shared.dyn")
    O = T.alloc_buffer((1, 16, 1, 16), "float32", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 2), T.float32(0)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), R.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), R.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 3), 0, 1, 16, 16, 8))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float32"), O.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float32"), C.data, 0, 256, 2)))
''',
    "special_function": r'''
@T.prim_func
def main_kernel_inner(A: T.Buffer((1, 8, 1, 16), "float16"), B: T.Buffer((1, 8, 1, 16), "float16")):
    L = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    O = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.rsqrt", T.tvm_access_ptr(T.type_annotation("float16"), O.data, 0, 128, 2), T.tvm_access_ptr(T.type_annotation("float16"), L.data, 0, 128, 1), 3))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), O.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 2)))
''',
    "gather": r'''
@T.prim_func
def main_kernel_inner(O: T.Buffer((1, 1, 4, 16), "float16"), A: T.Buffer((1, 1, 16, 16), "float16"), I: T.Buffer((1, 1, 4, 1), "uint32")):
    T.evaluate(T.call_extern("int32", "ppl.gather", T.tvm_access_ptr(T.type_annotation("float16"), O.data, 0, 64, 2), T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("uint32"), I.data, 0, 4, 1), 16))
''',
}


def test_gemm_legalization_materializes_full_m_n_k_contract():
    mod = _rv_module(GATE2_SOURCES["gemm"])
    calls = []

    def visit(node):
        if (isinstance(node, tvm.tir.Call) and
                node.op.name == "tir.call_extern" and node.args and
                isinstance(node.args[0], tvm.tir.StringImm) and
                node.args[0].value == "ppl.rv.gemm"):
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main_kernel_inner"].body, visit)
    assert len(calls) == 2
    assert all(len(call.args) == 9 for call in calls)
    assert [tuple(int(value) for value in call.args[6:9]) for call in calls] == [
        (16, 16, 16), (16, 16, 16)]


def test_sg2260e_rv_allocator_disables_async_unsafe_lifetime_reuse():
    first = _rv_module(GATE2_SOURCES["gemm"])["main_kernel_inner"]
    second = _rv_module(GATE2_SOURCES["gemm"])["main_kernel_inner"]
    assert int(first.attrs["tir.tpu.lmem_allow_lifetime_reuse"]) == 0
    names = ("L0", "R0", "L1", "R1", "O")
    intervals = []
    for name in names:
        prefix = f"tir.tpu.lmem.{name}."
        address = int(first.attrs[prefix + "address"])
        size = int(first.attrs[prefix + "size"])
        assert address == int(second.attrs[prefix + "address"])
        intervals.append((address, address + size, name))
    intervals.sort()
    assert all(lhs[1] <= rhs[0] for lhs, rhs in zip(intervals, intervals[1:]))
    assert intervals[-1][1] <= 16 * 16 * 1024


@pytest.mark.parametrize("pipeline", [False, True], ids=["serial", "pipeline"])
def test_gemm_accumulation_cmodel_numerics(monkeypatch, pipeline):
    monkeypatch.setenv(
        "PPL_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528"))
    rng = np.random.default_rng(20260805)
    lhs0 = rng.normal(scale=0.2, size=(1, 16, 1, 16)).astype(np.float16)
    rhs0 = rng.normal(scale=0.2, size=lhs0.shape).astype(np.float16)
    lhs1 = rng.normal(scale=0.2, size=lhs0.shape).astype(np.float16)
    rhs1 = rng.normal(scale=0.2, size=lhs0.shape).astype(np.float16)
    expected = (lhs0.reshape(16, 16).astype(np.float32) @
                rhs0.reshape(16, 16).astype(np.float32) +
                lhs1.reshape(16, 16).astype(np.float32) @
                rhs1.reshape(16, 16).astype(np.float32)).astype(np.float32)
    _run_cmodel(GATE2_SOURCES["gemm"], [lhs0, rhs0, lhs1, rhs1],
                expected.reshape(lhs0.shape), pipeline=pipeline, atol=1e-6)


@pytest.mark.parametrize(
    ("case", "instructions"),
    [
        ("fill", ("rvt_cp", "rvt_dma_st")),
        ("gemm", ("rvt_dma_ld", "rvt_fmm2a_nn", "rvt_dma_st")),
        ("gemm_nt", ("rvt_dma_ld", "rvt_fmm2a_nt", "rvt_dma_st")),
        ("special_function", ("rvt_cfg_rsqrt_iter", "rvt_sfu_rsqrt")),
        ("gather", ("rvt_cfg_dmaidx", "rvt_dma_hgather")),
    ],
)
def test_gate2_ops_match_golden_config_and_dataflow(case, instructions):
    generated = _rv_source(GATE2_SOURCES[case])
    golden = _golden_source("gemm" if case == "gemm_nt" else case)
    for instruction in instructions:
        assert instruction in generated
        if case != "gemm_nt":
            assert instruction in golden
            assert _instruction_class_signature(generated, instruction) == \
                _instruction_class_signature(golden, instruction)
    for config in ("RVT_CFGGR", "RVT_CFGTR"):
        # Gather is deliberately global-only in both TileLang and PPL.
        if config == "RVT_CFGTR" and case == "gather":
            continue
        assert config in generated
        assert config in golden
    assert "tpu_bdc_" not in generated
    assert "tpu_gdma_" not in generated
    if case in ("gemm", "gemm_nt"):
        assert generated.count("rvt_cfg_quant(0)") == (2 if case == "gemm" else 1)
        assert generated.count("rvt_cfg_satu(0, 0)") >= (2 if case == "gemm" else 1)
        expected = "rvt_fmm2a_nn" if case == "gemm" else "rvt_fmm2a_nt"
        assert generated.count(expected + "(") == (2 if case == "gemm" else 1)
    if case == "fill":
        assert re.search(r"RVT_CR\([^;]+, 0, TEEW_E16, 0, [^)]+\);", generated)
    if case == "gather":
        assert "rvt_cfg_dmaidx(0, &rv_dma_index)" in generated
        assert "uint64_t rv_dma_index = 0" in generated
        assert re.search(r"RVT_CFGGR_REG_SHAPE\([^;]+, 1, 1, 16, 16\)", generated)
        assert re.search(r"RVT_CFGGR_REG_SHAPE\([^;]+, 1, 1, 4, 1\)", generated)
        assert re.search(r"RVT_CFGGR_REG_SHAPE\([^;]+, 1, 1, 4, 16\)", generated)


@pytest.mark.parametrize("case", sorted(GATE2_SOURCES))
def test_gate2_source_is_valid_c_with_ppl_17_headers(case, tmp_path):
    ppl_root = Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528"
    if not ppl_root.is_dir():
        pytest.skip("PPL 1.7 SG2260E SDK is not available")
    layout = resolve_ppl_layout(str(ppl_root), "sg2260e")
    path = tmp_path / f"{case}.c"
    path.write_text(_rv_source(GATE2_SOURCES[case]), encoding="utf-8")
    command = ["cc", "-fsyntax-only", "-std=c11",
               *(f"-D{x}" for x in layout.compile_definitions),
               "-DTILELANG_PPL_HELPER_HAS_GET_DTYPE",
               *(f"-I{x}" for x in layout.include_dirs), str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _unsupported_source(operation, operand_count):
    operand = ('T.tvm_access_ptr(T.type_annotation("float16"), L.data, '
               '0, 128, 2)')
    operands = ", ".join([operand] * operand_count)
    return f'''\n@T.prim_func\ndef main_kernel_inner():\n    L = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")\n    T.evaluate(T.call_extern("int32", "{operation}", {operands}))\n'''


@pytest.mark.parametrize(
    ("operation", "operand_count", "diagnostic"),
    [
        ("ppl.exp", 5, "does not implement ppl.rv.exp"),
        ("ppl.sigmoid", 6, "does not implement ppl.rv.sigmoid"),
        ("ppl.reduce_sum", 3, "unavailable in PPL 1.7 SG2260E RV"),
        ("ppl.topk", 3, "PPL 1.7 SG2260E RV lowering"),
    ],
)
def test_gate2_unsupported_ops_fail_without_atomic_fallback(operation,
                                                            operand_count,
                                                            diagnostic):
    source = _unsupported_source(operation, operand_count)
    with pytest.raises(tvm.error.TVMError, match=diagnostic) as error:
        _rv_source(source)
    assert "tpu_bdc_" not in str(error.value)
    assert "tpu_gdma_" not in str(error.value)


def test_gate2_gemm_transpose_fails_clearly():
    source = GATE2_SOURCES["gemm_nt"].replace(
        "(1, 16, 1, 8)", "(1, 8, 1, 16)").replace(
            ", 0, 1, 16, 16, 8))", ", 1, 0, 16, 16, 8))")
    with pytest.raises(tvm.error.TVMError, match="has no SG2260E RV fmm2 variant"):
        _rv_source(source)


def test_gate2_gemm_dtype_fails_clearly():
    source = GATE2_SOURCES["gemm"].replace("float16", "float32")
    with pytest.raises(tvm.error.TVMError, match="fmm2a inputs must be fp16"):
        _rv_source(source)

    source = GATE2_SOURCES["gemm_nt"].replace('"float32"', '"float16"')
    with pytest.raises(tvm.error.TVMError,
                       match="rvt_fmm2a accumulator must be fp32"):
        _rv_source(source)
