import re
import subprocess
from pathlib import Path

import pytest
import tvm
from tvm.script import from_source

import tilelang
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout


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
