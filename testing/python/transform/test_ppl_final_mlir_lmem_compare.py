# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path

import pytest
import tvm
from tvm.script import from_source

import tilelang


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "testing/ppl_rv_golden/lmem_compare.py"
SPEC = importlib.util.spec_from_file_location("lmem_compare", MODULE_PATH)
assert SPEC and SPEC.loader
lmem_compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lmem_compare
SPEC.loader.exec_module(lmem_compare)


SOURCES = {
    "copy": r'''
@T.prim_func
def main(A: T.Buffer((1, 8, 1, 16), "float16"), B: T.Buffer((1, 8, 1, 16), "float16")):
    local = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), local.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), local.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 2)))
''',
    "elementwise": r'''
@T.prim_func
def main(A: T.Buffer((1, 8, 1, 16), "float16"), B: T.Buffer((1, 8, 1, 16), "float16"), C: T.Buffer((1, 8, 1, 16), "float16")):
    l = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    r = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    sum = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    result = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), l.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), r.data, 0, 128, 2)))
    T.evaluate(T.call_extern("int32", "ppl.add", T.tvm_access_ptr(T.type_annotation("float16"), sum.data, 0, 128, 2), T.tvm_access_ptr(T.type_annotation("float16"), l.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), r.data, 0, 128, 1)))
    T.evaluate(T.call_extern("int32", "ppl.mul_C", T.tvm_access_ptr(T.type_annotation("float16"), result.data, 0, 128, 2), T.tvm_access_ptr(T.type_annotation("float16"), sum.data, 0, 128, 1), T.float32(0.5)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), result.data, 0, 128, 1), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 128, 2)))
''',
    "gemm": r'''
@T.prim_func
def main(A: T.Buffer((1, 16, 1, 16), "float16"), B: T.Buffer((1, 16, 1, 16), "float16"), C: T.Buffer((1, 16, 1, 16), "float16")):
    l = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    r = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    result = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), l.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), r.data, 0, 256, 2)))
    T.evaluate(T.call_extern("int32", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), l.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), r.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), result.data, 0, 256, 2), 0, 0))
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), result.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 256, 2)))
''',
}


def _tilelang_allocations(source, chip="sg2260e"):
    mod = tvm.IRModule({"main": from_source(source, {"T": tvm.script.tir})})
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    func = mod["main"].with_attr("tir.tpu.chip", chip)
    mod = tilelang.transform.AddressAssign()(tvm.IRModule({"main": func}))
    attrs = {str(key): value for key, value in mod["main"].attrs._dict().items()}
    return mod["main"].attrs, lmem_compare.parse_tilelang_attrs(attrs)


@pytest.mark.parametrize("case", ["copy", "elementwise", "gemm"])
def test_tilelang_allocator_matches_ppl_final_structural_invariants(case):
    _, actual = _tilelang_allocations(SOURCES[case])
    final_path = ROOT / "testing/ppl_rv_golden/artifacts" / case / f"{case}_final.mlir"
    expected = lmem_compare.parse_final_mlir(final_path.read_text(encoding="utf-8"))
    lmem_compare.compare_allocations(actual, expected)


def test_chip_dispatch_is_observable_and_unknown_chip_is_rejected():
    attrs, _ = _tilelang_allocations(SOURCES["copy"], "sg2260e")
    assert attrs["tir.tpu.lmem_chip"] == "sg2260e"
    assert int(attrs["tir.tpu.lmem_bank_num"]) == 16
    assert int(attrs["tir.tpu.lmem_bank_size"]) == 16 * 1024
    with pytest.raises(tvm.error.TVMError, match="Unsupported TPU chip"):
        _tilelang_allocations(SOURCES["copy"], "unknown-chip")


def test_rejects_overlapping_live_allocations():
    mlir = """
      %0 = ppl.tensorbe LOCAL %none {address = 0 : i64, idx = 0 : i32, bank_conflict = [], live_range = [0, 2], ppl.vname = "a", size = 64 : i64}
      %1 = ppl.tensorbe LOCAL %none {address = 0 : i64, idx = 1 : i32, bank_conflict = [], live_range = [1, 3], ppl.vname = "b", size = 64 : i64}
    """
    tensors = lmem_compare.parse_final_mlir(mlir)
    with pytest.raises(AssertionError, match="illegal simultaneous"):
        lmem_compare.validate_allocations(tensors, bank_size=16 * 1024)


def test_can_disable_lifetime_address_reuse_for_async_rv_execution():
    mlir = """
      %0 = ppl.tensorbe LOCAL %none {address = 0 : i64, idx = 0 : i32, bank_conflict = [], live_range = [0, 1], ppl.vname = "a", size = 64 : i64}
      %1 = ppl.tensorbe LOCAL %none {address = 0 : i64, idx = 1 : i32, bank_conflict = [], live_range = [1, 2], ppl.vname = "b", size = 64 : i64}
    """
    tensors = lmem_compare.parse_final_mlir(mlir)
    lmem_compare.validate_allocations(tensors, bank_size=16 * 1024)
    with pytest.raises(AssertionError, match="lifetime reuse is disabled"):
        lmem_compare.validate_allocations(
            tensors, bank_size=16 * 1024, allow_lifetime_reuse=False)


def test_rejects_live_conflict_at_different_offsets_in_same_bank():
    mlir = """
      %0 = ppl.tensorbe LOCAL %none {address = 0 : i64, idx = 0 : i32, bank_conflict = [1], live_range = [0, 2], ppl.vname = "a", size = 64 : i64}
      %1 = ppl.tensorbe LOCAL %none {address = 64 : i64, idx = 1 : i32, bank_conflict = [0], live_range = [0, 2], ppl.vname = "b", size = 64 : i64}
    """
    tensors = lmem_compare.parse_final_mlir(mlir)
    with pytest.raises(AssertionError, match="live conflict shares bank span"):
        lmem_compare.validate_allocations(tensors, bank_size=16 * 1024)
