import pytest
import tvm
from tvm import tir
from tvm.script import from_source

import tilelang


SOURCE = r'''
@T.prim_func
def main(A: T.Buffer((16,), "float16"), B: T.Buffer((16,), "float16")):
    C = T.alloc_buffer((16,), "float16", scope="shared.dyn")
    D = T.alloc_buffer((16,), "float16", scope="shared.dyn")
    E = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    F = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    G = T.alloc_buffer((1, 16, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.copy", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 2)))
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 2), T.float16(1)))
    T.evaluate(T.call_extern("int32", "ppl.add", T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 16, 1)))
    T.evaluate(T.call_extern("int32", "ppl.gemm", T.tvm_access_ptr(T.type_annotation("float16"), E.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), F.data, 0, 256, 1), T.tvm_access_ptr(T.type_annotation("float16"), G.data, 0, 256, 3), 0, 0, 16, 16, 16))
    T.evaluate(T.call_extern("int32", "ppl.reduce_sum", T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 2)))
    T.evaluate(T.call_extern("int32", "ppl.exp", T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 16, 1)))
    T.evaluate(T.call_extern("int32", "ppl.gather", T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 16, 1), T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 1)))
    T.evaluate(T.call_extern("int32", "ppl.topk", T.tvm_access_ptr(T.type_annotation("float16"), C.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), D.data, 0, 16, 2), T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 16, 1)))
'''


def _module():
    return tvm.IRModule({"main": from_source(SOURCE, {"T": tvm.script.tir})})


def _extern_calls(mod):
    calls = []

    def visit(node):
        if isinstance(node, tir.Call) and node.op.name == "tir.call_extern":
            calls.append(node)

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return calls


def test_rv_legalization_covers_golden_operator_families():
    result = tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")(_module())
    func = result["main"]
    assert func.attrs["tir.tpu.rv.legalized"] == 1
    assert func.attrs["tir.tpu.rv.schema_version"] == 1
    assert func.attrs["tir.tpu.rv.operation_count"] == 8
    assert func.attrs["tir.tpu.chip"] == "sg2260e"
    assert func.attrs["tir.tpu.device_mode"] == "rv"

    names = [call.args[0].value for call in _extern_calls(result)]
    expected = {"copy", "fill", "add", "gemm", "reduce_sum", "exp", "gather", "topk"}
    assert {f"ppl.rv.{name}" for name in expected}.issubset(names)
    assert not any(name.startswith("ppl.") and not name.startswith("ppl.rv.") for name in names)
    assert "ppl.rv.tensor_view" in names
    assert "ppl.rv.scalar" in names


def test_rv_register_spaces_and_allocation_are_explicit_and_deterministic():
    rv_pass = tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")
    first = rv_pass(_module())
    second = rv_pass(_module())
    assert tvm.ir.structural_equal(first, second, map_free_vars=True)

    views = [call for call in _extern_calls(first) if call.args[0].value == "ppl.rv.tensor_view"]
    assert views
    for view in views:
        register_class = int(view.args[3])
        register_number = int(view.args[4])
        layout = int(view.args[5])
        assert register_class in (1, 2)  # TR or GR
        assert register_number >= (8 if register_class == 1 else 32)
        assert layout in (0, 1, 3)  # HW_ALIGN, CONTINUOUS, or FREE
    assert {int(view.args[10]) for view in views} == {0, 1, 2, 3}

    scalars = [call for call in _extern_calls(first) if call.args[0].value == "ppl.rv.scalar"]
    assert [int(call.args[1]) for call in scalars] == [1]


def test_rv_legalization_rejects_non_sg2260e_chip():
    with pytest.raises(tvm.error.InternalError, match="supports sg2260e only"):
        tilelang.transform.RVLegalizeAndAllocateRegisters("bm1690")(_module())
