# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import contextlib
import io

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import AssignTPUAddresses, LowerAndLegalize, OptimizeForTarget


BANK_SIZE = 16 * 1024
LMEM_ADDRESS_ATTR_PREFIX = "tilelang.tpu.lmem.address."


def _assigned_attrs(func):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    target = tvm.target.Target(
        "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
        mod = AssignTPUAddresses(mod, target)
    return mod["main"].attrs


def _addr(attrs, name):
    key = name if name.startswith(LMEM_ADDRESS_ATTR_PREFIX) else (
        LMEM_ADDRESS_ATTR_PREFIX + name)
    return int(attrs[key])


def _attr_keys(attrs):
    return [str(key) for key in attrs._dict().keys()]


def _single_attr_with_prefix(attrs, prefix):
    keys = [
        key for key in _attr_keys(attrs)
        if key.startswith(LMEM_ADDRESS_ATTR_PREFIX + prefix)
    ]
    assert len(keys) == 1, keys
    return keys[0]


def _raw_local_allocations(*, data_names, buffer_names=None):
    """Build nested Allocate/DeclBuffer nodes without parser name uniquing."""
    if buffer_names is None:
        buffer_names = data_names
    body = tvm.tir.Evaluate(0)
    for data_name, buffer_name in reversed(
            list(zip(data_names, buffer_names))):
        pointer_type = tvm.ir.PointerType(
            tvm.ir.PrimType("float32"), "shared")
        data = tvm.tir.Var(data_name, pointer_type)
        buffer = tvm.tir.decl_buffer(
            (32,), "float32", name=buffer_name, data=data, scope="shared")
        body = tvm.tir.Allocate(
            data, "float32", [32], tvm.tir.IntImm("bool", 1),
            tvm.tir.DeclBuffer(buffer, body))
    return tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")


def test_ppl_gemm_readwrite_accumulator_is_separated_from_both_inputs():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            a_shared = T.alloc_shared((64, 1024), "float16")
            b_shared = T.alloc_shared((1024, 64), "float16")
            c_shared = T.alloc_shared((64, 64), "float32")

            T.ppl_fill(c_shared, T.float32(0.0))
            T.ppl_gemm(a_shared, b_shared, c_shared, accumulate=True)

    attrs = _assigned_attrs(main)
    a_addr = _addr(attrs, "a_shared")
    b_addr = _addr(attrs, "b_shared")
    c_addr = _addr(attrs, "c_shared")

    # ppl_gemm implements C += A @ B.  All three operands are therefore read
    # during the GEMM and must not contend for the same local-memory bank.
    assert len({
        a_addr // BANK_SIZE,
        b_addr // BANK_SIZE,
        c_addr // BANK_SIZE,
    }) == 3


def test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            src0 = T.alloc_shared((64, 1024), "float32")
            src1 = T.alloc_shared((64, 1024), "float32")
            dst = T.alloc_shared((64, 1024), "float32")

            T.ppl_add(dst, src0, src1)

    attrs = _assigned_attrs(main)
    src0_addr = _addr(attrs, "src0")
    src1_addr = _addr(attrs, "src1")
    dst_addr = _addr(attrs, "dst")

    assert src0_addr // BANK_SIZE != src1_addr // BANK_SIZE
    assert dst_addr // BANK_SIZE == src0_addr // BANK_SIZE


def test_reduce_tmp_is_separated_from_input_bank():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            inp = T.alloc_shared((64, 1024), "float32")
            out = T.alloc_shared((64, 1), "float32")

            T.ppl_reduce_sum(inp, out, dim=1)

    attrs = _assigned_attrs(main)
    tmp_name = _single_attr_with_prefix(attrs, "tmp_buffer_sum")
    inp_addr = _addr(attrs, "inp")
    tmp_addr = _addr(attrs, tmp_name)

    assert inp_addr // BANK_SIZE != tmp_addr // BANK_SIZE


def test_exp_composite_operands_are_conservative_bank_clique():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            out = T.alloc_shared((64, 1024), "float32")
            work0 = T.alloc_shared((64, 1024), "float32")
            work1 = T.alloc_shared((64, 1024), "float32")
            coeff = T.alloc_shared((64, 32), "float32")
            T.ppl_exp(out, work0, work1, coeff)

    attrs = _assigned_attrs(main)
    banks = {
        _addr(attrs, "out") // BANK_SIZE,
        _addr(attrs, "work0") // BANK_SIZE,
        _addr(attrs, "work1") // BANK_SIZE,
        _addr(attrs, "coeff") // BANK_SIZE,
    }

    assert len(banks) == 4


def test_address_assignment_rejects_non_tpu_target():
    @T.prim_func
    def main():
        T.evaluate(0)

    mod = tvm.IRModule({"main": main})
    target = tvm.target.Target("c")
    mod = tvm.tir.transform.BindTarget(target)(mod)

    with pytest.raises(ValueError, match="requires a TPU target"):
        AssignTPUAddresses(mod, target)

    with pytest.raises(tvm.error.TVMError, match="requires a TPU Target"):
        tilelang.transform.AddressAssign()(mod)


@pytest.mark.parametrize("target_spec, message, native_message", [
    ("tpu", "explicit physical chip", "supported target chip"),
    (
        "tpu -mcpu=sg2260e",
        "explicit programming model",
        "requires a normalized target",
    ),
    (
        "tpu -mcpu=bm1690 -tpu-programming-model=rv",
        "does not support programming model",
        "does not support",
    ),
])
def test_address_assignment_requires_a_complete_supported_tpu_target(
        target_spec, message, native_message):
    @T.prim_func
    def main():
        T.evaluate(0)

    target = tvm.target.Target(target_spec)
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": main}))

    with pytest.raises(ValueError, match=message):
        AssignTPUAddresses(mod, target)

    # The native transform is independently fail-closed for callers that
    # bypass tilelang.engine.phase.
    with pytest.raises(tvm.error.TVMError, match=native_message):
        tilelang.transform.AddressAssign()(mod)


def test_lmem_address_attributes_are_namespaced_and_follow_the_data_var():
    # Buffer.name and Buffer.data.name are independent in legal TIR.  Codegen
    # owns the allocation data Var, so its identity must drive the hand-off.
    function = _raw_local_allocations(
        data_names=("target",), buffer_names=("unrelated_buffer_name",))
    target = tvm.target.Target(
        "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs

    assert attrs["target"].kind.name == "tpu"
    assert _addr(attrs, "target") == 0
    assert "unrelated_buffer_name" not in _attr_keys(attrs)


def test_address_assignment_rejects_duplicate_allocation_data_names():
    function = _raw_local_allocations(data_names=("duplicate", "duplicate"))
    target = tvm.target.Target(
        "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    with pytest.raises(
            tvm.error.TVMError,
            match="unique local allocation data-variable names.*duplicate"):
        tilelang.transform.AddressAssign()(mod)


def test_copy_aliases_use_canonical_allocations_and_write_effects():
    """Buffer presentation aliases must not disappear from liveness.

    RegionOp may carry a Buffer object distinct from the DeclBuffer while both
    share the same data Var and exact descriptor.  AddressAssign owns storage
    by that Var identity.  The copy source and destination must therefore be
    live together (no overlapping addresses), while a read/write pair need not
    be separated into different banks.
    """
    pointer_type = tvm.ir.PointerType(
        tvm.ir.PrimType("float32"), "shared")
    a_data = tvm.tir.Var("a", pointer_type)
    b_data = tvm.tir.Var("b", pointer_type)
    a_decl = tvm.tir.decl_buffer(
        (32,), "float32", name="a_decl", data=a_data, scope="shared")
    b_decl = tvm.tir.decl_buffer(
        (32,), "float32", name="b_decl", data=b_data, scope="shared")
    a_alias = tvm.tir.decl_buffer(
        (32,), "float32", name="a_alias", data=a_data, scope="shared")
    b_alias = tvm.tir.decl_buffer(
        (32,), "float32", name="b_alias", data=b_data, scope="shared")

    def region(buffer, access_mask):
        return tvm.tir.call_intrin(
            "handle", tvm.ir.Op.get("tl.region"),
            tvm.tir.BufferLoad(buffer, [0]), access_mask, 32)

    copy = tvm.tir.call_extern(
        "handle", "tl.tpu.copy", region(a_alias, 1), region(b_alias, 2))
    body = tvm.tir.Allocate(
        a_data, "float32", [32], tvm.tir.IntImm("bool", 1),
        tvm.tir.DeclBuffer(
            a_decl,
            tvm.tir.Allocate(
                b_data, "float32", [32], tvm.tir.IntImm("bool", 1),
                tvm.tir.DeclBuffer(b_decl, tvm.tir.Evaluate(copy)))))
    function = tvm.tir.PrimFunc(
        [], body).with_attr("global_symbol", "main")
    target = tvm.target.Target(
        "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(
        tvm.IRModule({"main": function}))

    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs
    a_addr = _addr(attrs, "a")
    b_addr = _addr(attrs, "b")
    assert abs(a_addr - b_addr) >= 32 * 4
    assert a_addr // BANK_SIZE == b_addr // BANK_SIZE


if __name__ == "__main__":
    test_ppl_gemm_readwrite_accumulator_is_separated_from_both_inputs()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_composite_operands_are_conservative_bank_clique()
    test_address_assignment_rejects_non_tpu_target()
