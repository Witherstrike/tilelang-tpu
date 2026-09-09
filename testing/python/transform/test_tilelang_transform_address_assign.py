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
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
        mod = AssignTPUAddresses(mod, target)
    return mod["main"].attrs


def _addr(attrs, name):
    key = name if name.startswith(LMEM_ADDRESS_ATTR_PREFIX) else (LMEM_ADDRESS_ATTR_PREFIX + name)
    return int(attrs[key])


def _attr_keys(attrs):
    return [str(key) for key in attrs._dict().keys()]


def _single_attr_with_prefix(attrs, prefix):
    keys = [key for key in _attr_keys(attrs) if key.startswith(LMEM_ADDRESS_ATTR_PREFIX + prefix)]
    assert len(keys) == 1, keys
    return keys[0]


def _raw_local_allocations(*, data_names, buffer_names=None):
    """Build nested Allocate/DeclBuffer nodes without parser name uniquing."""
    if buffer_names is None:
        buffer_names = data_names
    assert len(data_names) == len(buffer_names)
    body = tvm.tir.Evaluate(0)
    for data_name, buffer_name in reversed(list(zip(data_names, buffer_names))):  # noqa: B905
        pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
        data = tvm.tir.Var(data_name, pointer_type)
        buffer = tvm.tir.decl_buffer((32,), "float32", name=buffer_name, data=data, scope="shared")
        body = tvm.tir.Allocate(data, "float32", [32], tvm.tir.IntImm("bool", 1),
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
def test_address_assignment_requires_a_complete_supported_tpu_target(target_spec, message,
                                                                     native_message):

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
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs

    assert attrs["target"].kind.name == "tpu"
    assert _addr(attrs, "target") == 0
    assert "unrelated_buffer_name" not in _attr_keys(attrs)


def test_address_assignment_rejects_duplicate_allocation_data_names():
    function = _raw_local_allocations(data_names=("duplicate", "duplicate"))
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    with pytest.raises(
            tvm.error.TVMError, match="unique local allocation data-variable names.*duplicate"):
        tilelang.transform.AddressAssign()(mod)


def test_copy_aliases_use_canonical_allocations_and_write_effects():
    """Buffer presentation aliases must not disappear from liveness.

    RegionOp may carry a Buffer object distinct from the DeclBuffer while both
    share the same data Var and exact descriptor.  AddressAssign owns storage
    by that Var identity.  The copy source and destination must therefore be
    live together (no overlapping addresses), while a read/write pair need not
    be separated into different banks.
    """
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared")
    a_data = tvm.tir.Var("a", pointer_type)
    b_data = tvm.tir.Var("b", pointer_type)
    a_decl = tvm.tir.decl_buffer((32,), "float32", name="a_decl", data=a_data, scope="shared")
    b_decl = tvm.tir.decl_buffer((32,), "float32", name="b_decl", data=b_data, scope="shared")
    a_alias = tvm.tir.decl_buffer((32,), "float32", name="a_alias", data=a_data, scope="shared")
    b_alias = tvm.tir.decl_buffer((32,), "float32", name="b_alias", data=b_data, scope="shared")

    def region(buffer, access_mask):
        return tvm.tir.call_intrin("handle", tvm.ir.Op.get("tl.region"),
                                   tvm.tir.BufferLoad(buffer, [0]), access_mask, 32)

    copy = tvm.tir.call_extern("handle", "tl.tpu.copy", region(a_alias, 1), region(b_alias, 2))
    body = tvm.tir.Allocate(
        a_data, "float32", [32], tvm.tir.IntImm("bool", 1),
        tvm.tir.DeclBuffer(
            a_decl,
            tvm.tir.Allocate(b_data, "float32", [32], tvm.tir.IntImm("bool", 1),
                             tvm.tir.DeclBuffer(b_decl, tvm.tir.Evaluate(copy)))))
    function = tvm.tir.PrimFunc([], body).with_attr("global_symbol", "main")
    target = tvm.target.Target("tpu -mcpu=bm1690 -tpu-programming-model=tpukernel")
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))

    attrs = tilelang.transform.AddressAssign()(mod)["main"].attrs
    a_addr = _addr(attrs, "a")
    b_addr = _addr(attrs, "b")
    assert abs(a_addr - b_addr) >= 32 * 4
    assert a_addr // BANK_SIZE == b_addr // BANK_SIZE


def _liveness_buffer(name):
    return tvm.tir.decl_buffer((32,), "float32", name=name, scope="shared")


def _liveness_allocate(buffer, body):
    return tvm.tir.Allocate(buffer.data, buffer.dtype, list(buffer.shape),
                            tvm.tir.IntImm("bool", 1), tvm.tir.DeclBuffer(buffer, body))


def _liveness_fill(buffer):
    return tvm.tir.Evaluate(tvm.tir.call_extern("handle", "tl.tpu.fill", buffer.data, 1.0))


def _liveness_copy(src, dst):
    return tvm.tir.Evaluate(tvm.tir.call_extern("handle", "tl.tpu.copy", src.data, dst.data))


def _liveness_attrs(body, target_spec):
    function = tvm.tir.PrimFunc(tvm.tir.analysis.undefined_vars(body),
                                body).with_attr("global_symbol", "main")
    target = tvm.target.Target(target_spec)
    mod = tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": function}))
    return tilelang.transform.AddressAssign()(mod)["main"].attrs


TPU_ADDRESS_TARGETS = [
    "tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
    "tpu -mcpu=sg2260e -tpu-programming-model=tpukernel",
    "tpu -mcpu=sg2260e -tpu-programming-model=rv",
]


@pytest.mark.parametrize("target_spec", TPU_ADDRESS_TARGETS)
@pytest.mark.parametrize("loop_kind", ["for", "symbolic", "while", "nested"])
@pytest.mark.parametrize("initialize_before_loop", [False, True])
def test_loop_external_allocations_cannot_alias_later_scratch(target_spec, loop_kind,
                                                              initialize_before_loop):
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    read = _liveness_copy(weight, out)
    if loop_kind == "nested":
        # The read must reach the outer loop's end, beyond the inner loop.
        read = tvm.tir.For(tvm.tir.Var("j", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, read)
    body = tvm.tir.SeqStmt([read, _liveness_fill(scratch), _liveness_copy(scratch, out)])
    if loop_kind == "while":
        body = tvm.tir.While(tvm.tir.Var("keep_running", "bool"), body)
    else:
        extent = tvm.tir.Var("n", "int32") if loop_kind == "symbolic" else 2
        body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, extent, tvm.tir.ForKind.SERIAL, body)
    if initialize_before_loop:
        body = tvm.tir.SeqStmt([_liveness_fill(weight), body])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, target_spec)

    # Formerly weight=0, scratch=0, out=128: the later write destroyed the
    # next iteration's weight. A first use inside the loop needs protection
    # too; allocation scope, rather than a preceding use, determines this.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 128, 256]


@pytest.mark.parametrize("loop_extent", [None, 0, 1])
def test_no_backedge_preserves_sequential_address_reuse(loop_extent):
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    body = tvm.tir.SeqStmt(
        [_liveness_copy(weight, out),
         _liveness_fill(scratch),
         _liveness_copy(scratch, out)])
    if loop_extent is not None:
        body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, loop_extent, tvm.tir.ForKind.SERIAL, body)
    body = tvm.tir.SeqStmt([_liveness_fill(weight), body])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    # Static zero/one-trip loops cannot carry a value over a backedge. The
    # pass still visits their body, preserving its existing straight-line
    # allocation policy; dead-loop elimination belongs to other passes.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 0, 128]


def test_loop_local_allocations_keep_sequential_address_reuse():
    weight, scratch, out = [_liveness_buffer(name) for name in ("weight", "scratch", "out")]
    body = tvm.tir.SeqStmt([
        _liveness_fill(weight),
        _liveness_copy(weight, out),
        _liveness_fill(scratch),
        _liveness_copy(scratch, out)
    ])
    for buffer in (out, scratch, weight):
        body = _liveness_allocate(buffer, body)
    body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    # Each iteration owns a fresh weight allocation and initializes it before
    # use; the later scratch write cannot destroy a loop-carried value.
    assert [_addr(attrs, name) for name in ("weight", "scratch", "out")] == [0, 0, 128]


@pytest.mark.parametrize("nested", [False, True])
def test_loop_local_scratch_can_reuse_addresses_without_clobbering_external_weight(nested):
    weight, scratch0, scratch1, out = [
        _liveness_buffer(name) for name in ("weight", "scratch0", "scratch1", "out")
    ]
    body = tvm.tir.SeqStmt([
        _liveness_copy(weight, out),
        _liveness_allocate(
            scratch0, tvm.tir.SeqStmt([_liveness_fill(scratch0),
                                       _liveness_copy(scratch0, out)])),
        _liveness_allocate(
            scratch1, tvm.tir.SeqStmt([_liveness_fill(scratch1),
                                       _liveness_copy(scratch1, out)])),
    ])
    body = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    body = _liveness_allocate(weight, tvm.tir.SeqStmt([_liveness_fill(weight), body]))
    if nested:
        # Weight is local to the outer loop, external to the inner one;
        # scratch0/1 are local to both and retain sequential reuse.
        body = tvm.tir.For(tvm.tir.Var("j", "int32"), 0, 2, tvm.tir.ForKind.SERIAL, body)
    attrs = _liveness_attrs(_liveness_allocate(out, body), TPU_ADDRESS_TARGETS[0])

    expected = [128, 0, 256, 256] if nested else [0, 128, 256, 256]
    assert [_addr(attrs, name) for name in ("weight", "out", "scratch0", "scratch1")] == expected


def test_while_condition_local_read_survives_body_scratch_writes():
    condition, scratch, out = [_liveness_buffer(name) for name in ("condition", "scratch", "out")]
    body = tvm.tir.While(
        tvm.tir.BufferLoad(condition, [0]) > 0,
        tvm.tir.SeqStmt([_liveness_fill(scratch),
                         _liveness_copy(scratch, out)]))
    for buffer in (out, scratch, condition):
        body = _liveness_allocate(buffer, body)
    attrs = _liveness_attrs(body, TPU_ADDRESS_TARGETS[0])

    assert [_addr(attrs, name) for name in ("condition", "scratch", "out")] == [0, 128, 256]


if __name__ == "__main__":
    test_ppl_gemm_readwrite_accumulator_is_separated_from_both_inputs()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_composite_operands_are_conservative_bank_clique()
    test_address_assignment_rejects_non_tpu_target()
