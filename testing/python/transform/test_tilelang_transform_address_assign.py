# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import contextlib
import io

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


BANK_SIZE = 16 * 1024


def _assigned_attrs(func, chip="bm1690"):
    func = func.with_attr("tir.tpu.chip", chip)
    func = func.with_attr("tir.tpu.device_mode", "atomic")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    target = tvm.target.Target("tpu")
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    return mod["main"].attrs


def _addr(attrs, name):
    return int(attrs[name])


def _attr_keys(attrs):
    return [str(key) for key in attrs._dict().keys()]


def test_bm1690_keeps_lifetime_reuse_policy_for_sequential_buffers():
    source = r'''
@T.prim_func
def main():
    A = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    B = T.alloc_buffer((1, 8, 1, 16), "float16", scope="shared.dyn")
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float16"), A.data, 0, 128, 2), T.float32(0)))
    T.evaluate(T.call_extern("int32", "ppl.fill", T.tvm_access_ptr(T.type_annotation("float16"), B.data, 0, 128, 2), T.float32(0)))
'''
    from tvm.script import from_source
    func = from_source(source, {"T": tvm.script.tir}).with_attr(
        "tir.tpu.chip", "bm1690")
    mod = tvm.tir.transform.LowerOpaqueBlock()(tvm.IRModule({"main": func}))
    mod = tilelang.transform.AddressAssign()(mod)
    attrs = mod["main"].attrs
    assert int(attrs["tir.tpu.lmem_allow_lifetime_reuse"]) == 1
    assert int(attrs["tir.tpu.lmem.A.address"]) == int(
        attrs["tir.tpu.lmem.B.address"])


def _single_attr_with_prefix(attrs, prefix):
    keys = [key for key in _attr_keys(attrs) if key.startswith(prefix)]
    assert len(keys) == 1, keys
    return keys[0]


def test_ppl_gemm_accumulator_is_bank_separated_from_inputs():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            a_shared = T.alloc_shared((64, 1024), "float32")
            b_shared = T.alloc_shared((1024, 64), "float32")
            c_shared = T.alloc_shared((64, 64), "float32")

            T.ppl_fill(c_shared, T.float32(0.0))
            T.ppl_gemm(a_shared, b_shared, c_shared)

    attrs = _assigned_attrs(main)
    a_addr = _addr(attrs, "a_shared")
    b_addr = _addr(attrs, "b_shared")
    c_addr = _addr(attrs, "c_shared")

    assert a_addr // BANK_SIZE != b_addr // BANK_SIZE
    assert c_addr // BANK_SIZE != a_addr // BANK_SIZE
    assert c_addr // BANK_SIZE != b_addr // BANK_SIZE


def test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            src0 = T.alloc_shared((64, 1024), "float32")
            src1 = T.alloc_shared((64, 1024), "float32")
            dst = T.alloc_shared((64, 64), "float32")

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
            table = T.alloc_shared((64, 192), "float32")

            T.ppl_exp2(out, work0, work1, coeff, table)

    attrs = _assigned_attrs(main)
    banks = {
        _addr(attrs, "out") // BANK_SIZE,
        _addr(attrs, "work0") // BANK_SIZE,
        _addr(attrs, "work1") // BANK_SIZE,
        _addr(attrs, "coeff") // BANK_SIZE,
        _addr(attrs, "table") // BANK_SIZE,
    }

    assert len(banks) == 5


def test_sg2260e_uses_chip_description_and_preserves_allocation_contract():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((64, 1024), "float32")
            rhs = T.alloc_shared((64, 1024), "float32")
            out = T.alloc_shared((64, 64), "float32")
            T.ppl_add(out, lhs, rhs)

    attrs = _assigned_attrs(main, "sg2260e")
    assert attrs["tir.tpu.chip"] == "sg2260e"
    assert _addr(attrs, "lhs") // BANK_SIZE != _addr(attrs, "rhs") // BANK_SIZE


if __name__ == "__main__":
    test_ppl_gemm_output_write_phase_can_share_input_bank()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_composite_operands_are_conservative_bank_clique()
