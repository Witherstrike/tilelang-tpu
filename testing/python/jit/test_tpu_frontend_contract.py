# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Compile-time contracts for TPU semantic operation helpers."""

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T


def _target(chip="sg2260e"):
    return f"tpu -mcpu={chip} -tpu-programming-model=tpukernel"


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("float16", "DT_FP16"), ("bfloat16", "DT_BFP16"),
     ("float32", "DT_FP32")),
)
@pytest.mark.parametrize("chip", ("bm1690", "sg2260e"))
def test_rsqrt_selects_the_generic_instruction_for_all_declared_dtypes(
        chip, dtype, dtype_token):
    @T.prim_func
    def kernel(A: T.Tensor((1, 32), dtype), B: T.Tensor((1, 32), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((1, 32), dtype)
            dst = T.alloc_shared((1, 32), dtype)
            T.ppl_copy(A, src)
            T.ppl_rsqrt(dst, src)
            T.ppl_copy(dst, B)

    source = tilelang.lower(
        kernel, target=_target(chip), runtime_mode="cmodel").kernel_source
    assert "tpu_bdc_fp_rsqrt(" in source
    assert f"{dtype_token});" in source
    assert "tpu_bdc_fp32_rsqrt(" not in source


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("e4m3_float8", "DT_FP8E4M3"), ("e5m2_float8", "DT_FP8E5M2")),
)
@pytest.mark.parametrize("operation", ("add", "mul"))
def test_fp8_scalar_uses_generic_scalar_instruction(operation, dtype, dtype_token):
    @T.prim_func
    def kernel(A: T.Tensor((1, 32), dtype), B: T.Tensor((1, 32), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((1, 32), dtype)
            dst = T.alloc_shared((1, 32), dtype)
            T.ppl_copy(A, src)
            if operation == "add":
                T.ppl_add_C(dst, src, T.float32(-0.25))
            else:
                T.ppl_mul_C(dst, src, T.float32(0.75))
            T.ppl_copy(dst, B)

    source = tilelang.lower(
        kernel, target=_target(), runtime_mode="cmodel").kernel_source
    assert f"tpu_bdc_fp_{operation}_C(" in source
    assert f"tpu_cast(" in source and f", {dtype_token}, DT_FP32," in source
    assert f"tpu_bdc_fp8_{operation}_C(" not in source


@pytest.mark.parametrize("dtype", ("e4m3_float8", "e5m2_float8"))
def test_fp8_transposed_gemm_accumulation_reaches_public_frontend(dtype):
    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 64), dtype)
            rhs = T.alloc_shared((16, 64), dtype)
            out = T.alloc_shared((16, 16), "float32")
            T.ppl_gemm(
                lhs, rhs, out, transpose_B=True, accumulate=True)

    source = tilelang.lower(
        kernel, target=_target(), runtime_mode="cmodel").kernel_source
    assert "tpu_bdc_fp8_mm_R_trans(" in source
    assert "true, false, false" in source


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("e4m3_float8", "DT_FP8E4M3"), ("e5m2_float8", "DT_FP8E5M2")),
)
@pytest.mark.parametrize("operation", ("rope", "gather"))
def test_validated_fp8_backend_owned_ops_reach_codegen(
        operation, dtype, dtype_token):
    if operation == "rope":

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                source0 = T.alloc_shared((4, 32), dtype)
                source1 = T.alloc_shared((4, 32), dtype)
                output = T.alloc_shared((4, 32), dtype)
                T.ppl_rope_add(
                    output, source0, source1, source0, source1)
    else:

        @T.prim_func
        def kernel(param: T.Tensor((17, 32), dtype),
                   index: T.Tensor((7, 1), "uint32"),
                   output: T.Tensor((7, 32), dtype)):
            with T.Kernel(1, is_cpu=True) as _:
                T.ppl_gather(output, param, index, 17)

    source = tilelang.lower(
        kernel, target=_target(), runtime_mode="cmodel").kernel_source
    expected_instruction = (
        "tpu_bdc_fp_add(" if operation == "rope" else
        "tpu_gdma_h_gather_S2S(")
    assert expected_instruction in source
    assert dtype_token in source


def test_non_fp8_transposed_gemm_accumulation_fails_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                lhs = T.alloc_shared((16, 64), "float16")
                rhs = T.alloc_shared((16, 64), "float16")
                out = T.alloc_shared((16, 16), "float32")
                T.ppl_gemm(
                    lhs, rhs, out, transpose_B=True, accumulate=True)


def test_invalid_elementwise_broadcast_fails_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                lhs = T.alloc_shared((4, 32), "float32")
                rhs = T.alloc_shared((1, 32), "float32")
                out = T.alloc_shared((4, 32), "float32")
                T.ppl_add(out, lhs, rhs)


@pytest.mark.parametrize("operation", ("exp", "sigmoid", "rope"))
def test_scratch_or_output_aliases_fail_at_frontend(operation):
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                out = T.alloc_shared((4, 32), "float32")
                work0 = T.alloc_shared((4, 32), "float32")
                work1 = T.alloc_shared((4, 32), "float32")
                coeff = T.alloc_shared((64, 32), "float32")
                if operation == "exp":
                    T.ppl_exp(out, out, work1, coeff)
                elif operation == "sigmoid":
                    T.ppl_sigmoid(out, out, work0, work1, coeff)
                else:
                    T.ppl_rope_add(out, out, work0, work1, work0)


def test_invalid_copy_operand_fails_with_a_frontend_diagnostic():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                dst = T.alloc_shared((1, 32), "float32")
                T.ppl_copy(T.float32(1), dst)


def test_descriptor_extent_above_uint16_limit_fails_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                dst = T.alloc_shared((1, 65536), "float32")
                T.ppl_fill(dst, T.float32(0))


@pytest.mark.parametrize("operation", ("div", "exp", "sigmoid", "rsqrt", "reduce"))
def test_unvalidated_fp8_operations_fail_at_frontend(operation):
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                source0 = T.alloc_shared((4, 32), "e4m3_float8")
                source1 = T.alloc_shared((4, 32), "e4m3_float8")
                output = T.alloc_shared((4, 32), "e4m3_float8")
                coeff = T.alloc_shared((64, 32), "e4m3_float8")
                if operation == "div":
                    T.ppl_div(output, source0, source1)
                elif operation == "exp":
                    T.ppl_exp(output, source0, source1, coeff)
                elif operation == "sigmoid":
                    T.ppl_sigmoid(output, source0, source1, output, coeff)
                elif operation == "rsqrt":
                    T.ppl_rsqrt(output, source0)
                else:
                    reduced = T.alloc_shared((4, 1), "e4m3_float8")
                    T.ppl_reduce_sum(source0, reduced, dim=1)


def test_nonzero_fp8_fill_fails_closed_in_codegen():
    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            output = T.alloc_shared((1, 32), "e4m3_float8")
            T.ppl_fill(output, T.float32(1))

    with pytest.raises(tvm.error.TVMError, match="FP8 fill.*only.*zero"):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")
