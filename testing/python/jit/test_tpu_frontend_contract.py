# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Compile-time contracts for TPU semantic operation helpers."""

import pytest

import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.language.copy import buffer_to_tile_region


def _target(chip="sg2260e"):
    return f"tpu -mcpu={chip} -tpu-programming-model=tpukernel"


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("float16", "DT_FP16"), ("bfloat16", "DT_BFP16"), ("float32", "DT_FP32")),
)
@pytest.mark.parametrize("chip", ("bm1690", "sg2260e"))
def test_rsqrt_selects_the_generic_instruction_for_all_declared_dtypes(chip, dtype, dtype_token):

    @T.prim_func
    def kernel(A: T.Tensor((1, 32), dtype), B: T.Tensor((1, 32), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((1, 32), dtype)
            dst = T.alloc_shared((1, 32), dtype)
            T.ppl_copy(A, src)
            T.ppl_rsqrt(dst, src)
            T.ppl_copy(dst, B)

    source = tilelang.lower(kernel, target=_target(chip), runtime_mode="cmodel").kernel_source
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

    source = tilelang.lower(kernel, target=_target(), runtime_mode="cmodel").kernel_source
    assert f"tpu_bdc_fp_{operation}_C(" in source
    assert "tpu_cast(" in source and f", {dtype_token}, DT_FP32," in source
    assert f"tpu_bdc_fp8_{operation}_C(" not in source


@pytest.mark.parametrize("dtype", ("e4m3_float8", "e5m2_float8"))
def test_fp8_transposed_gemm_accumulation_reaches_public_frontend(dtype):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 64), dtype)
            rhs = T.alloc_shared((16, 64), dtype)
            out = T.alloc_shared((16, 16), "float32")
            T.ppl_gemm(lhs, rhs, out, transpose_B=True, accumulate=True)

    source = tilelang.lower(kernel, target=_target(), runtime_mode="cmodel").kernel_source
    assert "tpu_bdc_fp8_mm_R_trans(" in source
    assert "true, false, false" in source


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("e4m3_float8", "DT_FP8E4M3"), ("e5m2_float8", "DT_FP8E5M2")),
)
@pytest.mark.parametrize("operation", ("rope", "gather"))
def test_validated_fp8_backend_owned_ops_reach_codegen(operation, dtype, dtype_token):
    if operation == "rope":

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                source0 = T.alloc_shared((4, 32), dtype)
                source1 = T.alloc_shared((4, 32), dtype)
                output = T.alloc_shared((4, 32), dtype)
                T.ppl_rope_add(output, source0, source1, source0, source1)
    else:

        @T.prim_func
        def kernel(param: T.Tensor((17, 32), dtype), index: T.Tensor((7, 1), "uint32"),
                   output: T.Tensor((7, 32), dtype)):
            with T.Kernel(1, is_cpu=True) as _:
                T.ppl_gather(output, param, index, 17)

    source = tilelang.lower(kernel, target=_target(), runtime_mode="cmodel").kernel_source
    expected_instruction = ("tpu_bdc_fp_add(" if operation == "rope" else "tpu_gdma_h_gather_S2S(")
    assert expected_instruction in source
    assert dtype_token in source


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("float16", "DT_FP16"), ("bfloat16", "DT_BFP16")),
)
def test_transposed_gemm_accumulation_is_selected_by_programming_model(dtype, dtype_token):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 64), dtype)
            rhs = T.alloc_shared((16, 64), dtype)
            out = T.alloc_shared((16, 16), "float32")
            T.ppl_gemm(lhs, rhs, out, transpose_B=True, accumulate=True)

    with pytest.raises(
            tvm.error.TVMError, match="no accumulating right-transpose GEMM instruction"):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")

    rv_source = tilelang.lower(
        kernel,
        target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
        runtime_mode="cmodel",
    ).kernel_source
    assert rv_source.count("rvt_fmm2a_nt(10, 8, 9, 0, 0, 0);") == 1
    assert (f"rvt_tr(8, PRECISION({dtype_token}), FP8TYPE({dtype_token})" in rv_source)
    assert (f"rvt_tr(9, PRECISION({dtype_token}), FP8TYPE({dtype_token})" in rv_source)
    assert "rvt_tr(10, PRECISION(DT_FP32), FP8TYPE(DT_FP32)" in rv_source
    assert "rvt_fmm2_nt(" not in rv_source
    assert "rvt_fmm2_nn(" not in rv_source
    assert "rvt_fmm2a_nn(" not in rv_source


def test_rv_native_gemm_cannot_bypass_fp32_accumulator_contract():

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 16), "float16")
            rhs = T.alloc_shared((16, 16), "float16")
            out = T.alloc_shared((16, 16), "float16")
            T.evaluate(
                T.call_extern("handle", "tl.tpu.gemm", buffer_to_tile_region(lhs, "r"),
                              buffer_to_tile_region(rhs, "r"), buffer_to_tile_region(out, "rw"),
                              T.bool(False), T.bool(False), 16, 16, 16, T.bool(True)))

    with pytest.raises(tvm.error.TVMError, match="accumulating fmm2 requires an FP32 C tile"):
        tilelang.lower(
            kernel,
            target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
            runtime_mode="cmodel",
        )


def test_invalid_elementwise_broadcast_fails_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                lhs = T.alloc_shared((4, 32), "float32")
                rhs = T.alloc_shared((1, 32), "float32")
                out = T.alloc_shared((4, 32), "float32")
                T.ppl_add(out, lhs, rhs)


@pytest.mark.parametrize(
    ("chip", "programming_model", "instruction"),
    (
        ("bm1690", "tpukernel", "tpu_bdc_max("),
        ("sg2260e", "tpukernel", "tpu_bdc_max("),
        ("sg2260e", "rv", "rvt_fmax(10, 8, 9);"),
    ),
)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
@pytest.mark.parametrize("rhs_shape", ((4, 32), (4, 1)))
def test_portable_max_selects_backend_instruction(chip, programming_model, instruction, dtype,
                                                  rhs_shape):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((4, 32), dtype)
            rhs = T.alloc_shared(rhs_shape, dtype)
            out = T.alloc_shared((4, 32), dtype)
            T.ppl_max(out, lhs, rhs)

    source = tilelang.lower(
        kernel,
        target=f"tpu -mcpu={chip} -tpu-programming-model={programming_model}",
        runtime_mode="cmodel",
    ).kernel_source
    assert instruction in source
    if rhs_shape == (4, 1):
        if programming_model == "rv":
            assert "FREE_LAYOUT" in source
            assert "(int[4]){" in source
            assert ".stride.h, 0});" in source
        else:
            assert ".w = 0;" in source


@pytest.mark.parametrize(
    ("operation", "instruction"),
    (("add", "rvt_fadd"), ("sub", "rvt_fsub"), ("mul", "rvt_fmul"), ("div", "rvt_fdiv")),
)
def test_rv_w_broadcast_uses_a_zero_stride_descriptor(operation, instruction):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((4, 32), "float32")
            rhs = T.alloc_shared((4, 1), "float32")
            out = T.alloc_shared((4, 32), "float32")
            if operation == "add":
                T.ppl_add(out, lhs, rhs)
            elif operation == "sub":
                T.ppl_subtract(out, lhs, rhs)
            elif operation == "mul":
                T.ppl_mul(out, lhs, rhs)
            else:
                T.ppl_div(out, lhs, rhs)

    source = tilelang.lower(
        kernel,
        target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
        runtime_mode="cmodel",
    ).kernel_source
    assert f"{instruction}(10, 8, 9);" in source
    descriptor = next(line for line in source.splitlines() if "rvt_tr(9," in line)
    assert "FREE_LAYOUT" in descriptor
    assert "(int[4]){" in descriptor
    assert descriptor.rstrip().endswith(".stride.h, 0});")


@pytest.mark.parametrize("chip", ("bm1690", "sg2260e"))
@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    (("e4m3_float8", "DT_FP8E4M3"), ("e5m2_float8", "DT_FP8E5M2")),
)
@pytest.mark.parametrize("rhs_shape", ((4, 32), (4, 1)))
def test_tpukernel_fp8_max_selects_generic_instruction(chip, dtype, dtype_token, rhs_shape):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((4, 32), dtype)
            rhs = T.alloc_shared(rhs_shape, dtype)
            out = T.alloc_shared((4, 32), dtype)
            T.ppl_max(out, lhs, rhs)

    source = tilelang.lower(kernel, target=_target(chip), runtime_mode="cmodel").kernel_source
    assert "tpu_bdc_max(" in source
    assert dtype_token in source
    if rhs_shape == (4, 1):
        assert ".w = 0;" in source


@pytest.mark.parametrize("dtype", ("e4m3_float8", "e5m2_float8"))
def test_rv_fp8_max_fails_closed_at_codegen(dtype):

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((4, 32), dtype)
            rhs = T.alloc_shared((4, 32), dtype)
            out = T.alloc_shared((4, 32), dtype)
            T.ppl_max(out, lhs, rhs)

    with pytest.raises(tvm.error.TVMError, match="validated only for the TPU-Kernel"):
        tilelang.lower(
            kernel,
            target="tpu -mcpu=sg2260e -tpu-programming-model=rv",
            runtime_mode="cmodel",
        )


def test_gemm_output_alias_fails_at_frontend():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                lhs_and_out = T.alloc_shared((16, 16), "float16")
                rhs = T.alloc_shared((16, 16), "float16")
                T.ppl_gemm(
                    lhs_and_out,
                    rhs,
                    lhs_and_out,
                    accumulate=False,
                )


def test_shape_changing_local_view_fails_at_compiler_descriptor_boundary():

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            storage = T.alloc_shared((2, 6), "float32")
            reshaped = T.view(storage, (3, 4))
            T.ppl_fill(storage, T.float32(0))
            T.ppl_fill(reshaped, T.float32(0))

    with pytest.raises(
            tvm.error.TVMError,
            match=(r"multiple Allocate nodes for data Var|"
                   r"shape disagrees with (its descriptor owner|its Allocate)")):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")


def test_rank_changing_local_view_fails_at_compiler_descriptor_boundary():

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            storage = T.alloc_shared((12,), "float32")
            reshaped = T.reshape(storage, (3, 4))
            T.ppl_fill(storage, T.float32(0))
            T.ppl_fill(reshaped, T.float32(0))

    with pytest.raises(
            tvm.error.TVMError,
            match=(r"multiple Allocate nodes for data Var|"
                   r"rank disagrees with (its descriptor owner|its Allocate)")):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")


def test_rope_cannot_lose_a_reshaped_view_contract():

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            out = T.alloc_shared((2, 6), "float32")
            even0 = T.alloc_shared((2, 6), "float32")
            even1 = T.alloc_shared((2, 6), "float32")
            odd0 = T.alloc_shared((2, 6), "float32")
            odd1 = T.alloc_shared((2, 6), "float32")
            T.ppl_fill(out, T.float32(0))
            T.ppl_fill(even0, T.float32(0))
            T.ppl_fill(even1, T.float32(0))
            T.ppl_fill(odd0, T.float32(0))
            T.ppl_fill(odd1, T.float32(0))
            T.ppl_rope_add(
                T.view(out, (3, 4)), T.view(even0, (3, 4)), T.view(even1, (3, 4)),
                T.view(odd0, (3, 4)), T.view(odd1, (3, 4)))

    with pytest.raises(
            tvm.error.TVMError,
            match=(r"multiple Allocate nodes for data Var|"
                   r"shape disagrees with (its descriptor owner|its Allocate)")):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")


def test_direct_tensor_alias_cannot_bypass_logical_shape_validation():

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            storage = T.alloc_shared((2, 6), "float32")
            alias = T.Tensor((3, 4), "float32", storage.data)
            T.ppl_fill(storage, T.float32(0))
            T.ppl_fill(alias, T.float32(0))

    with pytest.raises(
            tvm.error.TVMError,
            match=(r"multiple Allocate nodes for data Var|"
                   r"shape disagrees with (its descriptor owner|its Allocate)")):
        tilelang.lower(kernel, target=_target(), runtime_mode="cmodel")


@pytest.mark.parametrize("alias_kind", ("view", "reshape", "tensor"))
def test_descriptor_equivalent_local_alias_is_allowed(alias_kind):

    def make_alias(storage):
        if alias_kind == "view":
            return T.view(storage, (2, 6))
        if alias_kind == "reshape":
            return T.reshape(storage, (2, 6))
        return T.Tensor((2, 6), "float32", storage.data)

    @T.prim_func
    def kernel():
        with T.Kernel(1, is_cpu=True) as _:
            storage = T.alloc_shared((2, 6), "float32")
            alias = make_alias(storage)
            T.ppl_fill(alias, T.float32(0))

    source = tilelang.lower(kernel, target=_target(), runtime_mode="cmodel").kernel_source
    assert source.count("tpu_bdc_set_C(") == 1


@pytest.mark.parametrize("reduce", (T.ppl_reduce_sum, T.ppl_reduce_max))
def test_reduction_input_output_alias_fails_at_frontend(reduce):
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def kernel():
            with T.Kernel(1, is_cpu=True) as _:
                tile = T.alloc_shared((4, 32), "float32")
                reduce(tile, tile, dim=1)


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


@pytest.mark.parametrize("chip", ("bm1690", "sg2260e"))
def test_rank4_singleton_slice_roundtrips_through_rank2_local_tile(chip):

    @T.prim_func
    def kernel(source: T.Tensor((1, 16, 1, 16), "float32"), destination: T.Tensor((1, 16, 1, 16),
                                                                                  "float32")):
        with T.Kernel(1, is_cpu=True) as _:
            local = T.alloc_shared((16, 16), "float32")
            T.ppl_copy(source[0:1, 0:16, 0:1, 0:16], local)
            T.ppl_copy(local, destination[0:1, 0:16, 0:1, 0:16])

    source = tilelang.lower(kernel, target=_target(chip), runtime_mode="cmodel").kernel_source
    assert source.count("tpu_gdma_cpy_") == 2


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
