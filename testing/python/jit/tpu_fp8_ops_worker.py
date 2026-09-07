# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Isolated numerical worker for the TPU-Kernel FP8 capability contract."""

import argparse
import os
import re
from typing import Tuple

import torch

import tilelang
import tilelang.language as T

_DTYPES = {
    "e4m3": ("e4m3_float8", torch.float8_e4m3fn),
    "e5m2": ("e5m2_float8", torch.float8_e5m2),
}

_PCIE_GATE_VARIABLES = (
    "TILELANG_TPU_ALLOW_PCIE_LOAD",
    "TILELANG_TPU_ALLOW_PCIE_PROFILE",
    "TILELANG_TPU_DEVICE_ID",
    "BMLIB_ENABLE_ALL_PROFILE",
    "PROFILE_RECORD_SIZE",
    "PROFILE_BOOK_KEEPING",
)


def _profile_selection() -> Tuple[str, str]:
    if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
        raise RuntimeError("FP8 worker must run through TPUInstructionProfiler")
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP", "")
    programming_model = os.environ.get("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE", "")
    if programming_model != "tpukernel":
        raise RuntimeError("FP8 capability worker requires programming_model=tpukernel")
    if chip not in ("bm1690", "sg2260e"):
        raise RuntimeError("FP8 capability worker requires a supported TPU chip")
    if runtime_mode not in ("cmodel", "pcie"):
        raise RuntimeError("FP8 capability worker requires runtime_mode=cmodel or pcie")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("FP8 profiling requires exactly one kernel launch")
    if runtime_mode == "cmodel":
        inherited_gates = tuple(name for name in _PCIE_GATE_VARIABLES if name in os.environ)
        if inherited_gates:
            raise RuntimeError(
                "CModel FP8 worker refuses PCIe recorder/device state: " +
                ", ".join(inherited_gates))
    else:
        required = {
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "BMLIB_ENABLE_ALL_PROFILE": "1",
        }
        invalid = tuple(
            name for name, expected in required.items()
            if os.environ.get(name) != expected)
        if invalid:
            raise RuntimeError(
                "PCIe FP8 worker requires explicit load/profile/recorder gates: " +
                ", ".join(invalid))
        device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
        if re.fullmatch(r"[0-9]+", device_id) is None or int(device_id) > 2**31 - 1:
            raise RuntimeError("PCIe FP8 worker requires a valid numeric device id")
    return chip, runtime_mode


def _target(chip: str) -> str:
    return f"tpu -mcpu={chip} -tpu-programming-model=tpukernel"


def _elementwise_operation(case: str) -> str:
    suffix = "-broadcast"
    if case.endswith(suffix):
        return case[:-len(suffix)]
    return case


def _compile(kernel, chip: str, runtime_mode: str):
    return tilelang.compile(
        kernel,
        out_idx=-1,
        target=_target(chip),
        runtime_mode=runtime_mode,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol=0.0, rtol=0.0) -> None:
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        max_abs = float(torch.max(torch.abs(actual.float() - expected.float())))
        raise RuntimeError(f"FP8 numerical mismatch; max_abs={max_abs}")


def _run_copy(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str, *,
              direct_global: bool) -> None:

    @T.prim_func
    def copy_kernel(A: T.Tensor((1, 64), dtype), B: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            if direct_global:
                T.ppl_copy(A, B)
            else:
                local = T.alloc_shared((1, 64), dtype)
                T.ppl_copy(A, local)
                T.ppl_copy(local, B)

    values = torch.linspace(-4.0, 4.0, 64, dtype=torch.float32).to(torch_dtype)
    source = values.reshape(1, 64)
    output = torch.zeros_like(source)
    _compile(copy_kernel, chip, runtime_mode)(source, output)
    if not torch.equal(source.view(torch.uint8), output.view(torch.uint8)):
        raise RuntimeError("same-format FP8 copy changed the encoded bytes")


def _run_fill_zero(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str) -> None:

    @T.prim_func
    def fill_kernel(B: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            local = T.alloc_shared((1, 64), dtype)
            T.ppl_fill(local, T.float32(0))
            T.ppl_copy(local, B)

    output = torch.full((1, 64), 2.0, dtype=torch.float32).to(torch_dtype)
    _compile(fill_kernel, chip, runtime_mode)(output)
    if torch.count_nonzero(output.view(torch.uint8)).item() != 0:
        raise RuntimeError("FP8 zero fill did not emit the all-zero encoding")


def _run_cast(direction: str, dtype: str, torch_dtype: torch.dtype, chip: str,
              runtime_mode: str) -> None:
    if direction == "to-fp8":

        @T.prim_func
        def cast_kernel(A: T.Tensor((1, 64), "float32"), B: T.Tensor((1, 64), dtype)):
            with T.Kernel(1, is_cpu=True) as _:
                src = T.alloc_shared((1, 64), "float32")
                dst = T.alloc_shared((1, 64), dtype)
                T.ppl_copy(A, src)
                T.ppl_copy(src, dst)
                T.ppl_copy(dst, B)

        source = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32).reshape(1, 64)
        output = torch.zeros((1, 64), dtype=torch.float32).to(torch_dtype)
        expected = source.to(torch_dtype)
    else:

        @T.prim_func
        def cast_kernel(A: T.Tensor((1, 64), dtype), B: T.Tensor((1, 64), "float32")):
            with T.Kernel(1, is_cpu=True) as _:
                src = T.alloc_shared((1, 64), dtype)
                dst = T.alloc_shared((1, 64), "float32")
                T.ppl_copy(A, src)
                T.ppl_copy(src, dst)
                T.ppl_copy(dst, B)

        source = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32).to(torch_dtype).reshape(1, 64)
        output = torch.zeros((1, 64), dtype=torch.float32)
        expected = source.float()

    _compile(cast_kernel, chip, runtime_mode)(source, output)
    if direction == "to-fp8":
        if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
            raise RuntimeError("FP32 to FP8 cast disagrees with the framework encoding")
    else:
        _assert_close(output, expected)


def _run_elementwise(operation: str, dtype: str, torch_dtype: torch.dtype, chip: str,
                     runtime_mode: str, *, broadcast_rhs: bool) -> None:
    rhs_shape = (1, 1) if broadcast_rhs else (1, 64)

    @T.prim_func
    def elementwise(A: T.Tensor((1, 64), dtype), B: T.Tensor(rhs_shape, dtype), C: T.Tensor((1, 64),
                                                                                            dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((1, 64), dtype)
            rhs = T.alloc_shared(rhs_shape, dtype)
            out = T.alloc_shared((1, 64), dtype)
            T.ppl_copy(A, lhs)
            T.ppl_copy(B, rhs)
            if operation == "add":
                T.ppl_add(out, lhs, rhs)
            elif operation == "sub":
                T.ppl_subtract(out, lhs, rhs)
            else:
                T.ppl_mul(out, lhs, rhs)
            T.ppl_copy(out, C)

    lhs_pattern = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
        dtype=torch.float32,
    ).repeat(8)
    if broadcast_rhs:
        rhs_pattern = torch.tensor([[0.5]], dtype=torch.float32)
    else:
        rhs_pattern = torch.tensor(
            [0.5, 1.0, 2.0, -1.0, -0.5, 2.0, -1.0, 0.5],
            dtype=torch.float32,
        ).repeat(8)
    lhs = lhs_pattern.to(torch_dtype).reshape(1, 64)
    rhs = rhs_pattern.to(torch_dtype).reshape(rhs_shape)
    output = torch.zeros_like(lhs)
    _compile(elementwise, chip, runtime_mode)(lhs, rhs, output)
    expected_fp32 = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
    }[operation](lhs.float(), rhs.float())
    expected = expected_fp32.to(torch_dtype)
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        _assert_close(output, expected, atol=0.25, rtol=0.0)


def _run_scalar(operation: str, dtype: str, torch_dtype: torch.dtype, chip: str,
                runtime_mode: str) -> None:
    value = -0.25 if operation == "add" else 0.75

    @T.prim_func
    def scalar_kernel(A: T.Tensor((1, 64), dtype), C: T.Tensor((1, 64), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            src = T.alloc_shared((1, 64), dtype)
            out = T.alloc_shared((1, 64), dtype)
            T.ppl_copy(A, src)
            if operation == "add":
                T.ppl_add_C(out, src, T.float32(value))
            else:
                T.ppl_mul_C(out, src, T.float32(value))
            T.ppl_copy(out, C)

    source = torch.tensor(
        [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=torch.float32,
    ).repeat(8).to(torch_dtype).reshape(1, 64)
    output = torch.zeros_like(source)
    _compile(scalar_kernel, chip, runtime_mode)(source, output)
    expected_fp32 = source.float() + value if operation == "add" else source.float() * value
    expected = expected_fp32.to(torch_dtype)
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        _assert_close(output, expected, atol=0.25, rtol=0.0)


def _run_rope(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str) -> None:
    shape = (4, 32)

    @T.prim_func
    def rope_kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype), C: T.Tensor(shape, dtype),
                    D: T.Tensor(shape, dtype), Output: T.Tensor(shape, dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            a = T.alloc_shared(shape, dtype)
            b = T.alloc_shared(shape, dtype)
            c = T.alloc_shared(shape, dtype)
            d = T.alloc_shared(shape, dtype)
            out = T.alloc_shared(shape, dtype)
            T.ppl_copy(A, a)
            T.ppl_copy(B, b)
            T.ppl_copy(C, c)
            T.ppl_copy(D, d)
            T.ppl_rope_add(out, a, b, c, d)
            T.ppl_copy(out, Output)

    inputs = tuple(
        (torch.arange(128, dtype=torch.float32).reshape(shape) % (7 + index) - 3.0).to(torch_dtype)
        for index in range(4))
    output = torch.zeros(shape, dtype=torch_dtype)
    _compile(rope_kernel, chip, runtime_mode)(*inputs, output)
    a, b, c, d = inputs
    expected_fp32 = torch.empty(shape, dtype=torch.float32)
    expected_fp32[:, 0::2] = a.float()[:, 0::2] + b.float()[:, 1::2]
    expected_fp32[:, 1::2] = c.float()[:, 1::2] + d.float()[:, 0::2]
    expected = expected_fp32.to(torch_dtype)
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        _assert_close(output, expected, atol=0.5, rtol=0.0)


def _run_gather(dtype: str, torch_dtype: torch.dtype, chip: str, runtime_mode: str) -> None:
    rows, width, count = 17, 32, 7

    @T.prim_func
    def gather_kernel(param_buffer: T.Tensor((rows, width), dtype), index_buffer: T.Tensor(
        (count, 1), "uint32"), output_buffer: T.Tensor((count, width), dtype)):
        with T.Kernel(1, is_cpu=True) as _:
            T.ppl_gather(output_buffer, param_buffer, index_buffer, rows)

    param = (torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) % 19 -
             9.0).to(torch_dtype)
    index_i32 = torch.tensor([16, 0, 8, 3, 12, 1, 15], dtype=torch.int32).reshape(count, 1)
    index_u32 = index_i32.view(torch.uint32)
    output = torch.zeros((count, width), dtype=torch_dtype)
    _compile(gather_kernel, chip, runtime_mode)(param, index_u32, output)
    expected = param[index_i32.long().reshape(-1)]
    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        raise RuntimeError("FP8 gather changed the selected encoded bytes")


def _run_gemm(case: str, dtype: str, torch_dtype: torch.dtype, chip: str,
              runtime_mode: str) -> None:
    transpose_b = case in ("gemm-nt-overwrite", "gemm-nt-accumulate")
    accumulate = case in ("gemm-nn-accumulate", "gemm-nt-accumulate")
    b_shape = (16, 64) if transpose_b else (64, 16)

    @T.prim_func
    def gemm(A: T.Tensor((16, 64), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((16, 16),
                                                                                    "float32")):
        with T.Kernel(1, is_cpu=True) as _:
            lhs = T.alloc_shared((16, 64), dtype)
            rhs = T.alloc_shared(b_shape, dtype)
            out = T.alloc_shared((16, 16), "float32")
            T.ppl_copy(A, lhs)
            T.ppl_copy(B, rhs)
            if accumulate:
                T.ppl_fill(out, T.float32(1))
            T.ppl_gemm(
                lhs,
                rhs,
                out,
                transpose_B=transpose_b,
                accumulate=accumulate,
            )
            T.ppl_copy(out, C)

    torch.manual_seed(7)
    lhs = (torch.randn(16, 64, dtype=torch.float32) * 0.5).to(torch_dtype)
    rhs = (torch.randn(*b_shape, dtype=torch.float32) * 0.5).to(torch_dtype)
    output = torch.zeros((16, 16), dtype=torch.float32)
    _compile(gemm, chip, runtime_mode)(lhs, rhs, output)
    expected = lhs.float() @ (rhs.float().T if transpose_b else rhs.float())
    if accumulate:
        expected += 1.0
    _assert_close(output, expected, atol=0.12, rtol=0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=tuple(_DTYPES), required=True)
    parser.add_argument(
        "--case",
        choices=(
            "copy",
            "copy-global-to-global",
            "fill-zero",
            "cast-to-fp8",
            "cast-from-fp8",
            "add",
            "sub",
            "mul",
            "add-broadcast",
            "sub-broadcast",
            "mul-broadcast",
            "add-scalar",
            "mul-scalar",
            "rope",
            "gather",
            "gemm-nn-overwrite",
            "gemm-nn-accumulate",
            "gemm-nt-overwrite",
            "gemm-nt-accumulate",
        ),
        required=True,
    )
    args = parser.parse_args()
    chip, runtime_mode = _profile_selection()
    dtype, torch_dtype = _DTYPES[args.dtype]
    if args.case in ("copy", "copy-global-to-global"):
        _run_copy(
            dtype,
            torch_dtype,
            chip,
            runtime_mode,
            direct_global=args.case == "copy-global-to-global",
        )
    elif args.case == "fill-zero":
        _run_fill_zero(dtype, torch_dtype, chip, runtime_mode)
    elif args.case.startswith("cast-"):
        _run_cast(args.case[len("cast-"):], dtype, torch_dtype, chip, runtime_mode)
    elif args.case in ("add", "sub", "mul", "add-broadcast", "sub-broadcast", "mul-broadcast"):
        operation = _elementwise_operation(args.case)
        _run_elementwise(
            operation,
            dtype,
            torch_dtype,
            chip,
            runtime_mode,
            broadcast_rhs=args.case.endswith("-broadcast"),
        )
    elif args.case in ("add-scalar", "mul-scalar"):
        _run_scalar(
            args.case[:-len("-scalar")],
            dtype,
            torch_dtype,
            chip,
            runtime_mode,
        )
    elif args.case == "rope":
        _run_rope(dtype, torch_dtype, chip, runtime_mode)
    elif args.case == "gather":
        _run_gather(dtype, torch_dtype, chip, runtime_mode)
    else:
        _run_gemm(args.case, dtype, torch_dtype, chip, runtime_mode)
    print(
        f"TPU_FP8_WORKER_OK chip={chip} dtype={args.dtype} case={args.case}",
        flush=True,
    )


if __name__ == "__main__":
    main()
