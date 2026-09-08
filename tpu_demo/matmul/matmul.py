# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tiled matrix multiplication for TPU-Kernel and RV Tensor."""

from typing import Optional

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance,
                             torch_dtype, validate_dimensions, validate_exact_tiling,
                             validate_selection)


def build_matmul(*, m: int = 32, n: int = 32, k: int = 32, block_m: int = 16,
                 block_n: int = 16, block_k: int = 16, dtype: str = "float16",
                 fp32_compute_dtype: str = "bfloat16"):
    if fp32_compute_dtype not in ("float16", "bfloat16"):
        raise ValueError("FP32 matmul compute dtype must be float16 or bfloat16")
    torch_dtype(dtype)
    validate_dimensions(
        "matmul", m=m, n=n, k=k, block_m=block_m, block_n=block_n, block_k=block_k)
    validate_exact_tiling(
        "matmul", ("m", m, block_m), ("n", n, block_n), ("k", k, block_k))
    compute_dtype = fp32_compute_dtype if dtype == "float32" else dtype

    @T.prim_func
    def kernel(A: T.Tensor((m, k), dtype), B: T.Tensor((k, n), dtype),
               C: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m),
                      is_cpu=True) as (bx, by):
            A_compute = T.alloc_shared((block_m, block_k), compute_dtype)
            B_compute = T.alloc_shared((block_k, block_n), compute_dtype)
            C_acc = T.alloc_shared((block_m, block_n), "float32")
            T.ppl_fill(C_acc, T.float32(0))
            for ko in T.serial(T.ceildiv(k, block_k)):
                if dtype == "float32":
                    A_input = T.alloc_shared((block_m, block_k), dtype)
                    B_input = T.alloc_shared((block_k, block_n), dtype)
                    T.ppl_copy(A[by * block_m, ko * block_k], A_input)
                    T.ppl_copy(B[ko * block_k, bx * block_n], B_input)
                    T.ppl_copy(A_input, A_compute)
                    T.ppl_copy(B_input, B_compute)
                else:
                    T.ppl_copy(A[by * block_m, ko * block_k], A_compute)
                    T.ppl_copy(B[ko * block_k, bx * block_n], B_compute)
                T.ppl_gemm(A_compute, B_compute, C_acc, accumulate=True)
            if dtype == "float32":
                T.ppl_copy(C_acc, C[by * block_m, bx * block_n])
            else:
                C_output = T.alloc_shared((block_m, block_n), dtype)
                T.ppl_copy(C_acc, C_output)
                T.ppl_copy(C_output, C[by * block_m, bx * block_n])

    return kernel


def run(*, dtype: str, chip: str, programming_model: str, runtime_mode: str,
        allow_pcie: bool = False, device_id: Optional[int] = None, seed: int = 0) -> dict:
    torch_dtype(dtype)
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=True,
        allow_pcie=allow_pcie,
        device_id=device_id,
    )
    shape = (32, 32, 32)
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    # Bounded inputs keep the deliberately reduced-precision FP32 matrix path
    # measurable without hiding its contract behind an excessively loose gate.
    a = (torch.randn((shape[0], shape[2]), generator=generator) * 0.25).to(host_dtype)
    b = (torch.randn((shape[2], shape[1]), generator=generator) * 0.25).to(host_dtype)
    dst = torch.zeros((shape[0], shape[1]), dtype=host_dtype)
    timing = compile_and_launch(
        build_matmul(m=shape[0], n=shape[1], k=shape[2], dtype=dtype),
        (a, b, dst),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
    )
    if dtype == "float32":
        a_ref = a.to(torch.bfloat16).float()
        b_ref = b.to(torch.bfloat16).float()
    else:
        a_ref, b_ref = a.float(), b.float()
    expected = torch.matmul(a_ref, b_ref).to(host_dtype)
    atol, rtol = tolerance(dtype, "matmul")
    metrics = comparison(dst, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="matmul", dtype=dtype, chip=chip, programming_model=programming_model,
        runtime_mode=runtime_mode, metrics=metrics, timing=timing,
        parameters={"m": shape[0], "n": shape[1], "k": shape[2], "block": 16,
                    "fp32_compute_dtype": "bfloat16" if dtype == "float32" else None,
                    "seed": seed})
