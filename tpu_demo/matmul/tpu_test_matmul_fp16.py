# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Numerical FP16 matrix-multiplication demo for the TPU backends.

The module is safe to import: compilation and device dispatch only happen from
``main``. PCIe execution is additionally guarded by ``--allow-pcie`` and an
explicit device id because loading the vendor runtime can initialize hardware.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import tilelang
import tilelang.language as T
import torch

M = 64
N = 64
K = 64
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32


def matmul(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    output_tile_dtype: str = "float16",
):
    """Build the user-facing TileLang kernel.

    ``T.ppl_*`` remains the stable frontend spelling. The compiler selects
    TPU-Kernel or RV Tensor lowering from the complete target at compile time.
    """

    @T.prim_func
    def main_kernel_inner(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), "float32")
            C_out = T.alloc_shared((block_M, block_N), output_tile_dtype)

            T.ppl_fill(C_shared, T.float32(0))
            # This demo establishes the backend-neutral GEMM contract.  Keep
            # producer loads ordered before their GEMM consumer; validated
            # TPU software-pipeline overlap is a separate optimization layer.
            for k in T.serial(T.ceildiv(K, block_K)):
                T.ppl_copy(A[by * block_M, k * block_K], A_shared)
                T.ppl_copy(B[k * block_K, bx * block_N], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared, accumulate=True)
            T.ppl_copy(C_shared, C_out)
            T.ppl_copy(C_out, C[by * block_M, bx * block_N])

    return main_kernel_inner


def _device_id(value: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) > 2**31 - 1:
        raise argparse.ArgumentTypeError("device id must be a non-negative 32-bit decimal integer")
    return int(value)


def _configure_runtime_safety(runtime_mode: str, allow_pcie: bool,
                              device_id: Optional[int]) -> None:
    """Establish the PCIe loader contract before TileLang compilation."""

    if runtime_mode == "cmodel":
        if allow_pcie or device_id is not None:
            raise ValueError("--allow-pcie/--device-id are only valid with --runtime-mode pcie")
        return

    if not allow_pcie:
        raise ValueError("PCIe execution is disabled by default; rerun with --allow-pcie "
                         "only from an isolated, supervised process")
    if (device_id is None or isinstance(device_id, bool) or not isinstance(device_id, int) or
            not 0 <= device_id <= 2**31 - 1):
        raise ValueError("PCIe execution requires an explicit non-negative 32-bit --device-id")

    requested_id = str(device_id)
    existing_gate = os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD")
    existing_id = os.environ.get("TILELANG_TPU_DEVICE_ID")
    if existing_gate not in (None, "1"):
        raise ValueError("TILELANG_TPU_ALLOW_PCIE_LOAD must be unset or equal to 1")
    if existing_id is not None and existing_id != requested_id:
        raise ValueError("--device-id conflicts with the existing TILELANG_TPU_DEVICE_ID")
    os.environ["TILELANG_TPU_ALLOW_PCIE_LOAD"] = "1"
    os.environ["TILELANG_TPU_DEVICE_ID"] = requested_id


def run(
    *,
    chip: str = "sg2260e",
    programming_model: str = "tpukernel",
    runtime_mode: str = "cmodel",
    allow_pcie: bool = False,
    device_id: Optional[int] = None,
    seed: int = 0,
) -> None:
    """Compile, execute, and numerically validate one backend selection."""

    if chip not in ("bm1690", "sg2260e"):
        raise ValueError(f"unsupported TPU chip: {chip!r}")
    if programming_model not in ("tpukernel", "rv"):
        raise ValueError(f"unsupported programming model: {programming_model!r}")
    if runtime_mode not in ("cmodel", "pcie"):
        raise ValueError(f"unsupported runtime mode: {runtime_mode!r}")
    if chip == "bm1690" and programming_model == "rv":
        raise ValueError("BM1690 does not support the RV programming model")
    _configure_runtime_safety(runtime_mode, allow_pcie, device_id)

    torch.manual_seed(seed)
    kernel = tilelang.compile(
        matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K),
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    actual = torch.zeros(M, N, dtype=torch.float16)
    kernel(a, b, actual)

    expected = torch.matmul(a, b).half()
    absolute_error = torch.abs(actual - expected)
    max_abs_error = float(torch.max(absolute_error))
    mean_abs_error = float(torch.mean(absolute_error))
    passed = torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
    print(
        "MATMUL_RESULT "
        f"chip={chip} programming_model={programming_model} runtime_mode={runtime_mode} "
        f"max_abs_error={max_abs_error:.8g} "
        f"mean_abs_error={mean_abs_error:.8g} passed={passed}",
        flush=True,
    )
    if not passed:
        raise AssertionError(f"FP16 matmul mismatch: max absolute error is {max_abs_error:.8g}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", choices=("bm1690", "sg2260e"), default="sg2260e")
    parser.add_argument("--programming-model", choices=("tpukernel", "rv"), default="tpukernel")
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"), default="cmodel")
    parser.add_argument(
        "--allow-pcie",
        action="store_true",
        help="acknowledge that this run may initialize and dispatch to a TPU board",
    )
    parser.add_argument("--device-id", type=_device_id)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run(
            chip=args.chip,
            programming_model=args.programming_model,
            runtime_mode=args.runtime_mode,
            allow_pcie=args.allow_pcie,
            device_id=args.device_id,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
