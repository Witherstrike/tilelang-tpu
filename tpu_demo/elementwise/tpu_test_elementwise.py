# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Numerical elementwise demo for TPU-Kernel and RV Tensor lowering.

The module is safe to import. Select ``add``, ``sub``, ``mul``, or ``div``
with ``--operation``; compilation and dispatch only happen from ``main``.
PCIe use is fail-closed and needs both ``--allow-pcie`` and ``--device-id``.
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
BLOCK_M = 32
BLOCK_N = 32


def elementwise(operation: str):
    """Build one operation while preserving the public ``T.ppl_*`` API."""

    if operation not in ("add", "sub", "mul", "div"):
        raise ValueError(f"unsupported elementwise operation: {operation!r}")

    @T.prim_func
    def main_kernel_inner(
            A: T.Tensor((M, N), "float32"),
            B: T.Tensor((M, N), "float32"),
            C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_N), "float32")
            B_shared = T.alloc_shared((BLOCK_M, BLOCK_N), "float32")
            C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), "float32")
            T.ppl_copy(A[by * BLOCK_M, bx * BLOCK_N], A_shared)
            T.ppl_copy(B[by * BLOCK_M, bx * BLOCK_N], B_shared)
            if operation == "add":
                T.ppl_add(C_shared, A_shared, B_shared)
            elif operation == "sub":
                T.ppl_subtract(C_shared, A_shared, B_shared)
            elif operation == "mul":
                T.ppl_mul(C_shared, A_shared, B_shared)
            else:
                T.ppl_div(C_shared, A_shared, B_shared)
            T.ppl_copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

    return main_kernel_inner


def _device_id(value: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) > 2**31 - 1:
        raise argparse.ArgumentTypeError("device id must be a non-negative 32-bit decimal integer")
    return int(value)


def _configure_runtime_safety(runtime_mode: str, allow_pcie: bool,
                              device_id: Optional[int]) -> None:
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
    operation: str = "add",
    chip: str = "sg2260e",
    programming_model: str = "tpukernel",
    runtime_mode: str = "cmodel",
    allow_pcie: bool = False,
    device_id: Optional[int] = None,
    seed: int = 0,
) -> None:
    """Compile, execute, and numerically validate one elementwise operation."""

    if operation not in ("add", "sub", "mul", "div"):
        raise ValueError(f"unsupported elementwise operation: {operation!r}")
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
        elementwise(operation),
        out_idx=-1,
        target=(f"tpu -mcpu={chip} "
                f"-tpu-programming-model={programming_model}"),
        runtime_mode=runtime_mode,
    )
    a = torch.randn(M, N, dtype=torch.float32)
    if operation == "div":
        # The RV instruction uses an approximate reciprocal path. Keeping the
        # divisor in [0.5, 2.0) avoids singular inputs and yields a useful test.
        b = torch.rand(M, N, dtype=torch.float32) * 1.5 + 0.5
    else:
        b = torch.randn(M, N, dtype=torch.float32)
    actual = torch.zeros(M, N, dtype=torch.float32)
    kernel(a, b, actual)

    expected = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
    }[operation](a, b)
    atol, rtol = ((1e-2, 1e-2) if operation == "div" else (1e-5, 1e-5))
    absolute_error = torch.abs(actual - expected)
    max_abs_error = float(torch.max(absolute_error))
    mean_abs_error = float(torch.mean(absolute_error))
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    print(
        "ELEMENTWISE_RESULT "
        f"operation={operation} chip={chip} programming_model={programming_model} "
        f"runtime_mode={runtime_mode} max_abs_error={max_abs_error:.8g} "
        f"mean_abs_error={mean_abs_error:.8g} passed={passed}",
        flush=True,
    )
    if not passed:
        raise AssertionError(f"elementwise {operation} mismatch: max absolute error is "
                             f"{max_abs_error:.8g}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("add", "sub", "mul", "div"), default="add")
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
            operation=args.operation,
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
