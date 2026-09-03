# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fresh-compile workers used by the TPU instruction-profile tests.

This file intentionally is not named ``test_*.py``: it is invoked by
``TPUInstructionProfiler`` in a new process and owns the complete JIT
compile/load/dispatch lifecycle.  It never loads a prebuilt TileLang TPU shared
object; PCIe is accepted only when the profiler supplies every safety gate.
"""

from __future__ import annotations

import argparse
import os

import torch

import tilelang
import tilelang.language as T


def _profile_selection():
    """Read the profiler's explicit chip/programming-model/runtime contract."""

    if os.environ.get("TILELANG_TPU_PROFILE_SESSION") != "1":
        raise RuntimeError("tpu_profile_worker must run through TPUInstructionProfiler.")
    chip = os.environ.get("TILELANG_TPU_PROFILE_CHIP")
    device_mode = os.environ.get("TILELANG_TPU_PROFILE_DEVICE_MODE")
    runtime_mode = os.environ.get("TILELANG_TPU_PROFILE_RUNTIME_MODE")
    if not chip or not device_mode or runtime_mode not in ("cmodel", "pcie"):
        raise RuntimeError("missing or invalid TileLang TPU profile selection.")
    if os.environ.get("TILELANG_TPU_BENCHMARK_RUNS") != "0":
        raise RuntimeError("instruction profiling requires exactly one TileLang launch.")
    if runtime_mode == "cmodel":
        for name in (
                "TILELANG_TPU_ALLOW_PCIE_LOAD",
                "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                "TILELANG_TPU_DEVICE_ID"):
            if name in os.environ:
                raise RuntimeError(
                    f"CModel profile worker inherited forbidden PCIe setting {name}.")
    else:
        for name in (
                "TILELANG_TPU_ALLOW_PCIE_LOAD",
                "TILELANG_TPU_ALLOW_PCIE_PROFILE",
                "BMLIB_ENABLE_ALL_PROFILE"):
            if os.environ.get(name) != "1":
                raise RuntimeError(f"PCIe profile worker is missing safety gate {name}=1.")
        device_id = os.environ.get("TILELANG_TPU_DEVICE_ID", "")
        if not device_id.isdecimal():
            raise RuntimeError("PCIe profile worker requires a numeric device ID.")
    return chip, device_mode, runtime_mode


def _matmul(chip: str, device_mode: str, runtime_mode: str) -> None:
    @T.prim_func
    def matmul(
            A: T.Tensor((64, 64), "float16"),
            B: T.Tensor((64, 64), "float16"),
            C: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_acc = T.alloc_shared((32, 32), "float32")
            C_out = T.alloc_shared((32, 32), "float16")

            T.ppl_fill(C_acc, T.float32(0))
            # Keep the core-op correctness matrix dependency-ordered.  TPU
            # software-pipeline overlap needs a separate hazard-aware contract;
            # num_stages=1 must not make a load race its immediate GEMM user.
            for k in T.serial(2):
                T.ppl_copy(A[by * 32, k * 32], A_shared)
                T.ppl_copy(B[k * 32, bx * 32], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_acc)
            T.ppl_copy(C_acc, C_out)
            T.ppl_copy(C_out, C[by * 32, bx * 32])

    torch.manual_seed(0)
    kernel = tilelang.compile(
        matmul,
        out_idx=-1,
        target=f"tpu -mcpu={chip}",
        device_mode=device_mode,
        runtime_mode=runtime_mode,
    )
    a = torch.randn(64, 64).half()
    b = torch.randn(64, 64).half()
    c = torch.zeros(64, 64).half()
    kernel(a, b, c)
    reference = torch.matmul(a, b).half()
    if not torch.allclose(c, reference, atol=1e-2, rtol=1e-2):
        difference = float(torch.max(torch.abs(reference - c)))
        raise RuntimeError(
            f"{device_mode} matmul mismatch; max abs difference={difference}")


def _elementwise(operation: str, chip: str, device_mode: str,
                 runtime_mode: str) -> None:
    @T.prim_func
    def elementwise(
            A: T.Tensor((64, 64), "float32"),
            B: T.Tensor((64, 64), "float32"),
            C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(2, 2, is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float32")
            B_shared = T.alloc_shared((32, 32), "float32")
            C_shared = T.alloc_shared((32, 32), "float32")
            T.ppl_copy(A[by * 32, bx * 32], A_shared)
            T.ppl_copy(B[by * 32, bx * 32], B_shared)
            if operation == "add":
                T.ppl_add(C_shared, A_shared, B_shared)
            elif operation == "sub":
                T.ppl_subtract(C_shared, A_shared, B_shared)
            elif operation == "mul":
                T.ppl_mul(C_shared, A_shared, B_shared)
            elif operation == "div":
                T.ppl_div(C_shared, A_shared, B_shared)
            T.ppl_copy(C_shared, C[by * 32, bx * 32])

    torch.manual_seed(0)
    kernel = tilelang.compile(
        elementwise,
        out_idx=-1,
        target=f"tpu -mcpu={chip}",
        device_mode=device_mode,
        runtime_mode=runtime_mode,
    )
    a = torch.randn(64, 64, dtype=torch.float32)
    if operation == "div":
        # Keep the divisor bounded away from zero; rvt_fdiv is a documented
        # reciprocal-sqrt approximation rather than strict IEEE division.
        b = torch.rand(64, 64, dtype=torch.float32) * 1.5 + 0.5
    else:
        b = torch.randn(64, 64, dtype=torch.float32)
    c = torch.zeros(64, 64, dtype=torch.float32)
    kernel(a, b, c)
    reference = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
    }[operation](a, b)
    atol, rtol = ((1e-2, 1e-2) if operation == "div" else (1e-5, 1e-5))
    if not torch.allclose(c, reference, atol=atol, rtol=rtol):
        difference = float(torch.max(torch.abs(reference - c)))
        raise RuntimeError(
            f"{device_mode} elementwise-{operation} mismatch; "
            f"max abs difference={difference}")


def _rv_control(chip: str, device_mode: str, runtime_mode: str) -> None:
    if device_mode != "rv":
        raise RuntimeError("rv-control requires device_mode='rv'.")

    @T.prim_func
    def control(A: T.Tensor((1,), "float32")):
        T.func_attr({"global_symbol": "profile_rv_control", "tir.noalias": T.bool(True)})
        T.rvt_kernel_start()
        T.rvt_sync_all()

    kernel = tilelang.compile(
        control,
        out_idx=[],
        target=f"tpu -mcpu={chip}",
        device_mode=device_mode,
        runtime_mode=runtime_mode,
    )
    kernel(torch.zeros(1, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "matmul",
            "tpukernel-matmul",
            "elementwise-add",
            "elementwise-sub",
            "elementwise-mul",
            "elementwise-div",
            "rv-control",
        ),
        required=True,
    )
    args = parser.parse_args()
    chip, device_mode, runtime_mode = _profile_selection()
    if args.case in ("matmul", "tpukernel-matmul"):
        _matmul(chip, device_mode, runtime_mode)
    elif args.case.startswith("elementwise-"):
        _elementwise(args.case.removeprefix("elementwise-"), chip, device_mode,
                     runtime_mode)
    else:
        _rv_control(chip, device_mode, runtime_mode)
    print(
        "TPU_PROFILE_WORKER_OK "
        f"case={args.case} chip={chip} device_mode={device_mode} runtime_mode={runtime_mode}",
        flush=True,
    )


if __name__ == "__main__":
    main()
