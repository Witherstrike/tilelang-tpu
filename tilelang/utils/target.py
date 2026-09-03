# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
from typing import Literal, Optional, Union
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

AVALIABLE_TARGETS = {
    "auto",
    "cuda",
    "hip",
    "webgpu",
    "c",  # represent c source backend
    "llvm",
    "tpu",
}


def is_tpu_target_spec(target: Union[str, Target]) -> bool:
    """Whether ``target`` names the TPU target kind, including chip variants.

    TPU chip selection uses the standard TVM ``-mcpu`` attribute, for example
    ``tpu -mcpu=sg2260e``.  Keep this narrow exception to the historical
    allow-list: arbitrary target strings remain rejected, while TPU gets the
    same target-specialisation spelling used by the GPU backends.
    """
    if isinstance(target, Target):
        return target.kind.name == "tpu"
    if not isinstance(target, str):
        return False
    try:
        return Target(target).kind.name == "tpu"
    except (TypeError, ValueError):
        return False


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system by locating the CUDA path.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        nvcc.find_cuda_path()
        return True
    except Exception:
        return False


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def _configured_tpu_auto_target() -> Optional[str]:
    """Return an explicitly configured, SDK-validated TPU target for ``auto``.

    A TPU cannot be safely inferred merely because the TileLang TPU backend was
    built: the physical chip determines PPL ABI/macros and a bare ``tpu`` used
    to silently select BM1690's PCIe-oriented compatibility default.  Require
    both an operator-selected chip and a usable PPL 1.7 layout before auto
    selection may choose TPU.
    """
    chip = os.environ.get("TILELANG_TPU_CHIP")
    if not chip:
        return None
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise ValueError(
            "TILELANG_TPU_CHIP is set, but PPL_PROJECT_ROOT is missing; "
            "set both to enable target='auto' TPU selection.")

    # Keep these imports local: target utilities are used by non-TPU builds,
    # which should not acquire a PPL dependency merely by importing TileLang.
    from tilelang.engine.tpu_config import get_tpu_chip_spec
    from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout

    spec = get_tpu_chip_spec(chip)
    resolve_ppl_layout(ppl_root, spec.name)
    return f"tpu -mcpu={spec.name}"


def determine_target(target: Union[str, Target, Literal["auto"]] = "auto",
                     return_object: bool = False) -> Union[str, Target]:
    """
    Determine the appropriate target for compilation.

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", CUDA/HIP are preferred. TPU is selected only when
              ``TILELANG_TPU_CHIP`` and a validated ``PPL_PROJECT_ROOT`` are
              explicitly configured; otherwise the portable C backend is the
              safe fallback.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target or a valid Target object.

    Raises:
        ValueError: If TPU auto selection was explicitly requested but its
            chip/PPL SDK configuration is invalid.
        AssertionError: If the target is invalid.
    """

    return_var: Union[str, Target] = target

    if target == "auto":
        # Check for CUDA and HIP availability
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()

        # Determine the target based on availability
        if is_cuda_available:
            return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        else:
            configured_tpu = _configured_tpu_auto_target()
            # Do not turn an unconfigured CPU-only host into an implicit
            # BM1690/PCIe job. The C backend is available in this build even
            # where LLVM is intentionally not enabled, so it is the portable
            # non-device fallback.
            return_var = configured_tpu or "c"
    else:
        # Validate the target if it's not "auto"
        assert isinstance(target, Target) or target in AVALIABLE_TARGETS or \
            is_tpu_target_spec(target), f"Target {target} is not supported"
        return_var = target

    if return_object:
        return Target(return_var)
    return return_var
