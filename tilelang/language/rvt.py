# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Low-level bindings for the PPL 1.7 RISC-V Tensor (RVT) ABI.

The bindings deliberately mirror ``rvt_api.h`` instead of attempting to turn
TileLang buffers into RVT CR/TR/GR descriptors automatically.  Use them with
``-tpu-programming-model=rv`` and configure descriptors/registers according to
the PPL RVT ABI. Existing TPUKernel-only TileLang externs require
``-tpu-programming-model=tpukernel``; portable ``tl.tpu.*`` operations are
selected by the target-aware emitter.
"""

import re

import tilelang.language as T


def rvt_call(intrinsic: str, *args):
    """Emit one PPL ``rvt_api.h`` call.

    ``intrinsic`` must be the literal name of an RVT API beginning with
    ``rvt_`` (for example ``\"rvt_fconv\"`` or ``\"rvt_dma_hscatter\"``).
    This generic entry point covers the direct subset of PPL 1.7 RVT whose C
    ABI arguments can be represented by TIR scalar/pointer expressions. It
    intentionally does not claim support for header APIs that take C structs
    by value (for example ``array4_t``); those require dedicated descriptor
    builder helpers before they can be called correctly from TileLang.
    Arguments are passed unchanged to the C ABI and therefore must follow the
    exact types and descriptor-register conventions in ``rvt_api.h``.
    """
    if not re.fullmatch(r"rvt_[A-Za-z0-9_]+", intrinsic):
        raise ValueError(
            "RVT intrinsics must be a C identifier beginning with 'rvt_'; "
            f"got {intrinsic!r}")
    return T.call_extern("handle", intrinsic, *args)


# Common sequencing/configuration operations receive named wrappers so a
# kernel can be written naturally while rvt_call remains the direct-ABI escape
# hatch for TIR-representable parts of the versioned vendor surface.
def rvt_kernel_start():
    return rvt_call("rvt_kernel_start")


def rvt_set_max_cmd_id(cmd_id):
    return rvt_call("rvt_set_max_cmd_id", cmd_id)


def rvt_parallel(enable):
    return rvt_call("rvt_parallel", enable)


def rvt_fence(pred, succ):
    return rvt_call("rvt_fence", pred, succ)


def rvt_sync_i(register_state, engine):
    return rvt_call("rvt_sync_i", register_state, engine)


def rvt_sync_all():
    return rvt_call("rvt_sync_all")


def rvt_sync_all_gdma():
    return rvt_call("rvt_sync_all_gdma")


def rvt_sync_all_tiu():
    return rvt_call("rvt_sync_all_tiu")


def rvt_cfg_satu(sym_satu, f_satu):
    return rvt_call("rvt_cfg_satu", sym_satu, f_satu)


def rvt_cfg_lanemask(lane_mask):
    return rvt_call("rvt_cfg_lanemask", lane_mask)


def rvt_cfg_round_mode(round_mode):
    return rvt_call("rvt_cfg_round_mode", round_mode)


# Frequently used TIU and DMA instructions.  Descriptors are raw RVT register
# encodings; TileLang does not yet synthesize them from Buffer metadata.
def rvt_fadd(out, lhs, rhs):
    return rvt_call("rvt_fadd", out, lhs, rhs)


def rvt_fsub(out, lhs, rhs):
    return rvt_call("rvt_fsub", out, lhs, rhs)


def rvt_fmul(out, lhs, rhs):
    return rvt_call("rvt_fmul", out, lhs, rhs)


def rvt_fmac(out, lhs, rhs):
    return rvt_call("rvt_fmac", out, lhs, rhs)


def rvt_dma_ld(tensor_desc, global_desc):
    return rvt_call("rvt_dma_ld", tensor_desc, global_desc)


def rvt_dma_st(global_desc, tensor_desc):
    return rvt_call("rvt_dma_st", global_desc, tensor_desc)


def rvt_dma_cp(dst, src):
    return rvt_call("rvt_dma_cp", dst, src)
