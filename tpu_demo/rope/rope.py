# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Interleaved rotary-position embedding for TPU-Kernel."""

from typing import Optional, Tuple

import tilelang.language as T
import torch

from tpu_demo.common import (comparison, compile_and_launch, result_payload, tolerance, torch_dtype,
                             validate_dimensions, validate_exact_tiling, validate_selection)


def build_rope(*,
               rows: int = 8,
               width: int = 32,
               block_rows: int = 4,
               block_width: int = 32,
               dtype: str = "float16"):
    torch_dtype(dtype)
    validate_dimensions(
        "rope", rows=rows, width=width, block_rows=block_rows, block_width=block_width)
    validate_exact_tiling("rope", ("rows", rows, block_rows), ("width", width, block_width))
    if width % 2 or block_width % 2:
        raise ValueError("RoPE width and block width must both be even")

    @T.prim_func
    def kernel(source: T.Tensor((rows, width), dtype), cosine: T.Tensor((rows, width), dtype),
               sine: T.Tensor((rows, width), dtype), destination: T.Tensor((rows, width), dtype)):
        with T.Kernel(
                T.ceildiv(rows, block_rows), T.ceildiv(width, block_width),
                is_cpu=True) as (bx, by):
            shape = (block_rows, block_width)
            x = T.alloc_shared(shape, dtype)
            cos = T.alloc_shared(shape, dtype)
            sin = T.alloc_shared(shape, dtype)
            x_cos = T.alloc_shared(shape, dtype)
            x_sin = T.alloc_shared(shape, dtype)
            neg_x = T.alloc_shared(shape, dtype)
            neg_x_sin = T.alloc_shared(shape, dtype)
            output = T.alloc_shared(shape, dtype)
            T.ppl_copy(source[bx * block_rows, by * block_width], x)
            T.ppl_copy(cosine[bx * block_rows, by * block_width], cos)
            T.ppl_copy(sine[bx * block_rows, by * block_width], sin)
            T.ppl_mul(x_cos, x, cos)
            T.ppl_mul(x_sin, x, sin)
            T.ppl_mul_C(neg_x, x, T.float32(-1.0))
            T.ppl_mul(neg_x_sin, neg_x, sin)
            T.ppl_rope_add(output, x_cos, neg_x_sin, x_cos, x_sin)
            T.ppl_copy(output, destination[bx * block_rows, by * block_width])

    return kernel


def _cosine_sine(rows: int, width: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    frequencies = 10000.0**(-2 * torch.arange(width // 2, dtype=torch.float32) / width)
    theta = positions * frequencies.unsqueeze(0)
    cos_half, sin_half = torch.cos(theta), torch.sin(theta)
    cosine = torch.repeat_interleave(cos_half, 2, dim=1).to(dtype).contiguous()
    sine = torch.repeat_interleave(sin_half, 2, dim=1).to(dtype).contiguous()
    return cosine, sine


def _reference(source: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    source_f32, cosine_f32, sine_f32 = source.float(), cosine.float(), sine.float()
    output = torch.empty_like(source_f32)
    output[:, 0::2] = (
        source_f32[:, 0::2] * cosine_f32[:, 0::2] - source_f32[:, 1::2] * sine_f32[:, 1::2])
    output[:, 1::2] = (
        source_f32[:, 1::2] * cosine_f32[:, 1::2] + source_f32[:, 0::2] * sine_f32[:, 0::2])
    return output.to(source.dtype)


def run(*,
        dtype: str,
        chip: str,
        programming_model: str,
        runtime_mode: str,
        allow_pcie: bool = False,
        device_id: Optional[int] = None,
        seed: int = 0) -> dict:
    torch_dtype(dtype)
    validate_selection(
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        supports_rv=False,
        allow_pcie=allow_pcie,
        device_id=device_id)
    rows, width = 8, 32
    generator = torch.Generator().manual_seed(seed)
    host_dtype = torch_dtype(dtype)
    source = torch.randn((rows, width), generator=generator).to(host_dtype)
    cosine, sine = _cosine_sine(rows, width, host_dtype)
    destination = torch.zeros_like(source)
    timing = compile_and_launch(
        build_rope(rows=rows, width=width, dtype=dtype), (source, cosine, sine, destination),
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode)
    expected = _reference(source, cosine, sine)
    atol, rtol = tolerance(dtype, "rope")
    metrics = comparison(destination, expected, atol=atol, rtol=rtol)
    return result_payload(
        operation="rope",
        dtype=dtype,
        chip=chip,
        programming_model=programming_model,
        runtime_mode=runtime_mode,
        metrics=metrics,
        timing=timing,
        parameters={
            "rows": rows,
            "width": width,
            "block_rows": 4,
            "block_width": 32,
            "seed": seed
        })
