# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Llama 2 API contracts and opt-in, one-case-per-process numerical tests."""
import argparse
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import tilelang
import tilelang.language as T

TARGET = "tpu -mcpu=sg2260e -tpu-programming-model=rv"
DTYPES = ("float16", "bfloat16", "float32")
CASES = tuple(f"{op}.{dt}" for op in ("rmsnorm", "softmax", "silu", "swiglu", "rope", "transpose",
                                      "embedding", "cache", "repeat", "chain", "mask")
              for dt in DTYPES) + tuple(
                  f"gemm-{layout}.{dt}" for layout in ("nn", "acc") for dt in DTYPES) + tuple(
                      f"gemm-nt.{dt}" for dt in DTYPES[:2])

CASES += tuple(
    f"{op}.{dt}" for op in ("rmsnorm-wide", "softmax-wide", "rope-head") for dt in DTYPES)


def resolve_shape(op, rows, width):
    if op.endswith("-wide"):
        return op[:-5], 1, 4096
    if op == "rope-head":
        return "rope", 65, 128
    return op, rows, width


def make_kernel(op, dtype, rows=3, width=32):
    op, rows, width = resolve_shape(op, rows, width)
    if op == "embedding":

        @T.prim_func
        def kernel(W: T.Tensor((17, width), dtype), I: T.Tensor((rows, 1), "uint32"), O: T.Tensor(
            (rows, width), dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_embedding(O, W, I)

        return kernel
    if op == "transpose":

        @T.prim_func
        def kernel(X: T.Tensor((rows, width), dtype), O: T.Tensor((width, rows), dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_transpose(O, X)

        return kernel
    if op == "mask":

        @T.prim_func
        def kernel(O: T.Tensor((rows, rows + 2), dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_causal_mask(O, 2)

        return kernel
    if op == "cache":

        @T.prim_func
        def kernel(X: T.Tensor((rows, width), dtype), Old: T.Tensor((rows + 4, width), dtype),
                   O: T.Tensor((rows + 4, width), dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_copy(Old, O)
                T.ppl_kv_cache_update(O, X, 2)

        return kernel
    if op == "repeat":

        @T.prim_func
        def kernel(X: T.Tensor((rows, width), dtype), O: T.Tensor((rows, width * 3), dtype)):
            with T.Kernel(1, is_cpu=True):
                T.ppl_repeat_kv(O, X, 3, width // 2)

        return kernel
    if op.startswith("gemm-"):
        trans = op == "gemm-nt"
        accum = op == "gemm-acc"
        bshape = (16, width) if trans else (width, 16)
        allocation_scope = "local.matrix" if dtype == "float32" else "shared"

        @T.prim_func
        def kernel(A: T.Tensor((rows, width), dtype), B: T.Tensor(bshape, dtype), O: T.Tensor(
            (rows, 16), "float32")):
            with T.Kernel(1, is_cpu=True):
                a = T.alloc_shared((rows, width), dtype, scope=allocation_scope)
                b = T.alloc_shared(bshape, dtype, scope=allocation_scope)
                c = T.alloc_shared((rows, 16), "float32", scope=allocation_scope)
                T.ppl_copy(A, a)
                T.ppl_copy(B, b)
                if dtype == "float32":
                    if accum:
                        T.ppl_gemm(a, b, c, transpose_B=trans, accumulate=False)
                else:
                    T.ppl_fill(c, T.float32(1))
                T.ppl_gemm(a, b, c, transpose_B=trans, accumulate=accum)
                T.ppl_copy(c, O)

        return kernel
    if op == "rope":

        @T.prim_func
        def kernel(X: T.Tensor((rows, width), dtype), C: T.Tensor((rows, width // 2), "float32"),
                   S: T.Tensor((rows, width // 2), "float32"), O: T.Tensor((rows, width), dtype)):
            with T.Kernel(1, is_cpu=True):
                x = T.alloc_shared((rows, width), dtype)
                y = T.alloc_shared((rows, width), dtype)
                c = T.alloc_shared((rows, width // 2), "float32")
                s = T.alloc_shared((rows, width // 2), "float32")
                T.ppl_copy(X, x)
                T.ppl_copy(C, c)
                T.ppl_copy(S, s)
                T.ppl_rope(y, x, c, s)
                T.ppl_copy(y, O)

        return kernel

    @T.prim_func
    def kernel(X: T.Tensor((rows, width), dtype), W: T.Tensor((rows, width), dtype), O: T.Tensor(
        (rows, width), dtype)):
        with T.Kernel(1, is_cpu=True):
            x = T.alloc_shared((rows, width), dtype)
            w = T.alloc_shared((rows, width), dtype)
            y = T.alloc_shared((rows, width), dtype)
            T.ppl_copy(X, x)
            T.ppl_copy(W, w)
            if op == "chain":
                z = T.alloc_shared((rows, width), dtype)
                T.ppl_rmsnorm(y, x, w)
                T.ppl_rmsnorm(z, y, w)
                T.ppl_silu(y, z)
                T.ppl_silu(z, y)
                T.ppl_copy(z, y)
            elif op == "rmsnorm":
                T.ppl_rmsnorm(y, x, w)
            elif op == "softmax":
                T.ppl_softmax(y, x)
            elif op == "silu":
                T.ppl_silu(y, x)
            elif op == "swiglu":
                T.ppl_swiglu(y, x, w)
            T.ppl_copy(y, O)

    return kernel


def inputs_reference(op, dtype, rows=3, width=32):
    op, rows, width = resolve_shape(op, rows, width)
    torch.manual_seed(2260)
    dt = getattr(torch, dtype)
    x = torch.randn(rows, width).to(dt)
    w = torch.randn(rows, width).to(dt)
    if op == "embedding":
        weight = torch.randn(17, width).to(dt)
        ids = torch.tensor(
            ([16, 0, 16] * ((rows + 2) // 3))[:rows], dtype=torch.uint32).reshape(rows, 1)
        return [weight, ids], weight[ids.long().flatten()]
    if op == "transpose":
        return [x], x.T.contiguous()
    if op == "repeat":
        return [x], x.reshape(rows, 2, width // 2).repeat_interleave(
            3, dim=1).reshape(rows, width * 3)
    if op == "cache":
        old = torch.randn(rows + 4, width).to(dt)
        ref = old.clone()
        ref[2:2 + rows] = x
        return [x, old], ref
    if op == "mask":
        ref = torch.where(
            torch.arange(rows + 2)[None, :] > torch.arange(rows)[:, None] + 2, float("-inf"), 0.)
        return [], ref.to(dt)
    if op.startswith("gemm-"):
        b = torch.randn((16, width) if op == "gemm-nt" else (width, 16)).to(dt)
        ref = x.float() @ (b.float().T if op == "gemm-nt" else b.float())
        if op == "gemm-acc":
            return [x, b], ref * 2 if dtype == "float32" else ref + 1
        return [x, b], ref
    if op == "rope":
        positions = torch.arange(rows).float() + 127
        theta = positions[:, None] * 10000**(-torch.arange(0, width, 2).float() / width)
        c, s = theta.cos(), theta.sin()
        z = torch.view_as_complex(x.float().reshape(rows, width // 2, 2))
        ref = torch.view_as_real(z * torch.complex(c, s)).reshape(rows, width).to(dt)
        return [x, c, s], ref
    if op == "chain":
        ref = x
        for _ in range(2):
            ref = (ref.float() *
                   torch.rsqrt(ref.float().square().mean(-1, keepdim=True) + 1e-5)).to(dt) * w
        for _ in range(2):
            ref = torch.nn.functional.silu(ref.float()).to(dt)
    elif op == "rmsnorm":
        # Non-unit weights distinguish true weighted RMSNorm from the old demo.
        ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).to(dt) * w
    elif op == "softmax":
        x[:, width // 2:] = float('-inf')
        ref = x.float().softmax(-1).to(dt)
    elif op == "silu":
        ref = torch.nn.functional.silu(x.float()).to(dt)
    else:
        ref = torch.nn.functional.silu(x.float()).to(dt) * w
    return [x, w], ref


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("model", ("rv", "tpukernel"))
def test_lower(case, model):
    op, dt = case.split('.')
    source = tilelang.lower(
        make_kernel(op, dt), target=TARGET.replace("=rv", "=" + model)).kernel_source
    if model == 'rv':
        assert 'rvt_' in source
        assert 'tpu_bdc_' not in source
        if case.startswith('gemm-nn.float32'):
            assert 'rvt_fmm_nn' in source
            assert 'rvt_fmm2_nn' not in source
        if case.startswith('gemm-acc.float32'):
            assert 'rvt_fmma_nn' in source
            assert 'rvt_fmm2a_nn' not in source
    else:
        assert 'rvt_' not in source
        if case.startswith(('gemm-nn.float32', 'gemm-acc.float32')):
            assert 'tpu_bdc_fp32_mm' in source


def test_reject_alias_and_invalid_contracts():
    from tvm import tir
    x = tir.decl_buffer((3, 32), 'float32', scope='shared')
    y = tir.decl_buffer((3, 32), 'float32', scope='shared')
    z = tir.decl_buffer((3, 32), 'float32', scope='shared')
    with pytest.raises(ValueError):
        T.ppl_softmax(x, x)
    with pytest.raises(ValueError):
        T.ppl_rmsnorm(x, y, y, epsilon=-1)
    with pytest.raises(ValueError):
        T.ppl_rope(x, y, x, y)
    with pytest.raises(ValueError):
        T.ppl_repeat_kv(x, y, 0, 16)
    matrix = tir.decl_buffer((3, 32), 'float32', scope='local.matrix')
    with pytest.raises(ValueError, match='local.matrix'):
        T.ppl_add(matrix, matrix, matrix)
    with pytest.raises(ValueError, match='local.matrix'):
        T.ppl_gemm(x, y, z, accumulate=False)


def run(case, runtime, output_dir, rows=3, width=32, model="rv"):
    op, dt = case.split('.')
    op, rows, width = resolve_shape(op, rows, width)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    kernel = tilelang.compile(
        make_kernel(op, dt, rows, width),
        out_idx=-1,
        target=TARGET.replace("=rv", "=" + model),
        runtime_mode=runtime)
    (path / "kernel.c").write_text(kernel.get_kernel_source())
    inputs, ref = inputs_reference(op, dt, rows, width)
    result = kernel(*inputs)
    exact = op in ('embedding', 'transpose', 'cache', 'repeat', 'mask')
    rtol, atol = (0, 0) if exact else {
        'float32': (3e-5, 5e-6),
        'float16': (4e-3, 3e-3),
        'bfloat16': (3e-2, 2e-2)
    }[dt]
    torch.save({'inputs': inputs, 'reference': ref, 'output': result}, path / 'tensors.pt')
    np.save(path / 'output.npy', result.contiguous().view(torch.uint8).numpy())
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)
    (path / 'result.json').write_text(
        json.dumps(
            {
                'case': case,
                'runtime': runtime,
                'programming_model': model,
                'rows': rows,
                'width': width,
                'rtol': rtol,
                'atol': atol,
                'passed': True
            },
            indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--runtime', choices=('cmodel', 'pcie'), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', choices=('rv', 'tpukernel'), default='rv')
    p.add_argument('--rows', type=int, default=3)
    p.add_argument('--width', type=int, default=32)
    a = p.parse_args()
    run(a.case, a.runtime, a.output_dir, a.rows, a.width, a.model)
