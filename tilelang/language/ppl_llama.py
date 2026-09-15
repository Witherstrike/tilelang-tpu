# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Explicit, synchronous TPU building blocks for Llama 2 inference.

These macros expand into checked public tensor operations. Local math uses
FP32; FP16/BF16 conversion occurs in local memory. Shapes are static and the
caller tiles large tensors to fit LMEM. No implicit model-wide allocation.
"""
import math
import tilelang.language as T
from .customize import (_require_local_buffer, _require_global_buffer, _require_rank,
                        _require_dtype, _require_same_dtype, _require_same_shape,
                        _require_distinct_storage, _static_positive_dim, _TPU_BASE_FLOAT_DTYPES)


def _tiles(op, **buffers):
    for name, buf in buffers.items():
        _require_local_buffer(name, buf)
        _require_rank(name, buf, 2)
        _require_dtype(name, buf, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype(op, *buffers.values())
    first = next(iter(buffers.values()))
    for buf in buffers.values():
        _require_same_shape(op, first, buf)
    _require_distinct_storage(op, **buffers)


@T.macro
def _rmsnorm(out, inp, weight, epsilon):
    x = T.alloc_shared(inp.shape, "float32")
    square = T.alloc_shared(inp.shape, "float32")
    r = T.alloc_shared((inp.shape[0], 1), "float32")
    rounded = T.alloc_shared(inp.shape, inp.dtype)
    T.ppl_copy(inp, x)
    T.ppl_mul(square, x, x)
    T.ppl_reduce_sum(square, r, 1)
    T.ppl_mul_C(r, r, T.float32(1.0 / inp.shape[1]))
    T.ppl_add_C(r, r, T.float32(epsilon))
    T.ppl_rsqrt(r, r)
    T.ppl_mul(x, x, r)
    T.ppl_copy(x, rounded)
    T.ppl_mul(out, rounded, weight)


def ppl_rmsnorm(out, inp, weight, epsilon=1e-5):
    """Weighted RMSNorm over W, with Meta's cast-before-weight rounding.

    All operands are distinct local (M,W) tiles of matching float dtype.
    Weight is explicitly broadcast/tiled by the caller to (M,W).
    """
    _tiles("ppl_rmsnorm", out=out, inp=inp, weight=weight)
    if isinstance(
            epsilon,
            bool) or not isinstance(epsilon,
                                    (float, int)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("ppl_rmsnorm epsilon must be finite and positive")
    return _rmsnorm(out, inp, weight, float(epsilon))


@T.macro
def _softmax(out, inp):
    x = T.alloc_shared(inp.shape, "float32")
    y = T.alloc_shared(inp.shape, "float32")
    r = T.alloc_shared((inp.shape[0], 1), "float32")
    w0 = T.alloc_shared(inp.shape, "float32")
    w1 = T.alloc_shared(inp.shape, "float32")
    coeff = T.alloc_shared((64, 32), "float32")
    T.ppl_copy(inp, x)
    T.ppl_reduce_max(x, r, 1)
    T.ppl_subtract(y, x, r)
    T.ppl_exp(y, w0, w1, coeff)
    T.ppl_reduce_sum(y, r, 1)
    T.ppl_div(y, y, r)
    T.ppl_copy(y, out)


def ppl_softmax(out, inp):
    """Stable row Softmax using FP32 accumulation, then cast to input dtype.

    Each row must contain a finite unmasked score; -Inf masks are supported.
    An entirely masked row has undefined probabilities (as in torch.softmax).
    """
    _tiles("ppl_softmax", out=out, inp=inp)
    return _softmax(out, inp)


@T.macro
def _silu(out, inp):
    x = T.alloc_shared(inp.shape, "float32")
    y = T.alloc_shared(inp.shape, "float32")
    w0 = T.alloc_shared(inp.shape, "float32")
    w1 = T.alloc_shared(inp.shape, "float32")
    coeff = T.alloc_shared((64, 32), "float32")
    T.ppl_copy(inp, x)
    T.ppl_sigmoid(y, x, w0, w1, coeff)
    T.ppl_mul(y, y, x)
    T.ppl_copy(y, out)


def ppl_silu(out, inp):
    """SiLU(x) = x * sigmoid(x), computed in FP32 for finite inputs."""
    _tiles("ppl_silu", out=out, inp=inp)
    return _silu(out, inp)


@T.macro
def _swiglu(out, gate, up):
    activated = T.alloc_shared(gate.shape, gate.dtype)
    T.ppl_silu(activated, gate)
    T.ppl_mul(out, activated, up)


def ppl_swiglu(out, gate, up):
    """SiLU(gate) * up, preserving the dtype rounding between the two ops."""
    _tiles("ppl_swiglu", out=out, gate=gate, up=up)
    return _swiglu(out, gate, up)


@T.macro
def _rope(out, inp, cos, sin):
    pair0 = T.alloc_shared((inp.shape[0], 1), inp.dtype)
    pair1 = T.alloc_shared((inp.shape[0], 1), inp.dtype)
    a = T.alloc_shared((inp.shape[0], 1), "float32")
    b = T.alloc_shared((inp.shape[0], 1), "float32")
    c = T.alloc_shared((inp.shape[0], 1), "float32")
    s = T.alloc_shared((inp.shape[0], 1), "float32")
    ac = T.alloc_shared((inp.shape[0], 1), "float32")
    bs = T.alloc_shared((inp.shape[0], 1), "float32")
    bc = T.alloc_shared((inp.shape[0], 1), "float32")
    as_ = T.alloc_shared((inp.shape[0], 1), "float32")
    for j in T.serial(inp.shape[1] // 2):
        T.ppl_copy(inp[0, 2 * j], pair0)
        T.ppl_copy(inp[0, 2 * j + 1], pair1)
        T.ppl_copy(pair0, a)
        T.ppl_copy(pair1, b)
        T.ppl_copy(cos[0, j], c)
        T.ppl_copy(sin[0, j], s)
        T.ppl_mul(ac, a, c)
        T.ppl_mul(bs, b, s)
        T.ppl_mul(bc, b, c)
        T.ppl_mul(as_, a, s)
        T.ppl_subtract(ac, ac, bs)
        T.ppl_add(bc, bc, as_)
        T.ppl_copy(ac, pair0)
        T.ppl_copy(bc, pair1)
        T.ppl_copy(pair0, out[0, 2 * j])
        T.ppl_copy(pair1, out[0, 2 * j + 1])


def ppl_rope(out, inp, cos, sin):
    """Meta Llama 2 adjacent-pair RoPE, with FP32 cosine/sine (M,W/2).

    Tables contain the absolute positions, already expanded across heads.
    This is not the split-half rotate_half convention of converted HF weights.
    """
    _tiles("ppl_rope", out=out, inp=inp)
    width = _static_positive_dim("ppl_rope width", inp.shape[1])
    if width % 2:
        raise ValueError("ppl_rope requires even width")
    for name, buf in (("cos", cos), ("sin", sin)):
        _require_local_buffer(name, buf)
        _require_rank(name, buf, 2)
        _require_dtype(name, buf, {"float32"})
        if tuple(int(x) for x in buf.shape) != (int(inp.shape[0]), width // 2):
            raise ValueError("ppl_rope cos/sin shape must be (M,W/2)")
    _require_distinct_storage("ppl_rope", out=out, inp=inp, cos=cos, sin=sin)
    return _rope(out, inp, cos, sin)


@T.macro
def _transpose(out, inp):
    scalar = T.alloc_shared((1, 1), inp.dtype)
    for i in T.serial(inp.shape[0]):
        for j in T.serial(inp.shape[1]):
            T.ppl_copy(inp[i, j], scalar)
            T.ppl_copy(scalar, out[j, i])


def ppl_transpose(out, inp):
    """Transpose a global rank-2 matrix. Synchronous functional baseline."""
    for name, buf in (("out", out), ("inp", inp)):
        _require_global_buffer(name, buf)
        _require_rank(name, buf, 2)
        _require_dtype(name, buf, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_transpose", out, inp)
    _require_distinct_storage("ppl_transpose", out=out, inp=inp)
    if tuple(int(x) for x in out.shape) != tuple(int(x) for x in reversed(inp.shape)):
        raise ValueError("ppl_transpose expects output shape (W,M)")
    return _transpose(out, inp)


@T.macro
def _causal_mask(out, past):
    zero = T.alloc_shared((1, 1), out.dtype)
    negative = T.alloc_shared((1, 1), out.dtype)
    T.ppl_fill(zero, T.float32(0))
    T.ppl_fill(negative, -T.infinity("float32"))
    for i in T.serial(out.shape[0]):
        for j in T.serial(out.shape[1]):
            if j > past + i:
                T.ppl_copy(negative, out[i, j])
            else:
                T.ppl_copy(zero, out[i, j])


def ppl_causal_mask(out, past_length=0):
    """Global additive causal mask shaped (Q,past_length+Q).

    Masked entries are -Inf, unmasked entries zero, including the cache prefix.
    """
    _require_global_buffer("out", out)
    _require_rank("out", out, 2)
    _require_dtype("out", out, _TPU_BASE_FLOAT_DTYPES)
    if type(past_length) is not int or past_length < 0:
        raise ValueError("ppl_causal_mask past_length must be a nonnegative static integer")
    if int(out.shape[1]) != past_length + int(out.shape[0]):
        raise ValueError("ppl_causal_mask requires shape (Q,past_length+Q)")
    return _causal_mask(out, past_length)


@T.macro
def _cache_update(cache, value, start):
    T.ppl_copy(value, cache[start:start + value.shape[0], 0:value.shape[1]])


def ppl_kv_cache_update(cache, value, start_pos):
    """Copy global (tokens,features) K or V rows into a disjoint cache.

    Features flatten KV heads and head width; start_pos is static, bounded.
    Existing prefix and suffix are preserved. Call separately for K and V.
    """
    for name, buf in (("cache", cache), ("value", value)):
        _require_global_buffer(name, buf)
        _require_rank(name, buf, 2)
        _require_dtype(name, buf, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_kv_cache_update", cache, value)
    _require_distinct_storage("ppl_kv_cache_update", cache=cache, value=value)
    if type(start_pos) is not int or start_pos < 0 or start_pos + int(value.shape[0]) > int(
            cache.shape[0]):
        raise ValueError("ppl_kv_cache_update start_pos is outside cache bounds")
    if int(cache.shape[1]) != int(value.shape[1]):
        raise ValueError("ppl_kv_cache_update feature widths must match")
    return _cache_update(cache, value, start_pos)


@T.macro
def _repeat_kv(out, inp, n_rep, head_dim):
    for t in T.serial(inp.shape[0]):
        for h in T.serial(inp.shape[1] // head_dim):
            for r in T.serial(n_rep):
                T.ppl_copy(inp[t:t + 1, h * head_dim:(h + 1) * head_dim],
                           out[t:t + 1, (h * n_rep + r) * head_dim:(h * n_rep + r + 1) * head_dim])


def ppl_repeat_kv(out, inp, n_rep, head_dim):
    """Global GQA repeat_interleave on heads, flattened as (tokens,heads*D)."""
    for name, buf in (("out", out), ("inp", inp)):
        _require_global_buffer(name, buf)
        _require_rank(name, buf, 2)
        _require_dtype(name, buf, _TPU_BASE_FLOAT_DTYPES)
    _require_same_dtype("ppl_repeat_kv", out, inp)
    _require_distinct_storage("ppl_repeat_kv", out=out, inp=inp)
    n_rep = _static_positive_dim("ppl_repeat_kv n_rep", n_rep)
    head_dim = _static_positive_dim("ppl_repeat_kv head_dim", head_dim)
    if int(inp.shape[1]) % head_dim or tuple(int(x) for x in out.shape) != (int(
            inp.shape[0]), int(inp.shape[1]) * n_rep):
        raise ValueError("ppl_repeat_kv requires whole heads and output (tokens,heads*n_rep*D)")
    return _repeat_kv(out, inp, n_rep, head_dim)
