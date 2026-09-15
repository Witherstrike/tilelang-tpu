# Llama 2 operators on SG2260E

TileLang provides composable `T.ppl_*` building blocks for a Llama 2 forward
pass on SG2260E. The same APIs lower to TPU-Kernel or RV Tensor according to
the target's `-tpu-programming-model` option.

The APIs operate on static, explicitly tiled tensors. They do not load model
weights, schedule a complete model, implement tokenization or sampling, or
manage distributed execution.

## Operator coverage

| Operation | API | Contract |
| --- | --- | --- |
| Embedding | `ppl_embedding(out, weight, indices)` | Global `(N,D)`, `(V,D)`, and `(N,1)` uint32 indices. |
| RMSNorm | `ppl_rmsnorm(out, inp, weight, epsilon)` | Local `(M,W)` tensors; FP32 reduction and cast-before-weight semantics. |
| Softmax | `ppl_softmax(out, inp)` | Stable row-wise Softmax with FP32 reduction. |
| SiLU | `ppl_silu(out, inp)` | FP32 activation math with output cast to the public dtype. |
| SwiGLU | `ppl_swiglu(out, gate, up)` | `cast(silu(gate)) * up`. |
| RoPE | `ppl_rope(out, inp, cos, sin)` | Adjacent-pair rotation; FP32 tables shaped `(M,W/2)`. |
| Causal mask | `ppl_causal_mask(out, past_length)` | Global `(Q,past_length+Q)` additive mask. |
| Transpose | `ppl_transpose(out, inp)` | Functional global rank-2 transpose. |
| KV-cache update | `ppl_kv_cache_update(cache, value, start_pos)` | Bounded global row update. |
| GQA KV repetition | `ppl_repeat_kv(out, inp, n_rep, head_dim)` | Repeats complete flattened KV heads. |
| Projection | `ppl_gemm(A, B, C, ...)` | Local rank-2 GEMM; backend-specific layout constraints apply. |

The public payload dtypes are FP32, FP16, and BF16. Composite normalization
and activation operators convert their intermediates to FP32. Large model
tensors must be tiled to fit local memory.

FP32 GEMM operands use explicit `local.matrix` storage, matching the standard
matrix DMA and MM layouts on both programming models. On RV Tensor it selects
the ISA's standard `fmm` path and requires weights stored as `(K,N)`. FP16 and
BF16 GEMM use ordinary local tiles and the high-performance `fmm2` path; they
also support `(N,K)` weights through `transpose_B=True`.

RoPE follows the adjacent-pair convention used by the original Llama 2 model.
Weights converted for a split-half `rotate_half` convention require the
corresponding permutation and cannot be used interchangeably.

## Validation

Source and ABI tests:

```bash
python -m pytest -q \
  testing/python/jit/test_tpu_llama_ops.py \
  testing/python/jit/test_tpu_rv_essential_ops.py
```

Run every numerical case in a fresh process on CModel before using PCIe:

```bash
python tools/run_llama_validation.py \
  --runtime cmodel \
  --model rv \
  --output-dir results/llama-cmodel
```

After confirming exclusive access to an idle board:

```bash
export TILELANG_TPU_PCIE_RUNTIME_PATH=/opt/tpuv7/tpuv7-runtime_1.9.3/lib
export TILELANG_TPU_DEVICE_ID=0
export TILELANG_TPU_ALLOW_PCIE_LOAD=1

python tools/run_llama_validation.py \
  --runtime pcie \
  --model rv \
  --output-dir results/llama-pcie \
  --smi /opt/tpuv7/tpuv7-current/bin/tpu-smi
```

PCIe execution also requires `TILELANG_TPU_PCIE_RUNTIME_PATH`,
`TILELANG_TPU_DEVICE_ID`, and `TILELANG_TPU_ALLOW_PCIE_LOAD=1`. The compiler
is discovered from the configured PPL SDK. The validation runner serializes
device access, checks board health before and after each case, and stops at the
first failure. Generated outputs belong in an ignored results directory and
are not source-controlled.

After manually investigating the recorded quarantine reason, confirming that
the board is again idle, and removing the quarantine/session markers, one
exact case can be retried in a new output directory with
`--case test_tpu_llama_ops/mask.float16`. Repeat `--case` to select several
cases. A targeted retry does not replace the required complete CModel run.
