# Llama 2 inference operator port for SG2260E RV

This work covers independent `T.ppl_*` tensor APIs for the Llama 2 **forward
pass**, including prefill and cached decoding. It does not load model weights,
implement a serving engine, or qualify full-model accuracy or performance.
Training, distributed collectives, tokenizer, and generation policy are outside
this operator port. Sampling is not a Transformer forward operator; the existing
TPUKernel-only `ppl_topk` is not advertised as an RV implementation.

## Research and semantic audit

Sources, accessed 2026-09-14:

- [Meta's Llama 2 model implementation](https://github.com/meta-llama/llama/blob/main/llama/model.py)
  defines embedding, weighted RMSNorm, adjacent-pair rotary embedding, bias-free
  projections, scaled masked attention, grouped KV repetition, residual addition,
  and the SiLU-gated feed-forward network. Norm and attention reductions use
  FP32. Norm casts back before multiplying the learned weight; SwiGLU also has
  a dtype boundary between SiLU and multiplication. These rounding boundaries
  matter for FP16/BF16 references.
- [Official model card](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md)
  lists 7B, 13B, and 70B variants with 4K context; 70B uses GQA.
- [Llama 2 paper](https://arxiv.org/abs/2307.09288) provides model background.
- [Generation code](https://github.com/meta-llama/llama/blob/main/llama/generation.py)
  keeps tokenization, decoding policy and optional log-probabilities outside the
  model forward. This port's coverage boundary follows that distinction.

## Operator coverage

All shapes are static. Large model tensors need explicit tiling. `local` below
means a compiler-owned TPU local-memory Buffer, not a host tensor. Public JIT
parameters are CPU torch tensors which the adapter transfers to global memory.

| Model operation | Public API | Contract and audit result |
| --- | --- | --- |
| Token embedding | **`ppl_embedding(out, weight, indices)`** | Global `(N,D)`, `(V,D)`, `(N,1) uint32`. Valid IDs in `[0,V)`; repeated IDs and boundary rows tested. FP16/BF16/FP32. New shared semantic ABI `tl.tpu.embedding`; RV DMA H-gather and existing TPUKernel S2S gather. |
| Weighted RMSNorm | **`ppl_rmsnorm(out, inp, weight, epsilon=1e-5)`** | Distinct local `(M,W)` buffers. Weight is caller-expanded to `(M,W)`. FP32 square/reduction/rsqrt, cast back, then weight multiply. Old demo normalized without learned weight and was not a complete model RMSNorm. |
| Q/K/V, output and FFN projections; logits | `ppl_gemm(A,B,C, transpose_B=..., accumulate=...)` | Existing NN/NT FP16/BF16 input semantics fit these projections. FP32 accumulation explicitly initialized; overwrite mode must ignore prior C. Numerical tests cover both layouts and accumulator modes. No transpose-A or FP32 GEMM claim. |
| Attention scale | `ppl_mul_C` | Existing finite compile-time scale, e.g. `1/sqrt(head_dim)`. |
| Residual and additive mask | `ppl_add` | Existing equal-shape or `(M,1)` right-hand broadcast. A `(1,W)` weight/mask is **not** implicitly broadcast; materialize the desired tile. |
| Causal mask | **`ppl_causal_mask(out, past_length=0)`** | Global FP32 `(Q,past_length+Q)`, zero for `j<=past_length+i`, `-Inf` otherwise. Includes cached prefix. |
| Attention probabilities | **`ppl_softmax(out, inp)`** | Distinct local `(M,W)`, stable FP32 max/subtract/exp/sum/div and cast back. Each row requires at least one finite score; `-Inf` masked entries yield zero. Caller must normalize over the whole key sequence, not independently normalize separate K tiles. |
| Rotary position embedding | **`ppl_rope(out, inp, cos, sin)`** | Adjacent pairs: `(a*c-b*s, b*c+a*s)`, FP32 math then cast. Local input/output `(M,2D)`, tables FP32 `(M,D)` with absolute positions already selected and head broadcasting materialized. Host may precompute constant tables. |
| SiLU | **`ppl_silu(out, inp)`** | FP32 computation of `x*sigmoid(x)`, then cast, for finite inputs. |
| SwiGLU gate | **`ppl_swiglu(out, gate, up)`** | `cast(silu(gate))*up`; preserves intermediate dtype rounding. Linear projections remain independent GEMM calls. |
| Matrix/layout transpose | **`ppl_transpose(out, inp)`** | Global rank-2 `(W,M)` from `(M,W)`, synchronous scalar-tile baseline. Batch/head loops and global regions express higher-dimensional layouts. |
| KV cache write | **`ppl_kv_cache_update(cache, value, start_pos)`** | Global `(capacity,features)` and `(tokens,features)`; bounded static offset. Flatten KV heads into features. Writes exactly the selected rows and preserves prefix/suffix. Invoke separately for K and V. |
| KV cache read / tile staging / output cast | `ppl_copy` | Existing global regions preserve parent strides. Casts are local-to-local, so mixed-dtype DMA must be split into transfer and conversion. Local C-offset regions remain rejected. |
| GQA KV repetition | **`ppl_repeat_kv(out, inp, n_rep, head_dim)`** | Global `(tokens,kv_heads*D)` to `(tokens,kv_heads*n_rep*D)`. Repeats each whole head consecutively, not the whole matrix. |

New mathematical APIs are macros over existing checked public operations. Their
scratch allocations and effects remain visible to address/liveness analysis.
No raw RVT calls or unvalidated scalar tensor loads were added. The embedding
extern has explicit frontend, residual region, memory effect, native shape,
dtype, storage and backend validation.

The pre-existing `ppl_rope_add` remains TPUKernel-only: it assembles already
computed terms and is not complete RoPE. The new `ppl_rope` uses Meta's original
adjacent-pair convention. Converted Hugging Face weights may use a split-half
convention; those layouts cannot be interchanged without the corresponding
weight permutation.

Macro expansion creates independent TIR Vars which can have identical name
hints. The public lowering pipeline now gives colliding local allocation names
unique suffixes before writing string-keyed LMEM attributes. Both allocation
definitions and all uses are renamed. Distinct DeclBuffers aliasing one data Var
still fail the native ownership check. Repeated RMSNorm and SiLU in one kernel
exercise this path numerically.

## Validation and limitations

Numerical entry: `testing/python/jit/test_tpu_llama_ops.py`. `pytest` only checks
lowering/contracts; it does not execute a device kernel. `CASES` includes three
float dtypes, distinct RMSNorm weights, partial `-Inf` masks, nonzero-position
RoPE, repeated/boundary embedding IDs, cache sentinels, GQA head order, repeated
macros, NN/NT/accumulating GEMM, 4096-wide normalization, and `(65,128)` RoPE.
Each case saves generated C, inputs/reference/output tensors and numeric status;
failed comparisons retain outputs for diagnosis.

`tools/run_llama_validation.py` runs the old 32-case essential matrix followed
by all Llama cases, one fresh process per case, with a timeout and immediate
stop on any failure. Its manifest fingerprints source (including TVM sources)
and the used chip SDK headers/libraries. Each result includes an output byte
hash. Source changes during a run are rejected. PCIe requires the complete
matching CModel manifest, the existing `/run/lock` exclusive/session/quarantine
protocol, and idle board checks before and after every case. It never resets a
board or removes quarantine markers. After a successful kernel process exits,
it allows up to ten status samples, one second apart, for utilization to settle
and requires two consecutive idle samples. Every sample is retained. Device
faults, retained memory, malformed telemetry and status-command failures still
stop immediately; the pre-launch check remains strict.

Example, after setting `PYTHONPATH`, `TILELANG_LIBRARY_PATH`, `TVM_LIBRARY_PATH`
and `PPL_PROJECT_ROOT` to the same source/build/SDK identity:

```bash
python tools/run_llama_validation.py --runtime cmodel --output-dir results/cmodel
# Only after CModel succeeds and board ownership/recovery is confirmed:
python tools/run_llama_validation.py --runtime pcie --output-dir results/pcie \
  --cmodel-manifest results/cmodel/manifest.json \
  --smi /opt/tpuv7/tpuv7-current/bin/tpu-smi
```

PCIe also needs the documented `PPL_RISCV_CC`,
`TILELANG_TPU_PCIE_RUNTIME_PATH`, `TILELANG_TPU_DEVICE_ID` and
`TILELANG_TPU_ALLOW_PCIE_LOAD=1` configuration. Shape bounds, valid token IDs,
finite model activations, LMEM capacity and exact tiling are caller obligations.
The functional transpose and column-wise RoPE are not performance implementations.
No full checkpoint inference, arbitrary dynamic shapes, multi-core scaling,
FP8, strict IEEE transcendental behavior, or training support is claimed.

## Verified results (2026-09-15)

Implementation revision: `74096491774009fb29e69bb9d770386a5cf609ab`.
[Machine-readable results and output hashes](validation/sg2260e-rv-llama2-results.json)
record these separately scoped checks:

- **547 static tests passed**, 8 SDK-conditional cases skipped locally; the
  independent build used clean TVM `a8a54d2b1f43c23a47f2fc08779654918eae6464`.
- **CModel 78/78 passed**: original 32 essential cases plus 46 Llama cases,
  including 4096-wide RMSNorm/Softmax and `(65,128)` RoPE. The clean-TVM and
  original-source builds produced identical generated C and output bytes for
  all 78 cases. The copied checkout's pre-existing TVM edits were preserved and
  are not part of the implementation commit.
- **Remote PCIe compilation/linking 46/46 passed**, without loading device
  libraries. All 46 generated kernels match the CModel sources byte-for-byte.
  Remote Llama API/registry checks passed 135 tests.
- **PCIe numerical execution 78/78 passed on device 0**, covering the same
  32 essential and 46 Llama cases. All generated C and output bytes match the
  complete CModel rerun with the final supervisor source fingerprint.
  Final telemetry was Active / 0% / 0MB; no session or quarantine marker remains.
- **Supervisor 12/12 tests passed locally and remotely**, including six new
  tests for bounded utilization settling and immediate rejection of faults or
  retained memory.

The initial PCIe attempt completed `fill.float16` with exact output, then stopped
because the immediate utilization sample was 9% despite 0MB device memory. Its
owner process exited, no visible device holders remained, and two subsequent
samples were idle. With user authorization and the exclusive lock held, the
admin2-owned markers were archived and cleared, preserving the lock file. No
hardware reset occurred. The revised supervisor then passed a fresh full CModel
matrix before the successful full PCIe matrix. Original attempt logs and all
status samples remain archived alongside the final validation artifacts.

The common extern capability contract registers embedding and its compiler
stages. Its stricter runtime-claim format has not been populated from this
separate result format; numerical capability rows are intentionally not promoted
using test-source evidence alone.
