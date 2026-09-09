# TileLang-TPU Status and Upstream Roadmap

## Current status

TileLang-TPU now separates hardware selection from runtime selection:

- A compilation target is `chip + programming_model`.
- BM1690 supports TPU-Kernel.
- SG2260E supports TPU-Kernel and RV Tensor.
- `cmodel` and `pcie` select the runtime and do not change instruction
  selection.

Shared frontend operations lower to neutral `tl.tpu.*` semantics before the
compiler selects TPU-Kernel or RV Tensor. Operations that exist only in
TPU-Kernel use `tl.tpukernel.*`. An unsupported mapping is rejected during
compilation; there is no silent fallback to another programming model.

The compiler uses the resolved target consistently in target passes, residual
IR verification, address assignment, code generation, PPL compilation, and
runtime loading. BM1690 uses eight cores and PPL architecture `tpub_7_1`.
SG2260E uses four cores and `tpub_7_1_e`.

Current numerical coverage includes:

- Core copy, elementwise, broadcast, maximum, and matrix-multiplication paths
  for both programming models on SG2260E
- A wider TPU-Kernel operation matrix, including reductions, math operations,
  gather, RoPE, top-k where available, and selected FP8 operations
- High-level elementwise, matmul, RMSNorm, split-K RMSNorm, RoPE, SwiGLU, and
  FlashAttention examples
- BM1690 and SG2260E CModel tests, followed by serialized SG2260E PCIe tests

The exact supported selectors and validation stages are recorded in
`research/tpu-op-contract/contract.json`. Fixed demo workloads do not imply
support for arbitrary shapes, dynamic dimensions, multicore schedules, or
every data-type combination.

## Related upstream projects

The following projects provide useful design references:

| Project | Relevant design |
| --- | --- |
| [TileLang](https://github.com/tile-ai/tilelang) | Backend dialects, code-generator registration, target resolution, region-based built-ins, reduction infrastructure, and buffer-initialization checks |
| [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend) | Explicit compute scopes, synchronization, pipelines, buffer reuse, multicore communication, and a broad operator set |
| [TileOPs](https://github.com/tile-ai/TileOPs) | Separation of high-level operator contracts from kernels, with manifests for signatures, workloads, references, tests, and benchmarks |
| [SOPHGO PPL](https://github.com/sophgo/PPL) | TPU-Kernel compilation and target-specific SDK interfaces |

TileLang-TPU should follow TileLang's backend registration and semantic IR
boundaries instead of adding TPU conditions to generic compiler code. The
Ascend backend is most useful as a reference for modeling dependencies and
multicore execution; its hardware-specific abstractions should not be copied
directly. TileOPs provides a practical model for connecting operator metadata,
tests, and performance workloads.

## Compiler boundaries

### Target selection

Target resolution validates the chip and programming model together. Runtime
mode remains separate so the same compiled semantics can run in CModel or on a
PCIe device. This prevents runtime configuration from changing generated
instructions.

The target-dependent pass runs after shared semantic lowering. It converts
neutral operations to the selected instruction family and leaves
TPU-Kernel-only operations explicit. A residual verifier rejects any operation
that has no legal mapping for the target.

### Buffer ABI

Semantic operations receive typed regions that preserve data type, rank,
shape, scope, access mode, and storage identity. Non-copy operations require a
whole-buffer region. Copy may use a checked contiguous subregion. Raw pointer
compatibility forms are not supported because they lose the information needed
to validate views and aliases.

### Instruction selection

TPU-Kernel and RV Tensor use separate instruction selectors. They share only
target-neutral utilities such as type formatting, shape extraction, and common
diagnostics. This keeps backend-specific legality rules in the selector that
owns them while avoiding duplicate general-purpose code.

## Operator coverage and remaining gaps

| Operator family | Current coverage | Main gaps |
| --- | --- | --- |
| Copy and cast | Base floating types, selected integer types, FP32 local casts, selected FP8 conversions; basic RV FP16/FP32 copy paths | More cross-type conversions, RV data types, rank and stride coverage |
| Fill | Nonzero FP16/BF16/FP32 and selected zero-fill paths | FP8 nonzero fill and broader value semantics |
| GEMM | FP16/BF16 TPU-Kernel variants, selected FP8 variants, FP16/BF16/FP32 demo interfaces, basic RV matmul | Batch GEMM, transpose-A, general tails, native FP32 operands, wider RV variants |
| Elementwise | FP16/BF16/FP32 dense and W broadcast on both models; selected TPU-Kernel FP8 | General broadcast, compare, select, clamp, and FP8 division |
| Scalar operations | Add and multiply for base floating types and selected FP8 formats | Subtract, divide, dynamic scalar values, and explicit saturation rules |
| Math | TPU-Kernel exp, sigmoid, and reciprocal square root | RV implementations, more functions, and documented error ranges |
| Reduction | TPU-Kernel row sum and maximum over tested boundary widths | Neutral reduction IR, RV support, more axes, tails, minimum, and arg-reduction |
| Gather and RoPE | TPU-Kernel base floating types and selected FP8 formats | RV support, index policy, and general layouts |
| Top-k | BM1690 FP32/INT32/UINT32 ascending and descending cases | BM1690 PCIe, wider K/length limits; SG2260E remains unsupported |
| Composite operators | TPU-Kernel RMSNorm, split-K RMSNorm, RoPE, SwiGLU, and FlashAttention for fixed workloads | General shapes, tails, dynamic parameters, multicore schedules, and RV lowering |

FP8 support is always recorded by exact format and operation. E4M3 and E5M2
results cannot be generalized to exceptional values, saturation modes,
untested shapes, BM1690 PCIe, or RV Tensor. Likewise, the FP32 matmul demo uses
BF16 matrix operands internally and must not be presented as native FP32
matrix multiplication.

## Priorities

Work is ordered by four factors:

1. How many operations could be affected by an incorrect address, dependency,
   or fallback rule
2. How broadly the feature is reused by normalization, softmax, attention, and
   other model operators
3. Whether source and CModel tests can reduce risk before PCIe execution
4. Whether the change reduces duplicated target rules and eases upstream
   maintenance

Compiler correctness and reusable primitives therefore come before isolated
high-level kernels or performance tuning.

## P0: compiler correctness and structure

### Central operator specification

Why: Operand roles, type rules, effects, and dispatch keys currently span the
frontend, verifier, address assignment, and both instruction selectors. A
missed update can allow one stage to accept an operation that another stage
interprets differently.

Work:

- Define a `TpuOpSpec` registry with semantic names, frontend aliases, operand
  roles, effects, type and layout constraints, workspace requirements,
  supported programming models, and instruction-selection keys.
- Generate or validate frontend guards, verifier entries, address effects, and
  selector dispatch from the same record.
- Express chip-specific behavior as capability predicates on the full selector.
- Generate positive and negative source tests and capability-contract entries.

Done when removing or changing one registration produces one clear compiler
error, operand effects agree across every stage, and the machine-readable
contract has no missing references.

### One source for target capabilities

Why: Python, native build code, and demo scheduling currently mirror valid
chip and programming-model combinations. Hand-maintained copies can drift.

Work: Generate the Python, C++, and test registries from one description that
contains the PPL architecture, compiler macros, core count, programming models,
and chip features.

Done when build-time checks prove that every layer accepts and rejects the same
target combinations.

### Tails, dynamic shapes, and aliases

Why: Current kernels use static, tile-aligned shapes. Rounding an unsupported
shape up can issue an out-of-bounds DMA command and may leave a PCIe device
unresponsive.

Work:

- Add a planner that handles static, non-divisible tiles with valid
  regions and controlled padding or cropping.
- Record masking support for every DMA and compute operation.
- Prove region bounds and alias rules before code generation.
- Add symbolic dimensions and runtime guards only after static tails are
  reliable.

Done when dimensions immediately below, at, and above EU and tile boundaries
pass numerical tests, while unsafe or unprovable regions fail during
compilation.

### Dependency-safe compiler passes

Why: GPU vectorization and software-pipeline assumptions do not describe TPU
DMA, BDC, or RV Tensor dependencies. Enabling them without a TPU dependency
model can change correct programs into incorrect ones.

Work:

- Model command tokens, buffer versions, producer-consumer edges, and barrier
  visibility for DMA, BDC, and RV Tensor commands.
- Reject read-after-write, write-after-read, and write-after-write hazards that
  lack a valid dependency.
- Define TPU vector lane, ramp, load/store, and reduction rules before enabling
  vector passes.
- Compare serial and pipelined schedules in CModel and on hardware.

Done when unsafe schedules fail at compile time and accepted schedules have
both numerical evidence and an explainable command trace.

### Standard TileLang operator API

Why: Public `T.ppl_*` names expose a vendor toolchain in otherwise portable
TileLang programs.

Work: Lower standard `T.copy`, `T.fill`, `T.gemm`, elementwise, and reduction
operations to the TPU semantic registry. Keep hardware-specific arguments in
explicit extensions or target capabilities rather than duplicating the API.

Done when standard entry points generate the same semantic IR and numerical
results as the current TPU entry points, after which only one public semantic
form remains.

## P1: reusable model operations

### Conversion and quantization

Define conversion support by source type, destination type, rounding,
saturation, scope, and programming model. Implement floating-point and integer
conversion plus quantize/dequantize operations. Reject every unlisted
combination. Keep E4M3 and E5M2 as separate capabilities.

### General elementwise and broadcast

Replace the fixed W-broadcast variant with a checked broadcast planner that can
represent row, column, and batch broadcasting. Add minimum, maximum, compare,
select, and clamp with explicit in-place, NaN, infinity, and division-by-zero
rules.

### Neutral reduction

Introduce `tl.tpu.reduce` with an axis, initial value, accumulator type,
workspace, and tail policy. Reuse the current TPU-Kernel implementation and add
RV Tensor mappings. Start with contiguous axes, then add multiple axes and
cross-tile reduction.

### Math and activation

Add documented error limits and workspace planning for exp, exp2, log,
reciprocal square root, sigmoid, GELU, and SiLU. A native instruction and a
composite lowering are separate backend implementations and require separate
validation.

### Normalization and softmax

Promote the current RMSNorm reference to a standard operation with an axis,
accumulator type, epsilon, workspace, and tail contract. Then add LayerNorm and
stable max-subtracted softmax. Split-K partial results and merge ownership must
be explicit in the launch plan.

### Layout operations

Separate zero-cost views from physical data movement. Validate global and local
strides, contiguity, alignment, and aliases. Implement 2D transpose before more
complex blocked layouts.

## P2: multicore execution and complex operators

### Explicit launch plans

A core count alone is not a partitioning strategy. Define per-core ranges,
address offsets, output ownership, cross-core synchronization, reduction, and
failure cleanup. Begin with write-disjoint elementwise work, then matmul, then
reduction.

### Asynchronous execution

Add double-buffer versions, explicit wait and signal operations, capacity
checks, and schedule legality on top of the P0 dependency model. Use profiling
to inspect overlap, but use recorder-free warm-up and repeated runs for
performance measurements.

### Batch GEMM and attention

Define batched descriptors and partitioning before optimizing attention. Use
the current FlashAttention implementation as the reference, then add stable
online softmax, masks, causal and tail handling, and multicore pipelines. Any
specialized path must match the reference for the same workload.

### Indexed and ordered operations

Move gather to neutral TPU semantics and define out-of-range index behavior.
Extend BM1690 top-k tests to K=1, K=length, limit cases, and PCIe. Keep
SG2260E top-k disabled until a valid runtime implementation exists. Add an RV
mapping only after its instruction and algorithm contracts are defined.

### Convolution

Start with an im2col-plus-GEMM reference covering NCHW, NHWC, padding,
dilation, and non-aligned boundaries. Introduce fused or dedicated paths only
through target capabilities.

### Broader FP8 support

Extend the existing regression set in small steps: nonzero fill, FP16/BF16
conversions, FP8 output GEMM, exceptional values, saturation behavior, and
larger shapes. Validate source and CModel behavior before running a small,
stop-on-error PCIe batch. RV FP8 requires its own descriptor, rounding,
saturation, and accumulator contract.

## P3: upstream integration and maintenance

### Backend registration

Move TPU target resolution, passes, language lowering, and code generation
behind TileLang's backend registry. Preserve behavior during the move, then use
the central operator specification to remove repeated string dispatch and
target checks. Re-run the BM1690 and SG2260E CModel baselines after each step.

The native build entry point should also require every `PrimFunc` to carry the
same complete target as the build request. This closes a path in which a direct
FFI caller could omit the function target and bypass normal lowering checks.

### Reproducible artifacts

Record compiler identity, SDK identity, target, source hash, case selector,
runtime mode, and numerical policy in one artifact manifest. Separate build
cache keys from runtime results so a cached binary cannot be reused for a
different chip or programming model.

### Sustainable testing

Maintain three layers:

- Fast source and verifier tests for all legal and illegal selectors
- CModel numerical matrices for every supported target
- Serialized, stop-on-error PCIe matrices promoted only after CModel passes

Keep correctness profiling separate from performance benchmarks. Instruction
traces diagnose mappings; performance reports require warm-up, repetition,
stable device conditions, and summary statistics.

## Delivery sequence

| Phase | Main result | Completion criterion |
| --- | --- | --- |
| P0 | One target model, one operator specification, safe regions and dependencies | Compiler stages agree on every legal selector and reject unsafe programs before runtime |
| P1 | Portable conversion, broadcast, reduction, math, normalization, and layout operations | Reusable operators pass the required CModel matrices on each valid target |
| P2 | Multicore, asynchronous execution, attention, indexing, convolution, and broader FP8 | Optimized paths match reference implementations and pass staged PCIe tests |
| P3 | TileLang backend integration, reproducible artifacts, and long-term testing | The backend can be maintained and reviewed without TPU branches in generic compiler code |

Every new capability should state its frontend semantics, exact target and
data-type constraints, generated instruction path, CModel result, PCIe result,
and failure behavior. Documentation or a neighboring passing case is not a
substitute for evidence for the exact selector.
