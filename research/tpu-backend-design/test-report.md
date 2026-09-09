# TileLang-TPU Test Report

## 1. Evidence set

The runtime evidence in this report is tied to implementation commit
`e5774525e3a6e11d0d6010e979203c55181a8872`. The artifact root is
`research/artifacts/2026-09-09/final-e5774525/`, which is ignored by Git.

The accepted set contains 39 `summary.json` records:

- Eight CModel matrices
- Two complete PCIe matrices
- Fourteen TPU-Kernel PCIe shards
- Fifteen demo PCIe shards

Each accepted summary records the same commit, a clean implementation tree,
`complete=true`, `status=passed`, and no failed or cancelled cases. BM1690 uses
the original four CModel summaries. SG2260E uses the four sequential
`*-cmodel-retry1` summaries. Failed runs, partial prefixes, concurrent CModel
runs, and recovery canaries remain diagnostic records and are not part of the
accepted set.

Each numerical case compiles, loads, and runs in a fresh process, then compares
its output with an exact result or a PyTorch reference. A PCIe summary also
records matching BM1690 and SG2260E CModel inputs and the resolved SDK and
runtime identity.

After the implementation cleanup, the native build, TPU unit tests, target
contract checks, and a CModel smoke test for each valid target combination were
run again. These checks confirm the cleanup but do not change the runtime
evidence tied to `e5774525`.

## 2. Results

| Matrix | BM1690 CModel | SG2260E CModel | SG2260E PCIe |
| --- | ---: | ---: | ---: |
| Core backend mapping | 28/28 | 56/56 | 56/56 |
| TPU-Kernel FP8 | 42/42 | 42/42 | 42/42 |
| Full TPU-Kernel operations | 152/152 | 146/146 | 146/146 |
| High-level demos | 36/36 | 51/51 | 51/51 |
| **Stage total** | **258/258** | **295/295** | **295/295** |

The accepted summaries contain 848 executions: 553 CModel and 295 PCIe. This
number counts matrix executions, not distinct capabilities. Several matrices
intentionally exercise the same instruction mapping at different levels.

The complete PCIe shard sets have no duplicate or missing case IDs:

| Shard group | Contents | Shards | Cases |
| --- | --- | ---: | ---: |
| TPU-Kernel core | Copy, fill/GEMM, and pointwise | 3 | 59 |
| TPU-Kernel extended | Exp, gather, RoPE, rsqrt, and sigmoid | 5 | 15 |
| TPU-Kernel reduction | Sum and max for FP16, BF16, and FP32 | 6 | 72 |
| TPU-Kernel demos | Elementwise, matmul, RMSNorm, split-K RMSNorm, RoPE, SwiGLU, and FlashAttention | 10 | 36 |
| RV demos | Elementwise and matmul | 5 | 15 |

The accepted BF16 reduce-max shard is
`reduce-max-bfloat16-retry1`; earlier failing extended and reduction shards
are excluded.

## 3. Core backend mapping

Each programming model has 28 core cases:

- Four FP16/FP32 copy paths
- Four basic elementwise arithmetic cases
- Twelve FP16/BF16/FP32 W-broadcast arithmetic cases
- Seven dense, broadcast, and negative-infinity max cases
- One matmul case

BM1690 schedules the TPU-Kernel set. SG2260E schedules the same set once for
TPU-Kernel and once for RV Tensor. The three stages completed as follows:

- BM1690 CModel: 28/28
- SG2260E CModel: 56/56
- SG2260E PCIe: 56/56

These results show that the shared frontend semantics select two independent
instruction paths on SG2260E. They do not use runtime fallback.

## 4. TPU-Kernel operation coverage

The 146 cases common to BM1690 and SG2260E are:

| Operation family | Cases | Scope |
| --- | ---: | --- |
| Copy and cast | 23 | Three base floating types, six integer types, S2S, FP32 rank-3, and local casts |
| Fill | 3 | Nonzero FP16, BF16, and FP32 values |
| GEMM | 6 | FP16/BF16 NN overwrite, NN accumulation, and NT overwrite |
| Tensor add/subtract/multiply/divide | 16 | Dense base floating types and FP32 W broadcast |
| Tensor max | 5 | Dense base floating types, FP32 broadcast, and negative infinity |
| Scalar add/multiply | 6 | Two operations and three base floating types |
| Exp and sigmoid | 6 | Two operations and three base floating types |
| Reduce sum/max | 72 | Two operations, three base floating types, and twelve widths |
| Rsqrt | 3 | FP16, BF16, and FP32 |
| RoPE and gather | 6 | Two operations and three base floating types |
| **Common total** | **146** | |

BM1690 adds six ascending and descending top-k cases for FP32, INT32, and
UINT32, producing 152 cases. SG2260E rejects top-k during code generation
because its runtime does not provide the required operation.

The three full TPU-Kernel results are BM1690 CModel 152/152,
SG2260E CModel 146/146, and SG2260E PCIe 146/146. The largest absolute error was
0.0625 in `reduce-sum.bfloat16.w65`, within the declared tolerance.

The final pass review also fixed loop-back-edge liveness in `AddressAssign`.
A buffer initialized outside a repeated loop can no longer share an address
with scratch storage written later in the loop. Parameterized tests cover all
three valid targets, symbolic and nested loops, `While`, zero- and one-trip
loops, and loop-local scratch.

## 5. FP8 coverage

Each E4M3 and E5M2 set contains 21 cases:

- Local and S2S copy
- Zero fill
- Conversion to and from FP32
- Dense and W-broadcast add, subtract, multiply, and max
- Scalar add and multiply
- Gather and RoPE
- NN and NT overwrite and accumulation GEMM

The combined 42-case set completed on BM1690 CModel, SG2260E CModel, and
SG2260E PCIe. The result applies only to the exact shapes, input ranges, types,
and attributes in the summaries. It does not cover nonzero FP8 fill, every
FP16/BF16-to-FP8 conversion, FP8 output GEMM, exceptional values, larger
shapes, BM1690 PCIe, or RV FP8.

## 6. High-level demos

The TPU-Kernel set contains 36 cases: elementwise arithmetic, matmul, RMSNorm,
split-K RMSNorm, RoPE, SwiGLU, and three FlashAttention input variants. The RV
set contains 15 elementwise and matmul cases. Both sets cover FP16, BF16, and
FP32 where applicable.

| Stage | TPU-Kernel | RV Tensor | Total |
| --- | ---: | ---: | ---: |
| BM1690 CModel | 36 | N/A | 36/36 |
| SG2260E CModel | 36 | 15 | 51/51 |
| SG2260E PCIe | 36 | 15 | 51/51 |

The largest absolute error was 0.015625. The largest mean absolute error was
approximately 0.00260836 in BF16 elementwise division. FP32 matmul and
FlashAttention use BF16 matrix operands with FP32 accumulation internally;
they do not establish native FP32 matrix multiplication.

## 7. Profiling

Every core, FP8, and demo CModel case produced raw command records. A compatible
CModel timing decoder was not available, so these records contain no decoded
duration.

SG2260E PCIe profiling used `bigTpuProfile 0.3.5` through
`bigTpuProfile.bmprofile_perfAI.ProfileParser.parse`:

| Matrix | Profiled cases | Decoded intervals | BDC | GDMA |
| --- | ---: | ---: | ---: | ---: |
| Core | 56 | 824 | 218 | 606 |
| FP8 | 42 | 150 | 42 | 108 |
| Demo | 51 | 2,594 | 2,168 | 426 |
| **Total** | **149** | **3,568** | **2,428** | **1,140** |

The full 146-case TPU-Kernel PCIe matrix is a numerical test and has no decoded
timing claim. Each profiling case records one launch, without warm-up,
repetition, confidence intervals, or recorder-overhead correction. The timing
is suitable for instruction inspection and fault diagnosis, not performance
ranking.

## 8. PCIe device handling

The PCIe runner holds the device session lock during preflight, execution, and
postflight checks. After each command, two samples at least 0.25 seconds apart
must report `Active` and 0% utilization within a 10-second monotonic deadline.
A `Fault` state, topology mismatch, invalid health record, probe error, or idle
timeout stops the remaining cases and preserves the quarantine state.

Several earlier complete or partially sharded runs reached a device-health
failure after their numerical check. The retained samples showed `F` for the
main temperature, clock, utilization, and voltage fields. This record does not
identify whether the source was the driver, firmware, runtime, or telemetry.
Those runs and their recovery canaries remain diagnostic evidence only.

After controlled processes exited and the board returned to two consecutive
`Active/0%` samples, the accepted cases were rerun in smaller serial shards.
The accepted shard union covers every listed case once. It establishes the
reported operation results under controlled execution, not long-duration board
stability.

## 9. Remaining work

1. Run the PCIe matrix on BM1690 hardware.
2. Add RV reductions and math operations before enabling RV RMSNorm, RoPE,
   SwiGLU, or FlashAttention.
3. Add tails, dynamic shapes, exceptional values, division by zero, general
   broadcasting, alias cases, and larger GEMM and reduction shapes.
4. Extend FP8 to nonzero fill, additional conversions, FP8 outputs, exceptional
   values, and wider shape coverage.
5. Add explicit SG2260E four-core partitioning, synchronization, and ownership
   tests.
6. Build a separate unrecorded benchmark with warm-up, repetition, and
   statistical thresholds.

New support must first appear in the case registry and the machine-readable
operation contract. Promotion then follows BM1690 CModel, SG2260E CModel, and
SG2260E PCIe in that order.
