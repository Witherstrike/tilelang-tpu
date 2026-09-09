# RV Tensor Test Procedure

This document defines the test order and acceptance rules for the SG2260E RV
Tensor backend. Detailed cross-backend results are recorded in
[`../tpu-backend-design/test-report.md`](../tpu-backend-design/test-report.md).

## Scope

The RV core matrix covers copy, elementwise add/subtract/multiply/divide, max,
and matmul. The high-level demo matrix covers elementwise arithmetic and
matmul. Each applicable demo runs with FP16, BF16, and FP32.

The reference evidence is tied to commit
`e5774525e3a6e11d0d6010e979203c55181a8872`. Its ignored artifact directory is
`research/artifacts/2026-09-09/final-e5774525/`.

| Stage | Target | Core cases | Demo cases |
| --- | --- | ---: | ---: |
| BM1690 CModel | TPU-Kernel reference | 28 | 36 |
| SG2260E CModel | TPU-Kernel and RV Tensor | 56 | 51 |
| SG2260E PCIe | TPU-Kernel and RV Tensor | 56 | 51 |

The SG2260E totals include both programming models. RV contributes 28 core
cases and 15 demo cases at each SG2260E stage.

## Test order

Use one clean commit and a new artifact directory. Run:

1. BM1690 CModel
2. SG2260E CModel
3. SG2260E PCIe

PCIe cases must have matching BM1690 and SG2260E CModel records from the same
source and toolchain. A failed or incomplete stage cannot be used as PCIe input.

## CModel commands

```bash
export TILELANG_RV_TEST_RUN="research/artifacts/rv-test"

python testing/python/jit/tpu_core_ops_matrix.py \
  --runtime-mode cmodel \
  --chip bm1690 \
  --output-dir "${TILELANG_RV_TEST_RUN}/core-bm1690-cmodel"

python testing/python/jit/tpu_core_ops_matrix.py \
  --runtime-mode cmodel \
  --chip sg2260e \
  --output-dir "${TILELANG_RV_TEST_RUN}/core-sg2260e-cmodel"
```

The SG2260E command schedules both TPU-Kernel and RV Tensor. To inspect only RV
cases during development, add `--programming-model rv`.

## PCIe command

Run PCIe on one device and keep all cases serial:

```bash
python testing/python/jit/tpu_core_ops_matrix.py \
  --runtime-mode pcie \
  --chip sg2260e \
  --device-id 0 \
  --allow-pcie \
  --allow-pcie-profile \
  --all-pcie-cases \
  --bm-cmodel-summary "${TILELANG_RV_TEST_RUN}/core-bm1690-cmodel/summary.json" \
  --sg-cmodel-summary "${TILELANG_RV_TEST_RUN}/core-sg2260e-cmodel/summary.json" \
  --output-dir "${TILELANG_RV_TEST_RUN}/core-sg2260e-pcie"
```

Add the decoder options when per-instruction time is required:

```bash
--require-decoded-timing \
--pcie-decoder-python /path/to/decoder/python \
--pcie-decoder-pythonpath /path/to/decoder/package
```

## Acceptance rules

Each case compiles and runs in a fresh process. A result is accepted only when:

- The output shape and data type match the reference
- Every output value is finite
- The numerical error is within the case tolerance
- The raw instruction record is present and non-empty
- Strict profiling, when requested, contains valid nanosecond intervals
- The device returns to `Active` with 0% utilization after the launch

The runner stops at the first compile, runtime, numerical, profiling, process,
or device-health error. It terminates the controlled process group within a
fixed deadline and records a quarantine marker when device recovery cannot be
confirmed.

## Evidence boundaries

- CModel raw records on the reference host do not include decoded device time.
- PCIe profiles contain one recorded launch per case and are diagnostic data,
  not steady-state performance measurements.
- BM1690 PCIe is outside this test environment.
- RV reduction, normalization, activation, attention, RoPE, and FP8 paths are
  outside the current implementation.
- Static example shapes do not imply support for tails, dynamic shapes, general
  broadcasting, multi-core scheduling, or asynchronous pipelines.

Add a new capability to the machine-readable contract before promoting its
result. The selector must include the chip, programming model, data types,
shape or layout constraints, and operation attributes.
