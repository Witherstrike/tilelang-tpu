# TileLang TPU Operator Capability Contract

This directory contains the machine-readable record of TPU operator support:

- `schema.json` defines the data format.
- `contract.json` records operator semantics, target constraints, failure
  policies, and evidence for each validation stage.
- `validate.py` checks the schema, cross-references, semantic rules, and local
  runtime artifacts.

The contract separates frontend availability, code generation, CModel
correctness, PCIe correctness, and instruction timing. An SDK declaration or
successful source generation is not evidence that a case runs correctly on
hardware.

## Target model

| Target ID | Chip | Programming model | PPL architecture | Cores | Valid |
| --- | --- | --- | --- | ---: | --- |
| `bm1690.tpukernel` | BM1690 | TPU-Kernel | `tpub_7_1` | 8 | Yes |
| `bm1690.rv` | BM1690 | RV Tensor | N/A | N/A | No |
| `sg2260e.tpukernel` | SG2260E | TPU-Kernel | `tpub_7_1_e` | 4 | Yes |
| `sg2260e.rv` | SG2260E | RV Tensor | `tpub_7_1_e` | 4 | Yes |

`cmodel` and `pcie` are runtime modes. They do not select instructions and are
not part of a target ID. The compiler rejects `bm1690.rv` during target
resolution.

Evidence has an explicit order of authority. PPL documentation and headers
describe candidate low-level operations. TileLang source and generated code
show that a mapping exists. Numerical CModel or PCIe results demonstrate that
a specific case executes correctly. General TPU-Kernel documentation alone
does not establish TPUv7, FP8, or target-specific support.

## Capability selectors

Each capability is identified by an operation, variant, data-type bindings,
constraints, and target. Matching must include relevant shape, transpose,
accumulation, broadcast, and layout attributes. Tools must not infer support
from a related type, chip, programming model, or runtime mode.

Target results use these stages:

| Stage | Meaning |
| --- | --- |
| `declared` | The public TileLang frontend can express the selector. |
| `codegen_passed` | Target lowering and source emission accept the selector. |
| `cmodel_numeric_passed` | CModel output matches the required reference. |
| `pcie_numeric_passed` | Hardware output matches the required reference. |
| `sdk_declared` | The SDK documentation or headers declare a candidate operation. |

Stage status is one of `passed`, `historical_passed`, `failed`, `unverified`,
or `not_applicable`. `historical_passed` preserves older evidence but does not
authorize a current hardware run. `support_status="supported"` also requires
the requested validation stage to be `passed`.

## Semantic buffer ABI

The `semantic-buffer.region-abi` invariant applies to all semantic TPU
operations in the contract. A typed buffer crosses this boundary as
`tl.region(BufferLoad, access_mask, logical_extents)`. The former raw
`tir.tvm_access_ptr` compatibility form is not accepted.

Operations other than copy require a zero-origin whole-buffer region. Copy may
use a subregion when its ramp, bounds, and local C-axis origin are valid. Native
code checks the data type, original rank, shape, scope, and storage identity
against the compiler-owned descriptor. A presentation alias is valid only when
the descriptor is unchanged; reshaping or repackaging a buffer into a different
logical descriptor is rejected.

## Runtime evidence

The accepted regression set is tied to implementation commit
`e5774525e3a6e11d0d6010e979203c55181a8872`. Its local artifact root is
`research/artifacts/2026-09-09/final-e5774525/`, which is ignored by Git.

| Matrix | BM1690 CModel | SG2260E CModel | SG2260E PCIe |
| --- | ---: | ---: | ---: |
| Core backend mapping | 28/28 | 56/56 | 56/56 |
| TPU-Kernel FP8 | 42/42 | 42/42 | 42/42 |
| Full TPU-Kernel operations | 152/152 | 146/146 | 146/146 |
| High-level demos | 36/36 | 51/51 | 51/51 |

The 39 accepted summaries contain 553 CModel and 295 PCIe executions. These
counts include intentional overlap between matrices and must not be interpreted
as distinct capabilities. The [test report](../tpu-backend-design/test-report.md)
defines the selected summaries and their acceptance rules.

BM1690 has six additional top-k cases. SG2260E rejects top-k during code
generation because its runtime does not provide the required operation. The
demo matrices also preserve a strict backend boundary: TPU-Kernel covers all
seven operator families, while RV Tensor covers only elementwise operations
and matmul.

FP8 evidence is limited to the exact E4M3 and E5M2 selectors recorded in the
contract. It does not imply support for exceptional values, arbitrary shapes,
BM1690 PCIe, or RV FP8.

## Evidence validation

Mixed-backend summaries identify every result with
`runtime/chip/backend/case_id`. The validator rejects a summary when it finds:

- Missing fields or an unknown backend
- A conflict between summary-level and result-level target fields
- A runtime mismatch, non-passing result, or duplicate canonical key
- Missing, duplicate, or unexpected required cases
- A target count that differs from the contract

The same closed-set rules apply to summaries stored as either `cases{}` or
`results[]`. A `cases{}` collection is not assumed to contain only one
programming model.

Profiling is independent of numerical validation. A raw trace without a
compatible decoder proves neither a duration nor stable performance. Decoded
single-launch timing can help inspect instruction selection, but benchmark
claims require warm-up, repetition, and statistical reporting.

## Rules for automated tools

- CModel scheduling requires `support_status="supported"` and
  `cmodel_numeric_passed.status="passed"`.
- PCIe scheduling additionally requires
  `pcie_numeric_passed.status="passed"`.
- `unverified`, `not_applicable`, `unsupported`, `experimental`, and
  `historical_passed` do not authorize a current PCIe run.
- A missing target result means the operation does not apply to that target.
- Evidence must match the runtime, commit, target, capability, and complete
  case selector. `claim_targets` and `capability_ids` only define candidates.
- `sdk_declared=passed` may guide implementation work but does not promote any
  TileLang validation stage.

When adding support, define the complete selector first, then add frontend and
code-generation checks, CModel evidence, and PCIe evidence in that order. A
runner-recorded artifact must include the full commit, a clean implementation
identity, target case counts, and the required case list.

## Validation commands

```bash
python3 -m json.tool research/tpu-op-contract/schema.json >/dev/null
python3 -m json.tool research/tpu-op-contract/contract.json >/dev/null
python3 research/tpu-op-contract/validate.py
```

On the validation machine, require all referenced local artifacts:

```bash
python3 research/tpu-op-contract/validate.py --require-local-artifacts
```

`validate.py` uses only the Python standard library. In addition to the schema,
it checks ID and reference closure, target consistency, complete target
coverage, operand effects and directions, stage ordering, portable paths,
source locations, compiler registries, and the runtime evidence closed set.
