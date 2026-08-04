# PPL SG2260E RV golden corpus

The checked-in corpus under `testing/ppl_rv_golden/` records the PPL
1.7.122 SG2260E RV compiler contract used by TileLang. It contains minimal
programs for copy, fill, elementwise, GEMM, reduction, special function,
gather, and top-k, plus a reproducible generator and compiler-produced
artifacts.

## Reproduce

```bash
export PPL_PROJECT_ROOT=../ppl_v1.7.122-g05ebfb36-20260528
bash testing/ppl_rv_golden/generate.sh /tmp/ppl-rv-golden
```

The script invokes `ppl-compile` with `--chip sg2260e -D__sg2260e__ --rv
--O3 --print-ir`. Seven cases produce frontend, opt, final and RV MLIR plus
device C. Top-k reproducibly exits with code 23 in PPL's RV instruction
optimization pass; its frontend/opt/final artifacts and diagnostic are retained
as a partial golden. See the corpus README and
`artifacts/topk/compile.log` for the precise failure.

## Rules consumed by TileLang

| rule | golden observation | TileLang invariant |
|---|---|---|
| RULE-RV-001 | Global operands use GR 32+ and `CONTINUOUS_LAYOUT` | A global tensor view carries class GR, an ID starting at 32, and continuous layout |
| RULE-RV-002 | Local operands use TR 8+ and normally `HW_ALIGN_LAYOUT` | A local tensor view carries class TR, an ID starting at 8, and HW-aligned layout |
| RULE-RV-003 | Partial copy/global regions can require `FREE_LAYOUT` | Layout belongs to an operand view, not only to the backing buffer |
| RULE-RV-004 | Scalar operands are configured through control registers | Fill and scalar elementwise operands carry explicit CR descriptors starting at 1 |
| RULE-RV-005 | final-to-RV lowering replaces tensors with register descriptors | TIR legalizes each supported `ppl.*` operation to `ppl.rv.*` only after explicit views exist |
| RULE-RV-006 | Independent physical register IDs can be permuted between runs | Tests compare class/layout/dataflow and deterministic TileLang allocation, not PPL C byte identity |
| RULE-RV-007 | Local allocation is banked and permits lifetime-based reuse | This iteration records the PPL rule; exact bank/liveness allocation remains a later codegen task |

The authoritative details, compiler hash, shapes, strides, local-bank examples,
and operator-to-RV-instruction mapping live in
`testing/ppl_rv_golden/README.md`. Generated artifacts must only be updated by
rerunning the generator against a recorded PPL release; they must not be edited
to match TileLang output.

## Known boundary

Top-k is legalizable at the TileLang TIR layer because its three operand views
are unambiguous. PPL 1.7.122 then rejects its own RV MLIR because
`ppl.hau.topK` still verifies memref operands after the RV pass has converted
them to integer register descriptors. Consequently top-k RV C generation and
cmodel execution remain blocked outside TileLang; the failure is neither a
`.pl` syntax problem nor handled by C text rewriting.
