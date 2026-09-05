# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU-Kernel FP8 CModel matrix in fresh supervised processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig


_CHIPS = ("sg2260e", "bm1690")
_DTYPES = ("e4m3", "e5m2")
_CASES = (
    "copy",
    "copy-global-to-global",
    "fill-zero",
    "cast-to-fp8",
    "cast-from-fp8",
    "add",
    "sub",
    "mul",
    "add-broadcast",
    "sub-broadcast",
    "mul-broadcast",
    "add-scalar",
    "mul-scalar",
    "rope",
    "gather",
    "gemm-nn-overwrite",
    "gemm-nn-accumulate",
    "gemm-nt-overwrite",
    "gemm-nt-accumulate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chip", choices=_CHIPS, action="append", dest="chips")
    parser.add_argument("--dtype", choices=_DTYPES, action="append", dest="dtypes")
    parser.add_argument("--case", choices=_CASES, action="append", dest="cases")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise RuntimeError("PPL_PROJECT_ROOT must identify the PPL 1.7 SDK")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    repo_root = Path(__file__).resolve().parents[3]
    worker = Path(__file__).with_name("tpu_fp8_ops_worker.py")
    environment = {
        "PPL_PROJECT_ROOT": ppl_root,
        "PYTHONPATH": os.pathsep.join(
            item for item in (str(repo_root), os.environ.get("PYTHONPATH", ""))
            if item),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(scratch),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "runtime_mode": "cmodel",
        "programming_model": "tpukernel",
        "complete": False,
        "cases": {},
    }
    summary_path = output_dir / "summary.json"
    try:
        for chip in args.chips or _CHIPS:
            for dtype in args.dtypes or _DTYPES:
                for case in args.cases or _CASES:
                    key = f"{chip}/tpukernel/{dtype}/{case}"
                    print(f"RUN {key}", flush=True)
                    profiler = TPUInstructionProfiler(
                        TPUProfilingConfig(
                            chip=chip,
                            programming_model="tpukernel",
                            runtime_mode="cmodel",
                            output_dir=output_dir,
                            label=f"{chip}-tpukernel-{dtype}-{case}",
                            timeout_s=args.timeout,
                            postprocess=False,
                        ))
                    command = [
                        sys.executable,
                        str(worker),
                        "--dtype",
                        dtype,
                        "--case",
                        case,
                    ]
                    try:
                        report = profiler.run_cmodel(
                            command, environment=environment)
                        if not report.has_raw_trace:
                            raise RuntimeError(
                                "successful dispatch produced no profiling trace")
                        summary["cases"][key] = {
                            "status": "passed",
                            "artifact_dir": str(report.output_dir),
                            "parser_status": report.parser_status,
                            "raw_trace_file_count": len(report.raw_trace_files),
                            "raw_instruction_count": len(report.raw_instructions),
                        }
                        print(
                            f"PASS {key} raw={len(report.raw_instructions)}",
                            flush=True,
                        )
                    except BaseException as exc:
                        summary["cases"][key] = {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        summary_path.write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        print(
                            f"STOP {key}: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 1
                    summary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
        summary["complete"] = True
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"MATRIX_OK {summary_path}", flush=True)
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
