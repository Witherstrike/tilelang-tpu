# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU-Kernel FP8 capability matrix in supervised fresh processes.

CModel remains the safe default. PCIe requires an explicit chip and device,
separate load and recorder acknowledgements, and uses the same one-launch
profiling supervisor as the non-FP8 core matrix. Numerical correctness and a
non-empty raw trace are always required; decoded nanosecond timing is an
additional opt-in acceptance layer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Optional

from tilelang.jit import TPUInstructionProfiler, TPUProfilingConfig

if __package__:
    from .tpu_matrix_common import git_source_identity
    from .tpu_profile_matrix_common import (
        profile_report_summary as _report_summary,
        validate_profile_report as _validate_profile_report,
    )
else:
    from tpu_matrix_common import git_source_identity
    from tpu_profile_matrix_common import (
        profile_report_summary as _report_summary,
        validate_profile_report as _validate_profile_report,
    )

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
_PCIE_ENVIRONMENT_VARIABLES = (
    "TILELANG_TPU_ALLOW_PCIE_LOAD",
    "TILELANG_TPU_ALLOW_PCIE_PROFILE",
    "TILELANG_TPU_DEVICE_ID",
    "BMLIB_ENABLE_ALL_PROFILE",
    "PROFILE_RECORD_SIZE",
    "PROFILE_BOOK_KEEPING",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"), default="cmodel")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chip", choices=_CHIPS, action="append", dest="chips")
    parser.add_argument("--dtype", choices=_DTYPES, action="append", dest="dtypes")
    parser.add_argument("--case", choices=_CASES, action="append", dest="cases")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--allow-pcie-profile", action="store_true")
    parser.add_argument(
        "--require-decoded-timing",
        action="store_true",
        help=("require a successful decoder preflight and at least one valid nanosecond "
              "device-command interval for every PCIe case"),
    )
    parser.add_argument(
        "--pcie-decoder-python",
        type=Path,
        help="Python executable used only by the offline PCIe decoder child",
    )
    parser.add_argument(
        "--pcie-decoder-pythonpath",
        type=Path,
        action="append",
        default=[],
        help="repeatable import root used only by the offline PCIe decoder child",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float)) or
            not math.isfinite(float(args.timeout)) or args.timeout <= 0):
        raise ValueError("--timeout must be a positive finite number")
    decoder_options = bool(args.pcie_decoder_python or args.pcie_decoder_pythonpath)
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_profile:
            raise RuntimeError("PCIe requires both --allow-pcie and --allow-pcie-profile")
        if (args.device_id is None or args.device_id < 0 or args.device_id > 2**31 - 1):
            raise RuntimeError("PCIe requires a valid non-negative --device-id")
        if args.chips is None or len(args.chips) != 1:
            raise RuntimeError("PCIe requires exactly one explicit --chip")
    else:
        if (args.device_id is not None or args.allow_pcie or args.allow_pcie_profile or
                args.require_decoded_timing or decoder_options):
            raise RuntimeError("PCIe acknowledgements and decoder options are invalid for CModel")


def _worker_environment(repo_root: Path, scratch: Path, runtime_mode: str,
                        device_id: Optional[int]) -> dict[str, str]:
    environment = os.environ.copy()
    ppl_root = environment.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise RuntimeError("PPL_PROJECT_ROOT must identify the configured PPL 1.7 SDK")
    environment["PPL_PROJECT_ROOT"] = str(Path(ppl_root).expanduser().resolve())
    inherited_paths = tuple(
        os.path.abspath(item)
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys((str(repo_root), *inherited_paths)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = str(scratch)
    for name in _PCIE_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    if runtime_mode == "pcie":
        assert device_id is not None
        environment.update({
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": str(device_id),
        })
    return environment


def _profiling_config(args: argparse.Namespace, output_dir: Path, chip: str,
                      label: str) -> TPUProfilingConfig:
    return TPUProfilingConfig(
        chip=chip,
        programming_model="tpukernel",
        runtime_mode=args.runtime_mode,
        output_dir=output_dir,
        label=label,
        timeout_s=args.timeout,
        postprocess=args.runtime_mode == "pcie",
        pcie_decoder_python=args.pcie_decoder_python,
        pcie_decoder_pythonpath=tuple(args.pcie_decoder_pythonpath),
    )


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_matrix(args: argparse.Namespace, repo_root: Path, output_dir: Path,
                environment: Mapping[str, str]) -> int:
    chips = tuple(args.chips or _CHIPS)
    dtypes = tuple(args.dtypes or _DTYPES)
    cases = tuple(args.cases or _CASES)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "runtime_mode": args.runtime_mode,
        "programming_model": "tpukernel",
        "acceptance":
            ("numeric-raw-and-decoded-timing" if args.require_decoded_timing else "numeric-and-raw"
            ),
        "decoded_timing_required": args.require_decoded_timing,
        "complete": False,
        "cases": {},
    }
    summary.update(git_source_identity(repo_root))
    summary_path = output_dir / "summary.json"
    worker = Path(__file__).with_name("tpu_fp8_ops_worker.py")

    if args.require_decoded_timing:
        try:
            preflight_profiler = TPUInstructionProfiler(
                _profiling_config(args, output_dir, chips[0], "pcie-decoder-preflight"))
            decoder_identity = preflight_profiler.preflight_pcie_decoder(environment=environment)
            summary["decoder_preflight"] = {
                "status": "passed",
                "identity": dict(decoder_identity),
            }
        except BaseException as exc:
            summary["decoder_preflight"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_summary(summary_path, summary)
            print(f"STOP decoder preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    for chip in chips:
        for dtype in dtypes:
            for case in cases:
                key = f"{chip}/tpukernel/{dtype}/{case}"
                print(f"RUN {key}", flush=True)
                profiler = TPUInstructionProfiler(
                    _profiling_config(args, output_dir, chip, f"{chip}-tpukernel-{dtype}-{case}"))
                command = [sys.executable, str(worker), "--dtype", dtype, "--case", case]
                try:
                    report = (
                        profiler.run_pcie(command, environment=environment) if args.runtime_mode
                        == "pcie" else profiler.run_cmodel(command, environment=environment))
                    _validate_profile_report(
                        report, require_decoded_timing=args.require_decoded_timing)
                    summary["cases"][key] = _report_summary(
                        report, require_decoded_timing=args.require_decoded_timing)
                    print(
                        f"PASS {key} raw={len(report.raw_trace_files)} "
                        f"timed={len(report.instruction_timings)}",
                        flush=True,
                    )
                except BaseException as exc:
                    summary["cases"][key] = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    _write_summary(summary_path, summary)
                    print(f"STOP {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                _write_summary(summary_path, summary)

    summary["complete"] = True
    _write_summary(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    try:
        environment = _worker_environment(repo_root, scratch, args.runtime_mode, args.device_id)
        return _run_matrix(args, repo_root, output_dir, environment)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
