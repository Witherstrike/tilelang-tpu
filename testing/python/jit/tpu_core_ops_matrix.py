# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU core-op numerical/profile matrix in isolated fresh processes.

Every case is compiled, loaded, launched once, checked against PyTorch, and
profiled by ``TPUInstructionProfiler``.  The first failure stops the matrix;
the profiler has already terminated that worker's process group before this
runner records the partial result.  PCIe additionally requires two explicit
CLI acknowledgements and a numeric device id.  PCIe decoded timing is
best-effort unless ``--require-decoded-timing`` explicitly makes it part of
the acceptance contract; numerical correctness and raw trace collection do
not depend on an optional vendor decoder being installed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Optional

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

_COPY_CASES = ("copy-fp32-local-roundtrip", "copy-fp32-global-to-global",
               "copy-fp16-local-roundtrip", "copy-fp16-global-to-global")
_CASES = ("elementwise-add", "elementwise-sub", "elementwise-mul", "elementwise-div", "matmul",
          *_COPY_CASES)
_CMODEL_CONFIGS = (("sg2260e", "tpukernel"), ("sg2260e", "rv"), ("bm1690", "tpukernel"))
_PCIE_CONFIGS = (("sg2260e", "tpukernel"), ("sg2260e", "rv"))


def _worker_environment(repo_root: Path, runtime_mode: str,
                        device_id: Optional[int]) -> dict[str, str]:
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise RuntimeError("PPL_PROJECT_ROOT must identify the configured PPL 1.7 SDK")
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    environment = {
        "PPL_PROJECT_ROOT":
            ppl_root,
        "PYTHONPATH":
            os.pathsep.join(item for item in (str(repo_root), inherited_pythonpath) if item),
    }
    if runtime_mode == "pcie":
        assert device_id is not None
        environment.update({
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": str(device_id),
        })
    return environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-mode", choices=("cmodel", "pcie"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chip", choices=("sg2260e", "bm1690"))
    parser.add_argument("--programming-model", choices=("tpukernel", "rv"))
    parser.add_argument("--case", choices=_CASES, action="append", dest="cases")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--allow-pcie", action="store_true")
    parser.add_argument("--allow-pcie-profile", action="store_true")
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
        help=("decoder-only import directory; repeat for multiple directories; "
              "never added to the compile/worker PYTHONPATH"),
    )
    parser.add_argument(
        "--require-decoded-timing",
        action="store_true",
        help=("make parser_status=ready and at least one valid decoded device "
              "timing row mandatory; by default PCIe accepts numerical success "
              "plus raw trace and treats decoding as best-effort"),
    )
    return parser.parse_args()


def _run_matrix(args: argparse.Namespace, repo_root: Path, output_dir: Path,
                configurations: tuple[tuple[str, str], ...], cases: tuple[str, ...],
                environment: dict[str, str]) -> int:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "runtime_mode": args.runtime_mode,
        "acceptance":
            ("numeric-raw-and-decoded-timing" if args.require_decoded_timing else "numeric-and-raw"
            ),
        "decoded_timing_required": args.require_decoded_timing,
        "complete": False,
        "cases": {},
    }
    summary.update(git_source_identity(repo_root))
    summary_path = output_dir / "summary.json"
    worker = Path(__file__).with_name("tpu_profile_worker.py")
    decoder_config = {
        "pcie_decoder_python": getattr(args, "pcie_decoder_python", None),
        "pcie_decoder_pythonpath": tuple(
            getattr(args, "pcie_decoder_pythonpath", ()) or ()),
    }

    if args.require_decoded_timing:
        chip, programming_model = configurations[0]
        print("PREFLIGHT PCIe decoder", flush=True)
        try:
            preflight_config = TPUProfilingConfig(
                chip=chip,
                programming_model=programming_model,
                runtime_mode="pcie",
                output_dir=output_dir,
                label="pcie-decoder-preflight",
                timeout_s=args.timeout,
                postprocess=True,
                **decoder_config,
            )
            decoder_identity = TPUInstructionProfiler(
                preflight_config).preflight_pcie_decoder(environment=environment)
        except BaseException as exc:
            summary["decoder_preflight"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"STOP decoder preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        summary["decoder_preflight"] = {
            "status": "ready",
            "identity": dict(decoder_identity),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for chip, programming_model in configurations:
        for case in cases:
            key = f"{chip}/{programming_model}/{case}"
            print(f"RUN {key}", flush=True)
            config = TPUProfilingConfig(
                chip=chip,
                programming_model=programming_model,
                runtime_mode=args.runtime_mode,
                output_dir=output_dir,
                label=f"{chip}-{programming_model}-{case}",
                timeout_s=args.timeout,
                postprocess=args.runtime_mode == "pcie",
                **decoder_config,
            )
            profiler = TPUInstructionProfiler(config)
            command = [sys.executable, str(worker), "--case", case]
            try:
                report = (
                    profiler.run_pcie(command, environment=environment) if args.runtime_mode
                    == "pcie" else profiler.run_cmodel(command, environment=environment))
                _validate_profile_report(report, require_decoded_timing=args.require_decoded_timing)
                summary["cases"][key] = _report_summary(
                    report, require_decoded_timing=args.require_decoded_timing)
                print(
                    f"PASS {key} raw={len(report.raw_instructions)} "
                    f"timed={len(report.instruction_timings)}",
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
                print(f"STOP {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    summary["complete"] = True
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.runtime_mode == "pcie":
        if not args.allow_pcie or not args.allow_pcie_profile:
            raise RuntimeError("PCIe requires both --allow-pcie and --allow-pcie-profile")
        if args.device_id is None or args.device_id < 0:
            raise RuntimeError("PCIe requires a non-negative --device-id")
    elif args.device_id is not None or args.allow_pcie or args.allow_pcie_profile:
        raise RuntimeError("PCIe acknowledgements must not be supplied to CModel")
    if args.runtime_mode != "pcie" and \
            (args.pcie_decoder_python is not None or args.pcie_decoder_pythonpath):
        raise RuntimeError("PCIe decoder options must not be supplied to CModel")
    if args.require_decoded_timing and args.runtime_mode != "pcie":
        raise RuntimeError("--require-decoded-timing is only valid with PCIe")

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = _PCIE_CONFIGS if args.runtime_mode == "pcie" else _CMODEL_CONFIGS
    configurations = tuple(
        item for item in configurations if (args.chip is None or item[0] == args.chip) and
        (args.programming_model is None or item[1] == args.programming_model))
    if not configurations:
        raise RuntimeError("the requested chip/programming-model pair is not in this matrix")
    cases = tuple(args.cases) if args.cases else _CASES
    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=output_dir))
    environment = _worker_environment(repo_root, args.runtime_mode, args.device_id)
    environment["TMPDIR"] = str(scratch_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        return _run_matrix(args, repo_root, output_dir, configurations, cases, environment)
    finally:
        # Delete only the unique directory created by this invocation.  Trace,
        # decoder, and report artifacts are siblings and remain intact.
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
