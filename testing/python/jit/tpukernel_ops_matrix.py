# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the TPU-Kernel numerical conformance matrix in isolated processes.

The default matrix is CModel-only and covers BM1690 plus SG2260E.  PCIe is
impossible to enter without ``--runtime-mode pcie --allow-pcie --device-id``.
Every case owns a fresh parent-death-supervised process group.  On a timeout
or first failure, the runner sends SIGTERM to that group, waits a bounded
grace interval, follows with SIGKILL if necessary, records the partial JSON
report, and stops.  If an outer watchdog kills the runner itself, the shared
TPU supervisor kills the worker group before it can become orphaned.

Examples::

    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-numeric

    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-reductions \
        --chip sg2260e --op reduce-sum --op reduce-max

    # Deliberately dangerous: run only after the CModel matrix passes.
    python testing/python/jit/tpukernel_ops_matrix.py \
        --output-dir research/artifacts/tpukernel-pcie \
        --runtime-mode pcie --allow-pcie --device-id 0 --chip sg2260e
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from tpukernel_ops_worker import (
    CaseSpec,
    _CHIPS,
    _CORE_COUNTS,
    _FLOAT_DTYPES,
    _RESULT_PREFIX,
    _RUNTIME_MODES,
    build_case_specs,
)

if __package__:
    from .tpu_matrix_common import git_source_identity
else:
    from tpu_matrix_common import git_source_identity


_INTEGER_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32")
_ALL_DTYPES = _FLOAT_DTYPES + _INTEGER_DTYPES
_OPERATIONS = tuple(sorted({case.operation for case in build_case_specs()}))
_SCHEMA_VERSION = 1
_PROCESS_PIPE_DRAIN_S = 5.0
_TPU_PROCESS_SUPERVISOR = (
    Path(__file__).resolve().parents[3] / "tilelang" / "jit" /
    "_tpu_profile_supervisor.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory in which summary.json and per-case JSON are written",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=_RUNTIME_MODES,
        default="cmodel",
        help="defaults to the board-safe CModel runtime",
    )
    parser.add_argument(
        "--chip",
        choices=_CHIPS,
        action="append",
        dest="chips",
        help=("repeat to select chips; CModel defaults to both chips, while "
              "PCIe requires exactly one explicit chip"),
    )
    parser.add_argument(
        "--op",
        choices=_OPERATIONS,
        action="append",
        dest="operations",
        help="repeat to select operation families",
    )
    parser.add_argument(
        "--dtype",
        choices=_ALL_DTYPES,
        action="append",
        dest="dtypes",
        help="repeat to select case output/payload dtypes",
    )
    parser.add_argument(
        "--case",
        choices=tuple(case.case_id for case in build_case_specs()),
        action="append",
        dest="case_ids",
        help="repeat to select exact case ids (intersects --op/--dtype)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="compile+load+one-launch timeout for each fresh worker",
    )
    parser.add_argument(
        "--kill-grace",
        type=float,
        default=3.0,
        help="seconds between process-group SIGTERM and SIGKILL",
    )
    parser.add_argument(
        "--allow-pcie",
        action="store_true",
        help="mandatory explicit acknowledgement for PCIe execution",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="mandatory non-negative board id for PCIe execution",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the filtered declarative case list as JSON and exit safely",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.kill_grace < 0:
        raise ValueError("--kill-grace must be non-negative")
    if args.runtime_mode == "pcie":
        if not args.allow_pcie:
            raise RuntimeError("PCIe numerical execution requires explicit --allow-pcie")
        if args.device_id is None or args.device_id < 0 or args.device_id > 2**31 - 1:
            raise RuntimeError("PCIe numerical execution requires a valid --device-id")
        if args.chips is None or len(set(args.chips)) != 1:
            raise RuntimeError(
                "PCIe numerical execution requires exactly one explicit --chip; "
                "run different chip targets in separate invocations")
    elif args.allow_pcie or args.device_id is not None:
        raise RuntimeError("--allow-pcie/--device-id are invalid for CModel execution")
    if not args.list_cases and args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --list-cases is used")


def _unique_in_order(values: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    source = values if values else default
    return tuple(dict.fromkeys(source))


def _selected_cases(args: argparse.Namespace) -> tuple[CaseSpec, ...]:
    operations = set(args.operations or _OPERATIONS)
    dtypes = set(args.dtypes or _ALL_DTYPES)
    case_ids = set(args.case_ids) if args.case_ids else None
    selected = tuple(
        case for case in build_case_specs()
        if case.operation in operations
        and case.dtype in dtypes
        and (case_ids is None or case.case_id in case_ids)
    )
    if not selected:
        raise RuntimeError("the requested case filters select no TPU-Kernel probes")
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _worker_environment(
        repo_root: Path,
        scratch_dir: Path,
        case: CaseSpec,
        chip: str,
        args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    ppl_root = environment.get("PPL_PROJECT_ROOT")
    if not ppl_root:
        raise RuntimeError("PPL_PROJECT_ROOT must identify the configured PPL 1.7 SDK")
    environment["PPL_PROJECT_ROOT"] = str(Path(ppl_root).expanduser().resolve())

    inherited_pythonpath = environment.get("PYTHONPATH", "")
    inherited_paths = tuple(
        os.path.abspath(item)
        for item in inherited_pythonpath.split(os.pathsep)
        if item)
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys((str(repo_root), *inherited_paths)))
    environment["TMPDIR"] = str(scratch_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TILELANG_TPU_BENCHMARK_RUNS"] = "0"
    environment.update({
        "TILELANG_TPU_NUMERIC_SESSION": "1",
        "TILELANG_TPU_NUMERIC_CASE": case.case_id,
        "TILELANG_TPU_NUMERIC_CHIP": chip,
        "TILELANG_TPU_NUMERIC_RUNTIME_MODE": args.runtime_mode,
    })

    # Never allow inherited shell state to turn a CModel command into a board
    # load or silently turn a numerical run into a profiling run.  Conversely,
    # install PCIe gates only after CLI validation.
    for name in (
            "TILELANG_TPU_NUMERIC_ALLOW_PCIE",
            "TILELANG_TPU_ALLOW_PCIE_LOAD",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE",
            "TILELANG_TPU_DEVICE_ID",
            "TILELANG_TPU_PROFILE_SESSION",
            "TILELANG_TPU_PROFILE_CHIP",
            "TILELANG_TPU_PROFILE_PROGRAMMING_MODEL",
            "TILELANG_TPU_PROFILE_RUNTIME_MODE",
            "TILELANG_TPU_PROFILE_OUTPUT_DIR",
            "BMLIB_ENABLE_ALL_PROFILE",
            "FILE_DUMP_CMD",
            "PROFILE_BOOK_KEEPING",
            "PROFILE_RECORD_SIZE",
            "TPU_RT_CORE_NUM"):
        environment.pop(name, None)
    if args.runtime_mode == "pcie":
        assert args.device_id is not None
        environment.update({
            "TILELANG_TPU_NUMERIC_ALLOW_PCIE": "1",
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_DEVICE_ID": str(args.device_id),
        })
    else:
        environment["TPU_RT_CORE_NUM"] = str(_CORE_COUNTS[chip])
    return environment


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat an inaccessible group as live; the subsequent signal error is
        # retained in the machine-readable termination report.
        return True


def _terminate_process_group(
        process: subprocess.Popen[str], grace_seconds: float) -> dict[str, Any]:
    """Terminate the worker's dedicated process group, escalating if live."""

    process_group = process.pid
    report: dict[str, Any] = {
        "process_group": process_group,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "errors": [],
    }
    try:
        os.killpg(process_group, signal.SIGTERM)
        report["sigterm_sent"] = True
    except ProcessLookupError:
        pass
    except OSError as error:
        report["errors"].append(f"SIGTERM: {type(error).__name__}: {error}")

    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()  # Reap the direct worker if SIGTERM made it exit.
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    process.poll()
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
            report["sigkill_sent"] = True
        except ProcessLookupError:
            pass
        except OSError as error:
            report["errors"].append(f"SIGKILL: {type(error).__name__}: {error}")

    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        report["errors"].append("direct worker did not reap after SIGKILL")
    report["returncode_after_termination"] = process.poll()
    report["group_exists_after_termination"] = _process_group_exists(process_group)
    return report


def _timeout_output_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_and_collect(
        process: subprocess.Popen[str], grace_seconds: float
) -> tuple[str, str, dict[str, Any]]:
    """Terminate a timed-out worker group and drain pipes with a hard bound.

    An escaped or uninterruptible descendant can retain an inherited stdout or
    stderr descriptor after the supervised group has died.  An unbounded final
    ``communicate()`` would then defeat the per-case TPU watchdog.  Preserve
    whatever output is available, close this runner's pipe handles, and return
    the machine-readable termination report even in that pathological case.
    """

    termination = _terminate_process_group(process, grace_seconds)
    termination["pipe_drain_timeout_seconds"] = _PROCESS_PIPE_DRAIN_S
    termination["pipe_drain_timed_out"] = False
    try:
        stdout, stderr = process.communicate(timeout=_PROCESS_PIPE_DRAIN_S)
        return stdout, stderr, termination
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output_as_text(error.output)
        stderr = _timeout_output_as_text(error.stderr)
        diagnostic = (
            "TileLang TPU numerical watchdog: process group did not close its "
            f"output pipes within {_PROCESS_PIPE_DRAIN_S:g}s after termination"
        )
        termination["pipe_drain_timed_out"] = True
        termination["errors"].append(diagnostic)
        stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        return stdout, stderr, termination


def _spawn_guarded_worker(
        command: Sequence[str], *, cwd: Path,
        environment: Mapping[str, str]) -> subprocess.Popen[str]:
    """Start one numerical worker under the shared TPU process-tree guard.

    A new session alone protects the runner's own process group, but it also
    lets a worker outlive the runner if an outer watchdog kills the latter.
    Reuse the profiling path's Linux parent-death supervisor so that such a
    death kills the private worker group before it can become an orphan.
    """

    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "safe TPU numerical execution requires Linux PR_SET_PDEATHSIG")
    if not _TPU_PROCESS_SUPERVISOR.is_file():
        raise RuntimeError(
            "TPU process-tree supervisor is missing: "
            f"{_TPU_PROCESS_SUPERVISOR}")
    guarded_command = [
        sys.executable,
        str(_TPU_PROCESS_SUPERVISOR),
        "--parent-pid",
        str(os.getpid()),
        "--",
        *command,
    ]
    return subprocess.Popen(
        guarded_command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _worker_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            try:
                payload = json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"worker result marker contains invalid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise RuntimeError("worker result marker must contain a JSON object")
            return payload
    return None


def _case_directory(output_dir: Path, chip: str, runtime_mode: str,
                    case: CaseSpec) -> Path:
    # Case ids are generated from a closed alnum/hyphen/dot vocabulary.
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-." for char in case.case_id):
        raise AssertionError(f"unsafe generated case id {case.case_id!r}")
    return output_dir / "cases" / runtime_mode / chip / case.case_id


def _run_one(
        args: argparse.Namespace,
        repo_root: Path,
        output_dir: Path,
        worker: Path,
        chip: str,
        case: CaseSpec) -> dict[str, Any]:
    case_dir = _case_directory(output_dir, chip, args.runtime_mode, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(tempfile.mkdtemp(prefix=".scratch-", dir=case_dir))
    command = [
        sys.executable,
        "-u",
        str(worker),
        "--case-id",
        case.case_id,
        "--chip",
        chip,
        "--runtime-mode",
        args.runtime_mode,
    ]
    try:
        environment = _worker_environment(repo_root, scratch_dir, case, chip, args)
    except BaseException:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        raise
    started_at = _utc_now()
    begin = time.monotonic()
    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    termination: dict[str, Any] | None = None
    timed_out = False
    launch_error: str | None = None
    returncode: int | None = None
    try:
        # Keep autotuner/vendor/compiler byproducts in disposable storage;
        # only the explicit JSON report escapes into case_dir.  The shared
        # supervisor additionally guarantees cleanup if this runner itself is
        # killed by an outer watchdog.
        process = _spawn_guarded_worker(
            command, cwd=scratch_dir, environment=environment)
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Do not append TimeoutExpired's partial copies: the bounded drain
            # returns the complete captured streams when all descriptors close,
            # or its own latest partial copies when a stuck/escaped descendant
            # retains an inherited pipe.
            stdout, stderr, termination = _terminate_and_collect(
                process, args.kill_grace)
        returncode = process.poll()
        if returncode == 0 and termination is None and \
                _process_group_exists(process.pid):
            # A successful direct worker is not allowed to background ordinary
            # children.  Once the supervisor exits, its parent-death contract
            # no longer protects such processes, so clean the group and fail
            # this case before another TPU workload can start.
            termination = _terminate_process_group(process, args.kill_grace)
            launch_error = (
                "RuntimeError: successful worker left a live descendant in "
                "its supervised process group")
            returncode = process.poll()
        if returncode not in (0, None) and termination is None:
            # The direct worker may have exited while compiler/runtime children
            # survived.  Clean its still-addressable process group on first
            # failure before any subsequent case could start.
            termination = _terminate_process_group(process, args.kill_grace)
            returncode = process.poll()
    except BaseException as error:
        launch_error = f"{type(error).__name__}: {error}"
        if process is not None:
            termination = _terminate_process_group(process, args.kill_grace)
            returncode = process.poll()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    elapsed = time.monotonic() - begin
    parsed_payload: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed_payload = _worker_payload(stdout)
    except RuntimeError as error:
        parse_error = str(error)

    passed = (
        launch_error is None
        and not timed_out
        and returncode == 0
        and parse_error is None
        and parsed_payload is not None
        and parsed_payload.get("status") == "passed"
    )
    if timed_out:
        failure = f"worker exceeded {args.timeout:g}s timeout"
    elif launch_error is not None:
        failure = launch_error
    elif parse_error is not None:
        failure = parse_error
    elif parsed_payload is not None and parsed_payload.get("status") != "passed":
        failure = str(parsed_payload.get("error", "worker reported failure"))
    elif returncode != 0:
        failure = f"worker exited with status {returncode}"
    elif parsed_payload is None:
        failure = "worker produced no machine-readable result marker"
    else:
        failure = None

    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "case": case.to_json(),
        "chip": chip,
        "programming_model": "tpukernel",
        "runtime_mode": args.runtime_mode,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "timeout_seconds": args.timeout,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "failure": failure,
        "termination": termination,
        "worker_result": parsed_payload,
        "stdout": stdout,
        "stderr": stderr,
    }
    _write_json(case_dir / "result.json", result)
    return result


def _summary_case(result: Mapping[str, Any], result_path: Path,
                  output_dir: Path) -> dict[str, Any]:
    worker_result = result.get("worker_result")
    compact: dict[str, Any] = {
        "status": result["status"],
        "case": result["case"],
        "chip": result["chip"],
        "runtime_mode": result["runtime_mode"],
        "elapsed_seconds": result["elapsed_seconds"],
        "result_file": str(result_path.relative_to(output_dir)),
    }
    if result.get("failure") is not None:
        compact["failure"] = result["failure"]
    if isinstance(worker_result, dict):
        if "metrics" in worker_result:
            compact["metrics"] = worker_result["metrics"]
        if "timing" in worker_result:
            compact["worker_timing"] = worker_result["timing"]
    return compact


def _run_matrix(args: argparse.Namespace, chips: tuple[str, ...],
                cases: tuple[CaseSpec, ...]) -> int:
    assert args.output_dir is not None
    repo_root = Path(__file__).resolve().parents[3]
    worker = Path(__file__).with_name("tpukernel_ops_worker.py")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    scheduled = [
        {"chip": chip, "case": case.to_json()}
        for chip in chips
        for case in cases
        if chip in case.supported_chips
    ]
    if not scheduled:
        raise RuntimeError(
            "none of the selected TPU-Kernel probes is supported by the "
            "selected chip set")
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "running",
        "complete": False,
        "programming_model": "tpukernel",
        "runtime_mode": args.runtime_mode,
        "chips": list(chips),
        "timeout_seconds_per_case": args.timeout,
        "kill_grace_seconds": args.kill_grace,
        "started_at": _utc_now(),
        "finished_at": None,
        "scheduled_case_count": len(scheduled),
        "completed_case_count": 0,
        "passed_case_count": 0,
        "failed_case_count": 0,
        "scheduled": scheduled,
        "results": [],
    }
    summary.update(git_source_identity(repo_root))
    _write_json(summary_path, summary)

    for chip in chips:
        for case in cases:
            if chip not in case.supported_chips:
                continue
            key = f"{args.runtime_mode}/{chip}/{case.case_id}"
            print(f"RUN {key}", flush=True)
            try:
                result = _run_one(
                    args, repo_root, output_dir, worker, chip, case)
            except BaseException as error:
                # This path covers runner bookkeeping failures.  No worker can
                # be live here: _run_one always terminates its group in finally.
                result = {
                    "schema_version": _SCHEMA_VERSION,
                    "status": "failed",
                    "case": case.to_json(),
                    "chip": chip,
                    "programming_model": "tpukernel",
                    "runtime_mode": args.runtime_mode,
                    "started_at": _utc_now(),
                    "elapsed_seconds": 0.0,
                    "timeout_seconds": args.timeout,
                    "timed_out": False,
                    "failure": f"runner error: {type(error).__name__}: {error}",
                }
                case_dir = _case_directory(output_dir, chip, args.runtime_mode, case)
                _write_json(case_dir / "result.json", result)

            result_path = (
                _case_directory(output_dir, chip, args.runtime_mode, case) /
                "result.json")
            summary["results"].append(
                _summary_case(result, result_path, output_dir))
            summary["completed_case_count"] += 1
            if result["status"] == "passed":
                summary["passed_case_count"] += 1
                print(f"PASS {key}", flush=True)
                _write_json(summary_path, summary)
                continue

            summary["failed_case_count"] += 1
            summary["status"] = "failed"
            summary["stopped_after"] = key
            summary["finished_at"] = _utc_now()
            _write_json(summary_path, summary)
            print(f"STOP {key}: {result.get('failure')}", file=sys.stderr, flush=True)
            return 1

    summary["status"] = "passed"
    summary["complete"] = True
    summary["finished_at"] = _utc_now()
    _write_json(summary_path, summary)
    print(f"MATRIX_OK {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    chips = _unique_in_order(args.chips, _CHIPS)
    cases = _selected_cases(args)
    if args.list_cases:
        print(json.dumps({
            "schema_version": _SCHEMA_VERSION,
            "programming_model": "tpukernel",
            "runtime_mode": args.runtime_mode,
            "chips": list(chips),
            "case_count_by_chip": {
                chip: sum(chip in case.supported_chips for case in cases)
                for chip in chips
            },
            "cases": [case.to_json() for case in cases],
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0
    return _run_matrix(args, chips, cases)


if __name__ == "__main__":
    raise SystemExit(main())
