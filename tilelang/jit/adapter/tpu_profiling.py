# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Safe PPL-style instruction profiling for TileLang TPU test programs.

TileLang emits raw PPL C directly and does not pass through ``ppl-compile``.
Consequently PPL's CLI-level ``--profiling``/``--autotune`` switch is not
directly applicable here. What is reusable is its runtime protocol:

* on CModel, execute one freshly compiled test worker in a dedicated directory
  with ``FILE_DUMP_CMD`` set, then optionally run an explicitly supplied
  PerfAI installation over the raw command dumps;
* on PCIe, enable TPUDNN command recording around exactly one launch in a
  supervised worker, behind separate load/profile/device safety gates, then
  optionally decode ``cdm_profile_data_dev*`` with already-installed vendor
  Python packages.

The child-process boundary is intentional.  Both the vendor runtime and the
process working directory are global state, and a profile worker must never
load an arbitrary prebuilt ``main.so`` from another JIT instance.  The command
passed to :class:`TPUInstructionProfiler` is therefore expected to compile and
load its own private TileLang JIT artifact.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout


PathLike = Union[str, os.PathLike]

_DEVICE_COMMAND_ENGINES = frozenset((
    "bd", "bdc", "tiu", "gdma", "sdma", "vsdma", "cdma", "dma",
))
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RAW_TRACE_ARTIFACT_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9][A-Za-z0-9_.-]*)-"
    r"(?P<launch>\d+)-(?P<group>\d+)\."
    r"(?P<engine>BD|GDMA|SDMA|VSDMA)\.(?P<core>\d+)(?:\.txt)?$")
_RAW_TRACE_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9][A-Za-z0-9_.-]*)-"
    r"(?P<launch>\d+)-(?P<group>\d+)\."
    r"(?P<engine>BD|GDMA|SDMA|VSDMA)\.(?P<core>\d+)\.txt$")
_PCIE_RAW_PROFILE_FILE_RE = re.compile(
    r"^(?:global|cdmlib\d+_\d+)\.profile$")

_PROFILE_SUPERVISOR_PATH = Path(__file__).parent.parent / "_tpu_profile_supervisor.py"
_PCIE_PROFILE_DECODER_PATH = Path(__file__).parent.parent / "_tpu_pcie_profile_decoder.py"
_PCIE_DECODED_REPORT_NAME = "tilelang_pcie_profile.json"


class TPUProfilingError(RuntimeError):
    """Base class for TPU profiling failures that preserve collected artifacts."""


class TPUProfilingTimeoutError(TPUProfilingError):
    """A profile worker exceeded its timeout and its process group was stopped."""


class TPUProfilingCommandError(TPUProfilingError):
    """The isolated profile worker exited unsuccessfully."""


class _PerfAIWorkspaceBusy(RuntimeError):
    """The vendor PerfAI work directory is already owned by another session."""


@dataclass(frozen=True)
class TPUProfilingConfig:
    """Configuration for one isolated TPU instruction-profile session.

    PCIe dispatch is fail-closed: :meth:`TPUInstructionProfiler.run_pcie`
    requires independent load/profile acknowledgements plus an explicit device
    ID.  Its worker must compile and load its own private JIT artifact, just as
    the CModel path does.
    """

    chip: str
    programming_model: str = "tpukernel"
    runtime_mode: str = "cmodel"
    output_dir: Optional[PathLike] = None
    label: str = "tilelang"
    timeout_s: float = 60.0
    perfai_root: Optional[PathLike] = None
    postprocess: bool = True
    profile_record_size: int = 4096
    profile_book_keeping: int = 1

    def __post_init__(self) -> None:
        if self.runtime_mode not in ("cmodel", "pcie"):
            raise ValueError(
                f"Unsupported TPU profiling runtime mode {self.runtime_mode!r}; "
                f"expected 'cmodel' or 'pcie'.")
        if isinstance(self.timeout_s, bool) or \
                not isinstance(self.timeout_s, (int, float)) or \
                not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("TPU profiling timeout_s must be a positive number.")
        if not _LABEL_RE.fullmatch(self.label):
            raise ValueError(
                "TPU profiling label must start with an ASCII letter/digit and "
                "contain only letters, digits, '.', '_' or '-'.")
        if not isinstance(self.profile_record_size, int) or self.profile_record_size <= 0:
            raise ValueError("TPU profile_record_size must be a positive integer.")
        if not isinstance(self.profile_book_keeping, int) or self.profile_book_keeping < 0:
            raise ValueError("TPU profile_book_keeping must be a non-negative integer.")
        # The dual-backend capability registry is the one authority for chip,
        # programming model and core topology.
        # Profiling must not maintain a second copy of this table.
        target_spec = TPUTargetSpec(
            chip=self.chip,
            programming_model=self.programming_model,
        )
        runtime_config = TPURuntimeConfig(runtime_mode=self.runtime_mode)
        object.__setattr__(self, "chip", target_spec.chip)
        object.__setattr__(self, "programming_model", target_spec.programming_model)
        object.__setattr__(self, "runtime_mode", runtime_config.runtime_mode)

    @property
    def target_spec(self) -> TPUTargetSpec:
        """Return the canonical compile-time identity for this session."""

        return TPUTargetSpec(
            chip=self.chip,
            programming_model=self.programming_model,
        )

    @property
    def runtime_config(self) -> TPURuntimeConfig:
        """Return the canonical host runtime selection for this session."""

        return TPURuntimeConfig(runtime_mode=self.runtime_mode)


@dataclass(frozen=True)
class TPUInstructionTiming:
    """One device-command timeline row emitted by PerfAI's ``profile_data.js``.

    ``instruction_timings`` intentionally excludes CPU/subnet/layer timeline
    events.  The complete PerfAI timeline remains available in
    :attr:`TPUProfileReport.timeline_events` for diagnostics.
    """

    engine: str
    begin: Any
    end: Any
    duration: Optional[float]
    unit: str
    core_id: Optional[int] = None
    command_id: Optional[int] = None
    opcode: Optional[str] = None
    fields: Mapping[str, Any] = field(default_factory=dict)


def _pcie_instruction_timing_error(timing: Any) -> Optional[str]:
    """Return why a decoded PCIe timing row is invalid, or ``None``."""

    for field_name in ("begin", "end", "duration"):
        value = getattr(timing, field_name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{field_name} must be a numeric nanosecond value"
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            return f"{field_name} must be finite"
    if timing.unit != "ns":
        return "unit must be exactly 'ns'"
    if timing.duration < 0:
        return "duration must be non-negative"
    if timing.end < timing.begin:
        return "end must not precede begin"
    return None


def _pcie_instruction_timings_error(timings: Sequence[Any]) -> Optional[str]:
    """Validate all decoded PCIe rows through one shared acceptance predicate."""

    if not timings:
        return "no decoded instruction timings"
    for index, timing in enumerate(timings):
        error = _pcie_instruction_timing_error(timing)
        if error is not None:
            return f"event {index}: {error}"
    return None


def _is_valid_pcie_raw_trace_artifact(raw_path: PathLike) -> bool:
    """Return whether one recorder directory contains non-empty profile data."""

    path = Path(raw_path)
    if not path.is_dir():
        return False
    try:
        profile_files = tuple(
            candidate for candidate in path.iterdir()
            if candidate.is_file() and
            _PCIE_RAW_PROFILE_FILE_RE.fullmatch(candidate.name) is not None)
        return bool(profile_files) and all(
            candidate.stat().st_size > 0 for candidate in profile_files)
    except OSError:
        # Recorder output may disappear or become inaccessible between worker
        # exit and collection.  Such a path is not usable raw evidence.
        return False


@dataclass(frozen=True)
class TPURawInstruction:
    """One decoded textual CModel command-dump line, without a fabricated time."""

    engine: str
    core_id: int
    command_id: Optional[int]
    opcode: Optional[str]
    text: str
    source_path: Path


@dataclass(frozen=True)
class TPUProfileReport:
    """Artifacts and optional decoded timing rows from one profile worker."""

    config: TPUProfilingConfig
    output_dir: Path
    command: Tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    raw_trace_files: Tuple[Path, ...]
    raw_instructions: Tuple[TPURawInstruction, ...]
    parser_status: str
    parser_message: Optional[str] = None
    perfai_report_path: Optional[Path] = None
    perfai_report_paths: Tuple[Path, ...] = ()
    decoded_report_path: Optional[Path] = None
    decoded_report_paths: Tuple[Path, ...] = ()
    timeline_events: Tuple[TPUInstructionTiming, ...] = ()
    instruction_timings: Tuple[TPUInstructionTiming, ...] = ()

    @property
    def has_raw_trace(self) -> bool:
        if self.config.runtime_mode == "pcie":
            return any(
                _is_valid_pcie_raw_trace_artifact(path)
                for path in self.raw_trace_files)
        return bool(self.raw_trace_files)

    @property
    def has_instruction_timings(self) -> bool:
        return bool(self.instruction_timings)


def _copy_environment(extra: Optional[Mapping[str, str]]) -> MutableMapping[str, str]:
    environment: MutableMapping[str, str] = dict(os.environ)
    if extra is not None:
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("TPU profiling environment keys and values must be strings.")
            environment[key] = value
    return environment


def _profile_output_dir(config: TPUProfilingConfig) -> Path:
    if config.output_dir is None:
        # A profile is an inspection artifact, not an anonymous scratch file.
        # Keep the default under the caller's working directory so successful
        # sessions never leave an undiscoverable directory in /tmp.
        output_root = Path.cwd().resolve() / "tilelang-tpu-profiles"
    else:
        output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Never emulate PPL's ``rmtree(<target>/profiling)`` behaviour. A caller
    # may preserve many regression profiles under one root, so every run owns
    # a new subdirectory and cannot erase a prior trace.
    return Path(tempfile.mkdtemp(prefix=f"{config.label}-", dir=output_root))


def _profile_deadline(timeout_s: float) -> float:
    """Return the absolute deadline shared by worker, lock, and PerfAI."""

    return time.monotonic() + timeout_s


def _remaining_timeout(deadline: float) -> float:
    """Return remaining wall-clock budget without ever extending a session."""

    return max(0.0, deadline - time.monotonic())


def _spawn_guarded_profile_process(command: Sequence[str], *, cwd: Path,
                                   environment: Mapping[str, str]) -> subprocess.Popen:
    """Spawn ``command`` below a parent-death-aware process-tree supervisor.

    The direct child is a standalone Python supervisor in its own session.
    It arms Linux parent-death ``SIGTERM`` immediately after exec, with a
    double parent-PID check that closes the setup race, then keeps the real
    worker/PerfAI command in its private process group.  Therefore an outer
    timeout that terminates pytest cannot leave an ordinary CModel worker or
    AutoRunner descendant running.
    """

    if not sys.platform.startswith("linux"):
        raise TPUProfilingError(
            "Safe TPU instruction profiling requires Linux PR_SET_PDEATHSIG; "
            "refusing to create a detached profile worker on this platform.")
    if not _PROFILE_SUPERVISOR_PATH.is_file():
        raise TPUProfilingError(
            f"TPU profile supervisor is missing: {_PROFILE_SUPERVISOR_PATH}")
    return subprocess.Popen(
        [sys.executable, str(_PROFILE_SUPERVISOR_PATH), "--parent-pid",
         str(os.getpid()), "--", *command],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )


_PROCESS_TERMINATION_GRACE_S = 5.0
_PROCESS_KILL_GRACE_S = 5.0
_PROCESS_PIPE_DRAIN_S = 5.0


def _kill_process_group(process: subprocess.Popen) -> bool:
    """Stop a guarded worker and all of its children without touching its parent shell.

    The direct process is the lightweight supervisor.  ``SIGTERM`` makes its
    handler kill the private group that contains the command and ordinary
    descendants before exiting.  The SIGKILL fallback only covers a broken
    supervisor and is intentionally not presented as a substitute for that
    guarded path.
    """

    if process.poll() is not None:
        return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_S)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_PROCESS_KILL_GRACE_S)
        return True
    except subprocess.TimeoutExpired:
        # A process stuck in uninterruptible kernel sleep cannot be reaped by
        # user space.  Never turn the TPU watchdog into another unbounded wait.
        return False


def _timeout_output_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_and_collect(process: subprocess.Popen) -> Tuple[str, str]:
    """Terminate a guarded group and drain its pipes with a final hard bound.

    A killed command descendant can retain an inherited pipe while stuck in a
    driver call.  ``Popen.communicate()`` without a timeout would then defeat
    the outer hardware watchdog.  Preserve whatever output is available,
    close this process's pipe handles, and return even in that pathological
    case.
    """

    reaped = _kill_process_group(process)
    try:
        stdout, stderr = process.communicate(timeout=_PROCESS_PIPE_DRAIN_S)
        return stdout, stderr
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_output_as_text(exc.output)
        stderr = _timeout_output_as_text(exc.stderr)
        diagnostic = (
            "TileLang TPU watchdog: process group did not close its output "
            f"pipes within {_PROCESS_PIPE_DRAIN_S:g}s after termination"
        )
        if not reaped:
            diagnostic += "; the supervisor process also could not be reaped"
        stderr = f"{stderr}\n{diagnostic}\n" if stderr else diagnostic + "\n"
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        return stdout, stderr


def _write_worker_logs(output_dir: Path, stdout: str, stderr: str) -> Tuple[Path, Path]:
    stdout_path = output_dir / "worker.stdout.log"
    stderr_path = output_dir / "worker.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _write_parser_logs(output_dir: Path, stdout: str, stderr: str) -> Tuple[Path, Path]:
    stdout_path = output_dir / "perfai.stdout.log"
    stderr_path = output_dir / "perfai.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


@contextmanager
def _exclusive_perfai_workspace(perfai_root: Path, deadline: float):
    """Serialize TileLang sessions that share PPL's mutable PerfAI workspace.

    PPL's own driver runs AutoRunner from the PerfAI directory, whose
    ``auto_build`` area is mutable.  Do not copy PPL's destructive cleanup;
    instead, own a Linux abstract-socket name keyed by the resolved root. This
    is a best-effort inter-process lock for TileLang sessions on the same
    machine; users must still keep unrelated direct PPL invocations out of
    that root.
    """

    # Linux abstract UNIX sockets are released automatically when the owning
    # process closes them.  Unlike a flock file in /tmp, this cannot leave a
    # stale intermediate behind after SIGKILL.
    token = hashlib.sha256(str(perfai_root).encode("utf-8")).hexdigest()[:20]
    address = "\0tilelang-perfai-" + token
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    while True:
        try:
            lock_socket.bind(address)
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                lock_socket.close()
                raise
            if time.monotonic() >= deadline:
                lock_socket.close()
                raise _PerfAIWorkspaceBusy(
                    f"PerfAI workspace is busy: {perfai_root}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock_socket.close()


def _find_perfai_root(config: TPUProfilingConfig,
                      environment: Mapping[str, str]) -> Optional[Path]:
    candidates = []
    if config.perfai_root is not None:
        candidates.append(Path(config.perfai_root))
    if environment.get("PPL_PERFAI_ROOT"):
        candidates.append(Path(environment["PPL_PERFAI_ROOT"]))
    if environment.get("PPL_THIRD_PARTY_PATH"):
        candidates.append(Path(environment["PPL_THIRD_PARTY_PATH"]) / "PerfAI")
    for candidate in candidates:
        runner = candidate.expanduser() / "AutoRunner.sh"
        if runner.is_file():
            return candidate.expanduser().resolve()
    return None


def _perfai_chip_name(config: TPUProfilingConfig) -> str:
    """Return the PPL chip spelling expected by PerfAI's ``-e`` flag.

    PPL invokes ``get_chip_name(chip_arch)`` before calling AutoRunner.  Its
    mapping distinguishes the RV profile decoder as ``sg2260erv`` even though
    TileLang deliberately exposes the physical chip as ``sg2260e`` plus a
    separate programming-model axis.  Keep that vendor-only spelling at this
    adapter boundary; it is not a third TileLang chip target.
    """

    if config.programming_model == "rv":
        return "sg2260erv"
    # This preserves PPL's CModel special case in ppl_compile.py: BM1690's
    # PerfAI target is named sg2260.
    return "sg2260" if config.chip == "bm1690" else config.chip


def _pcie_profile_arch(config: TPUProfilingConfig) -> str:
    """Return the decoder architecture used by PPL's TPUv7 PCIe path."""

    if config.programming_model == "rv":
        return "tpub_7_1_e_rv"
    return config.target_spec.chip_spec.ppl_arch


def _extract_js_array(text: str, variable: str) -> Sequence[Any]:
    """Read one array assignment from the small JS data file PerfAI emits.

    The output is data-only JavaScript, but it is not guaranteed to be strict
    JSON.  Try JSON first and then Python literal syntax after translating the
    three JSON scalar spellings.  The bracket scanner avoids a fragile
    line-oriented parser and handles arrays formatted over many lines.
    """

    match = re.search(rf"(?:let|var|const)\s+{re.escape(variable)}\s*=", text)
    if match is None:
        raise ValueError(f"PerfAI profile data does not define {variable!r}.")
    start = text.find("[", match.end())
    if start < 0:
        raise ValueError(f"PerfAI profile data has no array value for {variable!r}.")

    depth = 0
    quote: Optional[str] = None
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError(f"PerfAI profile data has an unterminated {variable!r} array.")

    source = text[start:end]
    try:
        value = json.loads(source)
    except json.JSONDecodeError:
        python_source = re.sub(r"\btrue\b", "True", source)
        python_source = re.sub(r"\bfalse\b", "False", python_source)
        python_source = re.sub(r"\bnull\b", "None", python_source)
        try:
            value = ast.literal_eval(python_source)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse PerfAI {variable!r} array as data.") from exc
    if not isinstance(value, list):
        raise ValueError(f"PerfAI {variable!r} is not an array.")
    return value


def _timing_unit(headers: Sequence[Any]) -> str:
    names = " ".join(str(item).lower() for item in headers)
    if "cycle" in names:
        return "cycles"
    if "usec" in names or "_us" in names or "(us" in names:
        return "us"
    if "_ms" in names or "(ms" in names:
        return "ms"
    return "unknown"


def _raw_command_id(text: str) -> Optional[int]:
    match = re.search(r"\b(?:cmd_id|bd_id|gdma_id|sdma_id)=(\d+)", text)
    return int(match.group(1)) if match is not None else None


def _raw_opcode(text: str) -> Optional[str]:
    match = re.search(r"\b(?:bd_func|gdma_func|sdma_func)=([^\s]+)", text)
    return match.group(1) if match is not None else None


def _integer_field(fields: Mapping[str, Any], *names: str) -> Optional[int]:
    normalized = {str(key).strip().lower(): value for key, value in fields.items()}
    for name in names:
        value = normalized.get(name)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            return int(value.strip())
    return None


def _string_field(fields: Mapping[str, Any], *names: str) -> Optional[str]:
    normalized = {str(key).strip().lower(): value for key, value in fields.items()}
    for name in names:
        value = normalized.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _timing_identity(fields: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Best-effort identity extraction without assuming one PerfAI schema.

    PerfAI releases have changed the extra columns in ``time_data``.  Preserve
    the original row in ``fields`` in all cases, and only project familiar
    fields when their names make the interpretation unambiguous.
    """

    core_id = _integer_field(fields, "core", "core_id", "coreid", "core_idx")
    command_id = _integer_field(
        fields, "cmd_id", "command_id", "bd_id", "gdma_id", "sdma_id")
    opcode = _string_field(
        fields, "opcode", "op_name", "func_name", "instruction")
    # PPL's historical PerfWeb schema stores ``bd_id=...`` / ``gdma_id=...``
    # in ``func_type`` rather than a separate command-id column. Preserve that
    # version-specific convention as a best-effort projection.
    for text in (_string_field(fields, "func_type"),
                 _string_field(fields, "type"),
                 _string_field(fields, "info")):
        if text is None:
            continue
        if command_id is None:
            command_id = _raw_command_id(text)
        if opcode is None:
            opcode = _raw_opcode(text)
    info = _string_field(fields, "info")
    if opcode is None and info is not None and "<br>" in info:
        # PPL's static BD/GDMA rows prefix the human-readable function name
        # before the first HTML line break. Do not mislabel metric-only rows.
        candidate = info.split("<br>", 1)[0].strip()
        if candidate and not re.match(r"(?:cycle|speed|bytes|ops_ratio)=", candidate):
            opcode = candidate
    return core_id, command_id, opcode


def parse_cmodel_raw_instruction_dumps(raw_trace_files: Sequence[PathLike]
                                      ) -> Tuple[TPURawInstruction, ...]:
    """Parse CModel text dumps into command metadata without inventing latency.

    The emulator's ``*.txt`` sidecars describe engine, core, dependency, and
    command-id information. They do not contain measured begin/end timestamps,
    so callers receive explicit raw records while PerfAI is unavailable rather
    than a misleading approximation labelled as execution time.
    """

    records = []
    for raw_path_like in raw_trace_files:
        raw_path = Path(raw_path_like)
        match = _RAW_TRACE_RE.fullmatch(raw_path.name)
        if match is None:
            continue
        engine = match.group("engine").lower()
        core_id = int(match.group("core"))
        try:
            lines = raw_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            text = line.strip()
            if not text:
                continue
            records.append(
                TPURawInstruction(
                    engine=engine,
                    core_id=core_id,
                    command_id=_raw_command_id(text),
                    opcode=_raw_opcode(text),
                    text=text,
                    source_path=raw_path,
                ))
    return tuple(records)


def parse_perfai_timeline_events(profile_data_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Extract every PerfAI timeline event from a web-report data file.

    PPL's own ``profiling_parser.py`` only prints an aggregate
    ``summary_data`` table.  PerfAI's ``time_data`` also contains CPU,
    subnet, and layer events, so this low-level parser preserves all rows
    rather than calling all of them instructions.
    """

    path = Path(profile_data_path)
    source = path.read_text(encoding="utf-8")
    categories = _extract_js_array(source, "categories")
    headers = _extract_js_array(source, "time_header")
    rows = _extract_js_array(source, "time_data")
    if len(headers) < 3:
        raise ValueError("PerfAI time_header must contain category, begin, and end fields.")

    unit = _timing_unit(headers)
    timings = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"PerfAI time_data row {row_index} is not a valid timeline row.")
        category = row[0]
        if isinstance(category, int) and 0 <= category < len(categories):
            engine = str(categories[category])
        else:
            engine = str(category)
        begin, end = row[1], row[2]
        duration: Optional[float] = None
        if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
            duration = float(end - begin)
        fields = {
            str(headers[index]): row[index]
            for index in range(min(len(headers), len(row)))
        }
        core_id, command_id, opcode = _timing_identity(fields)
        timings.append(
            TPUInstructionTiming(
                engine=engine,
                begin=begin,
                end=end,
                duration=duration,
                unit=unit,
                core_id=core_id,
                command_id=command_id,
                opcode=opcode,
                fields=fields,
            ))
    return tuple(timings)


def parse_perfai_instruction_timings(profile_data_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Return only recognized TPU command-engine rows from a PerfAI timeline.

    This deliberately filters PPL's host/subnet timeline rows.  Unknown
    vendor categories remain available through :func:`parse_perfai_timeline_events`
    instead of being misrepresented as individual TPU instructions.
    """

    return tuple(
        event for event in parse_perfai_timeline_events(profile_data_path)
        if event.engine.strip().lower() in _DEVICE_COMMAND_ENGINES)


def parse_pcie_decoded_instruction_timings(
        report_path: PathLike) -> Tuple[TPUInstructionTiming, ...]:
    """Read TileLang's stable JSON projection of a bigTpuProfile result."""

    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported TileLang PCIe profile schema: {path}")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"TileLang PCIe profile has no events array: {path}")
    events = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"PCIe profile event {index} is not an object: {path}")
        engine = raw_event.get("engine")
        begin = raw_event.get("begin")
        end = raw_event.get("end")
        unit = raw_event.get("unit")
        fields = raw_event.get("fields", {})
        if not isinstance(engine, str) or not isinstance(unit, str) or \
                not isinstance(begin, (int, float)) or \
                not isinstance(end, (int, float)) or \
                not isinstance(fields, dict):
            raise ValueError(f"PCIe profile event {index} has invalid fields: {path}")
        events.append(TPUInstructionTiming(
            engine=engine,
            begin=begin,
            end=end,
            duration=float(end - begin),
            unit=unit,
            core_id=raw_event.get("core_id"),
            command_id=raw_event.get("command_id"),
            opcode=raw_event.get("opcode"),
            fields=fields,
        ))
        error = _pcie_instruction_timing_error(events[-1])
        if error is not None:
            raise ValueError(
                f"PCIe profile event {index} is invalid: {error}: {path}")
    return tuple(events)


class TPUInstructionProfiler:
    """Run one fresh TileLang TPU test worker with PPL-compatible profiling."""

    def __init__(self, config: TPUProfilingConfig):
        self.config = config

    def _validate_ppl_dependencies(self, environment: Mapping[str, str]) -> None:
        """Fail before spawning when a TileLang/PPL worker declares its SDK.

        The profiler also accepts generic commands used by parser/unit tests,
        so an absent ``PPL_PROJECT_ROOT`` is not itself an error here.  Once a
        worker supplies the root, however, the runtime is known and only that
        runtime's profiling contract is validated.
        """

        ppl_root = environment.get("PPL_PROJECT_ROOT")
        if ppl_root is None:
            return
        if not ppl_root.strip():
            raise TPUProfilingError(
                "PPL_PROJECT_ROOT must not be empty for a TPU profile worker.")
        try:
            layout = resolve_ppl_layout(ppl_root, self.config.chip)
            layout.require_profiling(
                self.config.runtime_mode, environment=environment)
        except (OSError, ValueError) as exc:
            raise TPUProfilingError(
                "PPL 1.7 dependency preflight failed for "
                f"{self.config.runtime_mode} profiling: {exc}") from exc

    def pcie_profile_environment_overrides(
            self, environment: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
        """Return PPL's recorder overrides after PCIe safety preflight.

        The returned mapping is intentionally not a complete environment and
        this method never loads a library or touches a board.  The generated
        PCIe host enables TPUDNN recording only when :meth:`run_pcie` also marks
        the child as an isolated profiling session.

        No loading, initialization, or board dispatch occurs here.
        """

        if self.config.runtime_mode != "pcie":
            raise ValueError(
                "pcie_profile_environment_overrides requires runtime_mode='pcie'.")
        resolved = _copy_environment(environment)
        if resolved.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
            raise TPUProfilingError(
                "PCIe profiling requires TILELANG_TPU_ALLOW_PCIE_LOAD=1; "
                "this helper does not initialize a board.")
        if resolved.get("TILELANG_TPU_ALLOW_PCIE_PROFILE") != "1":
            raise TPUProfilingError(
                "PCIe profiling requires the separate acknowledgement "
                "TILELANG_TPU_ALLOW_PCIE_PROFILE=1.")
        device_id = resolved.get("TILELANG_TPU_DEVICE_ID", "")
        if re.fullmatch(r"[0-9]+", device_id) is None or int(device_id) > 2**31 - 1:
            raise TPUProfilingError(
                "PCIe profiling requires a non-negative integer "
                "TILELANG_TPU_DEVICE_ID.")
        return {
            "BMLIB_ENABLE_ALL_PROFILE": "1",
            "PROFILE_RECORD_SIZE": str(self.config.profile_record_size),
            "PROFILE_BOOK_KEEPING": str(self.config.profile_book_keeping),
        }

    def run_pcie(self,
                 command: Sequence[PathLike],
                 *,
                 environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
        """Run one explicitly authorized PCIe profile worker.

        The generated TileLang host wraps the same ``tpuRt`` stream/module in a
        TPUDNN handle, enables recording, performs exactly one kernel launch and
        synchronization, and disables recording before copying results back.
        The worker has the same parent-death/process-group watchdog as CModel.

        Offline decoding never installs packages.  When ``bigTpuProfile`` (and
        its PerfAI module) is absent, raw ``cdm_profile_data_dev*`` artifacts
        are retained and the report returns ``parser_status='unavailable'``.
        """

        if self.config.runtime_mode != "pcie":
            raise TPUProfilingError(
                "run_pcie only supports runtime_mode='pcie'.")
        if not command:
            raise ValueError("TPU profiling command must not be empty.")
        normalized_command = tuple(os.fspath(item) for item in command)
        deadline = _profile_deadline(float(self.config.timeout_s))
        worker_env = _copy_environment(environment)
        worker_env.update(self.pcie_profile_environment_overrides(worker_env))
        self._validate_ppl_dependencies(worker_env)
        output_dir = _profile_output_dir(self.config)
        worker_env.pop("FILE_DUMP_CMD", None)
        worker_env["TILELANG_TPU_PROFILE_SESSION"] = "1"
        worker_env["TILELANG_TPU_PROFILE_OUTPUT_DIR"] = str(output_dir)
        worker_env["TILELANG_TPU_PROFILE_CHIP"] = self.config.chip
        worker_env["TILELANG_TPU_PROFILE_PROGRAMMING_MODEL"] = \
            self.config.programming_model
        worker_env["TILELANG_TPU_PROFILE_RUNTIME_MODE"] = "pcie"
        # The host template independently suppresses benchmark loops during a
        # profile session; keep the worker contract explicit too.
        worker_env["TILELANG_TPU_BENCHMARK_RUNS"] = "0"

        try:
            process = _spawn_guarded_profile_process(
                normalized_command,
                cwd=output_dir,
                environment=worker_env,
            )
        except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
            stdout_path, stderr_path = _write_worker_logs(output_dir, "", str(exc))
            raise TPUProfilingCommandError(
                f"Could not start PCIe TPU profile worker. Logs: "
                f"{stdout_path}, {stderr_path}") from exc
        try:
            remaining = _remaining_timeout(deadline)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(normalized_command, 0)
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_and_collect(process)
            stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
            raise TPUProfilingTimeoutError(
                f"PCIe TPU profile session exceeded its total {self.config.timeout_s}s "
                f"deadline and its worker process group was terminated. Logs: "
                f"{stdout_path}, {stderr_path}")
        except BaseException:
            _kill_process_group(process)
            raise

        stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
        if process.returncode != 0:
            raise TPUProfilingCommandError(
                f"PCIe TPU profile worker exited with status {process.returncode}. "
                f"Logs: {stdout_path}, {stderr_path}")

        raw_files = tuple(sorted(
            path for path in output_dir.glob("cdm_profile_data_dev*")
            if _is_valid_pcie_raw_trace_artifact(path)))
        parser_status = "not-requested"
        parser_message: Optional[str] = None
        report_paths: Tuple[Path, ...] = ()
        decoded_report_paths: Tuple[Path, ...] = ()
        timeline_events: Tuple[TPUInstructionTiming, ...] = ()
        timings: Tuple[TPUInstructionTiming, ...] = ()

        if self.config.postprocess:
            if not raw_files:
                parser_status = "no-raw-trace"
                parser_message = (
                    "The PCIe worker completed but produced no "
                    "cdm_profile_data_dev* artifact; the decoder was not invoked.")
            elif not _PCIE_PROFILE_DECODER_PATH.is_file():
                parser_status = "unavailable"
                parser_message = (
                    "TileLang's offline PCIe profile decoder helper is missing; "
                    "raw recorder artifacts were kept.")
            else:
                decoder: Optional[subprocess.Popen] = None
                try:
                    remaining = _remaining_timeout(deadline)
                    if remaining <= 0:
                        parser_status = "deadline-exhausted"
                        parser_message = (
                            "The PCIe worker used the total profile deadline; "
                            "offline decoding was not started.")
                    else:
                        decoder_env = dict(worker_env)
                        # Decoding is offline and must not inherit permission to
                        # initialize the board a second time.
                        decoder_env.pop("TILELANG_TPU_ALLOW_PCIE_LOAD", None)
                        decoder_env.pop("TILELANG_TPU_ALLOW_PCIE_PROFILE", None)
                        decoder_env.pop("TILELANG_TPU_DEVICE_ID", None)
                        decoder_env.pop("BMLIB_ENABLE_ALL_PROFILE", None)
                        decoder = _spawn_guarded_profile_process(
                            [sys.executable, str(_PCIE_PROFILE_DECODER_PATH),
                             "--profile-dir", str(output_dir),
                             "--arch", _pcie_profile_arch(self.config)],
                            cwd=output_dir,
                            environment=decoder_env,
                        )
                        decoder_stdout, decoder_stderr = decoder.communicate(
                            timeout=_remaining_timeout(deadline))
                except subprocess.TimeoutExpired:
                    assert decoder is not None
                    decoder_stdout, decoder_stderr = _terminate_and_collect(decoder)
                    parser_stdout, parser_stderr = _write_parser_logs(
                        output_dir, decoder_stdout, decoder_stderr)
                    parser_status = "timed-out"
                    parser_message = (
                        "The offline PCIe decoder exceeded the remaining total "
                        f"profile deadline. Logs: {parser_stdout}, {parser_stderr}")
                    decoder = None
                except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
                    parser_stdout, parser_stderr = _write_parser_logs(
                        output_dir, "", str(exc))
                    parser_status = "failed"
                    parser_message = (
                        f"Could not start the offline PCIe decoder. Logs: "
                        f"{parser_stdout}, {parser_stderr}")
                    decoder = None
                except BaseException:
                    if decoder is not None:
                        _kill_process_group(decoder)
                    raise

                if decoder is not None:
                    parser_stdout, parser_stderr = _write_parser_logs(
                        output_dir, decoder_stdout, decoder_stderr)
                    if decoder.returncode == 3:
                        parser_status = "unavailable"
                        parser_message = (
                            "bigTpuProfile/PerfAI is not installed; no package was "
                            "installed automatically and raw PCIe traces were kept. "
                            f"Logs: {parser_stdout}, {parser_stderr}")
                    elif decoder.returncode != 0:
                        parser_status = "failed"
                        parser_message = (
                            f"The offline PCIe decoder exited with status "
                            f"{decoder.returncode}. Logs: {parser_stdout}, {parser_stderr}")
                    else:
                        decoded_report_paths = tuple(sorted(
                            output_dir.rglob(_PCIE_DECODED_REPORT_NAME)))
                        if decoded_report_paths:
                            try:
                                timings = tuple(
                                    event
                                    for path in decoded_report_paths
                                    for event in parse_pcie_decoded_instruction_timings(path))
                            except (json.JSONDecodeError, ValueError) as exc:
                                parser_status = "invalid-report"
                                parser_message = str(exc)
                            else:
                                timeline_events = timings
                                parser_status = (
                                    "ready" if timings else
                                    "no-device-command-events")
                                if not timings:
                                    parser_message = (
                                        "bigTpuProfile decoded the PCIe recorder "
                                        "output but returned no device-command events.")
                        else:
                            parser_status = "missing-report"
                            parser_message = (
                                "The offline PCIe decoder completed without a "
                                "canonical TileLang JSON artifact. Logs: "
                                f"{parser_stdout}, {parser_stderr}")

        return TPUProfileReport(
            config=self.config,
            output_dir=output_dir,
            command=normalized_command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            raw_trace_files=raw_files,
            raw_instructions=(),
            parser_status=parser_status,
            parser_message=parser_message,
            perfai_report_path=report_paths[0] if report_paths else None,
            perfai_report_paths=report_paths,
            decoded_report_path=(
                decoded_report_paths[0] if decoded_report_paths else None),
            decoded_report_paths=decoded_report_paths,
            timeline_events=timeline_events,
            instruction_timings=timings,
        )

    def run_cmodel(self,
                   command: Sequence[PathLike],
                   *,
                   environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
        """Run ``command`` in an isolated CModel profiling session.

        ``command`` must compile/load the TileLang kernel inside this child
        process.  The worker starts a new process group; a timeout terminates
        the whole group, matching the safety discipline used for CModel smoke
        tests.  PCIe is rejected here rather than accidentally running through
        the CModel collection mechanism.
        """

        if self.config.runtime_mode != "cmodel":
            raise TPUProfilingError(
                "run_cmodel only supports runtime_mode='cmodel'. PCIe profiling "
                "requires a separately reviewed one-launch worker and is not dispatched.")
        if not command:
            raise ValueError("TPU profiling command must not be empty.")
        normalized_command = tuple(os.fspath(item) for item in command)
        # This is one wall-clock budget for the whole session, not a fresh
        # timeout for each stage.  In particular, a contended PerfAI lock plus
        # AutoRunner cannot extend a user-visible 60s CModel watchdog into a
        # multi-minute operation.
        deadline = _profile_deadline(float(self.config.timeout_s))
        worker_env = _copy_environment(environment)
        self._validate_ppl_dependencies(worker_env)
        output_dir = _profile_output_dir(self.config)
        # This is the PPL CModel contract.  A relative label is required by the
        # emulator, hence the isolated worker cwd instead of a global parent
        # process chdir.
        worker_env["FILE_DUMP_CMD"] = self.config.label
        worker_env["TILELANG_TPU_PROFILE_SESSION"] = "1"
        worker_env["TILELANG_TPU_PROFILE_OUTPUT_DIR"] = str(output_dir)
        worker_env["TILELANG_TPU_PROFILE_CHIP"] = self.config.chip
        worker_env["TILELANG_TPU_PROFILE_PROGRAMMING_MODEL"] = \
            self.config.programming_model
        worker_env["TILELANG_TPU_PROFILE_RUNTIME_MODE"] = "cmodel"
        # Profiling is exactly one launch.  The generated TileLang host
        # template otherwise honors an inherited benchmark loop.
        worker_env["TILELANG_TPU_BENCHMARK_RUNS"] = "0"
        # PPL's CModel driver sets this for SG2260E.  Make the topology
        # explicit for both supported chips so a worker does not inherit a
        # previous process's emulator-core setting.
        worker_env["TPU_RT_CORE_NUM"] = str(
            self.config.target_spec.chip_spec.physical_core_count)
        # PPL's CModel flow uses FILE_DUMP_CMD, not BMLIB's PCIe recorder.
        worker_env.pop("BMLIB_ENABLE_ALL_PROFILE", None)
        # Never inherit a previously acknowledged board session into a CModel
        # test worker.  The command remains user-supplied, but TileLang's own
        # JIT loader cannot accidentally see a PCIe opt-in from its parent.
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_LOAD", None)
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_PROFILE", None)
        worker_env.pop("TILELANG_TPU_DEVICE_ID", None)

        try:
            process = _spawn_guarded_profile_process(
                normalized_command,
                cwd=output_dir,
                environment=worker_env,
            )
        except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
            stdout_path, stderr_path = _write_worker_logs(output_dir, "", str(exc))
            raise TPUProfilingCommandError(
                f"Could not start CModel TPU profile worker. Logs: "
                f"{stdout_path}, {stderr_path}") from exc
        try:
            remaining = _remaining_timeout(deadline)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(normalized_command, 0)
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_and_collect(process)
            stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
            raise TPUProfilingTimeoutError(
                f"CModel TPU profile session exceeded its total {self.config.timeout_s}s "
                f"deadline and its worker process group was terminated. Logs: "
                f"{stdout_path}, {stderr_path}")
        except BaseException:
            # KeyboardInterrupt, test-runner cancellation, and similar paths
            # must not bypass the same process-tree cleanup as a timeout.
            _kill_process_group(process)
            raise

        stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
        if process.returncode != 0:
            raise TPUProfilingCommandError(
                f"CModel TPU profile worker exited with status {process.returncode}. "
                f"Logs: {stdout_path}, {stderr_path}")

        raw_files = tuple(sorted(
            path for path in output_dir.glob(f"{self.config.label}-*")
            if _RAW_TRACE_ARTIFACT_RE.fullmatch(path.name) is not None))
        raw_instructions = parse_cmodel_raw_instruction_dumps(raw_files)
        parser_status = "not-requested"
        parser_message: Optional[str] = None
        perfai_report_path: Optional[Path] = None
        timeline_events: Tuple[TPUInstructionTiming, ...] = ()
        timings: Tuple[TPUInstructionTiming, ...] = ()

        if self.config.postprocess:
            perfai_root = _find_perfai_root(self.config, worker_env)
            if perfai_root is None:
                parser_status = "unavailable"
                parser_message = (
                    "PerfAI AutoRunner.sh was not found. Raw CModel command dumps were "
                    "kept; set PPL_PERFAI_ROOT or TPUProfilingConfig.perfai_root to "
                    "decode per-instruction timings.")
            elif not raw_files:
                parser_status = "no-raw-trace"
                parser_message = (
                    "The CModel worker completed but produced no FILE_DUMP_CMD artifacts; "
                    "PerfAI was not invoked.")
            else:
                runner = perfai_root / "AutoRunner.sh"
                parser: Optional[subprocess.Popen] = None
                parser_stdout: Optional[Path] = None
                parser_stderr: Optional[Path] = None
                remaining = _remaining_timeout(deadline)
                if remaining <= 0:
                    parser_status = "deadline-exhausted"
                    parser_message = (
                        "The CModel worker used the total profile deadline; PerfAI was not "
                        "started and raw command dumps were kept.")
                else:
                    try:
                        # PPL's AutoRunner uses a mutable ``auto_build`` directory.
                        # Both lock acquisition and the parser share the same absolute
                        # deadline as the CModel worker; neither gets a second timeout.
                        with _exclusive_perfai_workspace(perfai_root, deadline):
                            remaining = _remaining_timeout(deadline)
                            if remaining <= 0:
                                parser_status = "deadline-exhausted"
                                parser_message = (
                                    "The total profile deadline expired while waiting for the "
                                    "PerfAI workspace; AutoRunner was not started.")
                            else:
                                parser_env = dict(worker_env)
                                # This is a parser process, not a CModel worker. Keeping
                                # the raw-dump label here could make a future AutoRunner
                                # child accidentally write into the parsing directory.
                                parser_env.pop("FILE_DUMP_CMD", None)
                                parser = _spawn_guarded_profile_process(
                                    ["bash", str(runner), "-d", str(output_dir), "-e",
                                     _perfai_chip_name(self.config)],
                                    cwd=perfai_root,
                                    environment=parser_env,
                                )
                                remaining = _remaining_timeout(deadline)
                                if remaining <= 0:
                                    parser_stdout_text, parser_stderr_text = \
                                        _terminate_and_collect(parser)
                                    parser_stdout, parser_stderr = _write_parser_logs(
                                        output_dir, parser_stdout_text, parser_stderr_text)
                                    parser_status = "deadline-exhausted"
                                    parser_message = (
                                        "The total profile deadline expired while AutoRunner was "
                                        "starting; its process group was terminated. Logs: "
                                        f"{parser_stdout}, {parser_stderr}")
                                    parser = None
                                else:
                                    try:
                                        parser_stdout_text, parser_stderr_text = parser.communicate(
                                            timeout=remaining)
                                    except subprocess.TimeoutExpired:
                                        parser_stdout_text, parser_stderr_text = \
                                            _terminate_and_collect(parser)
                                        parser_stdout, parser_stderr = _write_parser_logs(
                                            output_dir, parser_stdout_text, parser_stderr_text)
                                        parser_status = "timed-out"
                                        parser_message = (
                                            "PerfAI AutoRunner.sh exceeded the remaining total profile "
                                            "deadline and its process group was terminated. Logs: "
                                            f"{parser_stdout}, {parser_stderr}")
                                        parser = None
                    except _PerfAIWorkspaceBusy as exc:
                        parser_status = "busy"
                        parser_message = (
                            f"{exc}; the total profile deadline expired before AutoRunner could "
                            "start.")
                    except (OSError, subprocess.SubprocessError, TPUProfilingError) as exc:
                        parser_stdout, parser_stderr = _write_parser_logs(output_dir, "", str(exc))
                        parser_status = "failed"
                        parser_message = (
                            f"Could not start PerfAI AutoRunner.sh. Logs: "
                            f"{parser_stdout}, {parser_stderr}")
                        parser = None
                    except BaseException:
                        # Preserve cancellation semantics, but never leave an
                        # AutoRunner process tree behind when pytest is stopped.
                        if parser is not None:
                            _kill_process_group(parser)
                        raise
                if parser is not None:
                    parser_stdout, parser_stderr = _write_parser_logs(
                        output_dir, parser_stdout_text, parser_stderr_text)
                if parser is not None and parser.returncode != 0:
                    parser_status = "failed"
                    parser_message = (
                        f"PerfAI AutoRunner.sh exited with status {parser.returncode}. "
                        f"Logs: {parser_stdout}, {parser_stderr}")
                elif parser is not None:
                    candidate = (
                        output_dir / "result_profiling" / "output" / "PerfWeb" /
                        "profile_data.js")
                    if not candidate.is_file():
                        parser_status = "missing-report"
                        parser_message = (
                            "PerfAI completed without the expected PerfWeb/profile_data.js "
                            f"artifact. Logs: {parser_stdout}, {parser_stderr}")
                    else:
                        perfai_report_path = candidate
                        try:
                            timeline_events = parse_perfai_timeline_events(candidate)
                            timings = tuple(
                                event for event in timeline_events
                                if event.engine.strip().lower() in _DEVICE_COMMAND_ENGINES)
                        except ValueError as exc:
                            parser_status = "invalid-report"
                            parser_message = str(exc)
                        else:
                            parser_status = "ready" if timings else "no-device-command-events"
                            if not timings:
                                parser_message = (
                                    "PerfAI produced a timeline but no recognized TPU command "
                                    "engine rows; inspect timeline_events and update the parser "
                                    "for this PerfAI schema.")

        return TPUProfileReport(
            config=self.config,
            output_dir=output_dir,
            command=normalized_command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            raw_trace_files=raw_files,
            raw_instructions=raw_instructions,
            parser_status=parser_status,
            parser_message=parser_message,
            perfai_report_path=perfai_report_path,
            perfai_report_paths=(perfai_report_path,) if perfai_report_path else (),
            decoded_report_path=None,
            decoded_report_paths=(),
            timeline_events=timeline_events,
            instruction_timings=timings,
        )


def run_tpu_cmodel_profile(command: Sequence[PathLike],
                           config: TPUProfilingConfig,
                           *,
                           environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
    """Convenience wrapper for a one-off isolated CModel profile worker."""

    return TPUInstructionProfiler(config).run_cmodel(command, environment=environment)


def run_tpu_pcie_profile(command: Sequence[PathLike],
                         config: TPUProfilingConfig,
                         *,
                         environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
    """Convenience wrapper for one explicitly authorized PCIe profile worker."""

    return TPUInstructionProfiler(config).run_pcie(command, environment=environment)


__all__ = [
    "TPUInstructionProfiler",
    "TPUInstructionTiming",
    "TPUProfileReport",
    "TPURawInstruction",
    "TPUProfilingCommandError",
    "TPUProfilingConfig",
    "TPUProfilingError",
    "TPUProfilingTimeoutError",
    "parse_perfai_timeline_events",
    "parse_perfai_instruction_timings",
    "parse_pcie_decoded_instruction_timings",
    "parse_cmodel_raw_instruction_dumps",
    "run_tpu_cmodel_profile",
    "run_tpu_pcie_profile",
]
