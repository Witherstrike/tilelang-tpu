# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Safe PPL-style instruction profiling for TileLang TPU test programs.

TileLang emits raw PPL C directly and does not pass through ``ppl-compile``.
Consequently PPL's deprecated ``--profiling``/``--autotune`` frontend switch is
not meaningful here.  What is reusable is its runtime protocol:

* on CModel, execute one freshly compiled test worker in a dedicated directory
  with ``FILE_DUMP_CMD`` set, then optionally run an explicitly supplied
  PerfAI installation over the raw command dumps;
* on PCIe, prepare (but do not dispatch) the PPL profiling environment behind
  separate, explicit safety gates.

The child-process boundary is intentional.  Both the vendor runtime and the
process working directory are global state, and a profile worker must never
load an arbitrary prebuilt ``main.so`` from another JIT instance.  The command
passed to :class:`TPUInstructionProfiler` is therefore expected to compile and
load its own private TileLang JIT artifact.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, os.PathLike]

_SUPPORTED_CHIPS = ("bm1690", "sg2260e")
_SUPPORTED_DEVICE_MODES = ("tpukernel", "rv")
_SUPPORTED_CHIP_DEVICE_MODES = {
    "bm1690": ("tpukernel",),
    "sg2260e": ("tpukernel", "rv"),
}
_CMODEL_CORE_COUNTS = {
    "bm1690": 8,
    "sg2260e": 4,
}
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


class TPUProfilingError(RuntimeError):
    """Base class for TPU profiling failures that preserve collected artifacts."""


class TPUProfilingTimeoutError(TPUProfilingError):
    """A profile worker exceeded its timeout and its process group was stopped."""


class TPUProfilingCommandError(TPUProfilingError):
    """The isolated profile worker exited unsuccessfully."""


@dataclass(frozen=True)
class TPUProfilingConfig:
    """Configuration for one isolated TPU instruction-profile session.

    ``runtime_mode='pcie'`` is accepted only to create the environment contract
    through :meth:`TPUInstructionProfiler.prepare_pcie_environment`.  Dispatch
    is deliberately not implemented by this module: a PCIe profile must have
    a separately reviewed TPUDNN host-launch path and an explicitly supervised
    worker.
    """

    chip: str
    device_mode: str = "tpukernel"
    runtime_mode: str = "cmodel"
    output_dir: Optional[PathLike] = None
    label: str = "tilelang"
    timeout_s: float = 60.0
    perfai_root: Optional[PathLike] = None
    postprocess: bool = True
    profile_record_size: int = 4096
    profile_book_keeping: int = 1

    def __post_init__(self) -> None:
        chip = self.chip.strip().lower() if isinstance(self.chip, str) else self.chip
        if chip not in _SUPPORTED_CHIPS:
            raise ValueError(
                f"Unsupported TPU profiling chip {self.chip!r}; "
                f"expected one of: {', '.join(_SUPPORTED_CHIPS)}")
        if self.device_mode not in _SUPPORTED_DEVICE_MODES:
            raise ValueError(
                f"Unsupported TPU profiling device mode {self.device_mode!r}; "
                f"expected 'tpukernel' or 'rv'.")
        if self.device_mode not in _SUPPORTED_CHIP_DEVICE_MODES[chip]:
            supported = ", ".join(_SUPPORTED_CHIP_DEVICE_MODES[chip])
            raise ValueError(
                f"TPU profiling chip {chip!r} does not support device_mode="
                f"{self.device_mode!r}; supported modes: {supported}.")
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
        object.__setattr__(self, "chip", chip)


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
    timeline_events: Tuple[TPUInstructionTiming, ...] = ()
    instruction_timings: Tuple[TPUInstructionTiming, ...] = ()

    @property
    def has_raw_trace(self) -> bool:
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
        return Path(tempfile.mkdtemp(prefix="tilelang-tpu-profile-")).resolve()
    output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Never emulate PPL's ``rmtree(<target>/profiling)`` behaviour. A caller
    # may preserve many regression profiles under one root, so every run owns
    # a new subdirectory and cannot erase a prior trace.
    return Path(tempfile.mkdtemp(prefix=f"{config.label}-", dir=output_root))


def _kill_process_group(process: subprocess.Popen) -> None:
    """Stop a worker and all of its children without touching its parent shell."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


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

    if config.device_mode == "rv":
        return "sg2260erv"
    # This preserves PPL's CModel special case in ppl_compile.py: BM1690's
    # PerfAI target is named sg2260.
    return "sg2260" if config.chip == "bm1690" else config.chip


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


class TPUInstructionProfiler:
    """Run one fresh TileLang TPU test worker with PPL-compatible profiling."""

    def __init__(self, config: TPUProfilingConfig):
        self.config = config

    def prepare_pcie_environment(
            self, environment: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
        """Return only candidate environment overrides for a future PCIe worker.

        The returned mapping is **not** a complete environment and does not
        enable TileLang PCIe instruction profiling: the present direct
        ``tpuRtKernelLaunch`` host ABI has no TPUDNN profile session. This
        preflight only documents PPL's environment contract and enforces the
        three explicit acknowledgements needed before a future, separately
        reviewed TPUDNN one-launch worker could be introduced.

        No loading, initialization, or board dispatch occurs here.
        """

        if self.config.runtime_mode != "pcie":
            raise ValueError("prepare_pcie_environment requires runtime_mode='pcie'.")
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
        output_dir = _profile_output_dir(self.config)
        worker_env = _copy_environment(environment)
        # This is the PPL CModel contract.  A relative label is required by the
        # emulator, hence the isolated worker cwd instead of a global parent
        # process chdir.
        worker_env["FILE_DUMP_CMD"] = self.config.label
        worker_env["TILELANG_TPU_PROFILE_SESSION"] = "1"
        worker_env["TILELANG_TPU_PROFILE_OUTPUT_DIR"] = str(output_dir)
        worker_env["TILELANG_TPU_PROFILE_CHIP"] = self.config.chip
        worker_env["TILELANG_TPU_PROFILE_DEVICE_MODE"] = self.config.device_mode
        worker_env["TILELANG_TPU_PROFILE_RUNTIME_MODE"] = "cmodel"
        # Profiling is exactly one launch.  The generated TileLang host
        # template otherwise honors an inherited benchmark loop.
        worker_env["TILELANG_TPU_BENCHMARK_RUNS"] = "0"
        # PPL's CModel driver sets this for SG2260E.  Make the topology
        # explicit for both supported chips so a worker does not inherit a
        # previous process's emulator-core setting.
        worker_env["TPU_RT_CORE_NUM"] = str(_CMODEL_CORE_COUNTS[self.config.chip])
        # PPL's CModel flow uses FILE_DUMP_CMD, not BMLIB's PCIe recorder.
        worker_env.pop("BMLIB_ENABLE_ALL_PROFILE", None)
        # Never inherit a previously acknowledged board session into a CModel
        # test worker.  The command remains user-supplied, but TileLang's own
        # JIT loader cannot accidentally see a PCIe opt-in from its parent.
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_LOAD", None)
        worker_env.pop("TILELANG_TPU_ALLOW_PCIE_PROFILE", None)
        worker_env.pop("TILELANG_TPU_DEVICE_ID", None)

        try:
            process = subprocess.Popen(
                normalized_command,
                cwd=output_dir,
                env=worker_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            stdout_path, stderr_path = _write_worker_logs(output_dir, "", str(exc))
            raise TPUProfilingCommandError(
                f"Could not start CModel TPU profile worker. Logs: "
                f"{stdout_path}, {stderr_path}") from exc
        try:
            stdout, stderr = process.communicate(timeout=float(self.config.timeout_s))
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            stdout, stderr = process.communicate()
            stdout_path, stderr_path = _write_worker_logs(output_dir, stdout, stderr)
            raise TPUProfilingTimeoutError(
                f"CModel TPU profile worker exceeded {self.config.timeout_s}s and its "
                f"process group was terminated. Logs: {stdout_path}, {stderr_path}")

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
                try:
                    parser_env = dict(worker_env)
                    # This is a parser process, not a CModel worker. Keeping
                    # the raw-dump label here could make a future AutoRunner
                    # child accidentally write into the parsing directory.
                    parser_env.pop("FILE_DUMP_CMD", None)
                    parser = subprocess.Popen(
                        ["bash", str(runner), "-d", str(output_dir), "-e",
                         _perfai_chip_name(self.config)],
                        cwd=perfai_root,
                        env=parser_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        errors="replace",
                        start_new_session=True,
                    )
                    try:
                        parser_stdout_text, parser_stderr_text = parser.communicate(
                            timeout=float(self.config.timeout_s))
                    except subprocess.TimeoutExpired:
                        _kill_process_group(parser)
                        parser_stdout_text, parser_stderr_text = parser.communicate()
                        parser_stdout, parser_stderr = _write_parser_logs(
                            output_dir, parser_stdout_text, parser_stderr_text)
                        parser_status = "timed-out"
                        parser_message = (
                            f"PerfAI AutoRunner.sh exceeded {self.config.timeout_s}s and its "
                            f"process group was terminated. Logs: {parser_stdout}, {parser_stderr}")
                        parser = None
                except OSError as exc:
                    parser_stdout, parser_stderr = _write_parser_logs(output_dir, "", str(exc))
                    parser_status = "failed"
                    parser_message = (
                        f"Could not start PerfAI AutoRunner.sh. Logs: "
                        f"{parser_stdout}, {parser_stderr}")
                    parser = None
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
            timeline_events=timeline_events,
            instruction_timings=timings,
        )


def run_tpu_cmodel_profile(command: Sequence[PathLike],
                           config: TPUProfilingConfig,
                           *,
                           environment: Optional[Mapping[str, str]] = None) -> TPUProfileReport:
    """Convenience wrapper for a one-off isolated CModel profile worker."""

    return TPUInstructionProfiler(config).run_cmodel(command, environment=environment)


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
    "parse_cmodel_raw_instruction_dumps",
    "run_tpu_cmodel_profile",
]
