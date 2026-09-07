# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Shared profiling-evidence policy for TPU matrix runners."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from tilelang.jit.adapter.tpu_profiling import _pcie_instruction_timings_error


def validate_profile_report(report: Any, *, require_decoded_timing: bool) -> None:
    """Require numerical-worker success, raw evidence, and optional timing."""

    if not report.has_raw_trace:
        raise RuntimeError("successful dispatch produced no profiling trace")
    if not require_decoded_timing:
        return
    if report.parser_status != "ready" or not report.has_instruction_timings:
        raise RuntimeError(
            "decoded instruction timing was explicitly required but the successful "
            "PCIe dispatch did not produce it "
            f"(parser_status={report.parser_status!r}, message={report.parser_message!r})")
    timing_error = _pcie_instruction_timings_error(report.instruction_timings)
    if timing_error is not None:
        raise RuntimeError("PCIe decoder produced an invalid instruction interval: " +
                           timing_error)


def profile_report_summary(report: Any, *, require_decoded_timing: bool) -> dict[str, Any]:
    """Project one profiler result into the stable matrix-summary shape."""

    raw_by_engine = Counter(item.engine for item in report.raw_instructions)
    raw_by_opcode = Counter(
        item.opcode for item in report.raw_instructions if item.opcode is not None)
    decoded_timing_error = (
        _pcie_instruction_timings_error(report.instruction_timings)
        if report.parser_status == "ready" else "decoder is not ready")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    if decoded_timing_error is None:
        for timing in report.instruction_timings:
            grouped[(timing.engine, timing.unit)].append(float(timing.duration))
    timing_by_engine = {
        f"{engine}:{unit}": {
            "count": len(durations),
            "sum": sum(durations),
            "min": min(durations),
            "max": max(durations),
        }
        for (engine, unit), durations in sorted(grouped.items())
    }
    return {
        "status": "passed",
        "artifact_dir": str(report.output_dir),
        "parser_status": report.parser_status,
        "raw_trace_file_count": len(report.raw_trace_files),
        "raw_instruction_count": len(report.raw_instructions),
        "raw_instruction_count_by_engine": dict(sorted(raw_by_engine.items())),
        "raw_instruction_count_by_opcode": dict(sorted(raw_by_opcode.items())),
        "timed_instruction_count": len(report.instruction_timings),
        "timing_by_engine_and_unit": timing_by_engine,
        "decoder_identity": dict(getattr(report, "decoder_identity", {})),
        "decoded_timing_required": require_decoded_timing,
        "decoded_timing_accepted": decoded_timing_error is None,
    }


__all__ = ["profile_report_summary", "validate_profile_report"]
