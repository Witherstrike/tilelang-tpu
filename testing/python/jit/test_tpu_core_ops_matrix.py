# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Acceptance-policy tests for the isolated TPU core-op matrix runner."""

from types import SimpleNamespace

import pytest

from testing.python.jit.tpu_core_ops_matrix import (
    _report_summary,
    _validate_profile_report,
)


def _report(*, parser_status="unavailable", timings=(), has_raw_trace=True):
    return SimpleNamespace(
        output_dir="profile",
        parser_status=parser_status,
        parser_message="decoder is not installed",
        raw_trace_files=("global.profile",) if has_raw_trace else (),
        raw_instructions=(SimpleNamespace(engine="bd", opcode="15"),),
        instruction_timings=tuple(timings),
        has_raw_trace=has_raw_trace,
        has_instruction_timings=bool(timings),
    )


def _timing(*, begin=2, end=7, duration=5):
    return SimpleNamespace(
        engine="bd", unit="cycles", begin=begin, end=end, duration=duration)


def test_numeric_and_raw_acceptance_does_not_require_vendor_decoder():
    report = _report()

    _validate_profile_report(report, require_decoded_timing=False)
    summary = _report_summary(report, require_decoded_timing=False)

    assert summary["status"] == "passed"
    assert summary["parser_status"] == "unavailable"
    assert summary["decoded_timing_required"] is False
    assert summary["decoded_timing_accepted"] is False


def test_raw_trace_remains_mandatory_for_every_acceptance_policy():
    report = _report(has_raw_trace=False)

    with pytest.raises(RuntimeError, match="no profiling trace"):
        _validate_profile_report(report, require_decoded_timing=False)


def test_explicit_decoded_timing_acceptance_rejects_missing_decoder():
    report = _report()

    with pytest.raises(RuntimeError, match="explicitly required"):
        _validate_profile_report(report, require_decoded_timing=True)


def test_explicit_decoded_timing_acceptance_accepts_valid_rows():
    report = _report(parser_status="ready", timings=(_timing(),))

    _validate_profile_report(report, require_decoded_timing=True)
    summary = _report_summary(report, require_decoded_timing=True)

    assert summary["decoded_timing_required"] is True
    assert summary["decoded_timing_accepted"] is True
    assert summary["timed_instruction_count"] == 1


@pytest.mark.parametrize(
    "timing",
    (
        _timing(duration=None),
        _timing(duration=-1),
        _timing(begin=8, end=7, duration=1),
    ),
)
def test_explicit_decoded_timing_acceptance_rejects_invalid_rows(timing):
    report = _report(parser_status="ready", timings=(timing,))

    with pytest.raises(RuntimeError, match="invalid instruction interval"):
        _validate_profile_report(report, require_decoded_timing=True)
