# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Acceptance-policy tests for the isolated TPU core-op matrix runner."""

from types import SimpleNamespace

import pytest

from testing.python.jit import tpu_core_ops_matrix as matrix_module
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


def _timing(*, begin=2, end=7, duration=5, unit="ns"):
    return SimpleNamespace(
        engine="bd", unit=unit, begin=begin, end=end, duration=duration)


_INVALID_TIMINGS = (
    _timing(duration=None),
    _timing(duration=True),
    _timing(duration=float("nan")),
    _timing(duration=float("inf")),
    _timing(duration=-1),
    _timing(begin=True),
    _timing(begin=float("nan")),
    _timing(end=float("inf")),
    _timing(begin=8, end=7, duration=1),
    _timing(unit=""),
    _timing(unit="cycles"),
)


def test_git_source_identity_records_revision_and_tracked_dirty_state(monkeypatch):
    responses = iter((
        SimpleNamespace(stdout="55c1c6d\n"),
        SimpleNamespace(stdout=" M tracked.py\n"),
    ))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(matrix_module.subprocess, "run", fake_run)

    identity = matrix_module._git_source_identity(matrix_module.Path("/repo"))

    assert identity == {
        "git_commit": "55c1c6d",
        "tracked_worktree_dirty": True,
    }
    assert calls[1][0][-1] == "--untracked-files=no"
    assert all(call[1]["timeout"] == 5 for call in calls)


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
    _INVALID_TIMINGS,
)
def test_explicit_decoded_timing_acceptance_rejects_invalid_rows(timing):
    report = _report(parser_status="ready", timings=(timing,))

    with pytest.raises(RuntimeError, match="invalid instruction interval"):
        _validate_profile_report(report, require_decoded_timing=True)


@pytest.mark.parametrize("timing", _INVALID_TIMINGS)
def test_optional_decoded_timing_never_accepts_invalid_rows(timing):
    report = _report(parser_status="ready", timings=(timing,))

    _validate_profile_report(report, require_decoded_timing=False)
    summary = _report_summary(report, require_decoded_timing=False)

    assert summary["status"] == "passed"
    assert summary["decoded_timing_accepted"] is False
    assert summary["timing_by_engine_and_unit"] == {}
