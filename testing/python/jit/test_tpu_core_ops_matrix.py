# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Acceptance-policy tests for the isolated TPU core-op matrix runner."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import tpu_core_ops_matrix as matrix_module
import tpu_fp8_ops_worker
import tpu_matrix_common
import tpu_profile_worker
from tpu_core_ops_matrix import (
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
    return SimpleNamespace(engine="bd", unit=unit, begin=begin, end=end, duration=duration)


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


@pytest.mark.parametrize(
    "source_name",
    (
        "testing/python/jit/tpukernel_ops_worker.py",
        "testing/python/jit/tpu_fp8_ops_worker.py",
        "testing/python/jit/tpu_profile_worker.py",
        "tpu_demo/elementwise/tpu_test_elementwise.py",
        "tpu_demo/matmul/tpu_test_matmul_fp16.py",
    ),
)
def test_tir_script_workers_keep_evaluated_annotations(source_name):
    """TVM Script consumes ``T.Tensor`` annotations as live objects."""

    repo_root = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo_root / source_name).read_text(encoding="utf-8"))
    annotation_futures = [
        alias.name for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" for alias in node.names
    ]

    assert "annotations" not in annotation_futures


def test_fp8_elementwise_case_names_preserve_unsuffixed_operations():
    cases = ("add", "sub", "mul", "add-broadcast", "sub-broadcast", "mul-broadcast")

    assert [tpu_fp8_ops_worker._elementwise_operation(case) for case in cases
           ] == ["add", "sub", "mul", "add", "sub", "mul"]


def test_git_source_identity_records_revision_and_tracked_dirty_state(monkeypatch):
    responses = iter((
        SimpleNamespace(stdout="55c1c6d\n"),
        SimpleNamespace(stdout=" M tracked.py\n"),
    ))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(tpu_matrix_common.subprocess, "run", fake_run)

    identity = tpu_matrix_common.git_source_identity(matrix_module.Path("/repo"))

    assert identity == {
        "git_commit": "55c1c6d",
        "implementation_worktree_dirty": True,
        "source_identity_scope": "tracked files excluding research/**",
    }
    assert calls[1][0][-2:] == [".", ":(exclude)research/**"]
    assert all(call[1]["timeout"] == 5 for call in calls)


def test_portable_copy_cases_are_worker_cases_and_default_matrix_cases():
    expected = (
        "copy-fp32-local-roundtrip",
        "copy-fp32-global-to-global",
        "copy-fp16-local-roundtrip",
        "copy-fp16-global-to-global",
    )

    assert tuple(tpu_profile_worker._COPY_CASES) == expected
    assert expected == matrix_module._COPY_CASES
    assert matrix_module._CASES[-len(expected):] == expected


def test_runner_dispatches_every_portable_copy_case(monkeypatch, tmp_path):
    calls = []
    configs = []

    class FakeProfiler:

        def __init__(self, config):
            configs.append(config)

        def run_cmodel(self, command, *, environment):
            calls.append((command, environment))
            return _report()

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(
        matrix_module,
        "git_source_identity",
        lambda _repo_root: {
            "git_commit": "test-revision",
            "implementation_worktree_dirty": False,
            "source_identity_scope": "tracked files excluding research/**",
        },
    )
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = SimpleNamespace(
        runtime_mode="cmodel",
        require_decoded_timing=False,
        timeout=10.0,
    )

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        matrix_module._COPY_CASES,
        {"PPL_PROJECT_ROOT": "/sdk"},
    )

    assert status == 0
    assert [call[0][-2:] for call in calls
           ] == [["--case", case] for case in matrix_module._COPY_CASES]
    assert all(call[0][0] == sys.executable for call in calls)
    assert [config.label for config in configs
           ] == [f"sg2260e-rv-{case}" for case in matrix_module._COPY_CASES]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert set(summary["cases"]) == set(f"sg2260e/rv/{case}" for case in matrix_module._COPY_CASES)


def test_required_decoding_preflights_before_any_pcie_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            events.append(("preflight", environment))
            return {
                "package": "bigTpuProfile",
                "package_version": "0.3.5",
                "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
            }

        def run_pcie(self, command, *, environment):
            events.append(("dispatch", command, environment))
            return _report(parser_status="ready", timings=(_timing(),))

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(
        matrix_module,
        "git_source_identity",
        lambda _repo_root: {
            "git_commit": "test-revision",
            "implementation_worktree_dirty": False,
            "source_identity_scope": "tracked files excluding research/**",
        },
    )
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = SimpleNamespace(
        runtime_mode="pcie",
        require_decoded_timing=True,
        timeout=10.0,
        pcie_decoder_python=None,
        pcie_decoder_pythonpath=(),
    )

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add",),
        {"PPL_PROJECT_ROOT": "/sdk"},
    )

    assert status == 0
    assert [event[0] for event in events] == ["preflight", "dispatch"]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["decoder_preflight"] == {
        "status": "ready",
        "identity": {
            "package": "bigTpuProfile",
            "package_version": "0.3.5",
            "parser_api": "bigTpuProfile.bmprofile_perfAI_2260.BMProfileParserPerfAI.parse",
        },
    }


def test_failed_decoder_preflight_stops_matrix_without_dispatch(monkeypatch, tmp_path):
    dispatches = []

    class FakeProfiler:

        def __init__(self, config):
            pass

        def preflight_pcie_decoder(self, *, environment):
            raise RuntimeError("decoder unavailable")

        def run_pcie(self, command, *, environment):
            dispatches.append(command)
            raise AssertionError("hardware dispatch must not be reached")

    monkeypatch.setattr(matrix_module, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(
        matrix_module,
        "git_source_identity",
        lambda _repo_root: {
            "git_commit": "test-revision",
            "implementation_worktree_dirty": False,
            "source_identity_scope": "tracked files excluding research/**",
        },
    )
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    args = SimpleNamespace(
        runtime_mode="pcie",
        require_decoded_timing=True,
        timeout=10.0,
        pcie_decoder_python=None,
        pcie_decoder_pythonpath=(),
    )

    status = matrix_module._run_matrix(
        args,
        tmp_path,
        output_dir,
        (("sg2260e", "rv"),),
        ("elementwise-add",),
        {"PPL_PROJECT_ROOT": "/sdk"},
    )

    assert status == 1
    assert dispatches == []
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["decoder_preflight"]["status"] == "failed"
    assert summary["cases"] == {}


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
