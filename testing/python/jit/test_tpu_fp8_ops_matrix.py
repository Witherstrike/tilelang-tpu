# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Safety and acceptance tests for the FP8 profiling matrix."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import tpu_fp8_ops_matrix as matrix
import tpu_fp8_ops_worker as worker


def _args(**overrides):
    values = {
        "runtime_mode": "cmodel",
        "timeout": 10.0,
        "chips": None,
        "dtypes": ["e4m3"],
        "cases": ["copy"],
        "device_id": None,
        "allow_pcie": False,
        "allow_pcie_profile": False,
        "require_decoded_timing": False,
        "pcie_decoder_python": None,
        "pcie_decoder_pythonpath": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _profile_environment(monkeypatch, runtime_mode):
    for name in worker._PCIE_GATE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TILELANG_TPU_PROFILE_SESSION", "1")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_CHIP", "sg2260e")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_PROGRAMMING_MODEL", "tpukernel")
    monkeypatch.setenv("TILELANG_TPU_PROFILE_RUNTIME_MODE", runtime_mode)
    monkeypatch.setenv("TILELANG_TPU_BENCHMARK_RUNS", "0")


def _timing():
    return SimpleNamespace(engine="bdc", unit="ns", begin=2, end=7, duration=5)


def _report():
    return SimpleNamespace(
        output_dir=Path("profile"),
        parser_status="ready",
        parser_message=None,
        raw_trace_files=(Path("cdm_profile_data_dev0-0"),),
        raw_instructions=(),
        instruction_timings=(_timing(),),
        has_raw_trace=True,
        has_instruction_timings=True,
        decoder_identity={
            "package": "bigTpuProfile",
            "package_version": "test",
            "parser_api": "BMProfileParserPerfAI.parse",
        },
    )


def test_cmodel_worker_rejects_inherited_pcie_gate(monkeypatch):
    _profile_environment(monkeypatch, "cmodel")
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")

    with pytest.raises(RuntimeError, match="refuses PCIe"):
        worker._profile_selection()


def test_pcie_worker_requires_all_gates_and_numeric_device(monkeypatch):
    _profile_environment(monkeypatch, "pcie")
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_LOAD", "1")
    monkeypatch.setenv("TILELANG_TPU_ALLOW_PCIE_PROFILE", "1")
    monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "0")

    with pytest.raises(RuntimeError, match="recorder gates"):
        worker._profile_selection()

    monkeypatch.setenv("BMLIB_ENABLE_ALL_PROFILE", "1")
    assert worker._profile_selection() == ("sg2260e", "pcie")
    monkeypatch.setenv("TILELANG_TPU_DEVICE_ID", "device-zero")
    with pytest.raises(RuntimeError, match="numeric device"):
        worker._profile_selection()


def test_pcie_cli_requires_dual_acknowledgement_device_and_one_chip():
    with pytest.raises(RuntimeError, match="both"):
        matrix._validate_args(_args(runtime_mode="pcie", chips=["sg2260e"]))
    with pytest.raises(RuntimeError, match="device-id"):
        matrix._validate_args(
            _args(
                runtime_mode="pcie",
                chips=["sg2260e"],
                allow_pcie=True,
                allow_pcie_profile=True,
            ))
    with pytest.raises(RuntimeError, match="exactly one"):
        matrix._validate_args(
            _args(
                runtime_mode="pcie",
                chips=None,
                device_id=0,
                allow_pcie=True,
                allow_pcie_profile=True,
            ))

    matrix._validate_args(
        _args(
            runtime_mode="pcie",
            chips=["sg2260e"],
            device_id=0,
            allow_pcie=True,
            allow_pcie_profile=True,
        ))


def test_cmodel_cli_rejects_pcie_and_decoder_options():
    with pytest.raises(RuntimeError, match="invalid for CModel"):
        matrix._validate_args(_args(allow_pcie=True))
    with pytest.raises(RuntimeError, match="invalid for CModel"):
        matrix._validate_args(
            _args(pcie_decoder_pythonpath=[Path("decoder")]))
    with pytest.raises(RuntimeError, match="invalid for CModel"):
        matrix._validate_args(_args(require_decoded_timing=True))


def test_worker_environment_removes_inherited_pcie_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PPL_PROJECT_ROOT", str(tmp_path / "ppl"))
    for name in matrix._PCIE_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "poison")

    cmodel = matrix._worker_environment(tmp_path, tmp_path / "scratch", "cmodel", None)
    assert not set(matrix._PCIE_ENVIRONMENT_VARIABLES).intersection(cmodel)

    pcie = matrix._worker_environment(tmp_path, tmp_path / "scratch", "pcie", 7)
    assert pcie["TILELANG_TPU_ALLOW_PCIE_LOAD"] == "1"
    assert pcie["TILELANG_TPU_ALLOW_PCIE_PROFILE"] == "1"
    assert pcie["TILELANG_TPU_DEVICE_ID"] == "7"
    assert "BMLIB_ENABLE_ALL_PROFILE" not in pcie


def test_required_decoder_is_preflighted_before_pcie_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            events.append(("preflight", dict(environment)))
            return {
                "package": "bigTpuProfile",
                "package_version": "test",
                "parser_api": "BMProfileParserPerfAI.parse",
            }

        def run_pcie(self, command, *, environment):
            events.append(("dispatch", command, dict(environment)))
            return _report()

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(
        matrix,
        "git_source_identity",
        lambda _root: {
            "git_commit": "test-revision",
            "implementation_worktree_dirty": False,
            "source_identity_scope": "tracked files excluding research/**",
        },
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    args = _args(
        runtime_mode="pcie",
        chips=["sg2260e"],
        device_id=0,
        allow_pcie=True,
        allow_pcie_profile=True,
        require_decoded_timing=True,
    )

    assert matrix._run_matrix(args, tmp_path, output_dir, {"PPL_PROJECT_ROOT": "/sdk"}) == 0
    assert [event[0] for event in events] == ["preflight", "dispatch"]
    summary = matrix.json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["decoder_preflight"]["status"] == "passed"
    case = summary["cases"]["sg2260e/tpukernel/e4m3/copy"]
    assert case["decoded_timing_accepted"] is True
    assert case["timed_instruction_count"] == 1


def test_failed_decoder_preflight_prevents_hardware_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeProfiler:

        def __init__(self, config):
            self.config = config

        def preflight_pcie_decoder(self, *, environment):
            events.append("preflight")
            raise RuntimeError("decoder unavailable")

        def run_pcie(self, command, *, environment):
            events.append("dispatch")
            raise AssertionError("hardware must not be touched")

    monkeypatch.setattr(matrix, "TPUInstructionProfiler", FakeProfiler)
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: {})
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    args = _args(
        runtime_mode="pcie",
        chips=["sg2260e"],
        device_id=0,
        allow_pcie=True,
        allow_pcie_profile=True,
        require_decoded_timing=True,
    )

    assert matrix._run_matrix(args, tmp_path, output_dir, {}) == 1
    assert events == ["preflight"]
    summary = matrix.json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is False
    assert summary["decoder_preflight"]["status"] == "failed"
