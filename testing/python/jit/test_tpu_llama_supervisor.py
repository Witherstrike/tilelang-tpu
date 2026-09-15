"""Fail-closed supervisor tests: no real runtime or board is loaded."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

spec = importlib.util.spec_from_file_location(
    "llama_supervisor",
    Path(__file__).resolve().parents[3] / "tools/run_llama_validation.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.mark.parametrize("status,util,memory", [("Fault", "0%", "0MB"), ("Active", "1%", "0MB"),
                                                ("Active", "0%", "1MB")])
def test_nonidle_board_is_rejected(monkeypatch, tmp_path, status, util, memory):
    state = {"card0": {"chip0": {"status": status, "tpu_util": util, "mem_usage": memory}}}

    def status_command(command, output, timeout):
        output.write(json.dumps(state))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "run_child", status_command)
    with pytest.raises(RuntimeError, match="not quiescent"):
        supervisor.board_state("smi", 0, tmp_path / "board.json")


def test_timeout_stops_matrix_and_poisons_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run", "--runtime", "cmodel", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(supervisor, "cases", lambda p: ["first", "must-not-run"])
    calls = []

    def timeout(command, *args):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(supervisor, "run_child", timeout)
    with pytest.raises(SystemExit, match="STOP"):
        supervisor.main()
    assert len(calls) == 1
    assert (tmp_path / "POISONED").exists()
    with pytest.raises(SystemExit, match="poisoned"):
        supervisor.main()
    assert len(calls) == 1


def test_pcie_requires_board_status_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run", "--runtime", "pcie", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(supervisor, "cases", lambda p: ["first"])
    with pytest.raises(SystemExit, match="requires --smi"):
        supervisor.main()


def test_pcie_requires_explicit_runtime_load_authorization(monkeypatch, tmp_path):
    smi = tmp_path / "tpu-smi"
    smi.touch()
    monkeypatch.delenv("TILELANG_TPU_ALLOW_PCIE_LOAD", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["run", "--runtime", "pcie", "--output-dir",
         str(tmp_path / "run"), "--smi",
         str(smi)])
    monkeypatch.setattr(supervisor, "cases", lambda p: ["first"])
    with pytest.raises(SystemExit, match="TILELANG_TPU_ALLOW_PCIE_LOAD=1"):
        supervisor.main()


def test_case_selector_runs_only_exact_requested_case(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run", "--runtime", "cmodel", "--output-dir",
        str(tmp_path), "--case", "test_tpu_llama_ops/second"
    ])
    monkeypatch.setattr(supervisor, "cases", lambda path: ["first", "second"])
    calls = []

    def succeed(command, log, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "run_child", succeed)
    supervisor.main()
    assert len(calls) == 1
    assert calls[0][1].endswith("test_tpu_llama_ops.py")
    assert calls[0][3] == "second"


def test_timeout_never_waits_for_a_stuck_driver_process(monkeypatch):
    calls = []

    class Child:
        pid = 12345

        def wait(self, timeout):
            calls.append(("wait", timeout))
            raise subprocess.TimeoutExpired("kernel", timeout)

        def poll(self):
            calls.append(("poll",))
            return None

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(supervisor.os, "killpg", lambda *a: calls.append(("killpg", *a)))
    with pytest.raises(subprocess.TimeoutExpired) as error:
        supervisor.run_child(["kernel"], None, 1)
    assert error.value.process_group == 12345
    assert [call[0] for call in calls] == ["wait", "killpg", "poll"]


@pytest.mark.parametrize("utils,passes,count", [
    (["9%", "0%", "0%"], True, 3),
    (["0%", "9%", "0%", "0%"], True, 4),
    (["9%"] * 10, False, 10),
    (["unknown"], False, 1),
])
def test_post_kernel_settle_is_bounded(monkeypatch, tmp_path, utils, passes, count):
    calls = []

    def status_command(command, output, timeout):
        util = utils[len(calls)]
        calls.append(util)
        output.write(
            json.dumps(
                {"card0": {
                    "chip0": {
                        "status": "Active",
                        "mem_usage": "0MB",
                        "tpu_util": util
                    }
                }}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "run_child", status_command)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    destination = tmp_path / "board.json"
    if passes:
        supervisor.board_state("smi", 0, destination, settle=True)
        assert destination.exists()
    else:
        with pytest.raises(RuntimeError, match="not quiescent"):
            supervisor.board_state("smi", 0, destination, settle=True)
    assert len(calls) == count
    assert len(list(tmp_path.glob("board-*.json"))) == count


@pytest.mark.parametrize("status,memory", [("Fault", "0MB"), ("Active", "1MB")])
def test_settle_does_not_retry_faults_or_retained_memory(monkeypatch, tmp_path, status, memory):

    def status_command(command, output, timeout):
        output.write(
            json.dumps(
                {"card0": {
                    "chip0": {
                        "status": status,
                        "mem_usage": memory,
                        "tpu_util": "0%"
                    }
                }}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "run_child", status_command)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: pytest.fail("must fail immediately"))
    with pytest.raises(RuntimeError, match="not quiescent"):
        supervisor.board_state("smi", 0, tmp_path / "board.json", settle=True)
