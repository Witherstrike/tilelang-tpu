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
    monkeypatch.setattr(supervisor, "sdk_identity", lambda: "sdk")
    monkeypatch.setattr(supervisor, "fingerprint", lambda: "fixed")
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


def test_pcie_requires_matching_complete_cmodel_proof(monkeypatch, tmp_path):
    smi = tmp_path / "smi"
    smi.touch()
    manifest = tmp_path / "proof.json"
    manifest.write_text(
        json.dumps({
            "runtime": "cmodel",
            "source_sha256": "old",
            "passed_cases": []
        }))
    monkeypatch.setattr(sys, "argv", [
        "run", "--runtime", "pcie", "--output-dir",
        str(tmp_path), "--smi",
        str(smi), "--cmodel-manifest",
        str(manifest)
    ])
    monkeypatch.setattr(supervisor, "sdk_identity", lambda: "sdk")
    monkeypatch.setattr(supervisor, "fingerprint", lambda: "new")
    monkeypatch.setattr(supervisor, "cases", lambda p: ["first"])
    with pytest.raises(SystemExit, match="proof incomplete"):
        supervisor.main()


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
