# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Unit tests for the isolated PPL-style TPU instruction profile worker."""

from pathlib import Path
import stat
import sys

import pytest

from tilelang.jit.adapter.tpu_profiling import (
    TPUInstructionProfiler,
    TPUProfilingConfig,
    TPUProfilingError,
    TPUProfilingTimeoutError,
    parse_perfai_instruction_timings,
    parse_perfai_timeline_events,
    run_tpu_cmodel_profile,
)


def _fake_trace_worker(label: str, device_mode: str = "tpukernel"):
    """A child command that models the CModel's relative FILE_DUMP_CMD output."""

    source = (
        "import os\n"
        "from pathlib import Path\n"
        "label = os.environ['FILE_DUMP_CMD']\n"
        "assert '/' not in label\n"
        "assert os.environ['TPU_RT_CORE_NUM'] == '4'\n"
        "assert os.environ['TILELANG_TPU_PROFILE_CHIP'] == 'sg2260e'\n"
        f"assert os.environ['TILELANG_TPU_PROFILE_DEVICE_MODE'] == '{device_mode}'\n"
        "assert os.environ['TILELANG_TPU_PROFILE_RUNTIME_MODE'] == 'cmodel'\n"
        "assert os.environ['TILELANG_TPU_BENCHMARK_RUNS'] == '0'\n"
        "assert 'TILELANG_TPU_ALLOW_PCIE_LOAD' not in os.environ\n"
        "assert 'TILELANG_TPU_ALLOW_PCIE_PROFILE' not in os.environ\n"
        "assert 'TILELANG_TPU_DEVICE_ID' not in os.environ\n"
        "Path(label + '-0-0.BD.0').write_bytes(b'raw-bd')\n"
        "Path(label + '-0-0.GDMA.0').write_bytes(b'raw-gdma')\n"
        "Path(label + '-0-0.BD.0.txt').write_text(\n"
        "    'bd cmd_id=7 bd_func=15\\n', encoding='utf-8')\n"
        "Path(label + '-0-0.GDMA.0.txt').write_text(\n"
        "    'gdma cmd_id=9 gdma_func=6\\n', encoding='utf-8')\n"
    )
    return [sys.executable, "-c", source]


def _create_fake_perfai(root: Path, expected_chip: str = "sg2260e") -> Path:
    perfai_root = root / "PerfAI"
    perfai_root.mkdir()
    runner = perfai_root / "AutoRunner.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -d) run_dir=$2; shift 2 ;;\n"
        "    -e) chip=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        f"test \"${{chip}}\" = {expected_chip}\n"
        "test \"${TILELANG_PROFILE_TEST_TOKEN:-}\" = inherited\n"
        "mkdir -p \"${run_dir}/result_profiling/output/PerfWeb\"\n"
        "cat > \"${run_dir}/result_profiling/output/PerfWeb/profile_data.js\" <<'EOF'\n"
        "let categories = [\"cpu\", \"bdc\", \"gdma\"];\n"
        "let time_header = [\"engine\", \"begin_us\", \"end_us\", \"type\", \"quality\", \"instruction\"];\n"
        "let time_data = [[0, 1, 2, 0, 1, \"host_call\"], [1, 10, 13, 4, 1, \"tiu_mul\"], [2, 11, 15, 0, 1, \"dma_load\"]];\n"
        "EOF\n",
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    return perfai_root


def test_cmodel_profile_worker_uses_a_private_cwd_and_keeps_raw_trace(tmp_path):
    output_dir = tmp_path / "profile"
    config = TPUProfilingConfig(
        chip="sg2260e", output_dir=output_dir, label="unit-trace", postprocess=False)

    report = run_tpu_cmodel_profile(
        _fake_trace_worker(config.label),
        config,
        environment={
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": "0",
        },
    )

    assert report.output_dir.parent == output_dir.resolve()
    assert report.parser_status == "not-requested"
    assert [path.name for path in report.raw_trace_files] == [
        "unit-trace-0-0.BD.0",
        "unit-trace-0-0.BD.0.txt",
        "unit-trace-0-0.GDMA.0",
        "unit-trace-0-0.GDMA.0.txt",
    ]
    assert report.stdout_path.is_file()
    assert report.stderr_path.is_file()
    assert [(item.engine, item.core_id, item.command_id, item.opcode)
            for item in report.raw_instructions] == [
                ("bd", 0, 7, "15"),
                ("gdma", 0, 9, "6"),
            ]


def test_cmodel_profile_runs_explicit_perfai_and_returns_instruction_durations(tmp_path):
    perfai_root = _create_fake_perfai(tmp_path)
    config = TPUProfilingConfig(
        chip="sg2260e",
        output_dir=tmp_path / "profile",
        label="unit-trace",
        perfai_root=perfai_root,
    )

    report = TPUInstructionProfiler(config).run_cmodel(
        _fake_trace_worker(config.label, device_mode="rv"),
        environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
    )

    assert report.parser_status == "ready"
    assert report.perfai_report_path is not None
    assert len(report.timeline_events) == 3
    assert [(item.engine, item.duration, item.unit, item.fields["instruction"])
            for item in report.instruction_timings] == [
                ("bdc", 3.0, "us", "tiu_mul"),
                ("gdma", 4.0, "us", "dma_load"),
            ]


def test_profile_parser_accepts_multiline_perfai_timeline(tmp_path):
    profile_data = tmp_path / "profile_data.js"
    profile_data.write_text(
        "const categories = [\n  \"bdc\",\n];\n"
        "let time_header = [\"engine\", \"start_cycle\", \"end_cycle\"];\n"
        "var time_data = [\n  [0, 4, 9],\n];\n",
        encoding="utf-8",
    )

    timings = parse_perfai_instruction_timings(profile_data)

    assert len(timings) == 1
    assert timings[0].engine == "bdc"
    assert timings[0].duration == 5.0
    assert timings[0].unit == "cycles"


def test_profile_parser_keeps_host_events_out_of_instruction_timings(tmp_path):
    profile_data = tmp_path / "profile_data.js"
    profile_data.write_text(
        "let categories = [\"cpu\", \"bdc\"];\n"
        "let time_header = [\"engine\", \"begin_us\", \"end_us\", \"func_type\", \"info\"];\n"
        "let time_data = [[0, 0, 8, \"host_call\", \"host\"], [1, 2, 7, \"bd_id=3\", \"tiu_mul<br>cycle=5\"]];\n",
        encoding="utf-8",
    )

    timeline = parse_perfai_timeline_events(profile_data)
    timings = parse_perfai_instruction_timings(profile_data)

    assert len(timeline) == 2
    assert [(item.engine, item.command_id, item.opcode, item.duration)
            for item in timings] == [("bdc", 3, "tiu_mul", 5.0)]


def test_rv_profile_uses_the_vendor_perfai_target_spelling(tmp_path):
    perfai_root = _create_fake_perfai(tmp_path, expected_chip="sg2260erv")
    config = TPUProfilingConfig(
        chip="sg2260e",
        device_mode="rv",
        output_dir=tmp_path / "profile",
        label="unit-trace",
        perfai_root=perfai_root,
    )

    report = TPUInstructionProfiler(config).run_cmodel(
        _fake_trace_worker(config.label),
        environment={"TILELANG_PROFILE_TEST_TOKEN": "inherited"},
    )

    assert report.parser_status == "ready"


def test_profile_config_rejects_invalid_chip_mode_and_nonfinite_timeout():
    with pytest.raises(ValueError, match="does not support device_mode='rv'"):
        TPUProfilingConfig(chip="bm1690", device_mode="rv")
    with pytest.raises(ValueError, match="positive number"):
        TPUProfilingConfig(chip="sg2260e", timeout_s=float("nan"))
    with pytest.raises(ValueError, match="positive number"):
        TPUProfilingConfig(chip="sg2260e", timeout_s=float("inf"))


def test_cmodel_profile_timeout_terminates_the_worker_process_group(tmp_path):
    config = TPUProfilingConfig(
        chip="sg2260e", output_dir=tmp_path, timeout_s=0.1, postprocess=False)
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "time.sleep(30)",
    ]

    with pytest.raises(TPUProfilingTimeoutError, match="process group was terminated"):
        TPUInstructionProfiler(config).run_cmodel(command)


def test_pcie_profile_environment_requires_two_acknowledgements():
    profiler = TPUInstructionProfiler(
        TPUProfilingConfig(chip="sg2260e", runtime_mode="pcie"))

    with pytest.raises(TPUProfilingError, match="ALLOW_PCIE_LOAD"):
        profiler.prepare_pcie_environment({})
    with pytest.raises(TPUProfilingError, match="ALLOW_PCIE_PROFILE"):
        profiler.prepare_pcie_environment({"TILELANG_TPU_ALLOW_PCIE_LOAD": "1"})
    with pytest.raises(TPUProfilingError, match="DEVICE_ID"):
        profiler.prepare_pcie_environment({
            "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
            "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
            "TILELANG_TPU_DEVICE_ID": str(2**31),
        })

    environment = profiler.prepare_pcie_environment({
        "TILELANG_TPU_ALLOW_PCIE_LOAD": "1",
        "TILELANG_TPU_ALLOW_PCIE_PROFILE": "1",
        "TILELANG_TPU_DEVICE_ID": "0",
    })
    assert environment == {
        "BMLIB_ENABLE_ALL_PROFILE": "1",
        "PROFILE_RECORD_SIZE": "4096",
        "PROFILE_BOOK_KEEPING": "1",
    }


def test_cmodel_runner_rejects_a_pcie_configuration_before_spawning(tmp_path):
    profiler = TPUInstructionProfiler(
        TPUProfilingConfig(chip="sg2260e", runtime_mode="pcie", output_dir=tmp_path))

    with pytest.raises(TPUProfilingError, match="not dispatched"):
        profiler.run_cmodel([sys.executable, "-c", "raise SystemExit(0)"])
