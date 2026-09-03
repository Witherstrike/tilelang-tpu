from pathlib import Path
import json

import pytest

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_ppl_17_layout(tmp_path, logical_chip="bm1690", arch="tpub_7_1"):
    (tmp_path / "deps/chip").mkdir(parents=True)
    (tmp_path / "deps/chip/chip_map.json").write_text(
        json.dumps({logical_chip: arch}), encoding="utf-8"
    )
    for directory in (
        f"deps/chip/{arch}/TPU1686/kernel/include",
        f"deps/chip/{arch}/lib",
        "deps/common/dev/kernel",
        "deps/common/dev/utils/include",
        "deps/common/host/include",
        "deps/runtime/tpuv7-runtime/include",
        "deps/runtime/tpuv7-runtime/lib",
    ):
        (tmp_path / directory).mkdir(parents=True)
    _touch(tmp_path / "deps/common/dev/utils/src/ppl_helper.c")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libtpuv7_emulator.so")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libfirmware_core.a")


def test_resolve_ppl_17_layout(tmp_path):
    arch = "tpub_7_1"
    _make_ppl_17_layout(tmp_path, arch=arch)

    layout = resolve_ppl_layout(str(tmp_path))

    assert layout.arch == arch
    assert layout.max_core_num == 8
    assert layout.compile_definitions == ("__tpub_7_1__", "__sg2260__")
    assert layout.firmware_archive == (
        tmp_path / f"deps/chip/{arch}/lib/libfirmware_core.a"
    )


def test_resolve_sg2260e_core_count(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, logical_chip="sg2260e", arch=arch)

    layout = resolve_ppl_layout(str(tmp_path), "sg2260e")

    assert layout.arch == arch
    assert layout.max_core_num == 4
    assert layout.compile_definitions == ("__tpub_7_1_e__", "__sg2260e__")


def test_rejects_legacy_ppl_layout(tmp_path):
    chip_root = tmp_path / "runtime/bm1690"
    emulator_root = chip_root / "tpuv7-runtime-emulator"
    for directory in (
        chip_root / "TPU1686/kernel/include",
        chip_root / "lib",
        tmp_path / "runtime/kernel",
        tmp_path / "runtime/customize/include",
        emulator_root / "include",
        emulator_root / "lib",
    ):
        directory.mkdir(parents=True)
    _touch(tmp_path / "runtime/customize/src/ppl_helper.c")
    _touch(emulator_root / "lib/libtpuv7_emulator.so")

    with pytest.raises(FileNotFoundError, match="PPL 1.7 SDK layout is required"):
        resolve_ppl_layout(str(tmp_path))


def test_rejects_unknown_chip_in_ppl_17_layout(tmp_path):
    _make_ppl_17_layout(tmp_path)

    with pytest.raises(ValueError, match="sg2260e"):
        resolve_ppl_layout(str(tmp_path), "sg2260e")
