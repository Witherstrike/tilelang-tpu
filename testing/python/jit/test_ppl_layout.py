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
    assert layout.root == tmp_path.resolve()
    assert layout.runtime_identity == (
        str(tmp_path.resolve()),
        str((tmp_path / "deps/runtime/tpuv7-runtime/lib").resolve()),
        str((tmp_path / f"deps/chip/{arch}/lib").resolve()),
    )


def test_resolve_sg2260e_core_count(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, logical_chip="sg2260e", arch=arch)

    layout = resolve_ppl_layout(str(tmp_path), "sg2260e")

    assert layout.arch == arch
    assert layout.max_core_num == 4
    assert layout.compile_definitions == ("__tpub_7_1_e__", "__sg2260e__")


def test_rvt_api_is_required_for_rv_mode(tmp_path):
    arch = "tpub_7_1_e"
    _make_ppl_17_layout(tmp_path, logical_chip="sg2260e", arch=arch)
    layout = resolve_ppl_layout(str(tmp_path), "sg2260e")

    with pytest.raises(FileNotFoundError, match="rvt_api.h"):
        layout.require_rvt_api()

    _touch(tmp_path / f"deps/chip/{arch}/TPU1686/kernel/include/rvt_api.h")
    assert layout.require_rvt_api() == layout.rvt_api_header


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


def test_rejects_chip_map_that_disagrees_with_the_capability_registry(tmp_path):
    _make_ppl_17_layout(
        tmp_path, logical_chip="sg2260e", arch="tpub_7_1")

    with pytest.raises(ValueError, match="disagrees with TileLang's validated capability"):
        resolve_ppl_layout(str(tmp_path), "sg2260e")


def test_pcie_cross_compiler_is_discovered_without_a_pinned_sdk_version(tmp_path):
    _make_ppl_17_layout(tmp_path, logical_chip="sg2260e", arch="tpub_7_1_e")
    gcc = tmp_path / (
        "third_party/toolchains_dir/any-ppl-release/bin/"
        "riscv64-unknown-linux-gnu-gcc")
    _touch(gcc)

    layout = resolve_ppl_layout(str(tmp_path), "sg2260e")
    assert layout.pcie_cross_gcc() == gcc


def test_pcie_cross_compiler_rejects_ambiguous_sdk_toolchains(tmp_path):
    _make_ppl_17_layout(tmp_path)
    for release in ("first", "second"):
        _touch(tmp_path / (
            f"third_party/toolchains_dir/{release}/bin/"
            "riscv64-unknown-linux-gnu-gcc"))

    layout = resolve_ppl_layout(str(tmp_path))
    with pytest.raises(ValueError, match="selection is ambiguous"):
        layout.pcie_cross_gcc()
