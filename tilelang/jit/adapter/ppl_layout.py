# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Resolve paths in the supported PPL 1.7 SDK release layout."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PPLLayout:
    logical_chip: str
    arch: str
    max_core_num: int
    compile_definitions: Tuple[str, ...]
    kernel_include: Path
    kernel_common_include: Path
    device_utils_include: Path
    host_include: Path
    runtime_include: Path
    runtime_lib: Path
    backend_lib: Path
    ppl_helper_source: Path
    emulator_library: Path
    firmware_archive: Path
    toolchain_dir: Path

    @property
    def include_dirs(self) -> Tuple[Path, ...]:
        return tuple(
            path for path in (
                self.kernel_include,
                self.kernel_common_include,
                self.device_utils_include,
                self.host_include,
                self.runtime_include,
            ) if path.is_dir())


_PPL_17_CHIP_DEFINITIONS = {
    "tpub_7_1": ("__tpub_7_1__", "__sg2260__"),
    "tpub_7_1_e": ("__tpub_7_1_e__", "__sg2260e__"),
}

_PPL_17_MAX_CORE_NUM = {
    "tpub_7_1": 8,
    "tpub_7_1_e": 4,
}


def _require_paths(layout: PPLLayout) -> PPLLayout:
    required = {
        "kernel headers": layout.kernel_include,
        "device helper headers": layout.device_utils_include,
        "host headers": layout.host_include,
        "TPUv7 runtime headers": layout.runtime_include,
        "TPUv7 runtime libraries": layout.runtime_lib,
        "chip backend libraries": layout.backend_lib,
        "ppl_helper.c": layout.ppl_helper_source,
        "TPUv7 emulator": layout.emulator_library,
        "firmware archive": layout.firmware_archive,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete PPL SDK layout:\n  " + "\n  ".join(missing))
    return layout


def resolve_ppl_layout(ppl_root: str, logical_chip: str = "bm1690") -> PPLLayout:
    """Return the PPL 1.7 toolchain paths for ``logical_chip``.

    TileLang-TPU deliberately supports only the PPL 1.7 ``deps/`` release
    layout. Keeping one layout avoids silently compiling a kernel with a
    legacy header/library mixture after the SDK has been upgraded.
    """
    root = Path(ppl_root).expanduser().resolve()
    chip_map_path = root / "deps/chip/chip_map.json"
    if not chip_map_path.is_file():
        raise FileNotFoundError(
            "PPL 1.7 SDK layout is required; expected chip map at "
            f"{chip_map_path}. The legacy PPL runtime/ layout is unsupported."
        )

    try:
        chip_map = json.loads(chip_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PPL 1.7 chip map: {chip_map_path}") from exc
    if not isinstance(chip_map, dict) or logical_chip not in chip_map:
        raise ValueError(f"Chip {logical_chip!r} is not present in {chip_map_path}")

    arch = chip_map[logical_chip]
    if not isinstance(arch, str):
        raise ValueError(f"Invalid architecture for chip {logical_chip!r} in {chip_map_path}")

    chip_root = root / "deps/chip" / arch
    runtime_root = root / "deps/runtime/tpuv7-runtime"
    common_root = root / "deps/common"
    definitions = _PPL_17_CHIP_DEFINITIONS.get(arch, (f"__{arch}__",))
    return _require_paths(
        PPLLayout(
            logical_chip=logical_chip,
            arch=arch,
            max_core_num=_PPL_17_MAX_CORE_NUM.get(arch, 1),
            compile_definitions=definitions,
            kernel_include=chip_root / "TPU1686/kernel/include",
            kernel_common_include=common_root / "dev/kernel",
            device_utils_include=common_root / "dev/utils/include",
            host_include=common_root / "host/include",
            runtime_include=runtime_root / "include",
            runtime_lib=runtime_root / "lib",
            backend_lib=chip_root / "lib",
            ppl_helper_source=common_root / "dev/utils/src/ppl_helper.c",
            emulator_library=chip_root / "lib/libtpuv7_emulator.so",
            firmware_archive=chip_root / "lib/libfirmware_core.a",
            toolchain_dir=root / "third_party/toolchains_dir"
            / "Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1",
        ))
