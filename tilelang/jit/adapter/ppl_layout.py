# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Resolve paths in the supported PPL 1.7 SDK release layout."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Tuple

from tilelang.engine.tpu_config import get_tpu_chip_spec


@dataclass(frozen=True)
class PPLLayout:
    root: Path
    logical_chip: str
    arch: str
    max_core_num: int
    compile_definitions: Tuple[str, ...]
    kernel_include: Path
    kernel_common_include: Path
    device_utils_include: Path
    host_include: Path
    runtime_include: Path
    tpudnn_include: Path
    runtime_lib: Path
    backend_lib: Path
    tpudnn_library: Path
    ppl_helper_source: Path
    emulator_library: Path
    firmware_archive: Path
    toolchains_root: Path

    @property
    def runtime_identity(self) -> Tuple[str, str, str]:
        """Canonical CModel SDK/runtime paths kept for compatibility."""
        return tuple(
            str(path.resolve())
            for path in (self.root, self.runtime_lib, self.backend_lib))

    def pcie_runtime_lib(self) -> Path:
        """Resolve the installed board runtime, never the SDK CModel runtime.

        PPL 1.7 compiles against headers in ``deps/`` but prepends the installed
        ``/opt/tpuv7/tpuv7-current/lib`` for PCIe execution.  The SDK's own
        ``deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so`` depends on
        ``libcdm_daemon_emulator.so`` and therefore cannot drive the board.
        An explicit override supports non-standard driver installations.
        """

        raw_path = os.environ.get(
            "TILELANG_TPU_PCIE_RUNTIME_PATH",
            "/opt/tpuv7/tpuv7-current/lib",
        )
        runtime_lib = Path(raw_path).expanduser().resolve()
        if runtime_lib == self.runtime_lib.resolve():
            raise ValueError(
                "TILELANG_TPU_PCIE_RUNTIME_PATH resolves to PPL's SDK CModel "
                "runtime. PCIe must use the installed TPUv7 board runtime.")
        runtime_so = runtime_lib / "libtpuv7_rt.so"
        if not runtime_so.is_file():
            raise FileNotFoundError(
                "TPUv7 PCIe board runtime is missing; expected "
                f"{runtime_so}. Install the matching TPUv7 driver runtime or "
                "set TILELANG_TPU_PCIE_RUNTIME_PATH to its lib directory.")
        return runtime_lib

    def runtime_identity_for(self, runtime_mode: str) -> Tuple[str, str, str]:
        """Return the SDK/runtime identity for one validated host mode."""

        if runtime_mode == "cmodel":
            runtime_lib = self.runtime_lib
        elif runtime_mode == "pcie":
            runtime_lib = self.pcie_runtime_lib()
        else:
            raise ValueError(f"Unsupported TPU runtime mode: {runtime_mode!r}")
        return tuple(
            str(path.resolve())
            for path in (self.root, runtime_lib, self.backend_lib))

    @property
    def include_dirs(self) -> Tuple[Path, ...]:
        return tuple(
            path for path in (
                self.kernel_include,
                self.kernel_common_include,
                self.device_utils_include,
                self.host_include,
                self.runtime_include,
                self.tpudnn_include,
            ) if path.is_dir())

    @property
    def rvt_api_header(self) -> Path:
        """The PPL 1.7 RV Tensor ABI header for this chip architecture."""
        return self.kernel_include / "rvt_api.h"

    def require_rvt_api(self) -> Path:
        """Return the RVT header or explain why ``device_mode=\"rv\"`` is invalid."""
        spec = get_tpu_chip_spec(self.logical_chip)
        if not spec.supports("rv"):
            raise ValueError(
                f"TPU chip {spec.name!r} does not support the RV programming model")
        header = self.rvt_api_header
        if not header.is_file():
            raise FileNotFoundError(
                "PPL 1.7 RV target requested, but this chip SDK does not provide "
                f"rvt_api.h: {header}")
        return header

    def pcie_cross_gcc(self) -> Path:
        """Find the one PPL-provided PCIe compiler without pinning an SDK version.

        CModel users do not need a cross compiler, so discovery is intentionally
        delayed until a PCIe build.  More than one candidate is an ambiguity,
        not a reason to silently select an arbitrary SDK revision.
        """
        candidates = tuple(sorted(
            self.toolchains_root.glob("*/bin/riscv64-unknown-linux-gnu-gcc")))
        if not candidates:
            raise FileNotFoundError(
                "PPL PCIe cross compiler is missing; expected "
                f"riscv64-unknown-linux-gnu-gcc under {self.toolchains_root}")
        if len(candidates) != 1:
            rendered = "\n  ".join(str(candidate) for candidate in candidates)
            raise ValueError(
                "PPL PCIe cross compiler selection is ambiguous; expected one candidate:\n  "
                + rendered)
        return candidates[0]


def _require_paths(layout: PPLLayout) -> PPLLayout:
    required = {
        "kernel headers": layout.kernel_include,
        "device helper headers": layout.device_utils_include,
        "host headers": layout.host_include,
        "TPUv7 runtime headers": layout.runtime_include,
        "TPUDNN profiling headers": layout.tpudnn_include,
        "TPUv7 runtime libraries": layout.runtime_lib,
        "chip backend libraries": layout.backend_lib,
        "TPUDNN profiling library": layout.tpudnn_library,
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
    chip_spec = get_tpu_chip_spec(logical_chip)
    logical_chip = chip_spec.name
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
    if arch != chip_spec.ppl_arch:
        raise ValueError(
            "PPL SDK chip map disagrees with TileLang's validated capability "
            f"for {logical_chip!r}: expected {chip_spec.ppl_arch!r}, got {arch!r}")

    chip_root = root / "deps/chip" / arch
    runtime_root = root / "deps/runtime/tpuv7-runtime"
    common_root = root / "deps/common"
    return _require_paths(
        PPLLayout(
            root=root,
            logical_chip=logical_chip,
            arch=arch,
            max_core_num=chip_spec.physical_core_count,
            compile_definitions=chip_spec.ppl_compile_definitions,
            kernel_include=chip_root / "TPU1686/kernel/include",
            kernel_common_include=common_root / "dev/kernel",
            device_utils_include=common_root / "dev/utils/include",
            host_include=common_root / "host/include",
            runtime_include=runtime_root / "include",
            tpudnn_include=chip_root / "TPU1686/tpuDNN/include",
            runtime_lib=runtime_root / "lib",
            backend_lib=chip_root / "lib",
            tpudnn_library=chip_root / "lib/libtpudnn.so",
            ppl_helper_source=common_root / "dev/utils/src/ppl_helper.c",
            emulator_library=chip_root / "lib/libtpuv7_emulator.so",
            firmware_archive=chip_root / "lib/libfirmware_core.a",
            toolchains_root=root / "third_party/toolchains_dir",
        ))
