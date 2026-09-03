# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional, Literal
from .utils import is_cuda_target, is_hip_target, is_cpu_target, is_tpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version
from tvm.target import Target
import ctypes
import os
import tempfile
import subprocess
import shutil
import logging
import re
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.jit.adapter.ppl_layout import PPLLayout, resolve_ppl_layout
from tilelang.engine.tpu_config import (
    bind_tpu_target,
    get_tpu_target_chip,
    TPUCompileConfig,
    resolve_tpu_compile_config,
)

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None
    mode: Optional[Literal["pcie", "cmodel"]] = None

    def __init__(self,
                 target: Target,
                 mode: Optional[Literal["pcie", "cmodel"]] = None,
        tpu_config: Optional[TPUCompileConfig] = None):
        self.target = Target(target)
        self.tpu_config = None
        self.mode = None
        self._ppl_layout: Optional[PPLLayout] = None
        # A TPU host module embeds the absolute path of its private
        # ``libkernel.so``.  Retain the path produced by this generator rather
        # than treating an arbitrary prebuilt ``main.so`` as interchangeable.
        # The latter requires a bundled, validated manifest and is deliberately
        # not supported yet.
        self._tpu_compiled_libpath: Optional[str] = None
        if is_tpu_target(self.target):
            target_chip = get_tpu_target_chip(self.target)
            self.tpu_config = tpu_config or resolve_tpu_compile_config(
                mode=mode, target_chip=target_chip)
            self.target = bind_tpu_target(self.target, self.tpu_config)
            self.mode = self.tpu_config.runtime_mode
        elif tpu_config is not None or mode is not None:
            raise ValueError("TPU configuration can only be used with target='tpu'")
        # TPU compilation emits several cooperating sources and shared objects.
        # Keep every generator in its own directory so a later compile cannot
        # replace an already-loaded CModel/PCIe kernel through global template
        # files or PPL_KERNEL_PATH.
        self.tpu_workspace_dir: Optional[str] = None
        if is_tpu_target(self.target):
            self._ensure_tpu_workspace()

    def _ensure_tpu_workspace(self) -> str:
        if self.tpu_workspace_dir is None:
            self.tpu_workspace_dir = tempfile.mkdtemp(prefix="tilelang-tpu-")
        return self.tpu_workspace_dir

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    def _tpu_runtime_sdk_identity(self, lib_path: str):
        """Return the PPL ABI identity captured while compiling ``lib_path``.

        A standalone TPU ``main.so`` has no manifest that proves which PPL
        headers, runtime libraries, private ``libkernel.so``, chip, model, or
        runtime it was built for.  Do not try to infer that from the current
        environment: only the private artifact just compiled by this generator
        is loadable.  Cache/database rehydration stays fail-closed until it
        carries and validates such a manifest.
        """
        if self._ppl_layout is None or self._tpu_compiled_libpath is None:
            raise RuntimeError(
                "TPU library loading only accepts an artifact compiled by this "
                "LibraryGenerator instance; prebuilt TPU artifacts require a "
                "verified manifest and are currently disabled.")
        requested_path = os.path.realpath(os.fspath(lib_path))
        if requested_path != self._tpu_compiled_libpath:
            raise RuntimeError(
                "TPU library loading rejected a path that was not produced by "
                "this LibraryGenerator instance; rebuild the kernel instead of "
                "loading a prebuilt TPU artifact.")
        return self._ppl_layout.runtime_identity

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        tpu_device_id = None
        tpu_sdk_identity = None
        if is_tpu_target(self.target) and self.mode == "pcie":
            if os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
                raise RuntimeError(
                    "PCIe TPU library loading is disabled by default because dlopen may "
                    "initialize the board runtime. Complete a CModel numerical smoke first, "
                    "then set TILELANG_TPU_ALLOW_PCIE_LOAD=1 and TILELANG_TPU_DEVICE_ID "
                    "for an explicitly supervised PCIe run.")
            # Keep the Python-side dlopen gate at least as strict as the host
            # template.  Waiting until main.so's init() means merely setting
            # ALLOW_PCIE can still load a library which links the TPU runtime.
            device_id = os.environ.get("TILELANG_TPU_DEVICE_ID")
            if device_id is None or re.fullmatch(r"[0-9]+", device_id) is None or \
                    int(device_id) > 2**31 - 1:
                raise RuntimeError(
                    "PCIe TPU library loading requires a non-negative integer "
                    "TILELANG_TPU_DEVICE_ID before dlopen.")
            assert self.tpu_config is not None
            # Keep CModel and PCIe from sharing a process-global vendor
            # runtime.  This happens before ctypes.CDLL, so an invalid mode
            # transition never initializes or touches a board runtime.
            from .tpu import reserve_tpu_runtime_profile
            tpu_device_id = int(device_id)
            tpu_sdk_identity = self._tpu_runtime_sdk_identity(lib_path)
            reserve_tpu_runtime_profile(
                self.tpu_config, tpu_device_id, tpu_sdk_identity)
        elif is_tpu_target(self.target) and self.mode == "cmodel":
            assert self.tpu_config is not None
            # The vendor CModel runtime is process-global. Reserve its core
            # topology before dlopen rather than allowing BM1690 (8 cores) and
            # SG2260E (4 cores) to silently share one initialized runtime.
            from .tpu import reserve_tpu_runtime_profile
            tpu_device_id = 0
            tpu_sdk_identity = self._tpu_runtime_sdk_identity(lib_path)
            reserve_tpu_runtime_profile(
                self.tpu_config, device_id=tpu_device_id,
                sdk_identity=tpu_sdk_identity)
        library = ctypes.CDLL(lib_path)
        if is_tpu_target(self.target):
            assert tpu_device_id is not None
            assert tpu_sdk_identity is not None
            try:
                bind_device = library.tilelang_tpu_bind_device
            except AttributeError as exc:
                raise RuntimeError(
                    "TPU host library lacks tilelang_tpu_bind_device; rebuild it "
                    "with the current TileLang TPU runtime safety template.") from exc
            bind_device.argtypes = [ctypes.c_int]
            bind_device.restype = ctypes.c_int
            status = bind_device(tpu_device_id)
            if status != 0:
                raise RuntimeError(
                    "TPU host library rejected the reserved runtime device "
                    f"{tpu_device_id} (status {status}).")
        return library

    def compile_lib(self, timeout: float = None, with_tl: bool = True):
        target = self.target
        mode = self.mode
        if is_cuda_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            libpath = src.name.replace(".cu", ".so")

            command = [
                "nvcc",
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        elif is_hip_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                src.name,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        elif is_tpu_target(target):
            assert self.tpu_config is not None
            self._tpu_compiled_libpath = None
            import os
            # 设置环境变量
            PPL_TOP = os.environ.get("PPL_PROJECT_ROOT", None)
            if not PPL_TOP:
                raise EnvironmentError("PPL_PROJECT_ROOT environment variable is not set.")
            ppl_layout = resolve_ppl_layout(PPL_TOP, self.tpu_config.chip)
            self._ppl_layout = ppl_layout

            if self.tpu_config.device_mode == "rv":
                # RVT is a direct PPL ABI bridge: the generated PPL source
                # includes rvt_api.h and users emit explicit rvt_* externs.
                # A missing header is a chip/SDK capability error, rather
                # than a silent fallback to the TPU-Kernel path.
                ppl_layout.require_rvt_api()

            if self.mode=="pcie":
                self.tpu_compile_pcie(timeout=timeout, layout=ppl_layout)
            elif self.mode=="cmodel":
                self.tpu_compile_cmodel(timeout=timeout, layout=ppl_layout)
            else:
                raise ValueError(f"Unsupported compile mode: {self.mode}")
            self.srcpath = self._ensure_tpu_workspace()
            self.libpath = os.path.join(self.srcpath, "main.so")
            self._tpu_compiled_libpath = os.path.realpath(self.libpath)
            return

        else:
            raise ValueError(f"Unsupported target: {target}")


        if with_tl:
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
            command += ["-diag-suppress=20013"]
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.tpu_workspace_dir is not None:
            shutil.rmtree(self.tpu_workspace_dir, ignore_errors=True)
            self.tpu_workspace_dir = None
            self.libpath = None
            self.srcpath = None
            self._tpu_compiled_libpath = None
            return
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath

    def _prepare_cmodel_kernel_source(self, kernel_path: str):
        with open(kernel_path, "r") as f:
            kernel_code = f.read()

        sanitized = kernel_code.replace("      tpu_parallel_start(); \n", "")
        sanitized = sanitized.replace("      tpu_parallel_end(); \n", "")
        sanitized = sanitized.replace("tpu_parallel_start(); \n", "")
        sanitized = sanitized.replace("tpu_parallel_end(); \n", "")

        if sanitized != kernel_code:
            logger.info("Stripping TPU pipeline parallel markers for cmodel execution")
            with open(kernel_path, "w") as f:
                f.write(sanitized)

    @staticmethod
    def _run_tpu_command(command, task_name, timeout):
        try:
            subprocess.run(command, timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"{task_name} failed: {e}") from e

    @staticmethod
    def _ppl_compile_flags(layout: PPLLayout,
                           src_dir: str,
                           device_mode: str = "tpukernel"):
        definitions = [
            *(f"-D{definition}" for definition in layout.compile_definitions),
            "-DTILELANG_PPL_HELPER_HAS_GET_DTYPE",
        ]
        if device_mode == "rv":
            layout.require_rvt_api()
            definitions.append("-DTILELANG_TPU_RV")
        elif device_mode == "tpukernel":
            definitions.append("-DTILELANG_TPU_TPUKERNEL")
        else:
            raise ValueError(
                f"Unsupported TPU device mode {device_mode!r}; expected 'tpukernel' or 'rv'")
        includes = [f"-I{path}" for path in layout.include_dirs]
        include_dir = os.path.join(src_dir, "include")
        if os.path.isdir(include_dir):
            includes.append(f"-I{include_dir}")
        return definitions, includes

    def tpu_compile_pcie(self, timeout, layout: PPLLayout):
        cross_gcc = str(layout.pcie_cross_gcc())

        src_dir = self._ensure_tpu_workspace()
        definitions, includes = self._ppl_compile_flags(
            layout, src_dir, self.tpu_config.device_mode)
        common = definitions + ["-Dlibkernel_EXPORTS"] + includes + [
            "-O3", "-DNDEBUG", "-fPIC", "-flto"
        ]
        kernel_o = os.path.join(src_dir, "kernel.o")
        helper_o = os.path.join(src_dir, "ppl_helper.o")
        libkernel = os.path.join(src_dir, "libkernel.so")

        self._run_tpu_command(
            [cross_gcc, *common, "-c", os.path.join(src_dir, "kernel.c"), "-o", kernel_o],
            "Compile TPU kernel", timeout)
        self._run_tpu_command(
            [cross_gcc, *common, "-c", str(layout.ppl_helper_source), "-o", helper_o],
            "Compile PPL helper", timeout)
        self._run_tpu_command(
            [cross_gcc, "-shared", "-fPIC", "-flto", "-Wl,--no-undefined",
             "-Wl,-soname,libkernel.so", "-o", libkernel, kernel_o, helper_o,
             "-Wl,--whole-archive", str(layout.firmware_archive),
             "-Wl,--no-whole-archive", "-Wl,-s", "-ldl", "-lm"],
            "Link PCIe libkernel.so", timeout)

        host_common = definitions + includes + [
            "-O3", "-DNDEBUG", "-std=c++17", "-fPIC",
            f'-DTILELANG_PPL_KERNEL_PATH="{libkernel}"',
        ]
        kernel_host_o = os.path.join(src_dir, "kernel_host.o")
        main_o = os.path.join(src_dir, "main.o")
        self._run_tpu_command(
            ["g++", *host_common, "-c", os.path.join(src_dir, "kernel.cpp"), "-o", kernel_host_o],
            "Compile TPU host wrapper", timeout)
        self._run_tpu_command(
            ["g++", *host_common, "-c", os.path.join(src_dir, "main.cpp"), "-o", main_o],
            "Compile TPU host entry", timeout)
        self._run_tpu_command(
            ["g++", "-shared", "-fPIC", "-Wl,--no-undefined", "-o",
             os.path.join(src_dir, "main.so"), kernel_host_o, main_o,
             f"-L{layout.runtime_lib}",
             f"-Wl,--disable-new-dtags,-rpath,{layout.runtime_lib}",
             "-ltpuv7_rt", "-Wl,--no-as-needed", "-lcdm_daemon_emulator",
             "-Wl,--as-needed", "-lpthread"],
            "Link PCIe main.so", timeout)

    def tpu_compile_cmodel(self, timeout, layout: PPLLayout):
        src_dir = self._ensure_tpu_workspace()
        definitions, includes = self._ppl_compile_flags(
            layout, src_dir, self.tpu_config.device_mode)
        definitions.append("-DUSING_CMODEL")
        common = definitions + includes + ["-O3", "-DNDEBUG", "-fPIC"]

        kernel_c = os.path.join(src_dir, "kernel.c")
        kernel_cpp_o = os.path.join(src_dir, "kernel_cpp.o")
        main_cpp_o = os.path.join(src_dir, "main_cpp.o")
        kernel_c_o = os.path.join(src_dir, "kernel_c.o")
        helper_o = os.path.join(src_dir, "ppl_helper_c.o")
        libkernel = os.path.join(src_dir, "libkernel.so")
        main_so = os.path.join(src_dir, "main.so")
        rpath = f"{layout.runtime_lib}:{layout.backend_lib}"
        # TPUv7 defaults to eight emulator cores. SG2260E exposes four, and
        # launching the extra scalar-emulator workers makes them address
        # non-existent cores before the first kernel can complete.  Embed the
        # value in main.so so cached/from-database processes do not depend on
        # this compiler process having set TPU_RT_CORE_NUM already.
        host_common = common + [
            f'-DTILELANG_PPL_KERNEL_PATH="{libkernel}"',
            f'-DTILELANG_TPU_CMODEL_CORE_NUM="{layout.max_core_num}"',
        ]

        logger.info("Compiling TPU cmodel kernel for PPL 1.7 architecture %s", layout.arch)
        self._prepare_cmodel_kernel_source(kernel_c)
        self._run_tpu_command(
            ["/usr/bin/c++", *host_common, "-std=c++17", "-c",
             os.path.join(src_dir, "kernel.cpp"), "-o", kernel_cpp_o],
            "Compile TPU host wrapper", timeout)
        self._run_tpu_command(
            ["/usr/bin/c++", *host_common, "-std=c++17", "-c",
             os.path.join(src_dir, "main.cpp"), "-o", main_cpp_o],
            "Compile TPU host entry", timeout)
        self._run_tpu_command(
            ["/usr/bin/cc", *common, "-Dkernel_EXPORTS", "-c", kernel_c, "-o", kernel_c_o],
            "Compile TPU cmodel kernel", timeout)
        self._run_tpu_command(
            ["/usr/bin/cc", *common, "-Dkernel_EXPORTS", "-c",
             str(layout.ppl_helper_source), "-o", helper_o],
            "Compile PPL helper", timeout)
        self._run_tpu_command(
            ["/usr/bin/cc", "-shared", "-fPIC", "-Wl,--no-undefined",
             "-Wl,-soname,libkernel.so", "-o", libkernel, kernel_c_o, helper_o,
             f"-Wl,-rpath,{rpath}", str(layout.emulator_library), "-lm"],
            "Link cmodel libkernel.so", timeout)
        self._run_tpu_command(
            ["/usr/bin/c++", "-shared", "-fPIC", "-o", main_so,
             kernel_cpp_o, main_cpp_o, f"-L{layout.runtime_lib}", f"-L{layout.backend_lib}",
             f"-Wl,--disable-new-dtags,-rpath,{rpath}", "-ltpuv7_rt",
             "-lcdm_daemon_emulator", "-lpthread"],
            "Link cmodel main.so", timeout)
