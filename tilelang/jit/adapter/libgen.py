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
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.jit.adapter.ppl_layout import PPLLayout, resolve_ppl_layout
from tilelang.engine.tpu_config import TPUCompileConfig, resolve_tpu_compile_config

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None
    mode: Literal["pcie", "cmodel"] = "pcie"

    def __init__(self,
                 target: Target,
                 mode: Optional[Literal["pcie", "cmodel"]] = None,
                 tpu_config: Optional[TPUCompileConfig] = None):
        self.target = target
        self.tpu_config = tpu_config or resolve_tpu_compile_config(mode=mode)
        self.mode = self.tpu_config.runtime_mode
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

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        if is_tpu_target(self.target) and self.mode == "pcie":
            if os.environ.get("TILELANG_TPU_ALLOW_PCIE_LOAD") != "1":
                raise RuntimeError(
                    "PCIe TPU library loading is disabled by default because dlopen may "
                    "initialize the board runtime. Complete a CModel numerical smoke first, "
                    "then set TILELANG_TPU_ALLOW_PCIE_LOAD=1 and TILELANG_TPU_DEVICE_ID "
                    "for an explicitly supervised PCIe run.")
        return ctypes.CDLL(lib_path)

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
            import os
            # 设置环境变量
            PPL_TOP = os.environ.get("PPL_PROJECT_ROOT", None)
            if not PPL_TOP:
                raise EnvironmentError("PPL_PROJECT_ROOT environment variable is not set.")
            ppl_layout = resolve_ppl_layout(PPL_TOP, self.tpu_config.chip)

            if self.tpu_config.device_mode == "rv":
                # RVT is a direct PPL ABI bridge: the generated PPL source
                # includes rvt_api.h and users emit explicit rvt_* externs.
                # A missing header is a chip/SDK capability error, rather
                # than a silent fallback to the atomic path.
                ppl_layout.require_rvt_api()

            if self.mode=="pcie":
                self.tpu_compile_pcie(timeout=timeout, layout=ppl_layout)
            elif self.mode=="cmodel":
                self.tpu_compile_cmodel(timeout=timeout, layout=ppl_layout)
            else:
                raise ValueError(f"Unsupported compile mode: {self.mode}")
            self.srcpath = self._ensure_tpu_workspace()
            self.libpath = os.path.join(self.srcpath, "main.so")
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
    def _ppl_compile_flags(layout: PPLLayout, src_dir: str, device_mode: str = "atomic"):
        definitions = [
            *(f"-D{definition}" for definition in layout.compile_definitions),
            "-DTILELANG_PPL_HELPER_HAS_GET_DTYPE",
        ]
        if device_mode == "rv":
            layout.require_rvt_api()
            definitions.append("-DTILELANG_TPU_RV")
        includes = [f"-I{path}" for path in layout.include_dirs]
        include_dir = os.path.join(src_dir, "include")
        if os.path.isdir(include_dir):
            includes.append(f"-I{include_dir}")
        return definitions, includes

    def tpu_compile_pcie(self, timeout, layout: PPLLayout):
        cross_compile = str(layout.toolchain_dir / "bin/riscv64-unknown-linux-gnu-")
        cross_gcc = cross_compile + "gcc"
        if not os.path.isfile(cross_gcc):
            raise FileNotFoundError(f"PPL PCIe cross compiler is missing: {cross_gcc}")

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
