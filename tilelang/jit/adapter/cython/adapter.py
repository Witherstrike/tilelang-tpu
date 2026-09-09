# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from ..base import BaseKernelAdapter
import ctypes
from typing import List, Optional, Union, Callable, Dict, Tuple, Any, Mapping
from tilelang import tvm as tvm
from tvm.target import Target
from tilelang.engine.param import KernelParam
from tvm import tir
from tvm.relay import TensorType
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.utils import is_cuda_target, is_hip_target, is_cpu_target, is_tpu_target
from tilelang.jit.adapter.tpu import (
    make_tpu_forward,
    reject_unverified_tpu_database_artifact,
)
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
from tilelang.utils.tensor import map_torch_type
from tilelang.contrib.cc import get_cplus_compiler
from tilelang.engine.tpu_config import TPURuntimeConfig, TPUTargetSpec
import torch
import sys
import sysconfig
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_CYTHON_CACHE_SCHEMA = 1
_CYTHON_BUILD_FLAGS = (
    "-shared",
    "-pthread",
    "-fPIC",
    "-fwrapv",
    "-O2",
    "-Wall",
    "-fno-strict-aliasing",
)
_CYTHON_WRAPPER_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cython_cache_root() -> Path:
    configured = os.environ.get("TILELANG_CACHE_DIR", "~/.tilelang/cache")
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / "cython-adapter" / f"v{_CYTHON_CACHE_SCHEMA}"


def _cython_build_identity(source_path: Path, compiler: Path,
                           cython_version: str) -> Dict[str, Any]:
    python_include_path = sysconfig.get_path("include")
    if not python_include_path:
        raise RuntimeError("Python's C include directory is unavailable.")
    return {
        "cache_schema": _CYTHON_CACHE_SCHEMA,
        "source_sha256": _sha256_file(source_path),
        "build_flags": list(_CYTHON_BUILD_FLAGS),
        "python": {
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
            "executable": str(Path(sys.executable).resolve()),
            "soabi": sysconfig.get_config_var("SOABI"),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "sysconfig_platform": sysconfig.get_platform(),
        },
        "cython_version": cython_version,
        "compiler": {
            "path": str(compiler),
            "sha256": _sha256_file(compiler),
        },
        "python_include_path": str(Path(python_include_path).resolve()),
    }


def _load_cython_extension(library_path: Path):
    if library_path.is_symlink() or not library_path.is_file():
        raise RuntimeError(f"Cython adapter cache is not a regular file: {library_path}")
    module_name = "cython_wrapper"
    spec = importlib.util.spec_from_file_location(module_name, library_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create an import spec for {library_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    origin = Path(module.__file__).resolve() if module.__file__ else None
    if origin != library_path.resolve():
        raise RuntimeError(
            f"Loaded Cython adapter from {origin}, expected {library_path.resolve()}")
    wrapper_type = getattr(module, "CythonKernelWrapper", None)
    if wrapper_type is None:
        raise RuntimeError("Cython adapter library has no CythonKernelWrapper")
    return wrapper_type


def _cached_cython_wrapper(entry: Path, identity: Mapping[str, Any]):
    library_path = entry / "cython_wrapper.so"
    manifest_path = entry / "manifest.json"
    if any(path.is_symlink() for path in (entry, library_path, manifest_path)):
        raise RuntimeError(f"Cython adapter cache must not contain symlinks: {entry}")
    if not library_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("identity") != identity:
        return None
    if manifest.get("library_sha256") != _sha256_file(library_path):
        return None
    return _load_cython_extension(library_path)


def _load_cython_kernel_wrapper():
    """Build and load the non-TPU Cython bridge on first actual use."""

    try:
        import Cython
    except ImportError as error:
        raise RuntimeError(
            "The Cython execution backend requires the Cython Python package.") from error
    compiler_name = get_cplus_compiler()
    if not compiler_name:
        raise RuntimeError("No C++ compiler is available for the Cython execution backend.")
    compiler_path = shutil.which(compiler_name)
    if compiler_path is None:
        raise RuntimeError(f"Cannot resolve the Cython execution backend compiler: {compiler_name}")
    compiler = Path(compiler_path).resolve(strict=True)
    source_path = Path(__file__).resolve().with_name("cython_wrapper.pyx")
    identity = _cython_build_identity(source_path, compiler, Cython.__version__)
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    cache_root = _cython_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / fingerprint
    if entry.is_symlink():
        raise RuntimeError(f"Cython adapter cache entry must not be a symlink: {entry}")
    entry.mkdir(parents=False, exist_ok=True)

    with _CYTHON_WRAPPER_LOCK:
        lock_path = entry / ".lock"
        with lock_path.open("a+b") as lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except ImportError:
                fcntl = None
            cached = _cached_cython_wrapper(entry, identity)
            if cached is not None:
                return cached

            library_path = entry / "cython_wrapper.so"
            manifest_path = entry / "manifest.json"
            with tempfile.TemporaryDirectory(
                    prefix=f".{fingerprint}.", dir=cache_root) as build_dir_name:
                build_dir = Path(build_dir_name)
                generated_cpp = build_dir / "cython_wrapper.cpp"
                generated_library = build_dir / "cython_wrapper.so"
                subprocess.run(
                    [
                        sys.executable, "-m", "cython",
                        str(source_path), "--cplus", "-o",
                        str(generated_cpp)
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        compiler, *_CYTHON_BUILD_FLAGS, f"-I{identity['python_include_path']}",
                        str(generated_cpp), "-o",
                        str(generated_library)
                    ],
                    check=True,
                )
                if generated_library.is_symlink() or not generated_library.is_file():
                    raise RuntimeError("Cython compilation produced no regular shared library")
                os.replace(generated_library, library_path)
                try:
                    wrapper_type = _load_cython_extension(library_path)
                except Exception:
                    library_path.unlink(missing_ok=True)
                    raise
                manifest = {
                    "identity": identity,
                    "library_sha256": _sha256_file(library_path),
                }
                temporary_manifest = entry / f".manifest.{os.getpid()}.tmp"
                try:
                    temporary_manifest.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary_manifest, manifest_path)
                finally:
                    temporary_manifest.unlink(missing_ok=True)
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return wrapper_type


class CythonKernelAdapter(BaseKernelAdapter):
    """Convert TVM/TIR functions to callable native kernels using Cython.
    
    This adapter handles:
    1. Converting TIR functions to compiled target libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping native kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target: Union[str, Target] = "cuda"
    ir_module: Optional[tvm.IRModule] = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    kernel_global_source: Optional[str] = None
    lib: Optional[ctypes.CDLL] = None  # Compiled library handle
    wrapped_source: Optional[str] = None  # Generated C++ wrapper code
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None
    # Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
    ptr_map: Optional[Dict[int, str]] = None
    # Maps buffer variables to their corresponding dtypes
    buffer_dtype_map: Optional[Dict[tir.Var, Tuple[int, torch.dtype]]] = None
    # Maps buffer variables to their corresponding static shapes
    # {
    #     "A": [(0, 16), (1, 16)] -> represents A.shape = (16, 16)
    # }
    static_shape_map: Optional[Dict[tir.Var, Tuple[int, List[Tuple[int, int]]]]] = None
    # Maps buffer variables to their corresponding devices
    buffer_device_map: Optional[Dict[tir.Var, Tuple[int, torch.device]]] = None
    # Pass configs for the compiler
    pass_configs: Optional[Dict[str, Any]] = None

    def __init__(self,
                 params: List[KernelParam],
                 result_idx: List[int],
                 target: Union[str, Target],
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 tpu_target: Optional[TPUTargetSpec] = None,
                 tpu_runtime: Optional[TPURuntimeConfig] = None):
        """Initialize the adapter with the given TIR function or module.
        
        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Compilation target (for example, ``cuda``, ``hip``, ``c``, or ``tpu``)
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.ptr_map = self._process_ptr_map()
        self.static_shape_map = self._process_static_shape()
        self.buffer_device_map = self._process_buffer_device()

        self.verbose = verbose
        self.tpu_target = None
        self.tpu_runtime = None
        if is_tpu_target(self.target):
            if tpu_target is None or tpu_runtime is None:
                raise ValueError("TPU adapter requires target and runtime configuration "
                                 "from the compiled artifact")
            self.tpu_target = tpu_target
            self.tpu_runtime = tpu_runtime
        elif tpu_target is not None or tpu_runtime is not None:
            raise ValueError("TPU configuration can only be used with target='tpu'")
        self.lib_generator = LibraryGenerator(
            self.target, tpu_target=self.tpu_target, tpu_runtime=self.tpu_runtime)
        self.wrapper = TLWrapper(
            self.target, tpu_workspace_dir=self.lib_generator.tpu_workspace_dir)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapper.assign_output_indices(self.result_idx)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
        if is_tpu_target(self.target):
            # The TPU host ABI is not a normal Cython ``call`` wrapper.
            self.func = make_tpu_forward(self.lib, self.params, self.result_idx,
                                         self.dynamic_symbolic_map)
        else:
            self.lib.get_last_error.restype = ctypes.c_char_p
            result = self.lib.init()
            if result != 0:
                error_msg = self.lib.get_last_error().decode('utf-8')
                raise RuntimeError(f"Initialization failed: {error_msg}")

            wrapper_type = _load_cython_kernel_wrapper()
            self.cython_wrapper = wrapper_type(self.result_idx, self.params, self.lib)
            self.cython_wrapper.set_dynamic_symbolic_map(self.dynamic_symbolic_map)
            self.cython_wrapper.set_buffer_dtype_map(self.buffer_dtype_map)
            self.cython_wrapper.set_static_shape_map(self.static_shape_map)
            self.cython_wrapper.set_buffer_device_map(self.buffer_device_map)
            self.cython_wrapper.set_ptr_map(self.ptr_map)
            self._post_init()

    @classmethod
    def from_database(cls,
                      params: List[TensorType],
                      result_idx: List[int],
                      target: str,
                      func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                      kernel_global_source: str,
                      kernel_lib_path: str,
                      verbose: bool = False,
                      pass_configs: Optional[Dict[str, Any]] = None,
                      tpu_target: Optional[TPUTargetSpec] = None,
                      tpu_runtime: Optional[TPURuntimeConfig] = None):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.kernel_global_source = kernel_global_source
        adapter.wrapped_source = kernel_global_source

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        adapter.target = Target.canon_target(determine_target(target))

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
        adapter.buffer_dtype_map = adapter._process_buffer_dtype()
        adapter.static_shape_map = adapter._process_static_shape()
        adapter.ptr_map = adapter._process_ptr_map()
        adapter.buffer_device_map = adapter._process_buffer_device()

        adapter.verbose = verbose
        adapter.tpu_target = None
        adapter.tpu_runtime = None
        if is_tpu_target(adapter.target):
            if tpu_target is None or tpu_runtime is None:
                raise ValueError("TPU adapter requires target and runtime configuration "
                                 "from the compiled artifact")
            adapter.tpu_target = tpu_target
            adapter.tpu_runtime = tpu_runtime
        elif tpu_target is not None or tpu_runtime is not None:
            raise ValueError("TPU configuration can only be used with target='tpu'")
        reject_unverified_tpu_database_artifact(adapter.target)
        adapter.lib_generator = LibraryGenerator(
            adapter.target,
            tpu_target=adapter.tpu_target,
            tpu_runtime=adapter.tpu_runtime,
        )
        adapter.lib = adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        # TPU database artifacts are rejected above until they carry a verified
        # manifest, so only the ordinary native-library path reaches this point.
        adapter.lib.get_last_error.restype = ctypes.c_char_p
        result = adapter.lib.init()
        if result != 0:
            error_msg = adapter.lib.get_last_error().decode('utf-8')
            raise RuntimeError(f"Initialization failed: {error_msg}")

        wrapper_type = _load_cython_kernel_wrapper()
        adapter.cython_wrapper = wrapper_type(adapter.result_idx, adapter.params, adapter.lib)
        adapter.cython_wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        adapter.cython_wrapper.set_buffer_dtype_map(adapter.buffer_dtype_map)
        adapter.cython_wrapper.set_static_shape_map(adapter.static_shape_map)
        adapter.cython_wrapper.set_buffer_device_map(adapter.buffer_device_map)
        adapter.cython_wrapper.set_ptr_map(adapter.ptr_map)

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self) -> Dict[tir.Var, Tuple[int, int]]:
        """Extract information about dynamic shapes from the TIR function.
        
        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if (isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and
                        (shape not in params)):
                        dynamic_symbolic_map[shape] = (i, j)
        return dynamic_symbolic_map

    def _process_buffer_dtype(self) -> Dict[tir.Var, Tuple[int, torch.dtype]]:
        """Extract information about buffer dtypes from the TIR function.
        
        Maps buffer variables to their corresponding dtypes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_dtype_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name, dtype = buffer.name, buffer.dtype
                buffer_dtype_map[name] = (i, map_torch_type(dtype))
        return buffer_dtype_map

    def _process_ptr_map(self) -> Dict[int, str]:
        """Extract information about pointer arguments from the TIR function.
        
        Maps pointer arguments to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        ptr_map = {}
        for i, param in enumerate(params):
            if param.dtype == 'handle':
                ptr_map[i] = param.name
        return ptr_map

    def _process_static_shape(self) -> Dict[tir.Var, List[Tuple[int, int]]]:
        """Extract information about static shapes from the TIR function.
        
        Maps buffer variables to their corresponding static shapes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        static_shape_map = {}
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                shape = buffer.shape
                static_shape = []
                for j, s in enumerate(shape):
                    if isinstance(s, tir.IntImm):
                        static_shape.append((j, s.value))
                static_shape_map[name] = (i, static_shape)
        return static_shape_map

    def _process_buffer_device(self) -> Dict[tir.Var, Tuple[int, torch.device]]:
        """Extract information about buffer devices from the TIR function.
        
        Maps buffer variables to their corresponding devices.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        buffer_device_map = {}
        device = None
        if is_cuda_target(self.target) or is_hip_target(self.target):
            device = torch.device("cuda")
        elif is_cpu_target(self.target) or is_tpu_target(self.target):
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported target: {self.target}")

        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                name = buffer.name
                buffer_device_map[name] = (i, device)
        return buffer_device_map

    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        """Low-level function to call the compiled CUDA kernel.
        
        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""

        def lambda_forward(*args, stream: int = -1):
            return self.cython_wrapper.forward([*args], stream=stream)

        return lambda_forward

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        """Returns the source path of the compiled library."""
        return self.lib_generator.srcpath

    @property
    def libpath(self):
        """Returns the path to the compiled library."""
        return self.lib_generator.libpath

    @property
    def lib_code(self):
        """Returns the code of the compiled library."""
        return self.lib_generator.lib_code

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        return (self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0)

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.kernel_global_source
        else:
            assert self.wrapped_source is not None, "Wrapped source is not available"
            return self.wrapped_source
