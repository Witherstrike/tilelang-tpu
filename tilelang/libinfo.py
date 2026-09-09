# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Library information. This is a standalone file that can be used to get various info. 
Modified from: https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/libinfo.py
"""

#! pylint: disable=protected-access
import os
import sys


def get_env_paths(env_var, splitter):
    """Get path in env variable"""
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(splitter)]
    return []


def get_dll_directories():
    """Get extra tile lang dll directories"""
    curr_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.abspath(os.path.join(curr_dir, ".."))
    # An explicit library path is an override, so list it before source-tree
    # and package defaults.  ``find_lib_path`` also limits its search to these
    # entries when the override is present; this prevents two native builds
    # from being mixed silently.
    dll_path = get_env_paths("TILELANG_LIBRARY_PATH", os.pathsep)
    dll_path += [
        curr_dir,
        os.path.join(source_dir, "build-tpu"),  # TPU-only local build
        os.path.join(source_dir, "build-tpu", "Release"),
        os.path.join(source_dir, "build"),  # local build
        os.path.join(source_dir, "build", "Release"),
        os.path.join(curr_dir, "lib"),  # pypi build
    ]
    if "CONDA_PREFIX" in os.environ:
        dll_path.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(get_env_paths("LD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(get_env_paths("DYLD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(get_env_paths("PATH", ";"))
    return [os.path.abspath(p) for p in dll_path if os.path.isdir(p)]


def find_lib_path(name, optional=False):
    """Find tile lang library

    Parameters
    ----------
    name : str
        The name of the library

    optional: boolean
        Whether the library is required
    """
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("win32"):
        lib_name = f"{name}.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    else:
        lib_name = f"lib{name}.so"

    explicit_library_paths = [
        path for path in get_env_paths("TILELANG_LIBRARY_PATH", os.pathsep) if path
    ]
    if explicit_library_paths:
        # Treat an explicit override as authoritative. Falling back to a local
        # build when it is misspelled can pair TileLang with an unrelated TVM
        # library and makes a validated build selection non-reproducible.
        dll_paths = [os.path.abspath(path) for path in explicit_library_paths]
    else:
        dll_paths = get_dll_directories()
    lib_dll_path = [os.path.join(p, lib_name) for p in dll_paths]
    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_found and not optional:
        message = (f"Cannot find libraries: {lib_name}\n" + "List of candidates:\n" +
                   "\n".join(lib_dll_path))
        raise RuntimeError(message)
    return lib_found
