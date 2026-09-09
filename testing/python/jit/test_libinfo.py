# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
import sys

import pytest

from tilelang import libinfo


def _library_filename(name: str) -> str:
    if sys.platform.startswith(("linux", "freebsd")):
        return f"lib{name}.so"
    if sys.platform.startswith("win32"):
        return f"{name}.dll"
    if sys.platform.startswith("darwin"):
        return f"lib{name}.dylib"
    return f"lib{name}.so"


def test_explicit_library_path_takes_priority(monkeypatch, tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    filename = _library_filename("tilelang_module")
    (first / filename).touch()
    (second / filename).touch()
    override = f"{first}{os.pathsep}{second}"
    monkeypatch.setenv("TILELANG_LIBRARY_PATH", override)

    found = libinfo.find_lib_path("tilelang_module")

    assert found == [str(first / filename), str(second / filename)]


def test_invalid_explicit_library_path_does_not_fall_back(monkeypatch, tmp_path: Path):
    explicit = tmp_path / "explicit"
    fallback = tmp_path / "fallback"
    explicit.mkdir()
    fallback.mkdir()
    filename = _library_filename("tilelang_module")
    (fallback / filename).touch()
    monkeypatch.setenv("TILELANG_LIBRARY_PATH", str(explicit))
    monkeypatch.setattr(libinfo, "get_dll_directories", lambda: [str(fallback)])

    with pytest.raises(RuntimeError, match=str(explicit)):
        libinfo.find_lib_path("tilelang_module")
