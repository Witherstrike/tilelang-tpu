# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from pathlib import Path
import subprocess

import pytest

from tilelang import tvm
from tilelang.jit.adapter import libgen
from tilelang.jit.adapter.libgen import LibraryGenerator


def _cpu_generator() -> LibraryGenerator:
    generator = LibraryGenerator(tvm.target.Target("c"))
    generator.update_lib_code("extern \"C\" int generated() { return 0; }")
    return generator


def _place_temporary_source(monkeypatch, tmp_path: Path) -> Path:
    source_path = tmp_path / "generated.cpp"

    def new_temporary_source(suffix: str) -> str:
        assert suffix == ".cpp"
        source_path.touch()
        return str(source_path)

    monkeypatch.setattr(libgen, "_new_temporary_source", new_temporary_source)
    return source_path


def test_non_tpu_close_removes_owned_artifacts_without_removing_caller_paths(monkeypatch, tmp_path):
    source_path = _place_temporary_source(monkeypatch, tmp_path)
    library_path = source_path.with_suffix(".so")

    def compile_success(command, *, timeout):
        assert timeout is None
        Path(command[command.index("-o") + 1]).write_bytes(b"library")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(libgen.subprocess, "run", compile_success)
    generator = _cpu_generator()
    generator.compile_lib()
    user_source = tmp_path / "user.cpp"
    user_source.write_text("// caller-owned\n", encoding="utf-8")
    user_library = tmp_path / "user.so"
    user_library.write_bytes(b"caller-owned")
    generator.set_src_path(str(user_source))
    generator.set_lib_path(str(user_library))

    generator.close()
    generator.close()

    assert not source_path.exists()
    assert not library_path.exists()
    assert user_source.read_text(encoding="utf-8") == "// caller-owned\n"
    assert user_library.read_bytes() == b"caller-owned"
    assert generator.get_source_path() == str(user_source)
    assert generator.get_lib_path() == str(user_library)


def test_non_tpu_recompile_removes_previous_owned_artifacts(monkeypatch, tmp_path):
    source_paths = [tmp_path / "first.cpp", tmp_path / "second.cpp"]

    def new_temporary_source(suffix: str) -> str:
        assert suffix == ".cpp"
        source_path = source_paths.pop(0)
        source_path.touch()
        return str(source_path)

    def compile_success(command, *, timeout):
        Path(command[command.index("-o") + 1]).write_bytes(b"library")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(libgen, "_new_temporary_source", new_temporary_source)
    monkeypatch.setattr(libgen.subprocess, "run", compile_success)
    generator = _cpu_generator()
    generator.compile_lib()
    first_source = Path(generator.get_source_path())
    first_library = Path(generator.get_lib_path())

    generator.compile_lib()

    assert not first_source.exists()
    assert not first_library.exists()
    assert Path(generator.get_source_path()).exists()
    assert Path(generator.get_lib_path()).exists()
    generator.close()


def test_non_tpu_compile_exception_removes_partial_outputs(monkeypatch, tmp_path):
    source_path = _place_temporary_source(monkeypatch, tmp_path)
    library_path = source_path.with_suffix(".so")

    def compile_failure(command, *, timeout):
        library_path.write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(libgen.subprocess, "run", compile_failure)
    generator = _cpu_generator()

    with pytest.raises(RuntimeError, match="Compile kernel failed"):
        generator.compile_lib(timeout=1)

    assert not source_path.exists()
    assert not library_path.exists()
    assert generator.get_source_path() is None
    assert generator.get_lib_path() is None


def test_non_tpu_compile_error_removes_partial_outputs(monkeypatch, tmp_path):
    source_path = _place_temporary_source(monkeypatch, tmp_path)
    library_path = source_path.with_suffix(".so")

    def compile_failure(command, *, timeout):
        assert timeout is None
        library_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(libgen.subprocess, "run", compile_failure)
    generator = _cpu_generator()

    with pytest.raises(RuntimeError, match="Compilation Failed"):
        generator.compile_lib()

    assert not source_path.exists()
    assert not library_path.exists()
    assert generator.get_source_path() is None
    assert generator.get_lib_path() is None


def test_non_tpu_keyboard_interrupt_removes_partial_outputs(monkeypatch, tmp_path):
    source_path = _place_temporary_source(monkeypatch, tmp_path)
    library_path = source_path.with_suffix(".so")

    def compile_interrupted(command, *, timeout):
        assert timeout is None
        library_path.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(libgen.subprocess, "run", compile_interrupted)
    generator = _cpu_generator()

    with pytest.raises(KeyboardInterrupt):
        generator.compile_lib()

    assert not source_path.exists()
    assert not library_path.exists()
    assert generator.get_source_path() is None
    assert generator.get_lib_path() is None
