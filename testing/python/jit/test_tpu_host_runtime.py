"""Host wrapper control flow tested without initializing a TPU runtime."""
import os
import importlib
from pathlib import Path
import shutil
import subprocess

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper


@pytest.fixture(scope="module")
def host_runner(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("g++ is required")
    root = tmp_path_factory.mktemp("tpu_host_runtime")
    # Generate into a private directory so tests cannot overwrite device artifacts.
    template_dir = Path(__file__).resolve().parents[3] / "src/tl_templates/tpu"
    shutil.copy(template_dir / "main_template.cpp", root)
    func = tvm.script.from_source('''
@T.prim_func
def main_kernel(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
    T.evaluate(0)
''', {"T": T})
    wrapper = object.__new__(TLTPUSourceWrapper)
    wrapper.mod = tvm.IRModule({"main_kernel": func})
    wrapper.output_indices = [1]
    wrapper.parse_func_args()
    module = importlib.import_module("tilelang.jit.adapter.wrapper")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "get_tpu_template_dir", lambda: str(root))
        wrapper.create_main_cpp()
    (root / "kernel.h").write_text("int main_kernel(unsigned long long, unsigned long long);\n")
    (root / "tpuv7_rt.h").write_text('''
#pragma once
using tpuRtStream_t = void*;
using tpuRtKernelModule_t = void*;
using tpuRtStatus_t = int;
constexpr int tpuRtSuccess = 0;
int tpuRtInit();
int tpuRtSetDevice(int);
int tpuRtStreamCreate(void**);
void* tpuRtKernelLoadModuleFile(const char*, void*);
int tpuRtKernelUnloadModule(void*, void*);
int tpuRtStreamDestroy(void*);
int tpuRtMalloc(void**, unsigned long long, int);
int tpuRtMemcpyS2D(void*, const void*, unsigned long long);
int tpuRtMemcpyD2S(void*, const void*, unsigned long long);
int tpuRtFree(void**, int);
''')
    (root / "fake.cpp").write_text('''
#include "tpuv7_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static int event(const char* name) {
  puts(name);
  const char* failure = getenv("FAIL_AT");
  return failure && !strcmp(failure, name) ? 7 : 0;
}
int tpuRtInit() { return event("init"); }
int tpuRtSetDevice(int d) { printf("device=%d\\n", d); return event("set_device"); }
int tpuRtStreamCreate(void** s) { int r=event("stream"); if (!r) *s=(void*)1; return r; }
void* tpuRtKernelLoadModuleFile(const char*, void*) { return event("load") ? nullptr : (void*)2; }
int tpuRtKernelUnloadModule(void*, void*) { return event("unload"); }
int tpuRtStreamDestroy(void*) { return event("destroy"); }
int tpuRtMalloc(void** p, unsigned long long, int) { int r=event("malloc"); if (!r) *p=(void*)3; return r; }
int tpuRtMemcpyS2D(void*, const void*, unsigned long long) { return event("h2d"); }
int tpuRtMemcpyD2S(void*, const void*, unsigned long long) { return event("d2h"); }
int tpuRtFree(void** p, int) { *p=nullptr; return event("free"); }
int main_kernel(unsigned long long, unsigned long long) { return event("launch"); }
extern "C" int tilelang_tpu_run(void**);
int main() { float a[4]={}; void* args[]={a,a}; return tilelang_tpu_run(args); }
''')
    exe = root / "runner"
    subprocess.run(["g++", "-std=c++17", f"-I{root}", str(root / "main.cpp"),
                    str(root / "fake.cpp"), "-o", str(exe)], check=True, capture_output=True)
    return exe


@pytest.mark.parametrize("failure,forbidden", [
    ("init", "set_device"), ("set_device", "stream"), ("stream", "load"),
    ("load", "malloc"), ("malloc", "h2d"), ("h2d", "launch"),
    ("launch", "d2h"), ("d2h", None), ("free", None), ("unload", None),
    ("destroy", None),
])
def test_runtime_failure_stops_work(host_runner, failure, forbidden):
    env = dict(os.environ, PPL_KERNEL_PATH="unused", TILELANG_TPU_DEVICE_ID="0", FAIL_AT=failure)
    result = subprocess.run([host_runner], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    events = result.stdout.splitlines()
    assert failure in events
    if forbidden:
        assert forbidden not in events
    if failure in ("load", "malloc", "h2d", "launch", "d2h", "free", "unload"):
        assert "destroy" in events


def test_success_launches_once(host_runner):
    env = dict(os.environ, PPL_KERNEL_PATH="unused", TILELANG_TPU_DEVICE_ID="3", FAIL_AT="")
    result = subprocess.run([host_runner], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["init", "device=3", "set_device", "stream", "load",
                                         "malloc", "malloc", "h2d", "h2d", "launch", "d2h",
                                         "free", "free", "unload", "destroy"]


@pytest.mark.parametrize("device", ["", "-1", "1x", "2147483648"])
def test_invalid_device_rejected_before_init(host_runner, device):
    env = dict(os.environ, PPL_KERNEL_PATH="unused", TILELANG_TPU_DEVICE_ID=device, FAIL_AT="")
    result = subprocess.run([host_runner], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not result.stdout
