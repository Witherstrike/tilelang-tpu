"""Public T.ppl API compile gates and opt-in device numerics.

Each device case executes in a fresh process; never initialize the PPL emulator
twice in one process. See docs/sg2260e_rv_all_apis_handoff.md.
"""
import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.engine.tpu_config import TPUCompileConfig
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout
from tilelang.jit.adapter.wrapper import TLTPUSourceWrapper


OPS = ["add", "subtract", "mul", "div", "add_C", "mul_C", "fill", "clear",
       "exp2", "sigmoid", "rsqrt", "reduce_sum",
       "reduce_max", "rope_add", "copy", "gemm",
       "gather", "topk"]
CASES = OPS + ["topk_ascending", "topk_nan", "topk_int32", "topk_uint32",
               "reduce_sum_wide", "reduce_max_accumulate", "scalar_chain",
               "gather_repeat", "copy_global", "copy_cast", "exp2_extremes"]


def test_every_exported_ppl_api_has_a_case():
    exported = {name[4:] for name in dir(T) if name.startswith("ppl_")}
    assert exported == set(OPS)


def case(op):
    if op.startswith("topk_"):
        source, _, _, indices = case("topk")
        dtype = op.removeprefix("topk_")
        a = np.array([3, -1, 3, 7, 0, 7, -4, 2], dtype="float32")
        descending = op != "topk_ascending"
        if op == "topk_nan":
            a = np.array([np.nan, -np.inf, np.nan, 7, np.inf, 7, -0., 0.], dtype="float32")
            source = source.replace('(4,)', '(8,)').replace(', 4, True, 8)', ', 8, True, 8)')
        elif dtype in ("int32", "uint32"):
            limits = np.iinfo(dtype)
            a = np.array([limits.max, limits.min, 3, 7, 0, 7, limits.max, 2], dtype=dtype)
            source = source.replace('"float32"', f'"{dtype}"')
        if not descending:
            source = source.replace('True', 'False')
        # Python integer conversion avoids unsigned negation wraparound.
        order = sorted(range(8), key=lambda i: (
            bool(np.isnan(a[i])), 0 if np.isnan(a[i]) else (-float(a[i]) if descending else float(a[i])), i))
        order = np.asarray(order[:8 if op == "topk_nan" else 4], dtype="int32")
        return source, [a], [a[order], order], indices
    if op == "reduce_sum_wide":
        source, _, _, indices = case("reduce_sum")
        source = source.replace('(8, 16)', '(65, 33)').replace('(8, 1)', '(65, 1)')
        a = np.random.default_rng(2260).uniform(-1, 1, (65, 33)).astype("float32")
        return source, [a, a.copy()], [a.sum(axis=1, keepdims=True)], indices
    if op == "reduce_max_accumulate":
        source, inputs, expected, indices = case("reduce_max")
        source = source.replace('T.ppl_reduce_max(X, Z, 1)', 'T.ppl_fill(Z, 1.75)\n    T.ppl_reduce_max(X, Z, 1, clear=False)')
        return source, inputs, [np.maximum(expected[0], 1.75)], indices
    if op == "scalar_chain":
        source, inputs, _, indices = case("add_C")
        source = source.replace('T.ppl_add_C(Z, X, 1.25)', 'T.ppl_add_C(Z, X, 1.25)\n    T.ppl_mul_C(Z, Z, 1.25)\n    T.ppl_add_C(Z, Z, 1.25)')
        return source, inputs, [(inputs[0] + 1.25) * 1.25 + 1.25], indices
    if op == "gather_repeat":
        source, inputs, expected, indices = case("gather")
        source += '    T.ppl_gather(C, A, I, 8)\n'
        return source, inputs, expected, indices
    if op == "copy_global":
        source, inputs, expected, indices = case("copy")
        source = source[:source.index('    X =')] + '    T.ppl_copy(A, C)\n'
        return source, inputs, expected, indices
    if op == "copy_cast":
        source, inputs, expected, indices = case("copy")
        source = source.replace('Z = T.alloc_shared((8, 16), "float32")', 'Z = T.alloc_shared((8, 16), "float16")')
        source = source.replace('C: T.Buffer((8, 16), "float32")', 'C: T.Buffer((8, 16), "float16")')
        return source, inputs, [expected[0].astype("float16")], indices
    if op == "exp2_extremes":
        source, inputs, _, indices = case("exp2")
        inputs[0][:] = np.resize(np.array([-np.inf, -104, -90, -1, 0, 1, 88, 89, np.inf, np.nan], dtype="float32"), inputs[0].shape)
        with np.errstate(over="ignore"):
            expected = np.exp(inputs[0])
        return source, inputs, [expected], indices
    rng = np.random.default_rng(2260)
    a = rng.uniform(0.25, 2, (8, 16)).astype("float32")
    b = rng.uniform(0.25, 2, (8, 16)).astype("float32")
    prefix = '''
@T.prim_func
def main_kernel_inner(A: T.Buffer((8, 16), "float32"), B: T.Buffer((8, 16), "float32"), C: T.Buffer(OUT_SHAPE, "float32")):
    X = T.alloc_shared((8, 16), "float32")
    Y = T.alloc_shared((8, 16), "float32")
    Z = T.alloc_shared(OUT_SHAPE, "float32")
    T.ppl_copy(A, X)
    T.ppl_copy(B, Y)
'''
    body = ""
    shape = (8, 16)
    inputs = [a, b]
    if op in ("add", "subtract", "mul", "div"):
        body = f"    T.ppl_{op}(Z, X, Y)\n"
        expected = {"add": np.add, "subtract": np.subtract,
                    "mul": np.multiply, "div": np.divide}[op](a, b)
    elif op in ("add_C", "mul_C"):
        body = f"    T.ppl_{op}(Z, X, 1.25)\n"
        expected = a + 1.25 if op == "add_C" else a * 1.25
    elif op in ("fill", "clear"):
        body = "    T.ppl_fill(Z, 1.25)\n" if op == "fill" else "    T.ppl_clear(Z)\n"
        expected = np.full_like(a, 1.25 if op == "fill" else 0)
    elif op == "copy":
        body = "    T.ppl_copy(X, Z)\n"
        expected = a
    elif op == "rsqrt":
        body = "    T.ppl_rsqrt(Z, X)\n"
        expected = 1 / np.sqrt(a)
    elif op in ("exp2", "sigmoid"):
        a[:] = rng.uniform(-12, 12, a.shape)
        body = '''    W0 = T.alloc_shared((8, 16), "float32")
    W1 = T.alloc_shared((8, 16), "float32")
    Coeff = T.alloc_shared((64, 32), "float32")
    Table = T.alloc_shared((64, 192), "float32")
'''
        if op == "exp2":
            body += "    T.ppl_exp2(X, W0, W1, Coeff, Table)\n    T.ppl_copy(X, Z)\n"
            expected = np.exp(a)
        else:
            body += "    T.ppl_sigmoid(Z, X, W0, W1, Coeff, Table)\n"
            expected = 1 / (1 + np.exp(-a))
    elif op.startswith("reduce_"):
        shape = (8, 1)
        body = f"    T.ppl_{op}(X, Z, 1)\n"
        expected = (a.sum(axis=1, keepdims=True) if "sum" in op else a.max(axis=1, keepdims=True))
    elif op == "rope_add":
        body = "    T.ppl_rope_add(Z, X, Y, X, Y)\n"
        expected = np.empty_like(a)
        expected[:, ::2] = a[:, ::2] + b[:, 1::2]
        expected[:, 1::2] = a[:, 1::2] + b[:, ::2]
    elif op == "gemm":
        a, b = a.astype("float16"), b.T.copy().astype("float16")
        inputs = [a, b]
        prefix = prefix.replace('B: T.Buffer((8, 16), "float32")', 'B: T.Buffer((16, 8), "float16")')
        prefix = prefix.replace('A: T.Buffer((8, 16), "float32")', 'A: T.Buffer((8, 16), "float16")')
        prefix = prefix.replace('X = T.alloc_shared((8, 16), "float32")', 'X = T.alloc_shared((8, 16), "float16")')
        prefix = prefix.replace('Y = T.alloc_shared((8, 16), "float32")', 'Y = T.alloc_shared((16, 8), "float16")')
        shape = (8, 8)
        body = "    T.ppl_clear(Z)\n    T.ppl_gemm(X, Y, Z)\n    T.ppl_gemm(X, Y, Z)\n"
        expected = 2 * (a.astype("float32") @ b.astype("float32"))
    elif op == "gather":
        a = a.astype("float16")
        indices = np.array([7, 1, 1, 0], dtype="uint32")
        return '''
@T.prim_func
def main_kernel_inner(A: T.Buffer((8, 16), "float16"), I: T.Buffer((4,), "uint32"), C: T.Buffer((4, 16), "float16")):
    T.ppl_gather(C, A, I, 8)
''', [a, indices], [a[indices]], [2]
    elif op == "topk":
        a = np.array([3, -1, 3, 7, 0, 7, -4, 2], dtype="float32")
        order = np.argsort(-a, kind="stable")[:4].astype("int32")
        return '''
@T.prim_func
def main_kernel_inner(A: T.Buffer((8,), "float32"), C: T.Buffer((4,), "float32"), I: T.Buffer((4,), "int32")):
    T.ppl_topk(C, I, A, 4, True, 8)
''', [a], [a[order], order], [1, 2]
    else:
        raise ValueError(op)
    source = prefix.replace("OUT_SHAPE", repr(shape)) + body + "    T.ppl_copy(Z, C)\n"
    return source, inputs, [expected], [2]


def lower_case(op, pipeline=False):
    source, inputs, outputs, indices = case(op)
    func = tvm.script.from_source(source, {"T": T})
    original = tvm.IRModule({"main_kernel_inner": func})
    mod = tvm.tir.transform.LowerOpaqueBlock()(original)
    func = mod["main_kernel_inner"].with_attr("tir.tpu.chip", "sg2260e").with_attr("tir.tpu.device_mode", "rv")
    mod = tilelang.transform.AddressAssign()(tvm.IRModule({"main_kernel_inner": func}))
    mod = tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")(mod)
    if pipeline:
        func = mod["main_kernel_inner"]
        marker = tvm.tir.IntImm("int32", 0)
        start = tvm.tir.AttrStmt(marker, "tpu_parallel_start", 0, tvm.tir.Evaluate(0))
        end = tvm.tir.AttrStmt(marker, "tpu_parallel_end", 0, tvm.tir.Evaluate(0))
        mod = tvm.IRModule({"main_kernel_inner": func.with_body(tvm.tir.SeqStmt([start, func.body, end]))})
    code = tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)
    return original, code, inputs, outputs, indices


@pytest.mark.parametrize("op,old,new,diagnostic", [
    ("topk", ", 4, True, 8)", ", 9, True, 8)", "length >= k"),
    ("rope_add", "T.ppl_rope_add(Z,", "T.ppl_rope_add(X,", "requires separate output"),
    ("exp2", "T.ppl_exp2(X, W0, W1,", "T.ppl_exp2(X, W0, W0,", "work0.buffer_name"),
])
def test_invalid_public_composites_rejected(op, old, new, diagnostic):
    source = case(op)[0].replace(old, new)
    func = tvm.script.from_source(source, {"T": T})
    mod = tvm.tir.transform.LowerOpaqueBlock()(tvm.IRModule({"main_kernel_inner": func}))
    func = mod["main_kernel_inner"].with_attr("tir.tpu.chip", "sg2260e").with_attr("tir.tpu.device_mode", "rv")
    mod = tilelang.transform.AddressAssign()(tvm.IRModule({"main_kernel_inner": func}))
    mod = tilelang.transform.RVLegalizeAndAllocateRegisters("sg2260e")(mod)
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        tvm.get_global_func("target.build.tilelang_ppl_rv")(mod)


def test_topk_allocator_reports_scratch():
    func = tvm.script.from_source(case("topk")[0], {"T": T})
    mod = tvm.tir.transform.LowerOpaqueBlock()(tvm.IRModule({"main_kernel_inner": func}))
    func = mod["main_kernel_inner"].with_attr("tir.tpu.chip", "sg2260e").with_attr("tir.tpu.device_mode", "rv")
    mod = tilelang.transform.AddressAssign()(tvm.IRModule({"main_kernel_inner": func}))
    attrs = mod["main_kernel_inner"].attrs
    assert int(attrs["tir.tpu.lmem.__rv_topk_scratch.size"]) == 7 * 64
    assert int(attrs["tir.tpu.lmem.__rv_topk_scratch.address"]) == int(attrs["tir.tpu.rv.topk_scratch"])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("runtime", ["cmodel", "pcie"])
def test_public_engine_lower(op, runtime):
    from tilelang.engine.lower import lower
    source, _, _, _ = case(op)
    func = tvm.script.from_source(source, {"T": T})
    artifact = lower(func, target="tpu", chip="sg2260e", device_mode="rv", runtime_mode=runtime)
    assert "rvt_" in artifact.kernel_source
    assert "ppl.rv." not in artifact.kernel_source
    assert artifact.tpu_config.runtime_mode == runtime


@pytest.mark.parametrize("op", CASES)
@pytest.mark.parametrize("pipeline", [False, True], ids=["serial", "pipeline"])
def test_public_api_compiles(op, pipeline, tmp_path):
    _, code, _, _, _ = lower_case(op, pipeline)
    assert "ppl.rv." not in code
    assert "tpu_bdc_" not in code and "tpu_gdma_" not in code and "tpu_hau_" not in code
    root = os.environ.get("PPL_PROJECT_ROOT")
    if not root:
        root = str(Path(__file__).resolve().parents[4] / "ppl_v1.7.122-g05ebfb36-20260528")
    if not Path(root).is_dir():
        pytest.skip("PPL SDK is required for generated C syntax checking")
    layout = resolve_ppl_layout(root, "sg2260e")
    path = tmp_path / f"{op}.c"
    path.write_text(code)
    subprocess.run(["cc", "-std=c11", "-fsyntax-only",
                    *(f"-D{x}" for x in layout.compile_definitions),
                    *(f"-I{x}" for x in layout.include_dirs), str(path)],
                   check=True, capture_output=True, text=True)


def run_device(op, runtime, pipeline=False, compile_only=False):
    original, code, inputs, expected, indices = lower_case(op, pipeline)
    target = SimpleNamespace(kind=SimpleNamespace(name="tpu"))
    TLTPUSourceWrapper(original, code, target, output_indices=indices)
    generator = LibraryGenerator(target, tpu_config=TPUCompileConfig("sg2260e", "rv", runtime))
    generator.update_lib_code(code)
    generator.compile_lib(timeout=120)
    if compile_only:
        print(f"COMPILED {runtime} {op} pipeline={pipeline}: {generator.libpath}", flush=True)
        return
    library = ctypes.CDLL(generator.libpath)
    library.tilelang_tpu_run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.tilelang_tpu_run.restype = ctypes.c_int
    actual = [np.full_like(x, 42) for x in expected]
    arrays = inputs + actual
    args = (ctypes.c_void_p * len(arrays))(*(x.ctypes.data for x in arrays))
    assert library.tilelang_tpu_run(args) == 0
    for got, want in zip(actual, expected):
        if np.issubdtype(want.dtype, np.integer):
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    print(f"PASS {runtime} {op} pipeline={pipeline}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES + ["all"])
    parser.add_argument("--runtime", required=True, choices=["cmodel", "pcie"])
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--compile-only", action="store_true", help="Build and link without loading or launching the runtime")
    parser.add_argument("--timeout", type=int, default=300, help="Per-case timeout for --case all")
    args = parser.parse_args()
    if args.case == "all":
        failed = []
        for op in CASES:
            command = [sys.executable, __file__, "--case", op, "--runtime", args.runtime]
            command += [flag for flag, enabled in [("--pipeline", args.pipeline), ("--compile-only", args.compile_only)] if enabled]
            try:
                result = subprocess.run(command, timeout=args.timeout)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
            if returncode:
                failed.append(op)
        if failed:
            raise SystemExit("Failed cases: " + ", ".join(failed))
    else:
        run_device(args.case, args.runtime, args.pipeline, args.compile_only)
