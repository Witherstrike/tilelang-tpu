# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Keep TPU compiler registries and the machine contract closed and aligned."""

import json
import re
import subprocess
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SEMANTIC_EXTERN = re.compile(
    r'"(tl\.(?:tpu|tpukernel)\.[a-z0-9_]+)"')

_PORTABLE_EXTERNS = {
    "tl.tpu.add",
    "tl.tpu.copy",
    "tl.tpu.div",
    "tl.tpu.fill",
    "tl.tpu.gemm",
    "tl.tpu.mul",
    "tl.tpu.sub",
}
_TPUKERNEL_EXTERNS = {
    "tl.tpukernel.add_scalar",
    "tl.tpukernel.exp",
    "tl.tpukernel.gather",
    "tl.tpukernel.mul_scalar",
    "tl.tpukernel.reduce_max",
    "tl.tpukernel.reduce_sum",
    "tl.tpukernel.rope_add",
    "tl.tpukernel.rsqrt",
    "tl.tpukernel.sigmoid",
    "tl.tpukernel.topk",
}


def _semantic_externs(relative_path):
    source = (_REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    return set(_SEMANTIC_EXTERN.findall(source))


def _machine_contract():
    path = _REPOSITORY_ROOT / "research/tpu-op-contract/contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_semantic_extern_registry_is_isomorphic_across_compiler_layers():
    """An extern is valid only when every compiler boundary owns it."""
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS

    assert _semantic_externs("tilelang/language/customize.py") == expected
    assert _semantic_externs("tilelang/engine/lower.py") == expected
    assert _semantic_externs("src/transform/address_assign.cc") == expected
    assert _semantic_externs(
        "src/target/codegen_tpu_common.cc") == _PORTABLE_EXTERNS
    assert _semantic_externs(
        "src/target/codegen_tpukernel.cc") == _TPUKERNEL_EXTERNS

    contract_externs = {
        symbol
        for operation in _machine_contract()["operations"]
        for symbol in operation["internal_symbols"]
    }
    assert contract_externs == expected


def test_machine_contract_standard_library_validator():
    """Run portable schema-adjacent and cross-reference validation in CI."""
    validator = _REPOSITORY_ROOT / "research/tpu-op-contract/validate.py"
    result = subprocess.run(
        [sys.executable, str(validator)],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "contract validation passed" in result.stdout
