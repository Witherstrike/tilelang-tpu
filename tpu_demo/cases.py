# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dependency-free registry for the public TPU demo validation matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DTYPES = ("float16", "bfloat16", "float32")
CHIP_CORE_COUNTS = {"bm1690": 8, "sg2260e": 4}
TARGET_CONFIGS = (
    ("bm1690", "tpukernel"),
    ("sg2260e", "tpukernel"),
    ("sg2260e", "rv"),
)
CHIPS = tuple(CHIP_CORE_COUNTS)
PROGRAMMING_MODELS = ("tpukernel", "rv")
RUNTIME_MODES = ("cmodel", "pcie")
OPERATIONS = (
    "elementwise-add",
    "elementwise-sub",
    "elementwise-mul",
    "elementwise-div",
    "matmul",
    "rmsnorm",
    "rmsnorm-splitk",
    "rope",
    "swiglu",
    "flashattn",
)
RV_SUPPORTED_OPERATIONS = frozenset(OPERATIONS)


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    operation: str
    dtype: str
    supports_rv: bool
    variant: str = "default"
    is_causal: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_cases() -> tuple[DemoCase, ...]:
    cases = []
    for operation in OPERATIONS:
        supports_rv = operation in RV_SUPPORTED_OPERATIONS
        for dtype in DTYPES:
            variants = (("balanced", "descending-max",
                         "weighted-keys") if operation == "flashattn" else ("default",))
            causal_modes = (False, True) if operation == "flashattn" else (False,)
            for is_causal in causal_modes:
                for variant in variants:
                    suffix = f".{variant}" if variant != "default" else ""
                    causal_suffix = ".causal" if is_causal else ""
                    cases.append(
                        DemoCase(
                            case_id=f"{operation}.{dtype}{suffix}{causal_suffix}",
                            operation=operation,
                            dtype=dtype,
                            supports_rv=supports_rv,
                            variant=variant,
                            is_causal=is_causal,
                        ))
    return tuple(cases)


def case_by_id(case_id: str) -> DemoCase:
    for case in build_cases():
        if case.case_id == case_id:
            return case
    raise ValueError(f"unknown TPU demo case: {case_id!r}")
