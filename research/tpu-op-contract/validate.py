#!/usr/bin/env python3
"""Validate the TileLang TPU operation contract with the Python standard library.

This checker deliberately does not require ``jsonschema``.  It verifies the
cross-reference and implementation-set invariants that JSON Schema cannot
express, while ignored runtime artifacts remain optional in another clone.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONTRACT_PATH = HERE / "contract.json"
SCHEMA_PATH = HERE / "schema.json"
ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
SEMANTIC_EXTERN = re.compile(r"tl\.(?:tpu|tpukernel)\.[A-Za-z0-9_]+")
REQUIRED_STAGES = {
    "declared",
    "codegen_passed",
    "cmodel_numeric_passed",
    "pcie_numeric_passed",
}
STAGE_STATUSES = {"passed", "failed", "unverified", "not_applicable"}


class ContractError(ValueError):
    """A deterministic contract validation failure."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path.relative_to(REPO_ROOT)}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path.relative_to(REPO_ROOT)} must contain one object")
    return value


def _unique_index(kind: str, rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ContractError(f"{kind} has a missing or non-string id")
        if identifier in result:
            raise ContractError(f"duplicate {kind} id: {identifier}")
        result[identifier] = row
    return result


def _check_portable_path(label: str, value: Any, *, symbolic: bool = False) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    if ABSOLUTE_PATH.match(value):
        raise ContractError(f"{label} is machine-absolute: {value}")
    if symbolic and value.startswith("${") and value.endswith("}"):
        return
    if ".." in PurePosixPath(value).parts:
        raise ContractError(f"{label} escapes its declared source root: {value}")


def _evidence_refs(owner: str, ids: Any, evidence: dict[str, dict[str, Any]]) -> None:
    if not isinstance(ids, list):
        raise ContractError(f"{owner}.evidence_ids must be an array")
    for evidence_id in ids:
        if evidence_id not in evidence:
            raise ContractError(f"{owner} references unknown evidence {evidence_id!r}")


def _extract_python_frozenset(path: Path, assignment: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == assignment
                   for target in node.targets):
            continue
        value = node.value
        if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
                or value.func.id != "frozenset" or len(value.args) != 1
                or not isinstance(value.args[0], ast.Set)):
            raise ContractError(f"{assignment} must remain a literal frozenset")
        result: set[str] = set()
        for element in value.args[0].elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                raise ContractError(f"{assignment} contains a non-literal member")
            result.add(element.value)
        return result
    raise ContractError(f"cannot find {assignment} in {path.relative_to(REPO_ROOT)}")


def _semantic_externs(path: Path) -> set[str]:
    return set(SEMANTIC_EXTERN.findall(path.read_text(encoding="utf-8")))


def _check_exact_set(label: str, expected: set[str], actual: set[str]) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ContractError(f"{label} mismatch; missing={missing}, extra={extra}")


def _validate_implementation_sets(operations: dict[str, dict[str, Any]]) -> None:
    internal_symbols = [
        symbol
        for operation in operations.values()
        for symbol in operation.get("internal_symbols", [])
    ]
    if len(internal_symbols) != len(set(internal_symbols)):
        raise ContractError("operation internal_symbols are not globally unique")
    contract_set = set(internal_symbols)

    lower_path = REPO_ROOT / "tilelang/engine/lower.py"
    lower_set = (
        _extract_python_frozenset(lower_path, "_PORTABLE_TPU_EXTERNS")
        | _extract_python_frozenset(lower_path, "_TPUKERNEL_EXTERNS")
    )
    _check_exact_set("contract vs lower.py closed extern set", contract_set, lower_set)

    frontend_set = _semantic_externs(REPO_ROOT / "tilelang/language/customize.py")
    _check_exact_set("contract vs frontend emitted extern set", contract_set, frontend_set)

    effects_set = _semantic_externs(REPO_ROOT / "src/transform/address_assign.cc")
    _check_exact_set("contract vs AddressAssign effect set", contract_set, effects_set)

    codegen_set = (
        _semantic_externs(REPO_ROOT / "src/target/codegen_tpu_common.cc")
        | _semantic_externs(REPO_ROOT / "src/target/codegen_tpukernel.cc")
    )
    _check_exact_set("contract vs TPU codegen dispatch set", contract_set, codegen_set)

    frontend_path = REPO_ROOT / "tilelang/language/customize.py"
    frontend_defs = {
        node.name
        for node in ast.parse(frontend_path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for operation in operations.values():
        for symbol in operation.get("frontend_symbols", []):
            function_name = symbol.removeprefix("T.")
            if function_name not in frontend_defs:
                raise ContractError(
                    f"{operation['id']} frontend symbol {symbol!r} has no definition in customize.py")


def validate() -> tuple[int, int, int, int]:
    # Parsing schema.json here catches syntax and duplicate-key errors even when
    # a JSON Schema implementation is intentionally unavailable.
    _load_json(SCHEMA_PATH)
    contract = _load_json(CONTRACT_PATH)

    source_roots = contract.get("source_roots")
    if not isinstance(source_roots, dict) or not source_roots:
        raise ContractError("source_roots must be a non-empty object")
    for root_id, root in source_roots.items():
        if not isinstance(root, dict):
            raise ContractError(f"source root {root_id!r} must be an object")
        _check_portable_path(f"source_roots.{root_id}.path", root.get("path"), symbolic=True)

    targets = _unique_index("target", contract.get("targets", []))
    operations = _unique_index("operation", contract.get("operations", []))
    evidence = _unique_index("evidence", contract.get("evidence", []))
    capabilities = _unique_index("capability", contract.get("capabilities", []))

    for target_id, target in targets.items():
        _evidence_refs(f"target {target_id}", target.get("evidence_ids", []), evidence)

    for evidence_id, item in evidence.items():
        root_id = item.get("source_root")
        if root_id not in source_roots:
            raise ContractError(f"evidence {evidence_id} uses unknown source root {root_id!r}")
        _check_portable_path(f"evidence {evidence_id}.path", item.get("path"))
        root = source_roots[root_id]
        if root.get("tracked") and item.get("kind") in {"source", "test_source"}:
            path = REPO_ROOT / root["path"] / item["path"]
            if not path.is_file():
                raise ContractError(f"tracked evidence is missing: {path.relative_to(REPO_ROOT)}")
            for token in (part.strip() for part in item["locator"].split(",")):
                if token and token not in path.read_text(encoding="utf-8"):
                    raise ContractError(
                        f"evidence {evidence_id} locator token {token!r} is absent from "
                        f"{path.relative_to(REPO_ROOT)}")

    for operation_id, operation in operations.items():
        _evidence_refs(f"operation {operation_id}", operation.get("evidence_ids", []), evidence)
        constraints = _unique_index(
            f"constraint in operation {operation_id}", operation.get("constraints", []))
        for constraint_id, constraint in constraints.items():
            _evidence_refs(
                f"constraint {operation_id}/{constraint_id}",
                constraint.get("evidence_ids", []),
                evidence,
            )

    for capability_id, capability in capabilities.items():
        operation_id = capability.get("operation_id")
        if operation_id not in operations:
            raise ContractError(
                f"capability {capability_id} references unknown operation {operation_id!r}")
        operation_constraints = {
            constraint["id"] for constraint in operations[operation_id].get("constraints", [])
        }
        for constraint_id in capability.get("constraints", []):
            if constraint_id not in operation_constraints:
                raise ContractError(
                    f"capability {capability_id} references unknown constraint {constraint_id!r}")

        results = capability.get("target_results")
        if not isinstance(results, dict) or not results:
            raise ContractError(f"capability {capability_id} has no target_results")
        for target_id, result in results.items():
            if target_id not in targets:
                raise ContractError(
                    f"capability {capability_id} references unknown target {target_id!r}")
            if not targets[target_id].get("applicable"):
                raise ContractError(
                    f"capability {capability_id} references inapplicable target {target_id}")
            verification = result.get("verification")
            if not isinstance(verification, dict):
                raise ContractError(f"{capability_id}/{target_id} has no verification object")
            missing_stages = REQUIRED_STAGES - verification.keys()
            if missing_stages:
                raise ContractError(
                    f"{capability_id}/{target_id} lacks stages {sorted(missing_stages)}")
            for stage_name, stage in verification.items():
                status = stage.get("status")
                if status not in STAGE_STATUSES:
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} has invalid status {status!r}")
                ids = stage.get("evidence_ids", [])
                _evidence_refs(f"{capability_id}/{target_id}/{stage_name}", ids, evidence)
                if status in {"passed", "failed"} and not ids:
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} is {status} without evidence")
            if (result.get("support_status") == "supported"
                    and verification["codegen_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} is supported without codegen proof")
            if (verification["cmodel_numeric_passed"]["status"] == "passed"
                    and verification["codegen_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} passes CModel without codegen proof")
            if (verification["pcie_numeric_passed"]["status"] == "passed"
                    and verification["cmodel_numeric_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} passes PCIe without CModel proof")

    # Lock the two easy-to-misstate characterization results.
    for capability in capabilities.values():
        if capability["operation_id"] != "topk" or capability["id"] == "topk.unsupported-floats":
            continue
        sg = capability["target_results"].get("sg2260e.tpukernel")
        bm = capability["target_results"].get("bm1690.tpukernel")
        if (not sg or sg["support_status"] != "unsupported"
                or sg["verification"]["codegen_passed"]["status"] != "failed"
                or sg["verification"]["cmodel_numeric_passed"]["status"] != "not_applicable"
                or sg["verification"]["pcie_numeric_passed"]["status"] != "not_applicable"):
            raise ContractError(f"{capability['id']} must keep SG2260E topk compile-time unsupported")
        if (not bm or bm["support_status"] != "supported"
                or bm["verification"]["cmodel_numeric_passed"]["status"] != "passed"):
            raise ContractError(f"{capability['id']} must retain BM1690 CModel verification")

    scalar = capabilities.get("scalar.add-mul-fp8")
    if scalar is None:
        raise ContractError("missing scalar.add-mul-fp8 capability")
    for target_id in ("bm1690.tpukernel", "sg2260e.tpukernel"):
        result = scalar["target_results"].get(target_id)
        if (not result or result["support_status"] != "supported"
                or result["verification"]["cmodel_numeric_passed"]["status"] != "passed"):
            raise ContractError(
                f"{target_id} FP8 scalar must reflect the valid generic-API CModel result")

    _validate_implementation_sets(operations)
    return len(targets), len(operations), len(capabilities), len(evidence)


def main() -> int:
    try:
        targets, operations, capabilities, evidence = validate()
    except ContractError as error:
        print(f"contract validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "contract validation passed: "
        f"{targets} targets, {operations} operations, "
        f"{capabilities} capabilities, {evidence} evidence entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
