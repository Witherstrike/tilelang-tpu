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
SUPPORT_STATUSES = {"supported", "unsupported", "unverified", "experimental"}
IMPLEMENTATION_CONFORMANCES = {
    "conforming",
    "known_gap",
    "unverified",
    "not_applicable",
}
DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "e4m3_float8",
    "e5m2_float8",
    "int32",
    "uint32",
    "int16",
    "uint16",
    "int8",
    "uint8",
    "bool",
}
OPTIONAL_STAGES = {"sdk_declared"}
BACKENDS = {"tpukernel", "rv"}
CONSTRAINT_STATUSES = {"enforced", "documented", "unverified", "known_gap"}
CONSTRAINT_CATEGORIES = {
    "dtype", "shape", "layout", "memory", "value", "aliasing", "target", "scheduling"
}
CONSTRAINT_PHASES = {
    "frontend", "lowering", "codegen", "compile", "runtime", "not_enforced"
}
FAILURE_PHASES = {
    "frontend", "lowering", "codegen", "compile", "runtime", "test_supervisor"
}
FAILURE_ACTIONS = {
    "reject", "abort_compile", "abort_run", "terminate_process_group_and_skip_remaining"
}
FAILURE_CONFORMANCES = {"required", "implemented", "known_gap"}


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


def _unique_named_index(kind: str, rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"{kind} has a missing or non-string name")
        if name in result:
            raise ContractError(f"duplicate {kind} name: {name}")
        result[name] = row
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
        if not isinstance(evidence_id, str):
            raise ContractError(f"{owner}.evidence_ids contains a non-string id")
        if evidence_id not in evidence:
            raise ContractError(f"{owner} references unknown evidence {evidence_id!r}")
    if len(ids) != len(set(ids)):
        raise ContractError(f"{owner}.evidence_ids contains duplicates")


def _require_nonempty_string(label: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")


def _validate_string_list(label: str, value: Any, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ContractError(f"{label} must be a {qualifier}array")
    for item in value:
        _require_nonempty_string(f"{label} item", item)
    if len(value) != len(set(value)):
        raise ContractError(f"{label} contains duplicates")
    return value


def _collect_evidence_refs(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_ids" and isinstance(child, list):
                result.update(item for item in child if isinstance(item, str))
            else:
                result.update(_collect_evidence_refs(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_evidence_refs(child))
    return result


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


def _extract_python_chip_specs(path: Path) -> dict[str, dict[str, Any]]:
    """Read the literal fields that define ``TPU_CHIP_SPECS`` without imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value: ast.expr | None = None
    for node in tree.body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "TPU_CHIP_SPECS"):
            value = node.value
            break
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "TPU_CHIP_SPECS" for target in node.targets)):
            value = node.value
            break
    if (not isinstance(value, ast.Call) or len(value.args) != 1
            or not isinstance(value.args[0], ast.Dict)):
        raise ContractError("TPU_CHIP_SPECS must remain a literal mapping constructor")

    result: dict[str, dict[str, Any]] = {}
    for key_node, spec_node in zip(
            value.args[0].keys, value.args[0].values):
        if (not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str)
                or not isinstance(spec_node, ast.Call)):
            raise ContractError("TPU_CHIP_SPECS contains a non-literal chip entry")
        fields = {keyword.arg: ast.literal_eval(keyword.value)
                  for keyword in spec_node.keywords if keyword.arg is not None}
        required = {"name", "ppl_arch", "physical_core_count", "programming_models"}
        if not required <= fields.keys():
            raise ContractError(
                f"TPU_CHIP_SPECS[{key_node.value!r}] lacks {sorted(required - fields.keys())}")
        if key_node.value in result:
            raise ContractError(f"duplicate TPU_CHIP_SPECS chip {key_node.value!r}")
        result[key_node.value] = fields
    return result


def _semantic_externs(path: Path) -> set[str]:
    return set(SEMANTIC_EXTERN.findall(path.read_text(encoding="utf-8")))


def _check_exact_set(label: str, expected: set[str], actual: set[str]) -> None:
    if expected == actual:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ContractError(f"{label} mismatch; missing={missing}, extra={extra}")


def _validate_effect_operands(operation_id: str, operation: dict[str, Any]) -> None:
    """Keep the machine contract's operand roles internally self-consistent."""

    parameters = _unique_named_index(
        f"parameter in operation {operation_id}", operation.get("parameters", []))
    effects = operation.get("effects")
    if not isinstance(effects, dict):
        raise ContractError(f"operation {operation_id} has no effects object")

    def check_group(owner: str, group: dict[str, Any]) -> None:
        access_sets: dict[str, set[str]] = {}
        for access in ("reads", "writes", "read_writes"):
            operands = group.get(access)
            if not isinstance(operands, list):
                raise ContractError(f"{owner}.{access} must be an array")
            if len(operands) != len(set(operands)):
                raise ContractError(f"{owner}.{access} contains duplicate operands")
            access_sets[access] = set(operands)
            for operand in operands:
                if operand not in parameters:
                    raise ContractError(f"{owner}.{access} references unknown operand {operand!r}")
                parameter = parameters[operand]
                if parameter.get("kind") not in {"buffer", "scratch"}:
                    raise ContractError(
                        f"{owner}.{access} references non-memory operand {operand!r}")
                direction = parameter.get("direction")
                allowed_directions = {
                    "reads": {"in", "inout"},
                    "writes": {"out", "inout"},
                    "read_writes": {"inout"},
                }[access]
                if direction not in allowed_directions:
                    raise ContractError(
                        f"{owner}.{access} conflicts with {operand!r} direction {direction!r}")
        if (access_sets["reads"] & access_sets["writes"]
                or access_sets["reads"] & access_sets["read_writes"]
                or access_sets["writes"] & access_sets["read_writes"]):
            raise ContractError(f"{owner} assigns one operand to multiple access classes")

    check_group(f"operation {operation_id}.effects", effects)
    conditional = effects.get("conditional")
    if not isinstance(conditional, list):
        raise ContractError(f"operation {operation_id}.effects.conditional must be an array")
    for index, group in enumerate(conditional):
        if not isinstance(group, dict):
            raise ContractError(
                f"operation {operation_id}.effects.conditional[{index}] must be an object")
        check_group(f"operation {operation_id}.effects.conditional[{index}]", group)


def _validate_targets(targets: dict[str, dict[str, Any]]) -> None:
    """Bind contract target rows to the canonical Python chip capability table."""

    specs = _extract_python_chip_specs(REPO_ROOT / "tilelang/engine/tpu_config.py")
    programming_models = {"tpukernel", "rv"}
    expected_ids = {
        f"{chip}.{programming_model}"
        for chip in specs
        for programming_model in programming_models
    }
    _check_exact_set("contract targets vs TPU_CHIP_SPECS cartesian pairs", expected_ids, set(targets))

    for target_id, target in targets.items():
        chip = target.get("chip")
        programming_model = target.get("backend")
        if target_id != f"{chip}.{programming_model}":
            raise ContractError(
                f"target id {target_id!r} disagrees with chip/backend fields")
        spec = specs[chip]
        applicable = programming_model in set(spec["programming_models"])
        if target.get("applicable") is not applicable:
            raise ContractError(
                f"target {target_id} applicability disagrees with TPU_CHIP_SPECS")
        if applicable:
            if target.get("ppl_arch") != spec["ppl_arch"]:
                raise ContractError(f"target {target_id} has stale ppl_arch")
            if target.get("physical_core_count") != spec["physical_core_count"]:
                raise ContractError(f"target {target_id} has stale physical_core_count")
        elif "ppl_arch" in target or "physical_core_count" in target:
            raise ContractError(
                f"inapplicable target {target_id} must not expose runtime metadata")


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
    invariants = _unique_index("invariant", contract.get("invariants", []))
    evidence = _unique_index("evidence", contract.get("evidence", []))
    capabilities = _unique_index("capability", contract.get("capabilities", []))

    vocabulary = contract.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ContractError("vocabulary must be an object")
    for field, expected in (
            ("support_status", SUPPORT_STATUSES),
            ("stage_status", STAGE_STATUSES),
            ("implementation_conformance", IMPLEMENTATION_CONFORMANCES)):
        values = vocabulary.get(field)
        if not isinstance(values, dict):
            raise ContractError(f"vocabulary.{field} must be an object")
        _check_exact_set(f"vocabulary.{field}", expected, set(values))

    _validate_targets(targets)

    scope = contract.get("scope")
    if not isinstance(scope, dict):
        raise ContractError("scope must be an object")
    included_operations = scope.get("included_operations")
    if not isinstance(included_operations, list):
        raise ContractError("scope.included_operations must be an array")
    _check_exact_set(
        "scope.included_operations vs operations",
        set(operations),
        set(included_operations),
    )

    for invariant_id, invariant in invariants.items():
        applicable_operations = set(_validate_string_list(
            f"invariant {invariant_id}.applies_to_operations",
            invariant.get("applies_to_operations"), nonempty=True))
        unknown_operations = applicable_operations - set(operations)
        if unknown_operations:
            raise ContractError(
                f"invariant {invariant_id} references unknown operations "
                f"{sorted(unknown_operations)}")
        if invariant.get("status") not in CONSTRAINT_STATUSES:
            raise ContractError(
                f"invariant {invariant_id} has invalid status "
                f"{invariant.get('status')!r}")
        if invariant.get("enforced_at") not in CONSTRAINT_PHASES:
            raise ContractError(
                f"invariant {invariant_id} has invalid enforcement phase "
                f"{invariant.get('enforced_at')!r}")
        if invariant.get("category") not in CONSTRAINT_CATEGORIES:
            raise ContractError(
                f"invariant {invariant_id} has invalid category "
                f"{invariant.get('category')!r}")
        for field in ("expression", "description"):
            _require_nonempty_string(
                f"invariant {invariant_id}.{field}", invariant.get(field))
        _evidence_refs(
            f"invariant {invariant_id}",
            invariant.get("evidence_ids", []), evidence)

    for target_id, target in targets.items():
        _evidence_refs(f"target {target_id}", target.get("evidence_ids", []), evidence)

    for evidence_id, item in evidence.items():
        root_id = item.get("source_root")
        if root_id not in source_roots:
            raise ContractError(f"evidence {evidence_id} uses unknown source root {root_id!r}")
        _check_portable_path(f"evidence {evidence_id}.path", item.get("path"))
        _require_nonempty_string(f"evidence {evidence_id}.locator", item.get("locator"))
        _require_nonempty_string(f"evidence {evidence_id}.summary", item.get("summary"))
        root = source_roots[root_id]
        if root.get("tracked") and item.get("kind") in {
                "source", "test_source", "test_report", "review_note"}:
            path = REPO_ROOT / root["path"] / item["path"]
            if not path.is_file():
                raise ContractError(f"tracked evidence is missing: {path.relative_to(REPO_ROOT)}")
            if item.get("kind") in {"source", "test_source"}:
                for token in (part.strip() for part in item["locator"].split(",")):
                    if token and token not in path.read_text(encoding="utf-8"):
                        raise ContractError(
                            f"evidence {evidence_id} locator token {token!r} is absent from "
                            f"{path.relative_to(REPO_ROOT)}")

    for operation_id, operation in operations.items():
        _evidence_refs(f"operation {operation_id}", operation.get("evidence_ids", []), evidence)
        _require_nonempty_string(f"operation {operation_id}.semantics", operation.get("semantics"))
        _validate_string_list(
            f"operation {operation_id}.frontend_symbols",
            operation.get("frontend_symbols"), nonempty=True)
        _validate_string_list(
            f"operation {operation_id}.internal_symbols",
            operation.get("internal_symbols"), nonempty=True)
        backends = set(_validate_string_list(
            f"operation {operation_id}.backend_applicability",
            operation.get("backend_applicability"), nonempty=True))
        if not backends <= BACKENDS:
            raise ContractError(
                f"operation {operation_id} has unknown backends {sorted(backends - BACKENDS)}")
        _validate_effect_operands(operation_id, operation)
        constraints = _unique_index(
            f"constraint in operation {operation_id}", operation.get("constraints", []))
        for constraint_id, constraint in constraints.items():
            if not constraint_id.startswith(f"{operation_id}."):
                raise ContractError(
                    f"constraint {constraint_id} must use its operation id as a prefix")
            if constraint.get("status") not in CONSTRAINT_STATUSES:
                raise ContractError(
                    f"constraint {constraint_id} has invalid status {constraint.get('status')!r}")
            if constraint.get("enforced_at") not in CONSTRAINT_PHASES:
                raise ContractError(
                    f"constraint {constraint_id} has invalid enforcement phase "
                    f"{constraint.get('enforced_at')!r}")
            for field in ("category", "expression", "description"):
                _require_nonempty_string(
                    f"constraint {constraint_id}.{field}", constraint.get(field))
            if constraint.get("category") not in CONSTRAINT_CATEGORIES:
                raise ContractError(
                    f"constraint {constraint_id} has invalid category "
                    f"{constraint.get('category')!r}")
            _evidence_refs(
                f"constraint {operation_id}/{constraint_id}",
                constraint.get("evidence_ids", []),
                evidence,
            )
        failure_policy = operation.get("failure_policy")
        if not isinstance(failure_policy, list) or not failure_policy:
            raise ContractError(f"operation {operation_id}.failure_policy must be non-empty")
        for index, policy in enumerate(failure_policy):
            if not isinstance(policy, dict):
                raise ContractError(
                    f"operation {operation_id}.failure_policy[{index}] must be an object")
            for field in ("condition", "diagnostic"):
                _require_nonempty_string(
                    f"operation {operation_id}.failure_policy[{index}].{field}",
                    policy.get(field))
            if policy.get("phase") not in FAILURE_PHASES:
                raise ContractError(
                    f"operation {operation_id}.failure_policy[{index}] has invalid phase")
            if policy.get("action") not in FAILURE_ACTIONS:
                raise ContractError(
                    f"operation {operation_id}.failure_policy[{index}] has invalid action")
            if policy.get("conformance") not in FAILURE_CONFORMANCES:
                raise ContractError(
                    f"operation {operation_id}.failure_policy[{index}] has invalid conformance")

    for capability_id, capability in capabilities.items():
        _evidence_refs(
            f"capability {capability_id}", capability.get("evidence_ids", []), evidence)
        operation_id = capability.get("operation_id")
        if operation_id not in operations:
            raise ContractError(
                f"capability {capability_id} references unknown operation {operation_id!r}")
        if not capability_id.startswith(f"{operation_id}."):
            raise ContractError(
                f"capability {capability_id} must use its operation id as a prefix")
        _require_nonempty_string(f"capability {capability_id}.variant", capability.get("variant"))
        bindings = capability.get("dtype_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ContractError(f"capability {capability_id}.dtype_bindings must be non-empty")
        normalized_bindings: set[tuple[tuple[str, str], ...]] = set()
        for index, binding in enumerate(bindings):
            if not isinstance(binding, dict) or not binding:
                raise ContractError(
                    f"capability {capability_id}.dtype_bindings[{index}] must be a non-empty object")
            for role, dtype in binding.items():
                _require_nonempty_string(
                    f"capability {capability_id}.dtype_bindings[{index}] role", role)
                if dtype not in DTYPES:
                    raise ContractError(
                        f"capability {capability_id}.dtype_bindings[{index}] has unknown "
                        f"dtype {dtype!r}")
            normalized = tuple(sorted(binding.items()))
            if normalized in normalized_bindings:
                raise ContractError(
                    f"capability {capability_id}.dtype_bindings contains duplicates")
            normalized_bindings.add(normalized)
        operation_constraints = {
            constraint["id"] for constraint in operations[operation_id].get("constraints", [])
        }
        capability_constraints = capability.get("constraints")
        if not isinstance(capability_constraints, list):
            raise ContractError(f"capability {capability_id}.constraints must be an array")
        if len(capability_constraints) != len(set(capability_constraints)):
            raise ContractError(f"capability {capability_id}.constraints contains duplicates")
        for constraint_id in capability_constraints:
            if constraint_id not in operation_constraints:
                raise ContractError(
                    f"capability {capability_id} references unknown constraint {constraint_id!r}")

        results = capability.get("target_results")
        if not isinstance(results, dict) or not results:
            raise ContractError(f"capability {capability_id} has no target_results")
        expected_targets = {
            target_id
            for target_id, target in targets.items()
            if target.get("applicable")
            and target.get("backend") in operations[operation_id].get("backend_applicability", [])
        }
        _check_exact_set(
            f"capability {capability_id} target coverage",
            expected_targets,
            set(results),
        )
        declared_statuses: set[str] = set()
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
            unknown_stages = verification.keys() - REQUIRED_STAGES - OPTIONAL_STAGES
            if unknown_stages:
                raise ContractError(
                    f"{capability_id}/{target_id} has unknown stages {sorted(unknown_stages)}")
            _require_nonempty_string(
                f"{capability_id}/{target_id}.reason", result.get("reason"))
            for stage_name, stage in verification.items():
                if not isinstance(stage, dict):
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} must be an object")
                status = stage.get("status")
                if status not in STAGE_STATUSES:
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} has invalid status {status!r}")
                ids = stage.get("evidence_ids", [])
                _evidence_refs(f"{capability_id}/{target_id}/{stage_name}", ids, evidence)
                _require_nonempty_string(
                    f"{capability_id}/{target_id}/{stage_name}.scope", stage.get("scope"))
                _require_nonempty_string(
                    f"{capability_id}/{target_id}/{stage_name}.reason", stage.get("reason"))
                if status in {"passed", "failed"} and not ids:
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} is {status} without evidence")
                if (stage_name == "declared" and status in {"passed", "failed"}
                        and "src.frontend" not in ids):
                    raise ContractError(
                        f"{capability_id}/{target_id}/declared is {status} without "
                        "src.frontend evidence")
            declared_statuses.add(verification["declared"]["status"])
            support_status = result.get("support_status")
            if support_status not in SUPPORT_STATUSES:
                raise ContractError(
                    f"{capability_id}/{target_id} has invalid support_status "
                    f"{support_status!r}")
            conformance = result.get("implementation_conformance")
            if conformance not in IMPLEMENTATION_CONFORMANCES:
                raise ContractError(
                    f"{capability_id}/{target_id} has invalid implementation_conformance "
                    f"{conformance!r}")
            if (support_status == "supported"
                    and verification["codegen_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} is supported without codegen proof")
            if (support_status == "unverified"
                    and verification["codegen_passed"]["status"] != "unverified"):
                raise ContractError(
                    f"{capability_id}/{target_id} is unverified but its codegen stage is "
                    f"{verification['codegen_passed']['status']!r}")
            if (support_status == "unsupported"
                    and verification["declared"]["status"] != "failed"
                    and verification["codegen_passed"]["status"] != "failed"):
                raise ContractError(
                    f"{capability_id}/{target_id} is unsupported without a frontend or "
                    "codegen rejection")
            if (verification["cmodel_numeric_passed"]["status"] == "passed"
                    and verification["codegen_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} passes CModel without codegen proof")
            if (verification["pcie_numeric_passed"]["status"] == "passed"
                    and verification["cmodel_numeric_passed"]["status"] != "passed"):
                raise ContractError(f"{capability_id}/{target_id} passes PCIe without CModel proof")
            if verification["codegen_passed"]["status"] == "failed":
                for numeric_stage in ("cmodel_numeric_passed", "pcie_numeric_passed"):
                    if verification[numeric_stage]["status"] != "not_applicable":
                        raise ContractError(
                            f"{capability_id}/{target_id} has failed codegen but "
                            f"{numeric_stage} is {verification[numeric_stage]['status']!r}")
            if verification["codegen_passed"]["status"] == "not_applicable":
                for numeric_stage in ("cmodel_numeric_passed", "pcie_numeric_passed"):
                    if verification[numeric_stage]["status"] != "not_applicable":
                        raise ContractError(
                            f"{capability_id}/{target_id} has non-applicable codegen but "
                            f"{numeric_stage} is {verification[numeric_stage]['status']!r}")
        if len(declared_statuses) != 1:
            raise ContractError(
                f"{capability_id} has target-dependent frontend declared statuses: "
                f"{sorted(declared_statuses)}")

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

    fp16_to_bf16 = capabilities.get("copy.cast-fp16-to-bf16")
    bf16_to_fp16 = capabilities.get("copy.cast-bf16-to-fp16")
    if fp16_to_bf16 is None or bf16_to_fp16 is None:
        raise ContractError("FP16/BF16 RV conversion directions must remain separate selectors")
    if (fp16_to_bf16["target_results"]["sg2260e.rv"]["support_status"] != "supported"
            or bf16_to_fp16["target_results"]["sg2260e.rv"]["support_status"]
            != "unverified"):
        raise ContractError(
            "RV FP16-to-BF16 has source proof, while BF16-to-FP16 must remain unverified")

    nt_accumulate = capabilities.get("gemm.nt-accumulate")
    if nt_accumulate is None:
        raise ContractError("missing gemm.nt-accumulate capability")
    for target_id in ("bm1690.tpukernel", "sg2260e.tpukernel"):
        result = nt_accumulate["target_results"].get(target_id)
        if (not result or result["support_status"] != "unsupported"
                or result["verification"]["declared"]["status"] != "passed"
                or result["verification"]["codegen_passed"]["status"] != "failed"
                or result["verification"]["cmodel_numeric_passed"]["status"]
                != "not_applicable"
                or result["verification"]["pcie_numeric_passed"]["status"]
                != "not_applicable"):
            raise ContractError(
                f"{target_id} base-float NT accumulation must remain a target-codegen "
                "rejection, not a frontend rejection")
    rv_nt = nt_accumulate["target_results"].get("sg2260e.rv")
    if (not rv_nt or rv_nt["support_status"] != "supported"
            or rv_nt["verification"]["declared"]["status"] != "passed"
            or rv_nt["verification"]["codegen_passed"]["status"] != "passed"
            or rv_nt["verification"]["cmodel_numeric_passed"]["status"] != "unverified"
            or rv_nt["verification"]["pcie_numeric_passed"]["status"] != "unverified"):
        raise ContractError(
            "SG2260E RV base-float NT accumulation has source-selection proof only; "
            "numeric stages must remain unverified")

    for capability_id in (
            "gemm.nn-overwrite-base-fp32", "gemm.nt-overwrite-base-fp32"):
        capability = capabilities.get(capability_id)
        if capability is None:
            raise ContractError(f"missing accepted-but-unverified selector {capability_id}")
        for target_id, result in capability["target_results"].items():
            if (result["support_status"] != "unverified"
                    or result["verification"]["declared"]["status"] != "passed"
                    or result["verification"]["codegen_passed"]["status"] != "unverified"
                    or result["verification"]["cmodel_numeric_passed"]["status"]
                    != "unverified"
                    or result["verification"]["pcie_numeric_passed"]["status"]
                    != "unverified"):
                raise ContractError(
                    f"{capability_id}/{target_id} is admitted by source but lacks exact "
                    "lowering and numerical evidence")

    _check_exact_set(
        "operations vs capability operation coverage",
        set(operations),
        {capability["operation_id"] for capability in capabilities.values()},
    )
    _check_exact_set(
        "operation constraints vs capability constraint references",
        {
            constraint["id"]
            for operation in operations.values()
            for constraint in operation.get("constraints", [])
        },
        {
            constraint_id
            for capability in capabilities.values()
            for constraint_id in capability.get("constraints", [])
        },
    )

    contract_without_evidence = {
        key: value for key, value in contract.items() if key != "evidence"
    }
    referenced_evidence = _collect_evidence_refs(contract_without_evidence)
    _check_exact_set("evidence entries vs referenced evidence", set(evidence), referenced_evidence)

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
