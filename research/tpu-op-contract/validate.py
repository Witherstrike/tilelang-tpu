#!/usr/bin/env python3
"""Validate the TileLang TPU operation contract with the Python standard library.

This checker deliberately does not require ``jsonschema``.  It verifies the
cross-reference and implementation-set invariants that JSON Schema cannot
express, while ignored runtime artifacts remain optional in another clone.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONTRACT_PATH = HERE / "contract.json"
SCHEMA_PATH = HERE / "schema.json"
ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
SEMANTIC_EXTERN = re.compile(r"tl\.(?:tpu|tpukernel)\.[A-Za-z0-9_]+")
REQUIRED_STAGES = {
    "declared",
    "codegen_passed",
    "cmodel_numeric_passed",
    "pcie_numeric_passed",
}
STAGE_STATUSES = {
    "passed", "historical_passed", "failed", "unverified", "not_applicable"
}
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
TOP_LEVEL_FIELDS = {
    "$schema",
    "schema_version",
    "contract_id",
    "reviewed_at",
    "validated_source_revision",
    "scope",
    "invariants",
    "vocabulary",
    "source_roots",
    "targets",
    "operations",
    "evidence",
    "capabilities",
}


class ContractError(ValueError):
    """A deterministic contract validation failure."""


class SchemaDefinitionError(ContractError):
    """The schema itself is unsupported or malformed, not an instance mismatch."""


def _json_schema_matches(value: Any, schema: dict[str, Any], root: dict[str, Any],
                         path: str) -> bool:
    try:
        _validate_json_schema(value, schema, root, path)
    except SchemaDefinitionError:
        raise
    except ContractError:
        return False
    return True


def _validate_json_schema(value: Any, schema: dict[str, Any], root: dict[str, Any],
                          path: str) -> None:
    """Validate the JSON-Schema subset used by this repository.

    Keeping this small evaluator next to the semantic checks preserves the
    standard-library-only contract validator while making ``schema.json``
    authoritative rather than merely syntax-checked.  Unsupported schema
    keywords must be added here before they are introduced in the schema.
    """

    supported = {
        "$schema", "$id", "$defs", "$ref", "title", "description",
        "type", "const", "enum", "properties", "required",
        "additionalProperties", "propertyNames", "dependentRequired",
        "items", "minItems", "uniqueItems", "minProperties", "minLength",
        "minimum", "pattern", "format", "allOf", "if", "then", "not",
    }
    unknown = set(schema) - supported
    if unknown:
        raise SchemaDefinitionError(
            f"schema uses unsupported keywords at {path}: {sorted(unknown)}")

    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise SchemaDefinitionError(
                f"schema has unsupported reference {reference!r}")
        name = reference.removeprefix("#/$defs/")
        definition = root.get("$defs", {}).get(name)
        if not isinstance(definition, dict):
            raise SchemaDefinitionError(
                f"schema reference {reference!r} is unresolved")
        _validate_json_schema(value, definition, root, path)

    for child in schema.get("allOf", []):
        _validate_json_schema(value, child, root, path)
    condition = schema.get("if")
    if isinstance(condition, dict) and _json_schema_matches(value, condition, root, path):
        then = schema.get("then")
        if isinstance(then, dict):
            _validate_json_schema(value, then, root, path)
    rejected = schema.get("not")
    if isinstance(rejected, dict) and _json_schema_matches(value, rejected, root, path):
        raise ContractError(f"{path} matches a forbidden schema")

    expected_type = schema.get("type")
    type_matches = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected_type is not None:
        predicate = type_matches.get(expected_type)
        if predicate is None:
            raise SchemaDefinitionError(
                f"schema has unsupported type {expected_type!r}")
        if not predicate(value):
            raise ContractError(f"{path} must have JSON type {expected_type}")

    if "const" in schema and (
            type(value) is not type(schema["const"]) or value != schema["const"]):
        raise ContractError(f"{path} does not equal its schema constant")
    if "enum" in schema and not any(
            type(value) is type(candidate) and value == candidate
            for candidate in schema["enum"]):
        raise ContractError(f"{path} is outside its schema enum")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = set(required) - set(value)
        if missing:
            raise ContractError(f"{path} lacks required fields {sorted(missing)}")
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value:
                _validate_json_schema(value[key], child, root, f"{path}.{key}")
        additional = schema.get("additionalProperties", True)
        for key in set(value) - set(properties):
            if additional is False:
                raise ContractError(f"{path} has unknown field {key!r}")
            if isinstance(additional, dict):
                _validate_json_schema(value[key], additional, root, f"{path}.{key}")
        property_names = schema.get("propertyNames")
        if isinstance(property_names, dict):
            for key in value:
                _validate_json_schema(key, property_names, root, f"{path} key")
        minimum_properties = schema.get("minProperties")
        if minimum_properties is not None and len(value) < minimum_properties:
            raise ContractError(f"{path} has too few properties")
        for dependency, dependents in schema.get("dependentRequired", {}).items():
            if dependency in value:
                missing_dependents = set(dependents) - set(value)
                if missing_dependents:
                    raise ContractError(
                        f"{path} field {dependency!r} requires "
                        f"{sorted(missing_dependents)}")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if minimum_items is not None and len(value) < minimum_items:
            raise ContractError(f"{path} has too few items")
        if schema.get("uniqueItems"):
            serialized = [
                json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                for item in value
            ]
            if len(serialized) != len(set(serialized)):
                raise ContractError(f"{path} contains duplicate items")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_json_schema(item, items, root, f"{path}[{index}]")

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None and len(value) < minimum_length:
            raise ContractError(f"{path} is shorter than its schema minimum")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ContractError(f"{path} does not match its schema pattern")
        if schema.get("format") == "date-time":
            _validate_timestamp(path, value)

    minimum = schema.get("minimum")
    if minimum is not None and isinstance(value, int) and not isinstance(value, bool):
        if value < minimum:
            raise ContractError(f"{path} is below its schema minimum")


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


def _validate_timestamp(label: str, value: Any) -> None:
    _require_nonempty_string(label, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ContractError(f"{label} must include a UTC offset")


def _validate_runtime_report(
        evidence_id: str, artifact: dict[str, Any], expectation: Any) -> None:
    """Apply the evidence row's exact assertions to a local runtime summary."""

    if not isinstance(expectation, dict):
        raise ContractError(f"evidence {evidence_id}.runtime_expectation must be an object")
    required = {
        "runtime_mode", "complete", "case_count", "all_cases_passed",
        "claim_targets", "capability_ids", "target_case_counts",
    }
    allowed = required | {"required_case_ids"}
    if not required <= set(expectation) or set(expectation) - allowed:
        raise ContractError(
            f"evidence {evidence_id}.runtime_expectation fields are incomplete or unknown")
    runtime_mode = expectation.get("runtime_mode")
    if runtime_mode not in {"cmodel", "pcie"}:
        raise ContractError(f"evidence {evidence_id} has invalid expected runtime mode")
    if expectation.get("complete") is not True:
        raise ContractError(f"evidence {evidence_id} must expect complete=true")
    case_count = expectation.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
        raise ContractError(f"evidence {evidence_id} has invalid expected case count")
    if expectation.get("all_cases_passed") is not True:
        raise ContractError(f"evidence {evidence_id} must expect all cases passed")
    claim_targets = set(_validate_string_list(
        f"evidence {evidence_id}.runtime_expectation.claim_targets",
        expectation.get("claim_targets"),
        nonempty=True,
    ))
    _validate_string_list(
        f"evidence {evidence_id}.runtime_expectation.capability_ids",
        expectation.get("capability_ids"),
        nonempty=True,
    )
    if artifact.get("runtime_mode") != runtime_mode:
        raise ContractError(
            f"evidence {evidence_id} runtime mode disagrees with its local artifact")
    if artifact.get("complete") is not True:
        raise ContractError(f"evidence {evidence_id} local artifact is incomplete")

    cases = artifact.get("cases")
    results = artifact.get("results")
    records: list[dict[str, Any]]
    target_counts: Counter[str] = Counter()
    actual_case_ids: set[str] = set()
    if isinstance(cases, dict):
        records = []
        for case_key, record in cases.items():
            if not isinstance(case_key, str) or not isinstance(record, dict):
                raise ContractError(f"evidence {evidence_id} has malformed cases")
            parts = case_key.split("/", 2)
            if len(parts) < 3:
                raise ContractError(
                    f"evidence {evidence_id} case key lacks chip/backend identity")
            target_counts[f"{parts[0]}.{parts[1]}"] += 1
            actual_case_ids.add(case_key)
            records.append(record)
    elif isinstance(results, list):
        records = []
        programming_model = artifact.get("programming_model")
        _require_nonempty_string(
            f"evidence {evidence_id}.artifact.programming_model", programming_model)
        for record in results:
            if not isinstance(record, dict):
                raise ContractError(f"evidence {evidence_id} has malformed results")
            chip = record.get("chip")
            _require_nonempty_string(f"evidence {evidence_id}.result.chip", chip)
            target_counts[f"{chip}.{programming_model}"] += 1
            case = record.get("case")
            case_id = case.get("case_id") if isinstance(case, dict) else None
            _require_nonempty_string(f"evidence {evidence_id}.result.case.case_id", case_id)
            actual_case_ids.add(f"{chip}/{programming_model}/{case_id}")
            records.append(record)
    else:
        raise ContractError(
            f"evidence {evidence_id} local artifact has no cases/results collection")

    if len(records) != case_count:
        raise ContractError(
            f"evidence {evidence_id} expected {case_count} cases, found {len(records)}")
    if len(actual_case_ids) != len(records):
        raise ContractError(f"evidence {evidence_id} local artifact has duplicate case ids")
    if any(record.get("status") != "passed" for record in records):
        raise ContractError(f"evidence {evidence_id} local artifact has a non-passing case")
    expected_targets = expectation.get("target_case_counts")
    if (not isinstance(expected_targets, dict) or not expected_targets
            or any(not isinstance(key, str) or not key for key in expected_targets)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                   for value in expected_targets.values())):
        raise ContractError(
            f"evidence {evidence_id} has invalid expected target counts")
    if dict(target_counts) != expected_targets:
        raise ContractError(
            f"evidence {evidence_id} target counts disagree with its local artifact; "
            f"expected={expected_targets}, actual={dict(target_counts)}")
    if not claim_targets <= set(expected_targets):
        raise ContractError(
            f"evidence {evidence_id} claims targets absent from its case counts")
    required_case_ids = expectation.get("required_case_ids")
    if required_case_ids is not None:
        requested = set(_validate_string_list(
            f"evidence {evidence_id}.runtime_expectation.required_case_ids",
            required_case_ids,
            nonempty=True,
        ))
        missing = requested - actual_case_ids
        if missing:
            raise ContractError(
                f"evidence {evidence_id} local artifact lacks required cases "
                f"{sorted(missing)}")


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


def validate(*, require_local_artifacts: bool = False) -> tuple[int, int, int, int, int]:
    # The repository-owned evaluator executes the schema subset used here, so
    # structural checks remain active without a third-party jsonschema package.
    schema = _load_json(SCHEMA_PATH)
    contract = _load_json(CONTRACT_PATH)
    _validate_json_schema(contract, schema, schema, "contract")
    if set(contract) != TOP_LEVEL_FIELDS:
        raise ContractError(
            "contract top-level fields mismatch; "
            f"missing={sorted(TOP_LEVEL_FIELDS - set(contract))}, "
            f"extra={sorted(set(contract) - TOP_LEVEL_FIELDS)}")
    if contract.get("$schema") != "./schema.json":
        raise ContractError("contract.$schema must be './schema.json'")
    if contract.get("schema_version") != "1.2.0":
        raise ContractError("contract.schema_version must be '1.2.0'")
    if contract.get("contract_id") != "tilelang-tpu.current-op-capabilities":
        raise ContractError("contract.contract_id is not canonical")
    _validate_timestamp("contract.reviewed_at", contract.get("reviewed_at"))
    validated_source_revision = contract.get("validated_source_revision")
    if (not isinstance(validated_source_revision, str)
            or GIT_REVISION.fullmatch(validated_source_revision) is None):
        raise ContractError(
            "validated_source_revision must be a full lowercase Git revision")

    source_roots = contract.get("source_roots")
    if not isinstance(source_roots, dict) or not source_roots:
        raise ContractError("source_roots must be a non-empty object")
    for root_id, root in source_roots.items():
        if not isinstance(root, dict):
            raise ContractError(f"source root {root_id!r} must be an object")
        _check_portable_path(f"source_roots.{root_id}.path", root.get("path"), symbolic=True)
    artifacts_root = source_roots.get("artifacts")
    if (not isinstance(artifacts_root, dict)
            or artifacts_root.get("path") != "research/artifacts"
            or artifacts_root.get("tracked") is not False):
        raise ContractError(
            "source_roots.artifacts must identify ignored research/artifacts")

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
        if evidence_id.startswith("artifact.") and root_id != "artifacts":
            raise ContractError(
                f"ignored runtime evidence {evidence_id} must use the untracked "
                "artifacts source root")
        if root_id == "artifacts" and not evidence_id.startswith("artifact."):
            raise ContractError(
                f"non-artifact evidence {evidence_id} cannot use the artifacts source root")
        if root_id == "artifacts" and source_roots[root_id].get("tracked") is not False:
            raise ContractError("the artifacts source root must remain explicitly untracked")
        _check_portable_path(f"evidence {evidence_id}.path", item.get("path"))
        _require_nonempty_string(f"evidence {evidence_id}.locator", item.get("locator"))
        _require_nonempty_string(f"evidence {evidence_id}.summary", item.get("summary"))
        _validate_timestamp(f"evidence {evidence_id}.observed_at", item.get("observed_at"))
        if item.get("time_precision") not in {"second", "minute", "hour", "day"}:
            raise ContractError(f"evidence {evidence_id} has invalid time_precision")
        source_revision = item.get("source_revision")
        identity_source = item.get("identity_source")
        if source_revision is not None:
            if (not isinstance(source_revision, str)
                    or GIT_REVISION.fullmatch(source_revision) is None):
                raise ContractError(
                    f"evidence {evidence_id}.source_revision is not a full Git revision")
            if identity_source not in {"runner_recorded", "operator_recorded"}:
                raise ContractError(
                    f"evidence {evidence_id} has invalid identity_source {identity_source!r}")
        elif identity_source is not None:
            raise ContractError(
                f"evidence {evidence_id} has identity_source without source_revision")
        root = source_roots[root_id]
        runtime_expectation = item.get("runtime_expectation")
        if identity_source == "runner_recorded" and runtime_expectation is None:
            raise ContractError(
                f"runner-recorded evidence {evidence_id} lacks runtime_expectation")
        if runtime_expectation is not None and item.get("kind") != "runtime_report":
            raise ContractError(
                f"non-runtime evidence {evidence_id} has runtime_expectation")
        if runtime_expectation is not None:
            claim_targets = set(_validate_string_list(
                f"evidence {evidence_id}.runtime_expectation.claim_targets",
                runtime_expectation.get("claim_targets"), nonempty=True))
            unknown_targets = claim_targets - set(targets)
            if unknown_targets:
                raise ContractError(
                    f"evidence {evidence_id} claims unknown targets "
                    f"{sorted(unknown_targets)}")
            inapplicable_targets = {
                target_id for target_id in claim_targets
                if not targets[target_id].get("applicable")
            }
            if inapplicable_targets:
                raise ContractError(
                    f"evidence {evidence_id} claims inapplicable targets "
                    f"{sorted(inapplicable_targets)}")
            target_case_counts = runtime_expectation.get("target_case_counts")
            if (not isinstance(target_case_counts, dict) or not target_case_counts
                    or any(not isinstance(key, str) or not key
                           for key in target_case_counts)
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or value < 1 for value in target_case_counts.values())):
                raise ContractError(
                    f"evidence {evidence_id} has invalid expected target counts")
            unknown_count_targets = set(target_case_counts) - set(targets)
            if unknown_count_targets:
                raise ContractError(
                    f"evidence {evidence_id} counts unknown targets "
                    f"{sorted(unknown_count_targets)}")
            inapplicable_count_targets = {
                target_id for target_id in target_case_counts
                if not targets[target_id].get("applicable")
            }
            if inapplicable_count_targets:
                raise ContractError(
                    f"evidence {evidence_id} counts inapplicable targets "
                    f"{sorted(inapplicable_count_targets)}")
            if not claim_targets <= set(target_case_counts):
                raise ContractError(
                    f"evidence {evidence_id} claims targets absent from its "
                    "expected case counts")
            claimed_capabilities = set(_validate_string_list(
                f"evidence {evidence_id}.runtime_expectation.capability_ids",
                runtime_expectation.get("capability_ids"), nonempty=True))
            unknown_capabilities = claimed_capabilities - set(capabilities)
            if unknown_capabilities:
                raise ContractError(
                    f"evidence {evidence_id} claims unknown capabilities "
                    f"{sorted(unknown_capabilities)}")
            incompatible_claims = {
                f"{target_id}/{capability_id}"
                for target_id in claim_targets
                for capability_id in claimed_capabilities
                if target_id not in capabilities[capability_id].get("target_results", {})
            }
            if incompatible_claims:
                raise ContractError(
                    f"evidence {evidence_id} claims incompatible target/capability pairs "
                    f"{sorted(incompatible_claims)}")
        if root_id == "artifacts" and runtime_expectation is not None:
            artifact_path = REPO_ROOT / root["path"] / item["path"]
            if not artifact_path.is_file():
                if require_local_artifacts:
                    raise ContractError(
                        f"required local artifact is missing: "
                        f"{artifact_path.relative_to(REPO_ROOT)}")
            else:
                artifact = _load_json(artifact_path)
                if (identity_source == "runner_recorded"
                        and artifact.get("git_commit") != source_revision):
                    raise ContractError(
                        f"evidence {evidence_id} source_revision disagrees with "
                        f"{artifact_path.relative_to(REPO_ROOT)}")
                if (identity_source == "runner_recorded"
                        and artifact.get("implementation_worktree_dirty") is not False):
                    raise ContractError(
                        f"evidence {evidence_id} was not recorded from a clean "
                        "implementation worktree")
                if (identity_source == "runner_recorded"
                        and artifact.get("source_identity_scope")
                        != "tracked files excluding research/**"):
                    raise ContractError(
                        f"evidence {evidence_id} has an unknown implementation identity scope")
                _validate_runtime_report(evidence_id, artifact, runtime_expectation)
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
                if status in {"passed", "historical_passed", "failed"} and not ids:
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} is {status} without evidence")
                if status == "historical_passed" and stage_name != "pcie_numeric_passed":
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} uses historical_passed; "
                        "only stale PCIe evidence may use that non-authorizing status")
                if (status == "historical_passed"
                        and not stage.get("scope", "").startswith("Historical ")):
                    raise ContractError(
                        f"{capability_id}/{target_id}/{stage_name} must label its "
                        "non-authorizing scope as Historical")
                if stage_name in {"cmodel_numeric_passed", "pcie_numeric_passed"}:
                    expected_runtime_mode = (
                        "cmodel" if stage_name == "cmodel_numeric_passed" else "pcie")
                    qualifying_runtime_ids = [
                        evidence_id for evidence_id in ids
                        if evidence[evidence_id].get("kind") == "runtime_report"
                        and evidence[evidence_id].get("runtime_expectation", {}).get(
                            "runtime_mode") == expected_runtime_mode
                        and target_id in evidence[evidence_id].get(
                            "runtime_expectation", {}).get("claim_targets", [])
                        and capability_id in evidence[evidence_id].get(
                            "runtime_expectation", {}).get("capability_ids", [])
                    ]
                    current_runtime_ids = [
                        evidence_id for evidence_id in qualifying_runtime_ids
                        if evidence[evidence_id].get("source_revision")
                        == validated_source_revision
                        and evidence[evidence_id].get("identity_source") == "runner_recorded"
                    ]
                    if status == "passed" and not current_runtime_ids:
                        raise ContractError(
                            f"{capability_id}/{target_id}/{stage_name} is passed "
                            "without runner-recorded evidence for "
                            "validated_source_revision")
                    if status == "historical_passed":
                        if not qualifying_runtime_ids:
                            raise ContractError(
                                f"{capability_id}/{target_id}/{stage_name} lacks "
                                "machine-checkable historical PCIe runtime evidence")
                        if current_runtime_ids:
                            raise ContractError(
                                f"{capability_id}/{target_id}/{stage_name} is historical "
                                "despite current runner-recorded evidence")
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
            if (verification["pcie_numeric_passed"]["status"] == "historical_passed"
                    and verification["cmodel_numeric_passed"]["status"] != "passed"):
                raise ContractError(
                    f"{capability_id}/{target_id} has historical PCIe evidence without "
                    "current CModel proof")
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
    applicable_targets = sum(
        target.get("applicable") is True for target in targets.values())
    return (
        len(targets), applicable_targets, len(operations),
        len(capabilities), len(evidence),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-local-artifacts",
        action="store_true",
        help=("require every runtime report with machine-checkable assertions to "
              "exist below ignored research/artifacts"),
    )
    args = parser.parse_args()
    try:
        target_entries, applicable_targets, operations, capabilities, evidence = validate(
            require_local_artifacts=args.require_local_artifacts)
    except ContractError as error:
        print(f"contract validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "contract validation passed: "
        f"{target_entries} target entries ({applicable_targets} applicable), "
        f"{operations} operations, "
        f"{capabilities} capabilities, {evidence} evidence entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
