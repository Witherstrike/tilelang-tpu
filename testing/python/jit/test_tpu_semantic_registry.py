# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Keep TPU compiler registries and the machine contract closed and aligned."""

import ast
import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


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


def _contract_validator_module():
    path = _REPOSITORY_ROOT / "research/tpu-op-contract/validate.py"
    spec = importlib.util.spec_from_file_location("tilelang_tpu_contract_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_assignment(relative_path, name):
    path = _REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == name
                    for target in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name} in {relative_path}")


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


def test_every_semantic_extern_has_exact_region_operand_positions():
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS
    positions = _literal_assignment(
        "tilelang/engine/lower.py", "_TPU_SEMANTIC_REGION_ARGS")

    assert set(positions) == expected
    assert all(tuple(indices) == tuple(range(1, len(indices) + 1))
               for indices in positions.values())


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


def test_machine_contract_schema_is_applied_by_standard_library_validator():
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()

    unknown_field = copy.deepcopy(contract)
    unknown_field["evidence"][0]["typo"] = True
    with pytest.raises(validator.ContractError, match="unknown field"):
        validator._validate_json_schema(unknown_field, schema, schema, "contract")

    bad_nested_type = copy.deepcopy(contract)
    bad_nested_type["scope"]["excluded_claims"] = "not-an-array"
    with pytest.raises(validator.ContractError, match="must have JSON type array"):
        validator._validate_json_schema(bad_nested_type, schema, schema, "contract")

    missing_target_counts = copy.deepcopy(contract)
    next(item for item in missing_target_counts["evidence"]
         if "runtime_expectation" in item)["runtime_expectation"].pop(
             "target_case_counts")
    with pytest.raises(validator.ContractError, match="lacks required fields"):
        validator._validate_json_schema(
            missing_target_counts, schema, schema, "contract")

    unsupported_nested_schema = copy.deepcopy(schema)
    unsupported_nested_schema["properties"]["source_roots"][
        "additionalProperties"]["properties"]["path"]["not"][
            "futureKeyword"] = True
    with pytest.raises(validator.SchemaDefinitionError, match="unsupported keywords"):
        validator._validate_json_schema(
            contract, unsupported_nested_schema, unsupported_nested_schema, "contract")


def test_numeric_stage_cannot_borrow_another_target_runtime_artifact(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    capability = next(
        item for item in contract["capabilities"]
        if item["id"] == "add.fp32-equal.numeric"
    )
    stage = capability["target_results"]["sg2260e.rv"]["verification"][
        "cmodel_numeric_passed"]
    stage["evidence_ids"] = ["test.numeric-worker", "artifact.cmodel-fp8-final"]

    real_load = validator._load_json

    def load_contract_override(path):
        if path == validator.CONTRACT_PATH:
            return contract
        if path == validator.SCHEMA_PATH:
            return schema
        return real_load(path)

    monkeypatch.setattr(validator, "_load_json", load_contract_override)
    with pytest.raises(validator.ContractError, match="runner-recorded evidence"):
        validator.validate()


def test_numeric_stage_cannot_borrow_same_target_wrong_capability(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    capability = next(
        item for item in contract["capabilities"] if item["id"] == "exp.float")
    stage = capability["target_results"]["sg2260e.tpukernel"]["verification"][
        "cmodel_numeric_passed"]
    stage["evidence_ids"] = ["test.tpukernel-ops-worker", "artifact.cmodel-core"]

    real_load = validator._load_json

    def load_contract_override(path):
        if path == validator.CONTRACT_PATH:
            return contract
        if path == validator.SCHEMA_PATH:
            return schema
        return real_load(path)

    monkeypatch.setattr(validator, "_load_json", load_contract_override)
    with pytest.raises(validator.ContractError, match="runner-recorded evidence"):
        validator.validate()


@pytest.mark.parametrize(
    ("field", "phantom", "message"),
    [
        ("claim_targets", "future-chip.rv", "claims unknown targets"),
        ("claim_targets", "bm1690.rv", "claims inapplicable targets"),
        ("capability_ids", "future-op.fp32", "claims unknown capabilities"),
        ("capability_ids", "exp.float", "claims incompatible target/capability pairs"),
    ],
)
def test_runtime_evidence_claims_form_a_closed_contract_set(
        monkeypatch, field, phantom, message):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    runtime_evidence = next(
        item for item in contract["evidence"] if "runtime_expectation" in item
    )
    runtime_evidence["runtime_expectation"][field].append(phantom)

    real_load = validator._load_json

    def load_contract_override(path):
        if path == validator.CONTRACT_PATH:
            return contract
        if path == validator.SCHEMA_PATH:
            return schema
        return real_load(path)

    monkeypatch.setattr(validator, "_load_json", load_contract_override)
    with pytest.raises(validator.ContractError, match=message):
        validator.validate()


def test_historical_pcie_scope_is_explicitly_non_authorizing(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    stage = next(
        result["verification"]["pcie_numeric_passed"]
        for capability in contract["capabilities"]
        for result in capability["target_results"].values()
        if result["verification"]["pcie_numeric_passed"]["status"]
        == "historical_passed"
    )
    stage["scope"] = stage["scope"].removeprefix("Historical ")

    real_load = validator._load_json

    def load_contract_override(path):
        if path == validator.CONTRACT_PATH:
            return contract
        if path == validator.SCHEMA_PATH:
            return schema
        return real_load(path)

    monkeypatch.setattr(validator, "_load_json", load_contract_override)
    with pytest.raises(validator.ContractError, match="label.*Historical"):
        validator.validate()
