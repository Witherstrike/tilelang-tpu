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
_SEMANTIC_EXTERN = re.compile(r'"(tl\.(?:tpu|tpukernel)\.[a-z0-9_]+)"')

_PORTABLE_EXTERNS = {
    "tl.tpu.embedding",
    "tl.tpu.add_scalar",
    "tl.tpu.mul_scalar",
    "tl.tpu.rsqrt",
    "tl.tpu.reduce_sum",
    "tl.tpu.reduce_max",
    "tl.tpu.exp",
    "tl.tpu.sigmoid",

    "tl.tpu.add",
    "tl.tpu.copy",
    "tl.tpu.div",
    "tl.tpu.fill",
    "tl.tpu.gemm",
    "tl.tpu.max",
    "tl.tpu.mul",
    "tl.tpu.sub",
}
_TPUKERNEL_EXTERNS = {
    "tl.tpukernel.gather",
    "tl.tpukernel.rope_add",
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


def _validate_contract_without_local_artifacts(monkeypatch, validator, schema, contract):
    """Exercise the checks that remain authoritative in a fresh clone."""

    real_load = validator._load_json
    real_is_file = validator.Path.is_file
    artifacts_root = validator.REPO_ROOT / "research/artifacts"

    def load_contract_override(path):
        if path == validator.CONTRACT_PATH:
            return contract
        if path == validator.SCHEMA_PATH:
            return schema
        return real_load(path)

    def hide_local_artifacts(path):
        try:
            path.relative_to(artifacts_root)
        except ValueError:
            return real_is_file(path)
        return False

    monkeypatch.setattr(validator, "_load_json", load_contract_override)
    monkeypatch.setattr(validator.Path, "is_file", hide_local_artifacts)
    return validator.validate()


def _literal_assignment(relative_path, name):
    path = _REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == name for target in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name} in {relative_path}")


def test_semantic_extern_registry_is_isomorphic_across_compiler_layers():
    """An extern is valid only when every compiler boundary owns it."""
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS

    assert _semantic_externs("tilelang/language/customize.py") == expected
    assert _semantic_externs("tilelang/engine/lower.py") == expected
    assert _semantic_externs("src/transform/address_assign.cc") == expected
    assert _semantic_externs("src/target/codegen_tpu.cc") == _PORTABLE_EXTERNS
    assert _semantic_externs("src/target/codegen_tpukernel.cc") == (_TPUKERNEL_EXTERNS | {
        "tl.tpu.embedding",
        "tl.tpu.add_scalar",
        "tl.tpu.mul_scalar",
        "tl.tpu.rsqrt",
        "tl.tpu.reduce_sum",
        "tl.tpu.reduce_max",
        "tl.tpu.exp",
        "tl.tpu.sigmoid",
    })

    contract_externs = {
        symbol for operation in _machine_contract()["operations"]
        for symbol in operation["internal_symbols"]
    }
    assert contract_externs == expected


def test_every_semantic_extern_has_exact_region_operand_positions():
    expected = _PORTABLE_EXTERNS | _TPUKERNEL_EXTERNS
    positions = _literal_assignment("tilelang/engine/lower.py", "_TPU_SEMANTIC_REGION_ARGS")

    assert set(positions) == expected
    assert all(
        tuple(indices) == tuple(range(1,
                                      len(indices) + 1)) for indices in positions.values())


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
         if "runtime_expectation" in item)["runtime_expectation"].pop("target_case_counts")
    with pytest.raises(validator.ContractError, match="lacks required fields"):
        validator._validate_json_schema(missing_target_counts, schema, schema, "contract")

    unsupported_nested_schema = copy.deepcopy(schema)
    unsupported_nested_schema["properties"]["source_roots"]["additionalProperties"]["properties"][
        "path"]["not"]["futureKeyword"] = True
    with pytest.raises(validator.SchemaDefinitionError, match="unsupported keywords"):
        validator._validate_json_schema(contract, unsupported_nested_schema,
                                        unsupported_nested_schema, "contract")


@pytest.mark.parametrize(
    "mutate_schema",
    [
        lambda schema: schema["$defs"].update({"unusedFuture": {
            "futureKeyword": True
        }}),
        lambda schema: schema["properties"].update({"unusedFuture": {
            "futureKeyword": True
        }}),
        lambda schema: schema["$defs"]["target"]["allOf"][0].update({"if": []}),
    ],
)
def test_machine_contract_schema_definition_fails_closed(mutate_schema):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    mutate_schema(schema)

    with pytest.raises(validator.SchemaDefinitionError):
        validator._validate_json_schema(contract, schema, schema, "contract")


def test_runtime_case_counts_are_static_without_local_artifacts(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    runtime_evidence = next(
        item for item in contract["evidence"] if item["id"] == "artifact.cmodel-rv-core-region-abi")
    runtime_evidence["runtime_expectation"]["case_count"] += 1

    with pytest.raises(validator.ContractError, match="case_count disagrees"):
        _validate_contract_without_local_artifacts(monkeypatch, validator, schema, contract)


def _mixed_backend_runtime_fixture():
    expectation = {
        "runtime_mode":
            "cmodel",
        "complete":
            True,
        "case_count":
            2,
        "all_cases_passed":
            True,
        "claim_targets": ["sg2260e.tpukernel", "sg2260e.rv"],
        "capability_ids": ["add.fp16-bf16.numeric"],
        "target_case_counts": {
            "sg2260e.tpukernel": 1,
            "sg2260e.rv": 1,
        },
        "required_case_ids": [
            "sg2260e/tpukernel/elementwise-add.float16",
            "sg2260e/rv/elementwise-add.float16",
        ],
    }
    artifact = {
        "runtime_mode":
            "cmodel",
        "complete":
            True,
        "results": [
            {
                "key": "cmodel/sg2260e/tpukernel/elementwise-add.float16",
                "chip": "sg2260e",
                "programming_model": "tpukernel",
                "numeric": {
                    "runtime_mode": "cmodel"
                },
                "case": {
                    "case_id": "elementwise-add.float16"
                },
                "status": "passed",
            },
            {
                "key": "cmodel/sg2260e/rv/elementwise-add.float16",
                "chip": "sg2260e",
                "programming_model": "rv",
                "numeric": {
                    "runtime_mode": "cmodel"
                },
                "case": {
                    "case_id": "elementwise-add.float16"
                },
                "status": "passed",
            },
        ],
    }
    return artifact, expectation


def test_runtime_report_accepts_per_result_programming_models():
    validator = _contract_validator_module()
    artifact, expectation = _mixed_backend_runtime_fixture()

    validator._validate_runtime_report("artifact.demo", artifact, expectation)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact["results"][1].update({"programming_model": "future"}),
         "invalid programming model"),
        (lambda artifact: artifact["results"][1].update({"runtime_mode": "pcie"}),
         "runtime mode disagrees"),
        (lambda artifact: artifact["results"][1].update(
            {"key": "cmodel/sg2260e/tpukernel/elementwise-add.float16"}), "key disagrees"),
        (lambda artifact: artifact["results"][1].pop("key"), "lacks a key"),
    ],
)
def test_runtime_report_rejects_invalid_mixed_backend_identity(mutate, message):
    validator = _contract_validator_module()
    artifact, expectation = _mixed_backend_runtime_fixture()
    mutate(artifact)

    with pytest.raises(validator.ContractError, match=message):
        validator._validate_runtime_report("artifact.demo", artifact, expectation)


def test_runtime_report_rejects_top_level_and_result_backend_disagreement():
    validator = _contract_validator_module()
    artifact, expectation = _mixed_backend_runtime_fixture()
    artifact["programming_model"] = "tpukernel"

    with pytest.raises(validator.ContractError, match="programming model disagrees"):
        validator._validate_runtime_report("artifact.demo", artifact, expectation)


def test_runtime_report_rejects_duplicate_result_keys():
    validator = _contract_validator_module()
    artifact, expectation = _mixed_backend_runtime_fixture()
    artifact["results"][1].update({
        "programming_model": "tpukernel",
        "key": artifact["results"][0]["key"],
    })

    with pytest.raises(validator.ContractError, match="duplicate result keys"):
        validator._validate_runtime_report("artifact.demo", artifact, expectation)


def test_required_runtime_case_belongs_to_a_claimed_target_without_artifacts(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    runtime_evidence = next(
        item for item in contract["evidence"] if item["id"] == "artifact.cmodel-rv-core-region-abi")
    runtime_evidence["runtime_expectation"]["required_case_ids"] = ["bm1690/tpukernel/matmul"]

    with pytest.raises(validator.ContractError, match="claimed target"):
        _validate_contract_without_local_artifacts(monkeypatch, validator, schema, contract)


@pytest.mark.parametrize(
    "case_id",
    [
        "bm1690/tpukernel/matmul",
        "sg2260e/tpukernel/e4m3/add",
        "sg2260e/rv/group_1/sub-group/case.name",
    ],
)
def test_runtime_case_id_accepts_safe_paths_with_at_least_three_segments(case_id):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    schema_pattern = schema["$defs"]["runtimeExpectation"]["properties"]["required_case_ids"][
        "items"]["pattern"]

    assert schema_pattern == validator.RUNTIME_CASE_ID.pattern
    assert validator.RUNTIME_CASE_ID.fullmatch(case_id)


@pytest.mark.parametrize(
    "case_id",
    [
        "sg2260e/tpukernel",
        "/sg2260e/tpukernel/add",
        "sg2260e/tpukernel/add/",
        "sg2260e/tpukernel//add",
        "sg2260e/tpukernel/fp8/add?",
        "SG2260E/tpukernel/fp8/add",
        "sg2260e/tpukernel/../escape",
        "sg2260e/tpukernel/.",
        "sg2260e/tpukernel/..",
        "sg2260e/tpukernel/a/../../escape",
        "sg2260e/tpukernel/Case.Name",
        "sg2260e/tpukernel/-leading",
        "sg2260e/tpukernel/trailing-",
    ],
)
def test_runtime_case_id_rejects_short_empty_or_unsafe_paths(case_id):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    schema_pattern = schema["$defs"]["runtimeExpectation"]["properties"]["required_case_ids"][
        "items"]["pattern"]

    assert schema_pattern == validator.RUNTIME_CASE_ID.pattern
    assert validator.RUNTIME_CASE_ID.fullmatch(case_id) is None


@pytest.mark.parametrize("path", [r"..\escape", r"\\server\share"])
def test_contract_paths_reject_non_posix_separators(path):
    validator = _contract_validator_module()

    with pytest.raises(validator.ContractError, match="portable POSIX separators"):
        validator._check_portable_path("test path", path)


def test_validated_revision_must_exist_without_local_artifacts(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    phantom_revision = "0" * 40
    contract["validated_source_revision"] = phantom_revision
    for evidence in contract["evidence"]:
        if evidence.get("identity_source") == "runner_recorded":
            evidence["source_revision"] = phantom_revision

    with pytest.raises(validator.ContractError, match="commit .* is unavailable"):
        _validate_contract_without_local_artifacts(monkeypatch, validator, schema, contract)


def test_numeric_stage_cannot_borrow_another_target_runtime_artifact(monkeypatch):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    capability = next(
        item for item in contract["capabilities"] if item["id"] == "add.fp32-equal.numeric")
    stage = capability["target_results"]["sg2260e.rv"]["verification"]["cmodel_numeric_passed"]
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
    capability = next(item for item in contract["capabilities"] if item["id"] == "exp.float")
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
        ("capability_ids", "rope.float", "claims incompatible target/capability pairs"),
    ],
)
def test_runtime_evidence_claims_form_a_closed_contract_set(monkeypatch, field, phantom, message):
    validator = _contract_validator_module()
    schema = validator._load_json(validator.SCHEMA_PATH)
    contract = _machine_contract()
    runtime_evidence = next(
        item for item in contract["evidence"] if item["id"] == "artifact.cmodel-rv-core-region-abi")
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
    stage = next(result["verification"]["pcie_numeric_passed"]
                 for capability in contract["capabilities"]
                 for result in capability["target_results"].values()
                 if result["verification"]["pcie_numeric_passed"]["status"] == "passed")
    # The current contract no longer needs a historical stage.  Synthesize one
    # so this invariant remains covered even when every old PCIe result has
    # been superseded by same-revision evidence.
    stage["status"] = "historical_passed"
    stage["scope"] = "Historical " + stage["scope"]
    stage["scope"] = stage["scope"][len("Historical "):]

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
