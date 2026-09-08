# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pure-Python safety contracts for the public TPU demo matrix."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tpu_demo_ops_matrix as matrix
from tpu_demo.cases import case_by_id


def _execution_args(**overrides):
    values = {
        "runtime_mode": "pcie",
        "output_dir": Path("matrix-output"),
        "timeout": 30.0,
        "chip": "sg2260e",
        "programming_model": "tpukernel",
        "operations": ("matmul",),
        "dtypes": None,
        "case_ids": None,
        "device_id": 0,
        "allow_pcie": True,
        "allow_pcie_profile": True,
        "pcie_decoder_python": None,
        "pcie_decoder_pythonpath": (),
        "require_decoded_timing": False,
        "bm_cmodel_summary": Path("bm-summary.json"),
        "sg_cmodel_summary": Path("sg-summary.json"),
        "all_pcie_cases": False,
        "list_cases": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_args_accepts_explicit_scoped_pcie_request():
    matrix.validate_args(_execution_args())
    matrix.validate_args(
        _execution_args(operations=None, case_ids=None, all_pcie_cases=True))


@pytest.mark.parametrize(
    ("overrides", "diagnostic"),
    (
        ({"allow_pcie": False}, "--allow-pcie and --allow-pcie-profile"),
        ({"allow_pcie_profile": False}, "--allow-pcie and --allow-pcie-profile"),
        ({"chip": "bm1690"}, "explicit --chip sg2260e"),
        ({"device_id": 1}, "only --device-id 0"),
        ({"device_id": False}, "non-negative 32-bit"),
        ({"operations": None, "case_ids": None}, "explicit --case/--op filter"),
        ({"bm_cmodel_summary": None}, "--bm-cmodel-summary"),
        ({"sg_cmodel_summary": None}, "--bm-cmodel-summary"),
    ),
)
def test_validate_args_rejects_unscoped_or_unpromoted_pcie(overrides, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        matrix.validate_args(_execution_args(**overrides))


def test_validate_args_rejects_pcie_controls_in_cmodel_mode():
    with pytest.raises(RuntimeError, match="invalid in CModel mode"):
        matrix.validate_args(
            _execution_args(
                runtime_mode="cmodel",
                allow_pcie=True,
                allow_pcie_profile=False,
                device_id=None,
                bm_cmodel_summary=None,
                sg_cmodel_summary=None,
            ))


def test_validate_args_requires_explicit_single_chip_cmodel_stage():
    with pytest.raises(RuntimeError, match="one explicit --chip"):
        matrix.validate_args(
            _execution_args(
                runtime_mode="cmodel",
                chip=None,
                allow_pcie=False,
                allow_pcie_profile=False,
                device_id=None,
                bm_cmodel_summary=None,
                sg_cmodel_summary=None,
            ))


def test_rv_case_listing_excludes_tpukernel_only_composites():
    args = _execution_args(programming_model="rv", operations=None, case_ids=None)
    cases = matrix.selected_cases(args)

    assert len(cases) == 15
    assert all(case.supports_rv for case in cases)


def _board_payload():
    return {
        "card_num": 1,
        "chip_num": 1,
        "card0": {
            "card_idx": 0,
            "chip_num_of_card": 1,
            "chip0": {
                "chip_index_of_card": 0,
                "status": "Active",
                "tpu_util": "0%",
            },
        },
    }


def _set_path(payload, path, value):
    cursor = payload
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value


def test_validate_board_snapshot_accepts_exact_idle_single_card():
    payload = _board_payload()
    snapshot = matrix.validate_board_snapshot(payload, 0)

    assert snapshot == {
        "device_id": 0,
        "status": "Active",
        "card_key": "card0",
        "chip_count": 1,
        "chips": [payload["card0"]["chip0"]],
    }


@pytest.mark.parametrize(
    ("path", "value", "diagnostic"),
    (
        (("card_num",), 2, "exactly one visible card"),
        (("card_num",), True, "exactly one visible card"),
        (("chip_num",), 2, "exactly one visible card"),
        (("card0", "card_idx"), 1, "inconsistent card_idx"),
        (("card0", "chip_num_of_card"), 2, "invalid chip count"),
        (("card0", "chip0", "chip_index_of_card"), 1, "inconsistent chip index"),
        (("card0", "chip0", "status"), "active", "exact Active status"),
        (("card0", "chip0", "tpu_util"), 0, "not idle"),
        (("card0", "chip0", "tpu_util"), "1%", "not idle"),
    ),
)
def test_validate_board_snapshot_rejects_ambiguous_or_nonidle_state(
        path, value, diagnostic):
    payload = _board_payload()
    _set_path(payload, path, value)

    with pytest.raises(RuntimeError, match=diagnostic):
        matrix.validate_board_snapshot(payload, 0)


def test_validate_board_snapshot_rejects_unprovable_logical_device():
    with pytest.raises(RuntimeError, match="only for device 0"):
        matrix.validate_board_snapshot(_board_payload(), 1)


def _numeric_payload(case, *, chip="sg2260e", programming_model="tpukernel",
                     runtime_mode="cmodel"):
    return {
        "status": "passed",
        "operation": case.operation,
        "dtype": case.dtype,
        "chip": chip,
        "programming_model": programming_model,
        "runtime_mode": runtime_mode,
        "metrics": {"passed": True, "finite": True},
        "parameters": {"variant": case.variant},
    }


def test_demo_worker_payload_rejects_duplicate_markers(tmp_path):
    case = case_by_id("elementwise-add.float16")
    marker = matrix.RESULT_PREFIX + json.dumps(_numeric_payload(case))
    stdout = tmp_path / "worker.stdout.log"
    stdout.write_text(f"{marker}\n{marker}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="multiple machine-readable"):
        matrix.worker_payload(stdout)


def test_validate_numeric_identity_accepts_exact_variant_and_backend():
    case = case_by_id("flashattn.float16.weighted-keys")
    matrix.validate_numeric_identity(
        _numeric_payload(case),
        chip="sg2260e",
        programming_model="tpukernel",
        runtime_mode="cmodel",
        case=case,
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("operation", "matmul"),
        ("dtype", "float32"),
        ("chip", "bm1690"),
        ("programming_model", "rv"),
        ("runtime_mode", "pcie"),
    ),
)
def test_validate_numeric_identity_rejects_wrong_case_or_backend(field, wrong_value):
    case = case_by_id("flashattn.float16.weighted-keys")
    payload = _numeric_payload(case)
    payload[field] = wrong_value

    with pytest.raises(RuntimeError, match=field):
        matrix.validate_numeric_identity(
            payload,
            chip="sg2260e",
            programming_model="tpukernel",
            runtime_mode="cmodel",
            case=case,
        )


@pytest.mark.parametrize(
    "metrics",
    (
        {},
        {"passed": False, "finite": True},
        {"passed": True, "finite": False},
    ),
)
def test_validate_numeric_identity_rejects_nonpassing_or_nonfinite_metrics(metrics):
    case = case_by_id("flashattn.float16.weighted-keys")
    payload = _numeric_payload(case)
    payload["metrics"] = metrics

    with pytest.raises(RuntimeError, match="metrics|finite"):
        matrix.validate_numeric_identity(
            payload,
            chip="sg2260e",
            programming_model="tpukernel",
            runtime_mode="cmodel",
            case=case,
        )


def test_validate_numeric_identity_rejects_missing_variant_identity():
    case = case_by_id("flashattn.float16.weighted-keys")
    payload = _numeric_payload(case)
    payload["parameters"] = {}

    with pytest.raises(RuntimeError, match="scheduled variant"):
        matrix.validate_numeric_identity(
            payload,
            chip="sg2260e",
            programming_model="tpukernel",
            runtime_mode="cmodel",
            case=case,
        )


def _toolchain_identity(runtime_mode):
    identity = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "ppl_project_root": "/sdk/ppl-1.7",
        "compiler_runtime": {"tilelang_library": {"sha256": "tilelang"}},
        "ppl_common": {"chip_map": {"sha256": "chip-map"}},
        "cmodel": {"runtime_tree": {"sha256": "cmodel-runtime"}},
        "chips": {
            "bm1690": {"kernel_include_tree": {"sha256": "bm1690"}},
            "sg2260e": {"kernel_include_tree": {"sha256": "sg2260e"}},
        },
        "runtime_mode": runtime_mode,
    }
    if runtime_mode == "pcie":
        identity["pcie"] = {"installed_runtime_library": {"sha256": "board-runtime"}}
    return identity


def _promotion_result(case, chip, programming_model):
    return {
        "status": "passed",
        "key": f"cmodel/{chip}/{programming_model}/{case.case_id}",
        "chip": chip,
        "programming_model": programming_model,
        "raw_instruction_count": 1,
        "numeric": _numeric_payload(
            case,
            chip=chip,
            programming_model=programming_model,
            runtime_mode="cmodel",
        ),
    }


def _promotion_summary(case, chip, programming_model):
    result = _promotion_result(case, chip, programming_model)
    return {
        "schema_version": matrix.SCHEMA_VERSION,
        "matrix_kind": matrix.MATRIX_KIND,
        "status": "passed",
        "complete": True,
        "runtime_mode": "cmodel",
        "implementation_worktree_dirty": False,
        "git_commit": "0123456789abcdef",
        "source_state_sha256": "source-state",
        "started_at": "2026-09-08T00:00:00+00:00",
        "finished_at": "2026-09-08T00:01:00+00:00",
        "toolchain_identity": _toolchain_identity("cmodel"),
        "scheduled_case_count": 1,
        "completed_case_count": 1,
        "passed_case_count": 1,
        "failed_case_count": 0,
        "target_scope": [{
            "chip": chip,
            "programming_model": programming_model,
        }],
        "scheduled": [{
            "chip": chip,
            "programming_model": programming_model,
            "case": case.to_json(),
        }],
        "results": [result],
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def promotion(tmp_path, monkeypatch):
    case = case_by_id("matmul.float16")
    current = {
        "implementation_worktree_dirty": False,
        "git_commit": "0123456789abcdef",
        "source_state_sha256": "source-state",
    }
    monkeypatch.setattr(matrix, "git_source_identity", lambda _repo_root: dict(current))

    bm = _promotion_summary(case, "bm1690", "tpukernel")
    sg = _promotion_summary(case, "sg2260e", "rv")
    sg["started_at"] = "2026-09-08T00:02:00+00:00"
    sg["finished_at"] = "2026-09-08T00:03:00+00:00"
    bm_path = tmp_path / "bm.json"
    sg_path = tmp_path / "sg.json"
    _write_json(bm_path, bm)
    _write_json(sg_path, sg)
    args = SimpleNamespace(bm_cmodel_summary=bm_path, sg_cmodel_summary=sg_path)
    return SimpleNamespace(
        args=args,
        bm=bm,
        sg=sg,
        bm_path=bm_path,
        sg_path=sg_path,
        case=case,
        current=current,
        scheduled=(("sg2260e", "rv", case),),
        toolchain=_toolchain_identity("pcie"),
        pcie_started_at="2026-09-08T00:04:00+00:00",
        repo_root=tmp_path,
    )


def _validate_promotion(promotion):
    return matrix.validate_pcie_promotion(
        promotion.repo_root,
        promotion.args,
        promotion.scheduled,
        promotion.toolchain,
        promotion.pcie_started_at,
    )


def test_promotion_accepts_matching_clean_complete_evidence(promotion):
    evidence = _validate_promotion(promotion)

    assert evidence["git_commit"] == promotion.current["git_commit"]
    assert evidence["validated_case_count"] == 1
    assert len(evidence["bm1690_summary_sha256"]) == 64
    assert len(evidence["sg2260e_summary_sha256"]) == 64


@pytest.mark.parametrize("location", ("current", "bm-summary"))
def test_promotion_rejects_dirty_source(promotion, location):
    if location == "current":
        promotion.current["implementation_worktree_dirty"] = True
    else:
        promotion.bm["implementation_worktree_dirty"] = True
        _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="clean worktree|worktree to be clean"):
        _validate_promotion(promotion)


@pytest.mark.parametrize("location", ("current", "sg-summary"))
def test_promotion_rejects_missing_or_mismatched_commit(promotion, location):
    if location == "current":
        promotion.current.pop("git_commit")
        diagnostic = "verifiable Git commit"
    else:
        promotion.sg["git_commit"] = "different-commit"
        _write_json(promotion.sg_path, promotion.sg)
        diagnostic = "commit does not match"

    with pytest.raises(RuntimeError, match=diagnostic):
        _validate_promotion(promotion)


@pytest.mark.parametrize("location", ("current", "bm-summary"))
def test_promotion_rejects_missing_source_state(promotion, location):
    if location == "current":
        promotion.current.pop("source_state_sha256")
        diagnostic = "source-state digest"
    else:
        promotion.bm.pop("source_state_sha256")
        _write_json(promotion.bm_path, promotion.bm)
        diagnostic = "source state does not match"

    with pytest.raises(RuntimeError, match=diagnostic):
        _validate_promotion(promotion)


def test_promotion_rejects_missing_toolchain_identity(promotion):
    promotion.sg.pop("toolchain_identity")
    _write_json(promotion.sg_path, promotion.sg)

    with pytest.raises(RuntimeError, match="no toolchain identity"):
        _validate_promotion(promotion)


def test_promotion_rejects_mismatched_toolchain_identity(promotion):
    promotion.bm["toolchain_identity"]["ppl_project_root"] = "/different-sdk"
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="does not match current toolchain"):
        _validate_promotion(promotion)


def test_promotion_rejects_incomplete_pcie_toolchain_identity(promotion):
    promotion.toolchain.pop("pcie")

    with pytest.raises(RuntimeError, match="complete PCIe toolchain identity"):
        _validate_promotion(promotion)


def test_promotion_rejects_missing_required_case(promotion):
    promotion.bm["results"] = []
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="scheduled work does not match"):
        _validate_promotion(promotion)


def test_promotion_rejects_duplicate_result_keys(promotion):
    promotion.bm["results"].append(dict(promotion.bm["results"][0]))
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="scheduled work does not match"):
        _validate_promotion(promotion)


def test_promotion_rejects_same_file_for_both_chip_stages(promotion):
    promotion.args.sg_cmodel_summary = promotion.bm_path

    with pytest.raises(RuntimeError, match="must be different files"):
        _validate_promotion(promotion)


def test_promotion_rejects_mixed_chip_stage(promotion):
    promotion.bm["target_scope"].append({
        "chip": "sg2260e",
        "programming_model": "rv",
    })
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="mixes target"):
        _validate_promotion(promotion)


def test_promotion_requires_bm_stage_to_finish_before_sg_stage(promotion):
    promotion.bm["finished_at"] = "2026-09-08T00:04:00+00:00"
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="BM1690 must finish before SG2260E"):
        _validate_promotion(promotion)


@pytest.mark.parametrize(
    ("stage", "started_at", "finished_at", "diagnostic"),
    (
        ("bm", "2026-09-08T00:02:00+00:00", "2026-09-08T00:01:00+00:00",
         "BM1690 starts after it finishes"),
        ("sg", "2026-09-08T00:03:00+00:00", "2026-09-08T00:02:00+00:00",
         "SG2260E starts after it finishes"),
    ),
)
def test_promotion_rejects_reversed_stage_interval(
        promotion, stage, started_at, finished_at, diagnostic):
    payload = getattr(promotion, stage)
    payload["started_at"] = started_at
    payload["finished_at"] = finished_at
    _write_json(getattr(promotion, f"{stage}_path"), payload)

    with pytest.raises(RuntimeError, match=diagnostic):
        _validate_promotion(promotion)


def test_promotion_requires_sg_stage_to_finish_before_pcie(promotion):
    promotion.pcie_started_at = "2026-09-08T00:02:30+00:00"

    with pytest.raises(RuntimeError, match="SG2260E must finish before PCIe"):
        _validate_promotion(promotion)


@pytest.mark.parametrize("field", ("matrix_kind", "schema_version", "target_scope"))
def test_promotion_rejects_wrong_matrix_contract(promotion, field):
    if field == "matrix_kind":
        promotion.sg[field] = "another_matrix"
        diagnostic = "matrix_kind"
    elif field == "schema_version":
        promotion.sg[field] = matrix.SCHEMA_VERSION + 1
        diagnostic = "schema_version"
    else:
        promotion.sg[field] = [{"chip": "bm1690", "programming_model": "tpukernel"}]
        diagnostic = "single-chip scope"
    _write_json(promotion.sg_path, promotion.sg)

    with pytest.raises(RuntimeError, match=diagnostic):
        _validate_promotion(promotion)


def test_promotion_rejects_inconsistent_summary_counts(promotion):
    promotion.sg["completed_case_count"] = 0
    _write_json(promotion.sg_path, promotion.sg)

    with pytest.raises(RuntimeError, match="inconsistent completed_case_count"):
        _validate_promotion(promotion)


def test_promotion_rejects_result_outside_declared_scope(promotion):
    promotion.bm["results"].append(
        _promotion_result(promotion.case, "sg2260e", "rv"))
    _write_json(promotion.bm_path, promotion.bm)

    with pytest.raises(RuntimeError, match="result outside target_scope"):
        _validate_promotion(promotion)


def test_pin_worker_environment_removes_shared_checkout_imports(tmp_path):
    repo = tmp_path / "repo"
    snapshot = tmp_path / "scratch/execution-source"
    external = tmp_path / "external-packages"
    tilelang_library = tmp_path / "native/tilelang/libtilelang_module.so"
    tvm_library = tmp_path / "native/tvm/libtvm.so"
    for path in (repo, snapshot / "3rdparty/tvm/python", external,
                 tilelang_library.parent, tvm_library.parent):
        path.mkdir(parents=True)
    tilelang_library.write_bytes(b"tilelang")
    tvm_library.write_bytes(b"tvm")
    environment = {
        "PYTHONPATH": os.pathsep.join(
            (str(repo), str(repo / "3rdparty/tvm/python"), str(external))),
        "LD_LIBRARY_PATH": str(external),
        "PPL_PROJECT_ROOT": "/sdk/ppl-1.7",
    }
    toolchain = {
        "compiler_runtime": {
            "tilelang_library": {"resolved_path": str(tilelang_library.resolve())},
            "tvm_library": {"resolved_path": str(tvm_library.resolve())},
        },
    }

    pinned = matrix.pin_worker_environment(
        environment,
        repo_root=repo.resolve(),
        snapshot_root=snapshot.resolve(),
        toolchain_identity=toolchain,
    )

    entries = pinned["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [str(snapshot.resolve()),
                           str((snapshot / "3rdparty/tvm/python").resolve())]
    assert str(external.resolve()) in entries
    assert str(repo.resolve()) not in entries
    assert pinned["TVM_IMPORT_PYTHON_PATH"] == entries[1]
    assert pinned["TL_TEMPLATE_PATH"] == str((snapshot / "src").resolve())
    assert pinned["TILELANG_LIBRARY_PATH"] == str(tilelang_library.parent.resolve())
    assert pinned["TVM_LIBRARY_PATH"] == str(tvm_library.parent.resolve())
    assert pinned["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(tilelang_library.parent.resolve()),
        str(tvm_library.parent.resolve()),
        str(external.resolve()),
    ]


@pytest.mark.parametrize(
    "toolchain",
    (
        {},
        {"compiler_runtime": {"tilelang_library": {"resolved_path": "relative.so"}}},
    ),
)
def test_pin_worker_environment_rejects_unusable_native_identity(tmp_path, toolchain):
    repo = tmp_path / "repo"
    snapshot = tmp_path / "snapshot"
    (snapshot / "3rdparty/tvm/python").mkdir(parents=True)
    repo.mkdir()

    with pytest.raises(RuntimeError, match="toolchain identity"):
        matrix.pin_worker_environment(
            {},
            repo_root=repo.resolve(),
            snapshot_root=snapshot.resolve(),
            toolchain_identity=toolchain,
        )


def test_worker_environment_separates_cmodel_and_board_runtime(monkeypatch, tmp_path):
    ppl = tmp_path / "ppl"
    sdk_runtime = ppl / "deps/runtime/tpuv7-runtime/lib"
    board_runtime = tmp_path / "installed-runtime/lib"
    external = tmp_path / "external/lib"
    perfai = tmp_path / "PerfAI"
    for path in (sdk_runtime, board_runtime, external, perfai):
        path.mkdir(parents=True)
    monkeypatch.setenv("PPL_PROJECT_ROOT", str(ppl))
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", os.pathsep.join((str(sdk_runtime), str(external))))
    monkeypatch.setenv("TILELANG_TPU_PCIE_RUNTIME_PATH", str(board_runtime))
    monkeypatch.setenv("PPL_PERFAI_ROOT", str(perfai))

    cmodel = matrix.worker_environment(tmp_path, "cmodel", None)
    pcie = matrix.worker_environment(tmp_path, "pcie", 0)

    assert cmodel["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(sdk_runtime.resolve())
    assert pcie["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(board_runtime.resolve())
    assert str(sdk_runtime.resolve()) not in pcie["LD_LIBRARY_PATH"].split(os.pathsep)
    assert pcie["TILELANG_TPU_PCIE_RUNTIME_PATH"] == str(board_runtime.resolve())
    assert cmodel["PPL_PERFAI_ROOT"] == str(perfai.resolve())
    assert pcie["PPL_PERFAI_ROOT"] == str(perfai.resolve())


def test_materialize_execution_snapshot_pins_parent_and_tvm_commits(
        monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "3rdparty/tvm").mkdir(parents=True)
    destination = tmp_path / "snapshot"
    archives = []

    def fake_archive(repository, commit, output, paths):
        archives.append((repository, commit, output, paths))
        output.mkdir(parents=True, exist_ok=True)

    def fake_run(command, **kwargs):
        del kwargs
        if command[1] == "ls-tree":
            return SimpleNamespace(
                returncode=0, stdout="160000 commit tvmcommit\t3rdparty/tvm\n", stderr="")
        if command[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout="tvmcommit\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(matrix, "_extract_git_archive", fake_archive)
    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    identity = matrix.materialize_execution_snapshot(repo, destination, "parentcommit")

    assert identity == {
        "kind": "git-archive",
        "git_commit": "parentcommit",
        "tvm_git_commit": "tvmcommit",
        "read_only": True,
    }
    assert archives[0][0:2] == (repo, "parentcommit")
    assert archives[0][3] == (
        "VERSION", "tilelang", "tpu_demo", "testing/python/jit", "src")
    assert archives[1][0:2] == (repo / "3rdparty/tvm", "tvmcommit")
    assert archives[1][3] == ("python",)
    assert destination.stat().st_mode & 0o222 == 0


def test_execution_snapshot_is_read_only_and_scratch_remains_removable(tmp_path):
    scratch = tmp_path / "scratch"
    snapshot = scratch / "execution-source"
    nested = snapshot / "tilelang"
    nested.mkdir(parents=True)
    source = nested / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")

    matrix._make_tree_read_only(snapshot)

    assert snapshot.stat().st_mode & 0o222 == 0
    assert nested.stat().st_mode & 0o222 == 0
    assert source.stat().st_mode & 0o222 == 0
    with pytest.raises(PermissionError):
        source.write_text("value = 2\n", encoding="utf-8")

    matrix.remove_execution_scratch(scratch)
    assert not scratch.exists()


def test_source_identity_guard_accepts_exact_snapshot_and_rejects_change(
        monkeypatch, tmp_path):
    current = {
        "implementation_worktree_dirty": False,
        "git_commit": "abc",
        "source_state_sha256": "state",
    }
    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: dict(current))
    matrix.assert_source_identity_unchanged(tmp_path, current)

    current["implementation_worktree_dirty"] = True
    with pytest.raises(RuntimeError, match="refusing the next PCIe launch"):
        matrix.assert_source_identity_unchanged(tmp_path, {
            "git_commit": "abc", "source_state_sha256": "state"})


def test_toolchain_identity_guard_requires_exact_content(monkeypatch):
    expected = _toolchain_identity("pcie")
    current = dict(expected)
    monkeypatch.setattr(matrix, "toolchain_identity", lambda _environment, _mode: current)
    matrix.assert_toolchain_identity_unchanged({}, expected)

    current = {**expected, "pcie": {"installed_runtime_library": {"sha256": "changed"}}}
    with pytest.raises(RuntimeError, match="refusing the next PCIe launch"):
        matrix.assert_toolchain_identity_unchanged({}, expected)


def test_main_refuses_even_an_existing_empty_output_directory(monkeypatch, tmp_path):
    output = tmp_path / "claimed"
    output.mkdir()
    monkeypatch.setattr(
        matrix,
        "parse_args",
        lambda: _execution_args(
            runtime_mode="cmodel",
            output_dir=output,
            chip="bm1690",
            programming_model="tpukernel",
            operations=None,
            device_id=None,
            allow_pcie=False,
            allow_pcie_profile=False,
            bm_cmodel_summary=None,
            sg_cmodel_summary=None,
        ),
    )

    with pytest.raises(RuntimeError, match="refusing to reuse"):
        matrix.main()


def test_failed_board_postflight_is_not_retried(monkeypatch, tmp_path):
    import importlib

    tilelang_jit = importlib.import_module("tilelang.jit")

    case = case_by_id("matmul.float16")
    args = _execution_args()
    output = tmp_path / "output"
    output.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = {
        "implementation_worktree_dirty": False,
        "git_commit": "0123456789abcdef",
        "source_state_sha256": "source-state",
    }
    toolchain = _toolchain_identity("pcie")
    toolchain["pcie"]["tpu_smi"] = {"path": "/sbin/tpu-smi", "sha256": "smi"}

    monkeypatch.setattr(matrix, "git_source_identity", lambda _root: dict(source))
    monkeypatch.setattr(matrix, "toolchain_identity", lambda _env, _mode: dict(toolchain))
    monkeypatch.setattr(
        matrix, "pin_native_worker_libraries",
        lambda environment, _identity: dict(environment))
    monkeypatch.setattr(
        matrix, "validate_pcie_promotion",
        lambda *_args: {"git_commit": source["git_commit"]})
    monkeypatch.setattr(
        matrix, "materialize_execution_snapshot",
        lambda *_args: {"kind": "git-archive", "read_only": True})
    monkeypatch.setattr(
        matrix, "pin_worker_environment",
        lambda environment, **_kwargs: dict(environment))
    monkeypatch.setattr(matrix, "assert_source_identity_unchanged", lambda *_args: None)
    monkeypatch.setattr(matrix, "assert_toolchain_identity_unchanged", lambda *_args: None)
    monkeypatch.setattr(
        matrix,
        "numeric_payload",
        lambda _path: _numeric_payload(
            case, chip="sg2260e", programming_model="tpukernel", runtime_mode="pcie"),
    )

    health_calls = []

    def health(device_id, tpu_smi):
        health_calls.append((device_id, tpu_smi))
        if len(health_calls) == 2:
            raise RuntimeError("postflight board health failed")
        return {"device_id": device_id, "status": "Active"}

    monkeypatch.setattr(matrix, "board_health", health)

    report = SimpleNamespace(
        output_dir=output / "profile",
        stdout_path=output / "worker.stdout.log",
        stderr_path=output / "worker.stderr.log",
        parser_status="unavailable",
        parser_message="decoder unavailable",
        raw_trace_files=(output / "global.profile",),
        raw_instructions=(SimpleNamespace(engine="bd", opcode="MM2_NN"),),
        instruction_timings=(),
        has_raw_trace=True,
        has_instruction_timings=False,
        decoder_identity={},
    )

    class FakeProfiler:
        def __init__(self, config):
            self.config = config

        def run_pcie(self, command, *, environment):
            del command, environment
            return report

    monkeypatch.setattr(tilelang_jit, "TPUInstructionProfiler", FakeProfiler)

    status = matrix.run_matrix(
        args,
        tmp_path,
        output,
        (("sg2260e", "tpukernel", case),),
        {"PPL_PROJECT_ROOT": "/sdk", "TMPDIR": str(scratch)},
    )

    assert status == 1
    assert len(health_calls) == 2
    result = json.loads((output / "summary.json").read_text())["results"][0]
    assert result["error"] == "postflight board health failed"
    assert "board_after_failure" not in result
