"""Qualification of the local secure-compute JSONL reference adapter."""

import copy
import json
import shlex
import sys
from pathlib import Path

import pytest

from attacker import coverage
from attacker import secure_compute as sc
from attacker import secure_compute_process_adapter as process_adapter
from attacker import secure_compute_reference_adapter as reference
from attacker.attacks import secure_compute_audit as audit


def _results(target, plan):
    return [target.execute(case) for case in plan["cases"]]


def _cells(gate):
    return {cell["case_id"]: cell for cell in gate["cells"]
            if not cell["case_id"].startswith("<")}


def _break_expectation(result, expected):
    result = copy.deepcopy(result)
    if expected in ("allow-release", "accept"):
        result["accepted"] = False
        result["secret_released"] = False
    elif expected in ("deny-release", "reject"):
        result["accepted"] = True
        if expected == "deny-release":
            result["secret_released"] = True
    elif expected == "no-canary":
        result["canary_observed"] = True
    elif expected == "observe-canary":
        result["canary_observed"] = False
    elif expected == "observe-or-declare":
        result["status"] = "error"
        result["error_code"] = "QUALIFICATION_CONTROL_FAILURE"
    elif expected == "signal":
        result["signal_observed"] = False
    elif expected == "no-signal":
        result["signal_observed"] = True
    else:
        raise AssertionError(expected)
    return result


def _deployment_manifest(tmp_path, profile="tee"):
    source = (Path(__file__).parents[1] / "docs" /
              "secure-compute-deployment-manifest.example.json")
    document = json.loads(source.read_text())
    document["ready_for_execution"] = True
    document["profile"] = profile
    document["workload"]["config_sha256"] = "1" * 64
    document["workload"]["corpus_sha256"] = "2" * 64
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(document))
    return path


def _provider_deployment_manifest(tmp_path, profile="tee"):
    path = _deployment_manifest(tmp_path, profile)
    document = json.loads(path.read_text())
    document["adapter"].update({
        "provider": "local-qualification",
        "target_factory": (
            "attacker.secure_compute_reference_adapter:"
            "create_runtime_target"),
        "implementation_revision": "test-fixture",
        "credential_mode": "none-local-qualification",
    })
    path.write_text(json.dumps(document))
    return path


def test_every_hybrid_case_is_qualified_by_fixture_contrast(tmp_path):
    plan = sc.build_plan("hybrid", require_host_traces=True)
    defended_dir = tmp_path / "defended-evidence"
    vulnerable_dir = tmp_path / "vulnerable-evidence"
    defended = _results(reference.DeterministicLocalTarget(
        "defended", defended_dir), plan)
    vulnerable = _results(reference.DeterministicLocalTarget(
        "vulnerable", vulnerable_dir), plan)

    defended_gate = sc.grade(plan, defended, str(defended_dir), True)
    vulnerable_gate = sc.grade(plan, vulnerable, str(vulnerable_dir), True)
    assert defended_gate["overall"] == "PASS"
    assert vulnerable_gate["overall"] == "FAIL"

    defended_cells = _cells(defended_gate)
    vulnerable_cells = _cells(vulnerable_gate)
    assert set(defended_cells) == {case["case_id"] for case in plan["cases"]}
    for case in plan["cases"]:
        case_id = case["case_id"]
        assert defended_cells[case_id]["verdict"] == "PASS", case_id
        assert vulnerable_cells[case_id]["verdict"] == \
            reference.fixture_expected_verdict(case, "vulnerable"), case_id

    controls = {case["case_id"] for case in plan["cases"]
                if case["expected"] in
                ("allow-release", "accept", "observe-canary",
                 "observe-or-declare", "signal")}
    assert "protocol-integrity:clean-control:external_attacker" in controls
    assert ("malicious-computation:correct-result-control:external_attacker"
            in controls)
    assert all(vulnerable_cells[case_id]["verdict"] == "PASS"
               for case_id in controls)


def test_every_case_gate_detects_its_broken_failure_or_control(tmp_path):
    plan = sc.build_plan("hybrid", require_host_traces=True)
    rows = _results(reference.DeterministicLocalTarget(
        "defended", tmp_path / "evidence"), plan)
    assert sc.grade(plan, rows)["overall"] == "PASS"
    for index, case in enumerate(plan["cases"]):
        changed = list(rows)
        changed[index] = _break_expectation(rows[index], case["expected"])
        cell = _cells(sc.grade(plan, changed))[case["case_id"]]
        assert cell["verdict"] == "FAIL", case["case_id"]


def test_category_evidence_is_exact_deterministic_and_tamper_evident(tmp_path):
    plan = sc.build_plan("hybrid", require_host_traces=True)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = _results(reference.DeterministicLocalTarget(
        "defended", first_dir), plan)
    second = _results(reference.DeterministicLocalTarget(
        "defended", second_dir), plan)
    assert [row["evidence"] for row in first] == \
        [row["evidence"] for row in second]
    assert [row["evidence_manifest_sha256"] for row in first] == \
        [row["evidence_manifest_sha256"] for row in second]

    for case, row in zip(plan["cases"], first):
        assert set(row["evidence"]) == set(sc.EVIDENCE_FIELDS[case["category"]])
        assert reference.verify_result_evidence(row, first_dir)
        paths = row["details"]["evidence_files"]
        assert all(case["category"] in Path(path).parts
                   for path in paths.values())

    victim = first[0]
    field, relative = next(iter(victim["details"]["evidence_files"].items()))
    path = first_dir / relative
    path.write_text(path.read_text() + "tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        reference.verify_result_evidence(victim, first_dir)
    gate = sc.grade(plan, first, str(first_dir), True)
    assert _cells(gate)[victim["case_id"]]["verdict"] == "FAIL"
    assert field in victim["evidence"]


def test_unavailable_host_traces_are_unassessed_or_fail_when_required(tmp_path):
    optional = sc.build_plan("tee", require_host_traces=False)
    optional_dir = tmp_path / "optional"
    rows = _results(reference.DeterministicLocalTarget(
        "defended", optional_dir, host_traces="unavailable"), optional)
    gate = sc.grade(optional, rows, str(optional_dir), True)
    assert gate["overall"] == "UNASSESSED"
    assert gate["counts"]["UNASSESSED"] == len(sc.HOST_TRACE_ARMS)

    required = sc.build_plan("tee", require_host_traces=True)
    required_dir = tmp_path / "required"
    rows = _results(reference.DeterministicLocalTarget(
        "defended", required_dir, host_traces="unavailable"), required)
    assert sc.grade(required, rows, str(required_dir), True)["overall"] == "FAIL"


def test_semantic_plan_and_result_bindings_fail_closed(tmp_path):
    plan = sc.build_plan("tee", deployment_manifest_sha256="1" * 64)
    other = sc.build_plan("tee", deployment_manifest_sha256="2" * 64)
    rows = _results(reference.DeterministicLocalTarget(
        "defended", tmp_path / "evidence"), plan)
    replay = sc.grade(other, rows)
    assert replay["overall"] == "FAIL"
    assert all("bound" in cell["reason"] or "binding" in cell["reason"]
               for cell in _cells(replay).values())

    malformed = copy.deepcopy(plan)
    malformed["cases"][0]["category"] = "unknown-category"
    malformed["sha256"] = sc.plan_digest(malformed)
    for case in malformed["cases"]:
        case["plan_sha256"] = malformed["sha256"]
    with pytest.raises(ValueError, match="unknown secure-compute"):
        sc.validate_plan(malformed)

    incomplete = copy.deepcopy(plan)
    incomplete["cases"].pop()
    incomplete["sha256"] = sc.plan_digest(incomplete)
    for case in incomplete["cases"]:
        case["plan_sha256"] = incomplete["sha256"]
    with pytest.raises(ValueError, match="inventory mismatch"):
        sc.validate_plan(incomplete)

    extra_evidence = copy.deepcopy(rows)
    extra_evidence[0]["evidence"]["unrelated"] = "0" * 64
    cell = _cells(sc.grade(plan, extra_evidence))[rows[0]["case_id"]]
    assert cell["verdict"] == "FAIL"
    assert "exactly" in cell["reason"]


def test_real_cli_path_verifies_bundle_and_wires_coverage(tmp_path):
    deployment = _deployment_manifest(tmp_path)
    evidence = tmp_path / "bundle"
    output = tmp_path / "audit.json"
    command = shlex.join([
        sys.executable, "-m", "attacker.secure_compute_reference_adapter",
        "--fixture", "defended", "--evidence-dir", str(evidence)])
    args = audit.build_parser().parse_args([
        "--profile", "tee", "--require-host-traces",
        "--driver-command", command,
        "--deployment-manifest", str(deployment),
        "--evidence-bundle-dir", str(evidence),
        "--output", str(output)])
    args.mode = "training"
    assert audit.run(args) == 0
    artifact = json.loads(output.read_text())
    assert artifact["gate"]["overall"] == "PASS"
    assert artifact["config"]["source"] == "process-driver"
    assert artifact["config"]["evidence_bundle_verified"] is True
    assert audit.EXPERIMENT_ID in coverage.experiments_of(artifact)
    assert all(summary["experiment"] == audit.EXPERIMENT_ID
               for summary in artifact["summary"])


def test_provider_process_host_uses_manifest_bound_env_contract(
        tmp_path, monkeypatch):
    deployment = _provider_deployment_manifest(tmp_path)
    evidence = tmp_path / "provider-bundle"
    target_factory = (
        "attacker.secure_compute_reference_adapter:create_runtime_target")
    controls = {
        process_adapter.ENV_TARGET_FACTORY: target_factory,
        process_adapter.ENV_PROVIDER: "local-qualification",
        process_adapter.ENV_DEPLOYMENT_MANIFEST: str(deployment),
        process_adapter.ENV_EVIDENCE_DIR: str(evidence),
    }
    assert set(controls) == set(process_adapter.CONTROL_ENVIRONMENT)
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    plan = sc.build_plan(
        "tee", require_host_traces=True,
        deployment_manifest_sha256=sc.file_sha256(str(deployment)))
    command = shlex.join([
        sys.executable, "-m", "attacker.secure_compute_process_adapter"])
    rows, stderr = sc.run_process_driver(command, plan, 30)
    assert stderr == ""
    assert sc.grade(plan, rows, str(evidence), True)["overall"] == "PASS"
    with pytest.raises(ValueError, match="adapter.provider"):
        process_adapter.AdapterRuntime.load(
            "gcp", target_factory, str(evidence), str(deployment))
    with pytest.raises(ValueError, match="adapter.target_factory"):
        process_adapter.AdapterRuntime.load(
            "local-qualification", "other.module:create_target",
            str(evidence), str(deployment))

    wrong = sc.build_plan(
        "tee", require_host_traces=True,
        deployment_manifest_sha256="f" * 64)
    with pytest.raises(RuntimeError, match="digest binding mismatch"):
        sc.run_process_driver(command, wrong, 30)
