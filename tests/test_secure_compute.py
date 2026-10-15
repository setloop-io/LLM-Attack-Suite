"""Confidential-compute plan, adapter contract, gate, and CLI."""

import json
import shlex
import sys
from pathlib import Path

import pytest

from attacker import secure_compute as sc
from attacker import surfaces, threat_model
from attacker.attacks import secure_compute_audit as audit


def _write_results(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _manifest(tmp_path, profile="tee"):
    source = (Path(__file__).parents[1] / "docs" /
              "secure-compute-deployment-manifest.example.json")
    doc = json.loads(source.read_text())
    doc["ready_for_execution"] = True
    doc["profile"] = profile
    doc["workload"]["config_sha256"] = "1" * 64
    doc["workload"]["corpus_sha256"] = "2" * 64
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(doc))
    return path


def test_contract_registry_and_torchless_help():
    from attacker.attacks import load, validate_contract
    assert validate_contract(audit) is audit
    assert load("secure-compute-audit") is audit
    assert audit.EXPERIMENT_ID == "secure_compute_audit"
    assert audit.MODES == ("training", "inference")
    with pytest.raises(SystemExit) as exc:
        audit.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_plan_is_finite_complete_and_digest_bound():
    tee = sc.build_plan("tee")
    fhe = sc.build_plan("fhe")
    assert sc.validate_plan(tee) is tee
    assert len(tee["cases"]) == 53
    assert len(fhe["cases"]) == 53 + len(sc.FHE_ARMS)
    assert {c["arm"] for c in tee["cases"]
            if c["category"] == "attestation-key-release"} == set(sc.ATTESTATION_ARMS)
    assert {c["arm"] for c in tee["cases"]
            if c["category"] == "protocol-integrity"} == set(sc.PROTOCOL_ARMS)
    assert {c["arm"] for c in tee["cases"]
            if c["category"] == "malicious-computation"} == set(sc.COMPUTATION_ARMS)
    assert not any(c["category"] == "fhe-feedback-oracle" for c in tee["cases"])
    assert {c["arm"] for c in fhe["cases"]
            if c["category"] == "fhe-feedback-oracle"} == set(sc.FHE_ARMS)
    changed = json.loads(json.dumps(tee))
    changed["cases"][0]["request_id"] = "changed-request"
    with pytest.raises(ValueError, match="sha256"):
        sc.validate_plan(changed)


def test_deployment_manifest_is_validated_and_bound(tmp_path):
    path = _manifest(tmp_path)
    doc = sc.load_deployment_manifest(str(path), expected_profile="tee")
    assert doc["ready_for_execution"] is True
    digest = sc.file_sha256(str(path))
    plan = sc.build_plan("tee", deployment_manifest_sha256=digest)
    assert plan["deployment_manifest_sha256"] == digest
    assert all(c["payload"]["deployment_manifest_sha256"] == digest
               for c in plan["cases"])
    with pytest.raises(ValueError, match="does not match"):
        sc.load_deployment_manifest(str(path), expected_profile="fhe")
    doc["gates"]["invalid_release_tolerance"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="exactly zero"):
        sc.load_deployment_manifest(str(path), expected_profile="tee")
    doc["gates"]["invalid_release_tolerance"] = 0
    doc["workload"]["config_sha256"] = "not-a-digest"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="workload.config_sha256"):
        sc.load_deployment_manifest(str(path), expected_profile="tee")
    template = (Path(__file__).parents[1] / "docs" /
                "secure-compute-deployment-manifest.example.json")
    with pytest.raises(ValueError, match="ready_for_execution"):
        sc.load_deployment_manifest(str(template))


def test_dual_view_boundary_cases_share_canary_but_never_view():
    plan = sc.build_plan("tee")
    boundary = [c for c in plan["cases"] if c["category"] == "boundary-canary"]
    assert len(boundary) == 2 * len(sc.BOUNDARY_PHASES)
    for phase in sc.BOUNDARY_PHASES:
        pair = [c for c in boundary if c["arm"] == phase]
        assert {c["view"] for c in pair} == set(sc.VIEWS)
        assert len({c["canary_id"] for c in pair}) == 1
        assert {c["expected"] for c in pair} == {"no-canary", "observe-canary"}


def test_secure_gate_and_unassessed_are_distinct():
    plan = sc.build_plan("tee")
    gate = sc.grade(plan, sc.make_toy_results(plan))
    assert gate["overall"] == "UNASSESSED"
    assert gate["counts"]["FAIL"] == 0
    assert gate["counts"]["UNASSESSED"] == len(sc.HOST_TRACE_ARMS)
    assert gate["invalid_secret_releases"] == 0
    assert gate["external_canary_exposures"] == 0


def test_exact_release_gate_and_binding_fail_closed():
    plan = sc.build_plan("tee")
    rows = sc.make_toy_results(plan, unavailable_host=False)
    invalid = next(i for i, c in enumerate(plan["cases"])
                   if c["expected"] == "deny-release")
    rows[invalid]["secret_released"] = True
    gate = sc.grade(plan, rows)
    assert gate["overall"] == "FAIL"
    assert gate["invalid_secret_releases"] == 1

    rows = sc.make_toy_results(plan, unavailable_host=False)
    rows[0]["request_id"] = "wrong-request"
    gate = sc.grade(plan, rows)
    assert gate["overall"] == "FAIL"
    assert "binding mismatch" in gate["cells"][0]["reason"]

    rows = sc.make_toy_results(plan, unavailable_host=False)
    rows[0]["evidence"].pop("quote")
    gate = sc.grade(plan, rows)
    assert gate["overall"] == "FAIL"
    assert "evidence fields must be exactly" in gate["cells"][0]["reason"]


def test_missing_extra_duplicate_and_required_unavailable_fail():
    plan = sc.build_plan("tee")
    rows = sc.make_toy_results(plan, unavailable_host=False)
    assert sc.grade(plan, rows)["overall"] == "PASS"
    assert sc.grade(plan, rows[:-1])["missing_case_ids"]
    extra = dict(rows[-1], case_id="unplanned")
    assert sc.grade(plan, rows + [extra])["extra_case_ids"] == ["unplanned"]
    dup = sc.grade(plan, rows + [rows[0]])
    assert dup["duplicate_case_ids"] == [rows[0]["case_id"]]
    required = sc.build_plan("tee", require_host_traces=True)
    assert sc.grade(required, sc.make_toy_results(required))["overall"] == "FAIL"


def test_fhe_positive_control_and_production_feedback():
    plan = sc.build_plan("fhe")
    rows = sc.make_toy_results(plan, unavailable_host=False)
    assert sc.grade(plan, rows)["overall"] == "PASS"
    prod = next(i for i, c in enumerate(plan["cases"])
                if c["arm"] == "production-feedback")
    rows[prod]["signal_observed"] = True
    assert sc.grade(plan, rows)["overall"] == "FAIL"
    rows = sc.make_toy_results(plan, unavailable_host=False)
    pos = next(i for i, c in enumerate(plan["cases"])
               if c["arm"] == "leaky-positive-control")
    rows[pos]["signal_observed"] = False
    assert sc.grade(plan, rows)["overall"] == "FAIL"


def test_surface_provider_and_strict_jsonl(tmp_path):
    plan = sc.build_plan("tee")
    rows = sc.make_toy_results(plan, unavailable_host=False)
    path = tmp_path / "results.jsonl"
    _write_results(path, rows[:2])
    provider = surfaces.get_surface("secure-compute-results")
    loaded = list(provider.rows(mode="training", source=str(path)))
    assert len(loaded) == 2 and len(loaded[0][0]) == 4
    bad = dict(rows[0])
    bad["extra"] = True
    _write_results(path, [bad])
    with pytest.raises(ValueError, match="keys"):
        sc.read_jsonl(str(path))


def test_process_driver_round_trip(tmp_path):
    plan = sc.build_plan("tee")
    evidence = tmp_path / "evidence"
    command = shlex.join([
        sys.executable, "-m", "attacker.secure_compute_reference_adapter",
        "--fixture", "defended", "--evidence-dir", str(evidence)])
    rows, stderr = sc.run_process_driver(command, plan, 30)
    assert stderr == ""
    assert sc.grade(plan, rows, str(evidence), True)["overall"] == "PASS"


def test_cli_toy_artifact_and_plan_only(tmp_path):
    output = tmp_path / "audit.json"
    plan_path = tmp_path / "plan.json"
    args = audit.build_parser().parse_args([
        "--toy", "--quick", "--profile", "hybrid",
        "--plan-output", str(plan_path), "--output", str(output)])
    args.mode = "training"
    assert audit.run(args) == 2
    art = json.loads(output.read_text())
    assert art["schema"] == audit.ARTIFACT_SCHEMA
    assert art["config"]["toy"] is True
    assert art["config"]["plan_sha256"] == art["plan"]["sha256"]
    assert art["gate"]["overall"] == "UNASSESSED"
    assert threat_model.validate(art["threat_model"]) is art["threat_model"]
    journal = [json.loads(line) for line in
               (tmp_path / "audit.json.jsonl").read_text().splitlines()]
    assert len(journal) == len(art["results"])
    assert all(row["experiment"] == audit.EXPERIMENT_ID for row in journal)
    assert json.loads(plan_path.read_text())["sha256"] == art["plan"]["sha256"]

    only = tmp_path / "only.json"
    args = audit.build_parser().parse_args(["--plan-only", "--plan-output", str(only)])
    args.mode = "inference"
    assert audit.run(args) == 0 and only.is_file()
    with pytest.raises(FileExistsError):
        sc.write_plan(str(only), sc.build_plan("tee"))
    assert only.is_file()


def test_cli_requires_exactly_one_execution_source():
    args = audit.build_parser().parse_args([])
    args.mode = "training"
    with pytest.raises(SystemExit, match="exactly one"):
        audit.run(args)
    args = audit.build_parser().parse_args(["--results", "x"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="deployment-manifest"):
        audit.run(args)
    args = audit.build_parser().parse_args(["--toy", "--results", "x"])
    args.mode = "training"
    with pytest.raises(SystemExit, match="exactly one"):
        audit.run(args)


def test_self_tests_pass():
    assert sc.self_test() == 0
    assert audit.self_test() == 0
