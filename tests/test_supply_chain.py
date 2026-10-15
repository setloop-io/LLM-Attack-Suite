"""supply-chain (AI-artifact registry verification under attack): parser +
contract, toy end-to-end (9 arms x enforcing/permissive + negative
control), per-arm expected detection stages, the vulnerable permissive
baseline, AI-BOM/provenance metrics, real --registry path, schema-drift
errors, determinism, and the self-test. All torch-free."""

import json
import os

import pytest

import attacker.attacks.supply_chain as sc
from attacker import threat_model


def _run_toy(tmp_path, *extra, name="sc.json", seed_args=()):
    args = sc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name),
         *seed_args, *extra])
    args.mode = "inference"
    assert sc.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def _cells(art, arm=None, verifier=None):
    recs = [r for r in art["results"]
            if "error" not in r and r.get("arm") != "negative-control"]
    if arm is not None:
        recs = [r for r in recs if r["arm"] == arm]
    if verifier is not None:
        recs = [r for r in recs if r["verifier"] == verifier]
    return recs


def test_contract_and_parser():
    from attacker.attacks import validate_contract
    assert validate_contract(sc) is sc
    assert sc.EXPERIMENT_ID == "supply_chain"
    assert sc.MODES == ("training", "inference")
    assert sc.REQUIRES_LABELS is False
    assert isinstance(sc.DESCRIPTION, str) and sc.DESCRIPTION
    ap = sc.build_parser()
    for flag in ("--toy", "--arms", "--registry", "--manifest",
                 "--work-dir", "--output", "--seed", "--seeds", "--quick",
                 "--evidence"):
        assert any(flag in o for a in ap._actions for o in a.option_strings)
    args = ap.parse_args(["--arms", "rollback", "plugin-origin"])
    assert args.arms == ["rollback", "plugin-origin"]
    assert set(sc.ARMS) == set(sc.EXPECTED_STAGE) == {
        "artifact-substitution", "digest-mismatch", "unsigned-promotion",
        "rollback", "tokenizer-drift", "adapter-confusion",
        "plugin-origin", "unsafe-serialization", "dependency-drift"}
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert sc.self_test() == 0


def test_toy_run_artifact_layout(tmp_path):
    art, journal = _run_toy(tmp_path)
    assert art["schema"] == "dtraining.attacker.supply_chain.v1"
    for key in ("config", "threat_model", "provenance", "results",
                "summary"):
        assert key in art
    assert art["config"]["attack"] == sc.EXPERIMENT_ID
    assert art["config"]["source"] == "toy"
    assert art["config"]["signature_scheme"].startswith("toy-hmac-sha256")
    assert threat_model.validate(art["threat_model"]) is art["threat_model"]
    assert art["threat_model"]["schema"] == threat_model.SCHEMA_V2
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == [
        "LLM04", "LLM05"]
    assert len(art["results"]) == 19 == len(journal)
    assert all(r["experiment"] == sc.EXPERIMENT_ID
               for r in art["results"])


def test_every_arm_produces_records(tmp_path):
    art, _ = _run_toy(tmp_path)
    for arm in sc.ARMS:
        recs = _cells(art, arm=arm)
        assert len(recs) == 2, arm
        assert {r["verifier"] for r in recs} == {"enforcing", "permissive"}
        assert all(r["experiment"] == sc.EXPERIMENT_ID for r in recs)
        assert all(r["arm"] == arm for r in recs)
    summ_arms = {s["arm"] for s in art["summary"]}
    assert set(sc.ARMS) <= summ_arms and "all" in summ_arms


def test_enforcing_rejects_at_expected_stage_before_load(tmp_path):
    art, _ = _run_toy(tmp_path)
    load_idx = sc.STAGES.index("load")
    for arm in sc.ARMS:
        (enf,) = _cells(art, arm=arm, verifier="enforcing")
        assert enf["rejected_all"] is True, arm
        assert enf["bypass_rate"] == 0.0, arm
        assert enf["detected_at_expected_stage"] is True, arm
        assert enf["first_failure_stage"] == sc.EXPECTED_STAGE[arm], arm
        assert enf["detected_before_load"] is True, arm
        assert enf["first_failure_stage_index"] < load_idx, arm
    (sub,) = _cells(art, arm="artifact-substitution", verifier="enforcing")
    assert sub["first_failure_stage"] == "pinned-digest"
    assert sub["first_failure_stage_index"] < sc.STAGES.index(
        "safe-serialization") < load_idx
    (un,) = _cells(art, arm="unsafe-serialization", verifier="enforcing")
    assert un["first_failure_stage"] == "safe-serialization"
    assert un["first_failure_stage_index"] < load_idx


def test_permissive_baseline_bypasses_everything(tmp_path):
    """The vulnerable world must show the attack working — otherwise the
    negative control is vacuous."""
    art, _ = _run_toy(tmp_path)
    for arm in sc.ARMS:
        (per,) = _cells(art, arm=arm, verifier="permissive")
        assert per["role"] == "vulnerable-baseline"
        assert per["bypass_rate"] == 1.0, arm
        assert per["rejected_all"] is False, arm
        assert all(t["loaded"] for t in per["targets"]), arm
        assert per["first_failure_stage"] is None, arm


def test_negative_control_clean(tmp_path):
    art, _ = _run_toy(tmp_path)
    nc = [r for r in art["results"] if r.get("arm") == "negative-control"]
    assert len(nc) == 1
    nc = nc[0]
    assert nc["role"] == "negative-control"
    assert nc["n_false_rejections"] == 0
    assert nc["false_rejections"] == []
    assert nc["n_artifacts"] == 5
    assert nc["environment_checked"] is True
    assert nc["ai_bom_coverage_loaded"] == 1.0
    assert nc["provenance_completeness_loaded"] == 1.0
    summ = [s for s in art["summary"] if s["arm"] == "all"][0]
    assert summ["negative_control_false_rejections"] == 0


def test_audit_gap_metrics(tmp_path):
    """Loaded permissive assets drop out of the AI-BOM / provenance
    records — the measurable audit signal; summary carries the rates."""
    art, _ = _run_toy(tmp_path)
    (per,) = _cells(art, arm="artifact-substitution",
                    verifier="permissive")
    assert per["ai_bom_coverage_loaded"] == 0.0
    (up,) = _cells(art, arm="unsigned-promotion", verifier="permissive")
    assert up["provenance_completeness_loaded"] == 0.0
    (sub_enf,) = _cells(art, arm="artifact-substitution",
                        verifier="enforcing")
    assert sub_enf["ai_bom_coverage_loaded"] is None
    summ = {s["arm"]: s for s in art["summary"]}
    for arm in sc.ARMS:
        assert summ[arm]["bypass_rate_enforcing"] == 0.0, arm
        assert summ[arm]["bypass_rate_permissive"] == 1.0, arm
        assert summ[arm]["detected_at_expected_stage"] is True, arm
        assert summ[arm]["time_to_detect_stage_index_mean"] \
            == sc.STAGES.index(sc.EXPECTED_STAGE[arm]), arm


def test_plugin_origin_both_variants_rejected(tmp_path):
    art, _ = _run_toy(tmp_path)
    (enf,) = _cells(art, arm="plugin-origin", verifier="enforcing")
    assert enf["n_targets"] == 2
    variants = {t["variant"]: t for t in enf["targets"]}
    assert set(variants) == {"unapproved-name",
                             "name-conflict-origin-squat"}
    assert all(t["rejected"] and t["failure_stage"] == "plugin-allowlist"
               for t in variants.values())


def test_registry_path_real_verification(tmp_path):
    """--registry verifies a real on-disk registry dir (pristine => all
    clean; attacked => the arm's target rejected at its stage)."""
    reg_dir = tmp_path / "real-reg"
    sc.make_toy_registry(str(reg_dir), seed=7)
    out = tmp_path / "reg.json"
    args = sc.build_parser().parse_args(
        ["--registry", str(reg_dir), "--output", str(out)])
    args.mode = "training"
    assert sc.run(args) == 0
    art = json.loads(out.read_text())
    assert art["config"]["source"] == "registry"
    recs = art["results"]
    assert len(recs) == 6
    assert all(r["experiment"] == sc.EXPERIMENT_ID for r in recs)
    assert all(not r["rejected"] for r in recs)
    assert art["summary"][0]["n_rejected"] == 0
    assert art["summary"][0]["ai_bom_coverage_loaded"] == 1.0

    sc.apply_arm(str(reg_dir), "artifact-substitution", 7)
    out2 = tmp_path / "reg2.json"
    args = sc.build_parser().parse_args(
        ["--registry", str(reg_dir), "--output", str(out2)])
    args.mode = "training"
    assert sc.run(args) == 0
    art2 = json.loads(out2.read_text())
    rej = [r for r in art2["results"] if r["rejected"]]
    assert len(rej) == 1 and rej[0]["target"] == "toy-model"
    assert rej[0]["failure_stage"] == "pinned-digest"
    assert rej[0]["detected_before_load"] is True


def test_manifest_override_applies(tmp_path):
    """--manifest overrides the registry policy: an approved-manifest that
    flips the adapter binding must reject the pristine adapter."""
    reg_dir = tmp_path / "real-reg"
    sc.make_toy_registry(str(reg_dir), seed=3)
    override = tmp_path / "approved.json"
    override.write_text(json.dumps({
        "schema": sc.APPROVED_MANIFEST_SCHEMA,
        "adapter_bindings": {"toy-adapter": {"model": "other-model",
                                             "tenant": "tenant-b"}}}))
    rep = sc.verify(str(reg_dir), manifest_override=str(override))
    rej = [r for r in rep["artifacts"] if not r["ok"]]
    assert len(rej) == 1 and rej[0]["name"] == "toy-adapter"
    assert rej[0]["failure_stage"] == "binding"


def test_malformed_inputs_are_descriptive_errors(tmp_path):
    with pytest.raises(ValueError, match="registry.json"):
        sc.verify(str(tmp_path / "nope"))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "registry.json").write_text(json.dumps({"schema": "wrong.v0"}))
    with pytest.raises(ValueError, match="registry.json"):
        sc.verify(str(bad))
    reg_dir = tmp_path / "drift"
    sc.make_toy_registry(str(reg_dir), seed=1)
    (reg_dir / "toy-model" / "manifest.json").write_text("{not json")
    rep = sc.verify(str(reg_dir))
    rej = [r for r in rep["artifacts"] if not r["ok"]]
    assert len(rej) == 1 and rej[0]["failure_stage"] == "manifest-parse"
    assert "manifest.json" in rej[0]["reason"]
    args = sc.build_parser().parse_args(["--registry", str(bad),
                                         "--output",
                                         str(tmp_path / "x.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="registry.json"):
        sc.run(args)
    badm = tmp_path / "bad-override.json"
    badm.write_text(json.dumps({"schema": "wrong.v0"}))
    good = tmp_path / "good"
    sc.make_toy_registry(str(good), seed=1)
    with pytest.raises(ValueError, match="bad-override.json"):
        sc.verify(str(good), manifest_override=str(badm))


def test_no_input_path_is_descriptive():
    args = sc.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--toy"):
        sc.run(args)


def test_determinism_same_seed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    art1, _ = _run_toy(tmp_path / "a", name="sc.json")
    art2, _ = _run_toy(tmp_path / "b", name="sc.json")
    assert art1["results"] == art2["results"]
    assert art1["summary"] == art2["summary"]
    assert art1["config"] == art2["config"]


def test_determinism_seed_changes_fixture(tmp_path):
    """Different base seeds change the fixture bytes (digests differ) —
    determinism is seed-derived, not hard-coded."""
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    m1 = sc.make_toy_registry(str(d1), seed=11)
    m2 = sc.make_toy_registry(str(d2), seed=22)
    assert m1["digests"] != m2["digests"]
    m1b = sc.make_toy_registry(str(tmp_path / "r1b"), seed=11)
    assert m1 == m1b


def test_arm_subset_and_work_dir(tmp_path):
    work = tmp_path / "work"
    art, journal = _run_toy(
        tmp_path, "--arms", "rollback", "--work-dir", str(work))
    assert art["config"]["arms"] == ["rollback"]
    arms = {r["arm"] for r in art["results"]}
    assert arms == {"negative-control", "rollback"}
    assert len(art["results"]) == 3 == len(journal)
    assert (work / "seed-0" / "pristine" / "registry.json").is_file()
    assert (work / "seed-0" / "attack-rollback" / "archive"
            / "toy-model-v1" / "manifest.json").is_file()
