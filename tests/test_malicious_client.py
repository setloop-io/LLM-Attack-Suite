"""malicious-client (untrusted authenticated client against the shared cloud
body): contract + torchless parser, toy end-to-end over the vulnerable /
defended / isolated worlds, the defended-world negative control (and the
non-vacuous vulnerable world), per-arm semantics, the isolated attacker-
alone control, the shifted-benign false-positive control, the --probe-file
real-replay integration path, malformed-input errors, determinism, and the
self-test. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import malicious_client as mc


def _run_toy(tmp_path, *extra, name="mc.json"):
    args = mc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = "training"
    assert mc.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(mc) is mc
    assert mc.EXPERIMENT_ID == "malicious_client"
    assert mc.MODES == ("training", "inference")
    assert mc.REQUIRES_LABELS is False
    ap = mc.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    flags = {a for opt in ap._actions for a in opt.option_strings}
    assert {"--toy", "--quick", "--arms", "--world", "--clients",
            "--colluders", "--budgets", "--extract-budget", "--probes",
            "--collusion-probes", "--rounds", "--noise-sigma",
            "--probe-file", "--seed", "--output"} <= flags


def test_self_test_passes():
    assert mc.self_test() == 0


def test_toy_quick_writes_artifact_and_journal(tmp_path):
    art, journal = _run_toy(tmp_path)
    assert art["schema"] == "dtraining.attacker.malicious_client.v1"
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["config"]["attack"] == mc.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert art["config"]["world"] == "both"
    assert len(journal) == len(art["results"]) > 0
    assert all(r["experiment"] == mc.EXPERIMENT_ID for r in journal)
    assert art["summary"]


def test_artifact_threat_model_is_valid_v2(tmp_path):
    art, _ = _run_toy(tmp_path)
    doc = threat_model.validate(art["threat_model"])
    assert doc["schema"] == "llm_attacker.threat_model.v2"
    assert doc["actor"]["intent"] == "malicious"
    assert doc["actor"]["role"] == "client"
    assert doc["actor"]["persistence"] == "cross-session"
    assert set(doc["scenario"]["security_properties"]) == {
        "confidentiality", "integrity", "availability"}
    assert set(doc["scenario"]["lifecycle_stage"]) == {"training",
                                                       "inference"}
    assert set(doc["controls"]["tested"]) == set(mc.DEFENSES)
    assert doc["mappings"]["owasp_llm_2026"] == ["LLM02", "LLM05", "LLM06"]


def test_every_arm_produces_records_in_all_worlds(tmp_path):
    art, _ = _run_toy(tmp_path)
    for arm in mc.ARMS:
        for world in ("vulnerable", "defended", "isolated"):
            recs = [r for r in art["results"]
                    if r["arm"] == arm and r["world"] == world]
            assert recs, f"{arm}/{world} missing"
            assert all(r["experiment"] == mc.EXPERIMENT_ID for r in recs)


def test_negative_control_defended_world_defeats_every_arm(tmp_path):
    art, _ = _run_toy(tmp_path)
    defended = [r for r in art["results"] if r["world"] == "defended"]
    assert defended and all(r["role"] == "negative-control" for r in defended)
    assert {r["arm"] for r in defended} == set(mc.ARMS)
    assert all(r["attack_succeeded"] is False for r in defended)
    ext = [r for r in defended if r["arm"] == "crafted-extraction"]
    assert all(r["queries_granted"] <= mc.PARAMS["budget_limit"]
               for r in ext)
    assert all(not r["determinate"] for r in ext)
    assert all(r["rel_frobenius"] > 0.2 for r in ext)
    be = [r for r in defended if r["arm"] == "budget-extraction"]
    assert min(r["rel_frobenius"] for r in be) > 0.2
    obs = [r for r in defended if r["arm"] == "neuron-observability"][0]
    assert obs["n_probes_granted"] == mc.PARAMS["budget_limit"]
    assert obs["mean_rel_err"] > mc.OBS_REL_GATE
    poi = [r for r in defended if r["arm"] == "gradient-poisoning"][0]
    assert poi["honest_mse_delta"] == 0.0
    assert poi["detection_rate"] == 1.0
    assert poi["false_positive_rate"] == 0.0
    bomb = [r for r in defended if r["arm"] == "instability-bomb"][0]
    assert bomb["detection_rate"] == 1.0
    assert bomb["honest_outputs_corrupted"] == 0
    assert bomb["server_corrupted"] is False
    assert bomb["shifted_benign_accepted"] == bomb["shifted_benign_total"] > 0
    col = [r for r in defended if r["arm"] == "collusion"][0]
    assert col["gain"] == 1.0
    assert col["queries_granted"] == mc.PARAMS["budget_limit"]
    assert col["cross_tenant_duplicate_accepted"] is True


def test_isolated_control_cells(tmp_path):
    art, _ = _run_toy(tmp_path)
    iso = [r for r in art["results"] if r["world"] == "isolated"]
    assert iso and all(r["role"] == "isolated-control" for r in iso)
    assert {r["arm"] for r in iso} == set(mc.ARMS)
    assert all(r["attack_succeeded"] is False for r in iso)
    by_arm = {r["arm"]: r for r in iso}
    assert by_arm["gradient-poisoning"]["honest_mse_delta"] == 0.0
    assert by_arm["gradient-poisoning"]["private_self_harm_delta"] > 0.0
    assert by_arm["instability-bomb"]["honest_outputs_corrupted"] == 0
    assert by_arm["instability-bomb"]["server_corrupted"] is True
    assert by_arm["crafted-extraction"]["rel_frobenius"] < 1e-9


def test_vulnerable_world_attack_works_non_vacuous(tmp_path):
    art, _ = _run_toy(tmp_path)
    vuln = [r for r in art["results"] if r["world"] == "vulnerable"]
    assert {r["arm"] for r in vuln} == set(mc.ARMS)
    for arm in mc.ARMS:
        assert any(r["attack_succeeded"] is True for r in vuln
                   if r["arm"] == arm), f"{arm} vacuous in vulnerable world"
    ext = {r["budget"]: r for r in vuln if r["arm"] == "crafted-extraction"}
    assert ext[max(ext)]["determinate"] is True
    assert ext[max(ext)]["rel_frobenius"] < 1e-9
    assert ext[min(ext)]["attack_succeeded"] is False
    be = {r["design"]: r for r in vuln if r["arm"] == "budget-extraction"}
    assert be["optimal-zero-axes"]["rel_frobenius"] < 1e-9
    assert be["naive-random"]["attack_succeeded"] is False
    obs = [r for r in vuln if r["arm"] == "neuron-observability"][0]
    assert obs["mean_rel_err"] <= mc.OBS_REL_GATE
    assert obs["rank_corr"] >= mc.OBS_RANK_GATE
    poi = [r for r in vuln if r["arm"] == "gradient-poisoning"][0]
    assert poi["honest_mse_delta"] > mc.POISON_GATE_ABS
    assert poi["mse_ratio"] > 2.0
    assert poi["detection_rate"] == 0.0
    assert poi["cross_client_excess"] == poi["honest_mse_delta"]
    bomb = [r for r in vuln if r["arm"] == "instability-bomb"][0]
    assert bomb["honest_outputs_corrupted"] == bomb["pre_bomb_honest_ok"] > 0
    assert bomb["compute_amplification"] > 10.0
    assert bomb["benign_parity"] < 2.0
    assert bomb["memory_amplification"] > 100.0
    col = [r for r in vuln if r["arm"] == "collusion"][0]
    assert col["gain"] >= mc.COLLUSION_GAIN_GATE
    assert col["rel_err_colluded"] < col["rel_err_single"]
    assert col["expected_gain_sqrt_c"] == 4 ** 0.5


def test_arms_subset(tmp_path):
    art, _ = _run_toy(tmp_path, "--arms", "gradient-poisoning",
                      "instability-bomb")
    assert {r["arm"] for r in art["results"]} == {"gradient-poisoning",
                                                  "instability-bomb"}


def test_world_subset_still_runs_isolated_control(tmp_path):
    art, _ = _run_toy(tmp_path, "--world", "vulnerable")
    assert {r["world"] for r in art["results"]} == {"vulnerable", "isolated"}
    assert art["config"]["world"] == "vulnerable"


def test_determinism_same_seed_identical_metrics(tmp_path):
    art1, _ = _run_toy(tmp_path, name="a.json")
    art2, _ = _run_toy(tmp_path, name="b.json")
    assert art1["results"] == art2["results"]
    assert art1["summary"] == art2["summary"]


def test_failed_cells_are_journaled_not_raised(tmp_path):
    args = mc.build_parser().parse_args(
        ["--toy", "--quick", "--arms", "crafted-extraction",
         "--budgets", "0", "--output", str(tmp_path / "err.json")])
    args.mode = "training"
    assert mc.run(args) == 0
    journal = [json.loads(l) for l in
               (tmp_path / "err.json.jsonl").read_text().splitlines()]
    errors = [r for r in journal if "error" in r]
    assert errors and all(r["experiment"] == mc.EXPERIMENT_ID
                          for r in errors)


def test_probe_file_replay_integration_path(tmp_path):
    probes = tmp_path / "stream.jsonl"
    lines = [
        *[json.dumps({"client": "h1", "kind": "query",
                      "x": [float(i), 0.5, -0.25, 0.0, 1.0, -1.0, 0.75,
                            0.125]}) for i in range(4)],
        json.dumps({"client": "h1", "kind": "update",
                    "g": [0.3] * mc.D_OUT}),
        json.dumps({"client": "h2", "kind": "update",
                    "g": [0.4] * mc.D_OUT}),
        json.dumps({"client": "h2", "kind": "update",
                    "g": [0.35] * mc.D_OUT}),
        *[json.dumps({"client": c, "tenant": "atk", "kind": "query",
                      "x": [1.0] * mc.D_IN}) for c in ("s1", "s2", "s3")],
        json.dumps({"client": "s1", "tenant": "atk", "kind": "update",
                    "g": [-8.0] * mc.D_OUT}),
        json.dumps({"client": "s1", "tenant": "atk", "kind": "update",
                    "g": [float("nan")] * mc.D_OUT}),
    ]
    probes.write_text("\n".join(lines) + "\n")
    out = tmp_path / "replay.json"
    args = mc.build_parser().parse_args(
        ["--probe-file", str(probes), "--output", str(out)])
    args.mode = "inference"
    assert mc.run(args) == 0
    art = json.loads(out.read_text())
    threat_model.validate(art["threat_model"])
    rec = art["results"][0]
    assert rec["arm"] == "policy-replay" and rec["world"] == "real"
    assert rec["n_records"] == len(lines)
    assert rec["n_rejected"] > 0
    assert rec["rejection_reasons"].get("duplicate-probe", 0) == 2
    assert rec["rejection_reasons"].get("invalid-update", 0) == 1
    assert rec["rejection_reasons"].get("anomalous-update", 0) == 1
    assert rec["per_tenant"]["h1"]["rejected"] == 0
    assert rec["per_tenant"]["h2"]["rejected"] == 0
    assert rec["attack_succeeded"] is False


def test_malformed_probe_file_descriptive(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"client": "h1", "kind": "query", "x": [1.0]}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        mc.read_probe_file(str(bad))
    bad2 = tmp_path / "bad2.jsonl"
    bad2.write_text('{"client": "h1", "kind": "query", "x": [0.0] * 1}\n')
    with pytest.raises(ValueError, match=r"bad2\.jsonl:1"):
        mc.read_probe_file(str(bad2))
    bad3 = tmp_path / "bad3.jsonl"
    bad3.write_text("not json\n")
    with pytest.raises(ValueError, match=r"bad3\.jsonl:1"):
        mc.read_probe_file(str(bad3))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="empty.jsonl"):
        mc.read_probe_file(str(empty))


def test_missing_probe_file_is_descriptive_system_exit(tmp_path):
    args = mc.build_parser().parse_args(
        ["--probe-file", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit, match="missing.jsonl"):
        mc.run(args)


def test_no_input_path_is_descriptive():
    args = mc.build_parser().parse_args([])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--toy"):
        mc.run(args)
