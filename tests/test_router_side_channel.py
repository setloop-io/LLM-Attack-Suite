"""router-side-channel: module contract, toy worlds (informative router
leaks / null router at baseline as the negative control), journal path +
schema drift, determinism, self-test. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import router_side_channel as rsc


def _run_toy(tmp_path, *extra, name="rs.json", mode="inference"):
    args = rsc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert rsc.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def _cells(art, arm=None, world=None):
    return [r for r in art["results"]
            if (arm is None or r["arm"] == arm)
            and (world is None or r["world"] == world)]


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(rsc) is rsc
    assert rsc.EXPERIMENT_ID == "router_side_channel"
    assert rsc.MODES == ("training", "inference")
    assert rsc.REQUIRES_LABELS is False
    assert isinstance(rsc.DESCRIPTION, str) and rsc.DESCRIPTION
    ap = rsc.build_parser()
    for flag in ("--toy", "--arms", "--journal", "--output", "--seed",
                 "--quick", "--n-sequences", "--n-probe-repeats",
                 "--slots", "--users"):
        assert any(flag in o for o in
                   [opt for a in ap._actions for opt in a.option_strings]), \
            flag
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert rsc.self_test() == 0


def test_toy_artifact_and_journal_layout(tmp_path):
    art, journal = _run_toy(tmp_path)
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art, key
    assert art["schema"] == "dtraining.attacker.router_side_channel.v1"
    assert art["config"]["attack"] == rsc.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert art["config"]["control_world"] == "null"
    assert len(journal) == len(art["results"]) > 0
    assert len(art["summary"]) > 0
    raw = (tmp_path / "rs.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw


def test_threat_model_is_v2_and_validates(tmp_path):
    art, _ = _run_toy(tmp_path)
    doc = art["threat_model"]
    assert threat_model.validate(doc) is doc
    assert doc["schema"] == "llm_attacker.threat_model.v2"
    assert doc["actor"]["intent"] == "passive"
    assert doc["actor"]["persistence"] == "cross-session"
    assert set(doc["scenario"]["security_properties"]) >= {
        "confidentiality", "privacy"}
    assert set(doc["scenario"]["lifecycle_stage"]) == {
        "inference", "operations"}
    assert set(doc["mappings"]["owasp_llm_2026"]) == {"LLM02", "LLM09"}


def test_every_arm_every_world_records(tmp_path):
    art, _ = _run_toy(tmp_path)
    for arm in rsc.ARMS:
        for world in ("informative", "null"):
            recs = _cells(art, arm, world)
            assert recs, (arm, world)
            assert all(r["experiment"] == rsc.EXPERIMENT_ID for r in recs)
    null = [r for r in art["results"] if r["world"] == "null"]
    assert null and all(r["control"] is True for r in null)
    checks = [s for s in art["summary"] if s.get("check")
              == "negative-control"]
    assert len(checks) == len(rsc.ARMS)
    for row in checks:
        assert row["attack_signal"] > 0.0, row
        assert "null_control" in row and "informative" in row


def test_informative_world_shows_the_attack(tmp_path):
    """The informative router must leak, otherwise the null control is
    vacuous."""
    art, _ = _run_toy(tmp_path)
    for r in _cells(art, "property-classification", "informative"):
        assert r["auc"] >= 0.9
        assert r["tpr_at_1pct_fpr"] >= 0.9
        assert 0.0 <= r["tpr_wilson_lo"] <= r["tpr_wilson_hi"] <= 1.0
    for r in _cells(art, "length-domain-inference", "informative"):
        assert r["length_mae"] < 0.6 * r["length_mae_baseline"]
        assert r["length_spearman"] >= 0.8
        assert r["mean_domain_auc"] >= 0.8
    for r in _cells(art, "linkability", "informative"):
        assert r["auc"] >= 0.9
        assert r["mean_cosine_same"] > r["mean_cosine_cross"]
    for r in _cells(art, "contention-fingerprinting", "informative"):
        assert r["auc"] >= 0.9
        assert r["n_active_slots"] > 0
        assert r["mean_depth_active"] > r["mean_depth_idle"]
    for r in _cells(art, "expert-map-extraction", "informative"):
        assert r["fidelity"] >= 0.85
        assert r["lift_pp"] > 50.0


def test_null_world_sits_at_baseline(tmp_path):
    """Negative control: the uniform router carries no routing-correlated
    signal (AUC ~ 0.5, extraction at the 1/E baseline, flat queue
    telemetry)."""
    art, _ = _run_toy(tmp_path)
    for r in _cells(art, "property-classification", "null"):
        assert abs(r["auc"] - 0.5) <= 0.2
        assert r["tpr_at_1pct_fpr"] <= 0.34
    for r in _cells(art, "length-domain-inference", "null"):
        assert r["length_mae"] == pytest.approx(
            r["length_mae_baseline"], abs=1e-9)
        assert r["length_mae_lift"] == pytest.approx(0.0, abs=1e-9)
        assert abs(r["mean_domain_auc"] - 0.5) <= 0.25
        assert r["domain_accuracy"] <= 0.5
    for r in _cells(art, "linkability", "null"):
        assert abs(r["auc"] - 0.5) <= 0.2
        assert r["tpr_at_1pct_fpr"] <= 0.34
    for r in _cells(art, "contention-fingerprinting", "null"):
        assert r["auc"] == pytest.approx(0.5, abs=1e-9)
        assert r["tpr_at_1pct_fpr"] == 0.0
        assert r["mean_depth_active"] == pytest.approx(0.0, abs=1e-9)
    for r in _cells(art, "expert-map-extraction", "null"):
        assert r["fidelity"] <= 0.35
        assert r["random_baseline"] == pytest.approx(1 / 8)


def test_arms_subset(tmp_path):
    art, _ = _run_toy(tmp_path, "--arms", "linkability",
                      "expert-map-extraction")
    assert {r["arm"] for r in art["results"]} == {
        "linkability", "expert-map-extraction"}


def test_determinism_same_seed_identical_metrics(tmp_path):
    a1, _ = _run_toy(tmp_path, name="d1.json")
    a2, _ = _run_toy(tmp_path, name="d2.json")
    assert a1["results"] == a2["results"]
    assert a1["summary"] == a2["summary"]


def test_toy_without_output_returns_zero(tmp_path):
    args = rsc.build_parser().parse_args(["--toy", "--quick"])
    args.mode = "inference"
    assert rsc.run(args) == 0


def _journal_record(eids, weights, qd, ovf, nbytes, ts, sid):
    return {"expert_ids": eids, "weights": weights, "queue_depth": qd,
            "overflow": ovf, "bytes": nbytes, "ts": ts, "session_id": sid}


JOURNAL_FIXTURE = (
    [_journal_record([2, 3], [0.6, 0.4], i % 3, False, 250,
                     1000.0 + 0.01 * i, "s1") for i in range(8)]
    + [_journal_record([4, 5], [0.55, 0.45], 0, False, 240,
                       1000.5 + 0.01 * i, "s1") for i in range(6)]
    + [_journal_record([2, 3], [0.7, 0.3], 4, i == 3, 260,
                       1002.0 + 0.01 * i, "s2") for i in range(6)]
)


def test_journal_path_analyzes_router_traces(tmp_path):
    jf = tmp_path / "traces.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in JOURNAL_FIXTURE))
    out = tmp_path / "j.json"
    args = rsc.build_parser().parse_args(
        ["--journal", str(jf), "--output", str(out)])
    args.mode = "inference"
    assert rsc.run(args) == 0
    art = json.loads(out.read_text())
    assert art["config"]["source"] == "journal"
    assert art["config"]["journal_schema"] == "llm_attacker.router_trace.v1"
    arms = {r["arm"] for r in art["results"]}
    assert arms == set(rsc.JOURNAL_ARMS)
    assert all(r["experiment"] == rsc.EXPERIMENT_ID for r in art["results"])
    ld = [r for r in art["results"]
          if r["arm"] == "length-domain-inference"][0]
    assert ld["n_sessions"] == 2 and ld["n_sequences"] == 3
    assert ld["records_per_sequence"]["max"] == 8
    assert ld["n_overflows"] == 1
    lk = [r for r in art["results"] if r["arm"] == "linkability"][0]
    assert lk["n_pairs"] == 1 and lk["cosine_mean"] is not None
    ct = [r for r in art["results"]
          if r["arm"] == "contention-fingerprinting"][0]
    assert ct["n_slots"] >= 3 and "active_intervals" in ct
    journal = (tmp_path / "j.json.jsonl").read_text().splitlines()
    assert len(journal) == len(art["results"])


def test_journal_schema_drift_names_file_and_line(tmp_path):
    jf = tmp_path / "traces.jsonl"
    jf.write_text(json.dumps(JOURNAL_FIXTURE[0]) + "\n")
    with jf.open("a") as f:
        f.write(json.dumps({"expert_ids": [1, 2], "weights": [0.7, 0.7],
                            "queue_depth": 0, "overflow": False,
                            "bytes": 10, "ts": 2.0,
                            "session_id": "s9"}) + "\n")
    with pytest.raises(ValueError, match=r"traces\.jsonl:2"):
        rsc.read_journal(str(jf))
    args = rsc.build_parser().parse_args(["--journal", str(jf)])
    args.mode = "inference"
    args.output = None
    with pytest.raises(SystemExit, match=r"traces\.jsonl:2"):
        rsc.run(args)


def test_journal_missing_keys_and_missing_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"expert_ids": [1]}) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        rsc.read_journal(str(bad))
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="missing.jsonl"):
        rsc.read_journal(str(missing))
    args = rsc.build_parser().parse_args(["--journal", str(missing)])
    args.mode = "inference"
    args.output = None
    with pytest.raises(SystemExit, match="missing.jsonl"):
        rsc.run(args)


def test_no_input_path_is_descriptive():
    args = rsc.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--toy"):
        rsc.run(args)
