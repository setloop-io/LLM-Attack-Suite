"""multi-tenant-isolation: module contract, toy worlds (vulnerable leaks /
defended matches the physically-isolated negative control), journal path +
schema drift, determinism, self-test. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import multi_tenant_isolation as mti


def _run_toy(tmp_path, *extra, name="mt.json", mode="inference"):
    args = mti.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert mti.run(args) == 0
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
    assert validate_contract(mti) is mti
    assert mti.EXPERIMENT_ID == "multi_tenant_isolation"
    assert mti.MODES == ("training", "inference")
    assert mti.REQUIRES_LABELS is False
    assert isinstance(mti.DESCRIPTION, str) and mti.DESCRIPTION
    ap = mti.build_parser()
    for flag in ("--toy", "--arms", "--world", "--requests", "--journal",
                 "--output", "--seed", "--quick"):
        assert any(flag in o for o in
                   [opt for a in ap._actions for opt in a.option_strings]), flag
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert mti.self_test() == 0


def test_toy_artifact_and_journal_layout(tmp_path):
    art, journal = _run_toy(tmp_path)
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art, key
    assert art["schema"] == "dtraining.attacker.multi_tenant_isolation.v1"
    assert art["config"]["attack"] == mti.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert art["config"]["control_world"] == "isolated"
    assert len(journal) == len(art["results"]) > 0
    assert len(art["summary"]) > 0


def test_threat_model_is_v2_and_validates(tmp_path):
    art, _ = _run_toy(tmp_path)
    doc = art["threat_model"]
    assert threat_model.validate(doc) is doc
    assert doc["schema"] == "llm_attacker.threat_model.v2"
    assert doc["actor"]["intent"] == "active"
    assert doc["actor"]["role"] == "co-tenant"
    assert doc["scope"]["deployment"] == "multi-tenant"
    assert set(doc["scenario"]["security_properties"]) >= {
        "confidentiality", "authorization"}
    assert set(doc["mappings"]["owasp_llm_2026"]) == {
        "LLM02", "LLM06", "LLM08", "LLM09"}


def test_every_arm_every_world_records(tmp_path):
    art, _ = _run_toy(tmp_path)
    for arm in mti.ARMS:
        for world in ("vulnerable", "defended", "isolated"):
            recs = _cells(art, arm, world)
            assert recs, (arm, world)
            assert all(r["experiment"] == mti.EXPERIMENT_ID for r in recs)
    iso = [r for r in art["results"] if r["world"] == "isolated"]
    assert iso and all(r["control"] is True for r in iso)


def test_vulnerable_world_leaks(tmp_path):
    """The attack must work in the vulnerable world, otherwise the
    negative control is vacuous."""
    art, _ = _run_toy(tmp_path)
    for arm in mti.ARMS:
        for r in _cells(art, arm, "vulnerable"):
            assert r["isolation_failures"] > 0, arm
            assert r["isolation_failures_per_million"] > 0.0, arm
    for arm in ("session-collision", "continuous-batch-mixup",
                "kv-content-canary", "allocator-reuse"):
        assert all(r["canary_recovery_exact"] > 0.0
                   for r in _cells(art, arm, "vulnerable")), arm
    assert all(r["canary_recovery_fuzzy"] > 0.0
               for r in _cells(art, "error-path-bleed", "vulnerable"))
    cache = _cells(art, "prefix-cache-isolation", "vulnerable")
    assert all(r["cache_membership_auc"] > 0.9 for r in cache)
    assert all(r["cache_tpr_at_1pct_fpr"] >= 0.9 for r in cache)
    assert all(r["cross_tenant_hits"] > 0 for r in cache)
    mis = _cells(art, "response-misattribution", "vulnerable")
    assert all(r["misattribution_rate"] > 0.0 for r in mis)


def test_negative_control_sits_at_baseline(tmp_path):
    """Physically-isolated world: every metric at baseline."""
    art, _ = _run_toy(tmp_path)
    for arm in mti.ARMS:
        for r in _cells(art, arm, "isolated"):
            assert r["canary_recovery_exact"] == 0.0, arm
            assert r["canary_recovery_fuzzy"] == 0.0, arm
            assert r["isolation_failures"] == 0, arm
            assert r["isolation_failures_per_million"] == 0.0, arm
    for r in _cells(art, "prefix-cache-isolation", "isolated"):
        assert abs(r["cache_membership_auc"] - 0.5) < 0.25
        assert r["cache_tpr_at_1pct_fpr"] == 0.0
        assert r["cross_tenant_hits"] == 0
    assert all(r["misattribution_rate"] == 0.0
               for r in _cells(art, "response-misattribution", "isolated"))


def test_defended_world_matches_isolated_control(tmp_path):
    """Defended world (all defenses on) must not leak and must match the
    isolated control exactly (identical seeded draws, no leak channel)."""
    art, _ = _run_toy(tmp_path)
    metric_keys = ("canary_recovery_exact", "canary_recovery_fuzzy",
                   "cache_membership_auc", "cache_tpr_at_1pct_fpr",
                   "misattribution_rate", "isolation_failures",
                   "isolation_failures_per_million")
    for arm in mti.ARMS:
        defended = _cells(art, arm, "defended")
        isolated = _cells(art, arm, "isolated")
        assert len(defended) == len(isolated) == 1
        d, i = defended[0], isolated[0]
        assert d["isolation_failures"] == 0, arm
        for key in metric_keys:
            if key in d:
                assert d[key] == pytest.approx(i[key], abs=1e-9), (arm, key)
    checks = [s for s in art["summary"]
              if s.get("check") == "negative-control"]
    assert len(checks) == len(mti.ARMS)
    for row in checks:
        assert row["attack_signal"] > 0.0
        assert row["defended_minus_control"] == pytest.approx(0.0, abs=1e-9)


def test_world_flag_runs_subset_plus_control(tmp_path):
    art, _ = _run_toy(tmp_path, "--world", "vulnerable")
    worlds = {r["world"] for r in art["results"]}
    assert worlds == {"vulnerable", "isolated"}
    assert art["config"]["worlds"] == ["vulnerable", "isolated"]


def test_arms_subset(tmp_path):
    art, _ = _run_toy(tmp_path, "--arms", "kv-content-canary",
                      "allocator-reuse")
    assert {r["arm"] for r in art["results"]} == {
        "kv-content-canary", "allocator-reuse"}


def test_determinism_same_seed_identical_metrics(tmp_path):
    a1, _ = _run_toy(tmp_path, name="d1.json")
    a2, _ = _run_toy(tmp_path, name="d2.json")
    assert a1["results"] == a2["results"]
    assert a1["summary"] == a2["summary"]


JOURNAL_FIXTURE = [
    {"ts": 1.0, "tenant_hint": "victim", "session_id": "s1",
     "request_id": "r1", "phase": "prefill", "cache_hit": False,
     "bytes": 100},
    {"ts": 2.0, "tenant_hint": "victim", "session_id": "s1",
     "request_id": "r2", "phase": "prefill", "cache_hit": True,
     "bytes": 100},
    {"ts": 3.0, "tenant_hint": "attacker", "session_id": "s1",
     "request_id": "r1", "phase": "prefill", "cache_hit": True,
     "bytes": 100},
    {"ts": 4.0, "tenant_hint": "attacker", "session_id": "s2",
     "request_id": "r3", "phase": "decode", "cache_hit": False,
     "bytes": 50},
]


def _write_journal(tmp_path, records, name="serving.jsonl"):
    jf = tmp_path / name
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    return jf


def test_journal_path_analyzes_real_serving_journal(tmp_path):
    jf = _write_journal(tmp_path, JOURNAL_FIXTURE)
    out = tmp_path / "j.json"
    args = mti.build_parser().parse_args(
        ["--journal", str(jf), "--output", str(out)])
    args.mode = "inference"
    assert mti.run(args) == 0
    art = json.loads(out.read_text())
    assert art["config"]["source"] == "journal"
    assert art["config"]["journal_schema"] == "llm_attacker.multi_tenant.v1"
    arms = {r["arm"] for r in art["results"]}
    assert arms == set(mti.JOURNAL_ARMS)
    sc = [r for r in art["results"] if r["arm"] == "session-collision"][0]
    assert sc["n_sessions"] == 2 and sc["n_colliding_sessions"] == 1
    assert sc["colliding_sessions"] == ["s1"]
    pc = [r for r in art["results"]
          if r["arm"] == "prefix-cache-isolation"][0]
    assert pc["n_cache_hits"] == 2
    assert pc["cross_tenant_attributed_hits"] == 1
    assert pc["per_tenant"] == {"attacker": 1}
    rm = [r for r in art["results"]
          if r["arm"] == "response-misattribution"][0]
    assert rm["n_request_ids"] == 3
    assert rm["n_cross_tenant_request_ids"] == 1
    assert rm["misattribution_rate"] == pytest.approx(1 / 3)
    assert all(r["experiment"] == mti.EXPERIMENT_ID for r in art["results"])


def test_journal_clean_journal_reports_no_failures(tmp_path):
    jf = _write_journal(tmp_path, JOURNAL_FIXTURE[:2])
    out = tmp_path / "clean.json"
    args = mti.build_parser().parse_args(
        ["--journal", str(jf), "--output", str(out)])
    args.mode = "inference"
    assert mti.run(args) == 0
    art = json.loads(out.read_text())
    assert all(r["isolation_failures"] == 0 for r in art["results"])


def test_journal_schema_drift_names_file_and_line(tmp_path):
    jf = _write_journal(tmp_path, [JOURNAL_FIXTURE[0]])
    with jf.open("a") as f:
        f.write(json.dumps({"ts": 9.0, "tenant_hint": "x",
                            "request_id": "r9", "phase": "sideways",
                            "cache_hit": True, "bytes": 1}) + "\n")
    with pytest.raises(ValueError, match=r"serving\.jsonl:2"):
        mti.read_journal(str(jf))
    args = mti.build_parser().parse_args(["--journal", str(jf)])
    args.mode = "inference"
    args.output = None
    with pytest.raises(SystemExit, match=r"serving\.jsonl:2"):
        mti.run(args)


def test_journal_missing_keys_and_missing_file(tmp_path):
    bad = _write_journal(tmp_path, [{"ts": 1.0, "request_id": "r1"}],
                         name="bad.jsonl")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        mti.read_journal(str(bad))
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="missing.jsonl"):
        mti.read_journal(str(missing))
    args = mti.build_parser().parse_args(["--journal", str(missing)])
    args.mode = "inference"
    args.output = None
    with pytest.raises(SystemExit, match="missing.jsonl"):
        mti.run(args)


def test_no_input_path_is_descriptive():
    args = mti.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--toy"):
        mti.run(args)
