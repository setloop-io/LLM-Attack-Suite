"""key-session-abuse (lifecycle abuse of seam-defense secrets): contract +
torchless parser, toy end-to-end over both worlds, the defended-world
negative control (and the non-vacuous vulnerable world), per-arm
semantics, the real --scan-log integration path, malformed-input errors,
determinism, and the self-test. All torch-free."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import key_session_abuse as ksa


def _run_toy(tmp_path, *extra, name="ksa.json"):
    args = ksa.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = "training"
    assert ksa.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help():
    from attacker.attacks import validate_contract
    assert validate_contract(ksa) is ksa
    assert ksa.EXPERIMENT_ID == "key_session_abuse"
    assert ksa.MODES == ("training", "inference")
    assert ksa.REQUIRES_LABELS == "insider (protocol-state)"
    ap = ksa.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    flags = {a for opt in ap._actions for a in opt.option_strings}
    assert {"--toy", "--quick", "--arms", "--world", "--epochs",
            "--sessions", "--frames", "--scan-log", "--known-keys",
            "--seed", "--output"} <= flags


def test_self_test_passes():
    assert ksa.self_test() == 0


def test_toy_quick_writes_artifact_and_journal(tmp_path):
    art, journal = _run_toy(tmp_path)
    assert art["schema"] == "dtraining.attacker.key_session_abuse.v1"
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art
    assert art["config"]["attack"] == ksa.EXPERIMENT_ID
    assert art["config"]["toy"] is True
    assert art["config"]["world"] == "both"
    assert len(journal) == len(art["results"]) > 0
    assert all(r["experiment"] == ksa.EXPERIMENT_ID for r in journal)
    assert art["summary"]


def test_artifact_threat_model_is_valid_v2(tmp_path):
    art, _ = _run_toy(tmp_path)
    doc = threat_model.validate(art["threat_model"])
    assert doc["schema"] == "llm_attacker.threat_model.v2"
    assert doc["actor"]["intent"] == "malicious"
    assert doc["actor"]["persistence"] == "cross-epoch"
    assert set(doc["controls"]["tested"]) == set(ksa.DEFENSES)
    assert doc["mappings"]["owasp_llm_2026"] == ["LLM02", "LLM04", "LLM08"]


def test_every_arm_produces_records_in_both_worlds(tmp_path):
    art, _ = _run_toy(tmp_path)
    for arm in ksa.ARMS:
        for world in ("vulnerable", "defended"):
            recs = [r for r in art["results"]
                    if r["arm"] == arm and r["world"] == world]
            assert recs, f"{arm}/{world} missing"
            assert all(r["experiment"] == ksa.EXPERIMENT_ID for r in recs)


def test_negative_control_defended_world_rejects_everything(tmp_path):
    art, _ = _run_toy(tmp_path)
    defended = [r for r in art["results"] if r["world"] == "defended"]
    assert defended and all(r["role"] == "negative-control" for r in defended)
    assert {r["arm"] for r in defended} == set(ksa.ARMS)
    assert all(r["attack_succeeded"] is False for r in defended)
    by_arm = {r["arm"]: r for r in defended}
    assert by_arm["nonce-reuse"]["same_scope_reuse_pairs"] == 0
    assert by_arm["nonce-reuse"]["rejected_reuse_attempts"] > 0
    assert by_arm["session-binding"]["foreign_acceptance_rate"] == 0.0
    assert by_arm["epoch-rollback"]["detection_delay_ops"] == 0
    assert by_arm["master-seed-exposure"]["sessions_compromised"] == 0
    assert by_arm["master-seed-exposure"]["master_present"] is False
    assert by_arm["forward-secrecy"]["past_recovery_fraction"] == 0.0
    assert by_arm["forward-secrecy"]["future_recovery_fraction"] == 0.0
    assert by_arm["key-in-logs"]["n_leaks"] == 0
    assert by_arm["key-in-logs"]["scrub_markers"] > 0
    assert by_arm["negotiation-downgrade"]["downgrade_accepted"] is False
    assert by_arm["collusion"]["current_session_recovery"] == 0.0
    assert by_arm["collusion"]["beyond_stale_epoch"] is False


def test_isolated_control_cells(tmp_path):
    art, _ = _run_toy(tmp_path)
    controls = [r for r in art["results"] if r["world"] == "control"]
    assert controls and all(r["arm"] == "key-in-logs" for r in controls)
    assert all(r["role"] == "isolated-control" for r in controls)
    assert all(r["n_leaks"] == 0 and r["attack_succeeded"] is False
               for r in controls)


def test_vulnerable_world_attack_works_non_vacuous(tmp_path):
    art, _ = _run_toy(tmp_path)
    vuln = [r for r in art["results"] if r["world"] == "vulnerable"]
    assert {r["arm"] for r in vuln} == set(ksa.ARMS)
    assert all(r["attack_succeeded"] is True for r in vuln)
    by_arm = {r["arm"]: r for r in vuln}
    nr = by_arm["nonce-reuse"]
    assert nr["same_scope_reuse_pairs"] > 0
    assert nr["same_scope_stream_identity"] == 1.0
    assert nr["two_time_pad_xor_leak_pairs"] == nr["same_scope_reuse_pairs"]
    assert nr["cross_scope_stream_identity"] < 0.2
    ms = by_arm["master-seed-exposure"]
    assert ms["blast_radius"] == 1.0
    assert ms["sessions_compromised"] == ms["total_sessions"] > 1
    assert ms["one_disclosure_one_scope"] is False
    rb = by_arm["epoch-rollback"]
    assert rb["rollback_acceptance_rate"] == 1.0
    assert rb["detection_delay_ops"] is None
    assert rb["stale_state_available"] is True
    fs = by_arm["forward-secrecy"]
    assert fs["past_recovery_fraction"] == 1.0
    assert fs["future_recovery_fraction"] == 1.0
    co = by_arm["collusion"]
    assert co["stale_epoch_recovery"] == 1.0
    assert co["current_session_recovery"] == 1.0
    assert co["control_recovery"] == 0.0
    co_d = [r for r in art["results"]
            if r["arm"] == "collusion" and r["world"] == "defended"][0]
    assert co_d["stale_epoch_recovery"] == 1.0
    for w in ("vulnerable", "defended"):
        sb = [r for r in art["results"]
              if r["arm"] == "session-binding" and r["world"] == w][0]
        assert sb["own_acceptance_rate"] == 1.0


def test_world_subset_skips_the_other(tmp_path):
    art, _ = _run_toy(tmp_path, "--world", "vulnerable")
    assert {r["world"] for r in art["results"]} == {"vulnerable", "control"}
    assert art["config"]["world"] == "vulnerable"


def test_arms_subset(tmp_path):
    art, _ = _run_toy(tmp_path, "--arms", "epoch-rollback", "collusion")
    assert {r["arm"] for r in art["results"]} == {"epoch-rollback",
                                                  "collusion"}


def test_determinism_same_seed_identical_metrics(tmp_path):
    art1, _ = _run_toy(tmp_path, name="a.json")
    art2, _ = _run_toy(tmp_path, name="b.json")
    assert art1["results"] == art2["results"]
    assert art1["summary"] == art2["summary"]


def test_scan_log_integration_path(tmp_path):
    key = ksa.kdf("epoch", b"\x11" * 32, 0).hex()
    log = tmp_path / "seam.log"
    log.write_text(f"INFO loaded master seed {key}\nINFO rotation ok\n")
    keys = tmp_path / "keys.txt"
    keys.write_text(f"master={key}\n# comment\n\n")
    out = tmp_path / "scan.json"
    args = ksa.build_parser().parse_args(
        ["--scan-log", str(log), "--known-keys", str(keys),
         "--output", str(out)])
    args.mode = "inference"
    assert ksa.run(args) == 0
    art = json.loads(out.read_text())
    rec = art["results"][0]
    assert rec["arm"] == "key-in-logs" and rec["world"] == "real"
    assert rec["n_leaks"] == 1 and rec["attack_succeeded"] is True
    assert any("master" in item for item in rec["leaked_items"])
    threat_model.validate(art["threat_model"])
    log.write_text("INFO all nominal\n")
    assert ksa.run(args) == 0
    assert json.loads(out.read_text())["results"][0]["n_leaks"] == 0


def test_malformed_known_keys_descriptive(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("not-hex-material\n")
    with pytest.raises(ValueError, match=r"bad\.txt:1"):
        ksa.read_known_keys(str(bad))
    empty = tmp_path / "empty.txt"
    empty.write_text("# only comments\n")
    with pytest.raises(ValueError, match="empty.txt"):
        ksa.read_known_keys(str(empty))


def test_missing_files_are_descriptive_system_exit(tmp_path):
    args = ksa.build_parser().parse_args(
        ["--scan-log", str(tmp_path / "x.log"),
         "--known-keys", str(tmp_path / "missing-keys.txt")])
    with pytest.raises(SystemExit, match="missing-keys.txt"):
        ksa.run(args)
    keys = tmp_path / "keys.txt"
    keys.write_text(ksa.kdf("epoch", b"\x00" * 32, 0).hex() + "\n")
    args = ksa.build_parser().parse_args(
        ["--scan-log", str(tmp_path / "missing.log"),
         "--known-keys", str(keys)])
    with pytest.raises(SystemExit, match="missing.log"):
        ksa.run(args)
    args = ksa.build_parser().parse_args(
        ["--scan-log", str(tmp_path / "missing.log")])
    with pytest.raises(SystemExit, match="--known-keys"):
        ksa.run(args)


def test_no_input_path_is_descriptive():
    args = ksa.build_parser().parse_args([])
    args.mode = "training"
    with pytest.raises(SystemExit, match="--toy"):
        ksa.run(args)


def test_summary_n_seeds_counts_distinct_seeds_not_records(tmp_path):
    """The key-in-logs isolated benign-log control emits one record per
    (seed, world), so its cell holds two records per seed. Counting records
    would let a 1-seed run report n_seeds=2 and clear a >=3-seed guard."""
    art, _ = _run_toy(tmp_path)
    assert art["config"]["seeds"] == [0]
    by = {}
    for r in art["results"]:
        by.setdefault((r["arm"], r["world"]), []).append(r)
    assert art["summary"]
    for row in art["summary"]:
        recs = by[(row["arm"], row["world"])]
        assert row["n_seeds"] == len({r["seed"] for r in recs})
    ctl_recs = by[("key-in-logs", "control")]
    assert len(ctl_recs) == 2 and {r["seed"] for r in ctl_recs} == {0}
    ctl = [r for r in art["summary"]
           if (r["arm"], r["world"]) == ("key-in-logs", "control")]
    assert len(ctl) == 1 and ctl[0]["n_seeds"] == 1


def test_full_tier_key_in_logs_control_reports_three_distinct_seeds(tmp_path):
    """Full tier: six control records over three distinct seeds must report
    n_seeds=3, not 6."""
    args = ksa.build_parser().parse_args(
        ["--toy", "--output", str(tmp_path / "full.json")])
    args.mode = "training"
    assert ksa.run(args) == 0
    art = json.loads((tmp_path / "full.json").read_text())
    recs = [r for r in art["results"]
            if (r["arm"], r["world"]) == ("key-in-logs", "control")]
    assert len(recs) == 6 and {r["seed"] for r in recs} == {0, 1, 2}
    ctl = next(r for r in art["summary"]
               if (r["arm"], r["world"]) == ("key-in-logs", "control"))
    assert ctl["n_seeds"] == 3


def test_duplicate_seeds_on_the_command_line_are_not_double_counted(tmp_path):
    """A repeated --seeds value produces duplicate records per seed; the
    reported n_seeds must stay the distinct count."""
    args = ksa.build_parser().parse_args(
        ["--toy", "--seeds", "0", "0", "1",
         "--output", str(tmp_path / "dup.json")])
    args.mode = "training"
    assert ksa.run(args) == 0
    art = json.loads((tmp_path / "dup.json").read_text())
    assert art["config"]["seeds"] == [0, 0, 1]
    by = {}
    for r in art["results"]:
        by.setdefault((r["arm"], r["world"]), []).append(r)
    for row in art["summary"]:
        recs = by[(row["arm"], row["world"])]
        assert row["n_seeds"] == len({r["seed"] for r in recs}) == 2
        assert len(recs) > row["n_seeds"]
