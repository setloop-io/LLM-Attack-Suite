"""protocol-integrity (active protocol-level attacks on the split link):
provider registration + schema, toy end-to-end in both worlds, the
mandatory negative control, cross-session metrics, journal input path,
schema-drift errors, determinism, dispatch guards. All torch-free.

NOTE: importing attacker.attacks.protocol_integrity registers the
"protocol-frames" surface provider at import (by design). To keep the
surface registry pristine for the rest of the session, this module imports
lazily inside a fixture and pops the registration it caused on teardown."""

import importlib
import json
import sys

import pytest

from attacker import surfaces, threat_model

FULL_DETECT = ("replay", "duplicate", "cross-session-swap", "epoch-desync",
               "downgrade", "tensor-tamper", "oversize", "truncate")


@pytest.fixture
def pi_mod():
    fresh = "attacker.attacks.protocol_integrity" not in sys.modules
    mod = importlib.import_module("attacker.attacks.protocol_integrity")
    present = "protocol-frames" in surfaces.list_surfaces()
    if not present:
        surfaces.register_surface(mod.ProtocolFramesProvider())
    cleanup = fresh or not present
    yield mod
    if cleanup:
        surfaces._REGISTRY.pop("protocol-frames", None)


def _run_toy(mod, tmp_path, *extra, name="pi.json", mode="inference"):
    args = mod.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(tmp_path / name), *extra])
    args.mode = mode
    assert mod.run(args) == 0
    art = json.loads((tmp_path / name).read_text())
    journal = [json.loads(l) for l in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_torchless_help(pi_mod):
    from attacker.attacks import validate_contract
    assert validate_contract(pi_mod) is pi_mod
    assert pi_mod.EXPERIMENT_ID == "protocol_integrity"
    assert pi_mod.MODES == ("training", "inference")
    assert pi_mod.REQUIRES_LABELS is False
    ap = pi_mod.build_parser()
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--help"])
    assert exc.value.code == 0
    ns = ap.parse_args([])
    for flag in ("journal", "session_keys", "toy", "arms", "world",
                 "output", "seed", "quick", "size_cap"):
        assert hasattr(ns, flag), flag
    assert ns.world == "both" and ns.arms == list(pi_mod.ARMS)


def test_self_test_passes(pi_mod):
    assert pi_mod.self_test() == 0


def test_surface_provider_registered_and_validates(pi_mod, tmp_path):
    prov = surfaces.get_surface("protocol-frames")
    assert isinstance(prov, pi_mod.ProtocolFramesProvider)
    assert prov.modes == ("training", "inference")
    records, _ = pi_mod.make_journal(3, 2, 8, hidden=8)
    norm = prov.validate(records[0])
    assert set(norm) == set(prov.schema)
    with pytest.raises(ValueError, match="schema"):
        prov.validate({"ts": 1.0, "bytes": 4})
    with pytest.raises(ValueError, match="dir"):
        prov.validate({**records[0], "dir": "sideways"})
    with pytest.raises(ValueError, match="auth_tag"):
        prov.validate({**records[0], "auth_tag": "not-hex"})
    with pytest.raises(ValueError, match="mode"):
        prov.rows(mode="sideways", source="x")
    jf = tmp_path / "frames.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records[:4]))
    rows = list(prov.rows(mode="inference", source=str(jf)))
    assert len(rows) == 4 and rows[0][1]["session_id"] == "s0"
    with pytest.raises(ValueError, match="source"):
        list(prov.rows(mode="training"))


def test_toy_artifact_layout_and_threat_model(pi_mod, tmp_path):
    art, journal = _run_toy(pi_mod, tmp_path)
    for key in ("schema", "config", "threat_model", "provenance",
                "results", "summary"):
        assert key in art, key
    assert art["schema"] == "dtraining.attacker.protocol_integrity.v1"
    assert art["config"]["attack"] == pi_mod.EXPERIMENT_ID
    assert art["config"]["journal_schema"] == "llm_attacker.protocol_frames.v1"
    assert threat_model.validate(art["threat_model"]) is art["threat_model"]
    assert art["threat_model"]["actor"]["intent"] == "active"
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == [
        "LLM02", "LLM05", "LLM06"]
    assert "interpretation" in art
    assert len(journal) == len(art["results"])
    assert all("error" not in r for r in art["results"])
    assert art["summary"]


def test_every_arm_and_world_represented(pi_mod, tmp_path):
    art, _ = _run_toy(pi_mod, tmp_path)
    for arm in pi_mod.ARMS:
        for world in ("vulnerable", "defended"):
            recs = [r for r in art["results"]
                    if r["arm"] == arm and r["world"] == world]
            assert recs, (arm, world)
            assert all(r["experiment"] == pi_mod.EXPERIMENT_ID for r in recs)
    control = [r for r in art["results"] if r["arm"] == "negative-control"]
    assert control and all(r["world"] == "defended" for r in control)


def test_negative_control_zero_false_rejections(pi_mod, tmp_path):
    art, _ = _run_toy(pi_mod, tmp_path)
    for r in art["results"]:
        if r["arm"] == "negative-control":
            assert r["n_rejected"] == 0
            assert r["false_rejection_rate"] == 0.0
    summ = [s for s in art["summary"] if s["arm"] == "negative-control"][0]
    assert summ["false_rejection_rate_mean"] == 0.0


def test_defended_world_detects_mutations(pi_mod, tmp_path):
    art, _ = _run_toy(pi_mod, tmp_path)
    defended = [r for r in art["results"] if r["world"] == "defended"
                and r["arm"] != "negative-control"]
    assert defended
    for r in defended:
        assert set(r["rejection_reasons"]) <= pi_mod.REJECTION_REASONS
        assert r["benign_rejection_rate"] == 0.0
        assert r["n_benign_rejected"] == 0
        assert r["crash_hang_rate"] == 0.0
        if r["arm"] in FULL_DETECT:
            assert r["undetected_mutation_rate"] == 0.0, r["arm"]
            assert r["n_detected"] == r["n_events"]
            assert r["mttd_frames"] >= 1
        elif r["arm"] == "reorder":
            assert 0.0 < r["undetected_mutation_rate"] < 1.0
            assert r["n_detected"] >= 1
        else:
            assert r["arm"] == "drop"
            assert r["n_detected"] == 0
            assert r["undetected_mutation_rate"] == 1.0


def test_vulnerable_world_accepts_everything(pi_mod, tmp_path):
    """The attack-works baseline: without it the controls would be vacuous."""
    art, _ = _run_toy(pi_mod, tmp_path)
    vulnerable = [r for r in art["results"] if r["world"] == "vulnerable"]
    assert vulnerable
    for r in vulnerable:
        assert r["n_detected"] == 0
        assert r["undetected_mutation_rate"] == 1.0
        assert r["rejection_reasons"] == []
        assert r["benign_rejection_rate"] == 0.0


def test_cross_session_acceptance_metric(pi_mod, tmp_path):
    art, _ = _run_toy(pi_mod, tmp_path)
    swap = [r for r in art["results"] if r["arm"] == "cross-session-swap"]
    assert swap
    for r in swap:
        assert r["n_foreign"] == 1
        if r["world"] == "defended":
            assert r["cross_session_acceptance_rate"] == 0.0
            assert r["rejection_reasons"] == ["session-mismatch"]
        else:
            assert r["cross_session_acceptance_rate"] == 1.0


def test_determinism_same_seed_same_metrics(pi_mod, tmp_path):
    a1, _ = _run_toy(pi_mod, tmp_path, name="d1.json")
    a2, _ = _run_toy(pi_mod, tmp_path, name="d2.json")
    assert a1["results"] == a2["results"]
    assert a1["summary"] == a2["summary"]


def test_journal_input_path(pi_mod, tmp_path):
    """A protocol-frames journal on disk plus the endpoints' key file runs
    the same cells; keys never enter the artifact."""
    records, keys = pi_mod.make_journal(11, 3, 12, hidden=8)
    jf = tmp_path / "frames.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps(keys))
    args = pi_mod.build_parser().parse_args(
        ["--journal", str(jf), "--session-keys", str(kf),
         "--output", str(tmp_path / "j.json")])
    args.mode = "inference"
    assert pi_mod.run(args) == 0
    art = json.loads((tmp_path / "j.json").read_text())
    assert art["config"]["source"] == "journal"
    assert art["config"]["journal"] == str(jf)
    blob = json.dumps(art)
    assert not any(k["0"] in blob or k["1"] in blob for k in keys.values())
    control = [r for r in art["results"] if r["arm"] == "negative-control"]
    assert control and all(r["false_rejection_rate"] == 0.0 for r in control)
    defended = [r for r in art["results"] if r["world"] == "defended"
                and r["arm"] in FULL_DETECT]
    assert defended and all(r["undetected_mutation_rate"] == 0.0
                            for r in defended)


def test_journal_schema_drift_is_descriptive(pi_mod, tmp_path):
    records, _ = pi_mod.make_journal(5, 2, 8, hidden=8)
    jf = tmp_path / "bad.jsonl"
    jf.write_text(json.dumps(records[0]) + "\n"
                  + json.dumps({"ts": 1.0, "bytes": 4}) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        pi_mod.read_journal(str(jf))
    jf.write_text(json.dumps({**records[0], "dir": "up"}) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        pi_mod.read_journal(str(jf))
    jf.write_text("not json\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        pi_mod.read_journal(str(jf))
    with pytest.raises(ValueError, match="missing.jsonl"):
        pi_mod.read_journal(str(tmp_path / "missing.jsonl"))


def test_journal_mode_requires_session_keys(pi_mod, tmp_path):
    records, _ = pi_mod.make_journal(5, 2, 8, hidden=8)
    jf = tmp_path / "frames.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in records))
    args = pi_mod.build_parser().parse_args(["--journal", str(jf)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--session-keys"):
        pi_mod.run(args)
    args = pi_mod.build_parser().parse_args(
        ["--journal", str(jf), "--session-keys",
         str(tmp_path / "nope.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="nope.json"):
        pi_mod.run(args)
    kf = tmp_path / "keys.json"
    kf.write_text(json.dumps({"s0": {"0": "ab"}}))
    args = pi_mod.build_parser().parse_args(
        ["--journal", str(jf), "--session-keys", str(kf)])
    args.mode = "inference"
    with pytest.raises(ValueError, match="s1"):
        pi_mod.run(args)


def test_no_input_path_is_descriptive(pi_mod):
    args = pi_mod.build_parser().parse_args([])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="--journal"):
        pi_mod.run(args)


def test_surface_registry_left_pristine(pi_mod):
    assert "protocol-frames" in surfaces.list_surfaces()
