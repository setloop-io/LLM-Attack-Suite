"""output-sink-injection: per-sink arms, defended/vulnerable contrast,
adaptive residual bypass, benign battery, corpus errors, artifact layout."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import output_sink_injection as osi


def _run(tmp_path, extra=()):
    out = tmp_path / "osi.json"
    args = osi.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert osi.run(args) == 0
    return out, json.loads(out.read_text())


def test_contract_members():
    assert osi.EXPERIMENT_ID == "output_sink_injection"
    assert osi.MODES == ("inference",)
    assert osi.REQUIRES_LABELS is False


def test_toy_artifact_layout_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == osi.ARTIFACT_SCHEMA
    threat_model.validate(art["threat_model"])
    lines = (tmp_path / "osi.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(art["results"])
    assert all(json.loads(l)["experiment"] == osi.EXPERIMENT_ID
               for l in lines)


def test_every_arm_recorded_in_both_worlds(tmp_path):
    _, art = _run(tmp_path)
    for arm in osi.ARMS:
        worlds = {r["world"] for r in art["results"] if r["arm"] == arm}
        assert worlds == {"vulnerable", "defended"}, arm


def test_vulnerable_pipeline_shows_payloads_land(tmp_path):
    _, art = _run(tmp_path)
    vuln = [r for r in art["results"] if r.get("world") == "vulnerable"]
    assert vuln and sum(r["effects"] for r in vuln) > 0
    for r in vuln:
        if r["arm"] != "adaptive":
            assert r["effects"] > 0, r["arm"]


def test_defended_base_arms_have_zero_effects(tmp_path):
    _, art = _run(tmp_path)
    for r in art["results"]:
        if r.get("world") == "defended" and r["arm"] != "adaptive":
            assert r["effects"] == 0, r["arm"]


def test_adaptive_arm_finds_residual_bypass(tmp_path):
    _, art = _run(tmp_path)
    adp = [r for r in art["results"] if r.get("arm") == "adaptive"
           and r.get("world") == "defended"]
    assert adp and adp[0]["effects"] > 0
    assert adp[0]["effects"] < adp[0]["trials"]


def test_benign_battery_not_false_blocked(tmp_path):
    _, art = _run(tmp_path)
    assert art["summary"][0]["defended_benign_blocked"] == 0


def test_outputs_corpus_replay_and_drift(tmp_path):
    good = tmp_path / "o.jsonl"
    good.write_text(json.dumps({"sink": "sql",
                                "output": "alice' OR '1'='1' --"}) + "\n")
    _, art = _run(tmp_path, ["--outputs", str(good)])
    assert art["config"]["outputs"] == str(good)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"sink": "sql"}\n')
    args = osi.build_parser().parse_args(
        ["--toy", "--quick", "--outputs", str(bad)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="bad.jsonl"):
        osi.run(args)
    bad2 = tmp_path / "bad2.jsonl"
    bad2.write_text('{"sink": "laser", "output": "x"}\n')
    args = osi.build_parser().parse_args(
        ["--toy", "--quick", "--outputs", str(bad2)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="unknown sink"):
        osi.run(args)


def test_determinism(tmp_path):
    _, a = _run(tmp_path)
    out2 = tmp_path / "osi2.json"
    args = osi.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out2)])
    args.mode = "inference"
    osi.run(args)
    assert a["results"] == json.loads(out2.read_text())["results"]


def test_nothing_executes_payloads():
    farm = osi.SinkFarm(defenses={})
    r = farm.process("shell", "rm -rf /")
    assert r["effect"] is True and farm.recorded


def test_self_test_passes():
    assert osi.self_test() == 0
