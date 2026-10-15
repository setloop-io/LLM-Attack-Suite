"""hidden-context-exposure: arm coverage, exact-vs-semantic split, empty
control, app-config errors, determinism, artifact layout."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import hidden_context_exposure as hc


def _run(tmp_path, extra=()):
    out = tmp_path / "hc.json"
    args = hc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert hc.run(args) == 0
    return out, json.loads(out.read_text())


def test_contract_members():
    assert hc.EXPERIMENT_ID == "hidden_context_exposure"
    assert hc.MODES == ("inference",)
    assert callable(hc.build_parser) and callable(hc.run)


def test_toy_artifact_layout_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == hc.ARTIFACT_SCHEMA
    threat_model.validate(art["threat_model"])
    lines = (tmp_path / "hc.json.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(art["results"])
    assert all(json.loads(l)["experiment"] == hc.EXPERIMENT_ID
               for l in lines)


def test_every_arm_and_world_recorded(tmp_path):
    _, art = _run(tmp_path)
    arms = {r["arm"] for r in art["results"]}
    assert set(hc.ARMS) == arms
    for arm in hc.ARMS:
        worlds = {r["world"] for r in art["results"] if r["arm"] == arm}
        assert worlds == {"vulnerable", "defended", "empty"}


def test_vulnerable_leaks_defended_and_empty_do_not(tmp_path):
    _, art = _run(tmp_path)
    s = art["summary"][0]
    assert s["vulnerable_exact_hits"] > 0
    assert s["defended_exact_hits"] == 0
    assert s["empty_control_phantom_hits"] == 0


def test_semantic_scored_separately(tmp_path):
    _, art = _run(tmp_path)
    bi = [r for r in art["results"]
          if r["arm"] == "behavioral-inference" and r["world"] == "vulnerable"]
    assert bi and bi[0]["semantic"]["accuracy"] == 1.0
    assert bi[0]["exact_recovery"] == {}


def test_multi_turn_uses_multiple_turns(tmp_path):
    _, art = _run(tmp_path)
    mt = [r for r in art["results"]
          if r["arm"] == "multi-turn-accumulation"
          and r["world"] == "vulnerable"]
    assert mt and mt[0]["mean_turns"] > 1


def test_determinism(tmp_path):
    _, a = _run(tmp_path)
    out2 = tmp_path / "hc2.json"
    args = hc.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out2)])
    args.mode = "inference"
    hc.run(args)
    assert a["results"] == json.loads(out2.read_text())["results"]


def test_app_config_validation(tmp_path):
    good = tmp_path / "app.json"
    good.write_text(json.dumps({"hidden": {"system_prompt": "x"},
                                "defenses": {"prompt_guard": True}}))
    assert hc.load_app_config(str(good))["defenses"]["prompt_guard"] is True
    bad = tmp_path / "bad.json"
    bad.write_text('{"hidden": {}}')
    with pytest.raises(ValueError, match="bad.json"):
        hc.load_app_config(str(bad))
    with pytest.raises(ValueError, match="missing.json"):
        hc.load_app_config(str(tmp_path / "missing.json"))


def test_endpoint_integration_point():
    args = hc.build_parser().parse_args(
        ["--toy", "--endpoint", "http://127.0.0.1:9"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="INTEGRATION POINT"):
        hc.run(args)


def test_self_test_passes():
    assert hc.self_test() == 0
