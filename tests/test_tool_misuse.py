"""tool-misuse attack module (attacker/attacks/tool_misuse.py): the toy
tier end to end, the three exact-outcome cases, the authorized-task control,
a parser/schema error, fail-closed journalling, and artifact layout."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import tool_misuse as tm
from attacker.attacks import validate_contract, load, ATTACKS


def _run(tmp_path, extra=(), name="tm.json"):
    out = tmp_path / name
    args = tm.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert tm.run(args) == 0
    return out, json.loads(out.read_text())


def test_contract_members():
    assert tm.EXPERIMENT_ID == "tool_misuse"
    assert tm.MODES == ("inference",)
    assert tm.REQUIRES_LABELS is False
    assert isinstance(tm.DESCRIPTION, str)
    assert callable(tm.build_parser) and callable(tm.run)
    validate_contract(tm, "tool-misuse")


def test_registered_in_attacks():
    assert ATTACKS.get("tool-misuse") == "attacker.attacks.tool_misuse"
    assert load("tool-misuse").EXPERIMENT_ID == "tool_misuse"


def test_toy_run_artifact_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == tm.ARTIFACT_SCHEMA
    for k in ("config", "threat_model", "provenance", "results", "summary"):
        assert k in art
    threat_model.validate(art["threat_model"])
    assert art["threat_model"]["scope"]["use_case"] == "agentic"
    assert art["config"]["attack"] == tm.EXPERIMENT_ID
    lines = (tmp_path / "tm.json.jsonl").read_text().strip().splitlines()
    assert all(json.loads(ln)["experiment"] == tm.EXPERIMENT_ID
               for ln in lines)


def test_every_arm_and_world_present(tmp_path):
    _, art = _run(tmp_path)
    cells = {(r["arm"], r["world"]) for r in art["results"]
             if "error" not in r}
    expected = {(a, w) for a in tm.ARMS for w in ("vulnerable", "defended")}
    assert cells == expected


def test_case1_vulnerable_violation_fires(tmp_path):
    _, art = _run(tmp_path)
    dp_v = _cell(art, "dangerous-parameters", "vulnerable")
    assert dp_v["total_violations"] > 0
    assert dp_v["violation_rate"] > 0
    assert dp_v["violations"]["param_cap_exceeded"] > 0


def test_case2_defended_blocked_no_effect(tmp_path):
    _, art = _run(tmp_path)
    dp_d = _cell(art, "dangerous-parameters", "defended")
    assert dp_d["total_violations"] == 0
    assert dp_d["unauthorized_effects"] == 0
    assert dp_d["wilson_upper"] > 0


def test_case3_control_executes_in_both_worlds(tmp_path):
    _, art = _run(tmp_path)
    for world in ("vulnerable", "defended"):
        ctrl = _cell(art, "authorized-task", world)
        assert ctrl["total_violations"] == 0
        assert ctrl["false_blocks"] == 0


def test_scope_confusion_and_amplification(tmp_path):
    _, art = _run(tmp_path)
    sc_v = _cell(art, "scope-confusion", "vulnerable")
    sc_d = _cell(art, "scope-confusion", "defended")
    assert sc_v["total_violations"] > 0 and sc_d["total_violations"] == 0
    assert sc_v["violations"]["principal_not_owner"] > 0
    amp_v = _cell(art, "side-effect-amplification", "vulnerable")
    amp_d = _cell(art, "side-effect-amplification", "defended")
    assert amp_v["unauthorized_effects"] > amp_v["trials"]
    assert amp_v["violations"]["over_authorized_budget"] > 0
    assert amp_d["total_violations"] == 0


def test_summary_is_non_vacuous(tmp_path):
    _, art = _run(tmp_path)
    s = art["summary"][0]
    assert s["defended_total_violations"] == 0
    assert s["vulnerable_total_violations"] > 0
    assert s["control_false_blocks"] == 0


def test_failed_cell_lands_in_results_not_only_sidecar(tmp_path, monkeypatch):
    out = tmp_path / "tm.json"
    calls = {"n": 0}
    real = tm.run_arm

    def boom(arm, world, seed, n_seeds):
        if arm == "scope-confusion" and world == "vulnerable":
            calls["n"] += 1
            raise RuntimeError("injected cell failure")
        return real(arm, world, seed, n_seeds)

    monkeypatch.setattr(tm, "run_arm", boom)
    args = tm.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out)])
    args.mode = "inference"
    assert tm.run(args) == 0
    art = json.loads(out.read_text())
    errs = [r for r in art["results"] if "error" in r]
    assert calls["n"] == 1
    assert any(r.get("arm") == "scope-confusion" and r.get("world")
               == "vulnerable" for r in errs)


def test_bad_arm_is_a_parser_error():
    with pytest.raises(SystemExit):
        tm.build_parser().parse_args(["--arms", "no-such-arm"])


def test_endpoint_is_an_integration_point():
    args = tm.build_parser().parse_args(["--endpoint", "https://agent.local"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="INTEGRATION POINT"):
        tm.run(args)


def test_determinism_same_seed_same_results(tmp_path):
    _, a = _run(tmp_path, name="a.json")
    _, b = _run(tmp_path, name="b.json")
    ra = {(r["arm"], r["world"]): r["total_violations"] for r in a["results"]
          if "error" not in r}
    rb = {(r["arm"], r["world"]): r["total_violations"] for r in b["results"]
          if "error" not in r}
    assert ra == rb


def test_real_artifact_covers_asi02_under_applicable_agentic(tmp_path):
    """acceptance 3(a) end-to-end: a REAL tool-misuse artifact (all arms)
    flips ASI02 to COVERED under --applicable-agentic; without the opt-in it
    stays OUT_OF_SCOPE."""
    from attacker import coverage
    _, art = _run(tmp_path)
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "COVERED"
    assert m["threats"]["ASI02"]["mapped_attacks"] == ["tool-misuse"]
    m0 = coverage.build_manifest([art])
    assert m0["threats"]["ASI02"]["status"] == "OUT_OF_SCOPE"


def test_self_test_passes():
    assert tm.self_test() == 0


def _cell(art, arm, world):
    for r in art["results"]:
        if r.get("arm") == arm and r.get("world") == world \
                and "error" not in r:
            return r
    raise AssertionError(f"no cell for {arm}/{world}")
