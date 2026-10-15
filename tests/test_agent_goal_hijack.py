"""agent-goal-hijack attack module (attacker/attacks/agent_goal_hijack.py):
the toy tier end to end, the three exact-outcome cases, the MANDATORY
provenance ablation (none vs provenance-only), the benign control, a
parser/schema error, fail-closed journalling, artifact layout, the ASI01
coverage flip, and that adding instruction_provenance leaves ASI02 green."""

import json

import pytest

from attacker import threat_model
from attacker.attacks import agent_goal_hijack as agh
from attacker.attacks import tool_misuse as tm
from attacker.attacks import validate_contract, load, ATTACKS


def _run(tmp_path, extra=(), name="agh.json"):
    out = tmp_path / name
    args = agh.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out), *extra])
    args.mode = "inference"
    assert agh.run(args) == 0
    return out, json.loads(out.read_text())


def _cell(art, arm, world):
    for r in art["results"]:
        if r.get("arm") == arm and r.get("world") == world \
                and "error" not in r:
            return r
    raise AssertionError(f"no cell for {arm}/{world}")


def test_contract_members():
    assert agh.EXPERIMENT_ID == "agent_goal_hijack"
    assert agh.MODES == ("inference",)
    assert agh.REQUIRES_LABELS is False
    assert isinstance(agh.DESCRIPTION, str)
    assert callable(agh.build_parser) and callable(agh.run)
    validate_contract(agh, "agent-goal-hijack")


def test_registered_in_attacks():
    assert ATTACKS.get("agent-goal-hijack") == \
        "attacker.attacks.agent_goal_hijack"
    assert load("agent-goal-hijack").EXPERIMENT_ID == "agent_goal_hijack"


def test_toy_run_artifact_and_journal(tmp_path):
    out, art = _run(tmp_path)
    assert art["schema"] == agh.ARTIFACT_SCHEMA
    for k in ("config", "threat_model", "provenance", "results", "summary"):
        assert k in art
    threat_model.validate(art["threat_model"])
    assert art["threat_model"]["scope"]["use_case"] == "agentic"
    assert art["threat_model"]["mappings"]["owasp_agentic_2026"] == ["ASI01"]
    assert art["threat_model"]["mappings"]["owasp_llm_2026"] == ["LLM01"]
    assert art["config"]["attack"] == agh.EXPERIMENT_ID
    lines = (tmp_path / "agh.json.jsonl").read_text().strip().splitlines()
    assert all(json.loads(ln)["experiment"] == agh.EXPERIMENT_ID
               for ln in lines)


def test_every_arm_and_world_present(tmp_path):
    _, art = _run(tmp_path)
    cells = {(r["arm"], r["world"]) for r in art["results"]
             if "error" not in r}
    expected = {(a, w) for a in agh.ARMS for w in agh.WORLDS}
    assert cells == expected


@pytest.mark.parametrize("arm", ["tool-output-hijack",
                                 "retrieved-document-hijack"])
def test_case_a_vulnerable_violation_fires(tmp_path, arm):
    _, art = _run(tmp_path)
    v = _cell(art, arm, "vulnerable")
    assert v["total_violations"] > 0
    assert v["violation_rate"] > 0
    assert v["violations"]["unsanctioned_effect"] > 0


@pytest.mark.parametrize("arm", ["tool-output-hijack",
                                 "retrieved-document-hijack"])
def test_case_b_defended_silent(tmp_path, arm):
    _, art = _run(tmp_path)
    d = _cell(art, arm, "defended")
    assert d["total_violations"] == 0
    assert d["wilson_upper"] > 0
    assert d["false_blocks"] == 0


def test_case_c_benign_control_executes_in_every_world(tmp_path):
    _, art = _run(tmp_path)
    for world in agh.WORLDS:
        ctrl = _cell(art, "benign-task", world)
        assert ctrl["total_violations"] == 0
        assert ctrl["false_blocks"] == 0


@pytest.mark.parametrize("arm", ["tool-output-hijack",
                                 "retrieved-document-hijack"])
def test_case_d_provenance_ablation_flips_on_the_toggle(tmp_path, arm):
    """none (all defenses off) FIRES; provenance-only (ONLY
    instruction_provenance on, identical everything else) is SILENT. Toggling
    only provenance flips the outcome on the clean own-order->attacker-account
    redirect that no OTHER defense catches — proving the control load-bearing."""
    _, art = _run(tmp_path)
    none = _cell(art, arm, "none")
    prov = _cell(art, arm, "provenance-only")
    assert none["total_violations"] > 0
    assert none["violations"]["unsanctioned_effect"] > 0
    assert prov["total_violations"] == 0
    assert prov["false_blocks"] == 0
    assert agh.WORLD_DEFENSES["none"] == {}
    assert agh.WORLD_DEFENSES["provenance-only"] == \
        {"instruction_provenance": True}


def test_summary_records_the_ablation(tmp_path):
    _, art = _run(tmp_path)
    s = art["summary"][0]
    assert s["vulnerable_total_violations"] > 0
    assert s["none_total_violations"] > 0
    assert s["defended_total_violations"] == 0
    assert s["provenance_only_total_violations"] == 0
    assert s["control_false_blocks"] == 0


def test_failed_cell_lands_in_results_not_only_sidecar(tmp_path, monkeypatch):
    out = tmp_path / "agh.json"
    calls = {"n": 0}
    real = agh.run_arm

    def boom(arm, world, seed, n_seeds):
        if arm == "tool-output-hijack" and world == "vulnerable":
            calls["n"] += 1
            raise RuntimeError("injected cell failure")
        return real(arm, world, seed, n_seeds)

    monkeypatch.setattr(agh, "run_arm", boom)
    args = agh.build_parser().parse_args(
        ["--toy", "--quick", "--output", str(out)])
    args.mode = "inference"
    assert agh.run(args) == 0
    art = json.loads(out.read_text())
    errs = [r for r in art["results"] if "error" in r]
    assert calls["n"] == 1
    assert any(r.get("arm") == "tool-output-hijack"
               and r.get("world") == "vulnerable" for r in errs)


def test_bad_arm_is_a_parser_error():
    with pytest.raises(SystemExit):
        agh.build_parser().parse_args(["--arms", "no-such-arm"])


def test_endpoint_is_an_integration_point():
    args = agh.build_parser().parse_args(["--endpoint", "https://agent.local"])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="INTEGRATION POINT"):
        agh.run(args)


def test_help_is_torch_less():
    assert agh.build_parser().format_help()


def test_determinism_same_seed_same_results(tmp_path):
    _, a = _run(tmp_path, name="a.json")
    _, b = _run(tmp_path, name="b.json")
    ra = {(r["arm"], r["world"]): r["total_violations"] for r in a["results"]
          if "error" not in r}
    rb = {(r["arm"], r["world"]): r["total_violations"] for r in b["results"]
          if "error" not in r}
    assert ra == rb


def test_real_artifact_covers_asi01_under_applicable_agentic(tmp_path):
    """A REAL agent-goal-hijack artifact (all arms x worlds) flips ASI01 to
    COVERED under --applicable-agentic; without the opt-in it is OUT_OF_SCOPE."""
    from attacker import coverage
    _, art = _run(tmp_path)
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert m["threats"]["ASI01"]["status"] == "COVERED"
    assert m["threats"]["ASI01"]["mapped_attacks"] == ["agent-goal-hijack"]
    m0 = coverage.build_manifest([art])
    assert m0["threats"]["ASI01"]["status"] == "OUT_OF_SCOPE"


def test_adding_provenance_leaves_asi02_green():
    """(f) instruction_provenance is inert for ASI02 (no injected content),
    so the tool-misuse battery still passes with it in shared ALL_DEFENSES."""
    assert "instruction_provenance" in tm.DEFENDED
    assert tm.self_test() == 0


def test_self_test_passes():
    assert agh.self_test() == 0
