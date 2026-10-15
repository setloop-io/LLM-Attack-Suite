"""Benchmark profiles (attacker/profiles.py): the four versioned profiles,
per-threat COVERED/PARTIAL/UNMEASURED/OUT_OF_SCOPE semantics, the
agentic applicability rule, and the CLI."""

import json
import subprocess
import sys

import pytest

from attacker import coverage, profiles


def art(exp, ok=True):
    rec = {"experiment": exp, "metric": 1.0}
    if not ok:
        rec = {"experiment": exp, "error": "boom"}
    return {"schema": "dtraining.attacker.test.v1", "config": {},
            "threat_model": "t", "provenance": {}, "results": [rec],
            "summary": []}


def _complete_art(experiment):
    """A COMPLETE artifact for one experiment: the full required cell set when
    it is COMPLETENESS-gated (so the C5 guard counts it measured), else a
    single plain record. Mirrors test_coverage._complete_records — needed for
    the split-privacy.v2 / genai-app.v1 modules gated in Lane C."""
    req = coverage.COMPLETENESS.get(experiment)
    if req is None:
        recs = [{"experiment": experiment, "metric": 1.0}]
    elif "cells" in req:
        recs = [{"experiment": experiment, "arm": a, "world": w,
                 **{k: 1.0 for k in metrics}}
                for (a, w), metrics in req["cells"].items()]
    else:
        recs = [{"experiment": experiment, "arm": a, "world": w,
                 **{k: 1.0 for k in req["metrics"]}}
                for a in req["arms"] for w in req["worlds"]]
    return {"schema": "dtraining.attacker.test.v1", "config": {},
            "threat_model": "t", "provenance": {}, "results": recs,
            "summary": []}


def test_fifteen_versioned_profiles():
    assert set(profiles.PROFILES) == {
        "benchmark.split-privacy.v2",
        "benchmark.split-integrity-availability.v1",
        "benchmark.genai-app.v1", "benchmark.agentic.v1",
        "benchmark.agentic.v2", "benchmark.agentic.v3",
        "benchmark.agentic.v4", "benchmark.agentic.v5",
        "benchmark.agentic.v6", "benchmark.agentic.v7",
        "benchmark.agentic.v8", "benchmark.agentic.v9",
        "benchmark.agentic.v10", "benchmark.agentic.v11",
        "benchmark.agentic.v12"}
    for name, spec in profiles.PROFILES.items():
        assert name.startswith("benchmark.")
        assert spec["description"]
        assert spec["threats"]


def test_agentic_v2_requires_tool_misuse_but_status_is_additive():
    """C6: the v2 profile lists tool-misuse as a required attack, but ASI02
    status derives from the GLOBAL manifest, not from required_attacks."""
    spec = profiles.PROFILES["benchmark.agentic.v2"]
    assert spec["required_attacks"] == ["tool-misuse"]
    assert spec.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v1"]["required_attacks"] == []


def test_agentic_v3_adds_goal_hijack_without_mutating_v2():
    """The ASI01 module lands as a NEW profile version; v2 is untouched
    (the versioning rule — never a silent edit of an existing profile)."""
    v3 = profiles.PROFILES["benchmark.agentic.v3"]
    assert v3["required_attacks"] == ["tool-misuse", "agent-goal-hijack"]
    assert v3.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v2"]["required_attacks"] == \
        ["tool-misuse"]


def test_agentic_v4_adds_human_agent_trust_without_mutating_v3():
    """The ASI09 module lands as a NEW profile version; v3 (and v2) untouched
    (the versioning rule — never a silent edit of an existing profile)."""
    v4 = profiles.PROFILES["benchmark.agentic.v4"]
    assert v4["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust"]
    assert v4.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v3"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack"]
    assert profiles.PROFILES["benchmark.agentic.v2"]["required_attacks"] == \
        ["tool-misuse"]


def test_agentic_v5_adds_memory_poisoning_without_mutating_v4():
    """The ASI06 module lands as a NEW profile version; v4 (and earlier)
    untouched (the versioning rule — never a silent edit of an existing
    profile)."""
    v5 = profiles.PROFILES["benchmark.agentic.v5"]
    assert v5["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning"]
    assert v5.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v4"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust"]


def test_agentic_v6_adds_supply_chain_without_mutating_v5():
    """The ASI04 module lands as a NEW profile version; v5 (and earlier)
    untouched (the versioning rule — never a silent edit of an existing
    profile)."""
    v6 = profiles.PROFILES["benchmark.agentic.v6"]
    assert v6["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain"]
    assert v6.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v5"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning"]


def test_agentic_v8_adds_identity_privilege_without_mutating_v6():
    """The ASI03 module lands as a NEW profile version; v6 (and earlier)
    untouched (the versioning rule — never a silent edit of an existing
    profile). v8 was reserved ahead of the ASI08 lane's v7 landing —
    versions are reserved, not landed in order."""
    v8 = profiles.PROFILES["benchmark.agentic.v8"]
    assert v8["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain",
        "identity-privilege-abuse"]
    assert v8.get("agentic") is True


def test_agentic_v7_adds_cascading_failure_without_mutating_v6():
    """The ASI08 module lands as a NEW profile version; v6 (and earlier)
    untouched (the versioning rule — never a silent edit of an existing
    profile)."""
    v7 = profiles.PROFILES["benchmark.agentic.v7"]
    assert v7["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain", "cascading-failure"]
    assert v7.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v6"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning", "agentic-supply-chain"]


def test_agentic_v9_adds_unexpected_code_execution_without_mutating_v8():
    """The ASI05 module lands as a NEW profile version on top of the union of
    v7+v8 (the first EIGHT agentic modules); v8 and v7 untouched (the
    versioning rule — never a silent edit of an existing profile)."""
    v9 = profiles.PROFILES["benchmark.agentic.v9"]
    assert v9["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain",
        "cascading-failure", "identity-privilege-abuse",
        "unexpected-code-execution"]
    assert v9.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v8"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning", "agentic-supply-chain",
         "identity-privilege-abuse"]
    assert profiles.PROFILES["benchmark.agentic.v7"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning", "agentic-supply-chain",
         "cascading-failure"]


def test_agentic_v10_adds_rogue_agent_without_mutating_v9():
    """The ASI10 module lands as a NEW profile version on top of v9 (the
    first NINE agentic modules); v9 untouched (the versioning rule — never a
    silent edit of an existing profile)."""
    v10 = profiles.PROFILES["benchmark.agentic.v10"]
    assert v10["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain",
        "cascading-failure", "identity-privilege-abuse",
        "unexpected-code-execution", "rogue-agent"]
    assert v10.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v9"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning", "agentic-supply-chain",
         "cascading-failure", "identity-privilege-abuse",
         "unexpected-code-execution"]


def test_agentic_v11_adds_insecure_agent_communication_without_mutating_v10():
    """The ASI07 module lands as a NEW profile version on top of v10 — with
    ALL TEN agentic modules required, the ASI01-ASI10 set is COMPLETE; v10
    untouched (the versioning rule — never a silent edit of an existing
    profile)."""
    v11 = profiles.PROFILES["benchmark.agentic.v11"]
    assert v11["required_attacks"] == [
        "tool-misuse", "agent-goal-hijack", "human-agent-trust",
        "memory-context-poisoning", "agentic-supply-chain",
        "cascading-failure", "identity-privilege-abuse",
        "unexpected-code-execution", "rogue-agent",
        "insecure-agent-communication"]
    assert v11.get("agentic") is True
    assert profiles.PROFILES["benchmark.agentic.v10"]["required_attacks"] == \
        ["tool-misuse", "agent-goal-hijack", "human-agent-trust",
         "memory-context-poisoning", "agentic-supply-chain",
         "cascading-failure", "identity-privilege-abuse",
         "unexpected-code-execution", "rogue-agent"]


def test_profile_threats_exist_in_the_coverage_universe():
    universe = coverage.threat_universe(applicable_agentic=True)
    for spec in profiles.PROFILES.values():
        for t in spec["threats"]:
            assert t in universe


def test_required_attacks_are_registered():
    from attacker.attacks import ATTACKS
    for spec in profiles.PROFILES.values():
        for a in spec["required_attacks"]:
            assert a in ATTACKS, a
        final = spec.get("final_composite_cell")
        if final:
            assert final in ATTACKS


def test_measured_vs_unmeasured_attacks():
    rep = profiles.evaluate_profile("benchmark.split-privacy.v2",
                                    [_complete_art("wire_shape_analysis")])
    assert rep["attacks"]["wire-shape"] == "MEASURED"
    assert rep["attacks"]["seam-inversion"] == "UNMEASURED"
    assert rep["summary"]["attacks_measured"] == 1
    rep2 = profiles.evaluate_profile(
        "benchmark.split-privacy.v2", [art("wire_shape_analysis", ok=False)])
    assert rep2["attacks"]["wire-shape"] == "UNMEASURED"


def test_unmeasured_applicable_risks_are_first_class():
    rep = profiles.evaluate_profile("benchmark.genai-app.v1", [])
    s = rep["summary"]
    assert s["UNMEASURED"] > 0
    assert "LLM01" in s["unmeasured_applicable"]
    assert s["COVERED"] == 0


def _full_agh():
    """A COMPLETE agent-goal-hijack artifact (all arms x worlds, full
    metrics) so it counts as measured even under the Phase-3 completeness
    guard (C5). Keeps the LLM01/S14 transition tests robust across phases."""
    from attacker.attacks import agent_goal_hijack as agh

    def cell(arm, world):
        return {"experiment": "agent_goal_hijack", "arm": arm, "world": world,
                "trials": 2, "violations": {"unsanctioned_effect": 0},
                "total_violations": 0, "violation_rate": 0.0,
                "wilson_upper": 0.5, "false_blocks": 0}
    return {"schema": "x", "config": {}, "threat_model": "t",
            "provenance": {},
            "results": [cell(a, w) for a in agh.ARMS for w in agh.WORLDS],
            "summary": []}


def _full_mcp():
    """A COMPLETE memory-context-poisoning artifact (all arms x worlds, full
    metrics) so it counts as measured under the completeness guard (C5)."""
    from attacker.attacks import memory_context_poisoning as mcp

    def cell(arm, world):
        return {"experiment": "memory_context_poisoning", "arm": arm,
                "world": world, "trials": 2,
                "violations": {"unsanctioned_effect": 0},
                "total_violations": 0, "violation_rate": 0.0,
                "wilson_upper": 0.5, "false_blocks": 0}
    return {"schema": "x", "config": {}, "threat_model": "t",
            "provenance": {},
            "results": [cell(a, w) for a in mcp.ARMS for w in mcp.WORLDS],
            "summary": []}


def _full_asc():
    """A COMPLETE agentic-supply-chain artifact (all arms x worlds, full
    metrics) so it counts as measured under the completeness guard (C5)."""
    from attacker.attacks import agentic_supply_chain as asc

    def cell(arm, world):
        return {"experiment": "agentic_supply_chain", "arm": arm,
                "world": world, "trials": 2,
                "violations": {"unsanctioned_effect": 0},
                "total_violations": 0, "violation_rate": 0.0,
                "wilson_upper": 0.5, "false_blocks": 0}
    return {"schema": "x", "config": {}, "threat_model": "t",
            "provenance": {},
            "results": [cell(a, w) for a in asc.ARMS for w in asc.WORLDS],
            "summary": []}


def _full_cf():
    """A COMPLETE cascading-failure artifact (all arms x worlds, full metrics)
    so it counts as measured under the completeness guard (C5)."""
    from attacker.attacks import cascading_failure as cf

    def cell(arm, world):
        return {"experiment": "cascading_failure", "arm": arm,
                "world": world, "trials": 2,
                "violations": {"unsanctioned_effect": 0},
                "total_violations": 0, "violation_rate": 0.0,
                "wilson_upper": 0.5, "false_blocks": 0}
    return {"schema": "x", "config": {}, "threat_model": "t",
            "provenance": {},
            "results": [cell(a, w) for a in cf.ARMS for w in cf.WORLDS],
            "summary": []}


def test_full_measurement_covers_a_profile_threat():
    rep = profiles.evaluate_profile("benchmark.genai-app.v1",
                                    [_complete_art("prompt_injection")])
    assert rep["threats"]["LLM01"]["status"] == "PARTIAL"
    assert rep["threats"]["LLM01"]["untested_attacks"] == \
        ["agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
         "memory-context-poisoning", "rag-vector-abuse"]
    rep2 = profiles.evaluate_profile(
        "benchmark.genai-app.v1",
        [_complete_art("prompt_injection"), _complete_art("rag_vector_abuse")])
    assert rep2["threats"]["LLM01"]["status"] == "PARTIAL"
    assert rep2["threats"]["LLM01"]["untested_attacks"] == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning"]
    rep3 = profiles.evaluate_profile(
        "benchmark.genai-app.v1",
        [_complete_art("prompt_injection"), _complete_art("rag_vector_abuse"),
         _full_agh(), _full_mcp(), _full_asc(), _full_cf()])
    assert rep3["threats"]["LLM01"]["status"] == "COVERED"
    assert "LLM01" not in rep3["summary"]["unmeasured_applicable"]


def test_agentic_profile_out_of_scope_by_default():
    rep = profiles.evaluate_profile("benchmark.agentic.v1", [])
    assert all(t["status"] == "OUT_OF_SCOPE"
               for t in rep["threats"].values())
    assert rep["summary"]["unmeasured_applicable"] == []
    rep2 = profiles.evaluate_profile("benchmark.agentic.v1", [],
                                     applicable_agentic=True)
    assert all(t["status"] == "UNMEASURED"
               for t in rep2["threats"].values())
    assert len(rep2["summary"]["unmeasured_applicable"]) == 10


def _tm_complete_artifact():
    """A COMPLETE tool-misuse artifact (all arms x worlds, metric keys) so
    the C5 completeness guard counts tool-misuse as measured."""
    arms = ("dangerous-parameters", "scope-confusion",
            "side-effect-amplification", "authorized-task")
    recs = [{"experiment": "tool_misuse", "arm": a, "world": w, "trials": 2,
             "total_violations": 0, "violation_rate": 0.0, "wilson_upper": 0.5}
            for a in arms for w in ("vulnerable", "defended")]
    return {"schema": "dtraining.attacker.test.v1", "config": {},
            "threat_model": "t", "provenance": {}, "results": recs,
            "summary": []}


def test_llm03_needs_tool_misuse_surfaced_genai_app_regression():
    """acceptance 3(e): mapping tool-misuse -> LLM03 (Excessive Agency) made
    mapped_attacks("LLM03") == [prompt-injection, tool-misuse];
    identity-privilege-abuse (ASI03) later extended it to THREE (LLM03 is the
    privilege/confused-deputy analogue, README.md),
    unexpected-code-execution (ASI05) extended it to FOUR (LLM03 is also the
    approval-bypass/tool-call analogue), rogue-agent (ASI10) extended it to
    FIVE (LLM03 is also the capability-budget/lifecycle analogue),
    insecure-agent-communication (ASI07) extended it to SIX (LLM03 is also
    the unauthenticated-instruction/confused-deputy-across-agents analogue),
    and cross-agent-deputy extended it to SEVEN (LLM03 is the
    Confused-Deputy-style cross-agent analogue — an AUTHENTIC forwarded
    request exercising the deputy's held authority).
    In benchmark.genai-app.v1 (which requires NONE of the agentic modules),
    running prompt-injection alone now yields LLM03 == PARTIAL with ALL SIX
    gaps named — deliberate, surfaced changes (LLM03 legitimately needs
    tool-misuse AND identity-privilege-abuse AND unexpected-code-execution
    AND rogue-agent AND insecure-agent-communication AND cross-agent-deputy
    measurements), NOT a silent regression."""
    assert set(coverage.mapped_attacks("LLM03")) == {
        "prompt-injection", "tool-misuse", "identity-privilege-abuse",
        "unexpected-code-execution", "rogue-agent",
        "insecure-agent-communication", "cross-agent-deputy"}
    rep = profiles.evaluate_profile("benchmark.genai-app.v1",
                                    [_complete_art("prompt_injection")])
    llm03 = rep["threats"]["LLM03"]
    assert llm03["status"] == "PARTIAL"
    assert llm03["untested_attacks"] == ["cross-agent-deputy",
                                         "identity-privilege-abuse",
                                         "insecure-agent-communication",
                                         "rogue-agent", "tool-misuse",
                                         "unexpected-code-execution"]
    rep2 = profiles.evaluate_profile(
        "benchmark.genai-app.v1",
        [_complete_art("prompt_injection"), _tm_complete_artifact()])
    assert rep2["threats"]["LLM03"]["status"] == "PARTIAL"
    assert rep2["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy", "identity-privilege-abuse",
         "insecure-agent-communication", "rogue-agent",
         "unexpected-code-execution"]
    rep3 = profiles.evaluate_profile(
        "benchmark.genai-app.v1",
        [_complete_art("prompt_injection"), _tm_complete_artifact(),
         _complete_art("identity_privilege_abuse"),
         _complete_art("unexpected_code_execution"),
         _complete_art("rogue_agent"),
         _complete_art("insecure_agent_communication"),
         _complete_art("cross_agent_deputy")])
    assert rep3["threats"]["LLM03"]["status"] == "COVERED"


def test_split_privacy_has_composite_final_cell():
    rep = profiles.evaluate_profile("benchmark.split-privacy.v2", [])
    assert "privacy-composition (final composite)" in rep["attacks"]


def test_unknown_profile_is_descriptive():
    with pytest.raises(ValueError, match="unknown profile"):
        profiles.evaluate_profile("benchmark.nope.v9", [])


def test_render_profile_md():
    rep = profiles.evaluate_profile("benchmark.split-privacy.v2",
                                    [art("wire_shape_analysis")])
    md = profiles.render_profile_md(rep)
    assert "benchmark.split-privacy.v2" in md
    assert "| threat | status |" in md
    assert "unmeasured applicable risks" in md


def _eid(attack_name):
    from attacker.attacks import load
    return load(attack_name).EXPERIMENT_ID


_LANE_C_PRIVACY = ["seam-inversion", "label-scoring", "gpu-memory",
                   "wire-shape", "lira-membership", "multi-tenant-isolation",
                   "artifact-telemetry-leak", "router-side-channel"]
_LANE_C_GENAI = ["prompt-injection", "rag-vector-abuse",
                 "hidden-context-exposure", "output-sink-injection",
                 "misinformation-integrity", "multi-tenant-isolation",
                 "artifact-telemetry-leak"]


@pytest.mark.parametrize("attack", _LANE_C_PRIVACY)
def test_split_privacy_v2_gates_required_attack(attack):
    eid = _eid(attack)
    assert eid in coverage.COMPLETENESS
    complete = profiles.evaluate_profile(
        "benchmark.split-privacy.v2", [_complete_art(eid)])
    assert complete["attacks"][attack] == "MEASURED"
    skeleton = profiles.evaluate_profile("benchmark.split-privacy.v2",
                                         [art(eid)])
    assert skeleton["attacks"][attack] == "UNMEASURED"


@pytest.mark.parametrize("attack", _LANE_C_GENAI)
def test_genai_app_v1_gates_required_attack(attack):
    eid = _eid(attack)
    assert eid in coverage.COMPLETENESS
    complete = profiles.evaluate_profile(
        "benchmark.genai-app.v1", [_complete_art(eid)])
    assert complete["attacks"][attack] == "MEASURED"
    skeleton = profiles.evaluate_profile("benchmark.genai-app.v1", [art(eid)])
    assert skeleton["attacks"][attack] == "UNMEASURED"


def test_deferred_privacy_attacks_remain_ungated():
    for attack in ("gradient-inversion", "membership"):
        eid = _eid(attack)
        assert eid not in coverage.COMPLETENESS
        rep = profiles.evaluate_profile("benchmark.split-privacy.v2",
                                        [art(eid)])
        assert rep["attacks"][attack] == "MEASURED"


def test_cli_writes_reports(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(art("wire_shape_analysis")))
    out = tmp_path / "p.json"
    md = tmp_path / "p.md"
    rc = profiles.main([str(a), "--output", str(out), "--md", str(md)])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == profiles.SCHEMA
    assert set(doc["profiles"]) == set(profiles.PROFILES)
    assert "benchmark.agentic.v1" in md.read_text()


def test_cli_single_profile_and_dispatch(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(art("wire_shape_analysis")))
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--profiles", str(a),
         "--profile", "benchmark.split-privacy.v2"],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "benchmark.split-privacy.v2" in r.stdout
    assert "agentic" not in r.stdout


def test_self_test_passes():
    assert profiles.self_test() == 0
