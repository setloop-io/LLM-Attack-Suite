"""Attack metadata registry (attacker/attack_metadata.py): schema
validation, registry completeness vs the attack registry (CI tripwire),
and the inversion helpers."""

import pytest

from attacker import attack_metadata as am
from attacker import threat_model
from attacker.attacks import ATTACKS


def _md(**kw):
    base = {"actor_intent": "passive", "persistence": "session",
            "surfaces": ("S1",), "security_properties": ("confidentiality",),
            "lifecycle_stages": ("training",), "owasp_llm_2026": ("LLM02",)}
    base.update(kw)
    return base


def test_validate_metadata_accepts_a_full_entry():
    md = _md(mitre_atlas=["AML.T0000"], nist_ai_100_2=["data-poisoning"],
             cwe=["CWE-200"])
    assert am.validate_metadata("x", md) is md


def test_validate_metadata_requires_every_field():
    for field in am._FIELDS:
        md = _md()
        del md[field]
        with pytest.raises(ValueError, match=field):
            am.validate_metadata("x", md)


def test_validate_metadata_rejects_bad_vocab():
    with pytest.raises(ValueError, match="actor_intent"):
        am.validate_metadata("x", _md(actor_intent="sneaky"))
    with pytest.raises(ValueError, match="persistence"):
        am.validate_metadata("x", _md(persistence="eternal"))
    with pytest.raises(ValueError, match="S99"):
        am.validate_metadata("x", _md(surfaces=("S99",)))
    with pytest.raises(ValueError, match="security_properties"):
        am.validate_metadata("x", _md(security_properties=("magic",)))
    with pytest.raises(ValueError, match="lifecycle_stages"):
        am.validate_metadata("x", _md(lifecycle_stages=("design",)))
    with pytest.raises(ValueError, match="LLM99"):
        am.validate_metadata("x", _md(owasp_llm_2026=("LLM99",)))
    with pytest.raises(ValueError, match="unknown fields"):
        am.validate_metadata("x", _md(mood="curious"))
    with pytest.raises(ValueError, match="non-empty"):
        am.validate_metadata("x", _md(surfaces=()))


def test_metadata_vocabularies_match_threat_model_enums():
    assert set(am._FIELDS["actor_intent"]) == set(threat_model.INTENTS)
    assert set(am._FIELDS["persistence"]) == set(threat_model.PERSISTENCE)
    assert am.OWASP_LLM_2026 == tuple(f"LLM{i:02d}" for i in range(1, 11))


def test_every_registered_attack_has_metadata_and_vice_versa():
    assert am.validate_registry(list(ATTACKS)) == sorted(ATTACKS)
    assert len(ATTACKS) == 65
    assert set(ATTACKS) == set(am.ATTACK_METADATA)
    assert "S19" not in am.SURFACES
    assert "S19" not in am.metadata_for("secure-compute-audit")["surfaces"]


def test_registry_gap_is_descriptive():
    with pytest.raises(ValueError, match="no-such-attack"):
        am.validate_registry(["accumulation", "no-such-attack"])
    with pytest.raises(ValueError, match="metadata entries for unknown"):
        am.validate_registry([])


def test_metadata_for_unknown_attack_names_the_registry():
    with pytest.raises(KeyError, match="attack_metadata"):
        am.metadata_for("no-such-attack")
    assert am.metadata_for("accumulation")["surfaces"] == ("S1",)


def test_attacks_for_owasp_inverts_mapping():
    llm02 = am.attacks_for_owasp("LLM02")
    assert "accumulation" in llm02 and "wire-shape" in llm02
    assert llm02 == sorted(llm02)
    with pytest.raises(ValueError, match="LLM99"):
        am.attacks_for_owasp("LLM99")


def test_attacks_for_surface_inverts_mapping():
    s7 = am.attacks_for_surface("S7")
    assert "wire-shape" in s7 and "mitm-proxy" in s7
    with pytest.raises(ValueError, match="S99"):
        am.attacks_for_surface("S99")


def test_schema_is_v2_with_closed_agentic_vocab():
    assert am.SCHEMA == "llm_attacker.attack_metadata.v2"
    assert am.OWASP_AGENTIC_2026 == tuple(f"ASI{i:02d}" for i in range(1, 11))
    assert "owasp_agentic_2026" in am._EXTRA_MAPPINGS
    assert "owasp_agentic_2026" not in am._FIELDS


def test_agentic_mapping_is_validated_against_the_closed_set():
    ok = _md(owasp_agentic_2026=("ASI02",))
    assert am.validate_metadata("x", ok) is ok
    with pytest.raises(ValueError, match="ASI99"):
        am.validate_metadata("x", _md(owasp_agentic_2026=("ASI99",)))


def test_tool_misuse_maps_asi02_and_llm03():
    md = am.metadata_for("tool-misuse")
    assert md["owasp_agentic_2026"] == ("ASI02",)
    assert "LLM03" in md["owasp_llm_2026"]


def test_attacks_for_agentic_inverts_mapping():
    asi02 = am.attacks_for_agentic("ASI02")
    assert asi02 == ["tool-misuse"]
    assert am.attacks_for_agentic("ASI01") == ["agent-goal-hijack"]
    with pytest.raises(ValueError, match="ASI99"):
        am.attacks_for_agentic("ASI99")


def test_lane_b_asi01_coverage_transitions_are_intended():
    """agent-goal-hijack (ASI01) extends three mappings — recorded, not
    silent (seed §4.3d): ASI01 [] -> [agent-goal-hijack]; LLM01 2 -> 3 (now 6
    with the ASI09/ASI06/ASI04/ASI08 modules); S14 6 -> 7 (now 16 with the
    ASI09, ASI06, ASI04, ASI08, ASI03, ASI05, ASI10 and ASI07 modules).
    Pinned so any future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI01") == ["agent-goal-hijack"]
    assert am.attacks_for_owasp("LLM01") == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning", "prompt-injection", "rag-vector-abuse"]
    s14 = am.attacks_for_surface("S14")
    assert "agent-goal-hijack" in s14 and len(s14) == 16


def test_lane_b_asi09_coverage_transitions_are_intended():
    """human-agent-trust (ASI09) extends three mappings — recorded, not silent
    (seed §4.3d): ASI09 [] -> [human-agent-trust]; LLM07 1 -> 2; S14 7 -> 8
    (now 16 with the ASI06, ASI04, ASI08, ASI03, ASI05, ASI10 and ASI07
    modules). Pinned so any future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI09") == ["human-agent-trust"]
    assert am.attacks_for_owasp("LLM07") == [
        "human-agent-trust", "misinformation-integrity"]
    s14 = am.attacks_for_surface("S14")
    assert "human-agent-trust" in s14 and len(s14) == 16
    md = am.metadata_for("human-agent-trust")
    assert md["owasp_agentic_2026"] == ("ASI09",)
    assert md["owasp_llm_2026"] == ("LLM07",)
    assert md["cwe"] == ("CWE-451",)


def test_lane_b_asi06_coverage_transitions_are_intended():
    """memory-context-poisoning (ASI06) extends three mappings — recorded, not
    silent (seed §3 Phase 3): ASI06 [] -> [memory-context-poisoning]; LLM01
    3 -> 4 (now 6 with the ASI04 and ASI08 modules); S14 8 -> 9 (now 16 with
    the ASI04, ASI08, ASI03, ASI05, ASI10 and ASI07 modules). Pinned so any
    future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI06") == ["memory-context-poisoning"]
    assert am.attacks_for_owasp("LLM01") == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning", "prompt-injection", "rag-vector-abuse"]
    s14 = am.attacks_for_surface("S14")
    assert "memory-context-poisoning" in s14 and len(s14) == 16
    md = am.metadata_for("memory-context-poisoning")
    assert md["owasp_agentic_2026"] == ("ASI06",)
    assert md["owasp_llm_2026"] == ("LLM01",)
    assert md["persistence"] == "cross-session"
    assert md["cwe"] == ("CWE-501",)


def test_lane_b_asi03_coverage_transitions_are_intended():
    """identity-privilege-abuse (ASI03) extends three mappings — recorded, not
    silent: ASI03 [] -> [identity-privilege-abuse]; LLM03 2 -> 3 (Excessive
    Agency — README.md names privilege/confused-deputy tests
    as LLM03's gap; deliberately NOT LLM01, the adoption-mechanism mapping of
    ASI01/ASI06/ASI04); S14 10 -> 11 (now 16 with the ASI08, ASI05, ASI10 and
    ASI07 modules). Pinned so any future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI03") == ["identity-privilege-abuse"]
    assert am.attacks_for_owasp("LLM03") == [
        "cross-agent-deputy", "identity-privilege-abuse",
        "insecure-agent-communication", "prompt-injection", "rogue-agent",
        "tool-misuse", "unexpected-code-execution"]
    assert "identity-privilege-abuse" not in am.attacks_for_owasp("LLM01")
    s14 = am.attacks_for_surface("S14")
    assert "identity-privilege-abuse" in s14 and len(s14) == 16
    md = am.metadata_for("identity-privilege-abuse")
    assert md["owasp_agentic_2026"] == ("ASI03",)
    assert md["owasp_llm_2026"] == ("LLM03",)
    assert md["persistence"] == "session"
    assert md["cwe"] == ("CWE-863",)


def test_lane_b_asi08_coverage_transitions_are_intended():
    """cascading-failure (ASI08) extends three mappings — recorded, not silent:
    ASI08 [] -> [cascading-failure]; LLM01 5 -> 6; S14 10 -> 11 (now 16 with
    the ASI03, ASI05, ASI10 and ASI07 modules). LLM05 is DELIBERATELY NOT used
    (it is
    the split-integrity-availability.v1 poisoning surface; mapping cascade
    there would disturb that frozen calibration), and LLM01 is not in that
    profile's threat set so it does not collide. Pinned so any future drift
    is a reviewable edit."""
    assert am.attacks_for_agentic("ASI08") == ["cascading-failure"]
    assert am.attacks_for_owasp("LLM01") == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning", "prompt-injection", "rag-vector-abuse"]
    s14 = am.attacks_for_surface("S14")
    assert "cascading-failure" in s14 and len(s14) == 16
    md = am.metadata_for("cascading-failure")
    assert md["owasp_agentic_2026"] == ("ASI08",)
    assert md["owasp_llm_2026"] == ("LLM01",)
    assert "LLM05" not in md["owasp_llm_2026"]
    assert md["persistence"] == "cross-session"
    assert md["cwe"] == ("CWE-345",)


def test_lane_b_asi05_coverage_transitions_are_intended():
    """unexpected-code-execution (ASI05) extends three mappings — recorded,
    not silent: ASI05 [] -> [unexpected-code-execution]; LLM03 3 -> 4
    (Excessive Agency — README.md names approval-bypass /
    tool-call tests as LLM03's gap; deliberately NOT LLM01, the
    adoption-mechanism mapping of ASI01/ASI06/ASI04/ASI08, because the modeled
    weakness and the load-bearing control is the missing code-digest approval
    binding at the execution gateway); S14 12 -> 13 (now 16 with the ASI10
    and ASI07 modules). Pinned so any future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI05") == ["unexpected-code-execution"]
    assert am.attacks_for_owasp("LLM03") == [
        "cross-agent-deputy", "identity-privilege-abuse",
        "insecure-agent-communication", "prompt-injection", "rogue-agent",
        "tool-misuse", "unexpected-code-execution"]
    assert "unexpected-code-execution" not in am.attacks_for_owasp("LLM01")
    s14 = am.attacks_for_surface("S14")
    assert "unexpected-code-execution" in s14 and len(s14) == 16
    md = am.metadata_for("unexpected-code-execution")
    assert md["owasp_agentic_2026"] == ("ASI05",)
    assert md["owasp_llm_2026"] == ("LLM03",)
    assert md["persistence"] == "session"
    assert md["cwe"] == ("CWE-94",)


def test_lane_b_asi10_coverage_transitions_are_intended():
    """rogue-agent (ASI10) extends three mappings — recorded, not silent:
    ASI10 [] -> [rogue-agent]; LLM03 4 -> 5 (Excessive Agency —
    README.md names capability-budget / tool-call tests as
    LLM03's gap; deliberately NOT LLM01, the adoption-mechanism mapping of
    ASI01/ASI06/ASI04/ASI08, because the modeled weakness and the load-bearing
    control is the missing operator attestation on the agent lifecycle at the
    gateway); S14 13 -> 14 (now 16 with the ASI07 module). Pinned so any
    future drift is a reviewable edit."""
    assert am.attacks_for_agentic("ASI10") == ["rogue-agent"]
    assert am.attacks_for_owasp("LLM03") == [
        "cross-agent-deputy", "identity-privilege-abuse",
        "insecure-agent-communication", "prompt-injection", "rogue-agent",
        "tool-misuse", "unexpected-code-execution"]
    assert "rogue-agent" not in am.attacks_for_owasp("LLM01")
    s14 = am.attacks_for_surface("S14")
    assert "rogue-agent" in s14 and len(s14) == 16
    md = am.metadata_for("rogue-agent")
    assert md["owasp_agentic_2026"] == ("ASI10",)
    assert md["owasp_llm_2026"] == ("LLM03",)
    assert md["persistence"] == "cross-session"
    assert md["cwe"] == ("CWE-693",)


def test_lane_b_asi07_coverage_transitions_are_intended():
    """insecure-agent-communication (ASI07) extends three mappings — recorded,
    not silent: ASI07 [] -> [insecure-agent-communication]; LLM03 5 -> 6
    (Excessive Agency — README.md names tool-call /
    approval-bypass / confused-deputy tests as LLM03's gap; deliberately NOT
    LLM01, the adoption-mechanism mapping of ASI01/ASI06/ASI04/ASI08, because
    the modeled weakness and the load-bearing control is the missing
    per-message authenticity/freshness verification at the receiving
    endpoint); S14 14 -> 15 (now 16 with cross-agent-deputy). Pinned so any future drift is a reviewable
    edit."""
    assert am.attacks_for_agentic("ASI07") == ["insecure-agent-communication"]
    assert am.attacks_for_owasp("LLM03") == [
        "cross-agent-deputy", "identity-privilege-abuse",
        "insecure-agent-communication", "prompt-injection", "rogue-agent",
        "tool-misuse", "unexpected-code-execution"]
    assert "insecure-agent-communication" not in am.attacks_for_owasp("LLM01")
    s14 = am.attacks_for_surface("S14")
    assert "insecure-agent-communication" in s14 and len(s14) == 16
    md = am.metadata_for("insecure-agent-communication")
    assert md["owasp_agentic_2026"] == ("ASI07",)
    assert md["owasp_llm_2026"] == ("LLM03",)
    assert md["persistence"] == "cross-session"
    assert md["cwe"] == ("CWE-294",)


def test_lane_b_cross_agent_deputy_coverage_transitions_are_intended():
    """cross-agent-deputy (the Confused-Deputy-style cross-agent scenario)
    extends two mappings — recorded, not silent: LLM03 6 -> 7 (Excessive
    Agency — README.md names confused-deputy tests as LLM03's gap; the
    cross-agent confused deputy is that analogue, deliberately NOT LLM01,
    the adoption-mechanism mapping of ASI01/ASI06/ASI04/ASI08, because the
    forwarded request is AUTHENTIC and the load-bearing control is the
    gateway's invocation-identity binding); S14 15 -> 16. It carries NO
    owasp_agentic_2026 mapping ON PURPOSE: the closed ASI01-ASI10 set is
    already completely mapped one-module-each (since benchmark.agentic.v11)
    and COVERED requires ALL mapped attacks measured, so mapping this
    scenario to ASI03 or ASI07 would silently demote those threats from
    COVERED to PARTIAL on every existing evidence bundle. Pinned so any
    future drift is a reviewable edit."""
    assert am.attacks_for_owasp("LLM03") == [
        "cross-agent-deputy", "identity-privilege-abuse",
        "insecure-agent-communication", "prompt-injection", "rogue-agent",
        "tool-misuse", "unexpected-code-execution"]
    assert "cross-agent-deputy" not in am.attacks_for_owasp("LLM01")
    s14 = am.attacks_for_surface("S14")
    assert "cross-agent-deputy" in s14 and len(s14) == 16
    assert am.attacks_for_agentic("ASI03") == ["identity-privilege-abuse"]
    assert am.attacks_for_agentic("ASI07") == ["insecure-agent-communication"]
    md = am.metadata_for("cross-agent-deputy")
    assert "owasp_agentic_2026" not in md
    assert md["owasp_llm_2026"] == ("LLM03",)
    assert md["persistence"] == "session"
    assert md["cwe"] == ("CWE-441",)


def test_surface_table_extends_the_paper_model():
    assert {"S1", "S2", "S5", "S9"} <= set(am.SURFACES)
    assert {"S10", "S11", "S12", "S13", "S14"} <= set(am.SURFACES)


def test_self_test_passes():
    assert am.self_test() == 0
