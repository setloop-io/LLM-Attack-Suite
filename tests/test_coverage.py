"""Coverage manifest (attacker/coverage.py): status semantics, the
never-conflate rule for UNMEASURED vs OUT_OF_SCOPE, artifact parsing, and
the CLI."""

import json
import subprocess
import sys

import pytest

from attacker import coverage


def art(*records):
    return {"schema": "dtraining.attacker.test.v1", "config": {},
            "threat_model": "t", "provenance": {}, "results": list(records),
            "summary": []}


GOOD = {"experiment": "wire_shape_analysis", "mae": 1.0}
BAD = {"experiment": "gpu_memory_residue", "error": "no GPU"}


def test_experiments_of_counts_only_valid_results():
    good = _complete_records("wire_shape_analysis")
    a = art(*good, BAD, {"no_experiment": True})
    assert coverage.experiments_of(a) == {"wire_shape_analysis"}
    assert coverage.experiments_of(art()) == set()
    a2 = art()
    a2["summary"] = [dict(r) for r in good]
    assert coverage.experiments_of(a2) == {"wire_shape_analysis"}


def test_load_artifacts_fail_closed(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(art(GOOD)))
    assert coverage.load_artifacts([str(p)]) == [art(GOOD)]
    with pytest.raises(ValueError, match="missing.json"):
        coverage.load_artifacts([str(tmp_path / "missing.json")])
    p.write_text("not json")
    with pytest.raises(ValueError, match="cannot read artifact"):
        coverage.load_artifacts([str(p)])


def test_unmeasured_when_no_artifacts():
    m = coverage.build_manifest([])
    assert m["threats"]["LLM02"]["status"] == "UNMEASURED"
    assert m["threats"]["S7"]["status"] == "UNMEASURED"
    assert m["summary"]["COVERED"] == 0
    assert "LLM02" in m["summary"]["unmeasured_applicable"]


def test_measured_attacks_drive_status():
    m = coverage.build_manifest(
        [art(*_complete_records("wire_shape_analysis"))])
    s7 = m["threats"]["S7"]
    assert s7["status"] in ("COVERED", "PARTIAL")
    assert "wire-shape" in s7["measured_attacks"]
    assert "gpu-memory" not in m["attacks_measured"]
    assert m["threats"]["S5"]["status"] == "UNMEASURED"


def test_partial_names_untested_attacks():
    m = coverage.build_manifest(
        [art(*_complete_records("wire_shape_analysis"))])
    s7 = m["threats"]["S7"]
    if set(s7["mapped_attacks"]) == {"wire-shape"}:
        assert s7["status"] == "COVERED"
    else:
        assert s7["status"] == "PARTIAL"
        assert s7["untested_attacks"]


def _complete_records(experiment):
    """Every result record a COMPLETENESS-gated experiment needs to count as
    measured (one per required cell, with its metric keys); a plain record for
    an ungated experiment. Keeps this test valid as guards are added — e.g.
    protocol-integrity (mapped to S7) now carries an integrity COMPLETENESS
    entry, so a skeleton one-liner correctly no longer marks S7 COVERED."""
    req = coverage.COMPLETENESS.get(experiment)
    if req is None:
        return [{"experiment": experiment, "ok": 1}]
    if "cells" in req:
        return [{"experiment": experiment, "arm": arm, "world": world,
                 **{k: 1.0 for k in metrics}}
                for (arm, world), metrics in req["cells"].items()]
    return [{"experiment": experiment, "arm": arm, "world": world,
             **{k: 1.0 for k in req["metrics"]}}
            for arm in req["arms"] for world in req["worlds"]]


def test_full_mapping_covered():
    from attacker import attack_metadata
    mapped = attack_metadata.attacks_for_surface("S7")
    from attacker.attacks import load
    recs = [r for a in mapped
            for r in _complete_records(load(a).EXPERIMENT_ID)]
    m = coverage.build_manifest([art(*recs)])
    assert m["threats"]["S7"]["status"] == "COVERED"
    assert m["threats"]["S7"]["untested_attacks"] == []


def _tm_cell(arm, world, **over):
    rec = {"experiment": "tool_misuse", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5}
    rec.update(over)
    return rec


_TM_ARMS = ("dangerous-parameters", "scope-confusion",
            "side-effect-amplification", "authorized-task")


def _tm_full(**over):
    return art(*[_tm_cell(a, w, **over)
                 for a in _TM_ARMS for w in ("vulnerable", "defended")])


def test_asi02_maps_to_tool_misuse():
    assert coverage.mapped_attacks("ASI02") == ["tool-misuse"]


def test_complete_tool_misuse_artifact_covers_asi02():
    m = coverage.build_manifest([_tm_full()], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "COVERED"
    assert "tool-misuse" in m["attacks_measured"]
    m0 = coverage.build_manifest([_tm_full()])
    assert m0["threats"]["ASI02"]["status"] == "OUT_OF_SCOPE"


def test_control_only_artifact_leaves_asi02_llm03_s14_unmeasured():
    control_only = art(_tm_cell("authorized-task", "vulnerable"),
                       _tm_cell("authorized-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "tool-misuse" not in m["attacks_measured"]
    assert m["threats"]["ASI02"]["status"] == "UNMEASURED"
    assert m["threats"]["LLM03"]["status"] == "UNMEASURED"
    assert m["threats"]["S14"]["status"] == "UNMEASURED"


def test_skeleton_single_record_does_not_cover():
    skeleton = art({"experiment": "tool_misuse", "note": "benign smoke"})
    m = coverage.build_manifest([skeleton], applicable_agentic=True)
    assert "tool-misuse" not in m["attacks_measured"]
    assert m["threats"]["ASI02"]["status"] == "UNMEASURED"


def test_completeness_is_outcome_independent():
    bad = _tm_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "COVERED"


def test_incomplete_missing_one_world_does_not_cover():
    cells = [_tm_cell(a, "vulnerable") for a in _TM_ARMS]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "UNMEASURED"


def test_missing_metric_key_does_not_cover():
    cells = []
    for a in _TM_ARMS:
        for w in ("vulnerable", "defended"):
            c = _tm_cell(a, w)
            del c["wilson_upper"]
            cells.append(c)
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "UNMEASURED"


_AGH_ARMS = ("tool-output-hijack", "retrieved-document-hijack", "benign-task")
_AGH_WORLDS = ("vulnerable", "defended", "none", "provenance-only")


def _agh_cell(arm, world, **over):
    rec = {"experiment": "agent_goal_hijack", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


def _agh_full(**over):
    return art(*[_agh_cell(a, w, **over)
                 for a in _AGH_ARMS for w in _AGH_WORLDS])


_HAT_ARMS = ("summary-mismatch", "hidden-action-detail", "benign-task")
_HAT_WORLDS = ("vulnerable", "defended", "none", "display-integrity-only")


def _hat_cell(arm, world, **over):
    rec = {"experiment": "human_agent_trust", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


def _hat_full(**over):
    return art(*[_hat_cell(a, w, **over)
                 for a in _HAT_ARMS for w in _HAT_WORLDS])


_MCP_ARMS = ("cross-session-memory", "delayed-benign-then-trigger",
             "benign-task")
_MCP_WORLDS = ("vulnerable", "defended", "none", "memory-provenance-only",
               "instruction-provenance-only")


def _mcp_cell(arm, world, **over):
    rec = {"experiment": "memory_context_poisoning", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


def _mcp_full(**over):
    return art(*[_mcp_cell(a, w, **over)
                 for a in _MCP_ARMS for w in _MCP_WORLDS])


def _measured(attack_name):
    """Complete measured records for an attack by its real EXPERIMENT_ID: the
    full required cell set when it is COMPLETENESS-gated (Lane C gated the
    genai-app modules), else a single plain record. Returns a LIST — splat it
    into art(...). Keeps the transition tests honest as guards are added."""
    from attacker.attacks import load
    return _complete_records(load(attack_name).EXPERIMENT_ID)


_ASC_ARMS = ("compromised-component", "unsigned-default-binding", "benign-task")
_ASC_WORLDS = ("vulnerable", "defended", "none", "component-provenance-only",
               "instruction-provenance-only")


def _asc_cell(arm, world, **over):
    rec = {"experiment": "agentic_supply_chain", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


def _asc_full(**over):
    return art(*[_asc_cell(a, w, **over)
                 for a in _ASC_ARMS for w in _ASC_WORLDS])


_IPA_ARMS = ("confused-deputy", "role-token-forgery", "lateral-session-use",
             "authorized-task")
_IPA_WORLDS = ("vulnerable", "defended", "none", "principal-binding-only",
               "instruction-provenance-only")


def _ipa_cell(arm, world, **over):
    rec = {"experiment": "identity_privilege_abuse", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


_CF_ARMS = ("broken-chain-redirect", "benign-task")
_CF_WORLDS = ("vulnerable", "defended", "none", "cascade-revalidation-only",
              "instruction-provenance-only", "parameter-typecheck-only",
              "tenant-containment-only", "component-provenance-only")


def _cf_cell(arm, world, **over):
    rec = {"experiment": "cascading_failure", "arm": arm, "world": world,
           "trials": 2, "total_violations": 0, "violation_rate": 0.0,
           "wilson_upper": 0.5, "violations": {"unsanctioned_effect": 0},
           "false_blocks": 0}
    rec.update(over)
    return rec


def _ipa_full(**over):
    return art(*[_ipa_cell(a, w, **over)
                 for a in _IPA_ARMS for w in _IPA_WORLDS])


def _cf_full(**over):
    return art(*[_cf_cell(a, w, **over)
                 for a in _CF_ARMS for w in _CF_WORLDS])


_UCE_ARMS = ("direct-injection", "document-injection", "approval-swap",
             "authorized-exec")
_UCE_WORLDS = ("vulnerable", "defended", "none", "exec-gate-only",
               "instruction-provenance-only", "parameter-typecheck-only",
               "tenant-containment-only", "component-provenance-only",
               "principal-binding-only")


def _uce_cell(arm, world, **over):
    rec = {"experiment": "unexpected_code_execution", "arm": arm,
           "world": world, "trials": 2, "total_violations": 0,
           "violation_rate": 0.0, "wilson_upper": 0.5,
           "violations": {"unsanctioned_effect": 0}, "false_blocks": 0}
    rec.update(over)
    return rec


def _uce_full(**over):
    return art(*[_uce_cell(a, w, **over)
                 for a in _UCE_ARMS for w in _UCE_WORLDS])


_RA_ARMS = ("unauthorized-replication", "goal-self-modification",
            "persistence-reload", "attested-lifecycle")
_RA_WORLDS = ("vulnerable", "defended", "none", "lifecycle-attestation-only",
              "instruction-provenance-only", "memory-provenance-only",
              "parameter-typecheck-only", "tenant-containment-only",
              "principal-binding-only")


def _ra_cell(arm, world, **over):
    rec = {"experiment": "rogue_agent", "arm": arm,
           "world": world, "trials": 2, "total_violations": 0,
           "violation_rate": 0.0, "wilson_upper": 0.5,
           "violations": {"unsanctioned_effect": 0}, "false_blocks": 0}
    rec.update(over)
    return rec


def _ra_full(**over):
    return art(*[_ra_cell(a, w, **over)
                 for a in _RA_ARMS for w in _RA_WORLDS])


_IAC_ARMS = ("sender-spoofing", "tamper-in-transit", "replay",
             "authentic-relay")
_IAC_WORLDS = ("vulnerable", "defended", "none",
               "message-authentication-only",
               "instruction-provenance-only", "memory-provenance-only",
               "component-provenance-only", "cascade-revalidation-only",
               "principal-binding-only", "parameter-typecheck-only",
               "tenant-containment-only")


def _iac_cell(arm, world, **over):
    rec = {"experiment": "insecure_agent_communication", "arm": arm,
           "world": world, "trials": 2, "total_violations": 0,
           "violation_rate": 0.0, "wilson_upper": 0.5,
           "violations": {"unsanctioned_effect": 0}, "false_blocks": 0}
    rec.update(over)
    return rec


def _iac_full(**over):
    return art(*[_iac_cell(a, w, **over)
                 for a in _IAC_ARMS for w in _IAC_WORLDS])


_CAD_ARMS = ("forwarded-deputy-request", "sanctioned-peer-request")
_CAD_WORLDS = ("vulnerable", "defended", "none",
               "principal-binding-only", "message-authentication-only",
               "instruction-provenance-only", "memory-provenance-only",
               "component-provenance-only", "cascade-revalidation-only",
               "tenant-containment-only", "parameter-typecheck-only")


def _cad_cell(arm, world, **over):
    rec = {"experiment": "cross_agent_deputy", "arm": arm,
           "world": world, "trials": 2, "total_violations": 0,
           "violation_rate": 0.0, "wilson_upper": 0.5,
           "violations": {"unsanctioned_effect": 0}, "false_blocks": 0}
    rec.update(over)
    return rec


def _cad_full(**over):
    return art(*[_cad_cell(a, w, **over)
                 for a in _CAD_ARMS for w in _CAD_WORLDS])


def test_asi01_maps_to_agent_goal_hijack():
    assert coverage.mapped_attacks("ASI01") == ["agent-goal-hijack"]


def test_complete_agent_goal_hijack_artifact_covers_asi01():
    m = coverage.build_manifest([_agh_full()], applicable_agentic=True)
    assert m["threats"]["ASI01"]["status"] == "COVERED"
    assert "agent-goal-hijack" in m["attacks_measured"]
    m0 = coverage.build_manifest([_agh_full()])
    assert m0["threats"]["ASI01"]["status"] == "OUT_OF_SCOPE"


def test_agh_control_only_leaves_asi01_llm01_s14_unmeasured():
    control_only = art(_agh_cell("benign-task", "vulnerable"),
                       _agh_cell("benign-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "agent-goal-hijack" not in m["attacks_measured"]
    assert m["threats"]["ASI01"]["status"] == "UNMEASURED"
    assert m["threats"]["LLM01"]["status"] == "UNMEASURED"
    assert m["threats"]["S14"]["status"] == "UNMEASURED"


def test_agh_completeness_is_outcome_independent():
    bad = _agh_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI01"]["status"] == "COVERED"


def test_agh_missing_one_world_does_not_cover():
    cells = [_agh_cell(a, w) for a in _AGH_ARMS
             for w in ("vulnerable", "defended", "none")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI01"]["status"] == "UNMEASURED"


def test_agh_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _AGH_ARMS:
            for w in _AGH_WORLDS:
                c = _agh_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI01"]["status"] == "UNMEASURED", key


def test_llm01_covered_to_partial_transition():
    base = [*_measured("prompt-injection"), *_measured("rag-vector-abuse")]
    m = coverage.build_manifest([art(*base)])
    assert m["threats"]["LLM01"]["status"] == "PARTIAL"
    assert m["threats"]["LLM01"]["untested_attacks"] == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "memory-context-poisoning"]
    m2 = coverage.build_manifest([art(*base), _agh_full()])
    assert m2["threats"]["LLM01"]["status"] == "PARTIAL"
    assert m2["threats"]["LLM01"]["untested_attacks"] == \
        ["agentic-supply-chain", "cascading-failure", "memory-context-poisoning"]
    m3 = coverage.build_manifest([art(*base), _agh_full(), _mcp_full()])
    assert m3["threats"]["LLM01"]["status"] == "PARTIAL"
    assert m3["threats"]["LLM01"]["untested_attacks"] == \
        ["agentic-supply-chain", "cascading-failure"]
    m4 = coverage.build_manifest(
        [art(*base), _agh_full(), _mcp_full(), _asc_full()])
    assert m4["threats"]["LLM01"]["status"] == "PARTIAL"
    assert m4["threats"]["LLM01"]["untested_attacks"] == ["cascading-failure"]
    m5 = coverage.build_manifest(
        [art(*base), _agh_full(), _mcp_full(), _asc_full(), _cf_full()])
    assert m5["threats"]["LLM01"]["status"] == "COVERED"


def test_s14_fifteen_to_sixteen_transition():
    others = art(*_measured("prompt-injection"), *_measured("rag-vector-abuse"),
                 *_measured("hidden-context-exposure"),
                 *_measured("output-sink-injection"),
                 *_measured("misinformation-integrity"))
    m = coverage.build_manifest([others, _tm_full()])
    assert m["threats"]["S14"]["status"] == "PARTIAL"
    assert m["threats"]["S14"]["untested_attacks"] == [
        "agent-goal-hijack", "agentic-supply-chain", "cascading-failure",
        "cross-agent-deputy", "human-agent-trust",
        "identity-privilege-abuse", "insecure-agent-communication",
        "memory-context-poisoning", "rogue-agent",
        "unexpected-code-execution"]
    m2 = coverage.build_manifest([others, _tm_full(), _agh_full()])
    assert m2["threats"]["S14"]["status"] == "PARTIAL"
    assert m2["threats"]["S14"]["untested_attacks"] == [
        "agentic-supply-chain", "cascading-failure", "cross-agent-deputy",
        "human-agent-trust", "identity-privilege-abuse",
        "insecure-agent-communication", "memory-context-poisoning",
        "rogue-agent", "unexpected-code-execution"]
    m3 = coverage.build_manifest([others, _tm_full(), _agh_full(), _hat_full()])
    assert m3["threats"]["S14"]["status"] == "PARTIAL"
    assert m3["threats"]["S14"]["untested_attacks"] == \
        ["agentic-supply-chain", "cascading-failure", "cross-agent-deputy",
         "identity-privilege-abuse", "insecure-agent-communication",
         "memory-context-poisoning", "rogue-agent",
         "unexpected-code-execution"]
    m4 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full()])
    assert m4["threats"]["S14"]["status"] == "PARTIAL"
    assert m4["threats"]["S14"]["untested_attacks"] == \
        ["agentic-supply-chain", "cascading-failure", "cross-agent-deputy",
         "identity-privilege-abuse", "insecure-agent-communication",
         "rogue-agent", "unexpected-code-execution"]
    m5 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full()])
    assert m5["threats"]["S14"]["status"] == "PARTIAL"
    assert m5["threats"]["S14"]["untested_attacks"] == \
        ["cascading-failure", "cross-agent-deputy",
         "identity-privilege-abuse", "insecure-agent-communication",
         "rogue-agent", "unexpected-code-execution"]
    m6 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full()])
    assert m6["threats"]["S14"]["status"] == "PARTIAL"
    assert m6["threats"]["S14"]["untested_attacks"] == \
        ["cascading-failure", "cross-agent-deputy",
         "insecure-agent-communication", "rogue-agent",
         "unexpected-code-execution"]
    m7 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full(), _cf_full()])
    assert m7["threats"]["S14"]["status"] == "PARTIAL"
    assert m7["threats"]["S14"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication", "rogue-agent",
         "unexpected-code-execution"]
    m8 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full(), _cf_full(), _uce_full()])
    assert m8["threats"]["S14"]["status"] == "PARTIAL"
    assert m8["threats"]["S14"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication", "rogue-agent"]
    m9 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full(), _cf_full(), _uce_full(), _ra_full()])
    assert m9["threats"]["S14"]["status"] == "PARTIAL"
    assert m9["threats"]["S14"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication"]
    m10 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full(), _cf_full(), _uce_full(), _ra_full(),
         _iac_full()])
    assert m10["threats"]["S14"]["status"] == "PARTIAL"
    assert m10["threats"]["S14"]["untested_attacks"] == \
        ["cross-agent-deputy"]
    m11 = coverage.build_manifest(
        [others, _tm_full(), _agh_full(), _hat_full(), _mcp_full(),
         _asc_full(), _ipa_full(), _cf_full(), _uce_full(), _ra_full(),
         _iac_full(), _cad_full()])
    assert m11["threats"]["S14"]["status"] == "COVERED"


def test_asi09_maps_to_human_agent_trust():
    assert coverage.mapped_attacks("ASI09") == ["human-agent-trust"]


def test_complete_human_agent_trust_artifact_covers_asi09():
    m = coverage.build_manifest([_hat_full()], applicable_agentic=True)
    assert m["threats"]["ASI09"]["status"] == "COVERED"
    assert m["threats"]["ASI09"]["mapped_attacks"] == ["human-agent-trust"]
    assert "human-agent-trust" in m["attacks_measured"]
    m0 = coverage.build_manifest([_hat_full()])
    assert m0["threats"]["ASI09"]["status"] == "OUT_OF_SCOPE"


def test_hat_control_only_leaves_asi09_llm07_s14_unmeasured():
    control_only = art(_hat_cell("benign-task", "vulnerable"),
                       _hat_cell("benign-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "human-agent-trust" not in m["attacks_measured"]
    assert m["threats"]["ASI09"]["status"] == "UNMEASURED"
    assert m["threats"]["LLM07"]["status"] == "UNMEASURED"
    assert m["threats"]["S14"]["status"] == "UNMEASURED"


def test_hat_completeness_is_outcome_independent():
    bad = _hat_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI09"]["status"] == "COVERED"


def test_hat_missing_one_world_does_not_cover():
    cells = [_hat_cell(a, w) for a in _HAT_ARMS
             for w in ("vulnerable", "defended", "none")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI09"]["status"] == "UNMEASURED"


def test_hat_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _HAT_ARMS:
            for w in _HAT_WORLDS:
                c = _hat_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI09"]["status"] == "UNMEASURED", key


def test_asi06_maps_to_memory_context_poisoning():
    assert coverage.mapped_attacks("ASI06") == ["memory-context-poisoning"]


def test_complete_memory_poisoning_artifact_covers_asi06():
    m = coverage.build_manifest([_mcp_full()], applicable_agentic=True)
    assert m["threats"]["ASI06"]["status"] == "COVERED"
    assert m["threats"]["ASI06"]["mapped_attacks"] == ["memory-context-poisoning"]
    assert "memory-context-poisoning" in m["attacks_measured"]
    m0 = coverage.build_manifest([_mcp_full()])
    assert m0["threats"]["ASI06"]["status"] == "OUT_OF_SCOPE"


def test_mcp_control_only_leaves_asi06_llm01_s14_unmeasured():
    control_only = art(_mcp_cell("benign-task", "vulnerable"),
                       _mcp_cell("benign-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "memory-context-poisoning" not in m["attacks_measured"]
    assert m["threats"]["ASI06"]["status"] == "UNMEASURED"
    assert m["threats"]["LLM01"]["status"] == "UNMEASURED"
    assert m["threats"]["S14"]["status"] == "UNMEASURED"


def test_mcp_completeness_is_outcome_independent():
    bad = _mcp_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI06"]["status"] == "COVERED"


def test_mcp_missing_one_world_does_not_cover():
    cells = [_mcp_cell(a, w) for a in _MCP_ARMS
             for w in ("vulnerable", "defended", "none",
                       "memory-provenance-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI06"]["status"] == "UNMEASURED"


def test_mcp_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _MCP_ARMS:
            for w in _MCP_WORLDS:
                c = _mcp_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI06"]["status"] == "UNMEASURED", key


def test_asi04_maps_to_agentic_supply_chain():
    assert coverage.mapped_attacks("ASI04") == ["agentic-supply-chain"]


def test_complete_supply_chain_artifact_covers_asi04():
    m = coverage.build_manifest([_asc_full()], applicable_agentic=True)
    assert m["threats"]["ASI04"]["status"] == "COVERED"
    assert m["threats"]["ASI04"]["mapped_attacks"] == ["agentic-supply-chain"]
    assert "agentic-supply-chain" in m["attacks_measured"]
    m0 = coverage.build_manifest([_asc_full()])
    assert m0["threats"]["ASI04"]["status"] == "OUT_OF_SCOPE"


def test_asc_control_only_leaves_asi04_llm01_s14_unmeasured():
    control_only = art(_asc_cell("benign-task", "vulnerable"),
                       _asc_cell("benign-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "agentic-supply-chain" not in m["attacks_measured"]
    assert m["threats"]["ASI04"]["status"] == "UNMEASURED"
    assert "agentic-supply-chain" not in \
        m["threats"]["LLM01"]["measured_attacks"]
    assert "agentic-supply-chain" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_asc_completeness_is_outcome_independent():
    bad = _asc_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI04"]["status"] == "COVERED"


def test_asc_missing_one_world_does_not_cover():
    cells = [_asc_cell(a, w) for a in _ASC_ARMS
             for w in ("vulnerable", "defended", "none",
                       "component-provenance-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI04"]["status"] == "UNMEASURED"


def test_asc_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _ASC_ARMS:
            for w in _ASC_WORLDS:
                c = _asc_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI04"]["status"] == "UNMEASURED", key


def test_asi03_maps_to_identity_privilege_abuse():
    assert coverage.mapped_attacks("ASI03") == ["identity-privilege-abuse"]


def test_complete_identity_abuse_artifact_covers_asi03():
    m = coverage.build_manifest([_ipa_full()], applicable_agentic=True)
    assert m["threats"]["ASI03"]["status"] == "COVERED"
    assert m["threats"]["ASI03"]["mapped_attacks"] == \
        ["identity-privilege-abuse"]
    assert "identity-privilege-abuse" in m["attacks_measured"]
    m0 = coverage.build_manifest([_ipa_full()])
    assert m0["threats"]["ASI03"]["status"] == "OUT_OF_SCOPE"


def test_ipa_control_only_leaves_asi03_llm03_s14_unmeasured():
    control_only = art(_ipa_cell("authorized-task", "vulnerable"),
                       _ipa_cell("authorized-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "identity-privilege-abuse" not in m["attacks_measured"]
    assert m["threats"]["ASI03"]["status"] == "UNMEASURED"
    assert "identity-privilege-abuse" not in \
        m["threats"]["LLM03"]["measured_attacks"]
    assert "identity-privilege-abuse" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_ipa_completeness_is_outcome_independent():
    bad = _ipa_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI03"]["status"] == "COVERED"


def test_ipa_missing_one_world_does_not_cover():
    cells = [_ipa_cell(a, w) for a in _IPA_ARMS
             for w in ("vulnerable", "defended", "none",
                       "principal-binding-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI03"]["status"] == "UNMEASURED"


def test_ipa_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _IPA_ARMS:
            for w in _IPA_WORLDS:
                c = _ipa_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI03"]["status"] == "UNMEASURED", key


def test_llm03_six_to_seven_transition():
    base = art(*_measured("prompt-injection"), *_measured("tool-misuse"))
    m = coverage.build_manifest([base])
    assert m["threats"]["LLM03"]["status"] == "PARTIAL"
    assert m["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy", "identity-privilege-abuse",
         "insecure-agent-communication", "rogue-agent",
         "unexpected-code-execution"]
    m2 = coverage.build_manifest([base, _ipa_full()])
    assert m2["threats"]["LLM03"]["status"] == "PARTIAL"
    assert m2["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication", "rogue-agent",
         "unexpected-code-execution"]
    m3 = coverage.build_manifest([base, _ipa_full(), _uce_full()])
    assert m3["threats"]["LLM03"]["status"] == "PARTIAL"
    assert m3["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication", "rogue-agent"]
    m4 = coverage.build_manifest([base, _ipa_full(), _uce_full(), _ra_full()])
    assert m4["threats"]["LLM03"]["status"] == "PARTIAL"
    assert m4["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy", "insecure-agent-communication"]
    m5 = coverage.build_manifest(
        [base, _ipa_full(), _uce_full(), _ra_full(), _iac_full()])
    assert m5["threats"]["LLM03"]["status"] == "PARTIAL"
    assert m5["threats"]["LLM03"]["untested_attacks"] == \
        ["cross-agent-deputy"]
    m6 = coverage.build_manifest(
        [base, _ipa_full(), _uce_full(), _ra_full(), _iac_full(),
         _cad_full()])
    assert m6["threats"]["LLM03"]["status"] == "COVERED"


def test_asi05_maps_to_unexpected_code_execution():
    assert coverage.mapped_attacks("ASI05") == ["unexpected-code-execution"]


def test_complete_unexpected_code_execution_artifact_covers_asi05():
    m = coverage.build_manifest([_uce_full()], applicable_agentic=True)
    assert m["threats"]["ASI05"]["status"] == "COVERED"
    assert m["threats"]["ASI05"]["mapped_attacks"] == \
        ["unexpected-code-execution"]
    assert "unexpected-code-execution" in m["attacks_measured"]
    m0 = coverage.build_manifest([_uce_full()])
    assert m0["threats"]["ASI05"]["status"] == "OUT_OF_SCOPE"


def test_uce_control_only_leaves_asi05_llm03_s14_unmeasured():
    control_only = art(_uce_cell("authorized-exec", "vulnerable"),
                       _uce_cell("authorized-exec", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "unexpected-code-execution" not in m["attacks_measured"]
    assert m["threats"]["ASI05"]["status"] == "UNMEASURED"
    assert "unexpected-code-execution" not in \
        m["threats"]["LLM03"]["measured_attacks"]
    assert "unexpected-code-execution" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_uce_completeness_is_outcome_independent():
    bad = _uce_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI05"]["status"] == "COVERED"


def test_uce_missing_one_world_does_not_cover():
    cells = [_uce_cell(a, w) for a in _UCE_ARMS
             for w in ("vulnerable", "defended", "none",
                       "instruction-provenance-only",
                       "parameter-typecheck-only", "tenant-containment-only",
                       "component-provenance-only", "principal-binding-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI05"]["status"] == "UNMEASURED"


def test_uce_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _UCE_ARMS:
            for w in _UCE_WORLDS:
                c = _uce_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI05"]["status"] == "UNMEASURED", key


def test_asi10_maps_to_rogue_agent():
    assert coverage.mapped_attacks("ASI10") == ["rogue-agent"]


def test_complete_rogue_agent_artifact_covers_asi10():
    m = coverage.build_manifest([_ra_full()], applicable_agentic=True)
    assert m["threats"]["ASI10"]["status"] == "COVERED"
    assert m["threats"]["ASI10"]["mapped_attacks"] == ["rogue-agent"]
    assert "rogue-agent" in m["attacks_measured"]
    m0 = coverage.build_manifest([_ra_full()])
    assert m0["threats"]["ASI10"]["status"] == "OUT_OF_SCOPE"


def test_ra_control_only_leaves_asi10_llm03_s14_unmeasured():
    control_only = art(_ra_cell("attested-lifecycle", "vulnerable"),
                       _ra_cell("attested-lifecycle", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "rogue-agent" not in m["attacks_measured"]
    assert m["threats"]["ASI10"]["status"] == "UNMEASURED"
    assert "rogue-agent" not in \
        m["threats"]["LLM03"]["measured_attacks"]
    assert "rogue-agent" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_ra_completeness_is_outcome_independent():
    bad = _ra_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI10"]["status"] == "COVERED"


def test_ra_missing_one_world_does_not_cover():
    cells = [_ra_cell(a, w) for a in _RA_ARMS
             for w in ("vulnerable", "defended", "none",
                       "instruction-provenance-only", "memory-provenance-only",
                       "parameter-typecheck-only", "tenant-containment-only",
                       "principal-binding-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI10"]["status"] == "UNMEASURED"


def test_ra_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _RA_ARMS:
            for w in _RA_WORLDS:
                c = _ra_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI10"]["status"] == "UNMEASURED", key


def test_asi07_maps_to_insecure_agent_communication():
    assert coverage.mapped_attacks("ASI07") == ["insecure-agent-communication"]


def test_complete_insecure_agent_communication_artifact_covers_asi07():
    m = coverage.build_manifest([_iac_full()], applicable_agentic=True)
    assert m["threats"]["ASI07"]["status"] == "COVERED"
    assert m["threats"]["ASI07"]["mapped_attacks"] == \
        ["insecure-agent-communication"]
    assert "insecure-agent-communication" in m["attacks_measured"]
    m0 = coverage.build_manifest([_iac_full()])
    assert m0["threats"]["ASI07"]["status"] == "OUT_OF_SCOPE"


def test_iac_control_only_leaves_asi07_llm03_s14_unmeasured():
    control_only = art(_iac_cell("authentic-relay", "vulnerable"),
                       _iac_cell("authentic-relay", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "insecure-agent-communication" not in m["attacks_measured"]
    assert m["threats"]["ASI07"]["status"] == "UNMEASURED"
    assert "insecure-agent-communication" not in \
        m["threats"]["LLM03"]["measured_attacks"]
    assert "insecure-agent-communication" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_iac_completeness_is_outcome_independent():
    bad = _iac_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI07"]["status"] == "COVERED"


def test_iac_missing_one_world_does_not_cover():
    cells = [_iac_cell(a, w) for a in _IAC_ARMS
             for w in ("vulnerable", "defended", "none",
                       "instruction-provenance-only", "memory-provenance-only",
                       "component-provenance-only",
                       "cascade-revalidation-only", "principal-binding-only",
                       "parameter-typecheck-only", "tenant-containment-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI07"]["status"] == "UNMEASURED"


def test_iac_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _IAC_ARMS:
            for w in _IAC_WORLDS:
                c = _iac_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI07"]["status"] == "UNMEASURED", key
def test_asi08_maps_to_cascading_failure():
    assert coverage.mapped_attacks("ASI08") == ["cascading-failure"]


def test_complete_cascading_failure_artifact_covers_asi08():
    m = coverage.build_manifest([_cf_full()], applicable_agentic=True)
    assert m["threats"]["ASI08"]["status"] == "COVERED"
    assert m["threats"]["ASI08"]["mapped_attacks"] == ["cascading-failure"]
    assert "cascading-failure" in m["attacks_measured"]
    m0 = coverage.build_manifest([_cf_full()])
    assert m0["threats"]["ASI08"]["status"] == "OUT_OF_SCOPE"


def test_cf_control_only_leaves_asi08_llm01_s14_unmeasured():
    control_only = art(_cf_cell("benign-task", "vulnerable"),
                       _cf_cell("benign-task", "defended"))
    m = coverage.build_manifest([control_only], applicable_agentic=True)
    assert "cascading-failure" not in m["attacks_measured"]
    assert m["threats"]["ASI08"]["status"] == "UNMEASURED"
    assert "cascading-failure" not in \
        m["threats"]["LLM01"]["measured_attacks"]
    assert "cascading-failure" not in \
        m["threats"]["S14"]["measured_attacks"]


def test_cf_completeness_is_outcome_independent():
    bad = _cf_full(total_violations=2, violation_rate=1.0)
    m = coverage.build_manifest([bad], applicable_agentic=True)
    assert m["threats"]["ASI08"]["status"] == "COVERED"


def test_cf_missing_one_world_does_not_cover():
    cells = [_cf_cell(a, w) for a in _CF_ARMS
             for w in ("vulnerable", "defended", "none",
                       "instruction-provenance-only", "parameter-typecheck-only",
                       "tenant-containment-only", "component-provenance-only")]
    m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
    assert m["threats"]["ASI08"]["status"] == "UNMEASURED"


def test_cf_missing_each_metric_key_does_not_cover():
    for key in ("trials", "total_violations", "violation_rate",
                "wilson_upper", "violations", "false_blocks"):
        cells = []
        for a in _CF_ARMS:
            for w in _CF_WORLDS:
                c = _cf_cell(a, w)
                del c[key]
                cells.append(c)
        m = coverage.build_manifest([art(*cells)], applicable_agentic=True)
        assert m["threats"]["ASI08"]["status"] == "UNMEASURED", key


def test_llm07_one_to_two_transition():
    m = coverage.build_manifest([art(*_measured("misinformation-integrity"))])
    assert m["threats"]["LLM07"]["status"] == "PARTIAL"
    assert m["threats"]["LLM07"]["untested_attacks"] == ["human-agent-trust"]
    m2 = coverage.build_manifest(
        [art(*_measured("misinformation-integrity")), _hat_full()])
    assert m2["threats"]["LLM07"]["status"] == "COVERED"


def test_out_of_scope_never_conflated_with_unmeasured():
    m = coverage.build_manifest([])
    asi = m["threats"]["ASI01"]
    assert asi["status"] == "OUT_OF_SCOPE"
    assert asi["out_of_scope_reason"]
    assert "ASI01" not in m["summary"]["unmeasured_applicable"]
    m2 = coverage.build_manifest([], applicable_agentic=True)
    assert m2["threats"]["ASI01"]["status"] == "UNMEASURED"
    assert "ASI01" in m2["summary"]["unmeasured_applicable"]


def test_extra_out_of_scope_and_custom_threats():
    m = coverage.build_manifest(
        [], extra_out_of_scope={"LLM07": "no misinformation target",
                                "X1": "custom scope exclusion"})
    assert m["threats"]["LLM07"]["status"] == "OUT_OF_SCOPE"
    assert m["threats"]["X1"]["status"] == "OUT_OF_SCOPE"
    assert "LLM07" not in m["summary"]["unmeasured_applicable"]


def test_unknown_experiment_ids_are_reported_not_dropped():
    m = coverage.build_manifest([art({"experiment": "mystery_exp"})])
    assert m["unknown_experiments"] == ["mystery_exp"]
    assert m["attacks_measured"] == []


def test_universe_covers_surfaces_owasp_and_agentic():
    u = coverage.threat_universe()
    assert {"S1", "S14", "LLM01", "LLM10", "ASI01", "ASI10"} <= set(u)
    assert all(u[t]["out_of_scope"] for t in u if t.startswith("ASI"))
    u2 = coverage.threat_universe(applicable_agentic=True)
    assert not any(u2[t]["out_of_scope"] for t in u2 if t.startswith("ASI"))


def test_summary_counts_add_up():
    m = coverage.build_manifest([art(GOOD)])
    s = m["summary"]
    total = s["COVERED"] + s["PARTIAL"] + s["UNMEASURED"] + s["OUT_OF_SCOPE"]
    assert total == len(m["threats"])
    assert s["applicable"] == total - s["OUT_OF_SCOPE"]


def test_render_manifest_md():
    m = coverage.build_manifest([art(GOOD)])
    md = coverage.render_manifest_md(m)
    assert "| threat | name | status |" in md
    assert "COVERED" in md and "UNMEASURED" in md
    assert "S7" in md


_V = 0.5

_LANE_C_REAL = {
    "seam_inversion": [
        (None, None, {"seq_accuracy_mean": _V, "control_accuracy_mean": _V,
                      "gate_verdicts": ["INCONCLUSIVE"]})],
    "norm_based_scoring": [
        (None, "null", {"auc_mean": _V, "gate_verdicts": ["x"]}),
        (None, "separable", {"auc_mean": _V, "gate_verdicts": ["x"]})],
    "lira_membership": [
        (None, "null", {"auc_mean": _V, "gate_verdicts": ["x"]}),
        (None, "separable", {"auc_mean": _V, "gate_verdicts": ["x"]})],
    "gpu_memory_residue": [
        ("allocator-residue", None,
         {"token_recovery_mean": _V, "gate_verdict": "x"}),
        ("scrub-check", None,
         {"canary_recovery_mean": _V, "gate_verdict": "x"}),
        ("um-timing", None,
         {"schedule_recovery_auc_mean": _V, "gate_verdict": "x"})],
    "wire_shape_analysis": [
        ("length-leak", None, {"mae_tokens_mean": _V, "rank_corr_mean": _V}),
        ("phase-detect", None, {"f1_mean": _V, "label_accuracy_mean": _V})],
    "multi_tenant_isolation": [
        (arm, world, {"canary_recovery_exact_mean": _V,
                      "isolation_failures_per_million_mean": _V})
        for arm in ("session-collision", "continuous-batch-mixup",
                    "error-path-bleed", "kv-content-canary",
                    "prefix-cache-isolation", "response-misattribution",
                    "allocator-reuse")
        for world in ("vulnerable", "defended", "isolated")],
    "router_side_channel": [
        (arm, world, {k: _V for k in keys})
        for arm, keys in (
            ("contention-fingerprinting",
             ("auc_mean", "tpr_at_1pct_fpr_mean")),
            ("expert-map-extraction", ("fidelity_mean", "lift_pp_mean")),
            ("length-domain-inference",
             ("domain_accuracy_mean", "mean_domain_auc_mean")),
            ("linkability", ("auc_mean", "tpr_at_1pct_fpr_mean")),
            ("property-classification", ("auc_mean", "tpr_at_1pct_fpr_mean")))
        for world in ("informative", "null")],
    "artifact_telemetry_leak": [
        ("store-scan", None, {"n_hits_total": 3,
                              "unauthorized_readability_total": 1,
                              "classes_exposed_overall": ["prompt"]})],
    "prompt_injection": [
        (arm, world, {"trials": 2, "total_violations": 0,
                      "violation_rate": 0.0, "wilson_upper": _V,
                      "violations": {"disclosure": 0}})
        for arm in ("direct-override", "indirect-document",
                    "cross-session-persistence", "payload-splitting",
                    "encoded-unicode", "adaptive")
        for world in ("vulnerable", "defended")],
    "rag_vector_abuse": [
        (arm, world, {"rate": 0.0, "trials": 2, "wilson_upper": _V, "hits": 0})
        for arm in ("cross-tenant-retrieval", "namespace-bypass",
                    "poisoned-ingestion", "embedding-inversion",
                    "membership-linkability", "stale-retention",
                    "retrieval-flooding", "citation-substitution")
        for world in ("vulnerable", "defended")],
    "hidden_context_exposure": [
        (arm, world, {"exact_recovery": 0.0, "semantic": 0.0, "mean_turns": 1})
        for arm in ("direct-extraction", "behavioral-inference",
                    "prefix-continuation", "encoded-extraction",
                    "multi-turn-accumulation")
        for world in ("vulnerable", "defended", "empty")],
    "output_sink_injection": [
        (arm, world, {"trials": 2, "effect_rate": 0.0, "wilson_upper": _V,
                      "effects": 0})
        for arm in ("html-xss", "sql-injection", "shell-command",
                    "template-ssti", "url-ssrf", "file-path",
                    "generated-code", "structured-output", "adaptive")
        for world in ("vulnerable", "defended")],
    "misinformation_integrity": [
        (arm, world, {**{k: 0.0 for k in keys}, "per_seed": [0.0]})
        for arm, keys in (
            ("grounded-qa", ("accuracy",)),
            ("citation-fidelity", ("entailment_rate",)),
            ("adversarial-premise", ("premise_acceptance_rate",)),
            ("fabricated-sources", ("fabricated_citation_rate",)),
            ("calibration", ("ece",)),
            ("contradiction", ("contradiction_rate",)),
            ("temporal-staleness", ("stale_answer_rate",)),
            ("downstream-decision", ("action_harm_rate",)))
        for world in ("vulnerable", "defended")],
}


def _lane_c_records(eid, cells):
    return [{"experiment": eid, "arm": arm, "world": world, **metrics}
            for (arm, world, metrics) in cells]


@pytest.mark.parametrize("eid", sorted(_LANE_C_REAL))
def test_lane_c_complete_artifact_is_measured(eid):
    a = art(*_lane_c_records(eid, _LANE_C_REAL[eid]))
    assert eid in coverage.experiments_of(a)


@pytest.mark.parametrize("eid", sorted(_LANE_C_REAL))
def test_lane_c_skeleton_one_row_bites(eid):
    skeleton = art({"experiment": eid, "note": "benign smoke"})
    assert eid not in coverage.experiments_of(skeleton)


@pytest.mark.parametrize("eid", sorted(_LANE_C_REAL))
def test_lane_c_dropping_one_cell_bites(eid):
    cells = _LANE_C_REAL[eid]
    if len(cells) < 2:
        pytest.skip("single-cell experiment: covered by the skeleton test")
    a = art(*_lane_c_records(eid, cells[:-1]))
    assert eid not in coverage.experiments_of(a)


@pytest.mark.parametrize("eid", sorted(_LANE_C_REAL))
def test_lane_c_gating_is_outcome_independent(eid):
    cells = [(arm, world, {k: 1.0 for k in metrics})
             for (arm, world, metrics) in _LANE_C_REAL[eid]]
    assert eid in coverage.experiments_of(art(*_lane_c_records(eid, cells)))


def test_lane_c_deferred_experiments_stay_ungated():
    for eid in ("dlgpp_gradient_inversion", "ea4_membership"):
        assert eid not in coverage.COMPLETENESS
        a = art({"experiment": eid, "metric": 1.0})
        assert eid in coverage.experiments_of(a)


def test_cli_writes_manifest_and_md(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(art(GOOD)))
    out = tmp_path / "cov.json"
    md = tmp_path / "cov.md"
    rc = coverage.main([str(a), "--output", str(out), "--md", str(md)])
    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["schema"] == coverage.SCHEMA
    assert manifest["artifact_count"] == 1
    assert "Coverage manifest" in md.read_text()


def test_cli_dispatches_through_main(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(art(GOOD)))
    r = subprocess.run([sys.executable, "-m", "attacker", "--coverage",
                        str(a)], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Coverage manifest" in r.stdout


def test_self_test_passes():
    assert coverage.self_test() == 0
