"""The additive extension of attacker/coverage.py for the seven integrity/
availability experiments (seed Phase 4): the non-Cartesian per-cell
COMPLETENESS entries, the fact that the four agentic entries are byte-identical
before/after (regression), and the profile reading 2 COVERED / 3 PARTIAL."""

import glob
import json
import os

from attacker import coverage, profiles

HERE = os.path.dirname(__file__)
ARTS = sorted(glob.glob(os.path.join(HERE, "data", "integrity_artifacts",
                                     "*.json")))
SEVEN = ("protocol_integrity", "malicious_client", "malicious_server_steering",
         "poisoning_backdoor", "resource_exhaustion", "key_session_abuse",
         "supply_chain")
FOUR_AGENTIC = ("tool_misuse", "agent_goal_hijack", "human_agent_trust",
                "memory_context_poisoning")


def test_four_agentic_entries_unchanged_rectangular_shape():
    for k in FOUR_AGENTIC:
        req = coverage.COMPLETENESS[k]
        assert set(req) == {"arms", "worlds", "metrics"}, k
        assert "cells" not in req


def _legacy_cells_complete(records, req):
    """The ORIGINAL rectangular-only _cells_complete (pre-extension), verbatim
    from the merged main, to prove the extension is ADDITIVE for the legacy
    shape (the four agentic entries score identically before/after)."""
    have = set()
    for rec in records:
        if not isinstance(rec, dict) or "error" in rec:
            continue
        cell = (rec.get("arm"), rec.get("world"))
        if cell[0] in req["arms"] and cell[1] in req["worlds"] \
                and all(k in rec for k in req["metrics"]):
            have.add(cell)
    need = {(a, w) for a in req["arms"] for w in req["worlds"]}
    return need <= have


def test_extension_is_additive_for_the_four_agentic_entries():
    """New _cells_complete == legacy _cells_complete for every agentic entry,
    on a complete record set, an incomplete one, and an error-laced one."""
    for k in FOUR_AGENTIC:
        req = coverage.COMPLETENESS[k]
        complete = [{"experiment": k, "arm": a, "world": w,
                     **{m: 1.0 for m in req["metrics"]}}
                    for a in req["arms"] for w in req["worlds"]]
        missing_one = complete[1:]
        with_error = [{**complete[0], "error": "boom"}] + complete[1:]
        for records in (complete, missing_one, with_error, []):
            assert coverage._cells_complete(records, req) \
                == _legacy_cells_complete(records, req), (k, len(records))


def test_seven_integrity_entries_are_the_explicit_cells_form():
    for k in SEVEN:
        req = coverage.COMPLETENESS[k]
        assert "cells" in req and req["cells"], k
        assert all(isinstance(key, tuple) and len(key) == 2
                   for key in req["cells"])


def test_topology_is_non_cartesian():
    assert len(coverage.COMPLETENESS["protocol_integrity"]["cells"]) == 21
    assert len(coverage.COMPLETENESS["key_session_abuse"]["cells"]) == 17
    ms = coverage.COMPLETENESS["malicious_server_steering"]["cells"]
    assert set(ms) == {(None, "honest"), (None, "crafted")}


def test_full_summaries_mark_all_seven_measured():
    measured = set(coverage.build_manifest(ARTS)["attacks_measured"])
    for name in ("protocol-integrity", "malicious-client", "malicious-server",
                 "poisoning-backdoor", "resource-exhaustion",
                 "key-session-abuse", "supply-chain"):
        assert name in measured, name


def test_profile_reads_two_covered_three_partial_seven_measured():
    rep = profiles.evaluate_profile(
        "benchmark.split-integrity-availability.v1", ARTS)
    status = {t: v["status"] for t, v in rep["threats"].items()}
    assert status["LLM04"] == "COVERED" and status["S13"] == "COVERED"
    assert status["S10"] == status["LLM05"] == status["LLM06"] == "PARTIAL"
    assert rep["summary"]["COVERED"] == 2 and rep["summary"]["PARTIAL"] == 3
    assert rep["summary"]["attacks_measured"] == 7


def test_dropping_one_required_cell_de_measures_the_experiment():
    sc = json.load(open(os.path.join(HERE, "data", "integrity_artifacts",
                                     "supply_chain.json")))
    dropped = {**sc, "summary": [r for r in sc["summary"]
                                 if r.get("arm") != "digest-mismatch"]}
    m = coverage.build_manifest([dropped])
    assert "supply-chain" not in m["attacks_measured"]
    rep = profiles.evaluate_profile(
        "benchmark.split-integrity-availability.v1", [dropped])
    assert rep["threats"]["LLM04"]["status"] != "COVERED"


def test_a_complete_tool_misuse_artifact_still_covered():
    """The agentic path is unchanged: a complete tool-misuse artifact still
    marks ASI02 COVERED (the extension did not disturb the legacy branch)."""
    req = coverage.COMPLETENESS["tool_misuse"]
    recs = [{"experiment": "tool_misuse", "arm": a, "world": w,
             **{m: 1.0 for m in req["metrics"]}}
            for a in req["arms"] for w in req["worlds"]]
    art = {"schema": "t", "config": {}, "threat_model": "t", "provenance": {},
           "results": recs, "summary": []}
    m = coverage.build_manifest([art], applicable_agentic=True)
    assert m["threats"]["ASI02"]["status"] == "COVERED"
