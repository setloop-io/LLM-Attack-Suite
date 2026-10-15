"""integrity-gate.v1 (attacker/integrity_gate.py): the 113-rule table, the
absolute (fail-closed) scoring, the D1 1.5x ratio contract, the D5 count
scale, the D7 `integrity_value` naming, and the tamper-sealed serialization.

Every test scores REAL calibration values against the real gate — no stubs."""

import math

import pytest

from attacker import integrity_gate as ig
from attacker.scorecard import _excess

G = ig.DEFAULT_INTEGRITY_GATE


def cell(exp, arm, world, seeds=3, **metrics):
    return {"experiment": exp, "arm": arm, "world": world, "n_seeds": seeds,
            **metrics}


def test_rule_table_has_113_unique_cells():
    assert len(G.rules) == 113
    coords = {(r.experiment, r.arm, r.world) for r in G.rules}
    assert len(coords) == 113


def test_per_axis_cell_counts_match_calibration_section4():
    counts = {}
    for r in G.rules:
        for ax in r.axes:
            counts[ax] = counts.get(ax, 0) + 1
    assert counts == {"S10": 39, "S13": 28, "LLM04": 25, "LLM05": 31,
                      "LLM06": 15}


def test_every_rule_role_in_vocabulary():
    assert all(r.role in ig.ROLES for r in G.rules)


def test_two_thresholds_flip_the_same_cell():
    fixed = cell("resource_exhaustion", "retry-storm", None,
                 amplification_vs_control_mean=1.400244)

    def gate(thr):
        return ig.IntegrityGate(rules=(ig.MetricRule(
            "t", "resource_exhaustion", "retry-storm", None, ("LLM06",),
            "amplification_vs_control_mean", "const", None, "ratio",
            "higher_worse", thr, "defense"),))
    assert ig.score_cell(fixed, [], gate(1.25))["verdict"] == "OPEN"
    assert ig.score_cell(fixed, [], gate(1.5))["verdict"] == "CLOSED"


def test_zero_rule_matches_is_unmeasured():
    s = ig.score_cell(cell("nope", "x", "y", metric=1.0), [], G)
    assert s["verdict"] == "UNMEASURED" and "no rule" in s["reason"]


def test_two_rule_matches_raises():
    r0 = G.rules[0]
    dup = ig.IntegrityGate(rules=G.rules + (r0,))
    with pytest.raises(ValueError):
        ig.score_cell(cell(r0.experiment, r0.arm, r0.world,
                           **{r0.signal_key: 1.0}), [], dup)


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), float("-inf"), True, "1.0", -0.1, 1.5])
def test_fail_closed_bad_values_are_unmeasured(bad):
    s = ig.score_cell(cell("protocol_integrity", "drop", "defended",
                           undetected_mutation_rate_mean=bad), [], G)
    assert s["verdict"] == "UNMEASURED"


@pytest.mark.parametrize("seeds", [2, None, 2.5, "3", True])
def test_fail_closed_bad_seed_counts_are_unmeasured(seeds):
    c = cell("protocol_integrity", "drop", "defended", seeds=seeds,
             undetected_mutation_rate_mean=1.0)
    assert ig.score_cell(c, [], G)["verdict"] == "UNMEASURED"


def test_nan_would_grade_closed_naively_but_is_unmeasured():
    assert (float("nan") > 0.05) is False
    s = ig.score_cell(cell("protocol_integrity", "drop", "defended",
                           undetected_mutation_rate_mean=float("nan")), [], G)
    assert s["verdict"] == "UNMEASURED" and s["integrity_value"] is None


def test_d7_ratio_contract_at_1_5x():
    def rex(arm, key, v):
        return ig.score_cell(cell("resource_exhaustion", arm, None,
                                  **{key: v}), [], G)["verdict"]
    assert rex("protocol-oversize", "amplification_vs_control_mean",
               23.254993) == "OPEN"
    assert rex("retry-storm", "amplification_vs_control_mean",
               1.400244) == "CLOSED"
    assert rex("benign-control", "amplification_parity_mean",
               1.018812) == "CLOSED"


def test_scored_field_is_integrity_value_not_excess():
    s = ig.score_cell(cell("resource_exhaustion", "protocol-oversize", None,
                           amplification_vs_control_mean=23.254993), [], G)
    assert "integrity_value" in s and s["integrity_value"] == 23.254993
    assert _excess(s) == (None, None)


def test_two_axis_cell_scored_once():
    r = ig.match_rule(cell("protocol_integrity", "oversize", "vulnerable"), G)
    assert r.axes == ("S10", "LLM06")
    s = ig.score_cell(cell("protocol_integrity", "oversize", "vulnerable",
                           undetected_mutation_rate_mean=1.0), [], G)
    assert s["verdict"] == "OPEN" and s["integrity_value"] == 1.0


def test_count_scale_zero_closed_one_open_no_divzero():
    zero = ig.score_cell(cell("supply_chain", "all", "negative-control",
                              negative_control_false_rejections=0), [], G)
    one = ig.score_cell(cell("supply_chain", "all", "negative-control",
                             negative_control_false_rejections=1), [], G)
    assert zero["verdict"] == "CLOSED" and zero["integrity_severity"] == 0.0
    assert one["verdict"] == "OPEN" and one["integrity_severity"] == 1.0


def test_out_of_profile_axis_cell_is_unmeasured_with_reason():
    s = ig.score_cell(cell("malicious_client", "budget-extraction",
                           "vulnerable", attack_success_rate=0.5), [], G)
    assert s["verdict"] == "UNMEASURED"
    assert s["reason"] == "no axis in the profile threat set"
    assert s["integrity_threshold"] == 0.05


def test_malicious_server_cells_are_unmeasured():
    honest = ig.score_cell(cell("malicious_server_steering", "bwd-direction",
                                "honest", seq_accuracy_mean=0.0297), [], G)
    agg = ig.score_cell(cell("malicious_server_steering", "bwd-direction",
                             None, crafted_minus_honest_pp=92.8572), [], G)
    assert honest["verdict"] == "UNMEASURED" and honest["role"] == "control"
    assert agg["verdict"] == "UNMEASURED" and agg["integrity_threshold"] is None


def test_gate_config_round_trips():
    cfg = ig.gate_config(G)
    assert ig.gate_digest(ig.gate_from_config(cfg)) == ig.gate_digest(G)


def test_hand_edited_threshold_breaks_the_digest():
    import copy
    cfg = copy.deepcopy(ig.gate_config(G))
    cfg["rules"][0]["threshold"] = 999.0
    assert ig.gate_digest(ig.gate_from_config(cfg)) != ig.gate_digest(G)


def test_self_test_passes():
    assert ig.self_test() == 0
