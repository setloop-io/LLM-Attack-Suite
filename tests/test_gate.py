"""The frozen gate: constants, monotonicity, verdicts, retargeting."""

import pytest

from attacker import gate
from attacker.gate import (DEFAULT_GATE, GateConfig, bonferroni_z, floor_pp,
                           verdict, wilson_upper)


def test_bonferroni_z_reproduces_frozen_value():
    assert round(bonferroni_z(), 4) == 2.5392


def test_default_gate_is_the_frozen_config():
    assert (DEFAULT_GATE.alpha, DEFAULT_GATE.n_trials, DEFAULT_GATE.gate_pp,
            DEFAULT_GATE.n_seeds_required, DEFAULT_GATE.control) == (
        0.05, 9, 1.0, 3, "label_free_majority")
    assert DEFAULT_GATE.version == gate.GATE_VERSION == "frozen-gate.v2"
    with pytest.raises(Exception):
        DEFAULT_GATE.gate_pp = 5.0


def test_wilson_upper_matches_latent_probe_math():
    from attacker.attacks import latent_probe
    for correct, total in ((0, 100), (50, 100), (100, 100), (316, 8192)):
        assert wilson_upper(correct, total) == latent_probe.wilson_upper(
            correct, total, 0.05, trials=9)


def test_wilson_upper_monotonic_in_correct():
    values = [wilson_upper(c, 100) for c in (0, 1, 25, 50, 99, 100)]
    assert values == sorted(values)
    assert 0.0 < values[0] and values[-1] <= 1.0


def test_excess_monotonic_in_recovery():
    excesses = [verdict(r, 0.04, 8192)["excess_pp"]
                for r in (0.04, 0.05, 0.08, 0.20)]
    assert excesses == sorted(excesses)


def test_floor_shrinks_with_n_and_sits_in_campaign_band():
    assert floor_pp(0.04, 8192) > floor_pp(0.04, 40960) > 0.0
    assert 0.3 < floor_pp(0.03857, 8192) < 0.8


def test_verdict_pass_at_control_is_at_floor():
    v = verdict(0.03857, 0.03857, 8192, n_seeds=3)
    assert v["verdict"] == "PASS"
    assert v["at_floor"] is True
    assert 0.0 < v["excess_pp"] <= DEFAULT_GATE.gate_pp
    assert v["gate"] == "frozen-gate.v2" and round(v["z"], 4) == 2.5392


def test_verdict_fail_on_real_break():
    v = verdict(0.10, 0.03857, 40960)
    assert v["verdict"] == "FAIL"
    assert v["excess_pp"] > DEFAULT_GATE.gate_pp
    assert v["at_floor"] is False


def test_verdict_seed_requirement_reported():
    assert verdict(0.04, 0.04, 8192, n_seeds=1)["seeds_sufficient"] is False
    assert verdict(0.04, 0.04, 8192, n_seeds=3)["seeds_sufficient"] is True
    assert verdict(0.04, 0.04, 8192)["seeds_sufficient"] is None


def test_gate_retargets_via_config():
    custom = GateConfig(alpha=0.01, n_trials=1, gate_pp=2.0,
                        n_seeds_required=5, control="chance",
                        version="defender-gate.v2")
    v = verdict(0.05, 0.04, 8192, config=custom, n_seeds=4)
    assert v["gate"] == "defender-gate.v2"
    assert v["z"] != bonferroni_z(DEFAULT_GATE)
    assert v["gate_pp"] == 2.0 and v["n_seeds_required"] == 5
    assert v["control_definition"] == "chance"
    assert v["seeds_sufficient"] is False


def test_bonferroni_z_full_precision_pin():
    assert abs(bonferroni_z() - 2.5391848136513127) < 1e-12
    assert abs(bonferroni_z() - 2.5392) < 5e-5


def test_bonferroni_z_trial_count_guard():
    z0 = bonferroni_z(GateConfig(n_trials=0, version="guard-test"))
    z1 = bonferroni_z(GateConfig(n_trials=1, version="guard-test"))
    assert z0 == z1
    assert abs(z0 - 1.6448536269514715) < 1e-12
    assert bonferroni_z() > bonferroni_z(
        GateConfig(n_trials=1, version="guard-test"))


def _wilson_hand(correct, total, z=2.5392):
    """Independent re-implementation of the Wilson upper bound using the
    paper's 4-dp z (PR-101), for cross-checking the module math."""
    import math
    p = correct / total
    den = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total
                           + z * z / (4.0 * total * total))
    return (center + radius) / den


def test_wilson_upper_matches_hand_computation():
    for correct, total in ((0, 100), (1, 100), (50, 100), (100, 100),
                           (316, 8192), (348, 8192)):
        assert abs(wilson_upper(correct, total)
                   - _wilson_hand(correct, total)) < 1e-5


def test_wilson_upper_full_precision_pins():
    expected = {(0, 100): 0.06056940717119347,
                (1, 100): 0.07815602520793363,
                (50, 100): 0.6230542636108086,
                (100, 100): 1.0,
                (316, 8192): 0.04434978515023801}
    for (correct, total), want in expected.items():
        assert abs(wilson_upper(correct, total) - want) < 1e-12


def test_wilson_upper_edges():
    import math
    assert math.isnan(wilson_upper(1, 0))
    assert math.isnan(wilson_upper(1, -5))
    assert wilson_upper(0, 8192) > 0.0
    assert abs(wilson_upper(8192, 8192) - 1.0) < 1e-12
    assert abs(wilson_upper(100, 100) - 1.0) < 1e-12


def test_pass_fail_boundary_at_campaign_config():
    v_pass = verdict(348 / 8192, 0.03857, 8192, n_seeds=3)
    v_fail = verdict(349 / 8192, 0.03857, 8192)
    assert abs(v_pass["excess_pp"] - 0.9937544651172845) < 1e-9
    assert v_pass["verdict"] == "PASS" and v_pass["at_floor"] is False
    assert abs(v_fail["excess_pp"] - 1.0067256572266134) < 1e-9
    assert v_fail["verdict"] == "FAIL" and v_fail["at_floor"] is False


def test_pass_fail_boundary_is_inclusive_at_exact_equality():
    base = verdict(0.05, 0.04, 8192)
    excess = base["excess_pp"]
    at = GateConfig(gate_pp=excess, version="boundary-test")
    just_below = GateConfig(gate_pp=excess - 1e-9, version="boundary-test")
    assert verdict(0.05, 0.04, 8192, config=at, n_seeds=3)["verdict"] == "PASS"
    assert verdict(0.05, 0.04, 8192, config=just_below)["verdict"] == "FAIL"


def test_floor_pp_full_precision_pins():
    assert abs(floor_pp(0.03857, 8192) - 0.5779785150238013) < 1e-12
    assert abs(floor_pp(0.04, 8192) - 0.5910578707115949) < 1e-12
    assert abs(floor_pp(0.04, 40960) - 0.25217724947377007) < 1e-12


def test_recovery_equal_control_scores_exactly_the_floor():
    v = verdict(0.03857, 0.03857, 8192, n_seeds=3)
    assert v["excess_pp"] == v["floor_pp"]
    assert v["at_floor"] is True and v["verdict"] == "PASS"


def test_seed_enforcement_is_reported_independently_of_verdict():
    blown = verdict(0.10, 0.03857, 40960, n_seeds=1)
    assert blown["verdict"] == "FAIL" and blown["seeds_sufficient"] is False
    clean = verdict(0.03857, 0.03857, 8192, n_seeds=2)
    assert clean["verdict"] == "INSUFFICIENT_EVIDENCE" and clean["seeds_sufficient"] is False
    assert verdict(0.03857, 0.03857, 8192,
                   n_seeds=99)["seeds_sufficient"] is True
    strict = GateConfig(n_seeds_required=5, version="seed-test")
    assert verdict(0.04, 0.04, 8192, config=strict,
                   n_seeds=4)["seeds_sufficient"] is False


def test_config_mutation_is_impossible_retarget_makes_new_version():
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_GATE.gate_pp = 9.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_GATE.version = "sneaky.v2"
    v2 = dataclasses.replace(DEFAULT_GATE, gate_pp=0.5,
                             version="tighter-gate.v2")
    assert v2.version == "tighter-gate.v2" and v2.gate_pp == 0.5
    assert (DEFAULT_GATE.gate_pp, DEFAULT_GATE.version) == (1.0,
                                                            "frozen-gate.v2")
    assert verdict(0.05, 0.04, 8192)["gate"] == "frozen-gate.v2"
    assert verdict(0.05, 0.04, 8192, config=v2)["gate"] == "tighter-gate.v2"


def test_verdict_record_schema_is_complete():
    v = verdict(0.05, 0.04, 8192, n_seeds=3)
    assert set(v) == {"gate", "z", "n", "recovery_pct", "control_pct",
                      "upper95_pct", "excess_pp", "gate_pp", "floor_pp",
                      "at_floor", "control_definition", "n_seeds_required",
                      "n_seeds", "seeds_sufficient", "verdict"}
    assert v["n"] == 8192 and v["n_seeds"] == 3
    assert v["recovery_pct"] == 5.0 and v["control_pct"] == 4.0
    assert abs(v["upper95_pct"] - 100.0
               * wilson_upper(round(0.05 * 8192), 8192)) < 1e-9


def test_self_test_passes():
    assert gate.self_test() == 0


def test_module_runs_standalone_torchless():
    import runpy
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.gate", run_name="__main__")
    assert exc.value.code == 0
