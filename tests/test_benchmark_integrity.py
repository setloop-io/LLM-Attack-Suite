"""benchmark.split-integrity-availability.v1 (attacker/benchmark_integrity.py):
the 88->113 normalization, Aggregation-B grading, the cell-for-cell match
against the acceptance oracle (tests/data/integrity_expected.json, generated
from calibration Appendix A at the locked 1.5x), regrade, and the error paths.

The oracle match is graded from the committed full-tier summaries in
tests/data/integrity_artifacts/ — the real measured values, not stubs."""

import glob
import importlib.util
import json
import os

import pytest

from attacker import benchmark_integrity as b, integrity_gate as ig

HERE = os.path.dirname(__file__)
ARTS = sorted(glob.glob(os.path.join(HERE, "data", "integrity_artifacts",
                                     "*.json")))
FIXTURE = os.path.join(HERE, "data", "integrity_expected.json")
FIELDS = ("axes", "role", "scale", "integrity_threshold", "verdict")


def _card():
    card, _ = b.grade_from_artifacts(ARTS, ig.DEFAULT_INTEGRITY_GATE,
                                     tier="full")
    return card


def _by_key(cells):
    return {(c["experiment"], c["arm"], c["world"], c["role"]):
            {k: c[k] for k in FIELDS} for c in cells}


def test_card_matches_oracle_cell_for_cell():
    oracle = json.load(open(FIXTURE))
    exp = _by_key(oracle["cells"])
    got = _by_key(_card()["cells"])
    assert set(got) == set(exp), \
        f"key mismatch: {set(got) ^ set(exp)}"
    diffs = {k: (exp[k], got[k]) for k in exp if exp[k] != got[k]}
    assert not diffs, f"field diffs: {diffs}"


def test_oracle_tally_is_51_47_15():
    card = _card()
    assert card["cell_tally"] == {"OPEN": 51, "CLOSED": 47, "UNMEASURED": 15}


def test_fixture_is_regenerated_from_calibration_csv_without_drift():
    """The committed oracle must equal what the generator produces from the
    retained calibration CSV — source and oracle cannot drift."""
    path = os.path.join(HERE, "data", "gen_integrity_expected.py")
    spec = importlib.util.spec_from_file_location("gen_integrity_expected",
                                                  path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    fresh = gen.build()
    committed = json.load(open(FIXTURE))
    assert fresh["cells"] == committed["cells"]
    assert fresh["tally"] == {"OPEN": 51, "CLOSED": 47, "UNMEASURED": 15}


def test_all_five_axes_at_risk_with_counts_and_named_open_cells():
    card = _card()
    assert card["overall"] == "AT-RISK"
    for axis, a in card["axes"].items():
        assert a["grade"] == "AT-RISK", axis
        assert a["n_defense"] == a["n_open"] + a["n_closed"] + a["n_unmeasured"]
        assert len(a["open_defense_cells"]) == a["n_open"] and a["n_open"] > 0


def test_retry_storm_flip_leaves_llm06_open_list():
    card = _card()
    llm06 = card["axes"]["LLM06"]
    assert (llm06["n_open"], llm06["n_closed"]) == (8, 3)
    assert not any("retry-storm" in n for n in llm06["open_defense_cells"])


def test_role_assertions_hold_no_baseline_closed_no_control_open():
    assert _card()["assertions"] == {"ok": True, "violations": []}


def test_every_row_accounted_for_88_rows_113_cells():
    card = _card()
    assert card["n_summary_rows"] == 88 and card["n_cells"] == 113


def _art(exp, summary, results=None):
    return {"schema": f"dtraining.attacker.{exp}.v1",
            "config": {"attack": exp}, "threat_model": "t", "provenance": {},
            "results": results or [], "summary": summary}


def test_duplicate_normalized_cell_is_a_fail_closed_error(tmp_path):
    dup = _art("protocol_integrity",
               [{"experiment": "protocol_integrity", "arm": "drop",
                 "world": "defended", "undetected_mutation_rate_mean": 1.0,
                 "n_seeds": 3},
                {"experiment": "protocol_integrity", "arm": "drop",
                 "world": "defended", "undetected_mutation_rate_mean": 0.0,
                 "n_seeds": 3}])
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate"):
        b.grade_from_artifacts([str(p)], ig.DEFAULT_INTEGRITY_GATE)


def test_wrong_attack_yields_unmeasured_not_a_grade(tmp_path):
    wrong = _art("some_other_experiment",
                 [{"experiment": "some_other_experiment", "arm": "x",
                   "world": "y", "metric": 1.0, "n_seeds": 3}])
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps(wrong))
    card, _ = b.grade_from_artifacts([str(p)], ig.DEFAULT_INTEGRITY_GATE)
    assert all(c["verdict"] == "UNMEASURED" for c in card["cells"])


def test_malformed_artifact_is_recorded_not_raised(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"results": []}))
    card, _ = b.grade_from_artifacts([str(p)], ig.DEFAULT_INTEGRITY_GATE)
    assert card["errors"] and "missing" in card["errors"][0]["error"]


def test_malicious_server_toy_is_unmeasured():
    card = _card()
    ms = [c for c in card["cells"]
          if c["experiment"] == "malicious_server_steering"]
    assert len(ms) == 3 and all(c["verdict"] == "UNMEASURED" for c in ms)
    assert not any(c["verdict"] == "OPEN" for c in ms)


def test_regrade_reproduces_card(tmp_path):
    card, index = b.grade_from_artifacts(ARTS, ig.DEFAULT_INTEGRITY_GATE,
                                         tier="full")
    b._write_outputs(str(tmp_path), card, index)
    assert b.regrade_integrity(str(tmp_path)) == 0
    re = json.load(open(tmp_path / "integrity_reportcard.json"))
    assert re["cell_tally"] == card["cell_tally"]
    assert _by_key(re["cells"]) == _by_key(card["cells"])


def test_regrade_catches_hand_edited_index_threshold(tmp_path):
    card, index = b.grade_from_artifacts(ARTS, ig.DEFAULT_INTEGRITY_GATE,
                                         tier="full")
    b._write_outputs(str(tmp_path), card, index)
    idx = json.load(open(tmp_path / b.CELL_INDEX_NAME))
    idx["gate_config"]["rules"][0]["threshold"] = 999.0
    (tmp_path / b.CELL_INDEX_NAME).write_text(json.dumps(idx))
    with pytest.raises(ValueError, match="tamper"):
        b.regrade_integrity(str(tmp_path))


def test_regrade_refuses_a_benchmark_v1_workdir(tmp_path):
    with pytest.raises(ValueError, match="not a"):
        b.regrade_integrity(str(tmp_path))
    (tmp_path / b.CELL_INDEX_NAME).write_text(json.dumps(
        {"battery_version": "benchmark.v1"}))
    with pytest.raises(ValueError, match="refusing to regrade"):
        b.regrade_integrity(str(tmp_path))


def test_scored_gate_is_the_card_gate(monkeypatch):
    seen = {}
    orig = ig.score_cell

    def spy(c, siblings, gate):
        seen["gate"] = gate
        return orig(c, siblings, gate)
    monkeypatch.setattr(ig, "score_cell", spy)
    g = ig.DEFAULT_INTEGRITY_GATE
    card, _ = b.grade_from_artifacts(ARTS, g, tier="full")
    assert seen["gate"] is g
    assert card["gate"]["digest"] == ig.gate_digest(g)


def test_capture_free_no_target_config():
    import inspect
    assert "--target-config" not in inspect.getsource(b.main)
    assert _card()["target"] is None
    assert "TargetConfig" not in inspect.getsource(b.regrade_integrity)


def test_run_battery_with_fake_runner(tmp_path):
    def fake_runner(cmd, timeout):
        out = cmd[cmd.index("--output") + 1]
        exp = os.path.basename(out)[:-5]
        summ = ([{"experiment": exp, "arm": "long-context",
                  "amplification_vs_control_mean": 2.5, "n_seeds": 3}]
                if exp == "resource_exhaustion" else [])
        with open(out, "w") as f:
            json.dump(_art(exp, summ), f)
        return 0, "ok"
    rc = b.run_integrity_battery("quick", str(tmp_path), runner=fake_runner)
    card = json.load(open(tmp_path / "integrity_reportcard.json"))
    assert rc == 0 and card["battery_version"] == b.BATTERY_VERSION
    for name in (b.CELL_INDEX_NAME, "integrity_reportcard.json",
                 "integrity_reportcard.md"):
        assert (tmp_path / name).is_file()


def test_failed_attack_is_journaled_not_raised(tmp_path):
    def dead_runner(cmd, timeout):
        return 1, "error: boom"
    rc = b.run_integrity_battery("quick", str(tmp_path), runner=dead_runner)
    card = json.load(open(tmp_path / "integrity_reportcard.json"))
    assert rc == 0 and card["errors"]


def test_render_markdown_carries_grades_and_named_cells():
    md = b.render_integrity_md(_card())
    assert "Split integrity/availability report card" in md
    assert "AT-RISK" in md and "OPEN defense cells" in md


def test_self_test_passes():
    assert b.self_test() == 0
