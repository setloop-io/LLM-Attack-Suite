"""Scorecard aggregation over attack artifact JSONs."""

import json
import os
import subprocess
import sys

import pytest

from attacker.scorecard import (LEGACY_SCORECARD_POLICY, SCHEMA,
                                SCORECARD_POLICY, build_scorecard,
                                load_artifact, load_positive_control,
                                render_report, write_scorecard)
from attacker.gate import DEFAULT_GATE, GateConfig, verdict as gate_verdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_artifact(experiment, mode="training", results=None, summary=None):
    return {"schema": f"dtraining.attacker.{experiment}.v1",
            "config": {"attack": experiment, "mode": mode},
            "threat_model": "pytest", "provenance": {},
            "results": results or [], "summary": summary or []}


@pytest.fixture
def artifacts(tmp_path):
    specs = [
        make_artifact("latent_probe", results=[
            {"experiment": "latent_probe", "top1_pct": 4.5}],
            summary=[{"experiment": "latent_probe",
                      "best_probe_top1_pct": 4.5, "n_seeds": 3,
                      "upper95_excess_over_majority_pp": 0.638}]),
        make_artifact("some_future_attack", mode="inference", summary=[
            {"experiment": "some_future_attack",
             "upper95_excess_over_majority_pp": 1.5, "n_seeds": 1}]),
        make_artifact("ica_bss", results=[
            {"experiment": "ica_bss", "budget": 512,
             "error": "error: torch is required"}],
            summary=[{"experiment": "ica_bss", "ica_top1_mean": 0.5}]),
    ]
    paths = []
    for i, art in enumerate(specs):
        p = tmp_path / f"art{i}.json"
        p.write_text(json.dumps(art))
        paths.append(str(p))
    return paths


def test_build_scorecard_verdicts(artifacts):
    card = build_scorecard(artifacts, policy=LEGACY_SCORECARD_POLICY)
    assert card["schema"] == SCHEMA
    verdicts = {r["experiment"]: r["verdict"] for r in card["rows"]}
    assert verdicts == {"latent_probe": "PASS",
                        "some_future_attack": "FAIL",
                        "ica_bss": "INFO"}
    rows = {r["experiment"]: r for r in card["rows"]}
    assert rows["latent_probe"]["labels"] == "labeled"
    assert rows["latent_probe"]["mode"] == "training"
    assert rows["ica_bss"]["labels"] == "label-free"
    assert rows["some_future_attack"]["labels"] == "unknown"
    assert rows["some_future_attack"]["warnings"] == ["n_seeds=1 < required 3"]
    assert rows["latent_probe"]["headline_metric"] == \
        "upper95_excess_over_majority_pp"
    assert rows["ica_bss"]["headline_metric"] == "ica_top1_mean"


def test_worst_cell_and_not_shown(artifacts):
    card = build_scorecard(artifacts, policy=LEGACY_SCORECARD_POLICY)
    assert card["worst_cell"]["experiment"] == "some_future_attack"
    assert card["worst_cell"]["excess_pp"] == 1.5
    assert len(card["not_shown"]) == 1
    ns = card["not_shown"][0]
    assert ns["attack"] == "ica_bss"
    assert ns["error"] == "error: torch is required"
    assert ns["cell"] == {"experiment": "ica_bss", "budget": 512}
    s = card["summary"]
    assert (s["n_pass"], s["n_fail"], s["n_info"], s["n_not_shown"]) == \
        (1, 1, 1, 1)
    assert s["overall"] == "FAIL"


def test_render_report_sections(artifacts):
    report = render_report(build_scorecard(artifacts))
    for needle in ("ATTACKER SCORECARD", "z=2.5392", "latent-probe",
                   "some_future_attack", "WORST CELL", "NOT SHOWN",
                   "error: torch is required", "OVERALL: FAIL"):
        assert needle in report


def test_write_scorecard_json_and_txt(artifacts, tmp_path):
    out = str(tmp_path / "scorecard.json")
    card = build_scorecard(artifacts)
    write_scorecard(card, out)
    written = json.loads(open(out).read())
    assert written["schema"] == SCHEMA and written["summary"]["n_rows"] == 3
    txt = (tmp_path / "scorecard.txt").read_text()
    assert "OVERALL: FAIL" in txt
    assert not os.path.exists(out + ".tmp")


def test_load_artifact_rejects_drift(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"results": []}))
    with pytest.raises(ValueError, match="missing"):
        load_artifact(str(bad))


def test_overall_pass_and_info_only(artifacts):
    assert build_scorecard([artifacts[0], artifacts[2]])[
        "summary"]["overall"] == "INCOMPLETE"
    assert build_scorecard([artifacts[2]])["summary"]["overall"] == "INCOMPLETE"
    assert build_scorecard([artifacts[2]])["worst_cell"] is None


def test_empty_summary_yields_single_info_row(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(make_artifact("ica_bss")))
    card = build_scorecard([str(p)])
    assert len(card["rows"]) == 1
    row = card["rows"][0]
    assert row["verdict"] == "INFO" and row["headline_metric"] is None


def test_cli_end_to_end(artifacts, tmp_path):
    out = str(tmp_path / "sc.json")
    r = subprocess.run([sys.executable, "-m", "attacker", "--scorecard",
                        *artifacts, "--output", out],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "OVERALL: FAIL" in r.stdout
    assert json.loads(open(out).read())["summary"]["overall"] == "FAIL"

    r = subprocess.run([sys.executable, "-m", "attacker", "--scorecard",
                        artifacts[0], artifacts[2]],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "OVERALL: INCOMPLETE" in r.stdout


def gated_artifact(tmp_path, recovery=0.10, control=0.04, n=8192, n_seeds=3,
                   config=DEFAULT_GATE, drop_stats=False):
    g = gate_verdict(recovery, control, n, config=config, n_seeds=n_seeds)
    if drop_stats:
        for k in ("recovery_pct", "control_pct", "n"):
            g.pop(k)
    cell = {"experiment": "vmi", "n_seeds": n_seeds, "gate": g}
    p = tmp_path / "gated.json"
    p.write_text(json.dumps(make_artifact("vmi", results=[cell])))
    return str(p), g


def bare_artifact(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(make_artifact("latent_probe", summary=[
        {"experiment": "latent_probe", "n_seeds": 3,
         "upper95_excess_over_majority_pp": 0.638}])))
    return str(p)


def test_rescore_identical_gate_reproduces_verdict(tmp_path):
    path, g = gated_artifact(tmp_path)
    card = build_scorecard([path], gate=DEFAULT_GATE)
    row = card["rows"][0]
    assert row["verdict"] == "FAIL"
    assert row["excess_pp"] == g["excess_pp"]
    assert row["warnings"] == []


def test_rescore_different_alpha_recomputes(tmp_path):
    path, g = gated_artifact(tmp_path)
    strict = GateConfig(alpha=0.01, version="audit-gate")
    row = build_scorecard([path], gate=strict)["rows"][0]
    expected = gate_verdict(0.10, 0.04, 8192, config=strict)
    assert row["excess_pp"] == pytest.approx(expected["excess_pp"])
    assert row["excess_pp"] != g["excess_pp"]
    assert any("recomputed" in w for w in row["warnings"])


def test_rescore_different_multiplicity_recomputes(tmp_path):
    path, _ = gated_artifact(tmp_path)
    wide = GateConfig(n_trials=27, version="audit-gate")
    row = build_scorecard([path], gate=wide)["rows"][0]
    expected = gate_verdict(0.10, 0.04, 8192, config=wide)
    assert row["excess_pp"] == pytest.approx(expected["excess_pp"])


def test_rescore_incompatible_bare_bound_refused(tmp_path):
    strict = GateConfig(alpha=0.01, version="audit-gate")
    with pytest.raises(ValueError, match="refusing to re-label"):
        build_scorecard([bare_artifact(tmp_path)], gate=strict)


def test_rescore_gate_cell_without_statistics_refused(tmp_path):
    path, _ = gated_artifact(tmp_path, drop_stats=True)
    strict = GateConfig(alpha=0.01, version="audit-gate")
    with pytest.raises(ValueError, match="refusing to re-label"):
        build_scorecard([path], gate=strict)


def test_rescore_control_mismatch_refused(tmp_path):
    path, _ = gated_artifact(tmp_path)
    other = GateConfig(control="uniform_random", version="audit-gate")
    with pytest.raises(ValueError, match="control"):
        build_scorecard([path], gate=other)


def test_rescore_cross_version_v1_v2(tmp_path):
    path, g = gated_artifact(tmp_path)
    v1 = GateConfig(version="frozen-gate.v1")
    card = build_scorecard([path], gate=v1)
    assert card["gate"]["version"] == "frozen-gate.v1"
    row = card["rows"][0]
    assert row["excess_pp"] == g["excess_pp"]
    assert row["verdict"] == "FAIL"
    v1_wide = GateConfig(n_trials=3, version="frozen-gate.v1")
    row = build_scorecard([path], gate=v1_wide)["rows"][0]
    expected = gate_verdict(0.10, 0.04, 8192, config=v1_wide)
    assert row["excess_pp"] == pytest.approx(expected["excess_pp"])
    assert any("recomputed" in w for w in row["warnings"])
    with pytest.raises(ValueError, match="refusing to re-label"):
        build_scorecard([bare_artifact(tmp_path)], gate=v1_wide)


def seam_artifact(tmp_path, n_seeds):
    g = gate_verdict(0.04, 0.04, 8192, n_seeds=n_seeds)
    cell = {"experiment": "seam_inversion", "arm": "vmi",
            "submode": "vmi-linearization", "n_tokens": 8192,
            "seq_accuracy": 0.04, "control_accuracy": 0.04, "gate": g,
            "gate_verdicts": [g["verdict"]]}
    p = tmp_path / "seam.json"
    p.write_text(json.dumps(make_artifact("seam_inversion",
                                          summary=[cell])))
    return str(p)


def test_nested_gate_n_seeds_allows_pass(tmp_path):
    card = build_scorecard([seam_artifact(tmp_path, n_seeds=3)],
                           policy=LEGACY_SCORECARD_POLICY)
    row = card["rows"][0]
    assert row["verdict"] == "PASS"
    assert row["warnings"] == []


def test_missing_n_seeds_downgrades_pass(tmp_path):
    card = build_scorecard([seam_artifact(tmp_path, n_seeds=None)])
    row = card["rows"][0]
    assert row["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert row["warnings"] == ["n_seeds=None < required 3"]


def pass_artifact(tmp_path, experiment="lira_membership", name="defended.json",
                  link=None):
    g = gate_verdict(0.04, 0.04, 8192, n_seeds=3)
    assert g["verdict"] == "PASS"
    cell = {"experiment": experiment, "n_seeds": 3, "gate": g}
    if link is not None:
        cell["positive_control"] = link
    p = tmp_path / name
    p.write_text(json.dumps(make_artifact(experiment, summary=[cell])))
    return str(p)


def control_artifact(tmp_path, experiment="lira_membership", recovery=0.20,
                     control=0.04, name="control.json"):
    g = gate_verdict(recovery, control, 8192, n_seeds=3)
    cell = {"experiment": experiment, "n_seeds": 3, "gate": g}
    p = tmp_path / name
    p.write_text(json.dumps(make_artifact(
        experiment, results=[cell],
        summary=[{"experiment": experiment, "gate_verdicts": [g["verdict"]],
                  "n_cells": 1}])))
    return str(p), g


def test_v3_pass_with_backing_control_stays_pass(tmp_path):
    defended = pass_artifact(tmp_path)
    control, g = control_artifact(tmp_path)
    card = build_scorecard([defended], positive_controls=[control])
    assert card["policy"] == SCORECARD_POLICY
    row = card["rows"][0]
    assert row["verdict"] == "PASS"
    assert row["positive_control"] == {
        "status": "backed", "by": control,
        "control_max_excess_pp": g["excess_pp"]}
    assert g["excess_pp"] > DEFAULT_GATE.gate_pp
    rec, = card["positive_controls"]
    assert rec["competent"] is True and rec["origin"] == "report"
    assert rec["max_excess_pp"] == pytest.approx(g["excess_pp"])
    assert card["summary"]["overall"] == "PASS"


def test_v3_pass_without_control_downgrades(tmp_path):
    card = build_scorecard([pass_artifact(tmp_path)])
    row = card["rows"][0]
    assert row["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert row["positive_control"]["status"] == "unbacked"
    assert len(row["warnings"]) == 1
    assert "no positive control linked" in row["warnings"][0]
    assert SCORECARD_POLICY in row["warnings"][0]
    assert card["summary"]["overall"] == "INCOMPLETE"
    assert card["positive_controls"] == []


def test_v3_control_below_threshold_does_not_back(tmp_path):
    control, _ = control_artifact(tmp_path, recovery=0.043)
    card = build_scorecard([pass_artifact(tmp_path)],
                           positive_controls=[control])
    row = card["rows"][0]
    assert row["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert card["positive_controls"][0]["competent"] is False
    assert "demonstrates attack competence" in row["warnings"][0]
    assert card["summary"]["overall"] == "INCOMPLETE"


def test_v3_family_mismatch_does_not_back(tmp_path):
    control, _ = control_artifact(tmp_path, experiment="lira_membership")
    defended = pass_artifact(tmp_path, experiment="seam_inversion")
    card = build_scorecard([defended], positive_controls=[control])
    row = card["rows"][0]
    assert row["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "no positive control linked" in row["warnings"][0]


def test_v3_explicit_cell_link_backs_across_families(tmp_path):
    control, _ = control_artifact(tmp_path, experiment="other_family")
    defended = pass_artifact(tmp_path, experiment="seam_inversion",
                             link="control.json")
    card = build_scorecard([defended])
    row = card["rows"][0]
    assert row["verdict"] == "PASS"
    assert row["positive_control"]["status"] == "backed"
    rec, = card["positive_controls"]
    assert rec["origin"] == "artifact"


def test_v3_unreadable_control_fails_closed(tmp_path):
    defended = pass_artifact(tmp_path, link="no-such-control.json")
    card = build_scorecard([defended])
    row = card["rows"][0]
    assert row["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "unreadable" in row["warnings"][0]


def test_legacy_policy_preserves_v2_behavior(tmp_path):
    defended = pass_artifact(tmp_path)
    control, _ = control_artifact(tmp_path)
    card = build_scorecard([defended], policy=LEGACY_SCORECARD_POLICY)
    assert card["policy"] == LEGACY_SCORECARD_POLICY
    assert card["rows"][0]["verdict"] == "PASS"
    assert "positive_control" not in card["rows"][0]
    assert card["positive_controls"] == []
    card = build_scorecard([defended], positive_controls=[control],
                           policy=LEGACY_SCORECARD_POLICY)
    assert card["rows"][0]["verdict"] == "PASS"
    assert card["positive_controls"] == []


def test_unknown_policy_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown scorecard policy"):
        build_scorecard([pass_artifact(tmp_path)],
                        policy="scorecard-policy.v9")


def test_load_positive_control_assesses_competence(tmp_path):
    control, g = control_artifact(tmp_path)
    rec = load_positive_control(control)
    assert rec["competent"] is True and rec["detail"] is None
    assert rec["attacks"] == ["lira_membership"]
    assert rec["max_excess_pp"] == pytest.approx(g["excess_pp"])
    weak, _ = control_artifact(tmp_path, recovery=0.043, name="weak.json")
    rec = load_positive_control(weak)
    assert rec["competent"] is False and "does not exceed" in rec["detail"]


def test_v3_cli_flag_and_policy_flag(tmp_path):
    defended = pass_artifact(tmp_path)
    control, _ = control_artifact(tmp_path)

    def run(*args):
        return subprocess.run([sys.executable, "-m", "attacker",
                               "--scorecard", *args],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=120)

    r = run(defended)
    assert r.returncode == 2 and "OVERALL: INCOMPLETE" in r.stdout
    assert "POSITIVE CONTROLS" in r.stdout
    r = run(defended, "--positive-control", control)
    assert r.returncode == 0 and "OVERALL: PASS" in r.stdout
    assert "backs PASS" in r.stdout
    r = run(defended, "--scorecard-policy", "scorecard-policy.v2")
    assert r.returncode == 0 and "OVERALL: PASS" in r.stdout


BATTERY_CONTROL = os.path.join(
    ROOT, "results", "attacker-positive-controls-v1", "membership",
    "lira-gpt2-auc.json")


@pytest.mark.skipif(not os.path.isfile(BATTERY_CONTROL),
                    reason="positive-control battery not present")
def test_v3_real_battery_control_backs_membership_pass(tmp_path):
    card = build_scorecard([pass_artifact(tmp_path)],
                           positive_controls=[BATTERY_CONTROL])
    row = card["rows"][0]
    assert row["verdict"] == "PASS"
    assert row["positive_control"]["by"] == BATTERY_CONTROL
    rec, = card["positive_controls"]
    assert rec["competent"] is True
    assert rec["max_excess_pp"] > DEFAULT_GATE.gate_pp
