"""Standard environment benchmark (attacker/benchmark.py): spec integrity,
the pure grade() mapping, the driver on a fake runner, and the CLI paths."""

import json
import os
import subprocess
import sys

import pytest

from attacker.benchmark import (AXES, BATTERY_ATTACKS, BATTERY_DEFENSES,
                                BATTERY_VERSION, CELL_INDEX_NAME,
                                REPORTCARD_SCHEMA, TargetConfig, _cell,
                                battery_spec, grade, render_reportcard_md,
                                regrade, run_battery)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_battery_spec_is_versioned_and_consistent():
    spec = battery_spec()
    assert spec["battery_version"] == BATTERY_VERSION == "benchmark.v1"
    json.dumps(spec)
    from attacker.attacks import ATTACKS
    from attacker.splitbench import _DEFENSES as SB_DEFENSES
    assert all(a.attack in ATTACKS and a.axis in AXES
               for a in BATTERY_ATTACKS)
    assert all(d.name in SB_DEFENSES for d in BATTERY_DEFENSES)
    assert [d.name for d in BATTERY_DEFENSES if d.quick] == ["none",
                                                             "mid-lite"]
    assert [a.attack for a in BATTERY_ATTACKS if a.quick] == [
        "seam-inversion", "label-scoring", "gpu-memory"]
    env = [a for a in BATTERY_ATTACKS if not a.per_defense]
    assert [a.attack for a in env] == ["gpu-memory"]
    assert "--toy" not in env[0].args


def test_target_config_validation():
    assert TargetConfig().validate().model == "gpt2"
    for kw in ({"mode": "inference"}, {"model": ""}, {"device": "tpu"},
               {"dtype": "fp16"}, {"seq_len": 1}, {"split": "0"},
               {"property_token": -1}, {"lr": 0.0}):
        with pytest.raises(ValueError):
            TargetConfig(**kw).validate()


def test_target_config_from_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"model": "gpt2", "split": "4"}))
    assert TargetConfig.from_json(str(p)).split == "4"
    p.write_text(json.dumps({"modle": "gpt2"}))
    with pytest.raises(ValueError, match="modle"):
        TargetConfig.from_json(str(p))


def acell(axis, defense, verdict, excess=None, status="ok"):
    c = _cell(f"{axis}:{defense}:x", "attack", axis, defense, "x", "arm")
    c["status"] = status
    c["verdict"] = verdict
    c["excess_pp"] = excess
    return c


def capcell(defense, loss):
    c = _cell(f"capture:{defense}", "capture", None, defense, "splitbench",
              "finetune")
    c["status"] = "ok"
    c["final_loss"] = loss
    return c


def test_grade_open_at_risk_closed_unmeasured():
    g = grade([acell("S1", "none", "FAIL", 86.0),
               acell("S1", "mid-lite", "FAIL", 2.08)])
    assert g["axes"]["S1"]["grade"] == "OPEN" and g["overall"] == "OPEN"

    g = grade([acell("S1", "none", "FAIL", 86.0),
               acell("S1", "mid-lite", "PASS", 0.29)])
    assert g["axes"]["S1"]["grade"] == "AT-RISK"
    assert g["axes"]["S1"]["best_defense"]["defense"] == "mid-lite"
    assert g["axes"]["S1"]["worst_cell"]["excess_pp"] == 86.0

    g = grade([acell("S1", "none", "PASS", 0.4),
               acell("S1", "mid-lite", "PASS", 0.3)])
    assert g["axes"]["S1"]["grade"] == "CLOSED" and g["overall"] == "INCOMPLETE"

    g = grade([acell("S1", "none", None, status="error"),
               acell("S5", None, "INFO")])
    assert g["axes"]["S1"]["grade"] == "UNMEASURED"
    assert g["overall"] == "UNMEASURED"


def test_grade_overall_is_worst_measured_and_utility():
    g = grade([acell("S1", "none", "PASS", 0.1),
               acell("S2", "none", "FAIL", 5.0),
               acell("S5", None, None, status="error"),
               capcell("none", 4.0), capcell("mid-lite", 7.6)])
    assert g["overall"] == "OPEN"
    assert g["axes"]["S5"]["grade"] == "UNMEASURED"
    assert g["utility"]["mid-lite"]["final_loss"] == 7.6
    card_md = render_reportcard_md(g)
    assert "# Environment report card" in card_md
    assert "## Utility (defense cost)" in card_md
    assert "Corpus caveat" in card_md


def fake_artifact(summary):
    return {"schema": "dtraining.attacker.fake.v1",
            "config": {"attack": "fake", "mode": "training"},
            "threat_model": "fake", "provenance": {},
            "results": [dict(summary[0])], "summary": summary}


def make_fake_runner(fail_gpu=True):
    def runner(cmd, timeout):
        out = cmd[cmd.index("--output") + 1]
        if "gpu-memory" in cmd and fail_gpu:
            return 1, "error: no hardware probe tier available"
        if "label-scoring" in cmd and "mid-lite" in out:
            summary = [{"experiment": "norm_based_scoring", "auc": 0.51,
                        "upper95_excess_over_majority_pp": 0.3,
                        "n_seeds": 3}]
        elif "label-scoring" in cmd:
            summary = [{"experiment": "norm_based_scoring", "auc": 0.83,
                        "upper95_excess_over_majority_pp": 33.0,
                        "n_seeds": 3}]
        elif "attacker.splitbench" in cmd:
            summary = [{"captures_written": 4, "final_loss": 7.6,
                        "mean_loss": 8.1}]
        else:
            summary = [{"experiment": "vmi",
                        "upper95_excess_over_majority_pp": 55.0,
                        "n_seeds": 3}]
        with open(out, "w") as f:
            json.dump(fake_artifact(summary), f)
        return 0, "ok"
    return runner


def test_run_battery_quick_fake_runner(tmp_path):
    rc = run_battery("quick", TargetConfig(model="gpt2"), str(tmp_path),
                     runner=make_fake_runner())
    assert rc == 0
    card = json.loads((tmp_path / "reportcard.json").read_text())
    assert card["schema"] == REPORTCARD_SCHEMA
    assert card["battery_version"] == "benchmark.v1"
    assert card["tier"] == "quick"
    assert card["axes"]["S1"]["grade"] == "OPEN"
    assert card["axes"]["S9"]["grade"] == "AT-RISK"
    assert card["axes"]["S5"]["grade"] == "UNMEASURED"
    assert card["axes"]["S2"]["grade"] == "UNMEASURED"
    assert card["overall"] == "OPEN"
    s5 = card["axes"]["S5"]["cells"][0]
    assert s5["status"] == "error" and "no hardware probe tier" in s5["error"]
    assert card["utility"]["none"]["final_loss"] == 7.6
    for name in (CELL_INDEX_NAME, "scorecard.json", "scorecard.txt",
                 "reportcard.md"):
        assert (tmp_path / name).is_file(), name
    sc = json.loads((tmp_path / "scorecard.json").read_text())
    assert any(ns["attack"] == "gpu-memory" for ns in sc["not_shown"])


def test_run_battery_never_raises_on_dead_environment(tmp_path):
    rc = run_battery("quick", TargetConfig(model="gpt2"), str(tmp_path),
                     runner=lambda cmd, timeout: (1, "error: boom"))
    assert rc == 0
    card = json.loads((tmp_path / "reportcard.json").read_text())
    assert card["overall"] == "UNMEASURED"
    index = json.loads((tmp_path / CELL_INDEX_NAME).read_text())
    assert all(c["status"] == "error" for c in index["cells"])
    assert any("skipped: capture cell failed" in (c["error"] or "")
               for c in index["cells"])


def test_regrade_round_trip(tmp_path):
    run_battery("quick", TargetConfig(model="gpt2"), str(tmp_path),
                runner=make_fake_runner())
    assert regrade(str(tmp_path)) == 0
    card = json.loads((tmp_path / "reportcard.json").read_text())
    assert card["overall"] == "OPEN"


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "attacker", *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=300)


def test_cli_benchmark_help():
    r = run_cli("--benchmark", "--help")
    assert r.returncode == 0 and "quick" in r.stdout and "full" in r.stdout


def test_cli_benchmark_bad_tier_and_config(tmp_path):
    r = run_cli("--benchmark", "quick", "--target-config",
                str(tmp_path / "missing.json"), "--workdir",
                str(tmp_path / "w"))
    assert r.returncode != 0
    assert "benchmark" in (r.stderr + r.stdout)


def test_cli_benchmark_grade(tmp_path):
    run_battery("quick", TargetConfig(model="gpt2"), str(tmp_path),
                runner=make_fake_runner())
    os.remove(str(tmp_path / "reportcard.json"))
    r = run_cli("--benchmark-grade", str(tmp_path))
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "overall OPEN" in r.stdout
    card = json.loads((tmp_path / "reportcard.json").read_text())
    assert card["overall"] == "OPEN"
