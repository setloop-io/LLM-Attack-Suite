"""Execution reports (attacker/report.py) + the --report CLI paths."""

import json
import os
import subprocess
import sys

from attacker.evidence import EvidenceLogger
from attacker.report import (CORPUS_CAVEAT, render_report, report_path_for,
                             write_report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_artifact(summary=None, results=None):
    return {"schema": "dtraining.attacker.pytest.v1",
            "config": {"attack": "pytest-attack", "mode": "training",
                       "budget": 64},
            "threat_model": "pytest threat model",
            "provenance": {"hostname": "pytest-host",
                           "timestamp_utc": "2026-01-01T00:00:00Z"},
            "results": results or [], "summary": summary or []}


def test_render_report_full_sections(tmp_path):
    art = make_artifact(summary=[{"experiment": "pytest_attack",
                                  "best_probe_top1_pct": 6.5,
                                  "upper95_excess_over_majority_pp": 2.5,
                                  "n_seeds": 3}])
    p = tmp_path / "run.json"
    p.write_text(json.dumps(art))
    md = render_report(str(p))
    assert "# Execution report — `pytest-attack`" in md
    assert "## Configuration" in md and "| `budget` | 64 |" in md
    assert "pytest threat model" in md
    assert "best_probe_top1_pct" in md and "+2.500 pp" in md
    assert "FAIL" in md and "frozen-gate.v2" in md and "z=2.5392" in md
    assert "degraded: no evidence manifest" in md
    assert CORPUS_CAVEAT in md


def test_render_report_sealed_evidence(tmp_path):
    art = make_artifact(summary=[{"experiment": "pytest_attack",
                                  "upper95_excess_over_majority_pp": 0.3,
                                  "n_seeds": 3}])
    p = tmp_path / "sealed.json"
    p.write_text(json.dumps(art))
    with EvidenceLogger(str(p), node_role="pytest/cpu") as ev:
        ev.event("cell_result", {"cell": 1})
    md = render_report(str(p))
    assert "OK — chain intact" in md
    assert "pytest/cpu" in md
    assert "(match)" in md
    assert "```bash" in md


def test_render_report_tampered_chain_reported(tmp_path):
    art = make_artifact(summary=[{"experiment": "pytest_attack",
                                  "upper95_excess_over_majority_pp": 0.3}])
    p = tmp_path / "t.json"
    p.write_text(json.dumps(art))
    with EvidenceLogger(str(p)):
        pass
    lines = (tmp_path / "t.json.evidence.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"] = {"tampered": True}
    lines[0] = json.dumps(rec)
    (tmp_path / "t.json.evidence.jsonl").write_text("\n".join(lines) + "\n")
    md = render_report(str(p))
    assert "FAILED" in md and "✗" in md


def test_render_report_never_raises(tmp_path):
    for bad in (tmp_path / "missing.json", str(tmp_path), __file__):
        md = render_report(str(bad))
        assert isinstance(md, str) and "degraded" in md


def test_write_report_path(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(make_artifact()))
    out = write_report(str(p))
    assert out == report_path_for(str(p)) == str(p) + ".report.md"
    assert os.path.isfile(out)


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "attacker", *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=300)


def test_cli_report_writes_and_prints(tmp_path):
    p = tmp_path / "cli.json"
    p.write_text(json.dumps(make_artifact(
        summary=[{"experiment": "x", "ica_top1_mean": 0.5}])))
    r = run_cli("--report", str(p))
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert os.path.isfile(str(p) + ".report.md")
    r = run_cli("--report", str(p), "--stdout")
    assert r.returncode == 0 and "# Execution report" in r.stdout


def test_cli_report_piggyback_on_attack(tmp_path):
    out = tmp_path / "gm.json"
    r = run_cli("--mode", "training", "--attack", "gpu-memory", "--toy",
                "--quick", "--arm", "scrub-check", "--output", str(out),
                "--report")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert os.path.isfile(str(out) + ".report.md")
    md = (tmp_path / "gm.json.report.md").read_text()
    assert "gpu_memory_residue" in md and "scrub" in md
