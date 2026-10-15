"""Tests for the test-reporting machinery (conftest_plugin + run_suite).

Runs a synthetic mini-suite through pytest in-process with the plugin and
asserts the report contract; the merger is tested on fabricated per-env
records (no real second venv is invoked).
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import run_suite
from conftest_plugin import render_markdown, tier_of


class _Item:
    def __init__(self, nodeid, markers=()):
        self.nodeid = nodeid
        self._markers = markers

    def iter_markers(self):
        class M:
            def __init__(self, name):
                self.name = name
        return iter([M(m) for m in self._markers])


def test_tier_classification():
    assert tier_of(_Item("tests/test_gate.py::test_x")) == "torch-free"
    assert tier_of(_Item("tests/test_splitbench.py::test_x",
                         ["slow"])) == "torch"
    assert tier_of(_Item("tests/test_defenses.py::test_run_scenarios",
                         ["slow"])) == "bdd"


def _record(tests):
    return {"schema": "llm_attacker.test_run.v1",
            "env": {"python": "3.14", "platform": "test",
                    "torch_present": False, "ts_utc": "2026-08-20T00:00:00Z"},
            "totals": {"tests": len(tests),
                       "passed": sum(t["outcome"] == "passed" for t in tests),
                       "failed": sum(t["outcome"] == "failed" for t in tests),
                       "skipped": sum(t["outcome"] == "skipped" for t in tests)},
            "tiers": {}, "verdict": "GREEN", "tests": tests}


def test_render_markdown_sections():
    rec = _record([
        {"nodeid": "t::a", "module": "t", "tier": "torch-free",
         "outcome": "passed", "duration_s": 0.1},
        {"nodeid": "t::b", "module": "t", "tier": "torch",
         "outcome": "skipped", "reason": "could not import 'torch'",
         "duration_s": 0.0},
        {"nodeid": "t::c", "module": "t", "tier": "torch-free",
         "outcome": "failed", "error": "assert 1 == 2", "duration_s": 0.2}])
    md = render_markdown(rec)
    assert "## Skipped (by reason)" in md and "could not import 'torch'" in md
    assert "## Failed" in md and "t::c" in md
    assert "## Slowest 10" in md


def test_plugin_writes_last_run(tmp_path):
    """A synthetic mini-suite through pytest with the plugin wired in."""
    suite = tmp_path / "mini"
    (suite / "tests").mkdir(parents=True)
    (suite / "conftest.py").write_text('pytest_plugins = ["conftest_plugin"]\n')
    (suite / "tests" / "test_mini.py").write_text(
        "import pytest\n"
        "def test_pass(): assert True\n"
        "def test_skip(): pytest.skip('demo reason')\n"
        "def test_fail(): assert 1 == 2\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "tests"))
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=suite, capture_output=True, text=True, env=env)
    assert r.returncode != 0
    rec = json.loads((suite / "tests" / "last_run.json").read_text())
    by_id = {t["nodeid"]: t for t in rec["tests"]}
    assert by_id["tests/test_mini.py::test_pass"]["outcome"] == "passed"
    assert by_id["tests/test_mini.py::test_fail"]["outcome"] == "failed"
    sk = by_id["tests/test_mini.py::test_skip"]
    assert sk["outcome"] == "skipped" and "demo reason" in sk["reason"]
    assert (suite / "tests" / "last_run.md").exists()


def test_merge_prefers_pass_and_names_gaps():
    env_a = ("torchfree", {"tests": [
        {"nodeid": "a::t", "outcome": "skipped", "tier": "torch",
         "module": "a", "duration_s": 0},
        {"nodeid": "b::t", "outcome": "passed", "tier": "torch-free",
         "module": "b", "duration_s": 0}]})
    env_b = ("torch", {"tests": [
        {"nodeid": "a::t", "outcome": "passed", "tier": "torch",
         "module": "a", "duration_s": 0},
        {"nodeid": "b::t", "outcome": "passed", "tier": "torch-free",
         "module": "b", "duration_s": 0},
        {"nodeid": "c::t", "outcome": "skipped", "tier": "bdd",
         "module": "c", "duration_s": 0}]})
    known = [{"nodeid": "c::t"}]
    merged = run_suite.merge([env_a, env_b], known)
    assert merged["best"]["a::t"]["outcome"] == "passed"
    assert merged["best"]["a::t"]["env"] == "torch"
    assert merged["never_ran"] == ["c::t"]
    assert merged["unacknowledged_gaps"] == []
    assert merged["acknowledged_gaps"] == ["c::t"]
    lines = run_suite.verdict_lines(merged, ["torchfree", "torch"])
    assert any("TDD" in l and "green" in l for l in lines)


def test_merge_flags_unacknowledged_gap():
    env = ("e", {"tests": [{"nodeid": "x::t", "outcome": "skipped",
                            "tier": "bdd", "module": "x", "duration_s": 0}]})
    merged = run_suite.merge([env], [])
    assert merged["unacknowledged_gaps"] == ["x::t"]
    lines = run_suite.verdict_lines(merged, ["e"])
    assert any("UNACKNOWLEDGED" in l for l in lines)


def test_known_gaps_file_is_valid_and_current():
    with open(os.path.join(ROOT, "tests", "known_gaps.json")) as f:
        gaps = json.load(f)["gaps"]
    assert len(gaps) >= 1 and all("nodeid" in g and "reason" in g
                                  for g in gaps)
