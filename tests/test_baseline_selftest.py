"""Regression tripwire: the torch-free baselines must stay green."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "attacker", *args],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=600)


def test_self_test_subprocess_exit_zero():
    r = run_cli("--self-test")
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "FRAMEWORK SELF-TEST PASSED" in r.stdout


def test_list_attacks_exit_zero():
    r = run_cli("--list-attacks")
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "latent-probe" in r.stdout


def test_scorecard_help_exit_zero():
    r = run_cli("--scorecard", "--help")
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])


def test_attack_help_torchless():
    r = run_cli("--mode", "training", "--attack", "latent-probe", "--help")
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
