"""Tests for the Study 2 CLM & SCR-Audit attack battery (clm-scr-audit)."""

from __future__ import annotations

import pytest

from attacker.attack_metadata import ATTACK_METADATA
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import clm_scr_audit


def test_registered_and_valid_contract() -> None:
    assert "clm-scr-audit" in ATTACKS
    assert "clm-scr-audit" in ATTACK_METADATA
    mod = load("clm-scr-audit")
    validate_contract(mod)
    assert mod.EXPERIMENT_ID == "clm_scr_audit"
    assert mod.MODES == ("inference",)


def test_all_18_attack_families_and_episode_counts() -> None:
    assert len(clm_scr_audit.ATTACK_FAMILIES) == 18
    eps = clm_scr_audit.build_all_episodes()
    adv = [e for e in eps if e.is_adversarial]
    ben = [e for e in eps if not e.is_adversarial]
    assert len(adv) == 96
    assert len(ben) == 36
    assert len({e.episode_id for e in eps}) == 132


def test_mla_rope_relocation_algebra_exact() -> None:
    linf = clm_scr_audit.verify_mla_rope_relocation_algebra()
    assert linf < 1e-12


def test_self_test_and_cli_modes(capsys: pytest.CaptureFixture[str]) -> None:
    res = clm_scr_audit.self_test()
    assert res["loopback_sandbox_verified"] is True
    assert res["attack_families"] == 18

    parser = clm_scr_audit.build_parser()
    assert clm_scr_audit.run(parser.parse_args(["--list-families"])) == 0
    out = capsys.readouterr().out
    assert "A7_split_payload_assembly" in out
    assert "A9_in_schema_param_sabotage" in out
    assert "G3_scr_rope_mla_joint_gcg" in out

    assert clm_scr_audit.run(parser.parse_args(["--self-test"])) == 0
