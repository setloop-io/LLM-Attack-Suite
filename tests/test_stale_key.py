"""stale-key (E-R4, ratcheted keys / stale-key attack): torch-free
machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented configuration),
  * self_test() — the pure-python one-way-ratchet fixtures,
  * the one-way property the whole attack relies on (stale keys + the
    derivation rule do not yield future keys),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * run()'s two arms with the numerical core scripted to known values:
    the zero-fresh-pairs arm scores every stale key in the label-free
    band with W rel_err ~= sqrt(2) and reports best_stale = the max; the
    fresh-pairs arm records the E-R1a-level recovery; fresh-pair solve
    failures journal structured "error:" records while the zero-fresh arm
    still lands in results.
"""

import json
import math
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import stale_key as sk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STALE_TOP1 = {1: 3.5, 2: 2.8, 3: 4.1}
STALE_REL_ERR = {1: 1.4142, 2: 1.4150, 3: 1.4136}
FRESH_TOP1 = {64: 38.0, 256: 61.0}


class _Idx:
    """randperm result: an opaque permutation handle carrying its length."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, sl):
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else self.n
        return _Idx(stop - start)


class _Mat:
    """Tensor stand-in: shape + provenance tag; every op the module
    performs returns another _Mat."""

    def __init__(self, n, h=64, tag=None):
        self.shape = (n, h)
        self.tag = tag

    def __getitem__(self, key):
        if isinstance(key, _Idx):
            return _Mat(key.n, self.shape[1], self.tag)
        if isinstance(key, slice):
            start = key.start or 0
            if start < 0:
                start += self.shape[0]
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
        return _Mat(1, self.shape[1], self.tag)

    def double(self):
        return self


class _W:
    """solve result stand-in: ("W", epoch) for a stale key handed to the
    decode directly, ("fresh", E) for a fresh-pairs solve."""

    def __init__(self, tag):
        self.tag = tag


class _FakeTorch:
    """Only the torch surface stale_key.run touches."""

    class Generator:
        def manual_seed(self, seed):
            return self

    @staticmethod
    def randperm(n, generator=None):
        return _Idx(n)


def _world(n_epochs=5):
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1], "wire")

    return {"n_epochs": n_epochs,
            "Ws": [_Mat(64, 64, ("W", t)) for t in range(n_epochs)],
            "public_h": _Mat(4096, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object(),
            "wire": wire}


def _patch_toy(monkeypatch, solve=None):
    """Run sk.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script the numerical core to known values."""
    monkeypatch.setattr(sk, "require_torch", lambda name: None)
    monkeypatch.setattr(sk, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world",
                        lambda **kw: _world())
    monkeypatch.setattr(
        sk, "w_rel_err",
        lambda a, b: STALE_REL_ERR[a.tag[1]])
    if solve is None:
        def solve(h, hw):
            return _W(("fresh", h.shape[0])), "lstsq+svd"
    monkeypatch.setattr(sk, "solve_w", solve)

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        kind, key = w_hat.tag
        return STALE_TOP1[key] if kind == "W" else FRESH_TOP1[key]

    monkeypatch.setattr(sk, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="er4.json"):
    out = tmp_path / name
    args = sk.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "training"
    assert sk.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert sk.EXPERIMENT_ID == "er4_stale_key"
    assert sk.MODES == ("training", "inference")
    assert sk.REQUIRES_LABELS == ("insider (stale keys) + optional fresh "
                                  "labeled pairs")
    assert "E-R4" in sk.DESCRIPTION
    assert validate_contract(sk, "stale-key") is sk
    assert ATTACKS["stale-key"] == "attacker.attacks.stale_key"
    assert load("stale-key") is sk
    md = attack_metadata.metadata_for("stale-key")
    assert md["actor_intent"] == "malicious"
    assert md["persistence"] == "cross-epoch"
    assert set(md["surfaces"]) == {"S1", "S10"}


def test_parser_defaults_are_the_documented_configuration():
    args = sk.build_parser().parse_args([])
    assert args.stale_epochs == 3 and args.target_epoch == 4
    assert args.fresh_pair_budgets == [64, 256]
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]
    assert args.toy is False and args.quick is False


def test_parser_rejects_bad_budget_and_help_works():
    with pytest.raises(SystemExit) as exc:
        sk.build_parser().parse_args(["--fresh-pair-budgets", "lots"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        sk.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert sk.self_test() == 0


def test_ratchet_is_one_way_so_stale_keys_cannot_bootstrapp():
    """The property E-R4 relies on: being HANDED the derivation rule plus
    stale keys W_1..W_k (equivalently their seeds) does not yield the next
    epoch's seed."""
    for master in (0, 1, 999, 12345):
        chain = [sk.ratchet_seed(master, t) for t in range(8)]
        assert len(set(chain)) == 8
        assert chain == [sk.ratchet_seed(master, t) for t in range(8)]
        for t in range(5):
            assert sk.ratchet_seed(chain[t], 1) != chain[t + 1]


def test_random_rel_err_reference_is_sqrt2():
    assert sk.RANDOM_REL_ERR == pytest.approx(math.sqrt(2.0))


def test_run_fails_closed_without_torch(tmp_path):
    if sk.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er4.json"
    args = sk.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        sk.run(args)
    assert not out.exists()
    assert not (tmp_path / "er4.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if sk.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er4.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "stale-key", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[er4_stale_key] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_non_toy_run_is_a_descriptive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sk, "require_torch", lambda name: None)
    args = sk.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="real-run driver"):
        sk.run(args)
    assert not (tmp_path / "x.json").exists()


def test_zero_fresh_arm_stale_keys_stay_in_the_band(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.er4.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["stale_epochs"] == 3
    assert art["config"]["target_epoch"] == 4
    zero = [r for r in art["results"] if r["arm"] == "zero_fresh_pairs"]
    assert {r["seed"] for r in zero} == {0, 1}
    for r in zero:
        assert r["target_epoch"] == 4
        assert r["random_rel_err_reference"] == pytest.approx(
            math.sqrt(2.0))
        assert [row["stale_epoch"] for row in r["stale_results"]] == [1, 2, 3]
        for row in r["stale_results"]:
            assert row["top1"] == STALE_TOP1[row["stale_epoch"]]
            assert row["w_rel_err"] == STALE_REL_ERR[row["stale_epoch"]]
        assert r["best_stale_top1"] == max(STALE_TOP1.values())
    assert len(journal) == len(art["results"])


def test_fresh_pairs_arm_matches_the_er1a_level(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    art, _ = _run(["--toy"], tmp_path)
    fresh = [r for r in art["results"] if r["arm"] == "fresh_pairs"]
    assert {(r["epoch_size"], r["seed"]) for r in fresh} == {
        (64, s) for s in (0, 1, 2)} | {(256, s) for s in (0, 1, 2)}
    for r in fresh:
        assert r["top1_mean"] == FRESH_TOP1[r["epoch_size"]]
        assert r["n_solves"] == 3
        assert r["target_epoch"] == 4
        assert "stale keys held but unused" in r["note"]


def test_failed_fresh_solves_journal_errors_and_keep_zero_arm(tmp_path,
                                                              monkeypatch):
    _patch_toy(monkeypatch,
               solve=lambda h, hw: (None, "error: epoch key solve failed"))
    art, journal = _run(["--toy", "--quick"], tmp_path)
    arms = {r["arm"] for r in art["results"]}
    assert arms == {"zero_fresh_pairs"}
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 2
    for j in errs:
        assert j["experiment"] == sk.EXPERIMENT_ID
        assert j["arm"] == "fresh_pairs" and j["epoch_size"] == 64
        assert j["error"].startswith("error:")
        assert "epoch key solve failed" in j["error"]
        assert "Traceback" not in j["error"]


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["config"] == b["config"]
