"""sharded (E-R5, block-diagonal sharded secrets + staggered rotation):
torch-free machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented sweep),
  * self_test() — the pure-python stagger-schedule fixtures,
  * the staggered schedule's security property itself (no 2-epoch window
    has all blocks on the same key),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * run()'s two arms with the numerical core scripted to known values:
    the static arm (negative control — sharding alone is no defense)
    reports the per-block mean; the staggered arm reports rotated blocks
    in the label-free band and unrotated blocks recovered, at block
    granularity, with rotated_blocks_in_window from the REAL schedule;
    solve failures journal structured "error:" records.
"""

import json
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import sharded as sh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_TOP1 = 88.0
ROTATED_TOP1 = 3.0
UNROTATED_TOP1 = 95.0


class _Mat:
    """Tensor stand-in: shape + provenance tag; every op the module
    performs returns another _Mat. The numeric payload lives in the
    scripted solve/recovery stubs."""

    def __init__(self, n, h=64, tag=None):
        self.shape = (n, h)
        self.tag = tag
        self.sets = []

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            if isinstance(cols, slice) and cols.start is not None:
                return _Mat(self.shape[0], self.shape[1],
                            ("block", cols.start))
            return _Mat(self.shape[0], self.shape[1], self.tag)
        if isinstance(key, slice):
            start = key.start or 0
            if start < 0:
                start += self.shape[0]
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
        return _Mat(1, self.shape[1], self.tag)

    def __setitem__(self, key, value):
        self.sets.append((key, value))

    def __matmul__(self, other):
        return _Mat(self.shape[0], getattr(other, "shape", (0, 64))[1],
                    self.tag)

    def __mul__(self, other):
        return _Mat(*self.shape, tag=self.tag)

    def clone(self):
        return _Mat(self.shape[0], self.shape[1], self.tag)

    def contiguous(self):
        return self

    def float(self):
        return self

    def double(self):
        return self


class _W:
    """solve_w result stand-in; w_hat[sl, sl] hands back a block-tagged
    factor so the staggered arm's per-block decode is attributable."""

    def __init__(self, tag):
        self.tag = tag

    def __getitem__(self, key):
        rows, cols = key
        return _W(("stag-block", cols.start))


class _FakeTorch:
    """Only the torch surface sharded.run touches."""

    class Generator:
        def manual_seed(self, seed):
            return self

    class linalg:
        @staticmethod
        def qr(m):
            return _Mat(*m.shape), _Mat(*m.shape)

    @staticmethod
    def randn(n, h, generator=None):
        return _Mat(n, h)

    @staticmethod
    def sign(m):
        return m

    @staticmethod
    def diagonal(m):
        return _Mat(1, m.shape[1])

    @staticmethod
    def cat(mats):
        return _Mat(sum(m.shape[0] for m in mats), mats[0].shape[1],
                    "cat")


def _world():
    return {"public_h": _Mat(4096, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object()}


def _patch_toy(monkeypatch, solve=None):
    """Run sh.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script the numerical core to known values."""
    monkeypatch.setattr(sh, "require_torch", lambda name: None)
    monkeypatch.setattr(sh, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world", lambda **kw: _world())
    monkeypatch.setattr(sh, "block_diag_secret",
                        lambda hidden, s, seed: _Mat(hidden, hidden, "W"))
    if solve is None:
        def solve(h, hw):
            if hw.tag == "cat":
                return _W("pooled"), "lstsq+svd"
            return _W("block-solve"), "lstsq+svd"
    monkeypatch.setattr(sh, "solve_w", solve)


def _patch_recovery(monkeypatch, rotated_block_starts):
    """Static per-block solves recover at STATIC_TOP1 (the negative
    control); staggered-arm decodes sit in the band exactly on the blocks
    the REAL schedule rotated in the window."""

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        if w_hat.tag == "block-solve":
            return STATIC_TOP1
        _, start = w_hat.tag
        return (ROTATED_TOP1 if start in rotated_block_starts
                else UNROTATED_TOP1)

    monkeypatch.setattr(sh, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="er5.json"):
    out = tmp_path / name
    args = sh.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "training"
    assert sh.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert sh.EXPERIMENT_ID == "er5_sharded"
    assert sh.MODES == ("training", "inference")
    assert sh.REQUIRES_LABELS is True
    assert "E-R5" in sh.DESCRIPTION
    assert validate_contract(sh, "sharded") is sh
    assert ATTACKS["sharded"] == "attacker.attacks.sharded"
    assert load("sharded") is sh
    md = attack_metadata.metadata_for("sharded")
    assert md["surfaces"] == ("S1",)
    assert md["persistence"] == "session"
    assert "LLM02" in md["owasp_llm_2026"]


def test_parser_defaults_are_the_documented_sweep():
    args = sh.build_parser().parse_args([])
    assert args.shards == [1, 4]
    assert args.pairs_per_block == 16
    assert args.rotation_period == 4
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]
    assert args.toy is False and args.quick is False


def test_parser_rejects_bad_shards_and_help_works():
    with pytest.raises(SystemExit) as exc:
        sh.build_parser().parse_args(["--shards", "many"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        sh.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_self_test_passes():
    assert sh.self_test() == 0


def test_staggered_schedule_never_rotates_all_blocks_in_one_window():
    """The defense claim, at the source: for s >= 2, no 2-epoch sliding
    window has all blocks on the same key, and every block rotates exactly
    once per period."""
    for s, period in ((2, 4), (4, 8), (4, 16), (8, 16)):
        got = {e: sh.stagger_rotating_blocks(e, s, period)
               for e in range(2 * period)}
        for e in range(2 * period - 1):
            window = set(got[e]) | set(got[e + 1])
            assert len(window) < s, (s, period, e, window)
        for b in range(s):
            assert sum(b in got[e] for e in range(period)) == 1


def test_run_fails_closed_without_torch(tmp_path):
    if sh.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er5.json"
    args = sh.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        sh.run(args)
    assert not out.exists()
    assert not (tmp_path / "er5.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if sh.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er5.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "sharded", "--toy", "--quick", "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[er5_sharded] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_non_toy_run_is_a_descriptive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "require_torch", lambda name: None)
    args = sh.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="real-run driver"):
        sh.run(args)
    assert not (tmp_path / "x.json").exists()


def test_static_arm_negative_control_and_staggered_mechanism(tmp_path,
                                                             monkeypatch):
    """s=4, period=4: block 3 rotates inside the window (real schedule),
    blocks 0-2 do not."""
    _patch_toy(monkeypatch)
    _patch_recovery(monkeypatch, rotated_block_starts={3 * 16})
    art, journal = _run(["--toy", "--quick", "--shards", "4"], tmp_path)
    assert art["schema"] == "dtraining.attacker.er5.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["shards"] == [4]
    static = [r for r in art["results"] if r["arm"] == "static"]
    stag = [r for r in art["results"] if r["arm"] == "staggered"]
    assert {r["seed"] for r in static} == {r["seed"] for r in stag} == {0, 1}
    for r in static:
        assert r["per_block_top1_mean"] == STATIC_TOP1
        assert r["pairs_per_block"] == 16 and r["shards"] == 4
    for r in stag:
        assert r["rotated_blocks_in_window"] == [3]
        assert r["per_block_top1"] == {"0": UNROTATED_TOP1,
                                       "1": UNROTATED_TOP1,
                                       "2": UNROTATED_TOP1,
                                       "3": ROTATED_TOP1}
        assert r["rotated_block_top1_mean"] == ROTATED_TOP1
        assert r["unrotated_block_top1_mean"] == UNROTATED_TOP1
        assert "label-free band" in r["note"]
    assert len(journal) == 4
    assert all(j["experiment"] == sh.EXPERIMENT_ID and "error" not in j
               for j in journal)


def test_single_shard_window_rotates_nothing(tmp_path, monkeypatch):
    """s=1: the whole key is one block; at epoch 1 nothing rotates, so the
    composed solve is scored on an unrotated-only window."""
    _patch_toy(monkeypatch)
    _patch_recovery(monkeypatch, rotated_block_starts=set())
    art, _ = _run(["--toy", "--quick", "--shards", "1"], tmp_path)
    stag = [r for r in art["results"] if r["arm"] == "staggered"]
    assert {r["seed"] for r in stag} == {0, 1}
    for r in stag:
        assert r["rotated_blocks_in_window"] == []
        assert r["per_block_top1"] == {"0": UNROTATED_TOP1}
        assert r["unrotated_block_top1_mean"] == UNROTATED_TOP1
        assert r["rotated_block_top1_mean"] == 0.0


def test_indivisible_shard_count_is_skipped(tmp_path, monkeypatch, capsys):
    _patch_toy(monkeypatch)
    _patch_recovery(monkeypatch, rotated_block_starts=set())
    art, _ = _run(["--toy", "--quick", "--shards", "1", "3"], tmp_path)
    assert {r["shards"] for r in art["results"]} == {1}
    assert "hidden=64 not divisible by s=3" in capsys.readouterr().out


def test_failed_solves_journal_structured_errors(tmp_path, monkeypatch):
    _patch_toy(monkeypatch,
               solve=lambda h, hw: (None, "error: gesdd non-convergence"))
    _patch_recovery(monkeypatch, rotated_block_starts={48})
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["results"] == []
    assert len(journal) == 2 * (2 + 5)
    for j in journal:
        assert j["experiment"] == sh.EXPERIMENT_ID
        assert j["error"].startswith("error:")
        assert "gesdd non-convergence" in j["error"]
        assert "Traceback" not in j["error"]
    stag_errs = [j for j in journal if j.get("arm") == "staggered"]
    block_errs = [j for j in journal if "block" in j]
    assert len(stag_errs) == 4 and len(block_errs) == 2 * (1 + 4)


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    _patch_recovery(monkeypatch, rotated_block_starts={48})
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["config"] == b["config"]
