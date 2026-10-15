"""subspace-joint (cross-epoch joint attacker over the stable low-dim
subspace): torch-free machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented sweep),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * joint_solve directly with a stand-in torch: the alternating
    minimization returns one lifted W per epoch plus diagnostics, and an
    lstsq failure comes back as (None, {"error": "error: ..."}) — never a
    raise out of the solve,
  * run()'s head-to-head wiring with the numerical core scripted to known
    values: each (k, E, seed) cell records joint vs independent recovery
    at the same budget, the summary averages per seed, and joint-solve
    failures (raised OR returned) journal structured error records.
"""

import json
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import subspace_joint as sj

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOINT_TOP1 = 72.0
IND_TOP1 = 8.0


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
    performs returns another _Mat so joint_solve/run dataflow executes."""

    def __init__(self, n, h=64, tag=None):
        self.shape = (n, h)
        self.tag = tag

    def __getitem__(self, key):
        if isinstance(key, _Idx):
            return _Mat(key.n, self.shape[1], self.tag)
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
        return _Mat(1, self.shape[1], self.tag)

    @property
    def T(self):
        return _Mat(self.shape[1], self.shape[0], self.tag)

    def __sub__(self, other):
        return _Mat(*self.shape, tag=self.tag)

    def __add__(self, other):
        return _Mat(*self.shape, tag=self.tag)

    def __matmul__(self, other):
        return _Mat(self.shape[0], getattr(other, "shape", (0, 64))[1],
                    self.tag)

    def mean(self, dim):
        return _Mat(1, self.shape[1])

    def double(self):
        return self

    def contiguous(self):
        return self


class _W:
    """solve result stand-in carrying the cell's provenance."""

    def __init__(self, tag):
        self.tag = tag


class _Lstsq:
    def __init__(self, n, k):
        self.solution = _Mat(n, k)


class _FakeTorch:
    """Only the torch surface joint_solve/run touch."""

    float64 = "float64"

    class Generator:
        def manual_seed(self, seed):
            return self

    class linalg:
        fail = False

        @staticmethod
        def lstsq(a, b):
            if _FakeTorch.linalg.fail:
                raise RuntimeError("pca driver did not converge\n"
                                   "second line of the LAPACK message")
            return _Lstsq(a.shape[1], b.shape[1])

    @staticmethod
    def randperm(n, generator=None):
        return _Idx(n)

    @staticmethod
    def cat(mats):
        return _Mat(sum(m.shape[0] for m in mats), mats[0].shape[1])

    @staticmethod
    def eye(n, dtype=None):
        return _Mat(n, n)

    @staticmethod
    def pca_lowrank(m, q):
        return _Mat(*m.shape), _Mat(1, m.shape[1]), _Mat(m.shape[1], q)


def _world(n_public=4096, n_epochs=3):
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1], ("wire", t))

    return {"n_epochs": n_epochs,
            "public_h": _Mat(n_public, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object(),
            "wire": wire}


def _patch_toy(monkeypatch, joint=None, solve=None):
    """Run sj.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script the numerical core to known values."""
    monkeypatch.setattr(sj, "require_torch", lambda name: None)
    monkeypatch.setattr(sj, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world",
                        lambda **kw: _world())
    if joint is None:
        def joint(canon_by_t, wire_by_t, k, n_iters):
            return ([_W(("joint", t)) for t in range(len(canon_by_t))],
                    {"k": k, "alt_iters": n_iters})
    monkeypatch.setattr(sj, "joint_solve", joint)
    if solve is None:
        def solve(h, hw):
            return _W(("ind", hw.tag[1])), "lstsq+svd"
    monkeypatch.setattr(sj, "solve_w", solve)

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        return JOINT_TOP1 if w_hat.tag[0] == "joint" else IND_TOP1

    monkeypatch.setattr(sj, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="sj.json"):
    out = tmp_path / name
    args = sj.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "training"
    assert sj.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert sj.EXPERIMENT_ID == "subspace_joint"
    assert sj.MODES == ("training", "inference")
    assert sj.REQUIRES_LABELS is True
    assert "joint" in sj.DESCRIPTION
    assert validate_contract(sj, "subspace-joint") is sj
    assert ATTACKS["subspace-joint"] == "attacker.attacks.subspace_joint"
    assert load("subspace-joint") is sj
    md = attack_metadata.metadata_for("subspace-joint")
    assert md["persistence"] == "cross-epoch"
    assert md["surfaces"] == ("S1",)


def test_parser_defaults_are_the_documented_sweep():
    args = sj.build_parser().parse_args([])
    assert args.subspace_dims == [8, 16]
    assert args.pairs_per_epoch == [4, 8, 16, 32]
    assert args.epochs == 4 and args.alt_iters == 5
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]


def test_parser_rejects_bad_dim_and_help_works():
    with pytest.raises(SystemExit) as exc:
        sj.build_parser().parse_args(["--subspace-dims", "wide"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        sj.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_run_fails_closed_without_torch(tmp_path):
    if sj.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "sj.json"
    args = sj.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        sj.run(args)
    assert not out.exists()
    assert not (tmp_path / "sj.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if sj.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "sj.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "subspace-joint", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[subspace_joint] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_non_toy_run_is_a_descriptive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(sj, "require_torch", lambda name: None)
    args = sj.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="integration"):
        sj.run(args)
    assert not (tmp_path / "x.json").exists()


def test_joint_solve_returns_one_lifted_w_per_epoch(monkeypatch):
    monkeypatch.setattr(sj, "torch", _FakeTorch)
    monkeypatch.setattr(sj, "polar", lambda sol: _Mat(*sol.shape, "Q"))
    _FakeTorch.linalg.fail = False
    canon = [_Mat(16, 64) for _ in range(3)]
    wire = [_Mat(16, 64) for _ in range(3)]
    Ws, diag = sj.joint_solve(canon, wire, k=8, n_iters=3)
    assert isinstance(Ws, list) and len(Ws) == 3
    assert diag == {"k": 8, "alt_iters": 3}


def test_joint_solve_lstsq_failure_is_a_structured_error(monkeypatch):
    monkeypatch.setattr(sj, "torch", _FakeTorch)
    monkeypatch.setattr(sj, "polar", lambda sol: _Mat(*sol.shape, "Q"))
    _FakeTorch.linalg.fail = True
    try:
        Ws, diag = sj.joint_solve([_Mat(16, 64)], [_Mat(16, 64)], k=8,
                                  n_iters=1)
    finally:
        _FakeTorch.linalg.fail = False
    assert Ws is None
    assert diag == {"error": "error: pca driver did not converge"}


def test_toy_run_records_joint_vs_independent_per_cell(tmp_path,
                                                       monkeypatch):
    _patch_toy(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.subspace_joint.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    cells = art["results"]
    assert {(r["k"], r["pairs_per_epoch"], r["seed"]) for r in cells} == {
        (8, E, s) for E in (4, 8, 16) for s in (0, 1)}
    for r in cells:
        assert r["experiment"] == sj.EXPERIMENT_ID
        assert r["joint_top1_mean"] == JOINT_TOP1
        assert r["independent_top1_mean"] == IND_TOP1
        assert r["joint_top1_mean"] > r["independent_top1_mean"]
        assert r["n_solve_seeds"] == 1
    assert len(journal) == len(cells)
    assert all(j["experiment"] == sj.EXPERIMENT_ID and "error" not in j
               for j in journal)
    assert {(s["k"], s["pairs_per_epoch"]) for s in art["summary"]} == {
        (8, E) for E in (4, 8, 16)}
    for s in art["summary"]:
        assert s["joint_mean"] == JOINT_TOP1
        assert s["independent_mean"] == IND_TOP1
        assert s["n_seeds"] == 2


def test_failed_joint_solves_journal_both_error_flavours(tmp_path,
                                                         monkeypatch):
    calls = {"n": 0}

    def flaky(canon_by_t, wire_by_t, k, n_iters):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic diverged")
        return None, {"error": "error: ill-conditioned subspace"}

    _patch_toy(monkeypatch, joint=flaky)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["results"] == [] and art["summary"] == []
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 6
    by_msg = {j["error"] for j in errs}
    assert "error: synthetic diverged" in by_msg
    assert "error: ill-conditioned subspace" in by_msg
    for j in errs:
        assert j["experiment"] == sj.EXPERIMENT_ID
        assert j["k"] == 8 and j["pairs_per_epoch"] in (4, 8, 16)
        assert "Traceback" not in j["error"]


def test_failed_independent_solve_drops_that_epoch_only(tmp_path,
                                                        monkeypatch):
    """A failed per-epoch independent solve is skipped (not journaled, no
    error cell): the head-to-head still records, averaged over the epochs
    whose independent solve succeeded."""
    def solve(h, hw):
        if hw.tag == ("wire", 1):
            return None, "error: epoch-1 solve failed"
        return _W(("ind", hw.tag[1])), "lstsq+svd"

    _patch_toy(monkeypatch, solve=solve)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    cells = art["results"]
    assert len(cells) == 6
    for r in cells:
        assert r["joint_top1_mean"] == JOINT_TOP1
        assert r["independent_top1_mean"] == IND_TOP1
    assert all("error" not in j for j in journal)


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    assert a["config"] == b["config"]
