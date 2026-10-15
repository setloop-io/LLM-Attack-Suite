"""leak-accumulation (persistent-APT cross-epoch leak budget): torch-free
machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented budget sweep),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * leak_completion directly with a stand-in torch: the leaked rows are
    restored EXACT and in their true positions, the completion basis is
    filled to H orthonormal rows, and a degenerate (zero-norm) candidate
    column is skipped at the 1e-8 floor,
  * run()'s budget sweep with the numerical core scripted to known
    values: cumulative_leaked_rows grows by budget per epoch (the
    per-epoch budget does NOT reset — the attack's whole point),
    composite decode is recorded per (budget, seed, epoch), the summary
    aggregates per budget, and a decode failure journals a structured
    error record and skips just that cell.
"""

import json
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import leak_accumulation as la

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DECODE = {1: 5.0, 4: 45.0}


class _Idx:
    """randperm result: an opaque row-index set carrying its size."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, sl):
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else self.n
        return _Idx(stop - start)


class _Mat:
    """Tensor stand-in: shape + provenance tag + a scripted norm; every
    op leak_completion performs returns another _Mat carrying the tag so
    the Gram-Schmidt dataflow executes and stays attributable."""

    def __init__(self, n, h=64, tag=None, norm_val=1.0, norms=None):
        self.shape = (n, h)
        self.tag = tag
        self.norm_val = norm_val
        self.norms = norms
        self.sets = []

    def __getitem__(self, key):
        if isinstance(key, _Idx):
            return _Mat(key.n, self.shape[1], self.tag)
        if isinstance(key, tuple):
            rows, col = key
            norm = self.norms[col] if self.norms else 1.0
            return _Mat(self.shape[0], 1, ("qcol", col), norm)
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
        return _Mat(1, self.shape[1], ("leakrow", key))

    def __setitem__(self, key, value):
        self.sets.append((key, value))

    def __matmul__(self, other):
        return _Mat(self.shape[0], 1, self.tag, self.norm_val)

    def __mul__(self, other):
        return _Mat(*self.shape, self.tag, self.norm_val)

    __rmul__ = __mul__

    def __sub__(self, other):
        return _Mat(*self.shape, self.tag, self.norm_val)

    def __truediv__(self, other):
        return _Mat(*self.shape, self.tag, self.norm_val)

    def norm(self):
        return self.norm_val

    def clone(self):
        return _Mat(self.shape[0], self.shape[1], self.tag,
                    self.norm_val, self.norms)

    def double(self):
        return self


class _FakeTorch:
    """Only the torch surface leak_accumulation touches."""

    float64 = "float64"
    qr_norms = None

    class Generator:
        def manual_seed(self, seed):
            return self

    class linalg:
        @staticmethod
        def qr(m):
            return _Mat(*m.shape, norms=_FakeTorch.qr_norms), _Mat(*m.shape)

    @staticmethod
    def randperm(n, generator=None):
        return _Idx(n)

    @staticmethod
    def randn(n, h, generator=None, dtype=None):
        return _Mat(n, h)

    @staticmethod
    def stack(rows):
        return _Mat(len(rows), rows[0].shape[1] if rows else 0,
                    [r.tag for r in rows])


def _world(n_epochs=4):
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1], ("wire", t))

    return {"n_epochs": n_epochs,
            "Ws": [_Mat(64, 64, ("W", t)) for t in range(n_epochs)],
            "public_h": _Mat(1024, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object(),
            "wire": wire}


def _patch_toy(monkeypatch, recovery=None):
    """Run la.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script leak_completion/recovery to known values.
    The fake completion tags its result with (budget, epoch) so the
    scripted decode is attributable."""
    monkeypatch.setattr(la, "require_torch", lambda name: None)
    monkeypatch.setattr(la, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world",
                        lambda **kw: _world())

    def completion(w_true, row_idx, seed):
        return _Mat(64, 64, ("leak", row_idx.n, w_true.tag[1], seed))

    monkeypatch.setattr(la, "leak_completion", completion)
    if recovery is None:
        def recovery(h_wire, w_tilde, ref_h, ref_tok, victim_tok):
            return DECODE[w_tilde.tag[1]]
    monkeypatch.setattr(la, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="leak.json"):
    out = tmp_path / name
    args = la.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "training"
    assert la.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert la.EXPERIMENT_ID == "leak_accumulation"
    assert la.MODES == ("training", "inference")
    assert la.REQUIRES_LABELS == "insider row-leak (breach-case parametrization)"
    assert "APT" in la.DESCRIPTION
    assert validate_contract(la, "leak-accumulation") is la
    assert ATTACKS["leak-accumulation"] == "attacker.attacks.leak_accumulation"
    assert load("leak-accumulation") is la
    md = attack_metadata.metadata_for("leak-accumulation")
    assert md["actor_intent"] == "malicious"
    assert md["persistence"] == "cross-epoch"
    assert md["surfaces"] == ("S1",)


def test_parser_defaults_are_the_documented_sweep():
    args = la.build_parser().parse_args([])
    assert args.epochs == 8
    assert args.leak_per_epoch == [1, 4, 16]
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]


def test_parser_rejects_bad_budget_and_help_works():
    with pytest.raises(SystemExit) as exc:
        la.build_parser().parse_args(["--leak-per-epoch", "some"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        la.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_run_fails_closed_without_torch(tmp_path):
    if la.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "leak.json"
    args = la.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        la.run(args)
    assert not out.exists()
    assert not (tmp_path / "leak.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if la.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "leak.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "leak-accumulation", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[leak_accumulation] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_non_toy_run_is_a_descriptive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(la, "require_torch", lambda name: None)
    args = la.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="integration point"):
        la.run(args)
    assert not (tmp_path / "x.json").exists()


def test_leak_completion_restores_leaked_rows_exactly(monkeypatch):
    """The e8 attack-4 construction: a random orthogonal completion of the
    leaked row space, with the leaked rows put back EXACT and in position."""
    monkeypatch.setattr(la, "torch", _FakeTorch)
    _FakeTorch.qr_norms = None
    row_idx = _Idx(2)
    out = la.leak_completion(_Mat(8, 8), row_idx, seed=7)
    assert out.shape == (8, 8)
    tags = out.tag
    assert tags[:2] == [("leakrow", 0), ("leakrow", 1)]
    assert sum(1 for t in tags if t[0] == "qcol") == 6
    assert len(out.sets) == 1
    key, leaked = out.sets[0]
    assert key is row_idx and leaked.shape == (2, 8)


def test_leak_completion_skips_degenerate_columns_at_the_floor(monkeypatch):
    """A candidate column Gram-Schmidt'd to ~(near-)zero norm is skipped
    (n > 1e-8 floor); the completion still fills the basis."""
    monkeypatch.setattr(la, "torch", _FakeTorch)
    _FakeTorch.qr_norms = [0.0] + [1.0] * 7
    try:
        out = la.leak_completion(_Mat(8, 8), _Idx(1), seed=3)
    finally:
        _FakeTorch.qr_norms = None
    assert out.shape == (8, 8)
    tags = out.tag
    assert ("qcol", 0) not in tags
    assert sum(1 for t in tags if t[0] == "qcol") == 7
    assert len(out.sets) == 1


def test_toy_run_cumulative_leak_does_not_reset_per_epoch(tmp_path,
                                                          monkeypatch):
    _patch_toy(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.leak_accumulation.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["leak_per_epoch"] == [1, 4]
    cells = art["results"]
    assert len(cells) == 16
    for r in cells:
        assert r["experiment"] == la.EXPERIMENT_ID
        assert r["cumulative_leaked_rows"] == r["leak_per_epoch"] * (
            r["epoch"] + 1)
        assert r["leaked_rows_this_epoch"] == r["leak_per_epoch"]
        assert r["composite_decode_top1"] == DECODE[r["leak_per_epoch"]]
    assert len(journal) == len(cells)
    assert all(j["experiment"] == la.EXPERIMENT_ID and "error" not in j
               for j in journal)
    summ = {s["leak_per_epoch"]: s for s in art["summary"]}
    assert set(summ) == {1, 4}
    for budget, s in summ.items():
        assert s["decode_mean"] == DECODE[budget]
        assert s["final_cumulative_leaked_rows"] == budget * 4
        assert s["n_cells"] == 8


def test_failed_decode_journals_an_error_and_skips_just_that_cell(
        tmp_path, monkeypatch):
    def recovery(h_wire, w_tilde, ref_h, ref_tok, victim_tok):
        _, budget, epoch, completion_seed = w_tilde.tag
        if (budget, epoch, completion_seed) == (4, 2, 1234 + 2):
            raise RuntimeError("composite decode diverged")
        return DECODE[budget]

    _patch_toy(monkeypatch, recovery=recovery)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    present = {(r["leak_per_epoch"], r["seed"], r["epoch"])
               for r in art["results"]}
    assert len(art["results"]) == 15
    assert (4, 0, 2) not in present
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 1
    err = errs[0]
    assert err["experiment"] == la.EXPERIMENT_ID
    assert (err["budget"], err["seed"], err["epoch"]) == (4, 0, 2)
    assert "composite decode diverged" in err["error"]
    assert "Traceback" not in err["error"]
    summ = {s["leak_per_epoch"]: s for s in art["summary"]}
    assert summ[4]["n_cells"] == 7
    assert summ[4]["decode_mean"] == DECODE[4]
    assert summ[4]["final_cumulative_leaked_rows"] == 16
    assert summ[1]["n_cells"] == 8


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    assert a["config"] == b["config"]
