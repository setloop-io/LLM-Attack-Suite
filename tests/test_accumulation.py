"""accumulation (E-R1a, per-epoch oracle accumulation budget): torch-free
machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented sweep),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * run()'s record/journal/summary wiring end to end with the numerical
    core (solve_w / recovery_with_what_nn) scripted to seeded known values
    and a stand-in tensor/torch: the within-epoch solve scores high, the
    cross-epoch pooling CONTROL sits in the label-free band, the summary's
    max_safe_epoch / K50 crossing are the exact interpolants of the
    scripted curve, and solve failures land in the journal as structured
    "error:" records (never tracebacks).
"""

import json
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import accumulation as acc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WITHIN_TOP1 = {64: 12.0, 256: 61.0}
POOLED_TOP1 = 4.0
EXPECTED_K50 = 64 + (50.0 - 12.0) / (61.0 - 12.0) * (256 - 64)


class _Idx:
    """randperm result: an opaque permutation handle carrying its length."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, sl):
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else self.n
        return _Idx(stop - start)


class _Mat:
    """Tensor stand-in: carries shape + a provenance tag; every tensor op
    the module performs returns another _Mat so run()'s dataflow executes.
    The numeric payload lives in the scripted solve/recovery stubs.
    `origin` is the absolute row offset of row 0 (slices accumulate it), so
    tests can pin WHICH rows of the canonical replay a slice came from."""

    def __init__(self, n, h=64, tag=None, origin=0):
        self.shape = (n, h)
        self.tag = tag
        self.origin = origin

    def __getitem__(self, key):
        if isinstance(key, _Idx):
            return _Mat(key.n, self.shape[1], self.tag, self.origin)
        if isinstance(key, slice):
            start = key.start or 0
            if start < 0:
                start += self.shape[0]
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag,
                        self.origin + start)
        return _Mat(1, self.shape[1], self.tag)

    def float(self):
        return self

    def double(self):
        return self


class _W:
    """solve_w result stand-in carrying the cell's provenance."""

    def __init__(self, tag):
        self.tag = tag


class _FakeTorch:
    """Only the torch surface accumulation.run/solve_and_score touch."""

    class Generator:
        def manual_seed(self, seed):
            return self

    @staticmethod
    def randperm(n, generator=None):
        return _Idx(n)

    @staticmethod
    def cat(mats):
        return _Mat(sum(m.shape[0] for m in mats), mats[0].shape[1],
                    "pooled")

    @staticmethod
    def load(path, map_location=None):
        return {"h": _Mat(512, 64), "tok": _Mat(512, 1)}


def _world(n_public=4096, n_epochs=5):
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1], "wire")

    return {"n_epochs": n_epochs,
            "public_h": _Mat(n_public, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object(),
            "wire": wire}


def _patch_toy(monkeypatch, world=None, solve=None):
    """Run acc.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script the numerical core to known values."""
    monkeypatch.setattr(acc, "require_torch", lambda name: None)
    monkeypatch.setattr(acc, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world",
                        lambda **kw: world or _world())
    if solve is None:
        def solve(h, hw):
            if hw.tag == "pooled":
                return _W(("pooled",)), "lstsq+svd"
            return _W(("within", hw.shape[0])), "lstsq+svd"
    monkeypatch.setattr(acc, "solve_w", solve)

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        if w_hat.tag[0] == "pooled":
            return POOLED_TOP1
        return WITHIN_TOP1[w_hat.tag[1]]

    monkeypatch.setattr(acc, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="er1.json"):
    out = tmp_path / name
    args = acc.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "training"
    assert acc.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert acc.EXPERIMENT_ID == "er1a_accumulation"
    assert acc.MODES == ("training", "inference")
    assert acc.REQUIRES_LABELS is True
    assert "E-R1a" in acc.DESCRIPTION
    assert validate_contract(acc, "accumulation") is acc
    assert ATTACKS["accumulation"] == "attacker.attacks.accumulation"
    assert load("accumulation") is acc
    md = attack_metadata.metadata_for("accumulation")
    assert md["surfaces"] == ("S1",)
    assert md["security_properties"] == ("confidentiality",)
    assert "LLM02" in md["owasp_llm_2026"]


def test_parser_defaults_are_the_documented_sweep():
    args = acc.build_parser().parse_args([])
    assert args.epoch_sizes == [64, 256, 1024]
    assert args.capture_dir is None and args.canonical_pt is None
    assert args.phase is None
    assert (args.hidden, args.seed, args.toy, args.quick) == (64, 1234,
                                                              False, False)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]
    assert args.output is None


def test_parser_accepts_capture_flags():
    args = acc.build_parser().parse_args(
        ["--epoch-sizes", "8", "32", "--capture-dir", "caps",
         "--canonical-pt", "replay.pt", "--phase", "bwd", "--toy"])
    assert args.epoch_sizes == [8, 32] and args.phase == "bwd"
    assert args.capture_dir == "caps" and args.canonical_pt == "replay.pt"
    assert args.toy is True


def test_parser_rejects_bad_epoch_size_and_help_works():
    with pytest.raises(SystemExit) as exc:
        acc.build_parser().parse_args(["--epoch-sizes", "lots"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        acc.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_run_fails_closed_without_torch(tmp_path):
    if acc.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er1.json"
    args = acc.build_parser().parse_args(["--toy", "--quick", "--output",
                                          str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        acc.run(args)
    assert not out.exists()
    assert not (tmp_path / "er1.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if acc.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er1.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "accumulation", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[er1a_accumulation] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_toy_run_within_epoch_high_pooled_control_in_band(tmp_path,
                                                          monkeypatch):
    _patch_toy(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.er1a.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["attack"] == acc.EXPERIMENT_ID
    assert art["config"]["epoch_sizes"] == [64, 256]
    cells = art["results"]
    assert {(r["epoch_size"], r["seed"]) for r in cells} == {
        (64, 0), (64, 1), (256, 0), (256, 1)}
    for r in cells:
        assert r["experiment"] == acc.EXPERIMENT_ID
        assert r["within_epoch_top1"] == WITHIN_TOP1[r["epoch_size"]]
        assert r["pooled_cross_epoch_top1"] == POOLED_TOP1
        assert r["n_solves"] == 2
    assert len(journal) == len(cells)
    assert all(j["experiment"] == acc.EXPERIMENT_ID and "error" not in j
               for j in journal)


def test_summary_interpolates_the_exact_k50_crossing(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    art, _ = _run(["--toy", "--quick"], tmp_path)
    assert len(art["summary"]) == 2
    for s in art["summary"]:
        assert s["experiment"] == acc.EXPERIMENT_ID
        assert s["max_safe_epoch"] == 64
        cross = s["k50_crossing"]
        assert cross["k50_method"] == "linear_in_K"
        assert cross["k50_bracket"] == [64, 256]
        assert cross["k50_interpolated"] == pytest.approx(EXPECTED_K50,
                                                          abs=1e-6)


def test_oversized_epoch_budgets_are_skipped_with_a_note(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    _patch_toy(monkeypatch, world=_world(n_public=300))
    art, _ = _run(["--toy"], tmp_path)
    assert {r["epoch_size"] for r in art["results"]} == {64}
    assert {r["seed"] for r in art["results"]} == {0, 1, 2}
    assert all(r["n_solves"] == 3 for r in art["results"])
    out = capsys.readouterr().out
    assert "only 300 pairs available, skipping" in out
    for s in art["summary"]:
        assert s["max_safe_epoch"] == 64
        assert s["k50_crossing"]["k50_method"] == "not_bracketed"


def test_failed_solves_journal_structured_errors(tmp_path, monkeypatch):
    _patch_toy(monkeypatch,
               solve=lambda h, hw: (None, "error: lstsq did not converge"))
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["results"] == [] and art["summary"] == []
    assert len(journal) == 4
    for j in journal:
        assert j["experiment"] == acc.EXPERIMENT_ID
        assert j["error"].startswith("error:")
        assert "lstsq did not converge" in j["error"]
        assert "Traceback" not in j["error"]
        assert j["epoch_size"] in (64, 256) and j["seed"] in (0, 1)


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    assert a["config"] == b["config"]


def test_capture_dir_branch_row_aligns_the_canonical_replay(tmp_path,
                                                            monkeypatch):
    """--capture-dir + --canonical-pt: captures grouped per epoch via the
    capture core, canonical replay rows row-aligned within each epoch, the
    victim set held out from epoch 0 — same cell/journal contract as toy."""
    _patch_toy(monkeypatch)
    monkeypatch.setattr(acc.captures, "scan_captures",
                        lambda d, mode: [({"epoch": 0}, "wire_0000.pt"),
                                         ({"epoch": 1}, "wire_0001.pt")])
    monkeypatch.setattr(
        acc.captures, "load_epoch_rows",
        lambda records, phase=None: {0: _Mat(256, 64, "wire"),
                                     1: _Mat(256, 64, "wire")})
    art, journal = _run(["--quick", "--capture-dir", "caps",
                         "--canonical-pt", "replay.pt"], tmp_path)
    assert art["config"]["toy"] is False
    assert art["config"]["capture_dir"] == "caps"
    cells = art["results"]
    assert {(r["epoch_size"], r["seed"]) for r in cells} == {
        (64, 0), (64, 1), (256, 0), (256, 1)}
    for r in cells:
        assert r["within_epoch_top1"] == WITHIN_TOP1[r["epoch_size"]]
        assert r["pooled_cross_epoch_top1"] == POOLED_TOP1
    assert len(journal) == len(cells)
    for s in art["summary"]:
        assert s["k50_crossing"]["k50_bracket"] == [64, 256]


def _patch_captures(monkeypatch, epoch_rows, n_canon):
    """Stand in the capture core + canonical replay for --capture-dir runs:
    epoch_rows maps epoch -> (n_rows, tag); the canonical replay is a
    single origin-tracked _Mat of n_canon rows, so slices reveal which
    replay rows (which epoch) they came from."""
    monkeypatch.setattr(
        acc.captures, "scan_captures",
        lambda d, mode: [({"epoch": ep}, f"wire_{i:04d}.pt")
                         for i, ep in enumerate(sorted(epoch_rows))])
    monkeypatch.setattr(
        acc.captures, "load_epoch_rows",
        lambda records, phase=None: {ep: _Mat(n, 64, tag)
                                     for ep, (n, tag) in epoch_rows.items()})
    monkeypatch.setattr(_FakeTorch, "load", staticmethod(
        lambda path, map_location=None: {"h": _Mat(n_canon, 64),
                                         "tok": _Mat(n_canon, 1)}))


def test_capture_victim_labels_aligned_to_epoch0_partial_final_epoch(
        tmp_path, monkeypatch):
    """Regression for the multi-epoch label misalignment, partial-final-
    epoch case. The victim wire rows are epoch 0's tail, so the labels must
    be epoch 0's canonical tail (replay rows [192, 256)); the old alignment
    scored against the LAST epoch's tail (rows [320, 384)), which trends
    toward a false PASS. Aligned labels score 100 below, the old misaligned
    ones 0 — the old code fails this test loudly."""
    _patch_toy(monkeypatch)
    _patch_captures(monkeypatch,
                    {0: (256, "wire-ep0"), 1: (128, "wire-ep1")},
                    n_canon=384)
    seen = []

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        seen.append((h_wire, victim_tok))
        aligned = (victim_tok.origin, victim_tok.shape[0]) == (192, 64)
        return 100.0 if aligned else 0.0

    monkeypatch.setattr(acc, "recovery_with_what_nn", recovery)
    art, _ = _run(["--capture-dir", "caps", "--canonical-pt", "replay.pt",
                   "--epoch-sizes", "64", "--seeds", "0",
                   "--solve-seeds", "0"], tmp_path)
    assert len(art["results"]) == 1
    cell = art["results"][0]
    assert cell["within_epoch_top1"] == 100.0
    assert cell["pooled_cross_epoch_top1"] == 100.0
    assert len(seen) == 2
    for h_wire, victim_tok in seen:
        assert h_wire.tag == "wire-ep0"
        assert (victim_tok.origin, victim_tok.shape[0]) == (192, 64)


def test_capture_victim_labels_single_epoch_byte_identical(tmp_path,
                                                           monkeypatch):
    """Single-epoch captures are unaffected by the alignment fix: epoch 0
    IS the last epoch, so the per-epoch label slice is the same canonical
    tail rows [N-nv, N) the old global indexing produced."""
    _patch_toy(monkeypatch)
    _patch_captures(monkeypatch, {0: (256, "wire-ep0")}, n_canon=256)
    seen = []

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        seen.append((h_wire, victim_tok))
        return 42.0

    monkeypatch.setattr(acc, "recovery_with_what_nn", recovery)
    art, _ = _run(["--capture-dir", "caps", "--canonical-pt", "replay.pt",
                   "--epoch-sizes", "64", "--seeds", "0",
                   "--solve-seeds", "0"], tmp_path)
    assert [r["within_epoch_top1"] for r in art["results"]] == [42.0]
    assert len(seen) == 2
    for h_wire, victim_tok in seen:
        assert h_wire.tag == "wire-ep0"
        assert (victim_tok.origin, victim_tok.shape[0]) == (192, 64)


def test_no_toy_and_no_captures_is_a_descriptive_exit(tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(acc, "require_torch", lambda name: None)
    args = acc.build_parser().parse_args(["--output",
                                          str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="pass --toy"):
        acc.run(args)
    assert not (tmp_path / "x.json").exists()


def _victim():
    return {"h_wire": _Mat(4), "ref_h": _Mat(4), "ref_tok": None,
            "victim_tok": None}


def test_solve_and_score_collects_one_recovery_per_solve_seed(monkeypatch):
    monkeypatch.setattr(acc, "torch", _FakeTorch)
    monkeypatch.setattr(acc, "solve_w", lambda h, hw: (_W("w"), "lstsq+svd"))
    monkeypatch.setattr(acc, "recovery_with_what_nn",
                        lambda *a: 42.0)
    recs, err = acc.solve_and_score(_Mat(8), _Mat(8), _victim(), [0, 1, 2],
                                    seed=7)
    assert recs == [42.0, 42.0, 42.0] and err is None


def test_solve_and_score_keeps_going_past_failed_solves(monkeypatch):
    monkeypatch.setattr(acc, "torch", _FakeTorch)
    calls = {"n": 0}

    def flaky(h, hw):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "error: gesdd failed"
        return _W("w"), "lstsq+eigh_polar_fallback"

    monkeypatch.setattr(acc, "solve_w", flaky)
    monkeypatch.setattr(acc, "recovery_with_what_nn", lambda *a: 42.0)
    recs, err = acc.solve_and_score(_Mat(8), _Mat(8), _victim(), [0, 1, 2],
                                    seed=7)
    assert recs == [42.0, 42.0]
    assert err == "error: gesdd failed"


def test_solve_and_score_all_failed_returns_empty_plus_error(monkeypatch):
    monkeypatch.setattr(acc, "torch", _FakeTorch)
    monkeypatch.setattr(acc, "solve_w",
                        lambda h, hw: (None, "error: singular"))
    recs, err = acc.solve_and_score(_Mat(8), _Mat(8), _victim(), [0, 1],
                                    seed=7)
    assert recs == [] and err == "error: singular"


def test_k50_primitives_used_by_the_summary():
    assert acc.K50_THRESHOLD_PCT == 50.0
    curve = [{"K": k, "top1_mean": t}
             for k, t in ((64, 12.0), (256, 61.0))]
    cross = acc.crossing_k(curve)
    assert cross["k50_bracket"] == [64, 256]
    assert cross["k50_interpolated"] == pytest.approx(EXPECTED_K50)
    assert acc.partition_pairs(250, 4) == [63, 63, 62, 62]
