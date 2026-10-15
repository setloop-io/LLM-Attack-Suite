"""wire-eval (per-epoch wire-capture scoring; e9_eval/er_train_eval
superset): torch-free machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
capture writer in attacker/synthetic.py is torch-only), so the numeric toy
tier is covered by the torch env. This file pins everything that stays
checkable torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented configuration),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * score_epochs directly with a scripted numerical core: per-epoch cells
    carry the within-epoch recovery next to the label-free band; the
    cross-epoch pooling control is scored per epoch key and sits in the
    band; under-populated epochs are skipped; failed solves journal
    structured "error:" records (per epoch AND for the pooled arm),
  * run() end to end on both input paths — --toy (capture writer +
    scanner exercised through the fakes) and --capture-dir +
    --canonical-pt (per-epoch canonical alignment with a held-out victim
    tail) — and the empty-capture-dir fail-closed exit.
"""

import json
import os
import subprocess
import sys
import types

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import wire_eval as we

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WITHIN_TOP1 = {0: 80.0, 1: 85.0, 2: 90.0, 3: 95.0}
POOLED_TOP1 = {0: 4.0, 1: 5.0, 2: 6.0, 3: 7.0}
BAND_TOP1 = 3.9


class _Mat:
    """Tensor stand-in: shape + provenance tag; every op the module
    performs returns another _Mat so the scoring dataflow executes."""

    def __init__(self, n, h=64, tag=None):
        self.shape = (n, h)
        self.tag = tag

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            if start < 0:
                start += self.shape[0]
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
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
    """Only the torch surface wire_eval touches."""

    @staticmethod
    def cat(mats):
        return _Mat(sum(m.shape[0] for m in mats), mats[0].shape[1],
                    "cat")

    @staticmethod
    def load(path, map_location=None):
        return {"h": _Mat(1024, 64), "tok": _Mat(1024, 1)}


def _world(n_epochs=4):
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1],
                    ("vw", t) if h.tag == "victim" else ("wire", t))

    return {"n_epochs": n_epochs,
            "public_h": _Mat(2048, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64, "victim"), "victim_tok": object(),
            "wire": wire}


def _patch_core(monkeypatch, solve=None):
    """Script the numerical core of score_epochs to known values: per-epoch
    solves recover at WITHIN_TOP1[epoch], the pooled cross-epoch solve sits
    at POOLED_TOP1[epoch], the label-free band is BAND_TOP1."""
    monkeypatch.setattr(we, "torch", _FakeTorch)
    if solve is None:
        def solve(can, wire_rows):
            if wire_rows.tag == "cat":
                return _W(("pooled",)), "lstsq+svd"
            return _W(("ep", wire_rows.tag[1])), "lstsq+svd"
    monkeypatch.setattr(we, "solve_w", solve)
    monkeypatch.setattr(we, "nn_mean_decode", lambda *a: BAND_TOP1)

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        if w_hat.tag[0] == "pooled":
            return POOLED_TOP1[h_wire.tag[1]]
        return WITHIN_TOP1[w_hat.tag[1]]

    monkeypatch.setattr(we, "recovery_with_what_nn", recovery)


def _patch_run(monkeypatch, records=None, by_epoch=None):
    """Run we.run() torch-free: no-op the torch guard, stand in for the
    toy world, the capture writer and the capture loader."""
    monkeypatch.setattr(we, "require_torch", lambda name: None)
    _patch_core(monkeypatch)
    monkeypatch.setattr(synthetic, "make_toy_world", lambda **kw: _world())
    monkeypatch.setattr(synthetic, "write_toy_captures",
                        lambda world, d, mode, rows_per_epoch: [])
    if records is None:
        records = [({"phase": "decode", "epoch": t}, f"wire_{t:04d}.pt")
                   for t in range(4)]
    monkeypatch.setattr(we.captures, "scan_captures",
                        lambda d, mode: records)
    if by_epoch is None:
        by_epoch = {t: _Mat(128, 64, ("wire", t)) for t in range(4)}
    monkeypatch.setattr(we.captures, "load_epoch_rows",
                        lambda recs, phase=None:
                        dict(by_epoch) if phase in ("fwd", "decode")
                        else {})


def _args(tmp_path, name="w.json"):
    return types.SimpleNamespace(output=str(tmp_path / name),
                                 min_epoch_rows=16)


def test_contract_and_registry():
    assert we.EXPERIMENT_ID == "wire_eval"
    assert we.MODES == ("training", "inference")
    assert we.REQUIRES_LABELS is True
    assert "e9_eval" in we.DESCRIPTION
    assert validate_contract(we, "wire-eval") is we
    assert ATTACKS["wire-eval"] == "attacker.attacks.wire_eval"
    assert load("wire-eval") is we
    md = attack_metadata.metadata_for("wire-eval")
    assert set(md["surfaces"]) == {"S1", "S2"}
    assert "LLM02" in md["owasp_llm_2026"]


def test_parser_defaults_are_the_documented_configuration():
    args = we.build_parser().parse_args([])
    assert args.capture_dir is None and args.canonical_pt is None
    assert args.min_epoch_rows == 16
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]


def test_parser_rejects_bad_min_rows_and_help_works():
    with pytest.raises(SystemExit) as exc:
        we.build_parser().parse_args(["--min-epoch-rows", "some"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        we.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_run_fails_closed_without_torch(tmp_path):
    if we.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "w.json"
    args = we.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "training"
    with pytest.raises(SystemExit, match="requires torch"):
        we.run(args)
    assert not out.exists()
    assert not (tmp_path / "w.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if we.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "w.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "training",
         "--attack", "wire-eval", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[wire_eval] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_no_toy_and_no_captures_is_a_descriptive_exit(tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(we, "require_torch", lambda name: None)
    args = we.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "training"
    with pytest.raises(SystemExit, match="pass --toy"):
        we.run(args)
    assert not (tmp_path / "x.json").exists()


def _victim():
    return {"h_wire": _Mat(512, 64, ("vw", 0)),
            "h_wire_by_ep": {t: _Mat(512, 64, ("vw", t)) for t in range(4)},
            "ref_h": _Mat(2048, 64), "ref_tok": object(),
            "victim_tok": object()}


def test_score_epochs_cells_band_and_pooled_control(tmp_path, monkeypatch):
    _patch_core(monkeypatch)
    rows = we.score_epochs(
        {0: _Mat(128, 64, ("wire", 0)), 1: _Mat(128, 64, ("wire", 1))},
        {0: _Mat(128, 64), 1: _Mat(128, 64)},
        None, _victim(), _args(tmp_path), "fwd")
    cells = [r for r in rows if "epoch" in r]
    assert [(r["epoch"], r["n_rows"]) for r in cells] == [(0, 128),
                                                          (1, 128)]
    for r in cells:
        assert r["experiment"] == we.EXPERIMENT_ID
        assert r["phase_arm"] == "fwd"
        assert r["within_epoch_top1"] == WITHIN_TOP1[r["epoch"]]
        assert r["labelfree_band_top1"] == BAND_TOP1
    pooled = [r for r in rows if r.get("arm") == "pooled_cross_epoch"]
    assert len(pooled) == 1
    p = pooled[0]
    assert p["pooled_top1_per_epoch"] == [POOLED_TOP1[0], POOLED_TOP1[1]]
    assert p["pooled_top1_mean_over_epochs"] == sum(
        POOLED_TOP1[e] for e in (0, 1)) / 2
    assert "label-free band" in p["note"]
    journal = [json.loads(ln) for ln in
               (tmp_path / "w.json.jsonl").read_text().splitlines()]
    assert len(journal) == 3
    assert all(j["experiment"] == we.EXPERIMENT_ID for j in journal)


def test_score_epochs_skips_underpopulated_epochs(tmp_path, monkeypatch,
                                                  capsys):
    _patch_core(monkeypatch)
    rows = we.score_epochs({0: _Mat(8, 64, ("wire", 0))}, {0: _Mat(8, 64)},
                           None, _victim(), _args(tmp_path), "fwd")
    assert rows == []
    assert "8 rows < --min-epoch-rows, skipped" in capsys.readouterr().out


def test_score_epochs_victim_fallbacks_without_per_epoch_dicts(tmp_path,
                                                               monkeypatch):
    """victim rows/tokens fall back to the shared entries when the
    per-epoch dicts have no entry for an epoch."""
    _patch_core(monkeypatch)
    victim = _victim()
    victim["h_wire_by_ep"] = {1: _Mat(512, 64, ("vw", 1))}
    seen = []

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        seen.append(h_wire.tag)
        return WITHIN_TOP1[w_hat.tag[1]] if w_hat.tag[0] == "ep" \
            else POOLED_TOP1.get(h_wire.tag[1], BAND_TOP1)

    monkeypatch.setattr(we, "recovery_with_what_nn", recovery)
    rows = we.score_epochs(
        {0: _Mat(128, 64, ("wire", 0)), 1: _Mat(128, 64, ("wire", 1))},
        {0: _Mat(128, 64), 1: _Mat(128, 64)},
        None, victim, _args(tmp_path), "fwd")
    assert [r["within_epoch_top1"] for r in rows if "epoch" in r] == [
        WITHIN_TOP1[0], WITHIN_TOP1[1]]
    assert ("vw", 0) in seen


def test_score_epochs_failed_solves_journal_structured_errors(tmp_path,
                                                              monkeypatch):
    _patch_core(monkeypatch,
                solve=lambda can, wire: (None, "error: epoch solve failed"))
    rows = we.score_epochs(
        {0: _Mat(128, 64, ("wire", 0)), 1: _Mat(128, 64, ("wire", 1))},
        {0: _Mat(128, 64), 1: _Mat(128, 64)},
        None, _victim(), _args(tmp_path), "bwd")
    assert rows == []
    journal = [json.loads(ln) for ln in
               (tmp_path / "w.json.jsonl").read_text().splitlines()]
    assert len(journal) == 3
    per_ep = [j for j in journal if "epoch" in j]
    pooled = [j for j in journal if j.get("arm") == "pooled"]
    assert len(per_ep) == 2 and len(pooled) == 1
    for j in journal:
        assert j["experiment"] == we.EXPERIMENT_ID
        assert j["phase_arm"] == "bwd"
        assert j["error"].startswith("error:")
        assert "epoch solve failed" in j["error"]
        assert "Traceback" not in j["error"]


def _run(args_list, tmp_path, name="wire.json", mode="inference"):
    out = tmp_path / name
    args = we.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = mode
    assert we.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_toy_run_per_epoch_cells_and_pooled_control(tmp_path, monkeypatch):
    _patch_run(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.wire_eval.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["toy"] is True
    cells = [r for r in art["results"] if "epoch" in r]
    assert {(r["epoch"], r["phase_arm"]) for r in cells} == {
        (t, "decode") for t in range(4)}
    assert len(cells) == 2 * 4
    for r in cells:
        assert r["within_epoch_top1"] == WITHIN_TOP1[r["epoch"]]
        assert r["labelfree_band_top1"] == BAND_TOP1
    pooled = [r for r in art["results"]
              if r.get("arm") == "pooled_cross_epoch"]
    assert len(pooled) == 2
    for p in pooled:
        assert p["pooled_top1_per_epoch"] == [POOLED_TOP1[t]
                                              for t in range(4)]
    assert len(journal) == len(art["results"]) == 2 * 5
    s = art["summary"][0]
    assert s["n_epoch_cells"] == 8
    assert s["within_epoch_top1_mean"] == sum(WITHIN_TOP1.values()) / 4


def test_training_mode_runs_fwd_arm_and_skips_empty_bwd(tmp_path,
                                                        monkeypatch):
    _patch_run(monkeypatch)
    art, _ = _run(["--toy", "--quick"], tmp_path, mode="training")
    assert {r["phase_arm"] for r in art["results"]} == {"fwd"}
    assert len([r for r in art["results"] if "epoch" in r]) == 8


def test_real_capture_path_aligns_canonical_rows_per_epoch(tmp_path,
                                                           monkeypatch):
    """--capture-dir + --canonical-pt: canonical rows aligned by row order
    within each epoch, the victim set held out from each epoch's own wire
    tail (nv = n//4), then scored per epoch. The solve pairs the canonical
    rows with the MATCHING wire-row prefix (row-equal operands), never the
    full epoch rows including the held-out tail."""
    _patch_run(monkeypatch,
               by_epoch={0: _Mat(256, 64, ("wire", 0)),
                         1: _Mat(256, 64, ("wire", 1))})
    seen = {"per_epoch": [], "pooled": []}

    def solve(can, wire_rows):
        if wire_rows.tag == "cat":
            seen["pooled"].append((can.shape[0], wire_rows.shape[0]))
            return _W(("pooled",)), "lstsq+svd"
        seen["per_epoch"].append((can.shape[0], wire_rows.shape[0]))
        return _W(("ep", wire_rows.tag[1])), "lstsq+svd"

    monkeypatch.setattr(we, "solve_w", solve)
    art, _ = _run(["--capture-dir", "caps", "--canonical-pt", "replay.pt",
                   "--quick"], tmp_path)
    assert art["config"]["toy"] is False
    assert art["config"]["capture_dir"] == "caps"
    assert seen["per_epoch"] and all(
        c == 192 and w == 192 for c, w in seen["per_epoch"])
    assert seen["pooled"] == [(384, 384)] * 2
    cells = [r for r in art["results"] if "epoch" in r]
    assert {r["epoch"] for r in cells} == {0, 1}
    for r in cells:
        assert r["within_epoch_top1"] == WITHIN_TOP1[r["epoch"]]
    assert art["summary"][0]["n_epoch_cells"] == len(cells)


def test_empty_capture_dir_fails_closed(tmp_path, monkeypatch):
    _patch_run(monkeypatch, records=[])
    args = we.build_parser().parse_args(
        ["--capture-dir", "caps", "--canonical-pt", "replay.pt",
         "--output", str(tmp_path / "x.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="no wire captures"):
        we.run(args)
    assert not (tmp_path / "x.json").exists()


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_run(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    assert a["config"] == b["config"]
