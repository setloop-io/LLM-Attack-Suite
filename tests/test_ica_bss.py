"""ica-bss attack module (attacker/attacks/ica_bss.py): contract, parser,
fail-closed guards, and the seeded toy tier end to end.

The toy tier's heavy machinery is torch-only (attacker/ica.py FastICA:
eigh/QR fixed-point; attacker/synthetic.py toy world; common.nn_mean_decode
cdist). These tests drive run() torch-free by substituting, at the
module's own namespace seams, a pure-python world / unmixing / NN-mean
decode with KNOWN answers:

  * clustered world (tokens near per-token directions): decode must sit
    far above the label-free band;
  * unstructured Gaussian world: everything at the floor (nothing to
    recover — the ICA-unidentifiable regime);
  * best-of-restarts selection: the good unmixing must beat a planted
    rank-deficient one;
  * a failed restart is journaled as a structured error record and the
    cell continues (fail-closed).

The genuine FastICA convergence claims (non-Gaussian mixture recovered /
Gaussian mixture at floor for the ALGORITHM itself) remain torch-gated in
attacker/ica.py and are not re-derived here.
"""

import json
import random

import pytest

from attacker import attacks
from attacker.attacks import ica_bss as ica


class _Mat:
    """Row-major matrix exposing the tensor verbs ica_bss.run uses:
    .double()/.float() (no-ops), row slicing, broadcast row-vector
    subtraction, and @ against another matrix."""

    def __init__(self, rows):
        self.rows = [list(map(float, r)) for r in rows]

    def double(self):
        return self

    def float(self):
        return self

    def __getitem__(self, sl):
        return _Mat(self.rows[sl])

    def __sub__(self, vec):
        return _Mat([[x - vec[j] for j, x in enumerate(row)]
                     for row in self.rows])

    def __matmul__(self, other):
        b = other.rows if isinstance(other, _Mat) else other
        k, n = len(b), len(b[0])
        return _Mat([[sum(row[i] * b[i][j] for i in range(k))
                      for j in range(n)] for row in self.rows])


def _qr_columns(a_rows):
    """Gram-Schmidt orthonormalization of the columns; returns Q as rows."""
    h = len(a_rows)
    cols = [[a_rows[i][j] for i in range(h)] for j in range(h)]
    q = []
    for c in cols:
        v = list(c)
        for u in q:
            d = sum(x * y for x, y in zip(v, u))
            v = [x - d * y for x, y in zip(v, u)]
        nrm = sum(x * x for x in v) ** 0.5
        q.append([x / nrm for x in v])
    return [[q[j][i] for j in range(h)] for i in range(h)]


def _nn_mean_decode(h_rec, ref_h, ref_tok, victim_tok):
    """Pure-python mirror of common.nn_mean_decode: per-token reference
    means, nearest mean per recovered row, top-1 %."""
    rec_rows = h_rec.rows if isinstance(h_rec, _Mat) else h_rec
    ref_rows = ref_h.rows if isinstance(ref_h, _Mat) else ref_h
    vocab = max(max(ref_tok), max(victim_tok)) + 1
    d = len(ref_rows[0])
    means = []
    for v in range(vocab):
        sel = [r for r, t in zip(ref_rows, ref_tok) if t == v]
        means.append([sum(r[j] for r in sel) / len(sel) for j in range(d)]
                     if sel else [0.0] * d)
    correct = 0
    for x, t in zip(rec_rows, victim_tok):
        ds = [sum((x[j] - m[j]) ** 2 for j in range(d)) for m in means]
        correct += int(ds.index(min(ds)) == t)
    return round(100.0 * correct / max(1, len(rec_rows)), 4)


def _fake_world(hidden=8, vocab=8, n_public=160, n_victim=48,
                structured=True, seed=0, **_ignored):
    """Pure-python stand-in for synthetic.make_toy_world (torch-only):
    same structured=True semantics — tokens clustered near random unit
    directions (4.0 signal vs 1.0 noise), per-epoch orthogonal wire keys."""
    rng = random.Random(seed)
    dirs = []
    for _ in range(vocab):
        v = [rng.gauss(0, 1) for _ in range(hidden)]
        nrm = sum(x * x for x in v) ** 0.5
        dirs.append([x / nrm for x in v])

    def rows(n_):
        out, toks = [], []
        for _ in range(n_):
            t = rng.randrange(vocab)
            toks.append(t)
            base = ([4.0 * dirs[t][j] for j in range(hidden)]
                    if structured else [0.0] * hidden)
            out.append([base[j] + rng.gauss(0, 1) for j in range(hidden)])
        return out, toks

    public_h, public_tok = rows(n_public)
    victim_h, victim_tok = rows(n_victim)
    ws = []
    for ep in range(2):
        g = random.Random(1000 + ep)
        ws.append(_qr_columns([[g.gauss(0, 1) for _ in range(hidden)]
                               for _ in range(hidden)]))
    return {"hidden": hidden, "vocab": vocab, "n_epochs": 2,
            "public_h": _Mat(public_h), "public_tok": public_tok,
            "victim_h": _Mat(victim_h), "victim_tok": victim_tok,
            "Ws": ws, "wire": lambda h, t: h @ ws[t]}


def _install_fakes(monkeypatch, structured=True,
                   quality=lambda iseed: "good"):
    """Substitute the torch-only pieces at ica_bss's own namespace seams.
    quality(iseed): 'good' = seeded orthogonal unmixing, 'bad' = rank-1
    replicate (destroys all but one source dim), 'raise' = RuntimeError."""
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    holder = {}

    def fake_world(**kwargs):
        w = _fake_world(structured=structured, seed=kwargs.get("seed", 0))
        holder["world"] = w
        return w

    def fake_fastica(x, max_iter=200, seed=0):
        iseed = round((seed - 1234) / 71)
        q = quality(iseed)
        if q == "raise":
            raise RuntimeError("injected: ICA failed to converge")
        rows = x.rows
        n, d = len(rows), len(rows[0])
        mu = [sum(r[j] for r in rows) / n for j in range(d)]
        g = random.Random(seed)
        base = _qr_columns([[g.gauss(0, 1) for _ in range(d)]
                            for _ in range(d)])
        unmixing = base if q == "good" else [[base[i][0]] * d
                                             for i in range(d)]
        return {"sources": x, "unmixing": unmixing, "mean": mu,
                "n_iter": 3, "converged": True}

    monkeypatch.setattr("attacker.synthetic.make_toy_world", fake_world)
    monkeypatch.setattr(ica, "fastica", fake_fastica)
    monkeypatch.setattr(ica, "nn_mean_decode", _nn_mean_decode)
    return holder


def _parse(*argv):
    args = ica.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def _run(tmp_path, monkeypatch, name="ica.json", *argv, structured=True,
         quality=None):
    holder = _install_fakes(monkeypatch, structured=structured,
                            quality=quality or (lambda iseed: "good"))
    out = tmp_path / name
    args = _parse("--toy", "--hidden", "8", "--seeds", "0",
                  "--output", str(out), *argv)
    assert ica.run(args) == 0
    return holder, json.loads(out.read_text()), \
        [json.loads(line) for line in
         (tmp_path / (name + ".jsonl")).read_text().splitlines()]


def test_contract_and_registry():
    assert ica.EXPERIMENT_ID == "ica_bss"
    assert ica.MODES == ("training", "inference")
    assert ica.REQUIRES_LABELS is False
    attacks.validate_contract(ica, "ica-bss")
    assert attacks.ATTACKS.get("ica-bss") == "attacker.attacks.ica_bss"
    assert attacks.load("ica-bss") is ica


def test_parser_defaults_and_choices():
    args = _parse()
    assert args.rows_per_epoch == 4096
    assert args.ica_seeds == [0, 1, 2]
    assert args.max_iter == 200
    assert args.hidden == 64 and args.seed == 1234
    args = _parse("--rows-per-epoch", "128", "--ica-seeds", "3", "4",
                  "--max-iter", "50")
    assert args.rows_per_epoch == 128 and args.ica_seeds == [3, 4]
    assert args.max_iter == 50
    with pytest.raises(SystemExit):
        ica.build_parser().parse_args(["--help"])


def test_requires_torch_guard(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        ica.run(_parse("--toy", "--quick", "--output",
                       str(tmp_path / "ica.json")))


def test_real_capture_path_fails_closed_and_points_at_toy(tmp_path,
                                                          monkeypatch):
    """Without --toy the module refuses: real captures must be grouped per
    epoch by the caller (attacker.captures.load_epoch_rows). The guard is
    pure python; bypass the torch gate with a sentinel to reach it."""
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    with pytest.raises(SystemExit, match="--toy"):
        ica.run(_parse("--output", str(tmp_path / "ica.json")))


def test_clustered_world_decodes_far_above_the_label_free_band(tmp_path,
                                                               monkeypatch):
    holder, art, journal = _run(tmp_path, monkeypatch, "ica.json",
                                "--ica-seeds", "1")
    assert art["schema"] == "dtraining.attacker.ica_bss.v1"
    assert art["config"]["attack"] == "ica_bss"
    assert art["config"]["toy"] is True and art["config"]["hidden"] == 8
    assert art["config"]["ica_seeds"] == [1]
    assert len(art["results"]) == 2
    assert len(journal) == 2
    for rec in art["results"]:
        assert rec["experiment"] == "ica_bss" and rec["seed"] == 0
        assert rec["ica_seed_best"] == 1 and rec["converged"] is True
        assert rec["n_restarts"] == 1 and rec["n_iter"] == 3
        assert rec["ica_top1_best"] > 80.0
        assert rec["labelfree_band_top1"] < 25.0
        assert rec["ica_top1_best"] > rec["labelfree_band_top1"] + 50.0
    (summary,) = art["summary"]
    assert summary["n_cells"] == 2
    assert summary["ica_top1_mean"] > 80.0
    assert summary["band_mean"] < 25.0


def test_decode_matches_plain_space_under_orthogonal_unmixing(tmp_path,
                                                              monkeypatch):
    """PIN (equal-transform invariance): ref and victim rows are mapped
    through the SAME unmixing (ica_bss.py:108-111), so an orthogonal
    unmixing must reproduce the plain-space decode exactly — the toy
    metric is insensitive to ICA recovery quality by construction."""
    holder, art, _ = _run(tmp_path, monkeypatch, "pin.json", "--ica-seeds",
                          "1")
    w = holder["world"]
    plain = _nn_mean_decode(w["victim_h"], w["public_h"],
                            w["public_tok"], w["victim_tok"])
    for rec in art["results"]:
        assert rec["ica_top1_best"] == pytest.approx(plain, abs=1e-6)


def test_best_of_restarts_selects_the_good_unmixing(tmp_path, monkeypatch):
    quality = lambda iseed: "bad" if iseed == 0 else "good"
    _, art_bad, _ = _run(tmp_path, monkeypatch, "bad.json", "--ica-seeds",
                         "0", quality=quality)
    _, art_good, _ = _run(tmp_path, monkeypatch, "good.json", "--ica-seeds",
                          "1", quality=quality)
    _, art_both, _ = _run(tmp_path, monkeypatch, "both.json", "--ica-seeds",
                          "0", "1", quality=quality)
    for bad, good, both in zip(art_bad["results"], art_good["results"],
                               art_both["results"]):
        assert bad["ica_seed_best"] == 0 and good["ica_seed_best"] == 1
        assert good["ica_top1_best"] > bad["ica_top1_best"]
        assert both["ica_seed_best"] == 1
        assert both["ica_top1_best"] == good["ica_top1_best"]
        assert both["n_restarts"] == 2


def test_failed_restart_is_journaled_and_the_cell_continues(tmp_path,
                                                            monkeypatch):
    def quality(iseed):
        return "raise" if iseed == 0 else "good"
    _, art, journal = _run(tmp_path, monkeypatch, "err.json", "--ica-seeds",
                           "0", "1", quality=quality)
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 2
    for e in errs:
        assert e["experiment"] == "ica_bss"
        assert e["ica_seed"] == 0 and "epoch" in e and "seed" in e
        assert "ICA failed to converge" in e["error"]
    assert len(art["results"]) == 2
    assert all(r["ica_seed_best"] == 1 for r in art["results"])


def test_all_restarts_failing_journals_the_cell_fail_closed(tmp_path,
                                                            monkeypatch):
    _, art, journal = _run(tmp_path, monkeypatch, "allfail.json",
                           "--ica-seeds", "0", "1",
                           quality=lambda iseed: "raise")
    assert art["results"] == []
    cell_errs = [j for j in journal
                 if "error" in j and "ica_seed" not in j]
    assert len(cell_errs) == 2
    for e in cell_errs:
        assert e["experiment"] == "ica_bss"
        assert "epoch" in e and "seed" in e
        assert e["error"] == "error: all ICA restarts failed"


def test_unstructured_gaussian_world_sits_at_the_floor(tmp_path, monkeypatch):
    """Null world (no token structure): nothing to recover — decode and
    band both sit at the 1/vocab floor. This is the higher-order analog
    of the Gaussian-mixture regime where ICA is unidentifiable."""
    _, art, _ = _run(tmp_path, monkeypatch, "null.json", "--ica-seeds", "1",
                     structured=False)
    chance = 100.0 / 8
    for rec in art["results"]:
        assert rec["ica_top1_best"] <= 3 * chance
        assert rec["labelfree_band_top1"] <= 3 * chance


def test_quick_flag_shrinks_the_grid(tmp_path, monkeypatch):
    _, art, _ = _run(tmp_path, monkeypatch, "quick.json", "--quick")
    cfg = art["config"]
    assert cfg["rows_per_epoch"] == 1024 and cfg["seeds"] == [0]
    assert cfg["ica_seeds"] == [0, 1]
    assert all(r["n_restarts"] == 2 for r in art["results"])


def test_toy_tier_is_deterministic(tmp_path, monkeypatch):
    _, a, _ = _run(tmp_path, monkeypatch, "d1.json", "--ica-seeds", "0", "1")
    _, b, _ = _run(tmp_path, monkeypatch, "d2.json", "--ica-seeds", "0", "1")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
