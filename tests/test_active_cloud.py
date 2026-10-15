"""active-cloud attack module (attacker/attacks/active_cloud.py): the
abstract live-driver protocol (stays abstract), the offline planner /
synthetic harness / observation analyzer driven torch-free through a
minimal pure-python matrix kit substituted at the module's own namespace
seams, the crafted-vs-honest injection contrast as a known answer, and
the full run() loop end to end.

The heavy numerics (torch QR / lstsq-polar solve / NN decode) are
torch-gated; the kit below implements the same operations with exact,
seeded float arithmetic so the module's real control flow is what gets
exercised. Nothing here re-derives the torch tiers' claims.
"""

import json
import random

import pytest

from attacker import attacks
from attacker.attacks import active_cloud as ac


class _Scalar:
    def __init__(self, v):
        self.v = float(v)

    def __truediv__(self, other):
        return _Scalar(self.v / other.v)

    def item(self):
        return self.v


class _Vec:
    def __init__(self, xs):
        self.xs = [float(x) for x in xs]

    def double(self):
        return self

    def float(self):
        return self

    def __mul__(self, s):
        return _Vec([x * s for x in self.xs])

    __rmul__ = __mul__

    def __matmul__(self, mat):
        b = mat.rows if isinstance(mat, _Mat) else mat
        return _Vec([sum(self.xs[i] * b[i][j] for i in range(len(b)))
                     for j in range(len(b[0]))])


class _Mat:
    def __init__(self, rows):
        self.rows = [list(map(float, r)) for r in rows]

    def double(self):
        return self

    def float(self):
        return self

    def __getitem__(self, i):
        if isinstance(i, slice):
            return _Mat(self.rows[i])
        return _Vec(self.rows[i])

    def __matmul__(self, other):
        b = other.rows if isinstance(other, _Mat) else other
        k, n = len(b), len(b[0])
        return _Mat([[sum(row[i] * b[i][j] for i in range(k))
                      for j in range(n)] for row in self.rows])

    def __add__(self, other):
        if isinstance(other, _Vec):
            return _Mat([[x + other.xs[j] for j, x in enumerate(row)]
                         for row in self.rows])
        return _Mat([[a + b for a, b in zip(ra, rb)]
                     for ra, rb in zip(self.rows, other.rows)])

    def __sub__(self, other):
        if isinstance(other, _Vec):
            return _Mat([[x - other.xs[j] for j, x in enumerate(row)]
                         for row in self.rows])
        if len(other.rows) == 1:
            return _Mat([[x - other.rows[0][j] for j, x in enumerate(row)]
                         for row in self.rows])
        return _Mat([[a - b for a, b in zip(ra, rb)]
                     for ra, rb in zip(self.rows, other.rows)])

    def __mul__(self, other):
        if isinstance(other, _Vec):
            return _Mat([[x * other.xs[j] for j, x in enumerate(row)]
                         for row in self.rows])
        return _Mat([[x * other for x in row] for row in self.rows])

    __rmul__ = __mul__

    def __truediv__(self, s):
        return _Mat([[x / s for x in row] for row in self.rows])

    def mean(self, dim=0, keepdim=False):
        assert dim == 0
        n = len(self.rows)
        col = [sum(row[j] for row in self.rows) / n
               for j in range(len(self.rows[0]))]
        return _Mat([col]) if keepdim else _Vec(col)

    def norm(self):
        return _Scalar(sum(x * x for row in self.rows for x in row) ** 0.5)

    def T(self):
        return _Mat(list(map(list, zip(*self.rows))))


class _Gen:
    def manual_seed(self, seed):
        self._rng = random.Random(seed)
        return self


def _qr(m):
    """Gram-Schmidt QR on the columns of _Mat m; R diagonal positive."""
    rows = m.rows
    h, w = len(rows), len(rows[0])
    cols = [[rows[i][j] for i in range(h)] for j in range(w)]
    q, r_diag = [], []
    for c in cols:
        v = list(c)
        for u in q:
            d = sum(x * y for x, y in zip(v, u))
            v = [x - d * y for x, y in zip(v, u)]
        nrm = sum(x * x for x in v) ** 0.5
        r_diag.append(nrm)
        q.append([x / nrm for x in v])
    qm = _Mat([[q[j][i] for j in range(w)] for i in range(h)])
    rm = _Mat([[r_diag[j] if i == j else 0.0 for j in range(w)]
               for i in range(w)])
    return qm, rm


class _TorchStub:
    """The slice of the torch API active_cloud uses, in pure python."""
    float64 = "float64"

    @staticmethod
    def Generator():
        return _Gen()

    @staticmethod
    def randn(rows, cols, generator=None, dtype=None):
        return _Mat([[generator._rng.gauss(0, 1) for _ in range(cols)]
                     for _ in range(rows)])

    class linalg:
        @staticmethod
        def qr(m):
            return _qr(m)

    @staticmethod
    def sign(v):
        return _Vec([1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
                     for x in v.xs])

    @staticmethod
    def diagonal(m):
        return _Vec([m.rows[i][i] for i in range(len(m.rows))])

    @staticmethod
    def stack(vecs):
        return _Mat([v.xs if isinstance(v, _Vec) else v for v in vecs])


def _inverse(a):
    n = len(a)
    m = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            return None
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [x / d for x in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [row[n:] for row in m]


def _matmul(a, b):
    return [[sum(row[i] * b[i][j] for i in range(len(b)))
             for j in range(len(b[0]))] for row in a]


def _transpose(a):
    return [list(col) for col in zip(*a)]


def _lstsq(d, o):
    """Independent normal-equations least squares (ridge-regularized)."""
    h = len(d[0])
    dtd = _matmul(_transpose(d), d)
    dtd = [[x + (1e-9 if i == j else 0.0) for j, x in enumerate(row)]
           for i, row in enumerate(dtd)]
    inv = _inverse(dtd)
    if inv is None:
        return None
    return _matmul(_matmul(inv, _transpose(d)), o)


def _pure_solve_w(deltas, obs):
    """(w_hat, tag) with solve_w's never-raises contract: ridge-regularized
    normal equations + Newton polar iteration, all in fp64 lists."""
    d = deltas.rows
    o = obs.rows if isinstance(obs, _Mat) else obs
    x = _lstsq(d, o)
    if x is None:
        return None, "error: singular normal equations"
    p = [[v + (1e-9 if i == j else 0.0) for j, v in enumerate(row)]
         for i, row in enumerate(x)]
    for _ in range(50):
        pinv = _inverse(_transpose(p))
        if pinv is None:
            return None, "error: polar iteration hit a singular factor"
        p = [[(a + b) / 2.0 for a, b in zip(ra, rb)]
             for ra, rb in zip(p, pinv)]
    return _Mat(p), "pure-lstsq+newton-polar"


def _nn_mean_decode(h_rec, ref_h, ref_tok, victim_tok):
    """Mirror of common.nn_mean_decode over kit matrices."""
    rec = h_rec.rows if isinstance(h_rec, _Mat) else h_rec
    ref = ref_h.rows if isinstance(ref_h, _Mat) else ref_h
    vocab = max(max(ref_tok), max(victim_tok)) + 1
    d = len(ref[0])
    means = []
    for v in range(vocab):
        sel = [r for r, t in zip(ref, ref_tok) if t == v]
        means.append([sum(r[j] for r in sel) / len(sel) for j in range(d)]
                     if sel else [0.0] * d)
    correct = 0
    for x, t in zip(rec, victim_tok):
        ds = [sum((x[j] - m[j]) ** 2 for j in range(d)) for m in means]
        correct += int(ds.index(min(ds)) == t)
    return round(100.0 * correct / max(1, len(rec)), 4)


def _orthogonal(seed, h=4):
    g = random.Random(seed)
    q, _ = _qr(_Mat([[g.gauss(0, 1) for _ in range(h)] for _ in range(h)]))
    return q


def _fake_world(hidden=4, vocab=4, n_public=48, n_victim=12, seed=0, **_):
    """Stand-in for synthetic.make_toy_world: structured clusters, one
    orthogonal epoch key, wire = h @ W."""
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
            out.append([4.0 * dirs[t][j] + rng.gauss(0, 1)
                        for j in range(hidden)])
        return out, toks

    public_h, public_tok = rows(n_public)
    victim_h, victim_tok = rows(n_victim)
    w = _orthogonal(1000 + seed, hidden)
    return {"hidden": hidden, "vocab": vocab, "n_epochs": 1,
            "public_h": _Mat(public_h), "public_tok": public_tok,
            "victim_h": _Mat(victim_h), "victim_tok": victim_tok,
            "Ws": [w], "wire": lambda h, t: h @ w}


def _install_torch_stub(monkeypatch):
    monkeypatch.setattr(ac, "torch", _TorchStub)


def _parse(*argv):
    args = ac.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_contract_and_registry():
    assert ac.EXPERIMENT_ID == "active_cloud"
    assert ac.MODES == ("training", "inference")
    assert ac.REQUIRES_LABELS is False
    attacks.validate_contract(ac, "active-cloud")
    assert attacks.ATTACKS.get("active-cloud") == "attacker.attacks.active_cloud"
    assert attacks.load("active-cloud") is ac


def test_parser_defaults():
    args = _parse()
    assert args.probes == 128
    assert args.magnitudes == [0.01, 0.1, 1.0]
    assert args.steer_budget == 0.1


def test_abstract_driver_stays_abstract():
    """The live driver is a documented integration point: the protocol
    base class must refuse to DO anything by itself (lines 95-101)."""
    harness = ac.ActiveCloudHarness()
    with pytest.raises(NotImplementedError):
        harness.inject([0.0])
    with pytest.raises(NotImplementedError):
        harness.observe()
    with pytest.raises(NotImplementedError):
        harness.epoch()
    assert harness.ground_truth_W() is None
    assert harness.close() is None


def test_planner_probe_schedule_is_deterministic_orthonormal_and_finite(
        monkeypatch):
    _install_torch_stub(monkeypatch)
    p1 = ac.AttackPlanner(hidden=4, magnitude=0.5, n_probes=6, seed=7)
    p2 = ac.AttackPlanner(hidden=4, magnitude=0.5, n_probes=6, seed=7)
    basis = p1.probe_basis.rows
    for row in basis:
        assert sum(x * x for x in row) == pytest.approx(1.0, abs=1e-12)
    for i in range(4):
        for j in range(i + 1, 4):
            assert sum(a * b for a, b in zip(basis[i], basis[j])) == \
                pytest.approx(0.0, abs=1e-12)
    assert p1.probe_basis.rows == p2.probe_basis.rows
    probes = []
    while True:
        d = p1.next_probe()
        if d is None:
            break
        probes.append(d.xs)
    assert len(probes) == 6
    assert probes[0] == [0.5 * x for x in basis[0]]
    assert probes[4] == [0.5 * x for x in basis[0]]
    p3 = ac.AttackPlanner(hidden=4, magnitude=0.5, n_probes=6, seed=8)
    assert p3.probe_basis.rows != basis


def test_synthetic_harness_crafted_vs_honest_is_a_known_answer(monkeypatch):
    """The ONLY difference between an honest and a crafted round is the
    injected delta: observe() after inject(delta) must exceed the honest
    observation by exactly delta @ resp @ W (same counter, same base
    draw)."""
    _install_torch_stub(monkeypatch)
    world = _fake_world(seed=3)
    w = world["Ws"][0]

    honest = ac.SyntheticHarness(world, epoch=0, seed=11)
    obs_h0 = honest.observe()
    obs_h1 = honest.observe()

    crafted = ac.SyntheticHarness(world, epoch=0, seed=11)
    delta = _Vec([0.25, -0.5, 1.0, 0.75])
    crafted.inject(delta)
    obs_c0 = crafted.observe()

    assert crafted.epoch() == 0
    assert crafted.ground_truth_W().rows == w.rows
    assert obs_h0.rows == obs_h0.rows
    diff = (obs_c0 - obs_h0).rows
    expect = (delta @ crafted._resp @ w).xs
    for drow in diff:
        for got, want in zip(drow, expect):
            assert got == pytest.approx(want, abs=1e-9)
    assert any(abs(x) > 1e-6 for row in diff for x in row)
    obs_c1 = crafted.observe()
    assert obs_c1.rows == obs_h1.rows


def test_analyzer_recovers_an_exact_orthogonal_response_map(monkeypatch):
    """Known answer: probes come in ± pairs (zero-mean) and every
    observation carries the SAME constant offset c (the harness base
    term). Because the analyzer de-means the observations
    (active_cloud.py:160-161), the offset cancels and the polar(lstsq)
    estimate must BE the planted orthogonal map M — if the de-meaning
    were dropped, the offset would bias the solve away from M."""
    _install_torch_stub(monkeypatch)
    monkeypatch.setattr(ac, "solve_w", _pure_solve_w)
    m = _orthogonal(42)
    offset = [0.3, -0.1, 0.2, 0.5]
    planner = ac.AttackPlanner(hidden=4, magnitude=1.0, n_probes=8, seed=5)
    deltas = [_Vec(r) for r in planner.probe_basis.rows[:4]] + \
        [_Vec([-x for x in r]) for r in planner.probe_basis.rows[:4]]
    for d in deltas:
        planner.record(d, _Mat([[(d @ m).xs[j] + offset[j]
                                 for j in range(4)]]))

    class MapHarness(ac.ActiveCloudHarness):
        def ground_truth_W(self):
            return m

    w_hat, diag = ac.ObservationAnalyzer().estimate(planner, MapHarness())
    assert w_hat is not None
    assert diag["n_pairs"] == 8
    assert diag["solver"] == "pure-lstsq+newton-polar"
    for ra, rb in zip(w_hat.rows, m.rows):
        for a, b in zip(ra, rb):
            assert a == pytest.approx(b, abs=1e-6)
    assert diag["w_rel_err"] == pytest.approx(0.0, abs=1e-6)


def test_analyzer_fails_closed_with_no_pairs(monkeypatch):
    _install_torch_stub(monkeypatch)
    monkeypatch.setattr(ac, "solve_w", _pure_solve_w)
    planner = ac.AttackPlanner(hidden=4, magnitude=1.0, n_probes=2, seed=5)
    w_hat, diag = ac.ObservationAnalyzer().estimate(planner)
    assert w_hat is None
    assert diag == "error: no probe/observation pairs recorded" or \
        diag.startswith("error:")


def test_analyzer_returns_the_polar_factor_of_the_lstsq_system(monkeypatch):
    """Assembly check, recomputed independently from the recorded pairs:
    w_hat must be orthogonal AND must be the orthogonal polar factor of
    X = lstsq(deltas, de-meaned obs) — i.e. S = w_hat^T X is symmetric
    and X = w_hat @ S. Any wiring error in stack/de-mean/solve breaks
    this identity."""
    _install_torch_stub(monkeypatch)
    monkeypatch.setattr(ac, "solve_w", _pure_solve_w)
    g = random.Random(9)
    m = _Mat([[g.gauss(0, 1) for _ in range(4)] for _ in range(4)])
    planner = ac.AttackPlanner(hidden=4, magnitude=1.0, n_probes=8, seed=1)
    for _ in range(8):
        delta = planner.next_probe()
        planner.record(delta, _Mat([(delta @ m).xs]))
    w_hat, diag = ac.ObservationAnalyzer().estimate(planner)
    assert w_hat is not None and diag["n_pairs"] == 8
    assert "w_rel_err" not in diag
    wt = list(map(list, zip(*w_hat.rows)))
    ident = _matmul(w_hat.rows, wt)
    for i in range(4):
        for j in range(4):
            assert ident[i][j] == pytest.approx(1.0 if i == j else 0.0,
                                                abs=1e-6)
    d = [v.xs for v, _ in planner.pairs]
    o = [v.xs for _, mat in planner.pairs for v in [mat.mean(0)]]
    omean = [sum(r[j] for r in o) / len(o) for j in range(4)]
    o = [[x - omean[j] for j, x in enumerate(row)] for row in o]
    x = _lstsq(d, o)
    s = _matmul(wt, x)
    for i in range(4):
        for j in range(4):
            assert s[i][j] == pytest.approx(s[j][i], abs=1e-6)
    recon = _matmul(w_hat.rows, s)
    for ra, rb in zip(recon, x):
        for a, b in zip(ra, rb):
            assert a == pytest.approx(b, abs=1e-5)


def _install_run_fakes(monkeypatch, seed=0):
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    _install_torch_stub(monkeypatch)
    monkeypatch.setattr("attacker.synthetic.make_toy_world",
                        lambda **kw: _fake_world(seed=kw.get("seed", seed)))
    monkeypatch.setattr(ac, "solve_w", _pure_solve_w)

    def fake_recovery(h_star_wire, w_hat, ref_h, ref_tok, victim_tok):
        wt = _Mat(list(map(list, zip(*w_hat.rows))))
        return _nn_mean_decode(h_star_wire @ wt, ref_h, ref_tok, victim_tok)

    monkeypatch.setattr(ac, "recovery_with_what_nn", fake_recovery)


def test_run_end_to_end_artifact_journal_and_cells(tmp_path, monkeypatch):
    _install_run_fakes(monkeypatch)
    out = tmp_path / "ac.json"
    assert ac.run(_parse("--toy", "--quick", "--hidden", "4",
                         "--output", str(out))) == 0
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.active_cloud.v1"
    cfg = art["config"]
    assert cfg["attack"] == "active_cloud" and cfg["toy"] is True
    assert cfg["probes"] == 16
    assert cfg["seeds"] == [0] and cfg["magnitudes"] == [0.01]
    (rec,) = art["results"]
    assert rec["experiment"] == "active_cloud"
    assert rec["mode"] == "training"
    assert rec["magnitude"] == 0.01 and rec["seed"] == 0
    assert rec["n_pairs"] == 16
    assert rec["solver"] == "pure-lstsq+newton-polar"
    assert 0.0 <= rec["w_rel_err"] <= 2.0
    assert 0.0 <= rec["victim_top1"] <= 100.0
    journal = [json.loads(line) for line in
               (tmp_path / "ac.json.jsonl").read_text().splitlines()]
    assert journal == art["results"]


def test_run_journals_cell_errors_fail_closed(tmp_path, monkeypatch):
    _install_run_fakes(monkeypatch)

    def boom(deltas, obs):
        raise RuntimeError("injected analyzer failure")

    monkeypatch.setattr(ac, "solve_w", boom)
    out = tmp_path / "ac_err.json"
    assert ac.run(_parse("--toy", "--quick", "--hidden", "4",
                         "--output", str(out))) == 0
    art = json.loads(out.read_text())
    assert art["results"] == []
    journal = [json.loads(line) for line in
               (tmp_path / "ac_err.json.jsonl").read_text().splitlines()]
    (err,) = journal
    assert err["experiment"] == "active_cloud"
    assert "injected analyzer failure" in err["error"]
    assert err["magnitude"] == 0.01 and err["seed"] == 0


def test_run_is_deterministic(tmp_path, monkeypatch):
    _install_run_fakes(monkeypatch)
    results = []
    for name in ("a.json", "b.json"):
        assert ac.run(_parse("--toy", "--quick", "--hidden", "4",
                             "--output", str(tmp_path / name))) == 0
        results.append(json.loads((tmp_path / name).read_text())["results"])
    assert results[0] == results[1]


def test_requires_torch_guard(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        ac.run(_parse("--toy", "--quick", "--output",
                      str(tmp_path / "x.json")))


def test_live_driver_is_a_documented_integration_point(tmp_path, monkeypatch):
    """Without --toy, run() refuses and points at the live-driver
    integration point (the protocol above stays abstract for a reason)."""
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    with pytest.raises(SystemExit, match="integration point"):
        ac.run(_parse("--output", str(tmp_path / "x.json")))
