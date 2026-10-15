"""alignment-search attack module (attacker/attacks/alignment_search.py):
contract, parser, runtime-args builder, fail-closed guards, and the
realignment behavior the module is ABOUT — edit-distance/DTW recovery of
known aligned pairs from a synthetic rotated+jittered scaffold, vs a
passive positional baseline that must miss them.

The module's realignment core is deliberately factored torch-free
(attacker/dtw.py; the module docstring calls it the "PURE-PYTHON core,
torch-free testable"); these tests drive it through the module's own
imported names (als.dtw_distance / als.edit_alignment) and drive
als.realign_pairs itself with a minimal pure-python stand-in for the
torch tensor/quantile calls. The ICP solve loop (attack_sessions), the
toy driver (run_toy) and the real-capture driver (run_real body) are
torch-gated and stay uncovered here.
"""

import math
import random

import pytest

from attacker import attacks
from attacker.attacks import alignment_search as als


def _parse(*argv):
    args = als.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def test_contract_and_registry():
    assert als.EXPERIMENT_ID == "er8_alignment_search"
    assert als.MODES == ("training", "inference")
    assert "known scaffold prefix" in als.REQUIRES_LABELS
    attacks.validate_contract(als, "alignment-search")
    assert attacks.ATTACKS.get("alignment-search") == \
        "attacker.attacks.alignment_search"
    assert attacks.load("alignment-search") is als


def test_parser_defaults_and_aligner_choices():
    args = _parse()
    assert args.sessions == 8 and args.prefix_len == 32
    assert args.aligner == "edit"
    assert args.rounds == 6 and args.match_quantile == 0.5
    assert args.solve_rows == 16384 and args.victim_rows == 1024
    assert args.capture_dir is None and args.run_json is None
    with pytest.raises(SystemExit):
        als.build_parser().parse_args(["--aligner", "nope"])
    with pytest.raises(SystemExit):
        als.build_parser().parse_args(["--help"])


def test_make_runtime_args_known_answer():
    cfg = {"model": "m0", "dtype": "fp32", "seq_len": 16}
    a = als._make_runtime_args(cfg, "cpu")
    assert (a.model, a.dtype, a.device, a.seq_len) == ("m0", "fp32", "cpu", 16)
    assert a.toy is False and a.attn_impl == "sdpa"
    assert (a.max_pairs, a.epochs, a.batch_size) == (20000, 30, 256)
    b = als._make_runtime_args(cfg, "cuda", max_pairs=10, epochs=2,
                               batch_size=4)
    assert (b.max_pairs, b.epochs, b.batch_size) == (10, 2, 4)


def test_requires_torch_guard(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        als.run(_parse("--toy", "--quick", "--output",
                       str(tmp_path / "er8.json")))


def test_real_mode_fails_at_the_split_training_boundary(tmp_path,
                                                        monkeypatch):
    """run_real imports the split-training trainer stack (absent from this
    checkout); that integration boundary must surface descriptively."""
    monkeypatch.setattr("attacker.attacks.common.torch", object())
    with pytest.raises(ModuleNotFoundError, match="split_trainer"):
        als.run(_parse("--capture-dir", str(tmp_path)))


HIDDEN = 8
N_ROWS = 24


def _sign_perm(seed):
    rng = random.Random(seed)
    perm = list(range(HIDDEN))
    rng.shuffle(perm)
    signs = [rng.choice([-1.0, 1.0]) for _ in range(HIDDEN)]
    return perm, signs


def _apply_q(row, perm, signs):
    return [signs[j] * row[perm[j]] for j in range(HIDDEN)]


def _unapply_q(row, perm, signs):
    out = [0.0] * HIDDEN
    for j in range(HIDDEN):
        out[perm[j]] = signs[j] * row[j]
    return out


def _make_world(seed=0, n_pads=3, noise=1e-3, permute_slots=True):
    """Returns (assumed_rows, wire_rows_derot_noisy, true_pairs).
    true_pairs: (assumed_i, wire_j) for every non-pad wire row. With
    permute_slots=False the jitter is pads-only (order preserved) — the
    insert/delete mechanism the edit aligner is documented to match."""
    rng = random.Random(seed)
    dirs = []
    for _ in range(N_ROWS):
        v = [rng.gauss(0, 1) for _ in range(HIDDEN)]
        nrm = sum(x * x for x in v) ** 0.5
        dirs.append([4.0 * x / nrm for x in v])
    perm, signs = _sign_perm(seed + 1)
    tagged = [(i, _apply_q(r, perm, signs)) for i, r in enumerate(dirs)]
    pads = sorted(rng.sample(range(N_ROWS + n_pads), n_pads))
    for pos in pads:
        pad = [rng.gauss(0, 1) for _ in range(HIDDEN)]
        tagged.insert(min(pos, len(tagged)), (None, pad))
    if permute_slots:
        q = max(1, len(tagged) // 4)
        slots = ([tagged[i * q:(i + 1) * q] for i in range(3)]
                 + [tagged[3 * q:]])
        order = rng.sample(range(4), 4)
        tagged = [t for i in order for t in slots[i]]
    true_pairs = [(i, j) for j, (i, _) in enumerate(tagged)
                  if i is not None]
    derot = [_unapply_q(r, perm, signs) for _, r in tagged]
    noisy = [[x + rng.gauss(0, noise) for x in r] for r in derot]
    return [list(r) for r in dirs], noisy, true_pairs


def _positional_hits(wire_len, true_pairs):
    """The passive-alignment baseline E-R3 actually validated: row i of the
    wire is assumed to be prefix position i."""
    true_pos = {j: i for i, j in true_pairs}
    return sum(1 for j in range(wire_len) if true_pos.get(j) == j)


def test_edit_alignment_recovers_every_pair_under_pad_jitter():
    """The mechanism edit alignment exists for: inserted pads, order
    preserved. Known answer: exact recovery of all 24 true pairs."""
    assumed, wire, true_pairs = _make_world(seed=0, permute_slots=False)
    pairs = als.edit_alignment(assumed, wire)
    assert set(pairs) == set(true_pairs)
    assert len(pairs) == N_ROWS


def test_passive_positional_baseline_misses_them():
    """The same world positionally paired (no realignment): the inserted
    pads shift every later row, so the baseline gets barely half — the
    gap the realignment attacker exists to close."""
    assumed, wire, true_pairs = _make_world(seed=0, permute_slots=False)
    realigned = len(set(als.edit_alignment(assumed, wire))
                    & set(true_pairs))
    passive = _positional_hits(len(wire), true_pairs)
    assert realigned == N_ROWS
    assert passive <= 0.6 * N_ROWS
    assert realigned - passive >= N_ROWS // 3


def test_slot_permutation_bounds_monotone_realignment():
    """Honest limit, pinned: edit/DTW alignment is MONOTONE, so the 4-slot
    permutation in jitter_rows cannot be fully undone — recovery lands
    strictly between the passive baseline (0 here) and full recall. The
    false matches it does make are all far (>= 3.4) while true matches
    are ~1e-3 — the distance gap the quantile filter exploits."""
    assumed, wire, true_pairs = _make_world(seed=0, permute_slots=True)
    pairs = als.edit_alignment(assumed, wire)
    true_rec = set(pairs) & set(true_pairs)
    false = set(pairs) - set(true_pairs)
    passive = _positional_hits(len(wire), true_pairs)
    assert passive < len(true_rec) < N_ROWS
    assert false
    for i, j in true_rec:
        assert math.dist(assumed[i], wire[j]) < 0.01
    for i, j in false:
        assert math.dist(assumed[i], wire[j]) > 3.0


def test_dtw_aligner_realigns_cheaper_than_the_diagonal():
    assumed, wire, true_pairs = _make_world(seed=0, permute_slots=False)
    window = max(len(assumed), len(wire)) // 2
    cost, path = als.dtw_distance(assumed, wire, window=window)
    diag = sum(math.dist(assumed[min(i, len(assumed) - 1)], wire[i])
               for i in range(len(wire))) / len(wire)
    assert 0.0 < cost < diag / 4
    assert path[0] == (0, 0) and path[-1] == (len(assumed) - 1,
                                              len(wire) - 1)
    assert all(b0 <= b1 and a0 <= a1
               for (a0, b0), (a1, b1) in zip(path, path[1:]))
    assert set(true_pairs) <= set(path)


def test_rotation_roundtrip_is_exact():
    """The scaffold rotation itself: h @ W de-rotated by W^T is the
    identity — the fixed point the ICP loop drives W_hat towards."""
    rng = random.Random(7)
    row = [rng.gauss(0, 1) for _ in range(HIDDEN)]
    perm, signs = _sign_perm(8)
    assert _unapply_q(_apply_q(row, perm, signs), perm, signs) == row


class _Row:
    def __init__(self, xs):
        self.xs = [float(x) for x in xs]

    def __sub__(self, other):
        return _Row([a - b for a, b in zip(self.xs, other.xs)])

    def __pow__(self, p):
        return _Row([x ** p for x in self.xs])

    def sum(self):
        return _Scalar(sum(self.xs))


class _Scalar:
    def __init__(self, v):
        self.v = float(v)

    def item(self):
        return self.v


class _Rows:
    """The two verbs realign_pairs uses: .tolist() and row indexing."""

    def __init__(self, rows):
        self.rows = [list(map(float, r)) for r in rows]

    def tolist(self):
        return [list(r) for r in self.rows]

    def __getitem__(self, i):
        return _Row(self.rows[i])


class _Distances:
    def __init__(self, xs):
        self.xs = [float(x) for x in xs]

    def tolist(self):
        return list(self.xs)


class _TorchStub:
    @staticmethod
    def tensor(xs):
        return _Distances(xs)

    @staticmethod
    def quantile(vec, q):
        """torch.quantile semantics: linear interpolation over sorted xs."""
        xs = sorted(vec.xs)
        pos = q * (len(xs) - 1)
        lo = int(math.floor(pos))
        hi = min(len(xs) - 1, lo + 1)
        frac = pos - lo
        return _Scalar(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def test_realign_pairs_quantile_filter_keeps_only_true_close_pairs(
        monkeypatch):
    monkeypatch.setattr(als, "torch", _TorchStub)
    assumed, wire, true_pairs = _make_world(seed=0, permute_slots=False)
    for aligner in ("edit", "dtw"):
        got = als.realign_pairs(_Rows(assumed), _Rows(wire), aligner, 0.5)
        assert got, aligner
        assert set(got) <= set(true_pairs)
        assert len(got) >= N_ROWS // 2


def test_realign_pairs_empty_alignment_returns_empty(monkeypatch):
    monkeypatch.setattr(als, "torch", _TorchStub)
    assert als.realign_pairs(_Rows([]), _Rows([]), "edit", 0.5) == []


def test_realign_pairs_deterministic(monkeypatch):
    monkeypatch.setattr(als, "torch", _TorchStub)
    assumed, wire, _ = _make_world(seed=0)
    a = als.realign_pairs(_Rows(assumed), _Rows(wire), "edit", 0.5)
    b = als.realign_pairs(_Rows(assumed), _Rows(wire), "edit", 0.5)
    assert a == b
