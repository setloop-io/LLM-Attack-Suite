"""latent-sensitivity compromise battery (attacker/attacks/latent_sensitivity.py):
contract, parser, the fail-closed torch guard, the majority-baseline CONTROL
computation (_majority) on planted vs null label worlds, and the chaff-retain
frame-retaining counter-policy against the discard-identified-chaff arm.

The battery arms train the local invariant_only probe with torch; no private
defense SDK is required. The control math also has a pure-Python stand-in for
dependency-free validation.
"""

import sys
import types

import pytest

from attacker import attacks
from attacker.attacks import latent_sensitivity as ls


def _parse(*argv):
    return ls.build_parser().parse_args(list(argv))


def test_contract_and_registry():
    assert ls.EXPERIMENT_ID == "latent_sensitivity"
    assert ls.MODES == ("training",)
    assert ls.REQUIRES_LABELS is True
    attacks.validate_contract(ls, "latent-sensitivity")
    assert attacks.ATTACKS.get("latent-sensitivity") == \
        "attacker.attacks.latent_sensitivity"
    assert attacks.load("latent-sensitivity") is ls


def test_parser_requires_bundle_and_output():
    with pytest.raises(SystemExit):
        _parse()
    with pytest.raises(SystemExit):
        _parse("--bundle", "b.pt")
    args = _parse("--bundle", "b.pt", "--output", "o.json")
    assert args.bundle == "b.pt" and args.output == "o.json"
    assert args.restarts == 2 and args.epochs == 20
    assert args.batch_size == 512 and args.lr == 1e-3
    assert args.evidence is False
    args = _parse("--bundle", "b.pt", "--output", "o.json", "--evidence")
    assert args.evidence is True
    with pytest.raises(SystemExit):
        ls.build_parser().parse_args(["--help"])


def test_run_fails_closed_without_torch(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        ls.run(_parse("--bundle", str(tmp_path / "b.pt"),
                      "--output", str(tmp_path / "o.json")))


class _BoolV:
    def __init__(self, bs):
        self.bs = [bool(b) for b in bs]

    def float(self):
        return _V([1.0 if b else 0.0 for b in self.bs])


class _V:
    """The 1-D tensor verbs _majority uses."""

    def __init__(self, xs):
        self.xs = list(xs)

    def reshape(self, *_):
        return self

    def __ge__(self, v):
        return _BoolV([x >= v for x in self.xs])

    def __getitem__(self, mask):
        return _V([x for x, b in zip(self.xs, mask.bs) if b])

    def __eq__(self, v):
        return _BoolV([x == v for x in self.xs])

    def float(self):
        return _V([float(x) for x in self.xs])

    def mean(self):
        return sum(self.xs) / max(1, len(self.xs))


class _Counts:
    def __init__(self, counts):
        self.counts = counts

    def argmax(self):
        return self.counts.index(max(self.counts))


def _bincount(v):
    counts = [0] * (max(v.xs) + 1 if v.xs else 0)
    for x in v.xs:
        counts[int(x)] += 1
    return _Counts(counts)


@pytest.fixture
def torch_stub(monkeypatch):
    stub = types.SimpleNamespace(bincount=_bincount)
    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


def test_majority_control_planted_world(torch_stub):
    """Planted: class 0 dominates train (7/10) and eval is all class 0 —
    the majority-class control scores a perfect 1.0."""
    train_y = _V([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    eval_y = _V([0, 0, 0, 0])
    assert ls._majority(train_y, eval_y) == 1.0


def test_majority_control_null_world_at_chance(torch_stub):
    """Null: balanced classes and a uniform eval — the control sits at
    exactly 1/C, the chance line the cells must beat."""
    train_y = _V([0, 1, 2, 3] * 25)
    eval_y = _V([0, 1, 2, 3] * 10)
    assert ls._majority(train_y, eval_y) == pytest.approx(0.25)


def test_majority_masks_padding_labels(torch_stub):
    """Padded rows (label -1 from the chaff arms) leave both denominators."""
    train_y = _V([0, 0, 1, -1, -1])
    eval_y = _V([1, 1, -1, 0])
    assert ls._majority(train_y, eval_y) == pytest.approx(1.0 / 3.0)


def _chaff_fixture(tmp_path, torch, name="bundle.pt", seq=8, dim=16, vocab=4,
                   train_blocks=32, eval_blocks=16, sigma=0.02, seed=0):
    """Synthetic bundle where the connectivity identifier marks exactly the
    class-signal rows as chaff.

    Each frame: seq-2 filler rows (class 0, one shared direction, tight
    noise -> highly Gram-connected, majority by construction) and 2 signal
    rows (classes 1..vocab-1, class-dependent norm scale on near-orthogonal
    directions -> least-connected, and decodable from gauge-invariant
    features). sigma blurs everything: at large sigma the signal rows stop
    being decodable (the defended contrast)."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(vocab, dim, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    def blocks(n, off):
        gg = torch.Generator().manual_seed(seed + off)
        n_sig = 2
        toks = torch.zeros(n, seq, dtype=torch.long)
        toks[:, -n_sig:] = torch.randint(1, vocab, (n, n_sig), generator=gg)
        rows = 3.0 * dirs[toks] + 0.02 * torch.randn(n, seq, dim, generator=gg)
        sig_scale = 4.0 + toks[:, -n_sig:].float()
        rows[:, -n_sig:] = (sig_scale[..., None] * dirs[toks[:, -n_sig:]]
                            + 0.02 * torch.randn(n, n_sig, dim, generator=gg))
        if sigma > 0.02:
            rows = rows + sigma * torch.randn(n, seq, dim, generator=gg)
        return rows.float(), toks

    train_x, train_y = blocks(train_blocks, 101)
    eval_x, eval_y = blocks(eval_blocks, 202)
    path = tmp_path / name
    torch.save({"train_wire": train_x, "train_tokens": train_y,
                "eval_wire": eval_x, "eval_tokens": eval_y}, path)
    return path


def _run_battery(tmp_path, torch, bundle, name, epochs=200, lr=1e-2):
    """Run the battery on the toy fixture. The toy groups are one batch per
    epoch at the default --batch-size, so epochs/lr are scaled up to reach
    the trained regime the battery targets on real bundles (where one
    epoch is many steps at the shipped defaults)."""
    output = tmp_path / name
    assert ls.run(_parse("--bundle", str(bundle), "--output", str(output),
                         "--epochs", str(epochs), "--restarts", "1",
                         "--lr", str(lr))) == 0
    import json
    return json.loads(output.read_text()), output


@pytest.mark.slow
def test_chaff_retain_cells_complete_and_consistent(tmp_path):
    """The counter-policy reports canonical/chaff/combined cells on the
    (0.25, 0.5, 0.9) sweep, and the combined cell is the two decode groups
    scored together."""
    import json
    torch = pytest.importorskip("torch")
    bundle = _chaff_fixture(tmp_path, torch)
    artifact, output = _run_battery(tmp_path, torch, bundle, "out.json")
    retain = [r for r in artifact["results"] if r["arm"] == "chaff_retain"]
    assert sorted((r["fraction"], r["group"]) for r in retain) == \
        sorted((x, g) for x in (0.25, 0.5, 0.9)
               for g in ("canonical", "chaff", "combined"))
    for x in (0.25, 0.5, 0.9):
        cells = {r["group"]: r for r in retain if r["fraction"] == x}
        assert cells["combined"]["eval_rows"] == \
            cells["canonical"]["eval_rows"] + cells["chaff"]["eval_rows"]
        group_top1 = [cells[g]["top1_pct"] for g in ("canonical", "chaff")]
        assert min(group_top1) - 1e-6 <= cells["combined"]["top1_pct"] \
            <= max(group_top1) + 1e-6
    journal = [json.loads(line) for line in open(str(output) + ".jsonl")]
    assert len(journal) == len(artifact["results"])
    assert artifact["summary"][0]["arms"] == sorted(
        {"capture", "chaff_id", "chaff_retain", "known_pt"})


@pytest.mark.slow
def test_chaff_retain_recovers_what_discard_throws_away(tmp_path):
    """The paper's counter-test: on a bundle where the identifier marks the
    class-signal rows as chaff, discarding them blindfolds the attacker
    while the frame-retaining split-decode policy decodes them (vulnerable
    contrast); blurring the bundle collapses the chaff group's decodability
    (defended contrast)."""
    torch = pytest.importorskip("torch")
    vulnerable = _chaff_fixture(tmp_path, torch, name="vuln.pt")
    artifact, _ = _run_battery(tmp_path, torch, vulnerable, "vuln.json")

    def cell(art, arm, fraction, group=None):
        return [r for r in art["results"]
                if r["arm"] == arm and r["fraction"] == fraction
                and r.get("group") == group][0]

    discard = cell(artifact, "chaff_id", 0.25)
    chaff = cell(artifact, "chaff_retain", 0.25, "chaff")
    combined = cell(artifact, "chaff_retain", 0.25, "combined")
    assert chaff["excess_pp"] > 30.0, chaff
    assert combined["excess_pp"] > discard["excess_pp"] + 5.0
    assert chaff["top1_pct"] > 75.0

    defended = _chaff_fixture(tmp_path, torch, name="def.pt", sigma=2.0)
    defended_art, _ = _run_battery(tmp_path, torch, defended, "def.json")
    defended_chaff = cell(defended_art, "chaff_retain", 0.25, "chaff")
    assert defended_chaff["excess_pp"] < chaff["excess_pp"] - 15.0


@pytest.mark.slow
def test_battery_is_seeded_reproducible_and_evidence_sealed(tmp_path):
    import json
    torch = pytest.importorskip("torch")
    bundle = _chaff_fixture(tmp_path, torch)
    artifact1, out1 = _run_battery(tmp_path, torch, bundle, "a.json",
                                   epochs=3)
    argv = ("--bundle", str(bundle), "--output", str(out1), "--epochs", "3",
            "--restarts", "1", "--lr", "1e-2", "--evidence")
    assert ls.run(_parse(*argv)) == 0
    artifact2, _ = _run_battery(tmp_path, torch, bundle, "b.json", epochs=3)
    sealed = json.loads(out1.read_text())
    assert sealed["results"] == artifact2["results"]
    assert sealed["summary"] == artifact2["summary"]
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(str(out1) + ".evidence.json")
    assert ok and problems == []