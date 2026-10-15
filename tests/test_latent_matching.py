"""latent-matching attack module (attacker/attacks/latent_matching.py):
contract, parser, the fail-closed torch guard, and the majority-baseline
CONTROL computation (_majority) on planted vs null label worlds.

The matching arms use the local latent feature implementation and run with
torch; no private defense SDK is required. The control math also has a
pure-Python stand-in for dependency-free validation.
"""

import sys
import types

import pytest

from attacker import attacks
from attacker.attacks import latent_matching as lm


def _parse(*argv):
    return lm.build_parser().parse_args(list(argv))


def test_contract_and_registry():
    assert lm.EXPERIMENT_ID == "latent_matching"
    assert lm.MODES == ("training",)
    assert lm.REQUIRES_LABELS is True
    attacks.validate_contract(lm, "latent-matching")
    assert attacks.ATTACKS.get("latent-matching") == \
        "attacker.attacks.latent_matching"
    assert attacks.load("latent-matching") is lm


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
        lm.build_parser().parse_args(["--help"])


def test_run_fails_closed_without_torch(tmp_path):
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        lm.run(_parse("--bundle", str(tmp_path / "b.pt"),
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
    train_c = _V([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    eval_c = _V([0, 0, 0, 0])
    assert lm._majority(train_c, eval_c) == 1.0


def test_majority_control_null_world_at_chance(torch_stub):
    """Null: balanced classes and a uniform eval — the control sits at
    exactly 1/C, the chance line the arms must beat."""
    train_c = _V([0, 1, 2, 3] * 25)
    eval_c = _V([0, 1, 2, 3] * 10)
    assert lm._majority(train_c, eval_c) == pytest.approx(0.25)


def test_majority_counts_unknown_eval_labels_as_misses(torch_stub):
    """Unknown evaluation labels remain in the denominator as misses."""
    train_c = _V([0, 0, 1, -1, -1])
    eval_c = _V([1, 1, -1, 0])
    assert lm._majority(train_c, eval_c) == pytest.approx(1.0 / 4.0)


def test_majority_bincount_tiebreak_is_deterministic(torch_stub):
    """A train tie resolves to the lowest class index (torch.argmax's
    first-max rule), so the control is reproducible."""
    train_c = _V([2, 5, 2, 5])
    eval_c = _V([2, 5])
    assert lm._majority(train_c, eval_c) == pytest.approx(0.5)


@pytest.mark.slow
def test_real_bundle_unseen_vocabulary_is_scored_without_private_sdk(tmp_path):
    import json
    torch = pytest.importorskip("torch")
    bundle = tmp_path / "bundle.pt"
    torch.save({"train_wire": torch.randn(2, 3, 4), "eval_wire": torch.randn(2, 3, 4),
                "train_tokens": torch.tensor([[1,2,1], [2,1,2]]),
                "eval_tokens": torch.full((2,3), 100)}, bundle)
    output = tmp_path / "out.json"
    assert lm.run(_parse("--bundle", str(bundle), "--output", str(output))) == 0
    records = json.loads(output.read_text())["results"]
    assert len(records) == 2
    assert all(r["eval_rows"] == 6 and r["top1_pct"] == 0 for r in records)


def _order_fixture(tmp_path, torch, name="bundle.pt", seq=6, dim=12, vocab=4,
                   train_blocks=24, eval_blocks=12, noise=0.05,
                   position_scale=True, permute=True, seed=0,
                   include_order=True):
    """Synthetic permutation fixture with a known ground-truth order.

    Canonical rows are class-clustered (scale_p * dir[token] + small noise);
    with position_scale the scale grows with the TRUE position, so the
    gauge-invariant norm channel carries the position leak the metric must
    find. Rows are then permuted within each frame; train_order/eval_order
    record the true index of each released row."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(vocab, dim, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    def blocks(n, off):
        gg = torch.Generator().manual_seed(seed + off)
        toks = torch.randint(0, vocab, (n, seq), generator=gg)
        pos = torch.arange(seq).float()
        scale = 4.0 + pos if position_scale else torch.full((seq,), 4.0)
        rows = (scale[None, :, None] * dirs[toks]
                + noise * torch.randn(n, seq, dim, generator=gg))
        if permute:
            order = torch.stack(
                [torch.randperm(seq, generator=gg) for _ in range(n)])
        else:
            order = torch.arange(seq).expand(n, -1).clone()
        rows = torch.gather(rows, 1, order.unsqueeze(-1).expand(-1, -1, dim))
        toks = torch.gather(toks, 1, order)
        return rows.float(), toks, order

    train_x, train_y, train_order = blocks(train_blocks, 101)
    eval_x, eval_y, eval_order = blocks(eval_blocks, 202)
    bundle = {"train_wire": train_x, "train_tokens": train_y,
              "eval_wire": eval_x, "eval_tokens": eval_y}
    if include_order:
        bundle["train_order"] = train_order
        bundle["eval_order"] = eval_order
    path = tmp_path / name
    torch.save(bundle, path)
    return path


@pytest.mark.slow
def test_order_recovery_finds_planted_position_leak(tmp_path):
    """Planted positional norm structure: the arms must recover the true
    permutation far above the 1/seq chance line, alongside token top-1.

    The signature arm's unrestricted 1-NN geometry reads the norm leak
    directly; the centroid arm's two-stage resolve is bounded by its
    (weaker) token matching, so it is held to a looser but still
    far-above-chance bar — the gap between the arms is the honest cost of
    hard class restriction."""
    import json
    torch = pytest.importorskip("torch")
    bundle = _order_fixture(tmp_path, torch)
    output = tmp_path / "out.json"
    assert lm.run(_parse("--bundle", str(bundle), "--output",
                         str(output))) == 0
    artifact = json.loads(output.read_text())
    records = artifact["results"]
    assert [r["arm"] for r in records] == \
        ["vma_centroid_match", "sorted_signature_matching_position_free"]
    chance = 100.0 / 6
    bars = {"vma_centroid_match": 2.5 * chance,
            "sorted_signature_matching_position_free": 75.0}
    for r in records:
        assert "top1_pct" in r
        assert r["order_chance_pct"] == pytest.approx(chance)
        assert r["order_recovery_pct"] >= bars[r["arm"]], r
        assert r["order_excess_pp"] == \
            pytest.approx(r["order_recovery_pct"] - chance)
    summary = artifact["summary"][0]
    assert summary["order_recovery_evaluated"] is True
    assert summary["worst_order_excess_pp"] >= 50.0


@pytest.mark.slow
def test_order_recovery_null_fixture_sits_far_below_planted(tmp_path):
    """No positional structure: the per-frame assignment can only overfit
    noise, so recovery must land far below the planted fixture's."""
    import json
    torch = pytest.importorskip("torch")
    planted_out = tmp_path / "planted.json"
    lm.run(_parse("--bundle",
                  str(_order_fixture(tmp_path, torch, name="p.pt")),
                  "--output", str(planted_out)))
    null_out = tmp_path / "null.json"
    lm.run(_parse("--bundle",
                  str(_order_fixture(tmp_path, torch, name="n.pt",
                                     position_scale=False)),
                  "--output", str(null_out)))
    for planted, null in zip(json.loads(planted_out.read_text())["results"],
                             json.loads(null_out.read_text())["results"]):
        assert null["order_recovery_pct"] < 50.0, null
        assert planted["order_recovery_pct"] \
            - null["order_recovery_pct"] > 30.0


@pytest.mark.slow
def test_order_recovery_is_fail_closed_without_order_ground_truth(tmp_path):
    """Bundles without order fields get token metrics only, with the
    omission journaled — the metric is never fabricated."""
    import json
    torch = pytest.importorskip("torch")
    bundle = _order_fixture(tmp_path, torch, include_order=False)
    output = tmp_path / "out.json"
    assert lm.run(_parse("--bundle", str(bundle), "--output",
                         str(output))) == 0
    artifact = json.loads(output.read_text())
    assert artifact["summary"][0]["order_recovery_evaluated"] is False
    assert artifact["summary"][0]["worst_order_excess_pp"] is None
    for r in artifact["results"]:
        assert r["order_recovery_pct"] is None
        assert r["order_excess_pp"] is None
        assert "no order ground truth" in r["order_note"]
    import torch as _t
    half = _t.load(bundle, map_location="cpu")
    half.pop("train_order", None)
    half["eval_order"] = _t.zeros_like(half["eval_tokens"])
    half_path = tmp_path / "half.pt"
    _t.save(half, half_path)
    half_out = tmp_path / "half.json"
    assert lm.run(_parse("--bundle", str(half_path), "--output",
                         str(half_out))) == 0
    half_rec = json.loads(half_out.read_text())["results"][0]
    assert half_rec["order_recovery_pct"] is None
    assert "BOTH" in half_rec["order_note"]


@pytest.mark.slow
def test_run_is_seeded_reproducible_and_evidence_sealed(tmp_path):
    import json
    torch = pytest.importorskip("torch")
    bundle = _order_fixture(tmp_path, torch)
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    argv = ("--bundle", str(bundle), "--output", str(out1), "--evidence")
    assert lm.run(_parse(*argv)) == 0
    assert lm.run(_parse("--bundle", str(bundle), "--output",
                         str(out2))) == 0
    a, b = (json.loads(p.read_text()) for p in (out1, out2))
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    from attacker.evidence import verify_evidence
    ok, problems = verify_evidence(str(out1) + ".evidence.json")
    assert ok and problems == []
    journal = [json.loads(line)
               for line in open(str(out1) + ".jsonl")]
    assert [j["arm"] for j in journal] == [r["arm"] for r in a["results"]]
