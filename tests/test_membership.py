"""membership attack module (attacker/attacks/membership.py): the toy
fixture generator as a known-answer world (planted membership signal vs
effect=0 null at chance), the run() artifact/journal wiring driven through
a REAL import of a pure-python stand-in for split-training/
membership_inference.py (the sibling tree is absent from this checkout;
_load_mi falls back to sys.path by design), and the fail-closed seams.

Torch-free throughout: the module itself never imports torch.
"""

import json
import sys

import pytest

from attacker import attacks
from attacker.attacks import membership as mb


def _parse(*argv):
    args = mb.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


_FIXTURE_MI = '''
"""Stand-in for split-training/membership_inference.py (not in this repo
checkout): document-disjoint logistic-probe stand-in, deterministic."""
import json


def _centroid(rows):
    n, d = len(rows), len(rows[0])
    return [sum(r[j] for r in rows) / n for j in range(d)]


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def run(paths, out, seeds):
    records = []
    for p in paths:
        with open(p) as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    by_cond = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    conditions = {}
    for cond, rs in sorted(by_cond.items()):
        idx = [int(r["document_id"].rsplit("doc", 1)[1]) for r in rs]
        fit = [r for r, i in zip(rs, idx) if i < len(rs) // 2]
        held = [r for r, i in zip(rs, idx) if i >= len(rs) // 2]
        cm = _centroid([r["features"] for r in fit if r["membership"] == 1])
        cn = _centroid([r["features"] for r in fit if r["membership"] == 0])
        correct = sum(
            1 for r in held
            if int(_dist2(r["features"], cm) < _dist2(r["features"], cn))
            == r["membership"])
        conditions[cond] = {
            "membership_accuracy": correct / max(1, len(held)),
            "n_fit": len(fit), "n_held": len(held)}
    summary = {"experiment": "ea4_membership_inner", "seeds": list(seeds),
               "n_records": len(records), "conditions": conditions}
    with open(out, "w") as fh:
        json.dump(summary, fh)
    return summary
'''

_BOOM_MI = '''
def run(paths, out, seeds):
    raise RuntimeError("injected probe failure")
'''


def _install_mi(monkeypatch, tmp_path, source=_FIXTURE_MI):
    """Drop a stand-in membership_inference.py onto sys.path and force a
    fresh import (the module may be cached from an earlier test)."""
    monkeypatch.delitem(sys.modules, "membership_inference", raising=False)
    (tmp_path / "membership_inference.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))


def test_contract_and_registry():
    assert mb.EXPERIMENT_ID == "ea4_membership"
    assert mb.MODES == ("training",)
    assert mb.REQUIRES_LABELS is False
    assert isinstance(mb.DESCRIPTION, str)
    attacks.validate_contract(mb, "membership")
    assert attacks.ATTACKS.get("membership") == "attacker.attacks.membership"
    assert attacks.load("membership") is mb


def _read_features(path):
    return [json.loads(line) for line in
            open(path).read().strip().splitlines()]


def test_toy_features_schema_and_balance(tmp_path):
    p = tmp_path / "toy.jsonl"
    assert mb.make_toy_features(str(p), n_docs=120, n_feat=16, seed=0) == str(p)
    recs = _read_features(p)
    assert len(recs) == 240
    assert {r["condition"] for r in recs} == {"split_ft", "fedavg"}
    for r in recs:
        assert set(r) == {"document_id", "condition", "membership",
                          "property", "features"}
        assert len(r["features"]) == 16
        assert r["membership"] in (0, 1) and r["property"] in (0, 1)
    per_cond = {}
    for r in recs:
        per_cond.setdefault(r["condition"], []).append(r)
    for rs in per_cond.values():
        assert len(rs) == 120
        assert sum(r["membership"] for r in rs) == 60


def test_toy_features_planted_signal_vs_null(tmp_path):
    """Members are shifted by `effect` along a fixed direction: the
    member/non-member centroid gap is large in the planted world and
    vanishes in the effect=0 null world."""
    def gap_norm(effect):
        recs = _read_features(mb.make_toy_features(
            str(tmp_path / f"f{effect}.jsonl"), n_docs=120, n_feat=16,
            seed=0, effect=effect))
        mem = [r["features"] for r in recs if r["membership"] == 1]
        non = [r["features"] for r in recs if r["membership"] == 0]
        gap = [sum(f[j] for f in mem) / len(mem)
               - sum(f[j] for f in non) / len(non) for j in range(16)]
        return sum(g * g for g in gap) ** 0.5
    assert gap_norm(0.8) > 2.0
    assert gap_norm(0.0) < 1.0


def test_toy_features_deterministic_per_seed(tmp_path):
    a = mb.make_toy_features(str(tmp_path / "a.jsonl"), seed=7)
    b = mb.make_toy_features(str(tmp_path / "b.jsonl"), seed=7)
    c = mb.make_toy_features(str(tmp_path / "c.jsonl"), seed=8)
    assert open(a).read() == open(b).read()
    assert open(a).read() != open(c).read()


def test_run_toy_end_to_end_planted_world(tmp_path, monkeypatch):
    _install_mi(monkeypatch, tmp_path)
    out = tmp_path / "ea4.json"
    args = _parse("--toy", "--quick", "--output", str(out))
    assert mb.run(args) == 0
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.ea4.v1"
    assert art["config"]["attack"] == "ea4_membership"
    assert art["config"]["toy"] is True and art["config"]["seeds"] == [0]
    assert art["config"]["features"][0].endswith("toy.jsonl")
    (summary,) = art["results"]
    assert art["summary"] == art["results"]
    assert summary["n_records"] == 240
    for cond in ("split_ft", "fedavg"):
        cell = summary["conditions"][cond]
        assert cell["n_fit"] == 60 and cell["n_held"] == 60
        assert cell["membership_accuracy"] > 0.75
    inner = json.loads((tmp_path / "ea4.json.membership.json").read_text())
    assert inner == summary


def test_run_features_null_world_at_chance(tmp_path, monkeypatch):
    _install_mi(monkeypatch, tmp_path)
    null = mb.make_toy_features(str(tmp_path / "null.jsonl"), seed=3,
                                effect=0.0)
    out = tmp_path / "ea4_null.json"
    args = _parse("--features", null, "--output", str(out))
    assert mb.run(args) == 0
    (summary,) = json.loads(out.read_text())["results"]
    for cond in ("split_ft", "fedavg"):
        acc = summary["conditions"][cond]["membership_accuracy"]
        assert 0.3 <= acc <= 0.7


def test_run_fails_closed_and_journals_probe_error(tmp_path, monkeypatch):
    _install_mi(monkeypatch, tmp_path, source=_BOOM_MI)
    out = tmp_path / "ea4_boom.json"
    with pytest.raises(RuntimeError, match="injected probe failure"):
        mb.run(_parse("--toy", "--quick", "--output", str(out)))
    journal = [json.loads(line) for line in
               (tmp_path / "ea4_boom.json.jsonl").read_text().splitlines()]
    assert len(journal) == 1
    assert journal[0]["experiment"] == "ea4_membership"
    assert "injected probe failure" in journal[0]["error"]


def test_run_requires_toy_or_features(tmp_path, monkeypatch):
    _install_mi(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="--toy or --features"):
        mb.run(_parse("--output", str(tmp_path / "x.json")))


def test_run_without_split_training_tree_is_a_descriptive_import_error(
        monkeypatch):
    """The math module lives in the sibling split-training/ tree (absent in
    this checkout); _load_mi must surface that, not crash obscurely."""
    monkeypatch.delitem(sys.modules, "membership_inference", raising=False)
    with pytest.raises(ModuleNotFoundError, match="membership_inference"):
        mb._load_mi()


def test_run_determinism(tmp_path, monkeypatch):
    _install_mi(monkeypatch, tmp_path)
    results = []
    for name in ("d1.json", "d2.json"):
        out = tmp_path / name
        assert mb.run(_parse("--toy", "--quick", "--output", str(out))) == 0
        results.append(json.loads(out.read_text())["results"])
    assert results[0] == results[1]
