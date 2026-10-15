"""max-effort attack module (attacker/attacks/max_effort.py): the
distribution-reporting driver end to end with known-answer per-cell cores
registered through the module's own _CORES seam (the built-in cores are
torch-gated toy wrappers; the aggregation/distribution machinery being
tested is pure python), plus pct() quartile math and the fail-closed
guards.
"""

import json
import statistics

import pytest

from attacker import attacks
from attacker.attacks import max_effort as me


def _parse(*argv):
    args = me.build_parser().parse_args(list(argv))
    args.mode = "training"
    return args


def _torch_guard_bypass(monkeypatch):
    """require_torch reads common.torch; a sentinel gets us to the
    pure-python driver logic (the torch cores themselves are faked)."""
    monkeypatch.setattr("attacker.attacks.common.torch", object())


def _known_core(args, budget, init_seed):
    """Deterministic known answer: 100*B/(B+32) + init_seed."""
    return 100.0 * budget / (budget + 32) + init_seed


def test_contract_and_registry():
    assert me.EXPERIMENT_ID == "max_effort"
    assert me.MODES == ("training", "inference")
    assert "inherits" in me.REQUIRES_LABELS
    attacks.validate_contract(me, "max-effort")
    assert attacks.ATTACKS.get("max-effort") == "attacker.attacks.max_effort"
    assert attacks.load("max-effort") is me
    assert sorted(me._CORES) == ["accumulation", "ica-bss"]


def test_parser_defaults():
    args = _parse()
    assert args.target == "accumulation"
    assert args.budgets == [32, 64, 128, 256]
    assert args.inits == 8
    helptext = me.build_parser().format_help()
    assert "accumulation" in helptext and "ica-bss" in helptext


def test_pct_known_answers():
    assert me.pct([], 0.5) is None
    assert me.pct([5.0], 0.0) == me.pct([5.0], 1.0) == 5.0
    vals = [4.0, 1.0, 3.0, 2.0]
    assert me.pct(vals, 0.0) == 1.0
    assert me.pct(vals, 0.25) == 2.0
    assert me.pct(vals, 0.5) == 3.0
    assert me.pct(vals, 0.75) == 3.0
    assert me.pct(vals, 1.0) == 4.0


def test_requires_torch_guard():
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch present: the guard does not fire")
    with pytest.raises(SystemExit, match="requires torch"):
        me.run(_parse("--toy", "--quick"))


def test_unknown_target_is_descriptive(monkeypatch, tmp_path):
    _torch_guard_bypass(monkeypatch)
    with pytest.raises(SystemExit, match="no factorized core"):
        me.run(_parse("--toy", "--target", "no-such-attack",
                      "--output", str(tmp_path / "x.json")))


def test_toy_only_guard(monkeypatch, tmp_path):
    _torch_guard_bypass(monkeypatch)
    with pytest.raises(SystemExit, match="--toy only"):
        me.run(_parse("--output", str(tmp_path / "x.json")))


def test_run_distribution_is_a_known_answer(tmp_path, monkeypatch):
    _torch_guard_bypass(monkeypatch)
    monkeypatch.setitem(me._CORES, "accumulation", _known_core)
    out = tmp_path / "me.json"
    args = _parse("--toy", "--quick", "--output", str(out))
    assert me.run(args) == 0
    art = json.loads(out.read_text())
    assert art["schema"] == "dtraining.attacker.max_effort.v1"
    assert art["config"]["attack"] == "max_effort"
    assert art["config"]["target"] == "accumulation"
    assert art["config"]["budgets"] == [32, 64, 128]
    assert art["config"]["inits"] == 4
    assert art["results"] == art["summary"]
    assert [r["budget"] for r in art["results"]] == [32, 64, 128]
    rec = art["results"][0]
    assert rec["all_runs"] == [50.0, 51.0, 52.0, 53.0]
    dist = rec["distribution"]
    assert dist["min"] == 50.0 and dist["max"] == 53.0
    assert dist["mean"] == 51.5
    assert dist["p25"] == 51.0 and dist["p75"] == 52.0
    assert dist["std"] == pytest.approx(statistics.pstdev([50, 51, 52, 53]))
    assert dist["n_ok"] == 4 and dist["n_error"] == 0
    journal = [json.loads(line) for line in
               (tmp_path / "me.json.jsonl").read_text().splitlines()]
    assert len(journal) == 3
    assert all(j["experiment"] == "max_effort" for j in journal)


def test_run_journals_cell_errors_and_counts_silent_none(tmp_path,
                                                         monkeypatch):
    _torch_guard_bypass(monkeypatch)

    def flaky_core(args, budget, init_seed):
        if init_seed == 1:
            raise RuntimeError("injected solve failure")
        if init_seed == 2:
            return None
        return 10.0 + init_seed

    monkeypatch.setitem(me._CORES, "accumulation", flaky_core)
    out = tmp_path / "me_err.json"
    assert me.run(_parse("--toy", "--quick", "--budgets", "32",
                         "--output", str(out))) == 0
    (rec,) = json.loads(out.read_text())["results"]
    assert rec["all_runs"] == [10.0, 13.0]
    assert rec["distribution"]["n_ok"] == 2
    assert rec["distribution"]["n_error"] == 2
    journal = [json.loads(line) for line in
               (tmp_path / "me_err.json.jsonl").read_text().splitlines()]
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 1
    assert errs[0]["init"] == 1 and errs[0]["budget"] == 32
    assert "injected solve failure" in errs[0]["error"]
    assert errs[0]["experiment"] == "max_effort"


def test_run_null_world_core_sits_at_chance(tmp_path, monkeypatch):
    """A target whose reruns never beat chance must REPORT a flat
    chance-level distribution — the pre-emptor is only honest if the
    null case is visibly flat."""
    _torch_guard_bypass(monkeypatch)
    monkeypatch.setitem(me._CORES, "accumulation",
                        lambda args, budget, init_seed: 3.125)
    out = tmp_path / "me_null.json"
    assert me.run(_parse("--toy", "--quick", "--budgets", "32", "64",
                         "--output", str(out))) == 0
    for rec in json.loads(out.read_text())["results"]:
        dist = rec["distribution"]
        assert dist["min"] == dist["max"] == dist["mean"] == 3.125
        assert dist["std"] == 0.0 and dist["n_error"] == 0


def test_run_is_deterministic(tmp_path, monkeypatch):
    _torch_guard_bypass(monkeypatch)
    monkeypatch.setitem(me._CORES, "accumulation", _known_core)
    results = []
    for name in ("a.json", "b.json"):
        assert me.run(_parse("--toy", "--quick", "--output",
                             str(tmp_path / name))) == 0
        results.append(json.loads((tmp_path / name).read_text())["results"])
    assert results[0] == results[1]
